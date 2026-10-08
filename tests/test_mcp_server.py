"""The MCP server, driven through the SDK's in-memory client against a local tile server."""

from __future__ import annotations

import io
import json
import math
import sys
import threading
import time
from collections.abc import Awaitable, Callable, Iterator
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any, TypeVar

import pytest
from PIL import Image
from typer.testing import CliRunner

pytest.importorskip("mcp")

import anyio
from mcp.client import Client
from mcp.types import CallToolResult

from mapcv import planning
from mapcv.cli import app
from mapcv.mcp_server import build_server

T = TypeVar("T")

ZOOM = 18
X0, Y0, NX, NY = 134_700, 86_100, 6, 5  # 30 tiles near Amsterdam, as in tests/e2e/journey.py
READ_TOOLS = {
    "describe_config_schema",
    "validate_config",
    "inspect_labels",
    "plan",
    "info",
    "stats",
    "verify",
}
WRITE_TOOLS = {"write_config", "generate", "split"}


def _lon(x: float) -> float:
    return float(x / 2**ZOOM * 360 - 180)


def _lat(y: float) -> float:
    return float(math.degrees(math.atan(math.sinh(math.pi * (1 - 2 * y / 2**ZOOM)))))


class _Tiles(BaseHTTPRequestHandler):
    delay = 0.0
    requests: list[str] = []

    def log_message(self, *args: object) -> None:
        pass

    def do_GET(self) -> None:
        type(self).requests.append(self.path)
        if type(self).delay:
            time.sleep(type(self).delay)
        if "FAIL" in self.path:
            self.send_response(403)
            self.send_header("Content-Length", "0")
            self.end_headers()
            return
        z, x, y = (
            int(part) for part in self.path.strip("/").split("?")[0].split(".")[0].split("/")[-3:]
        )
        buffer = io.BytesIO()
        Image.new("RGB", (256, 256), ((x * 37) % 256, (y * 53) % 256, z * 9)).save(buffer, "PNG")
        body = buffer.getvalue()
        self.send_response(200)
        self.send_header("Content-Type", "image/png")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)


@pytest.fixture()
def tile_server() -> Iterator[str]:
    _Tiles.delay = 0.0
    _Tiles.requests = []
    ThreadingHTTPServer.request_queue_size = 128
    server = ThreadingHTTPServer(("127.0.0.1", 0), _Tiles)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    yield f"http://127.0.0.1:{server.server_address[1]}"
    server.shutdown()


def _bbox() -> tuple[float, float, float, float]:
    eps = 1e-7
    return _lon(X0) + eps, _lat(Y0 + NY) + eps, _lon(X0 + NX) - eps, _lat(Y0) - eps


def _labels() -> str:
    west, south, east, north = _bbox()
    w, h = (east - west) / 4, (north - south) / 4
    features = [
        {
            "type": "Feature",
            "properties": {"class": name, "id": index},
            "geometry": {
                "type": "Polygon",
                "coordinates": [[[x, y], [x + w, y], [x + w, y + h], [x, y + h], [x, y]]],
            },
        }
        for index, (name, x, y) in enumerate(
            (
                ("building", west + w / 2, south + h / 2),
                ("water", west + 2.5 * w, south + 2 * h),
                ("building", west + 2 * w, south + 0.5 * h),
            )
        )
    ]
    return json.dumps({"type": "FeatureCollection", "features": features})


def config_text(
    template: str,
    staging: str = "dataset",
    extra_imagery: str = "",
    region: tuple[float, float, float, float] | None = None,
    zoom: int = ZOOM,
) -> str:
    west, south, east, north = region or _bbox()
    return f"""\
region:
  west: {west}
  south: {south}
  east: {east}
  north: {north}
imagery:
  type: xyz
  zoom: {zoom}
  url_template: "{template}"
  max_connections: 4
{extra_imagery}
labels:
  path: labels.geojson
  label_field: class
sampler:
  patch_size: 256
writer:
  staging_dir: {staging}
split:
  strategy: spatial
  test_ratio: 0.2
  val_ratio: 0.1
"""


@pytest.fixture()
def project(tmp_path: Path, tile_server: str) -> Path:
    """A folder with labels and a config that reads the local tile server."""
    root = tmp_path / "proj"
    root.mkdir()
    (root / "labels.geojson").write_text(_labels(), encoding="utf-8")
    (root / "mapcv.yaml").write_text(
        config_text(f"{tile_server}/{{z}}/{{x}}/{{y}}.png"), encoding="utf-8"
    )
    return root


def run_client(
    root: Path,
    scenario: Callable[[Client], Awaitable[T]],
    write: bool = True,
    local_urls: bool = True,
) -> T:
    # The tests' tile and STAC servers run on this machine: --allow-local-urls.
    server = build_server(root, allow_write=write, allow_local_urls=local_urls)

    async def main() -> T:
        async with Client(server) as client:
            return await scenario(client)

    return anyio.run(main)


def text_of(result: CallToolResult) -> str:
    """Everything a model or a client could read from a result."""
    parts = [block.text for block in result.content if hasattr(block, "text")]
    return "\n".join([*parts, json.dumps(result.structured_content)])


async def call(client: Client, tool: str, **arguments: Any) -> CallToolResult:
    return await client.call_tool(tool, arguments)


def data(result: CallToolResult) -> dict[str, Any]:
    assert result.structured_content is not None
    return dict(result.structured_content)


def tree(path: Path) -> dict[str, bytes]:
    return {
        p.relative_to(path).as_posix(): p.read_bytes()
        for p in sorted(path.rglob("*"))
        if p.is_file()
    }


# ── Tools offered ────────────────────────────────────────────────────────────


def test_read_only_server_offers_no_write_tools(project: Path) -> None:
    async def scenario(client: Client) -> None:
        names = {tool.name for tool in (await client.list_tools()).tools}
        assert names == READ_TOOLS
        for name in WRITE_TOOLS:
            result = await call(client, name, config="mapcv.yaml", path="x.yaml", dataset="d")
            assert result.is_error

    run_client(project, scenario, write=False)
    assert not (project / "dataset").exists()


def test_write_mode_offers_every_tool_with_annotations(project: Path) -> None:
    async def scenario(client: Client) -> None:
        tools = {tool.name: tool for tool in (await client.list_tools()).tools}
        assert set(tools) == READ_TOOLS | WRITE_TOOLS
        for name in READ_TOOLS | WRITE_TOOLS:
            annotations = tools[name].annotations
            assert annotations is not None
            # stats and verify can also write (stats.json, SHA256SUMS) when writing is allowed.
            assert annotations.read_only_hint is (name in READ_TOOLS - {"stats", "verify"})
        assert tools["generate"].input_schema["properties"]["confirm_large"]["default"] is False

    run_client(project, scenario)


# ── Happy paths ──────────────────────────────────────────────────────────────


def test_journey_inspect_validate_plan_generate_info_split(project: Path) -> None:
    events: list[tuple[float, float | None]] = []

    async def on_progress(progress: float, total: float | None, message: str | None) -> None:
        events.append((progress, total))

    async def scenario(client: Client) -> None:
        labels = await call(client, "inspect_labels", path="labels.geojson")
        assert not labels.is_error
        found = data(labels)
        assert found["features"] == 3
        assert found["geometry_types"] == {"Polygon": 3}
        assert found["label_field_candidates"] == ["class", "id"]
        by_name = {field["name"]: field for field in found["fields"]}
        assert by_name["class"]["values"] == [
            {"value": "building", "count": 2},
            {"value": "water", "count": 1},
        ]
        west, _south, _east, north = _bbox()
        assert found["extent"]["west"] > west and found["extent"]["north"] < north

        valid = await call(client, "validate_config", path="mapcv.yaml")
        assert data(valid)["valid"] is True
        assert data(valid)["summary"]["task"] == "segmentation"
        inline = await call(
            client, "validate_config", yaml_text=(project / "mapcv.yaml").read_text()
        )
        assert data(inline)["valid"] is True

        planned = data(await call(client, "plan", config="mapcv.yaml"))
        assert planned["tiles"] == NX * NY
        assert planned["patches"] == NX * NY
        assert planned["large"] is False and planned["large_reason"] is None
        assert planned["output"] == "dataset"
        assert planned["labels"]["features"] == 3

        generated = await client.call_tool(
            "generate", {"config": "mapcv.yaml"}, progress_callback=on_progress
        )
        assert not generated.is_error, text_of(generated)
        done = data(generated)
        assert done["patches"] == NX * NY and done["new_patches"] == NX * NY
        assert done["dataset"] == "dataset"
        assert set(done["splits"]) >= {"train", "val", "test"}

        summary = data(await call(client, "info", dataset="dataset"))
        assert summary["patches"] == NX * NY
        assert summary["task"] == "segmentation"
        assert {row["name"] for row in summary["class_balance"]} >= {"building", "water"}

        resplit = data(
            await call(client, "split", dataset="dataset", test_ratio=0.25, strategy="random")
        )
        assert resplit["strategy"] == "random"
        assert sum(resplit["splits"][name] for name in ("train", "val", "test")) == NX * NY

        # A finished run resumes as a no-op.
        again = data(await call(client, "generate", config="mapcv.yaml"))
        assert again["new_patches"] == 0

    run_client(project, scenario)
    assert events and events[-1][0] == events[-1][1]
    assert [done for done, _ in events] == sorted(done for done, _ in events)
    assert (project / "dataset" / "manifest.json").exists()


def test_inspect_labels_reads_kml(project: Path) -> None:
    kml = project / "places.kml"
    kml.write_text(
        """<?xml version="1.0" encoding="UTF-8"?>
<kml xmlns="http://www.opengis.net/kml/2.2"><Document>
<Placemark><ExtendedData><SchemaData><SimpleData name="kind">pond</SimpleData></SchemaData>
</ExtendedData><Polygon><outerBoundaryIs><LinearRing><coordinates>
4.0,52.0 4.1,52.0 4.1,52.1 4.0,52.1 4.0,52.0</coordinates></LinearRing></outerBoundaryIs>
</Polygon></Placemark>
<Placemark><Point><coordinates>4.05,52.05</coordinates></Point></Placemark>
</Document></kml>""",
        encoding="utf-8",
    )

    async def scenario(client: Client) -> None:
        found = data(await call(client, "inspect_labels", path="places.kml"))
        assert found["features"] == 2
        assert found["geometry_types"] == {"Polygon": 1, "points/lines": 1}
        assert found["fields"][0]["name"] == "kind"
        assert found["extent"] == {"west": 4.0, "south": 52.0, "east": 4.1, "north": 52.1}

    run_client(project, scenario, write=False)


def test_inspect_labels_reports_projected_coordinates(project: Path) -> None:
    (project / "utm.geojson").write_text(
        json.dumps(
            {
                "type": "FeatureCollection",
                "features": [
                    {
                        "type": "Feature",
                        "properties": {},
                        "geometry": {
                            "type": "Polygon",
                            "coordinates": [
                                [
                                    [500000, 5000000],
                                    [500100, 5000000],
                                    [500100, 5000100],
                                    [500000, 5000000],
                                ]
                            ],
                        },
                    }
                ],
            }
        ),
        encoding="utf-8",
    )

    async def scenario(client: Client) -> None:
        found = data(await call(client, "inspect_labels", path="utm.geojson"))
        assert any("projected" in note for note in found["notes"])

    run_client(project, scenario, write=False)


def test_write_config_validates_then_writes(project: Path) -> None:
    text = (project / "mapcv.yaml").read_text()

    async def scenario(client: Client) -> None:
        bad = await call(client, "write_config", path="new.yaml", yaml_text=text + "bogus: 1\n")
        assert bad.is_error
        assert data(bad)["errors"][0]["field"] == "bogus"
        assert not (project / "new.yaml").exists()

        wrong_suffix = await call(client, "write_config", path="new.txt", yaml_text=text)
        assert wrong_suffix.is_error

        good = await call(client, "write_config", path="sub/new.yaml", yaml_text=text)
        assert not good.is_error, text_of(good)
        assert (project / "sub" / "new.yaml").read_text() == text

        again = await call(client, "write_config", path="sub/new.yaml", yaml_text=text)
        assert again.is_error and "overwrite" in text_of(again)
        replaced = await call(
            client, "write_config", path="sub/new.yaml", yaml_text=text, overwrite=True
        )
        assert not replaced.is_error
        # Paths in a config written to sub/ resolve against sub/, so labels.path leaves nothing
        # behind: it points at sub/labels.geojson, which does not exist yet.
        checked = data(await call(client, "validate_config", path="sub/new.yaml"))
        assert checked["valid"] is False
        assert checked["errors"][0]["field"] == "labels.path"
        assert "file not found: sub/labels.geojson" in checked["errors"][0]["message"]

    run_client(project, scenario)


# ── Same answers as the CLI ──────────────────────────────────────────────────


def test_validate_errors_match_the_cli(project: Path) -> None:
    broken = (project / "mapcv.yaml").read_text().replace("zoom: 18", "zoom: 99\n  frobnicate: 1")
    (project / "broken.yaml").write_text(broken)
    cli = CliRunner().invoke(app, ["validate", str(project / "broken.yaml")], terminal_width=200)
    assert cli.exit_code == 1

    async def scenario(client: Client) -> None:
        result = await call(client, "validate_config", path="broken.yaml")
        assert not result.is_error
        found = data(result)
        assert found["valid"] is False
        for error in found["errors"]:
            assert f"{error['field']}: {error['message']}" in " ".join(cli.output.split())

    run_client(project, scenario, write=False)


def test_generate_is_byte_identical_to_the_cli(project: Path) -> None:
    text = (project / "mapcv.yaml").read_text()
    (project / "cli.yaml").write_text(text.replace("staging_dir: dataset", "staging_dir: by_cli"))
    cli = CliRunner().invoke(app, ["generate", str(project / "cli.yaml"), "--yes"])
    assert cli.exit_code == 0, cli.output

    async def scenario(client: Client) -> None:
        result = await call(client, "generate", config="mapcv.yaml")
        assert not result.is_error, text_of(result)

    run_client(project, scenario)
    by_cli, by_mcp = tree(project / "by_cli"), tree(project / "dataset")
    assert by_cli.keys() == by_mcp.keys()
    assert len(by_cli) > NX * NY
    assert by_cli == by_mcp


# ── Safety: the root ─────────────────────────────────────────────────────────


def test_paths_outside_the_root_are_refused(project: Path, tmp_path: Path) -> None:
    outside = tmp_path / "outside"
    outside.mkdir()
    (outside / "labels.geojson").write_text(_labels(), encoding="utf-8")
    (outside / "secret.yaml").write_text((project / "mapcv.yaml").read_text())
    (project / "link").symlink_to(outside, target_is_directory=True)
    (project / "linked.geojson").symlink_to(outside / "labels.geojson")

    async def scenario(client: Client) -> None:
        for path in (
            "../outside/labels.geojson",
            str(outside / "labels.geojson"),
            "link/labels.geojson",
            "linked.geojson",
            "/etc/passwd",
        ):
            result = await call(client, "inspect_labels", path=path)
            assert result.is_error and "outside the folder" in text_of(result), path
        for path in ("../outside/secret.yaml", "link/secret.yaml", str(outside / "secret.yaml")):
            for tool, argument in (("validate_config", "path"), ("plan", "config")):
                result = await call(client, tool, **{argument: path})
                assert result.is_error and "outside the folder" in text_of(result), (tool, path)
        for dataset in ("..", "link", str(outside)):
            assert (await call(client, "info", dataset=dataset)).is_error
            assert (await call(client, "split", dataset=dataset)).is_error
        text = (project / "mapcv.yaml").read_text()
        for path in ("../escape.yaml", "link/new.yaml", str(outside / "new.yaml")):
            result = await call(client, "write_config", path=path, yaml_text=text)
            assert result.is_error and "outside the folder" in text_of(result), path
        assert not list(outside.glob("new.yaml")) and not (tmp_path / "escape.yaml").exists()

    run_client(project, scenario)


def test_config_paths_that_leave_the_root_are_refused(project: Path, tmp_path: Path) -> None:
    outside = tmp_path / "outside"
    outside.mkdir()
    (outside / "labels.geojson").write_text(_labels(), encoding="utf-8")
    (project / "link").symlink_to(outside, target_is_directory=True)
    base = (project / "mapcv.yaml").read_text()
    cases = {
        "labels ..": base.replace("path: labels.geojson", "path: ../outside/labels.geojson"),
        "labels absolute": base.replace("path: labels.geojson", f"path: {outside}/labels.geojson"),
        "labels symlink": base.replace("path: labels.geojson", "path: link/labels.geojson"),
        "staging ..": base.replace("staging_dir: dataset", "staging_dir: ../stolen"),
        "staging absolute": base.replace("staging_dir: dataset", f"staging_dir: {outside}/ds"),
        "staging symlink": base.replace("staging_dir: dataset", "staging_dir: link/ds"),
        "imagery file": base.replace(
            base[base.index("imagery:") : base.index("labels:")],
            "imagery:\n  type: geotiff\n  path: ../outside/image.tif\n",
        ),
    }
    for name, text in cases.items():
        (project / f"{name.replace(' ', '_')}.yaml").write_text(text)

    async def scenario(client: Client) -> None:
        for name, text in cases.items():
            file = f"{name.replace(' ', '_')}.yaml"
            for tool, arguments in (
                ("validate_config", {"path": file}),
                ("validate_config", {"yaml_text": text}),
                ("plan", {"config": file}),
                ("generate", {"config": file}),
                ("write_config", {"path": f"w_{file}", "yaml_text": text}),
            ):
                result = await call(client, tool, **arguments)
                assert result.is_error and "outside the folder" in text_of(result), (name, tool)
        assert not (project / "w_labels_...yaml").exists()

    run_client(project, scenario)
    assert not list(tmp_path.glob("stolen")) and not (outside / "ds").exists()
    assert not list(project.glob("w_*"))


def test_a_symlink_inside_an_existing_output_folder_is_refused(
    project: Path, tmp_path: Path
) -> None:
    outside = tmp_path / "outside"
    outside.mkdir()
    staging = project / "dataset"
    staging.mkdir()
    (staging / "Images").symlink_to(outside, target_is_directory=True)

    async def scenario(client: Client) -> None:
        result = await call(client, "generate", config="mapcv.yaml")
        assert result.is_error and "outside the folder" in text_of(result)
        assert (await call(client, "split", dataset="dataset")).is_error

    run_client(project, scenario)
    assert not list(outside.iterdir())


# ── Safety: large jobs ───────────────────────────────────────────────────────


def test_large_jobs_are_refused_until_confirmed(project: Path, tile_server: str) -> None:
    west, south, _east, _north = _bbox()
    # About 44 700 tiles at zoom 18 (the limit is 20 000): planned, never fetched here.
    wide = (west, south, west + 0.3, south + 0.17)
    template = f"{tile_server}/{{z}}/{{x}}/{{y}}.png"
    (project / "large.yaml").write_text(config_text(template, "big", region=wide))

    async def scenario(client: Client) -> None:
        planned = data(await call(client, "plan", config="large.yaml"))
        assert planned["large"] is True
        assert planned["tiles"] > planning.LARGE_JOB_TILES
        assert "tiles" in planned["large_reason"]
        refused = await call(client, "generate", config="large.yaml")
        assert refused.is_error
        failure = data(refused)
        assert failure["confirmation_required"] is True
        assert failure["plan"]["large"] is True
        assert "confirm_large" in text_of(refused)
        explicit = await call(client, "generate", config="large.yaml", confirm_large=False)
        assert explicit.is_error

    run_client(project, scenario)
    assert not (project / "big").exists()
    assert not _Tiles.requests


def test_confirmed_large_job_starts(project: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(planning, "LARGE_JOB_TILES", 5)

    async def scenario(client: Client) -> None:
        planned = data(await call(client, "plan", config="mapcv.yaml"))
        assert planned["large"] is True and "limit" in planned["large_reason"]
        assert (await call(client, "generate", config="mapcv.yaml")).is_error
        ok = await call(client, "generate", config="mapcv.yaml", confirm_large=True)
        assert not ok.is_error, text_of(ok)

    run_client(project, scenario)
    assert (project / "dataset" / "manifest.json").exists()


# ── Safety: credentials ──────────────────────────────────────────────────────


def test_credentials_in_url_template_never_appear(project: Path, tile_server: str) -> None:
    template = f"{tile_server}/SECRETPATH/{{z}}/{{x}}/{{y}}.png?key=SECRET"
    failing = f"{tile_server}/FAIL-SECRETPATH/{{z}}/{{x}}/{{y}}.png?key=SECRET"
    (project / "secret.yaml").write_text(config_text(template, "secret_out"))
    (project / "failing.yaml").write_text(config_text(failing, "failing_out"))
    broken = config_text(template).replace("zoom: 18", "zoom: 99")
    texts: list[str] = []

    async def scenario(client: Client) -> None:
        results: list[CallToolResult] = []
        for tool, arguments in (
            ("validate_config", {"path": "secret.yaml"}),
            ("validate_config", {"yaml_text": broken}),
            ("validate_config", {"yaml_text": config_text(template) + "\n  : bad: [yaml"}),
            ("plan", {"config": "secret.yaml"}),
            ("plan", {"yaml_text": broken}),
            ("write_config", {"path": "w.yaml", "yaml_text": broken}),
            ("write_config", {"path": "w.yaml", "yaml_text": config_text(template)}),
            ("generate", {"config": "secret.yaml"}),
            ("generate", {"config": "failing.yaml"}),
            ("info", {"dataset": "secret_out"}),
            ("info", {"dataset": "failing_out"}),
            ("split", {"dataset": "secret_out"}),
            ("describe_config_schema", {}),
            ("inspect_labels", {"path": "secret.yaml"}),
        ):
            results.append(await call(client, tool, **arguments))
        texts.extend(text_of(result) for result in results)
        generated = results[7]
        assert not generated.is_error, text_of(generated)
        shown = data(results[0])["summary"]["imagery"]
        assert shown.startswith("XYZ http://127.0.0.1/")

    run_client(project, scenario)
    assert any("failed" in text for text in texts)  # the failing source was really exercised
    for text in texts:
        assert "SECRET" not in text


# ── Schema ───────────────────────────────────────────────────────────────────


def _model_paths(model: Any, prefix: str = "", seen: tuple[Any, ...] | None = None) -> list[str]:
    """Dotted names of every field of a pydantic model and of the models inside it."""
    from typing import get_args

    from pydantic import BaseModel

    seen = (*(seen or ()), model)
    paths: list[str] = []
    for field, info in model.model_fields.items():
        name = info.alias or field  # the schema (and the YAML) use the alias
        paths.append(f"{prefix}{name}")
        stack = [info.annotation]
        while stack:
            annotation = stack.pop()
            if (
                isinstance(annotation, type)
                and issubclass(annotation, BaseModel)
                and annotation not in seen
            ):
                paths.extend(_model_paths(annotation, f"{prefix}{name}.", seen))
            stack.extend(get_args(annotation))
    return paths


def _schema_paths(schema: dict[str, Any], node: dict[str, Any], prefix: str = "") -> list[str]:
    """The same dotted names, read from the JSON schema the tool returns."""
    definitions = schema.get("$defs", {})

    def resolve(part: dict[str, Any]) -> list[dict[str, Any]]:
        if "$ref" in part:
            return resolve(definitions[part["$ref"].rsplit("/", 1)[1]])
        found = [part]
        for key in ("anyOf", "oneOf", "allOf"):
            for option in part.get(key, []):
                found.extend(resolve(option))
        items = part.get("items") or part.get("additionalProperties")
        if isinstance(items, dict):
            found.extend(resolve(items))
        return found

    paths: list[str] = []
    for part in resolve(node):
        for name, child in part.get("properties", {}).items():
            paths.append(f"{prefix}{name}")
            paths.extend(_schema_paths(schema, child, f"{prefix}{name}."))
    return paths


def test_schema_tool_has_every_config_field(project: Path) -> None:
    from mapcv.config import MapcvConfig

    async def scenario(client: Client) -> dict[str, Any]:
        result = await call(client, "describe_config_schema", full_schema=True)
        assert not result.is_error
        return data(result)

    found = run_client(project, scenario, write=False)
    expected = set(_model_paths(MapcvConfig))
    assert {
        "task",
        "region.west",
        "imagery.zoom",
        "imagery.url_template",
        "writer.staging_dir",
        "labels.label_field",
        "labels.classes",
        "sampler.patch_size",
        "split.strategy",
        "detection.min_visible",
        "instance.id_mask",
    } <= expected
    actual = set(_schema_paths(found["schema"], found["schema"]))
    assert expected - actual == set()
    rules = found["rules"]
    assert rules["tasks"] == [
        "segmentation",
        "detection",
        "instance",
        "classification",
        "change",
        "regression",
    ]
    assert rules["multi_source_tasks"] == ["segmentation", "change", "regression"]
    assert rules["labels_allowed_per_task"]["regression"] == ["continuous"]
    assert rules["labels_allowed_per_task"]["segmentation"] == ["none", "vector", "raster"]
    assert rules["labels_allowed_per_task"]["change"] == ["vector", "raster"]
    assert rules["imagery_types"] == ["xyz", "eopf_zarr", "geotiff", "stac_cog"]
    assert rules["labels_allowed_per_task"]["segmentation"] == ["none", "vector", "raster"]
    assert rules["labels_allowed_per_task"]["detection"] == ["vector"]
    assert rules["labels_allowed_per_task"]["classification"] == ["vector", "raster"]
    assert "classification" in rules["task_options_block"]
    assert "npy" not in rules["image_formats_per_imagery"]["xyz"]
    assert "npy" in rules["image_formats_per_imagery"]["geotiff"]
    # Every invalid combination carries the message the validators give.
    assert all(item["error"] for item in rules["invalid_combinations"])
    assert found["required"]["MapcvConfig"] == ["region", "imagery", "sampler", "writer"]


# ── Cancellation ─────────────────────────────────────────────────────────────


def test_cancel_stops_after_a_chunk_and_the_run_resumes(project: Path, tmp_path: Path) -> None:
    _Tiles.delay = 0.15
    text = (
        (project / "mapcv.yaml")
        .read_text()
        .replace("max_connections: 4", "max_connections: 4\n  strip_rows: 1")
    )
    (project / "mapcv.yaml").write_text(text)
    (project / "reference.yaml").write_text(
        text.replace("staging_dir: dataset", "staging_dir: reference")
    )
    cancelled = threading.Event()

    async def scenario(client: Client) -> None:
        with anyio.CancelScope() as scope:

            async def on_progress(
                progress: float, total: float | None, message: str | None
            ) -> None:
                if progress >= 1 and total and progress < total:
                    cancelled.set()
                    scope.cancel()

            await client.call_tool(
                "generate", {"config": "mapcv.yaml"}, progress_callback=on_progress
            )
        assert cancelled.is_set()
        manifest = project / "dataset" / "manifest.json"
        # The worker finishes its chunk and stops; wait until it lets go of the folder.
        deadline = time.monotonic() + 30
        while True:
            result = await call(client, "generate", config="mapcv.yaml")
            if not result.is_error:
                break
            assert "still running" in text_of(result) or "cancelled" in text_of(result)
            assert time.monotonic() < deadline
            await anyio.sleep(0.2)
        # The cancelled run did not finish: this call still had patches to write.
        assert 0 < data(result)["new_patches"] < NX * NY
        final = json.loads(manifest.read_text())
        assert len(final["patches"]) == NX * NY
        reference = await call(client, "generate", config="reference.yaml")
        assert not reference.is_error

    run_client(project, scenario)
    partial_run, uninterrupted = tree(project / "dataset"), tree(project / "reference")
    assert partial_run == uninterrupted


# ── Over a real stdio pipe ───────────────────────────────────────────────────


def test_mapcv_mcp_serves_over_stdio(project: Path) -> None:
    """The command itself: progress bars and logs must not corrupt the protocol stream."""
    from mcp.client.stdio import StdioServerParameters

    parameters = StdioServerParameters(
        command=sys.executable,
        args=["-c", "from mapcv.cli import app; app()", "mcp", "--root", str(project)]
        + ["--allow-write", "--allow-local-urls"],
    )

    async def main() -> None:
        async with Client(parameters) as client:
            names = {tool.name for tool in (await client.list_tools()).tools}
            assert names == READ_TOOLS | WRITE_TOOLS
            events: list[float] = []

            async def on_progress(
                progress: float, total: float | None, message: str | None
            ) -> None:
                events.append(progress)

            result = await client.call_tool(
                "generate", {"config": "mapcv.yaml"}, progress_callback=on_progress
            )
            assert not result.is_error, text_of(result)
            assert events and events[-1] == 2
            assert (await client.call_tool("info", {"dataset": "dataset"})).is_error is False

    anyio.run(main)
    assert len(json.loads((project / "dataset" / "manifest.json").read_text())["patches"]) == 30


# ── What the model is told when a tool breaks ────────────────────────────────


def test_tool_exceptions_become_error_results(
    project: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from mapcv import agent_tools

    raised: list[Exception] = [
        ValueError("a user mistake"),
        agent_tools.ConfigInvalid("the config has errors", [{"field": "region", "message": "bad"}]),
        TypeError("internal detail with SECRET"),
    ]

    def boom(*args: Any) -> None:
        raise raised.pop(0)

    monkeypatch.setattr(agent_tools, "plan", boom)

    async def scenario(client: Client) -> None:
        mistake = await call(client, "plan", config="mapcv.yaml")
        assert mistake.is_error and "a user mistake" in text_of(mistake)
        invalid = await call(client, "plan", config="mapcv.yaml")
        assert invalid.is_error and data(invalid)["errors"] == [
            {"field": "region", "message": "bad"}
        ]
        crash = await call(client, "plan", config="mapcv.yaml")
        assert crash.is_error and "Internal error (TypeError)" in text_of(crash)
        assert "internal detail" not in text_of(crash)  # a bug's text stays in the server log

    run_client(project, scenario, write=False)


def test_a_failing_generate_step_is_reported(
    project: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from mapcv import agent_tools

    def boom(*args: Any) -> None:
        raise OSError("the disk is gone")

    monkeypatch.setattr(agent_tools, "prepare_generate", boom)

    async def scenario(client: Client) -> None:
        result = await call(client, "generate", config="mapcv.yaml")
        assert result.is_error and "the disk is gone" in text_of(result)

    run_client(project, scenario)


def test_serve_runs_over_stdio(project: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from mapcv import mcp_server

    seen: list[Any] = []

    class _Server:
        def run(self, transport: str) -> None:
            seen.append(transport)

    monkeypatch.setattr(
        mcp_server, "build_server", lambda root, allow_write, allow_local_urls: _Server()
    )
    mcp_server.serve(project, allow_write=True)
    # stdout carries the protocol; the library never prints (tests/test_python_api.py).
    assert seen == ["stdio"]


# ── Agent usability ──────────────────────────────────────────────────────────


def test_plan_runs_while_a_generate_is_running(project: Path) -> None:
    _Tiles.delay = 0.1
    text = (project / "mapcv.yaml").read_text().replace("max_connections: 4", "strip_rows: 1")
    (project / "mapcv.yaml").write_text(text)
    (project / "other.yaml").write_text(text.replace("staging_dir: dataset", "staging_dir: other"))
    outcome: dict[str, Any] = {}

    async def scenario(client: Client) -> None:
        started = anyio.Event()
        finished = anyio.Event()

        async def on_progress(progress: float, total: float | None, message: str | None) -> None:
            started.set()

        async def generate() -> None:
            outcome["generate"] = await client.call_tool(
                "generate", {"config": "mapcv.yaml"}, progress_callback=on_progress
            )
            finished.set()

        async with anyio.create_task_group() as group:
            group.start_soon(generate)
            with anyio.fail_after(60):
                await started.wait()
            planned = await call(client, "plan", config="other.yaml")
            outcome["plan"] = planned
            outcome["generate_running"] = not finished.is_set()
            # The other quick tools were never blocked.
            assert not (await call(client, "validate_config", path="other.yaml")).is_error

    run_client(project, scenario)
    assert not outcome["plan"].is_error, text_of(outcome["plan"])
    assert "busy" not in text_of(outcome["plan"])
    assert data(outcome["plan"])["tiles"] == NX * NY
    assert outcome["generate_running"] is True
    assert not outcome["generate"].is_error, text_of(outcome["generate"])
    assert data(outcome["generate"])["patches"] == NX * NY


def _generated(project: Path) -> None:
    async def scenario(client: Client) -> None:
        assert not (await call(client, "generate", config="mapcv.yaml")).is_error

    run_client(project, scenario)


def test_info_lists_every_source_and_whether_the_dataset_is_complete(project: Path) -> None:
    _generated(project)
    manifest_path = project / "dataset" / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    second = dict(manifest["sources"][0], name="after")
    manifest["sources"].append(second)
    manifest["complete"] = False
    manifest_path.write_text(json.dumps(manifest))

    async def scenario(client: Client) -> None:
        result = await call(client, "info", dataset="dataset")
        found = data(result)
        assert [source["name"] for source in found["sources"]][1] == "after"
        assert found["source"] == found["sources"][0]
        assert found["complete"] is False
        assert "Incomplete" in text_of(result) and "call generate again" in text_of(result)

    run_client(project, scenario, write=False)


def test_stats_and_verify_over_a_read_only_server(project: Path) -> None:
    _generated(project)
    before = tree(project / "dataset")

    async def scenario(client: Client) -> None:
        tools = {tool.name: tool for tool in (await client.list_tools()).tools}
        assert tools["stats"].annotations is not None and tools["stats"].annotations.read_only_hint
        stats = data(await call(client, "stats", dataset="dataset", split="all"))
        assert stats["patches"] == NX * NY and stats["saved"] is None
        weights = stats["classes"]["median_frequency_weights"]
        assert {"background", "building", "water"} <= set(weights)
        verified = data(await call(client, "verify", dataset="dataset", deep=True))
        assert verified["ok"] is True and verified["patches"] == NX * NY
        refused = await call(client, "stats", dataset="dataset", save=True)
        assert refused.is_error and "read-only" in text_of(refused)
        sums = await call(client, "verify", dataset="dataset", write_sums=True)
        assert sums.is_error and "--allow-write" in text_of(sums)

    run_client(project, scenario, write=False)
    assert tree(project / "dataset") == before  # nothing written: no stats.json, no SHA256SUMS


def test_stats_and_verify_write_only_when_asked_and_allowed(project: Path) -> None:
    _generated(project)

    async def scenario(client: Client) -> None:
        assert (project / "dataset" / "stats.json").exists() is False
        saved = data(await call(client, "stats", dataset="dataset", save=True))
        assert saved["saved"] == "dataset/stats.json"
        summed = data(await call(client, "verify", dataset="dataset", write_sums=True))
        assert summed["ok"] and summed["checksums_written"] == "dataset/SHA256SUMS"
        # A deleted patch is reported with its name.
        victim = next((project / "dataset" / "Images").iterdir())
        victim.unlink()
        broken = data(await call(client, "verify", dataset="dataset"))
        assert broken["ok"] is False and broken["problem_count"] >= 1
        assert any(victim.name in problem for problem in broken["problems"])

    run_client(project, scenario)
    assert (project / "dataset" / "stats.json").exists()
    assert (project / "dataset" / "SHA256SUMS").exists()


def test_stats_and_verify_refuse_manifests_that_point_outside(project: Path) -> None:
    _generated(project)
    manifest_path = project / "dataset" / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    entry = manifest["patches"][0]
    entry["files"]["image"] = "../../outside.png"
    manifest_path.write_text(json.dumps(manifest))

    async def scenario(client: Client) -> None:
        for tool in ("stats", "verify"):
            result = await call(client, tool, dataset="dataset")
            # Manifest.load refuses it already (the dataset path rule of mapcv._confine).
            assert result.is_error and "not a path inside the dataset folder" in text_of(result)

    run_client(project, scenario, write=False)


def test_instructions_say_which_mode_the_server_is_in(project: Path) -> None:
    async def scenario(client: Client) -> tuple[str, str]:
        instructions = client.instructions
        assert instructions is not None
        return instructions, ""

    read_only, _ = run_client(project, scenario, write=False)
    writing, _ = run_client(project, scenario, write=True)
    assert "read-only" in read_only and "do not exist" in read_only
    assert "--allow-write" in read_only
    assert "read and write" in writing and "do not exist" not in writing


def test_a_config_error_is_not_repeated_as_json(project: Path) -> None:
    broken = (project / "mapcv.yaml").read_text().replace("zoom: 18", "zoom: 99")

    async def scenario(client: Client) -> None:
        result = await call(client, "write_config", path="b.yaml", yaml_text=broken)
        assert result.is_error
        text = "\n".join(block.text for block in result.content if hasattr(block, "text"))
        assert text.count("imagery.zoom") == 1 and '"valid"' not in text
        assert data(result)["errors"][0]["field"] == "imagery.zoom"  # still structured

    run_client(project, scenario)


def test_osm_labels_validate_with_a_warning_and_plan_still_refuses(project: Path) -> None:
    osm = (project / "mapcv.yaml").read_text()
    start = osm.index("labels:")
    osm = (
        osm[:start]
        + ("labels:\n  osm:\n    classes:\n      - {name: building, tags: {building: '*'}}\n")
        + osm[osm.index("sampler:") :]
    )

    async def scenario(client: Client) -> None:
        checked = data(await call(client, "validate_config", yaml_text=osm))
        assert checked["valid"] is True and "labels.osm" in checked["warnings"][0]
        refused = await call(client, "plan", yaml_text=osm)
        assert refused.is_error and "labels.osm" in text_of(refused)

    run_client(project, scenario, write=False)


def test_a_single_class_is_named_with_a_label_file_class(project: Path) -> None:
    text = (project / "mapcv.yaml").read_text()
    named = text.replace(
        "labels:\n  path: labels.geojson\n  label_field: class",
        "labels:\n  files:\n    - {path: labels.geojson, class: building}",
    )
    assert named != text
    (project / "named.yaml").write_text(named.replace("staging_dir: dataset", "staging_dir: named"))

    async def scenario(client: Client) -> None:
        planned = data(await call(client, "plan", config="named.yaml"))
        assert planned["labels"]["classes"] == {"building": 1}
        assert not (await call(client, "generate", config="named.yaml")).is_error
        found = data(await call(client, "info", dataset="named"))
        assert found["classes"] == {"building": 1}
        assert {row["name"] for row in found["class_balance"]} == {"background", "building"}

    run_client(project, scenario)
