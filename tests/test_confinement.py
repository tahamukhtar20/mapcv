"""Reads and writes stay where they belong: links and paths in a dataset folder, STAC
catalogs, the MCP server's root, and credentials in what mapcv records or shows."""

from __future__ import annotations

import itertools
import json
import os
import threading
import time
import urllib.error
from collections.abc import Iterator
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any

import pytest
import yaml
from typer.testing import CliRunner

pytest.importorskip("rasterio", reason="the GeoTIFF is written with rasterio")
from test_geotiff_imagery import config_for, make_raster, write_labels

from mapcv import stac
from mapcv._confine import (
    LinkEscapeError,
    check_folder_links,
    first_outside_path,
    is_dataset_path,
)
from mapcv.cli import app
from mapcv.config import LabelsConfig, MapcvConfig, OsmLabelsSource
from mapcv.manifest import Manifest, ManifestMismatchError
from mapcv.pipeline import run_generate
from mapcv.verify import verify_dataset

runner = CliRunner()


def _config(folder: Path, staging: str = "dataset") -> Path:
    """A GeoTIFF + labels config file in ``folder`` (no network)."""
    folder.mkdir(parents=True, exist_ok=True)
    raster = make_raster(folder, width=320, height=256, count=3)
    region = raster.region()
    labels = write_labels(folder, region)
    config = config_for(folder, {"path": str(raster.path)}, region, labels=labels, staging=staging)
    data = config.model_dump(mode="json", exclude_unset=True)
    data["split"] = {"strategy": "random", "test_ratio": 0.2, "val_ratio": 0.1}
    path = folder / "mapcv.yaml"
    path.write_text(yaml.safe_dump(data), encoding="utf-8")
    return path


def _dataset(folder: Path) -> Path:
    config = MapcvConfig.from_yaml(_config(folder))
    run_generate(config)
    return Path(config.writer.staging_dir)


# ── Links in a dataset folder (D-1) ──────────────────────────────────────────


def test_generate_does_not_write_through_a_link_out_of_the_staging_dir(tmp_path: Path) -> None:
    config = _config(tmp_path)
    outside = tmp_path / "outside"
    outside.mkdir()
    (outside / "a.txt").write_text("precious\n")
    staging = tmp_path / "dataset"
    (staging / "Images").mkdir(parents=True)
    (staging / "Images" / "patch_0000000.png").symlink_to(outside / "a.txt")
    (staging / "splits").mkdir()
    (staging / "splits" / "train.txt").symlink_to(outside / "c.txt")

    result = runner.invoke(app, ["generate", str(config), "--yes"])

    assert result.exit_code == 1
    assert "Unsafe link" in result.output and "outside" in result.output
    assert (outside / "a.txt").read_text() == "precious\n"
    assert not (outside / "c.txt").exists()


@pytest.mark.parametrize(
    "command",
    [
        ["stats"],
        ["card", "--force"],
        ["split", "--strategy", "random"],
        ["verify", "--write-checksums"],
        ["export", "-f", "terratorch"],
    ],
)
def test_dataset_commands_refuse_a_link_out_of_the_dataset(
    tmp_path: Path, command: list[str]
) -> None:
    staging = _dataset(tmp_path)
    outside = tmp_path / "outside.txt"
    outside.write_text("precious\n")
    for name in ("stats.json", "README.md", "terratorch.yaml", "SHA256SUMS"):
        (staging / name).unlink(missing_ok=True)
        (staging / name).symlink_to(outside)

    result = runner.invoke(app, [command[0], str(staging), *command[1:]])

    assert result.exit_code == 1, result.output
    assert "Unsafe link" in result.output
    assert outside.read_text() == "precious\n"


def test_links_inside_the_dataset_are_fine(tmp_path: Path) -> None:
    staging = _dataset(tmp_path)
    (staging / "manifest_alias.json").symlink_to(staging / "manifest.json")
    result = runner.invoke(app, ["stats", str(staging)])
    assert result.exit_code == 0, result.output


def test_check_folder_links(tmp_path: Path) -> None:
    folder = tmp_path / "ds"
    (folder / "sub").mkdir(parents=True)
    (folder / "sub" / "a.txt").write_text("a")
    (folder / "inside").symlink_to(folder / "sub" / "a.txt")
    check_folder_links(folder, "ds")
    check_folder_links(tmp_path / "missing", "ds")  # nothing to check
    (tmp_path / "other.txt").write_text("b")
    os.link(tmp_path / "other.txt", folder / "sub" / "hard.txt")
    check_folder_links(folder, "ds")  # hard links count only when asked
    with pytest.raises(LinkEscapeError, match="sub/hard.txt is a hard link"):
        check_folder_links(folder, "ds", hard_links=True)
    (folder / "sub" / "hard.txt").unlink()
    (folder / "sub" / "out").symlink_to(tmp_path, target_is_directory=True)
    with pytest.raises(LinkEscapeError, match="sub/out is a link to"):
        check_folder_links(folder, "ds")


# ── Paths in a manifest and in SHA256SUMS (D-3) ──────────────────────────────


@pytest.mark.parametrize(
    "rel", ["Images/patch_0000000.png", "Images/s2/patch_0.npy", "Masks/x.png", "A/p.tif"]
)
def test_dataset_paths(rel: str) -> None:
    assert is_dataset_path(rel)


@pytest.mark.parametrize(
    "rel", ["", "/etc/passwd", "../x.png", "Images/../../x", "Images//x", "./x", "C:/x", "a\\b"]
)
def test_paths_that_leave_the_dataset(rel: str) -> None:
    assert not is_dataset_path(rel)
    assert first_outside_path(["Images/a.png", rel, "Masks/b.png"]) == rel


def test_the_fast_check_agrees_with_the_path_rule() -> None:
    """``first_outside_path`` scans a whole manifest at once; it must refuse exactly what
    ``is_dataset_path`` refuses."""
    alphabet = ["a", "/", ".", "..", "\\", ":", "\n", "\x00"]
    for length in range(5):
        for parts in itertools.product(alphabet, repeat=length):
            rel = "".join(parts)
            found = first_outside_path(["Images/x.png", rel])
            assert (found is None) == is_dataset_path(rel), repr(rel)
    assert first_outside_path([]) is None


@pytest.mark.parametrize("bad", ["{outside}", "../{name}"])
def test_a_manifest_path_outside_the_dataset_is_refused(tmp_path: Path, bad: str) -> None:
    staging = _dataset(tmp_path / "work")
    outside = tmp_path / "outside.txt"
    outside.write_text("OUTSIDE-FILE-CONTENT\n")
    path = bad.format(outside=outside, name=outside.name)
    manifest = staging / "manifest.json"
    first = Manifest.load(manifest).patches[0]["files"]["image"]
    manifest.write_text(manifest.read_text().replace(f'"{first}"', json.dumps(path), 1))

    with pytest.raises(ManifestMismatchError, match="not a path inside the dataset folder"):
        Manifest.load(manifest)
    report = verify_dataset(staging)
    assert not report.ok and "not a path inside the dataset folder" in report.problems[0]
    exported = runner.invoke(
        app, ["export", str(staging), "-f", "webdataset", "-o", str(tmp_path / "wds")]
    )
    assert exported.exit_code == 1
    shards = list((tmp_path / "wds").glob("*.tar"))
    assert not any(b"OUTSIDE-FILE-CONTENT" in shard.read_bytes() for shard in shards)


def test_a_checksum_path_outside_the_dataset_is_a_problem(tmp_path: Path) -> None:
    staging = _dataset(tmp_path)
    (staging / "SHA256SUMS").write_text("0000  /etc/hostname\n0000  ../mapcv.yaml\n")
    report = verify_dataset(staging)
    assert report.checked_hashes == 0
    assert report.problems == [
        "SHA256SUMS lists '/etc/hostname', which is not a path inside the dataset folder",
        "SHA256SUMS lists '../mapcv.yaml', which is not a path inside the dataset folder",
    ]


# ── STAC catalogs (E-13, E-14) ───────────────────────────────────────────────

_WORLD = {
    "type": "Polygon",
    "coordinates": [[[-180, -85], [180, -85], [180, 85], [-180, 85], [-180, -85]]],
}


def _item(href: str) -> dict[str, Any]:
    return {
        "type": "Feature",
        "id": "item",
        "geometry": _WORLD,
        "properties": {"datetime": "2024-06-01T00:00:00Z", "eo:cloud_cover": 1.0},
        "assets": {"red": {"href": href}, "green": {"href": href}, "blue": {"href": href}},
    }


@pytest.fixture
def catalog() -> Iterator[dict[str, Any]]:
    """A STAC API answering every search with ``state["page"]``."""
    state: dict[str, Any] = {"page": {"features": [], "links": []}}

    class Handler(BaseHTTPRequestHandler):
        def do_POST(self) -> None:
            self.rfile.read(int(self.headers.get("Content-Length", 0)))
            body = json.dumps(state["page"]).encode()
            self.send_response(200)
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, *args: Any) -> None:
            pass

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    state["url"] = f"http://127.0.0.1:{server.server_address[1]}"
    yield state
    server.shutdown()
    server.server_close()


def _search(url: str) -> Any:
    from mapcv.config import StacSearchBase

    return StacSearchBase.model_validate({"catalog": url, "datetime": "2024-06-01"})


@pytest.mark.parametrize(
    "href", ["file:///etc/passwd", "http://169.254.169.254/latest", "https://elsewhere.example/p"]
)
def test_a_next_link_stays_on_the_catalog_host(catalog: dict[str, Any], href: str) -> None:
    catalog["page"] = {"features": [], "links": [{"rel": "next", "href": href, "method": "GET"}]}
    with pytest.raises(RuntimeError, match="next links on the catalog's own scheme and host"):
        list(stac.search_items(_search(catalog["url"]), (0, 0, 1, 1)))


def test_a_relative_next_link_on_the_catalog_host_is_followed(
    catalog: dict[str, Any], monkeypatch: pytest.MonkeyPatch
) -> None:
    catalog["page"] = {
        "features": [],
        "links": [{"rel": "next", "href": "/search?page=2", "method": "POST"}],
    }
    pages = 0
    original = stac._post

    def counting(url: str, body: dict[str, Any]) -> dict[str, Any]:
        nonlocal pages
        pages += 1
        if pages > 1:
            assert url == catalog["url"] + "/search?page=2"
            return {"features": [], "links": []}
        return original(url, body)

    monkeypatch.setattr(stac, "_post", counting)
    assert list(stac.search_items(_search(catalog["url"]), (0, 0, 1, 1))) == []
    assert pages == 2


def test_local_assets_go_through_the_active_check(catalog: dict[str, Any], tmp_path: Path) -> None:
    catalog["page"] = {"features": [_item(str(tmp_path / "outside.tif"))], "links": []}
    search = _search(catalog["url"])
    assert stac.find_item(search, (0, 0, 1, 1))[0]["id"] == "item"  # no check: allowed
    checked: list[Path] = []

    def refuse(path: Path) -> None:
        checked.append(path)
        raise PermissionError(f"{path} refused")

    with (
        stac.local_paths_checked(refuse),
        pytest.raises(PermissionError, match="outside.tif refused"),
    ):
        stac.find_item(search, (0, 0, 1, 1))
    assert checked == [tmp_path / "outside.tif"]
    catalog["page"] = {"features": [_item("https://example.org/a.tif")], "links": []}
    with stac.local_paths_checked(refuse):
        stac.find_item(search, (0, 0, 1, 1))  # remote assets are not local files


class _Endless:
    """A response body that never ends (``delay`` seconds per read)."""

    def __init__(self, delay: float = 0.0) -> None:
        self.delay = delay

    def read1(self, size: int) -> bytes:
        time.sleep(self.delay)
        return b" " * min(size, 1 << 16)


def test_a_stac_page_has_a_size_cap(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(stac, "_MAX_PAGE_BYTES", 1 << 20)
    with pytest.raises(RuntimeError, match="sent more than 1 MiB for one page"):
        stac._read_json(_Endless(), "http://127.0.0.1:1/search")


def test_a_stac_page_has_a_deadline(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(stac, "_DEADLINE_S", 0.3)
    started = time.monotonic()
    with pytest.raises(RuntimeError, match="took longer than 0.3 s"):
        stac._read_json(_Endless(delay=0.1), "http://127.0.0.1:1/search")
    assert time.monotonic() - started < 2


def test_a_stac_page_must_be_a_json_object() -> None:
    class Once:
        def __init__(self, body: bytes) -> None:
            self.body = body

        def read1(self, size: int) -> bytes:
            body, self.body = self.body, b""
            return body

    with pytest.raises(RuntimeError, match="something other than JSON"):
        stac._read_json(Once(b"<html>"), "http://127.0.0.1:1/search")
    with pytest.raises(RuntimeError, match="not a STAC search page"):
        stac._read_json(Once(b"[1, 2]"), "http://127.0.0.1:1/search")


# ── The Overpass URL's key (M-7) ─────────────────────────────────────────────


def _osm(url: str) -> OsmLabelsSource:
    return OsmLabelsSource.model_validate(
        {"overpass_url": url, "classes": [{"name": "building", "tags": {"building": "*"}}]}
    )


def test_the_overpass_key_is_not_recorded(tmp_path: Path) -> None:
    from mapcv.targets.segmentation import label_settings

    url = "http://127.0.0.1:8705/api/interpreter?key=SUPERSECRET123&format=json"
    labels = LabelsConfig.model_validate({"osm": _osm(url).model_dump()})
    settings = label_settings(labels, set())
    assert settings["osm"]["overpass_url"] == (
        "http://127.0.0.1:8705/api/interpreter?key=***&format=***"
    )
    assert "SUPERSECRET123" not in json.dumps(settings)


def test_the_overpass_key_is_not_shown_in_errors(monkeypatch: pytest.MonkeyPatch) -> None:
    from mapcv import osm

    def fail(*args: Any, **kwargs: Any) -> None:
        raise urllib.error.URLError("connection refused")

    monkeypatch.setattr("mapcv.osm.urlopen", fail)
    source = _osm("http://127.0.0.1:1/api/interpreter?key=SUPERSECRET123")
    with pytest.raises(RuntimeError) as excinfo:
        osm._fetch(source, "[out:json];")
    assert "SUPERSECRET123" not in str(excinfo.value)
    assert "http://127.0.0.1:1/api/interpreter failed" in str(excinfo.value)
