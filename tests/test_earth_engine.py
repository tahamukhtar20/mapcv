"""Earth Engine imagery (``imagery.earth_engine``) with a stand-in ``ee`` module.

The stand-in records what mapcv asks Earth Engine for and hands out map URLs on a
local tile server laid out like Earth Engine's (``/v1/projects/<p>/maps/<map id>/tiles/
{z}/{x}/{y}``). The map ID must never reach the manifest, the dataset or messages.
"""

from __future__ import annotations

import builtins
import sys
import threading
from collections.abc import Iterator
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from io import BytesIO
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
from PIL import Image
from pydantic import ValidationError

from mapcv import earth_engine
from mapcv.cli import app
from mapcv.config import EarthEngineImageryConfig, MapcvConfig
from mapcv.manifest import Manifest, ManifestMismatchError
from mapcv.pipeline import run_generate

REGION = {"west": 4.8900, "south": 52.3700, "east": 4.8990, "north": 52.3750}
MAP_ID = "0123456789abcdef-SECRETMAPID"


class Tiles(BaseHTTPRequestHandler):
    def do_GET(self) -> None:
        parts = self.path.strip("/").split("/")
        if any(part.startswith("expired") for part in parts):
            self.send_response(404)
            self.end_headers()
            return
        z, x, y = (int(part) for part in parts[-3:])
        buffer = BytesIO()
        Image.new("RGB", (256, 256), (x % 251, y % 241, z)).save(buffer, "PNG")
        body = buffer.getvalue()
        self.send_response(200)
        self.send_header("Content-Type", "image/png")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, *args: Any) -> None:
        pass


@pytest.fixture
def server() -> Iterator[str]:
    httpd = ThreadingHTTPServer(("127.0.0.1", 0), Tiles)
    threading.Thread(target=httpd.serve_forever, daemon=True).start()
    yield f"http://127.0.0.1:{httpd.server_address[1]}"
    httpd.shutdown()
    httpd.server_close()


class FakeEE:
    """The parts of ``ee`` mapcv uses, recording the calls."""

    def __init__(self, base: str, map_id: str = MAP_ID, fail: str | None = None) -> None:
        self.base, self.map_id, self.fail = base, map_id, fail
        self.calls: list[Any] = []

    def Initialize(self, project: str | None = None) -> None:
        if self.fail:
            raise RuntimeError(self.fail)
        self.calls.append(("Initialize", project))

    def _image(self, description: Any) -> Any:
        fake = self

        class Image:
            def getMapId(self, vis: dict[str, Any]) -> dict[str, Any]:
                fake.calls.append(("getMapId", description, vis))
                url = f"{fake.base}/v1/projects/p/maps/{fake.map_id}/tiles/{{z}}/{{x}}/{{y}}"
                return {"tile_fetcher": SimpleNamespace(url_format=url)}

        return Image()

    def Image(self, asset: str) -> Any:
        return self._image(("Image", asset))

    Geometry = SimpleNamespace(Rectangle=lambda coords: ("Rectangle", tuple(coords)))
    Filter = SimpleNamespace(lte=lambda name, value: ("lte", name, value))

    def ImageCollection(self, asset: str) -> Any:
        fake = self

        class Collection:
            def __init__(self, steps: list[Any]) -> None:
                self.steps = steps

            def filterDate(self, start: str, end: str) -> Any:
                return Collection([*self.steps, ("filterDate", start, end)])

            def filterBounds(self, geometry: Any) -> Any:
                return Collection([*self.steps, ("filterBounds", geometry)])

            def filter(self, condition: Any) -> Any:
                return Collection([*self.steps, ("filter", condition)])

            def linkCollection(self, other: Any, bands: list[str]) -> Any:
                return Collection([*self.steps, ("linkCollection", other.steps, bands)])

            def map(self, function: Any) -> Any:
                return Collection([*self.steps, ("map", function(_FakePixels()))])

            def __getattr__(self, reducer: str) -> Any:
                return lambda: fake._image(("ImageCollection", asset, *self.steps, reducer))

        return Collection([])


class _FakePixels:
    """An image inside ``collection.map``: records what is done with its bands."""

    def select(self, band: str) -> Any:
        return SimpleNamespace(gte=lambda value: ("gte", band, value))

    def updateMask(self, mask: Any) -> Any:
        return ("updateMask", mask)


def _engine(**changes: Any) -> dict[str, Any]:
    engine: dict[str, Any] = {
        "image": "USGS/NAIP/DOQQ/m_4207148_nw_19_060_20180901",
        "vis": {"bands": ["R", "G", "B"], "min": 0, "max": 255},
        "project": "my-project",
    }
    engine.update(changes)
    return engine


def _config(tmp_path: Path, policy: str = "lenient", **engine: Any) -> MapcvConfig:
    return MapcvConfig.model_validate(
        {
            "region": REGION,
            "imagery": {
                "type": "xyz",
                "zoom": 16,
                "policy": policy,
                "earth_engine": _engine(**engine),
            },
            "sampler": {"patch_size": 256, "edge_strategy": "drop"},
            "writer": {"staging_dir": str(tmp_path / "dataset")},
        }
    )


# ── config ───────────────────────────────────────────────────────────────────


@pytest.mark.parametrize(
    ("change", "message"),
    [
        ({"collection": "COPERNICUS/S2_SR_HARMONIZED"}, "'image' or 'collection', not both"),
        ({"image": None}, "'image' or 'collection', not both"),
        ({"start": "2024-01-01"}, "filter a 'collection'"),
        ({"vis": {"bands": ["B4", "B3"]}}, "1 or 3 bands"),
        ({"vis": {"bands": ["B4", "B3", "B2"], "palette": ["red"]}}, "exactly one band"),
        ({"image": None, "collection": "C", "start": "June"}, "a date like 2024-06-01"),
        ({"vis": {"bands": ["B4"], "stretch": 1}}, "Extra inputs are not permitted"),
    ],
)
def test_invalid_settings_are_refused(change: dict[str, Any], message: str) -> None:
    with pytest.raises(ValidationError, match=message):
        EarthEngineImageryConfig.model_validate(_engine(**change))


def test_earth_engine_is_a_third_way_to_name_xyz_tiles(tmp_path: Path) -> None:
    with pytest.raises(ValidationError, match="set 'source' or 'earth_engine', not both"):
        MapcvConfig.model_validate(
            {
                "region": REGION,
                "imagery": {
                    "type": "xyz",
                    "zoom": 16,
                    "source": "esri_satellite",
                    "earth_engine": _engine(),
                },
                "sampler": {"patch_size": 256},
                "writer": {"staging_dir": str(tmp_path)},
            }
        )


def test_visualization_parameters() -> None:
    vis = EarthEngineImageryConfig.model_validate(
        _engine(vis={"bands": ["B4", "B3", "B2"], "min": [0, 0, 0], "max": 3000, "gamma": 1.4})
    ).vis.params()
    assert vis == {"bands": "B4,B3,B2", "min": "0.0,0.0,0.0", "max": "3000.0", "gamma": "1.4"}
    one = EarthEngineImageryConfig.model_validate(
        _engine(vis={"bands": ["NDVI"], "min": -1, "max": 1, "palette": ["blue", "ffffff"]})
    ).vis.params()
    assert one["palette"] == "blue,ffffff" and one["bands"] == "NDVI"


# ── tile URLs ────────────────────────────────────────────────────────────────


def test_an_image_and_a_reduced_collection(monkeypatch: pytest.MonkeyPatch) -> None:
    fake = FakeEE("http://x")
    monkeypatch.setitem(sys.modules, "ee", fake)
    url = earth_engine.tile_url(EarthEngineImageryConfig.model_validate(_engine()))
    assert url == f"http://x/v1/projects/p/maps/{MAP_ID}/tiles/{{z}}/{{x}}/{{y}}"
    assert fake.calls[0] == ("Initialize", "my-project")
    assert fake.calls[1][1] == ("Image", "USGS/NAIP/DOQQ/m_4207148_nw_19_060_20180901")

    fake.calls.clear()
    config = EarthEngineImageryConfig.model_validate(
        _engine(
            image=None,
            collection="COPERNICUS/S2_SR_HARMONIZED",
            start="2024-06-01",
            reducer="mosaic",
        )
    )
    earth_engine.tile_url(config)
    steps = fake.calls[1][1]
    assert steps == (
        "ImageCollection",
        "COPERNICUS/S2_SR_HARMONIZED",
        ("filterDate", "2024-06-01", "2100-01-01"),
        "mosaic",
    )
    assert earth_engine.product_id(config) == "earth-engine:COPERNICUS/S2_SR_HARMONIZED"


def test_clouds_are_filtered_and_masked_over_the_region(monkeypatch: pytest.MonkeyPatch) -> None:
    fake = FakeEE("http://x")
    monkeypatch.setitem(sys.modules, "ee", fake)
    config = EarthEngineImageryConfig.model_validate(
        _engine(
            image=None,
            collection="COPERNICUS/S2_SR_HARMONIZED",
            start="2024-06-01",
            end="2024-09-01",
            max_cloud=40,
            cloud_score_plus=0.6,
        )
    )
    earth_engine.tile_url(config, (4.9, 52.3, 5.0, 52.4))
    steps = fake.calls[1][1]
    assert steps == (
        "ImageCollection",
        "COPERNICUS/S2_SR_HARMONIZED",
        ("filterBounds", ("Rectangle", (4.9, 52.3, 5.0, 52.4))),
        ("filterDate", "2024-06-01", "2024-09-01"),
        ("filter", ("lte", "CLOUDY_PIXEL_PERCENTAGE", 40.0)),
        ("linkCollection", [], ["cs_cdf"]),
        ("map", ("updateMask", ("gte", "cs_cdf", 0.6))),
        "median",
    )


@pytest.mark.parametrize(
    ("collection", "expected"),
    [
        ("COPERNICUS/S2_SR_HARMONIZED", "CLOUDY_PIXEL_PERCENTAGE"),
        ("LANDSAT/LC08/C02/T1_L2", "CLOUD_COVER"),
    ],
)
def test_the_cloud_property_is_detected(collection: str, expected: str) -> None:
    config = EarthEngineImageryConfig.model_validate(
        _engine(image=None, collection=collection, max_cloud=20)
    )
    assert config.cloud_filter_property == expected


@pytest.mark.parametrize(
    ("change", "message"),
    [
        ({"max_cloud": 20}, "filter a 'collection'"),
        ({"image": None, "collection": "MY/COLLECTION", "max_cloud": 20}, "set cloud_property"),
        (
            {"image": None, "collection": "LANDSAT/LC08/C02/T1_L2", "cloud_score_plus": 0.6},
            "Sentinel-2 collections",
        ),
        ({"image": None, "collection": "COPERNICUS/S2", "cloud_score_plus": 1.5}, "less than 1"),
    ],
)
def test_cloud_settings_are_checked(change: dict[str, Any], message: str) -> None:
    with pytest.raises(ValidationError, match=message):
        EarthEngineImageryConfig.model_validate(_engine(**change))
    custom = _engine(image=None, collection="MY/COLLECTION", max_cloud=5, cloud_property="CLOUDS")
    assert EarthEngineImageryConfig.model_validate(custom).cloud_filter_property == "CLOUDS"


def test_the_template_is_valid_and_the_wizard_writes_each_preset(tmp_path: Path) -> None:
    from typer.testing import CliRunner

    from mapcv.config import MapcvConfig

    runner = CliRunner()
    out = tmp_path / "template.yaml"
    assert runner.invoke(app, ["init", str(out), "--template", "earth-engine"]).exit_code == 0
    engine = MapcvConfig.from_yaml(out).primary_imagery.earth_engine  # type: ignore[union-attr]
    assert engine is not None and engine.max_cloud == 40 and engine.cloud_score_plus == 0.6

    bbox = "4.9375,52.3725,4.9515,52.3780"
    for dataset, collection in (
        ("sentinel2", "COPERNICUS/S2_SR_HARMONIZED"),
        ("landsat", "LANDSAT/LC08/C02/T1_L2"),
        ("naip", "USDA/NAIP/DOQQ"),
    ):
        answers = ["gee", bbox, dataset, "", ""]
        if dataset != "naip":
            answers.append("")  # max_cloud
        answers += ["my-project", "", "", "", "256", "./ds", "y"]
        out = tmp_path / f"{dataset}.yaml"
        result = runner.invoke(
            app, ["init", str(out), "--interactive"], input="\n".join(answers) + "\n"
        )
        assert result.exit_code == 0, result.output
        imagery = MapcvConfig.from_yaml(out).primary_imagery
        assert imagery.earth_engine is not None  # type: ignore[union-attr]
        assert imagery.earth_engine.collection == collection  # type: ignore[union-attr]
        assert imagery.earth_engine.project == "my-project"  # type: ignore[union-attr]

    custom = ["gee", bbox, "custom", "image", "USGS/SRTMGL1_003", "elevation", "0", "500", "30"]
    custom += ["", "", "", "", "256", "./ds", "y"]
    out = tmp_path / "custom.yaml"
    result = runner.invoke(app, ["init", str(out), "--interactive"], input="\n".join(custom) + "\n")
    assert result.exit_code == 0, result.output
    engine = MapcvConfig.from_yaml(out).primary_imagery.earth_engine  # type: ignore[union-attr]
    assert engine is not None and engine.image == "USGS/SRTMGL1_003"
    assert engine.vis.bands == ["elevation"]
    assert engine.project == "YOUR-CLOUD-PROJECT"  # left blank: a placeholder to fill in


def test_a_refusal_says_how_to_log_in(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setitem(sys.modules, "ee", FakeEE("http://x", fail="Please authorize access"))
    config = EarthEngineImageryConfig.model_validate(_engine())
    with pytest.raises(RuntimeError, match="earthengine authenticate") as raised:
        earth_engine.tile_url(config)
    assert "Please authorize access" in str(raised.value)


def test_without_the_extra(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delitem(sys.modules, "ee", raising=False)
    real_import = builtins.__import__

    def blocked(name: str, *args: Any, **kwargs: Any) -> Any:
        if name == "ee":
            raise ImportError("No module named 'ee'")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", blocked)
    with pytest.raises(RuntimeError, match=r'pip install "mapcv\[gee\]"'):
        earth_engine.tile_url(EarthEngineImageryConfig.model_validate(_engine()))


# ── generate ─────────────────────────────────────────────────────────────────


def _files_text(root: Path) -> str:
    return "\n".join(
        path.read_bytes().decode("latin-1") for path in sorted(root.rglob("*")) if path.is_file()
    )


def test_a_dataset_records_the_asset_never_the_map_id(
    tmp_path: Path, server: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setitem(sys.modules, "ee", FakeEE(server))
    config = _config(tmp_path)
    result = run_generate(config)
    manifest = Manifest.load(config.writer.staging_dir / "manifest.json")
    assert len(manifest.patches) == len(result.manifest.patches) > 0
    source = manifest.source
    assert source.product_id == "earth-engine:USGS/NAIP/DOQQ/m_4207148_nw_19_060_20180901"
    assert MAP_ID not in _files_text(config.writer.staging_dir)

    # A later run gets a new map ID for the same image: it resumes (nothing to do).
    monkeypatch.setitem(sys.modules, "ee", FakeEE(server, map_id="another-map-id"))
    assert run_generate(config).manifest.patches == manifest.patches
    # Other rendering is other data: it is refused.
    other = _config(tmp_path, vis={"bands": ["R", "G", "B"], "min": 0, "max": 100})
    with pytest.raises(ManifestMismatchError):
        run_generate(other)


def test_failed_tiles_are_reported_without_the_map_id(
    tmp_path: Path, server: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setitem(sys.modules, "ee", FakeEE(server, map_id="expired-SECRETMAPID"))
    config = _config(tmp_path, policy="strict")
    with pytest.raises(Exception, match="HTTP 404") as raised:
        run_generate(config)
    assert "SECRETMAPID" not in str(raised.value) and "127.0.0.1" in str(raised.value)


def test_the_plan_and_card_name_the_asset(
    tmp_path: Path, server: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    from typer.testing import CliRunner

    monkeypatch.setitem(sys.modules, "ee", FakeEE(server))
    config_path = tmp_path / "mapcv.yaml"
    config_path.write_text(
        "region: {west: 4.89, south: 52.37, east: 4.899, north: 52.375}\n"
        "imagery:\n  type: xyz\n  zoom: 16\n  earth_engine:\n"
        "    image: USGS/NAIP/DOQQ/m_4207148_nw_19_060_20180901\n"
        "    vis: {bands: [R, G, B], min: 0, max: 255}\n"
        f"sampler: {{patch_size: 256, edge_strategy: drop}}\nwriter: {{staging_dir: {tmp_path / 'd'}}}\n"
    )
    runner = CliRunner()
    planned = runner.invoke(app, ["plan", str(config_path)])
    assert planned.exit_code == 0, planned.output
    assert "Earth Engine USGS/NAIP/DOQQ/m_4207148_nw_19_060_20180901" in " ".join(
        planned.output.split()
    )
    generated = runner.invoke(app, ["generate", str(config_path), "--yes"])
    assert generated.exit_code == 0, generated.output
    assert MAP_ID not in generated.output
    card = runner.invoke(app, ["card", str(tmp_path / "d")])
    assert card.exit_code == 0, card.output
    assert "Google Earth Engine" in (tmp_path / "d" / "README.md").read_text(encoding="utf-8")
