"""``type: stac_cog``: Sentinel-2 bands as separate COGs of a STAC item.

A local STAC API names band COGs served by a local HTTP server that answers range
requests, as Earth Search's S3 bucket does. Bands of 10 m and 20 m and a 20 m scene
classification are written with rasterio on one UTM grid; every patch must equal the
10 m bands' pixels and the 20 m band's pixels repeated (nearest neighbour), and
masked scene classes must count as pixels without imagery.
"""

from __future__ import annotations

import json
import re
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any, Dict, Iterator

import numpy as np
import numpy.typing as npt
import pytest
from pydantic import ValidationError

pytest.importorskip("rasterio", reason="the band COGs are written with rasterio")
import rasterio  # noqa: E402
from pyproj import Transformer  # noqa: E402
from rasterio.transform import from_origin  # noqa: E402

from mapcv.config import MapcvConfig, StacCogImageryConfig  # noqa: E402
from mapcv.manifest import Manifest  # noqa: E402
from mapcv.pipeline import run_generate  # noqa: E402
from test_stac import Catalog, _item, _square  # noqa: E402

X0, Y0, EPSG = 600000.0, 5400000.0, 32631
W10, H10 = 160, 128


class Files:
    """Static files with HTTP range requests (what a COG reader needs)."""

    def __init__(self, folder: Path) -> None:
        self.folder = folder
        self.ranges = 0
        owner = self

        class Handler(BaseHTTPRequestHandler):
            def _send(self, body: bool) -> None:
                path = owner.folder / self.path.lstrip("/")
                if not path.is_file():
                    self.send_response(404)
                    self.end_headers()
                    return
                data = path.read_bytes()
                match = re.match(r"bytes=(\d+)-(\d*)", self.headers.get("Range", ""))
                if match:
                    owner.ranges += 1
                    start = int(match.group(1))
                    stop = min(len(data) - 1, int(match.group(2) or len(data) - 1))
                    part = data[start : stop + 1]
                    self.send_response(206)
                    self.send_header("Content-Range", f"bytes {start}-{stop}/{len(data)}")
                else:
                    part = data
                    self.send_response(200)
                self.send_header("Accept-Ranges", "bytes")
                self.send_header("Content-Length", str(len(part)))
                self.end_headers()
                if body:
                    self.wfile.write(part)

            def do_GET(self) -> None:  # noqa: N802
                self._send(True)

            def do_HEAD(self) -> None:  # noqa: N802
                self._send(False)

            def log_message(self, *args: Any) -> None:
                pass

        self.server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self.url = f"http://127.0.0.1:{self.server.server_address[1]}"
        threading.Thread(target=self.server.serve_forever, daemon=True).start()


def _cog(path: Path, data: npt.NDArray[Any], size: float) -> None:
    with rasterio.open(
        path,
        "w",
        driver="GTiff",
        height=data.shape[0],
        width=data.shape[1],
        count=1,
        dtype=str(data.dtype),
        crs=f"EPSG:{EPSG}",
        transform=from_origin(X0, Y0, size, size),
        tiled=True,
        blockxsize=64,
        blockysize=64,
        compress="deflate",
        nodata=0,
    ) as dst:
        dst.write(data, 1)


@pytest.fixture
def scene(tmp_path: Path) -> Iterator[Dict[str, Any]]:
    rng = np.random.default_rng(3)
    bands: Dict[str, npt.NDArray[Any]] = {
        "red": rng.integers(1, 9000, (H10, W10)).astype(np.uint16),
        "green": rng.integers(1, 9000, (H10, W10)).astype(np.uint16),
        "swir16": rng.integers(1, 9000, (H10 // 2, W10 // 2)).astype(np.uint16),
        "scl": rng.integers(0, 12, (H10 // 2, W10 // 2)).astype(np.uint8),
    }
    folder = tmp_path / "cogs"
    folder.mkdir()
    for name, data in bands.items():
        _cog(folder / f"{name}.tif", data, 10.0 if data.shape == (H10, W10) else 20.0)
    files = Files(folder)
    catalog = Catalog([])
    to_lonlat = Transformer.from_crs(f"EPSG:{EPSG}", "EPSG:4326", always_xy=True)
    west, north = to_lonlat.transform(X0 + 10 * 8 + 1, Y0 - 10 * 8 - 1)
    east, south = to_lonlat.transform(X0 + 10 * 140 - 1, Y0 - 10 * 112 + 1)
    region = {"west": west, "south": south, "east": east, "north": north}
    item = _item("S2A_31UFT_20250509_0_L2A", 3.0, "2025-05-09T10:40:00Z")
    item["geometry"] = _square(west - 0.1, south - 0.1, east + 0.1, north + 0.1)
    item["assets"] = {name: {"href": f"{files.url}/{name}.tif"} for name in bands}
    catalog.items = [item]
    yield {"bands": bands, "region": region, "catalog": catalog, "files": files}
    catalog.close()
    files.server.shutdown()
    files.server.server_close()


def _config(tmp_path: Path, scene: Dict[str, Any], **imagery: Any) -> MapcvConfig:
    return MapcvConfig.model_validate(
        {
            "region": scene["region"],
            "imagery": {
                "type": "stac_cog",
                "search": {"catalog": scene["catalog"].url, "datetime": "2025-05-01/2025-05-31"},
                **imagery,
            },
            "sampler": {"patch_size": 32, "edge_strategy": "drop", "max_empty_ratio": 1.0},
            "writer": {"staging_dir": str(tmp_path / "dataset"), "image_format": "npy"},
        }
    )


def test_bands_of_two_resolutions_and_masked_clouds(tmp_path: Path, scene: Dict[str, Any]) -> None:
    run_generate(_config(tmp_path, scene, bands=["swir16", "red", "green"], scl_mask=[8, 9, 10]))
    manifest = Manifest.load(tmp_path / "dataset" / "manifest.json")
    source = manifest.source
    assert (source.source_type, source.product_id, source.dtype) == (
        "stac_cog",
        "S2A_31UFT_20250509_0_L2A",
        "uint16",
    )
    assert source.bands == ["swir16", "red", "green"] and source.crs == f"EPSG:{EPSG}"
    assert source.fingerprint == {
        "stac": {
            "catalog": scene["catalog"].url,
            "collection": "sentinel-2-l2a",
            "item": "S2A_31UFT_20250509_0_L2A",
        },
        "scl_mask": [8, 9, 10],
    }
    bands = scene["bands"]
    swir = np.repeat(np.repeat(bands["swir16"], 2, 0), 2, 1)
    scl = np.repeat(np.repeat(bands["scl"], 2, 0), 2, 1)
    assert len(manifest.patches) >= 9 and scene["files"].ranges > 0
    for entry in manifest.patches:
        _, _, c, _, _, f = manifest.patch_transform(entry)
        col, row = round((c - X0) / 10), round((Y0 - f) / 10)
        window = (slice(row, row + 32), slice(col, col + 32))
        patch = np.load(tmp_path / "dataset" / entry["files"]["image"])
        np.testing.assert_array_equal(patch[0], swir[window])
        np.testing.assert_array_equal(patch[1], bands["red"][window])
        np.testing.assert_array_equal(patch[2], bands["green"][window])
        # Masked classes, and class 0 (SCL's no data, also the COG's NoData value).
        masked = np.isin(scl[window], [0, 8, 9, 10])
        assert entry["summary"]["empty_ratio"] == pytest.approx(float(masked.mean()))


def test_refusals(tmp_path: Path, scene: Dict[str, Any], monkeypatch: pytest.MonkeyPatch) -> None:
    with pytest.raises(ValueError, match="has no 'nir' asset; it has: green, red, scl, swir16"):
        run_generate(_config(tmp_path, scene, bands=["red", "nir"]))
    item = scene["catalog"].items[0]
    _cog(scene["files"].folder / "float.tif", np.ones((H10, W10), np.float32), 10.0)
    item["assets"]["float"] = {"href": f"{scene['files'].url}/float.tif"}
    with pytest.raises(ValueError, match="different data types"):
        run_generate(_config(tmp_path, scene, bands=["red", "float"]))
    item["assets"]["red"] = {"href": "/etc/secret.tif"}
    from mapcv.config import RegionConfig
    from mapcv.imagery import StacCogRasterSource

    remote = StacCogImageryConfig.model_validate(
        {
            "type": "stac_cog",
            "search": {"datetime": "2025-05-01", "catalog": "https://stac.example.org"},
        }
    )
    monkeypatch.setattr("mapcv.stac.find_item", lambda search, bbox: (item, 1))
    with pytest.raises(ValueError, match="may only point to https:// or s3:// files"):
        StacCogRasterSource(RegionConfig.model_validate(scene["region"]), remote)


@pytest.mark.parametrize(
    ("imagery", "writer", "message"),
    [
        ({"bands": ["red", "red"]}, {}, "must not contain duplicates"),
        ({"bands": []}, {}, "band assets"),
        ({"scl_mask": [13]}, {}, "scene classes 0-11"),
        ({}, {"image_format": "png"}, "requires writer.image_format='npy'"),
    ],
)
def test_config_refusals(
    tmp_path: Path, imagery: Dict[str, Any], writer: Dict[str, Any], message: str
) -> None:
    data = {
        "region": {"west": 3.0, "south": 48.0, "east": 3.01, "north": 48.01},
        "imagery": {"type": "stac_cog", "search": {"datetime": "2025-05-01"}, **imagery},
        "sampler": {"patch_size": 32},
        "writer": {"staging_dir": str(tmp_path / "d"), "image_format": "npy", **writer},
    }
    with pytest.raises(ValidationError, match=message):
        MapcvConfig.model_validate(data)
    defaults = StacCogImageryConfig.model_validate(
        {"type": "stac_cog", "search": {"datetime": "2025-05-01"}}
    )
    assert defaults.search.catalog == "https://earth-search.aws.element84.com/v1"
    assert defaults.bands == ["red", "green", "blue", "nir"]


def test_plan_label_and_card(tmp_path: Path, scene: Dict[str, Any]) -> None:
    from typer.testing import CliRunner

    import yaml
    from mapcv.card import card_text
    from mapcv.cli import app
    from mapcv.planning import plan

    config = _config(tmp_path, scene, bands=["red", "green"])
    estimate = plan(config)
    assert estimate.resolution_m == 10.0 and "Sentinel-2 COGs" in estimate.imagery
    path = tmp_path / "c.yaml"
    path.write_text(yaml.safe_dump(config.model_dump(mode="json", exclude_none=True)))
    result = CliRunner().invoke(app, ["validate", str(path)], env={"COLUMNS": "200"})
    assert result.exit_code == 0 and "Sentinel-2 COGs · search sentinel-2-l2a" in result.output
    run_generate(config)
    assert "Copernicus Sentinel" in card_text(tmp_path / "dataset")
    json.loads((tmp_path / "dataset" / "manifest.json").read_text())
