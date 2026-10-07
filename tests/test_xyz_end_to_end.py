"""XYZ imagery, vector labels and a spatial split, end to end over real HTTP.

A local tile server paints every pixel from its global Web Mercator pixel position, so
each written patch can be checked pixel for pixel against where the manifest says it
is. Masks are compared with rasterio's rasterization of the labels (projected with
pyproj) on each patch's grid, and the split lists must be disjoint and free of
overlapping patches across splits.
"""

from __future__ import annotations

import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from io import BytesIO
from pathlib import Path
from typing import Any, Dict, Iterator, List

import numpy as np
import numpy.typing as npt
import pytest
from PIL import Image

pytest.importorskip("rasterio", reason="masks are compared with rasterio's rasterization")
from pyproj import Transformer  # noqa: E402
from rasterio.features import rasterize  # noqa: E402
from rasterio.transform import Affine  # noqa: E402
from shapely.geometry import Polygon, shape  # noqa: E402
from shapely.ops import transform as shapely_transform  # noqa: E402

from mapcv.config import MapcvConfig  # noqa: E402
from mapcv.manifest import Manifest  # noqa: E402
from mapcv.pipeline import run_generate  # noqa: E402

ZOOM = 17


def _pixels(x: int, y: int) -> npt.NDArray[np.uint8]:
    """Tile (x, y): every pixel's colour encodes its global pixel column and row."""
    rows, cols = np.mgrid[0:256, 0:256]
    gx, gy = x * 256 + cols, y * 256 + rows
    return np.stack([gx % 251, gy % 241, (gx + gy) % 239], axis=-1).astype(np.uint8)


class Tiles(BaseHTTPRequestHandler):
    def do_GET(self) -> None:  # noqa: N802 - http.server's name
        _, x, y = (int(part) for part in self.path.strip("/").removesuffix(".png").split("/"))
        buffer = BytesIO()
        Image.fromarray(_pixels(x, y)).save(buffer, "PNG")
        body = buffer.getvalue()
        self.send_response(200)
        self.send_header("Content-Type", "image/png")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, *args: Any) -> None:
        pass


@pytest.fixture
def template() -> Iterator[str]:
    server = ThreadingHTTPServer(("127.0.0.1", 0), Tiles)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    yield f"http://127.0.0.1:{server.server_address[1]}/{{z}}/{{x}}/{{y}}.png"
    server.shutdown()
    server.server_close()


REGION = {"west": 4.8900, "south": 52.3700, "east": 4.8990, "north": 52.3750}


def _labels(path: Path) -> List[Dict[str, Any]]:
    west, south = REGION["west"], REGION["south"]
    dx, dy = REGION["east"] - west, REGION["north"] - south

    def ring(fx: float, fy: float, w: float, h: float) -> List[List[float]]:
        x0, y0 = west + fx * dx, south + fy * dy
        x1, y1 = x0 + w * dx, y0 + h * dy
        return [[x0, y0], [x1, y0], [x1, y1], [x0, y1], [x0, y0]]

    features = [
        {"type": "Feature", "properties": {"kind": kind}, "geometry": geometry}
        for kind, geometry in [
            ("house", {"type": "Polygon", "coordinates": [ring(0.1, 0.1, 0.3, 0.4)]}),
            (
                "park",
                {
                    "type": "Polygon",
                    "coordinates": [ring(0.5, 0.2, 0.4, 0.6), ring(0.6, 0.3, 0.1, 0.1)],
                },
            ),
            ("house", {"type": "Polygon", "coordinates": [ring(0.35, 0.55, 0.3, 0.3)]}),
        ]
    ]
    path.write_text(json.dumps({"type": "FeatureCollection", "features": features}))
    return features


@pytest.mark.parametrize("image_format", ["png", "tif"])
def test_xyz_labels_and_spatial_split(tmp_path: Path, template: str, image_format: str) -> None:
    features = _labels(tmp_path / "labels.geojson")
    config = MapcvConfig.model_validate(
        {
            "region": REGION,
            "imagery": {"type": "xyz", "zoom": ZOOM, "url_template": template, "cache": False},
            "labels": {
                "path": str(tmp_path / "labels.geojson"),
                "label_field": "kind",
                "classes": {"house": 1, "park": 2},
            },
            "sampler": {"patch_size": 128, "stride": 96, "edge_strategy": "shift"},
            "writer": {"staging_dir": str(tmp_path / "dataset"), "image_format": image_format},
            "split": {"strategy": "spatial", "test_ratio": 0.25, "val_ratio": 0.2, "seed": 7},
        }
    )
    result = run_generate(config)
    staging = tmp_path / "dataset"
    manifest = Manifest.load(staging / "manifest.json")
    assert result.tiles_failed == 0 and len(manifest.patches) > 10

    to_mercator = Transformer.from_crs("EPSG:4326", "EPSG:3857", always_xy=True).transform
    shapes = [
        (
            shapely_transform(to_mercator, shape(f["geometry"])),
            {"house": 1, "park": 2}[f["properties"]["kind"]],
        )
        for f in features
    ]
    assert all(isinstance(geometry, Polygon) for geometry, _ in shapes)
    origin = manifest.source.transform
    assert origin is not None
    tile_x0 = round((origin[2] + 20037508.342789244) / (origin[0] * 256))
    tile_y0 = round((20037508.342789244 - origin[5]) / (origin[0] * 256))

    labelled = 0
    for entry in manifest.patches:
        patch = Affine(*manifest.patch_transform(entry))
        image = _read(staging / entry["files"]["image"])
        gx0, gy0 = tile_x0 * 256 + entry["col"], tile_y0 * 256 + entry["row"]
        rows, cols = np.mgrid[0:128, 0:128]
        gx, gy = gx0 + cols, gy0 + rows
        expected = np.stack([gx % 251, gy % 241, (gx + gy) % 239], axis=-1).astype(np.uint8)
        np.testing.assert_array_equal(image, expected)

        mask = _read(staging / entry["files"]["mask"])[:, :, 0]
        reference = rasterize(shapes, out_shape=(128, 128), transform=patch, fill=0, dtype="uint8")
        mismatched = int((mask != reference).sum())
        assert mismatched == 0, f"{entry['files']['mask']}: {mismatched} pixels differ"
        assert entry["summary"]["class_pixels"] == {
            str(v): int((mask == v).sum()) for v in np.unique(mask)
        }
        labelled += int((mask > 0).any())
    assert labelled > 3

    lists = {
        s: (staging / "splits" / f"{s}.txt").read_text().split() for s in ("train", "val", "test")
    }
    by_name = {manifest.patch_name(e): e for e in manifest.patches}
    assert all(lists.values())
    assert not set(lists["train"]) & set(lists["test"]) and not set(lists["val"]) & set(
        lists["test"]
    )
    for name in lists["train"] + lists["val"]:
        for held in lists["test"]:
            a, b = by_name[name], by_name[held]
            assert abs(a["row"] - b["row"]) >= 128 or abs(a["col"] - b["col"]) >= 128


def _read(path: Path) -> npt.NDArray[np.uint8]:
    if path.suffix == ".tif":
        import rasterio

        with rasterio.open(path) as src:
            return np.moveaxis(src.read(), 0, -1)
    array = np.asarray(Image.open(path))
    return array[:, :, None] if array.ndim == 2 else array
