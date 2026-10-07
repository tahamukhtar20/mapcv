"""GeoTIFF / COG imagery end to end: config, source, planning, manifest, CLI.

Rasters are written with rasterio (GDAL) and every generated patch is compared with
what rasterio reads from the same file at the patch's position, and every mask with
``rasterio.features.rasterize`` of the labels reprojected by rasterio, on the patch's
own transform. Alignment errors would otherwise be silent.
"""

from __future__ import annotations

import json
import math
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import numpy.typing as npt
import pytest
from PIL import Image
from pydantic import ValidationError
from typer.testing import CliRunner

pytest.importorskip("rasterio", reason="GeoTIFF tests write their rasters with rasterio")
import rasterio
import rasterio.features
import rasterio.transform
import rasterio.warp
from pyproj import Transformer
from rasterio.crs import CRS
from rasterio.enums import Resampling
from rasterio.transform import Affine
from rasterio.windows import Window

from mapcv.cli import app
from mapcv.config import GeoTiffImageryConfig, MapcvConfig, RegionConfig
from mapcv.imagery import (
    GeoTiffRasterSource,
    geotiff_fingerprint,
    open_raster_source,
)
from mapcv.manifest import Manifest, ManifestEntry, ManifestMismatchError
from mapcv.pipeline import run_generate
from mapcv.planning import plan

runner = CliRunner()


def flat(text: str) -> str:
    """Console output without the line wrapping and padding of narrow terminals."""
    return "".join(text.split())


CENTER_LON, CENTER_LAT = 3.0, 48.85
# (EPSG code, pixel size in CRS units). 1 m pixels in the projected CRSs, ~1 m in degrees.
CRS_CASES = [(32631, 1.0), (3857, 1.5), (4326, 1.2e-5)]


# ── Fixtures: rasters, regions and labels ────────────────────────────────────


@dataclass
class Raster:
    path: Path
    epsg: int
    width: int
    height: int
    transform: Affine
    data: npt.NDArray[Any]
    nodata: float | None

    @property
    def lonlat_bounds(self) -> tuple[float, float, float, float]:
        corners = [
            self.transform * (col, row) for col in (0, self.width) for row in (0, self.height)
        ]
        xs, ys = zip(*corners)
        bounds: tuple[float, float, float, float] = rasterio.warp.transform_bounds(
            CRS.from_epsg(self.epsg),
            CRS.from_epsg(4326),
            min(xs),
            min(ys),
            max(xs),
            max(ys),
            densify_pts=21,
        )
        return bounds

    def region(self, margin: float = 0.05) -> dict[str, float]:
        west, south, east, north = self.lonlat_bounds
        dx, dy = (east - west) * margin, (north - south) * margin
        return {
            "west": west + dx,
            "south": south + dy,
            "east": east - dx,
            "north": north - dy,
        }


def make_raster(
    directory: Path,
    *,
    epsg: int = 32631,
    pixel: float = 1.0,
    width: int = 640,
    height: int = 512,
    count: int = 3,
    dtype: str = "uint8",
    nodata: float | None = None,
    block: tuple[int, int, int, int] | None = None,
    overviews: tuple[int, ...] = (),
    name: str = "scene.tif",
    seed: int = 7,
    tag_nodata: bool = True,
    rotation: float = 0.0,
) -> Raster:
    """A tiled, deflate-compressed GeoTIFF centred on (3E, 48.85N).

    ``block = (row0, row1, col0, col1)`` is filled with the NoData value (or NaN for
    float rasters without one) in every band.
    """
    to_crs = Transformer.from_crs("EPSG:4326", f"EPSG:{epsg}", always_xy=True)
    center_x, center_y = to_crs.transform(CENTER_LON, CENTER_LAT)
    transform = Affine(
        pixel, 0.0, center_x - width * pixel / 2, 0.0, -pixel, center_y + height * pixel / 2
    )
    if rotation:
        transform = transform * Affine.rotation(rotation, (width / 2, height / 2))
    rng = np.random.default_rng(seed)
    if np.dtype(dtype).kind == "f":
        data = rng.uniform(1.0, 1000.0, size=(count, height, width)).astype(dtype)
    else:
        high = min(np.iinfo(dtype).max, 250)
        data = rng.integers(1, high, size=(count, height, width)).astype(dtype)
    if block is not None:
        r0, r1, c0, c1 = block
        if nodata is not None:
            data[:, r0:r1, c0:c1] = nodata
        else:
            data[:, r0:r1, c0:c1] = np.nan
    path = directory / name
    profile: dict[str, Any] = {
        "driver": "GTiff",
        "height": height,
        "width": width,
        "count": count,
        "dtype": dtype,
        "crs": CRS.from_epsg(epsg),
        "transform": transform,
        "tiled": True,
        "blockxsize": 128,
        "blockysize": 128,
        "compress": "deflate",
    }
    if nodata is not None and tag_nodata:
        profile["nodata"] = nodata
    with rasterio.open(path, "w", **profile) as dst:
        dst.write(data)
        if overviews:
            dst.build_overviews(list(overviews), Resampling.nearest)
    return Raster(path, epsg, width, height, transform, data, nodata)


def write_labels(directory: Path, region: dict[str, float]) -> Path:
    """Two irregular polygons (classes a and b) inside ``region``, in lon/lat."""
    west, south = region["west"], region["south"]
    dx, dy = region["east"] - west, region["north"] - south

    def at(fx: float, fy: float) -> list[float]:
        return [west + fx * dx, south + fy * dy]

    shapes = {
        "a": [at(0.15, 0.20), at(0.50, 0.17), at(0.46, 0.55), at(0.22, 0.60), at(0.15, 0.20)],
        "b": [at(0.55, 0.45), at(0.90, 0.40), at(0.78, 0.88), at(0.55, 0.45)],
    }
    features = [
        {
            "type": "Feature",
            "properties": {"kind": kind},
            "geometry": {"type": "Polygon", "coordinates": [ring]},
        }
        for kind, ring in shapes.items()
    ]
    path = directory / "labels.geojson"
    path.write_text(json.dumps({"type": "FeatureCollection", "features": features}))
    return path


def config_for(
    tmp_path: Path,
    imagery: dict[str, Any],
    region: dict[str, float],
    *,
    labels: Path | None = None,
    image_format: str = "png",
    staging: str = "dataset",
    **sampler: Any,
) -> MapcvConfig:
    data: dict[str, Any] = {
        "region": region,
        "imagery": {"type": "geotiff", **imagery},
        "sampler": {
            "patch_size": 64,
            "mode": "grid",
            "edge_strategy": "drop",
            **sampler,
        },
        "writer": {"staging_dir": str(tmp_path / staging), "image_format": image_format},
    }
    if labels is not None:
        data["labels"] = {
            "path": str(labels),
            "label_field": "kind",
            "classes": {"a": 1, "b": 2},
        }
    return MapcvConfig.model_validate(data)


# ── Independent expectations (rasterio) ──────────────────────────────────────


def _read_image(staging: Path, entry: ManifestEntry, npy: bool) -> npt.NDArray[Any]:
    """The written patch as ``(bands, h, w)``."""
    path = staging / entry["files"]["image"]
    if npy:
        loaded: npt.NDArray[Any] = np.load(path)
        return loaded
    return np.moveaxis(np.asarray(Image.open(path)), -1, 0)


def _file_window(src: Any, manifest: Manifest, entry: ManifestEntry, size: int) -> Window:
    """Where the patch sits in the file, derived from its own georeferencing."""
    patch = Affine(*manifest.patch_transform(entry))
    col, row = ~src.transform * (patch.c, patch.f)
    assert col == pytest.approx(round(col), abs=1e-6)
    assert row == pytest.approx(round(row), abs=1e-6)
    assert (patch.a, patch.b, patch.d, patch.e) == pytest.approx(
        (src.transform.a, src.transform.b, src.transform.d, src.transform.e), rel=1e-9
    )
    return Window(round(col), round(row), size, size)


def _expected_patch(
    src: Any, window: Window, nodata: float | None, indexes: list[int] | None = None
) -> tuple[npt.NDArray[Any], npt.NDArray[np.bool_]]:
    """rasterio's pixels (zeros outside the file) and which of them have imagery."""
    rows, cols = int(window.height), int(window.width)
    whole = (
        window.row_off >= 0
        and window.col_off >= 0
        and window.row_off + rows <= src.height
        and window.col_off + cols <= src.width
    )
    # Boundless reads go through a VRT, which GDAL does not keep exact for rotated grids.
    data = src.read(indexes=indexes, window=window, boundless=not whole, fill_value=0)
    inside = np.zeros((rows, cols), dtype=bool)
    r0, c0 = int(window.row_off), int(window.col_off)
    ra, rb = max(r0, 0), min(r0 + rows, src.height)
    ca, cb = max(c0, 0), min(c0 + cols, src.width)
    if ra < rb and ca < cb:
        inside[ra - r0 : rb - r0, ca - c0 : cb - c0] = True
    empty = np.all(~np.isfinite(data), axis=0) if data.dtype.kind == "f" else np.zeros_like(inside)
    if nodata is not None:
        empty |= np.all(data == np.asarray(nodata, dtype=data.dtype), axis=0)
    return data, inside & ~empty


def _label_shapes(labels: Path, epsg: int) -> list[tuple[Any, int]]:
    ids = {"a": 1, "b": 2}
    shapes = []
    for feature in json.loads(labels.read_text())["features"]:
        geometry = rasterio.warp.transform_geom(
            "EPSG:4326", f"EPSG:{epsg}", feature["geometry"], precision=-1
        )
        shapes.append((geometry, ids[feature["properties"]["kind"]]))
    return shapes


def assert_dataset_matches_rasterio(
    config: MapcvConfig,
    raster: Raster,
    *,
    labels: Path | None = None,
    indexes: list[int] | None = None,
    overview_level: int | None = None,
    expect_patches: int = 4,
) -> Manifest:
    """Generate the dataset and check every image and mask against rasterio."""
    result = run_generate(config)
    manifest = result.manifest
    assert len(manifest.patches) >= expect_patches
    staging = config.writer.staging_dir
    npy = config.writer.image_format == "npy"
    size = config.sampler.patch_size
    ignore = 255
    class_pixels = 0
    kwargs = {} if overview_level is None else {"overview_level": overview_level}
    shapes = _label_shapes(labels, raster.epsg) if labels is not None else []
    with rasterio.open(raster.path, **kwargs) as src:
        for entry in manifest.patches:
            window = _file_window(src, manifest, entry, size)
            data, valid = _expected_patch(src, window, raster.nodata, indexes)
            image = _read_image(staging, entry, npy)
            np.testing.assert_array_equal(
                image, data, err_msg=f"image at {entry['row']},{entry['col']}"
            )
            if npy:
                # The PNG/JPG writer counts black pixels instead; NPY records the valid mask's.
                assert entry["summary"]["empty_ratio"] == pytest.approx(1 - valid.mean())
            if labels is None:
                continue
            patch_transform = Affine(*manifest.patch_transform(entry))
            expected = rasterio.features.rasterize(
                shapes, out_shape=(size, size), transform=patch_transform, fill=0, dtype="uint8"
            )
            expected[~valid] = ignore
            mask = np.asarray(Image.open(staging / entry["files"]["mask"]))
            np.testing.assert_array_equal(
                mask, expected, err_msg=f"mask at {entry['row']},{entry['col']}"
            )
            class_pixels += int(np.count_nonzero((expected > 0) & (expected != ignore)))
    if labels is not None:
        assert class_pixels > 0, "the labels never reached a patch: the test would prove nothing"
    return manifest


# ── Pixel-exact datasets in several CRSs ─────────────────────────────────────


@pytest.mark.parametrize(("epsg", "pixel"), CRS_CASES)
def test_png_dataset_matches_rasterio_in_each_crs(tmp_path: Path, epsg: int, pixel: float) -> None:
    raster = make_raster(tmp_path, epsg=epsg, pixel=pixel)
    region = raster.region()
    labels = write_labels(tmp_path, region)
    config = config_for(tmp_path, {"path": str(raster.path)}, region, labels=labels)
    manifest = assert_dataset_matches_rasterio(config, raster, labels=labels, expect_patches=20)
    source = manifest.source
    assert source.source_type == "geotiff"
    assert source.product_id == "scene.tif"
    assert source.crs == f"EPSG:{epsg}"
    assert source.bands == ["b1", "b2", "b3"]
    assert source.dtype == "uint8"
    assert source.patch_shape == [64, 64, 3]


def test_rotated_raster_matches_rasterio(tmp_path: Path) -> None:
    # A rotated grid has no axis-aligned window: the region maps through the inverse
    # transform, and patches and masks carry the rotation.
    raster = make_raster(tmp_path, epsg=32631, width=700, height=600, rotation=12.0)
    assert raster.transform.b != 0 and raster.transform.d != 0
    region = raster.region(margin=0.25)
    labels = write_labels(tmp_path, region)
    config = config_for(tmp_path, {"path": str(raster.path)}, region, labels=labels)
    manifest = assert_dataset_matches_rasterio(config, raster, labels=labels, expect_patches=6)
    assert manifest.source.transform is not None
    assert manifest.source.transform[1] != 0


def test_multiband_uint16_npy_matches_rasterio(tmp_path: Path) -> None:
    raster = make_raster(tmp_path, epsg=32631, count=4, dtype="uint16")
    region = raster.region()
    labels = write_labels(tmp_path, region)
    config = config_for(
        tmp_path, {"path": str(raster.path)}, region, labels=labels, image_format="npy"
    )
    manifest = assert_dataset_matches_rasterio(config, raster, labels=labels, expect_patches=20)
    assert manifest.source.dtype == "uint16"
    assert manifest.source.bands == ["b1", "b2", "b3", "b4"]
    assert manifest.source.patch_shape == [4, 64, 64]


def test_band_selection_keeps_the_configured_order(tmp_path: Path) -> None:
    raster = make_raster(tmp_path, epsg=3857, pixel=1.5, count=4, dtype="uint8")
    region = raster.region()
    config = config_for(tmp_path, {"path": str(raster.path), "bands": [3, 1, 2]}, region)
    manifest = assert_dataset_matches_rasterio(config, raster, indexes=[3, 1, 2])
    assert manifest.source.bands == ["b3", "b1", "b2"]
    npy = config_for(
        tmp_path,
        {"path": str(raster.path), "bands": [4, 2]},
        region,
        image_format="npy",
        staging="npy",
    )
    manifest = assert_dataset_matches_rasterio(npy, raster, indexes=[4, 2])
    assert manifest.source.bands == ["b4", "b2"]
    assert manifest.source.patch_shape == [2, 64, 64]


def test_single_band_uint8_png_repeats_the_band_as_gray(tmp_path: Path) -> None:
    raster = make_raster(tmp_path, count=1)
    region = raster.region()
    config = config_for(tmp_path, {"path": str(raster.path)}, region)
    result = run_generate(config)
    staging = config.writer.staging_dir
    with rasterio.open(raster.path) as src:
        for entry in result.manifest.patches[:5]:
            window = _file_window(src, result.manifest, entry, 64)
            expected = src.read(1, window=window)
            image = np.asarray(Image.open(staging / entry["files"]["image"]))
            for channel in range(3):
                np.testing.assert_array_equal(image[..., channel], expected)
    assert result.manifest.source.patch_shape == [64, 64, 3]


# ── NoData, edges and overviews ──────────────────────────────────────────────


def test_nodata_and_the_raster_edge_are_ignored_in_masks(tmp_path: Path) -> None:
    # The block sits under the polygons so its patches have labels over NoData pixels.
    raster = make_raster(tmp_path, nodata=0, block=(120, 300, 150, 380))
    region = raster.region()
    labels = write_labels(tmp_path, region)
    config = config_for(tmp_path, {"path": str(raster.path)}, region, labels=labels)
    manifest = assert_dataset_matches_rasterio(config, raster, labels=labels, expect_patches=20)
    staging = config.writer.staging_dir
    ignored = sum(
        int(np.count_nonzero(np.asarray(Image.open(staging / e["files"]["mask"])) == 255))
        for e in manifest.patches
    )
    assert ignored > 0
    assert any(0 < e["summary"]["empty_ratio"] < 1 for e in manifest.patches)


def test_region_beyond_the_file_is_padded_with_ignore_index(tmp_path: Path) -> None:
    raster = make_raster(tmp_path, epsg=32631, width=650, height=530)
    west, south, east, north = raster.lonlat_bounds
    # Extends past the file on every side, so the window is the whole file and patches at
    # its edges are padded (padding inside the file would look different).
    dx, dy = (east - west) * 0.1, (north - south) * 0.1
    region = {"west": west - dx, "south": south - dy, "east": east + dx, "north": north + dy}
    labels = write_labels(tmp_path, raster.region())
    config = config_for(
        tmp_path, {"path": str(raster.path)}, region, labels=labels, edge_strategy="pad"
    )
    with pytest.warns(UserWarning, match="extends beyond the GeoTIFF"):
        manifest = assert_dataset_matches_rasterio(config, raster, labels=labels, expect_patches=20)
    assert any(entry["padded"] for entry in manifest.patches)


def test_float_nan_pixels_are_invalid_without_a_nodata_tag(tmp_path: Path) -> None:
    raster = make_raster(tmp_path, count=2, dtype="float32", block=(100, 260, 120, 300))
    region = raster.region()
    labels = write_labels(tmp_path, region)
    config = config_for(
        tmp_path, {"path": str(raster.path)}, region, labels=labels, image_format="npy"
    )
    manifest = assert_dataset_matches_rasterio(config, raster, labels=labels, expect_patches=20)
    assert manifest.source.dtype == "float32"
    assert any(e["summary"]["empty_ratio"] > 0 for e in manifest.patches)


def test_nodata_can_be_overridden_in_the_config(tmp_path: Path) -> None:
    # The file does not declare NoData; the config says 0 is NoData.
    raster = make_raster(tmp_path, nodata=0, block=(120, 300, 150, 380), tag_nodata=False)
    region = raster.region()
    labels = write_labels(tmp_path, region)
    config = config_for(
        tmp_path,
        {"path": str(raster.path), "nodata": 0},
        region,
        labels=labels,
        image_format="npy",
    )
    assert_dataset_matches_rasterio(config, raster, labels=labels, expect_patches=20)
    # Without the override the same pixels are plain imagery.
    plain = config_for(
        tmp_path, {"path": str(raster.path)}, region, image_format="npy", staging="plain"
    )
    result = run_generate(plain)
    assert all(entry["summary"]["empty_ratio"] == 0 for entry in result.manifest.patches)


def test_overview_levels_match_rasterio_overviews(tmp_path: Path) -> None:
    raster = make_raster(tmp_path, width=1024, height=768, overviews=(2, 4))
    region = raster.region()
    labels = write_labels(tmp_path, region)
    for level, expected_patches in ((1, 12), (2, 4)):
        config = config_for(
            tmp_path,
            {"path": str(raster.path), "overview": level},
            region,
            labels=labels,
            staging=f"ov{level}",
        )
        manifest = assert_dataset_matches_rasterio(
            config, raster, labels=labels, overview_level=level - 1, expect_patches=expected_patches
        )
        with rasterio.open(raster.path, overview_level=level - 1) as ov:
            transform = manifest.source.transform
            assert transform is not None
            # Same pixel size as rasterio's overview, and the window starts on a whole pixel.
            assert (transform[0], transform[4]) == pytest.approx((ov.transform.a, ov.transform.e))
            col, row = ~ov.transform * (transform[2], transform[5])
            assert (col, row) == pytest.approx((round(col), round(row)), abs=1e-6)


def test_window_covers_the_region_on_the_pixel_grid(tmp_path: Path) -> None:
    for epsg, pixel in CRS_CASES:
        raster = make_raster(tmp_path, epsg=epsg, pixel=pixel, name=f"w{epsg}.tif")
        region_dict = raster.region(margin=0.2)
        region = RegionConfig.model_validate(region_dict)
        source = GeoTiffRasterSource(region, GeoTiffImageryConfig(path=str(raster.path)))
        meta = source.metadata
        left, bottom, right, top = rasterio.warp.transform_bounds(
            CRS.from_epsg(4326),
            CRS.from_epsg(epsg),
            region.west,
            region.south,
            region.east,
            region.north,
            densify_pts=21,
        )
        a, _, c, _, e, f = meta.transform
        w_left, w_right = c, c + a * meta.width
        w_top, w_bottom = f, f + e * meta.height
        # Covers the region, with less than one pixel of excess per side (snapped outward).
        tolerance = abs(a) * 1e-6
        assert w_left <= left + tolerance and w_right >= right - tolerance
        assert w_bottom <= bottom + tolerance and w_top >= top - tolerance
        assert left - w_left < abs(a) + tolerance and w_right - right < abs(a) + tolerance
        assert bottom - w_bottom < abs(e) + tolerance and w_top - top < abs(e) + tolerance
        # On the file's grid: whole pixels from the file origin.
        file_t = raster.transform
        assert (c - file_t.c) / file_t.a == pytest.approx(
            round((c - file_t.c) / file_t.a), abs=1e-6
        )
        assert (f - file_t.f) / file_t.e == pytest.approx(
            round((f - file_t.f) / file_t.e), abs=1e-6
        )
        source.close()


def test_region_outside_the_file_is_an_error(tmp_path: Path) -> None:
    raster = make_raster(tmp_path)
    region = RegionConfig(west=10.0, south=40.0, east=10.01, north=40.01)
    with pytest.raises(ValueError, match="does not intersect the GeoTIFF"):
        open_raster_source(region, GeoTiffImageryConfig(path=str(raster.path)))


def test_unusable_files_and_settings_fail_clearly(tmp_path: Path) -> None:
    raster = make_raster(tmp_path, count=2, overviews=(2,))
    region = RegionConfig.model_validate(raster.region())
    with pytest.raises(ValueError, match="has 2 band"):
        GeoTiffRasterSource(region, GeoTiffImageryConfig(path=str(raster.path), bands=[3]))
    with pytest.raises(ValueError, match="1 overview level"):
        GeoTiffRasterSource(region, GeoTiffImageryConfig(path=str(raster.path), overview=2))
    with pytest.raises(FileNotFoundError):
        GeoTiffRasterSource(region, GeoTiffImageryConfig(path=str(tmp_path / "missing.tif")))
    # 2 bands cannot be a PNG, a uint16 file cannot be a PNG.
    with pytest.raises(ValueError, match="writer.image_format: npy"):
        GeoTiffRasterSource(region, GeoTiffImageryConfig(path=str(raster.path)), image_format="png")
    wide = make_raster(tmp_path, dtype="uint16", name="u16.tif")
    with pytest.raises(ValueError, match="uint16"):
        GeoTiffRasterSource(
            RegionConfig.model_validate(wide.region()),
            GeoTiffImageryConfig(path=str(wide.path)),
            image_format="jpg",
        )


# ── Remote COG ───────────────────────────────────────────────────────────────


def _serve(httpserver: Any, payload: bytes, etag: str = '"v1"') -> str:
    from werkzeug import Request, Response

    def handler(request: Request) -> Response:
        header = request.headers.get("Range")
        if header is None:
            return Response(payload, status=200, content_type="image/tiff")
        start_text, end_text = header.removeprefix("bytes=").split("-")
        start, end = int(start_text), min(int(end_text), len(payload) - 1)
        return Response(
            payload[start : end + 1],
            status=206,
            content_type="image/tiff",
            headers={"Content-Range": f"bytes {start}-{end}/{len(payload)}", "ETag": etag},
        )

    httpserver.expect_request("/scene.tif").respond_with_handler(handler)
    return str(httpserver.url_for("/scene.tif"))


def test_remote_cog_produces_the_same_dataset_as_the_local_file(
    tmp_path: Path, httpserver: Any
) -> None:
    raster = make_raster(tmp_path, epsg=32631, count=4, dtype="uint16")
    region = raster.region()
    labels = write_labels(tmp_path, region)
    url = _serve(httpserver, raster.path.read_bytes())

    local = config_for(
        tmp_path,
        {"path": str(raster.path)},
        region,
        labels=labels,
        image_format="npy",
        staging="local",
    )
    remote = config_for(
        tmp_path, {"path": url}, region, labels=labels, image_format="npy", staging="remote"
    )
    local_manifest = assert_dataset_matches_rasterio(
        local, raster, labels=labels, expect_patches=20
    )
    remote_manifest = run_generate(remote).manifest

    assert len(remote_manifest.patches) == len(local_manifest.patches)
    for a, b in zip(local_manifest.patches, remote_manifest.patches):
        assert (a["row"], a["col"]) == (b["row"], b["col"])
        for role in ("image", "mask"):
            assert (tmp_path / "local" / a["files"][role]).read_bytes() == (
                tmp_path / "remote" / b["files"][role]
            ).read_bytes()
    assert remote_manifest.source.transform == local_manifest.source.transform
    fingerprint = remote_manifest.source.fingerprint
    assert fingerprint is not None and fingerprint["kind"] == "url"
    assert fingerprint["url"] == url
    assert fingerprint["etag"] == '"v1"'
    assert fingerprint["size"] == raster.path.stat().st_size


# ── Resuming and fingerprints ────────────────────────────────────────────────


def test_resume_accepts_the_same_file_and_refuses_another(tmp_path: Path) -> None:
    raster = make_raster(tmp_path)
    region = raster.region()
    config = config_for(tmp_path, {"path": str(raster.path)}, region)
    first = run_generate(config)
    again = run_generate(config)
    assert again.new_patches == 0 and len(again.manifest.patches) == len(first.manifest.patches)
    fingerprint = first.manifest.source.fingerprint
    assert fingerprint is not None and fingerprint["kind"] == "file"
    assert fingerprint["size"] == raster.path.stat().st_size
    assert len(fingerprint["sha256_head_tail"]) == 64
    assert "mtime" not in fingerprint

    # A copy or touch (new path, new modification time) is still the same file.
    copied = tmp_path / "elsewhere" / "scene.tif"
    copied.parent.mkdir()
    copied.write_bytes(raster.path.read_bytes())
    os.utime(copied, (1_000_000_000, 1_000_000_000))
    os.utime(raster.path, (1_100_000_000, 1_100_000_000))
    moved = config_for(tmp_path, {"path": str(copied)}, region)
    assert run_generate(moved).new_patches == 0

    # Same name and size, different pixels.
    other = make_raster(tmp_path, seed=99)
    assert other.path == raster.path
    with pytest.raises(ManifestMismatchError, match="imagery file"):
        run_generate(config)


def test_resume_refuses_changed_read_settings(tmp_path: Path) -> None:
    raster = make_raster(tmp_path, nodata=0, block=(0, 50, 0, 50))
    region = raster.region()
    run_generate(config_for(tmp_path, {"path": str(raster.path)}, region))
    changed = config_for(tmp_path, {"path": str(raster.path), "nodata": 7}, region)
    with pytest.raises(ManifestMismatchError, match="imagery file or read settings"):
        run_generate(changed)


def test_file_fingerprint_notices_header_size_and_tail_changes(tmp_path: Path) -> None:
    path = tmp_path / "blob.bin"
    path.write_bytes(bytes(range(256)) * 1024)  # 256 KiB
    base = geotiff_fingerprint(str(path))
    assert geotiff_fingerprint(str(path)) == base

    def changed(offset: int, size: int | None = None) -> dict[str, Any]:
        data = bytearray(path.read_bytes())
        data[offset] ^= 0xFF
        if size is not None:
            data = data[:size]
        path.write_bytes(bytes(data))
        result = geotiff_fingerprint(str(path))
        path.write_bytes(bytes(range(256)) * 1024)
        return result

    assert changed(10)["sha256_head_tail"] != base["sha256_head_tail"]
    assert (
        changed(len(bytes(range(256)) * 1024) - 5)["sha256_head_tail"] != base["sha256_head_tail"]
    )
    assert changed(0, size=1000)["size"] != base["size"]


def test_other_sources_do_not_record_a_fingerprint() -> None:
    from mapcv.manifest import SourceRecord

    assert "fingerprint" not in SourceRecord(source_type="xyz").model_dump(mode="json")
    record = SourceRecord(source_type="geotiff", fingerprint={"kind": "file"})
    assert SourceRecord.model_validate(record.model_dump(mode="json")).fingerprint == {
        "kind": "file"
    }


# ── Config ───────────────────────────────────────────────────────────────────

_BASE: dict[str, Any] = {
    "region": {"west": 3.0, "south": 48.8, "east": 3.01, "north": 48.81},
    "sampler": {"patch_size": 64},
    "writer": {"staging_dir": "out", "image_format": "png"},
}


def _validate(imagery: dict[str, Any], **overrides: Any) -> MapcvConfig:
    return MapcvConfig.model_validate({**_BASE, "imagery": imagery, **overrides})


@pytest.mark.parametrize(
    "path",
    [
        "scene.tif",
        "/data/scene.tif",
        "file:///data/scene.tif",
        "https://example.com/cog/scene.tif",
        "s3://sentinel-cogs/path/scene.tif",
        "http://127.0.0.1:8000/scene.tif",
        "http://localhost/scene.tif",
    ],
)
def test_config_accepts_local_paths_and_safe_urls(path: str) -> None:
    config = _validate({"type": "geotiff", "path": path})
    assert isinstance(config.imagery, GeoTiffImageryConfig)
    assert config.imagery.bands is None and config.imagery.overview == 0


@pytest.mark.parametrize(
    ("path", "message"),
    [
        ("https://user:pw@example.com/a.tif", "credentials"),
        ("https://example.com/a.tif?token=secret", "query strings"),
        ("https://example.com/a.tif#frag", "fragments"),
        ("http://example.com/a.tif", "https://"),
        ("ftp://example.com/a.tif", "https://"),
        ("", "must not be empty"),
    ],
)
def test_config_rejects_unsafe_paths(path: str, message: str) -> None:
    with pytest.raises(ValidationError, match=message):
        _validate({"type": "geotiff", "path": path})


@pytest.mark.parametrize(
    ("imagery", "message"),
    [
        ({"bands": []}, "at least one band"),
        ({"bands": [0, 1]}, "1-based"),
        ({"bands": [1, 1]}, "duplicates"),
        ({"overview": -1}, "greater than or equal to 0"),
        ({"nodata": float("inf")}, "number or .nan"),
        ({"zoom": 12}, "Extra inputs"),
    ],
)
def test_config_rejects_bad_settings(imagery: dict[str, Any], message: str) -> None:
    with pytest.raises(ValidationError, match=message):
        _validate({"type": "geotiff", "path": "a.tif", **imagery})


def test_config_checks_the_writer_format_against_selected_bands() -> None:
    with pytest.raises(ValidationError, match="selects 2 bands"):
        _validate({"type": "geotiff", "path": "a.tif", "bands": [1, 2]})
    # 1 or 3 bands fit PNG/JPG (the dtype is checked when the file is opened); NPY takes any.
    _validate({"type": "geotiff", "path": "a.tif", "bands": [1]})
    _validate({"type": "geotiff", "path": "a.tif", "bands": [3, 2, 1]}, writer={"staging_dir": "o"})
    _validate(
        {"type": "geotiff", "path": "a.tif", "bands": [1, 2]},
        writer={"staging_dir": "o", "image_format": "npy"},
    )
    _validate(
        {"type": "geotiff", "path": "a.tif"}, writer={"staging_dir": "o", "image_format": "npy"}
    )


def test_config_resolves_a_relative_path_against_the_config_file(tmp_path: Path) -> None:
    config_dir = tmp_path / "project"
    config_dir.mkdir()
    (config_dir / "mapcv.yaml").write_text(
        "region: {west: 3.0, south: 48.8, east: 3.01, north: 48.81}\n"
        "imagery: {type: geotiff, path: data/scene.tif}\n"
        "sampler: {patch_size: 64}\n"
        "writer: {staging_dir: out}\n"
    )
    config = MapcvConfig.from_yaml(config_dir / "mapcv.yaml")
    assert isinstance(config.imagery, GeoTiffImageryConfig)
    assert Path(config.imagery.path) == config_dir / "data" / "scene.tif"


# ── Planning ─────────────────────────────────────────────────────────────────


def test_plan_sizes_the_window_on_the_files_grid(tmp_path: Path) -> None:
    raster = make_raster(tmp_path, epsg=32631, count=4, dtype="uint16")
    region = raster.region()
    config = config_for(tmp_path, {"path": str(raster.path)}, region, image_format="npy")
    estimate = plan(config)
    source = GeoTiffRasterSource(config.region, config.imagery)  # type: ignore[arg-type]
    assert estimate.raster_px == (source.metadata.width, source.metadata.height)
    # 1 m UTM pixels are 1 m on the map; on the ground UTM's scale factor (0.9996 at the
    # central meridian) makes them a little larger.
    assert estimate.resolution_m == pytest.approx(1.0, rel=1e-3)
    assert estimate.patches == (source.metadata.width // 64) * (source.metadata.height // 64)
    assert estimate.output_bytes == estimate.patches * 4 * 64 * 64 * 2
    assert "scene.tif" in estimate.imagery and "EPSG:32631" in estimate.imagery
    assert estimate.tiles is None and estimate.download_bytes is None


def test_plan_converts_degrees_to_metres(tmp_path: Path) -> None:
    raster = make_raster(tmp_path, epsg=4326, pixel=1.2e-5)
    config = config_for(tmp_path, {"path": str(raster.path)}, raster.region())
    # 1.2e-5 deg: 1.34 m north-south, 0.88 m east-west at 48.85N; the geometric mean.
    assert plan(config).resolution_m == pytest.approx(
        1.2e-5 * 111_320 * math.cos(math.radians(CENTER_LAT)), rel=0.01
    )


def test_plan_reports_a_region_that_leaves_the_file(tmp_path: Path) -> None:
    raster = make_raster(tmp_path)
    west, south, east, north = raster.lonlat_bounds
    region = {"west": west, "south": south, "east": east + 0.01, "north": north}
    estimate = plan(config_for(tmp_path, {"path": str(raster.path)}, region))
    assert any("extends beyond the GeoTIFF" in message for message in estimate.warnings)


# ── CLI ──────────────────────────────────────────────────────────────────────


def test_cli_validate_plan_generate_info(tmp_path: Path) -> None:
    raster = make_raster(tmp_path, epsg=32631)
    region = raster.region()
    write_labels(tmp_path, region)
    config_path = tmp_path / "mapcv.yaml"
    config_path.write_text(
        "region:\n"
        + "".join(f"  {key}: {value}\n" for key, value in region.items())
        + "imagery:\n  type: geotiff\n  path: scene.tif\n"
        "labels:\n  path: labels.geojson\n  label_field: kind\n"
        "sampler:\n  patch_size: 64\n  edge_strategy: drop\n"
        "writer:\n  staging_dir: out\n  image_format: png\n"
        "split:\n  strategy: spatial\n"
    )

    result = runner.invoke(app, ["validate", str(config_path)])
    assert result.exit_code == 0, result.output
    assert "GeoTIFF" in result.output

    result = runner.invoke(app, ["plan", str(config_path)])
    assert result.exit_code == 0, result.output
    assert flat("GeoTIFF scene.tif") in flat(result.output) and flat("EPSG:32631") in flat(
        result.output
    )

    result = runner.invoke(app, ["generate", str(config_path), "--yes"])
    assert result.exit_code == 0, result.output
    assert flat("Dataset ready") in flat(result.output)
    assert flat("geotiff") in flat(result.output) and flat("scene.tif") in flat(result.output)
    # The guide's URL stays on one line at 80 columns, so it can be clicked and copied.
    train = next(line for line in result.output.splitlines() if "Train on it:" in line)
    assert train.rstrip().endswith("/guides/use-your-dataset/")

    result = runner.invoke(app, ["info", str(tmp_path / "out")])
    assert result.exit_code == 0, result.output
    assert flat("geotiff") in flat(result.output) and flat("scene.tif") in flat(result.output)
    assert flat("EPSG:32631") in flat(result.output) and flat("b1, b2, b3") in flat(result.output)
    manifest = Manifest.load(tmp_path / "out" / "manifest.json")
    assert manifest.source.fingerprint is not None
    assert manifest.source.fingerprint["kind"] == "file"

    # Re-running resumes; a different file is refused with a clear message.
    assert runner.invoke(app, ["generate", str(config_path), "--yes"]).exit_code == 0
    make_raster(tmp_path, seed=3)
    refused = runner.invoke(app, ["generate", str(config_path), "--yes"])
    assert refused.exit_code == 1
    assert flat("Cannot resume") in flat(refused.output) and flat("imagery file") in flat(
        refused.output
    )


def test_cli_validate_warns_about_a_missing_local_file(tmp_path: Path) -> None:
    config_path = tmp_path / "mapcv.yaml"
    config_path.write_text(
        "region: {west: 3.0, south: 48.8, east: 3.01, north: 48.81}\n"
        "imagery: {type: geotiff, path: nope.tif}\n"
        "sampler: {patch_size: 64}\n"
        "writer: {staging_dir: out}\n"
    )
    result = runner.invoke(app, ["validate", str(config_path)])
    assert result.exit_code == 0
    assert flat("imagery.path not found") in flat(result.output)


def test_cli_generate_explains_a_format_mismatch(tmp_path: Path) -> None:
    raster = make_raster(tmp_path, dtype="uint16")
    region = raster.region()
    config_path = tmp_path / "mapcv.yaml"
    config_path.write_text(
        "region:\n"
        + "".join(f"  {key}: {value}\n" for key, value in region.items())
        + "imagery: {type: geotiff, path: scene.tif}\n"
        "sampler: {patch_size: 64}\n"
        "writer: {staging_dir: out, image_format: png}\n"
    )
    result = runner.invoke(app, ["plan", str(config_path)])
    assert result.exit_code == 1
    assert flat("uint16") in flat(result.output) and flat("npy") in flat(result.output)


def test_init_geotiff_template_is_a_valid_config(tmp_path: Path) -> None:
    out = tmp_path / "geotiff.yaml"
    result = runner.invoke(app, ["init", str(out), "--template", "geotiff"])
    assert result.exit_code == 0, result.output
    config = MapcvConfig.from_yaml(out)
    assert isinstance(config.imagery, GeoTiffImageryConfig)


def test_init_wizard_geotiff_shows_the_file_and_writes_a_config(tmp_path: Path) -> None:
    raster = make_raster(tmp_path, epsg=32631, count=4, dtype="uint8")
    out = tmp_path / "mapcv.yaml"
    # imagery, path, bands (4 uint8 bands: pick RGB), area (Enter = whole file), labels (none),
    # patch size, output folder, split.
    answers = "\n".join(["geotiff", str(raster.path), "1,2,3", "", "", "128", "./ds", "n"]) + "\n"
    result = runner.invoke(app, ["init", str(out), "--interactive"], input=answers)
    assert result.exit_code == 0, result.output
    assert flat("EPSG:32631") in flat(result.output) and flat("640 × 512 px") in flat(result.output)
    assert flat("4 band(s) · uint8") in flat(result.output)
    config = MapcvConfig.from_yaml(out)
    assert isinstance(config.imagery, GeoTiffImageryConfig)
    assert config.imagery.bands == [1, 2, 3]
    assert config.writer.image_format == "png"
    west, south, east, north = raster.lonlat_bounds
    # The whole file, shrunk by a hair so it maps back inside the file.
    assert west <= config.region.west < west + 1e-4
    assert east - 1e-4 < config.region.east <= east
    assert south <= config.region.south < south + 1e-4
    assert north - 1e-4 < config.region.north <= north
    # The config the wizard wrote works as it is.
    estimate = plan(config)
    assert estimate.patches > 0 and not estimate.warnings


def test_init_wizard_asks_again_when_the_file_cannot_be_read(tmp_path: Path) -> None:
    raster = make_raster(tmp_path, count=2, dtype="uint16")
    out = tmp_path / "mapcv.yaml"
    answers = "\n".join(
        ["geotiff", str(tmp_path / "nope.tif"), str(raster.path), "", "", "128", "./ds", "n"]
    )
    result = runner.invoke(app, ["init", str(out), "--interactive"], input=answers + "\n")
    assert result.exit_code == 0, result.output
    assert flat("Cannot read that file") in flat(result.output)
    config = MapcvConfig.from_yaml(out)
    assert config.writer.image_format == "npy"


@pytest.mark.parametrize(
    ("task", "outcome"),
    [
        ("segmentation", "every mask will be background"),
        ("detection", "no patch will have objects"),
    ],
)
def test_generate_warns_about_an_empty_label_file(tmp_path: Path, task: str, outcome: str) -> None:
    raster = make_raster(tmp_path)
    labels = tmp_path / "empty.geojson"
    labels.write_text('{"type": "FeatureCollection", "features": []}')
    config = config_for(tmp_path, {"path": str(raster.path)}, raster.region(), labels=labels)
    config = config.model_copy(update={"task": task})
    what = "polygon" if task == "segmentation" else "feature"
    with pytest.warns(UserWarning, match=f"no usable label {what} in the label file, so {outcome}"):
        run_generate(config)
