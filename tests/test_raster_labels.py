"""Raster labels (``labels.type: raster``): alignment, settings, planning, CLI, resume.

Label rasters are written with rasterio (GDAL) on grids that differ from the imagery's
in CRS, resolution, origin and rotation. Every generated mask is compared with an
independent reference: ``rasterio.warp.reproject`` (nearest neighbour, exact
transformation: ``tolerance=0``) of the label raster onto the patch's own transform
(``Manifest.patch_transform``), mapped through the class table here. Alignment errors
(a half-pixel shift, a swapped axis) would otherwise be silent.

Masks must equal the reference except where a pixel centre lies within 1e-9 label
pixels of a label-pixel boundary: there the side a centre falls on depends on float
rounding. mapcv uses GDAL's tie rule (``floor(x + 1e-10)``), so in practice these
match too (``test_ties_take_the_right_and_lower_label_pixel`` pins the rule).
"""

from __future__ import annotations

import io
import json
import threading
from dataclasses import dataclass
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any, Dict, Iterator, Optional, Sequence, Tuple

import numpy as np
import numpy.typing as npt
import pytest
from PIL import Image
from pydantic import ValidationError
from typer.testing import CliRunner

pytest.importorskip("rasterio", reason="raster-label tests write their rasters with rasterio")
import rasterio  # noqa: E402
import rasterio.warp  # noqa: E402
from pyproj import Transformer  # noqa: E402
from rasterio.crs import CRS  # noqa: E402
from rasterio.enums import Resampling  # noqa: E402
from rasterio.transform import Affine  # noqa: E402

from mapcv.cli import app  # noqa: E402
from mapcv.config import (  # noqa: E402
    LabelsConfig,
    MapcvConfig,
    RasterLabelsConfig,
    RegionConfig,
    XYZImageryConfig,
)
from mapcv.imagery import open_raster_source  # noqa: E402
from mapcv.manifest import Manifest, ManifestMismatchError  # noqa: E402
from mapcv.pipeline import run_generate  # noqa: E402
from mapcv.planning import plan  # noqa: E402
from mapcv.targets import RasterSegmentationTarget, create_target  # noqa: E402
from mapcv.targets.raster_labels import LabelRasterSampler  # noqa: E402

runner = CliRunner()

CENTER_LON, CENTER_LAT = 3.0, 48.85
IMAGERY_EPSG = 32631
IGNORE = 255
SENTINEL = -999_999  # reference value of pixels outside the label raster
TIE_TOLERANCE = 1e-9
# rasterio 1.5 honours ``reproject(..., tolerance=0)``; older versions always approximate
# the transformation between two CRSs (up to 0.125 pixel), which is no exact reference.
EXACT_WARP = tuple(int(part) for part in rasterio.__version__.split(".")[:2]) >= (1, 5)

# Raster value -> mask ID. 99 is unmapped, NODATA is the file's NoData.
CLASSES = {0: 0, 10: 1, 20: 2, 30: 3, 40: 3}
NODATA = 250
VALUES_U8 = (0, 10, 20, 30, 40, 99, NODATA)
CLASSES_U16 = {1000: 1, 2000: 2, 3000: 3}
VALUES_U16 = (1000, 2000, 3000, 4000, 65535)
NODATA_U16 = 65535


def flat(text: str) -> str:
    """Console output without the line wrapping and padding of narrow terminals."""
    return "".join(text.split())


def utm(lon: float, lat: float, epsg: int = IMAGERY_EPSG) -> Tuple[float, float]:
    x, y = Transformer.from_crs("EPSG:4326", f"EPSG:{epsg}", always_xy=True).transform(lon, lat)
    return float(x), float(y)


# ── Fixtures: imagery, label rasters, configs ────────────────────────────────


@dataclass
class Imagery:
    path: Path
    transform: Affine
    width: int
    height: int

    def region(self, margin: float = 0.04) -> Dict[str, float]:
        corners = [
            self.transform * (col, row) for col in (0, self.width) for row in (0, self.height)
        ]
        xs, ys = zip(*corners)
        west, south, east, north = rasterio.warp.transform_bounds(
            CRS.from_epsg(IMAGERY_EPSG), CRS.from_epsg(4326), min(xs), min(ys), max(xs), max(ys)
        )
        dx, dy = (east - west) * margin, (north - south) * margin
        return {"west": west + dx, "south": south + dy, "east": east - dx, "north": north - dy}


def make_imagery(
    directory: Path,
    *,
    width: int = 320,
    height: int = 288,
    pixel: float = 1.0,
    rotation: float = 0.0,
    nodata_block: Optional[Tuple[int, int, int, int]] = None,
) -> Imagery:
    """A 3-band uint8 GeoTIFF in UTM 31N (NoData 0) centred on (3E, 48.85N)."""
    cx, cy = utm(CENTER_LON, CENTER_LAT)
    transform = Affine(pixel, 0.0, cx - width * pixel / 2, 0.0, -pixel, cy + height * pixel / 2)
    if rotation:
        transform = transform * Affine.rotation(rotation, (width / 2, height / 2))
    data = np.random.default_rng(3).integers(1, 250, size=(3, height, width), dtype=np.uint8)
    if nodata_block is not None:
        r0, r1, c0, c1 = nodata_block
        data[:, r0:r1, c0:c1] = 0
    path = directory / "scene.tif"
    with rasterio.open(
        path,
        "w",
        driver="GTiff",
        height=height,
        width=width,
        count=3,
        dtype="uint8",
        crs=CRS.from_epsg(IMAGERY_EPSG),
        transform=transform,
        nodata=0,
        tiled=True,
        blockxsize=128,
        blockysize=128,
    ) as dst:
        dst.write(data)
    return Imagery(path, transform, width, height)


def make_labels(
    directory: Path,
    transform: Affine,
    width: int,
    height: int,
    *,
    epsg: int = IMAGERY_EPSG,
    dtype: str = "uint8",
    values: Sequence[int] = VALUES_U8,
    nodata: Optional[int] = NODATA,
    name: str = "labels.tif",
    point: bool = False,
    seed: int = 11,
) -> Path:
    """A label raster with an independent random value per pixel: any misalignment,
    even by one pixel, changes most of the mask."""
    rng = np.random.default_rng(seed)
    data = rng.choice(np.asarray(values, dtype=dtype), size=(height, width))
    path = directory / name
    profile: Dict[str, Any] = dict(
        driver="GTiff",
        height=height,
        width=width,
        count=1,
        dtype=dtype,
        crs=CRS.from_epsg(epsg),
        transform=transform,
        tiled=True,
        blockxsize=64,
        blockysize=64,
        compress="deflate",
    )
    if nodata is not None:
        profile["nodata"] = nodata
    with rasterio.open(path, "w", **profile) as dst:
        if point:
            dst.update_tags(AREA_OR_POINT="Point")
        dst.write(data, 1)
    return path


def labels_around(
    imagery: Imagery,
    epsg: int,
    pixel: float,
    *,
    offset: Tuple[float, float] = (0.0, 0.0),
    cover: float = 0.8,
) -> Tuple[Affine, int, int]:
    """A north-up label grid in ``epsg`` with ``pixel`` size covering the central
    ``cover`` of the imagery, its origin shifted by ``offset`` (in label pixels)."""
    corners = [
        imagery.transform * (col, row) for col in (0, imagery.width) for row in (0, imagery.height)
    ]
    to_label = Transformer.from_crs(f"EPSG:{IMAGERY_EPSG}", f"EPSG:{epsg}", always_xy=True)
    xs, ys = to_label.transform([x for x, _ in corners], [y for _, y in corners])
    west, east, south, north = min(xs), max(xs), min(ys), max(ys)
    span_x, span_y = (east - west) * cover, (north - south) * cover
    left = (west + east) / 2 - span_x / 2 + offset[0] * pixel
    top = (south + north) / 2 + span_y / 2 - offset[1] * pixel
    return (
        Affine(pixel, 0.0, left, 0.0, -pixel, top),
        int(round(span_x / pixel)),
        int(round(span_y / pixel)),
    )


def config_for(
    tmp_path: Path,
    imagery: Dict[str, Any],
    region: Dict[str, float],
    labels: Dict[str, Any],
    *,
    staging: str = "dataset",
    image_format: str = "png",
    split: Optional[Dict[str, Any]] = None,
    **sampler: Any,
) -> MapcvConfig:
    data: Dict[str, Any] = {
        "region": region,
        "imagery": imagery,
        "labels": {"type": "raster", "classes": CLASSES, **labels},
        "sampler": {"patch_size": 64, "mode": "grid", "edge_strategy": "drop", **sampler},
        "writer": {"staging_dir": str(tmp_path / staging), "image_format": image_format},
    }
    if split is not None:
        data["split"] = split
    return MapcvConfig.model_validate(data)


def sampler_for(config: MapcvConfig) -> LabelRasterSampler:
    assert isinstance(config.labels, RasterLabelsConfig)
    return LabelRasterSampler(config.labels, f"EPSG:{IMAGERY_EPSG}")


def six(transform: Affine) -> Tuple[float, float, float, float, float, float]:
    a, b, c, d, e, f = tuple(transform)[:6]
    return (a, b, c, d, e, f)


# ── Independent reference (rasterio) ─────────────────────────────────────────


@dataclass
class Comparison:
    patches: int = 0
    pixels: int = 0
    ties: int = 0
    tie_mismatches: int = 0
    classes: int = 0
    ignored: int = 0


def reference_values(
    label_path: Path, patch: Affine, size: int, dst_crs: str
) -> npt.NDArray[np.int64]:
    """Raw label values at the patch's pixel centres, by GDAL's nearest-neighbour warp
    (exact transformation); ``SENTINEL`` outside the label raster."""
    with rasterio.open(label_path) as src:
        exact: Dict[str, Any] = {"tolerance": 0} if EXACT_WARP else {}
        if not EXACT_WARP and src.crs != CRS.from_user_input(dst_crs):
            pytest.skip("rasterio < 1.5 cannot reproject without approximating (tolerance)")
        raw = src.read(1).astype(np.int32)
        destination = np.full((size, size), SENTINEL, dtype=np.int32)
        rasterio.warp.reproject(
            raw,
            destination,
            src_transform=src.transform,
            src_crs=src.crs,
            src_nodata=None,  # map the raw values here, NoData included
            dst_transform=patch,
            dst_crs=CRS.from_user_input(dst_crs),
            dst_nodata=SENTINEL,
            resampling=Resampling.nearest,
            **exact,
        )
    return destination.astype(np.int64)


def tie_mask(label_path: Path, patch: Affine, size: int, dst_crs: str) -> npt.NDArray[np.bool_]:
    """Pixels whose centre lies within ``TIE_TOLERANCE`` of a label-pixel boundary."""
    with rasterio.open(label_path) as src:
        label_transform, label_crs = src.transform, src.crs
    cols, rows = np.meshgrid(np.arange(size) + 0.5, np.arange(size) + 0.5)
    xs, ys = patch * (cols, rows)
    if CRS.from_user_input(dst_crs) != label_crs:
        to_label = Transformer.from_crs(dst_crs, label_crs.to_string(), always_xy=True)
        xs, ys = to_label.transform(xs, ys)
    fc, fr = ~label_transform * (xs, ys)
    near = (np.abs(fc - np.round(fc)) < TIE_TOLERANCE) | (np.abs(fr - np.round(fr)) < TIE_TOLERANCE)
    return np.asarray(near, dtype=np.bool_)


def expected_mask(
    raw: npt.NDArray[np.int64],
    classes: Dict[int, int],
    *,
    nodata: Optional[int],
    ignore_values: Sequence[int] = (),
    unmapped: str = "background",
    ignore: Optional[int] = IGNORE,
) -> npt.NDArray[np.uint8]:
    """The class table applied value by value, written independently of mapcv's."""
    ignore_value = 0 if ignore is None else ignore
    out = np.empty(raw.shape, dtype=np.uint8)
    for index, value in np.ndenumerate(raw):
        if value == SENTINEL or value == nodata or value in ignore_values:
            out[index] = ignore_value
        elif value in classes:
            out[index] = classes[int(value)]
        else:
            out[index] = ignore_value if unmapped == "ignore" else 0
    return out


def assert_masks_match_reference(
    config: MapcvConfig,
    label_path: Path,
    *,
    classes: Dict[int, int] = CLASSES,
    nodata: Optional[int] = NODATA,
    ignore_values: Sequence[int] = (),
    unmapped: str = "background",
    ignore: Optional[int] = IGNORE,
    min_patches: int = 4,
) -> Tuple[Manifest, Comparison]:
    """Generate the dataset and compare every mask with the rasterio reference."""
    manifest = run_generate(config).manifest
    assert len(manifest.patches) >= min_patches
    staging = config.writer.staging_dir
    size = config.sampler.patch_size
    crs = manifest.source.crs
    assert crs is not None
    stats = Comparison()
    for entry in manifest.patches:
        patch = Affine(*manifest.patch_transform(entry))
        raw = reference_values(label_path, patch, size, crs)
        expected = expected_mask(
            raw,
            classes,
            nodata=nodata,
            ignore_values=ignore_values,
            unmapped=unmapped,
            ignore=ignore,
        )
        image = np.asarray(Image.open(staging / entry["files"]["image"]))
        # Imagery NoData (0 in every band) stays ignored whatever the label says.
        if ignore is not None:
            expected[~image.any(axis=-1)] = ignore
        mask = np.asarray(Image.open(staging / entry["files"]["mask"]))
        ties = tie_mask(label_path, patch, size, crs)
        differ = mask != expected
        assert not (differ & ~ties).any(), (
            f"mask at {entry['row']},{entry['col']} differs from rasterio at "
            f"{int((differ & ~ties).sum())} pixel(s) away from label-pixel boundaries"
        )
        stats.patches += 1
        stats.pixels += mask.size
        stats.ties += int(ties.sum())
        stats.tie_mismatches += int((differ & ties).sum())
        stats.classes += int(np.count_nonzero((expected > 0) & (expected != IGNORE)))
        stats.ignored += int(np.count_nonzero(expected == IGNORE))
        # The summary counts what the mask holds.
        counts = {str(v): int(c) for v, c in zip(*np.unique(mask, return_counts=True))}
        assert entry["summary"]["class_pixels"] == counts
    assert stats.classes > 0, "no class reached a mask: the comparison would prove nothing"
    print(f"\n{label_path.name}: {stats}")
    return manifest, stats


# ── Alignment: every mask against rasterio ───────────────────────────────────


def _geotiff(imagery: Imagery, **extra: Any) -> Dict[str, Any]:
    return {"type": "geotiff", "path": str(imagery.path), "chunk_rows": 100, **extra}


def test_same_grid_is_copied_and_matches_rasterio(tmp_path: Path) -> None:
    imagery = make_imagery(tmp_path, nodata_block=(40, 90, 60, 150))
    # The imagery's own grid, shifted by whole pixels and smaller than the imagery.
    transform = imagery.transform * Affine.translation(17, 9)
    labels = make_labels(tmp_path, transform, 260, 240)
    config = config_for(tmp_path, _geotiff(imagery), imagery.region(), {"path": str(labels)})
    manifest, stats = assert_masks_match_reference(config, labels)
    assert stats.ignored > 0  # imagery NoData, label NoData and the area off the label raster
    # The direct copy and the general path give the same masks.
    sampler = sampler_for(config)
    for row, col, height, width in [(0, 0, 100, 320), (100, 5, 164, 300), (250, 200, 64, 64)]:
        window = six(imagery.transform * Affine.translation(col, row))
        assert sampler.grid_offset(window) == (row - 9, col - 17)
        fast = sampler.sample(window, height, width)
        general = sampler.sample(window, height, width, fast=False)
        np.testing.assert_array_equal(fast, general)


@pytest.mark.parametrize(
    ("pixel", "offset"),
    [(2.0, (0.37, 0.61)), (0.5, (0.37, 0.61)), (3.0, (-0.5, 0.25))],
    ids=["coarser-2x", "finer-0.5x", "coarser-3x-half-pixel"],
)
def test_other_resolution_and_origin_match_rasterio(
    tmp_path: Path, pixel: float, offset: Tuple[float, float]
) -> None:
    imagery = make_imagery(tmp_path)
    transform, width, height = labels_around(imagery, IMAGERY_EPSG, pixel, offset=offset)
    labels = make_labels(tmp_path, transform, width, height)
    config = config_for(tmp_path, _geotiff(imagery), imagery.region(), {"path": str(labels)})
    sampler = sampler_for(config)
    assert sampler.grid_offset(six(imagery.transform)) is None
    assert_masks_match_reference(config, labels)


@pytest.mark.parametrize(
    ("epsg", "pixel", "dtype"),
    [(4326, 1.3e-5, "uint8"), (4326, 0.9e-5, "uint16"), (3857, 1.7, "uint8"), (3857, 1.1, "int16")],
    ids=["wgs84-coarser", "wgs84-finer-uint16", "mercator-coarser", "mercator-finer-int16"],
)
def test_labels_in_another_crs_match_rasterio(
    tmp_path: Path, epsg: int, pixel: float, dtype: str
) -> None:
    imagery = make_imagery(tmp_path)
    transform, width, height = labels_around(imagery, epsg, pixel, offset=(0.29, 0.43))
    classes: Dict[int, int]
    values: Sequence[int]
    if dtype == "uint8":
        classes, values, nodata = CLASSES, VALUES_U8, NODATA
    elif dtype == "uint16":
        classes, values, nodata = CLASSES_U16, VALUES_U16, NODATA_U16
    else:
        classes, values, nodata = {-5: 1, 0: 0, 7: 2, 300: 3}, (-5, 0, 7, 300, 12, -1), -1
    labels = make_labels(
        tmp_path, transform, width, height, epsg=epsg, dtype=dtype, values=values, nodata=nodata
    )
    config = config_for(
        tmp_path, _geotiff(imagery), imagery.region(), {"path": str(labels), "classes": classes}
    )
    assert_masks_match_reference(config, labels, classes=classes, nodata=nodata)


def test_rotated_imagery_grid_matches_rasterio(tmp_path: Path) -> None:
    imagery = make_imagery(tmp_path, width=352, height=352, rotation=17.0)
    transform, width, height = labels_around(imagery, IMAGERY_EPSG, 0.8, offset=(0.21, 0.66))
    labels = make_labels(tmp_path, transform, width, height)
    config = config_for(tmp_path, _geotiff(imagery), imagery.region(0.2), {"path": str(labels)})
    assert_masks_match_reference(config, labels)


def test_pixel_is_point_label_raster_matches_rasterio(tmp_path: Path) -> None:
    imagery = make_imagery(tmp_path)
    transform, width, height = labels_around(imagery, IMAGERY_EPSG, 2.0, offset=(0.1, 0.2))
    labels = make_labels(tmp_path, transform, width, height, point=True)
    with rasterio.open(labels) as src:
        assert src.tags().get("AREA_OR_POINT") == "Point"
        gdal_transform = tuple(src.transform)[:6]
    config = config_for(tmp_path, _geotiff(imagery), imagery.region(), {"path": str(labels)})
    sampler = sampler_for(config)
    assert sampler.transform == pytest.approx(gdal_transform, abs=1e-9)
    assert_masks_match_reference(config, labels)


def test_ties_take_the_right_and_lower_label_pixel(tmp_path: Path) -> None:
    """A label grid twice as fine as the imagery, on the same origin: every imagery
    pixel centre falls on a label-pixel corner and takes the label pixel to its
    lower right, as GDAL's nearest-neighbour kernel does (``floor(x + 1e-10)``)."""
    imagery = make_imagery(tmp_path)
    transform = imagery.transform * Affine.scale(0.5)
    labels = make_labels(tmp_path, transform, imagery.width * 2, imagery.height * 2)
    config = config_for(tmp_path, _geotiff(imagery), imagery.region(), {"path": str(labels)})
    sampler = sampler_for(config)
    mask = sampler.sample(six(imagery.transform), 40, 50)
    with rasterio.open(labels) as src:
        raw = src.read(1)[1:80:2, 1:100:2].astype(np.int64)
    np.testing.assert_array_equal(mask, expected_mask(raw, CLASSES, nodata=NODATA))
    # GDAL agrees here as well.
    reference = reference_values(labels, imagery.transform, 40, f"EPSG:{IMAGERY_EPSG}")
    np.testing.assert_array_equal(reference[:40, :40], raw[:40, :40])


# ── XYZ imagery (Web Mercator) end to end, offline ───────────────────────────


class _Tiles(BaseHTTPRequestHandler):
    def log_message(self, *args: object) -> None:
        pass

    def do_GET(self) -> None:  # noqa: N802 - http.server API
        z, x, y = (int(part) for part in self.path.strip("/").split(".")[0].split("/"))
        buffer = io.BytesIO()
        Image.new("RGB", (256, 256), (x % 200 + 20, y % 200 + 20, z)).save(buffer, "PNG")
        body = buffer.getvalue()
        self.send_response(200)
        self.send_header("Content-Type", "image/png")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)


@pytest.fixture()
def tile_server() -> Iterator[str]:
    server = ThreadingHTTPServer(("127.0.0.1", 0), _Tiles)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    yield f"http://127.0.0.1:{server.server_address[1]}/{{z}}/{{x}}/{{y}}.png"
    server.shutdown()


XYZ_REGION = {"west": 2.995, "south": 48.847, "east": 3.004, "north": 48.853}


@pytest.mark.parametrize("case", ["utm", "same-grid"])
def test_xyz_imagery_with_raster_labels_matches_rasterio(
    tmp_path: Path, tile_server: str, case: str
) -> None:
    xyz = {"type": "xyz", "zoom": 16, "url_template": tile_server, "strip_rows": 1}
    source = open_raster_source(RegionConfig(**XYZ_REGION), XYZImageryConfig.model_validate(xyz))
    meta = source.metadata
    source.close()
    assert meta.crs == "EPSG:3857"
    if case == "same-grid":
        # Labels made on the tile grid (whole tiles off the imagery's first one).
        transform = Affine(*meta.transform) * Affine.translation(256, 0)
        labels = make_labels(tmp_path, transform, meta.width - 256, meta.height, epsg=3857)
    else:
        west, north = utm(XYZ_REGION["west"] + 0.001, XYZ_REGION["north"] - 0.0005)
        transform = Affine(2.5, 0.0, west + 0.4, 0.0, -2.5, north - 0.7)
        labels = make_labels(tmp_path, transform, 230, 230)
    config = config_for(tmp_path, xyz, XYZ_REGION, {"path": str(labels)}, edge_strategy="pad")
    target = create_target(config)
    assert isinstance(target, RasterSegmentationTarget)
    target.prepare(meta)
    expect_fast = case == "same-grid"
    assert (target.sampler.grid_offset(meta.transform) is not None) is expect_fast
    manifest, stats = assert_masks_match_reference(config, labels)
    assert manifest.source.crs == "EPSG:3857"


# ── Settings: NoData, ignore values, unmapped values, ignore_index ───────────


def _small_case(tmp_path: Path) -> Tuple[Imagery, Path]:
    imagery = make_imagery(tmp_path, nodata_block=(0, 64, 0, 64))
    transform, width, height = labels_around(imagery, IMAGERY_EPSG, 2.0, offset=(0.3, 0.3))
    return imagery, make_labels(tmp_path, transform, width, height)


@pytest.mark.parametrize(
    "settings",
    [
        {"unmapped": "ignore"},
        {"ignore_values": [99]},
        {"nodata": 99},
        {"ignore_index": None},
        {"ignore_index": 200, "unmapped": "ignore"},
    ],
    ids=["unmapped-ignore", "ignore-values", "nodata-override", "no-ignore-index", "ignore-200"],
)
def test_label_settings_match_the_reference(tmp_path: Path, settings: Dict[str, Any]) -> None:
    imagery, labels = _small_case(tmp_path)
    config = config_for(
        tmp_path, _geotiff(imagery), imagery.region(), {"path": str(labels), **settings}
    )
    ignore = settings.get("ignore_index", IGNORE)
    manifest = run_generate(config).manifest
    size = config.sampler.patch_size
    staging = config.writer.staging_dir
    for entry in manifest.patches:
        patch = Affine(*manifest.patch_transform(entry))
        raw = reference_values(labels, patch, size, f"EPSG:{IMAGERY_EPSG}")
        expected = expected_mask(
            raw,
            CLASSES,
            nodata=settings.get("nodata", NODATA),
            ignore_values=settings.get("ignore_values", ()),
            unmapped=settings.get("unmapped", "background"),
            ignore=ignore,
        )
        image = np.asarray(Image.open(staging / entry["files"]["image"]))
        if ignore is not None:
            expected[~image.any(axis=-1)] = ignore
        mask = np.asarray(Image.open(staging / entry["files"]["mask"]))
        ties = tie_mask(labels, patch, size, f"EPSG:{IMAGERY_EPSG}")
        assert not ((mask != expected) & ~ties).any()
    assert manifest.ignore_index == ignore
    if ignore is not None:
        assert any(str(ignore) in entry["summary"]["class_pixels"] for entry in manifest.patches)


def test_imagery_nodata_stays_ignored_under_a_class(tmp_path: Path) -> None:
    imagery = make_imagery(tmp_path, nodata_block=(64, 128, 64, 128))
    # Every label pixel is class 1: only imagery NoData and the label edge are ignored.
    labels = make_labels(
        tmp_path, imagery.transform, imagery.width, imagery.height, values=(10,), nodata=None
    )
    config = config_for(tmp_path, _geotiff(imagery), imagery.region(), {"path": str(labels)})
    manifest = run_generate(config).manifest
    for entry in manifest.patches:
        mask = np.asarray(Image.open(config.writer.staging_dir / entry["files"]["mask"]))
        image = np.asarray(Image.open(config.writer.staging_dir / entry["files"]["image"]))
        np.testing.assert_array_equal(mask == IGNORE, ~image.any(axis=-1))
        assert set(np.unique(mask)) <= {1, IGNORE}
    assert any(entry["summary"]["class_pixels"].get("255") for entry in manifest.patches)


def test_min_label_ratio_and_stratified_split_use_the_raster_classes(tmp_path: Path) -> None:
    imagery = make_imagery(tmp_path)
    # Class 1 only in the top half of the label raster; the rest is background (0).
    data_transform = imagery.transform
    path = tmp_path / "half.tif"
    data = np.zeros((imagery.height, imagery.width), dtype=np.uint8)
    data[: imagery.height // 2] = 10
    with rasterio.open(
        path,
        "w",
        driver="GTiff",
        height=imagery.height,
        width=imagery.width,
        count=1,
        dtype="uint8",
        crs=CRS.from_epsg(IMAGERY_EPSG),
        transform=data_transform,
    ) as dst:
        dst.write(data, 1)
    everything = config_for(
        tmp_path, _geotiff(imagery), imagery.region(), {"path": str(path)}, staging="all"
    )
    kept = config_for(
        tmp_path,
        _geotiff(imagery),
        imagery.region(),
        {"path": str(path)},
        staging="kept",
        min_label_ratio=0.5,
        split={"strategy": "stratified", "test_ratio": 0.3, "val_ratio": 0.0},
    )
    all_patches = run_generate(everything).manifest.patches
    result = run_generate(kept)
    labeled = [
        entry
        for entry in all_patches
        if entry["summary"]["class_pixels"].get("1", 0) >= 0.5 * 64 * 64
    ]
    assert 0 < len(result.manifest.patches) == len(labeled) < len(all_patches)
    assert result.split_counts is not None
    assert sum(result.split_counts[name] for name in ("train", "val", "test")) == len(labeled)
    assert result.manifest.class_map == {"value_10": 1, "value_20": 2, "value_30_40": 3}


def test_geotiff_masks_hold_the_same_labels_on_the_patch_grid(tmp_path: Path) -> None:
    imagery, labels = _small_case(tmp_path)
    png = config_for(
        tmp_path, _geotiff(imagery), imagery.region(), {"path": str(labels)}, staging="png"
    )
    data = png.model_dump(mode="json")
    data["writer"].update(staging_dir=str(tmp_path / "tif"), image_format="tif", mask_format="tif")
    tif = MapcvConfig.model_validate(data)
    png_manifest = run_generate(png).manifest
    tif_manifest = run_generate(tif).manifest
    assert len(png_manifest.patches) == len(tif_manifest.patches) > 0
    for a, b in zip(png_manifest.patches, tif_manifest.patches):
        expected = np.asarray(Image.open(tmp_path / "png" / a["files"]["mask"]))
        with rasterio.open(tmp_path / "tif" / b["files"]["mask"]) as src:
            np.testing.assert_array_equal(src.read(1), expected)
            assert tuple(src.transform)[:6] == pytest.approx(tif_manifest.patch_transform(b))
            assert src.nodata == IGNORE


# ── Manifest, resume, planning ───────────────────────────────────────────────


def test_manifest_records_the_settings_and_a_fingerprint(tmp_path: Path) -> None:
    imagery, labels = _small_case(tmp_path)
    classes = {10: {"id": 1, "name": "tree"}, 20: 2, 30: {"id": 3, "name": "water"}, 40: 3}
    config = config_for(
        tmp_path, _geotiff(imagery), imagery.region(), {"path": str(labels), "classes": classes}
    )
    manifest = run_generate(config).manifest
    target = manifest.target
    assert target is not None
    assert target.type == "segmentation" and target.dtype == "uint8"
    assert target.class_map == {"tree": 1, "value_20": 2, "water": 3}
    assert target.ignore_index == IGNORE
    recorded = target.labels
    assert recorded is not None
    assert recorded["type"] == "raster" and recorded["band"] == 1
    assert recorded["classes"]["40"] == {"id": 3, "name": "water"}
    assert recorded["unmapped"] == "background" and recorded["resampling"] == "nearest"
    assert "path" not in recorded and "ignore_index" not in recorded
    assert recorded["fingerprint"]["kind"] == "file"
    assert recorded["fingerprint"]["size"] == labels.stat().st_size
    # Round-trips through the file unchanged.
    saved = json.loads((config.writer.staging_dir / "manifest.json").read_text())
    assert saved["target"]["labels"] == recorded


def test_resume_accepts_the_same_labels_and_refuses_changes(tmp_path: Path) -> None:
    imagery, labels = _small_case(tmp_path)
    config = config_for(tmp_path, _geotiff(imagery), imagery.region(), {"path": str(labels)})
    first = run_generate(config)
    assert first.new_patches > 0
    again = run_generate(config)
    assert again.new_patches == 0
    # Another class table.
    remapped = config_for(
        tmp_path,
        _geotiff(imagery),
        imagery.region(),
        {"path": str(labels), "classes": {10: 1, 20: 2}},
    )
    with pytest.raises(ManifestMismatchError, match="labels"):
        run_generate(remapped)
    # Another label file under the same name.
    transform, width, height = labels_around(imagery, IMAGERY_EPSG, 2.0, offset=(0.3, 0.3))
    make_labels(tmp_path, transform, width, height, seed=99)
    with pytest.raises(ManifestMismatchError, match="labels"):
        run_generate(config)


def test_vector_labels_record_no_type(tmp_path: Path) -> None:
    """Manifests of polygon labels are unchanged by labels.type (resume keeps working)."""
    labels = LabelsConfig(path=tmp_path / "x.geojson")
    assert labels.type == "vector"
    config = MapcvConfig.model_validate(
        {
            "region": {"west": 3.0, "south": 48.8, "east": 3.01, "north": 48.81},
            "imagery": {"type": "xyz", "zoom": 15, "source": "esri_satellite"},
            "labels": {"path": str(tmp_path / "x.geojson")},
            "sampler": {"patch_size": 64},
            "writer": {"staging_dir": str(tmp_path / "out")},
        }
    )
    assert isinstance(config.labels, LabelsConfig)
    (tmp_path / "x.geojson").write_text('{"type": "FeatureCollection", "features": []}')
    target = create_target(config)
    source_meta = open_raster_source(config.region, config.primary_imagery).metadata
    target.prepare(source_meta)
    record = target.record()
    assert record is not None and record.labels is not None
    assert set(record.labels) == {"label_field", "classes", "all_touched", "sha256"}


def test_plan_describes_the_label_raster(tmp_path: Path) -> None:
    imagery, labels = _small_case(tmp_path)
    config = config_for(tmp_path, _geotiff(imagery), imagery.region(), {"path": str(labels)})
    estimate = plan(config)
    assert estimate.labels is not None
    assert estimate.labels.raster is not None
    assert "EPSG:32631" in estimate.labels.raster and "2.00 m/px" in estimate.labels.raster
    assert estimate.labels.classes == {"value_10": 1, "value_20": 2, "value_30_40": 3}
    assert not estimate.warnings


def test_plan_and_generate_warn_when_the_raster_misses_the_imagery(tmp_path: Path) -> None:
    imagery = make_imagery(tmp_path)
    far = Affine(2.0, 0.0, imagery.transform.c + 50_000, 0.0, -2.0, imagery.transform.f)
    labels = make_labels(tmp_path, far, 50, 50)
    config = config_for(tmp_path, _geotiff(imagery), imagery.region(), {"path": str(labels)})
    estimate = plan(config)
    assert any("does not overlap" in message for message in estimate.warnings)
    with pytest.warns(UserWarning, match="does not overlap the imagery"):
        manifest = run_generate(config).manifest
    for entry in manifest.patches:
        assert set(entry["summary"]["class_pixels"]) == {"255"}


def test_unreadable_label_rasters_fail_clearly(tmp_path: Path) -> None:
    imagery = make_imagery(tmp_path)
    floats = tmp_path / "float.tif"
    with rasterio.open(
        floats,
        "w",
        driver="GTiff",
        height=10,
        width=10,
        count=1,
        dtype="float32",
        crs=CRS.from_epsg(IMAGERY_EPSG),
        transform=imagery.transform,
    ) as dst:
        dst.write(np.zeros((1, 10, 10), dtype=np.float32))
    config = config_for(tmp_path, _geotiff(imagery), imagery.region(), {"path": str(floats)})
    with pytest.raises(ValueError, match="integer class values"):
        run_generate(config)
    labels = make_labels(tmp_path, imagery.transform, 10, 10)
    config = config_for(
        tmp_path, _geotiff(imagery), imagery.region(), {"path": str(labels), "band": 2}
    )
    with pytest.raises(ValueError, match="1 band"):
        run_generate(config)
    missing = config_for(
        tmp_path, _geotiff(imagery), imagery.region(), {"path": str(tmp_path / "nope.tif")}
    )
    with pytest.raises(FileNotFoundError):
        run_generate(missing)
    assert plan(missing).labels is not None
    assert any("not found" in message for message in plan(missing).warnings)


def _serve(httpserver: Any, payload: bytes) -> str:
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
            headers={"Content-Range": f"bytes {start}-{end}/{len(payload)}", "ETag": '"l1"'},
        )

    httpserver.expect_request("/labels.tif").respond_with_handler(handler)
    return str(httpserver.url_for("/labels.tif"))


def test_remote_label_raster_gives_the_same_masks(tmp_path: Path, httpserver: Any) -> None:
    imagery, labels = _small_case(tmp_path)
    url = _serve(httpserver, labels.read_bytes())
    local = config_for(
        tmp_path, _geotiff(imagery), imagery.region(), {"path": str(labels)}, staging="local"
    )
    remote = config_for(
        tmp_path, _geotiff(imagery), imagery.region(), {"path": url}, staging="remote"
    )
    local_manifest = run_generate(local).manifest
    remote_manifest = run_generate(remote).manifest
    assert len(local_manifest.patches) == len(remote_manifest.patches) > 0
    for a, b in zip(local_manifest.patches, remote_manifest.patches):
        assert (tmp_path / "local" / a["files"]["mask"]).read_bytes() == (
            tmp_path / "remote" / b["files"]["mask"]
        ).read_bytes()
    assert remote_manifest.target is not None and remote_manifest.target.labels is not None
    fingerprint = remote_manifest.target.labels["fingerprint"]
    assert fingerprint["kind"] == "url" and fingerprint["url"] == url
    assert fingerprint["etag"] == '"l1"'


# ── Config validation ────────────────────────────────────────────────────────


def _labels(**labels: Any) -> RasterLabelsConfig:
    return RasterLabelsConfig.model_validate({"type": "raster", "path": "lc.tif", **labels})


def test_classes_accept_ids_and_names_and_merge_values() -> None:
    labels = _labels(
        classes={10: 1, "20": {"id": 2, "name": " grass "}, 90: 4, 95: 4, 0: 0, 30: {"id": 2}}
    )
    assert labels.class_map() == {"value_10": 1, "grass": 2, "value_90_95": 4}
    assert labels.classes[30].name == "grass"
    assert labels.classes[0].id == 0 and labels.classes[0].name is None
    assert labels.band == 1 and labels.unmapped == "background" and labels.ignore_index == 255


@pytest.mark.parametrize(
    ("labels", "message"),
    [
        ({}, "classes"),
        ({"classes": {}}, "at least one raster value"),
        ({"classes": {10: 255}}, "ignore_index"),
        ({"classes": {10: 256}}, "less than or equal to 255"),
        ({"classes": {10: {"id": 1, "name": "a"}, 20: {"id": 1, "name": "b"}}}, "two names"),
        ({"classes": {10: {"id": 1, "name": "a"}, 20: {"id": 2, "name": "a"}}}, "name 'a'"),
        ({"classes": {10: {"id": 0, "name": "bg"}}}, "background"),
        ({"classes": {True: 1}}, "integer"),
        ({"classes": {"ten": 1}}, "valid integer"),
        ({"classes": [10, 20]}, "valid dictionary"),
        ({"classes": {10: {"id": 1, "name": " "}}}, "must not be empty"),
        ({"classes": {10: 1}, "nodata": True}, "must be integers"),
        ({"classes": {10: 1}, "ignore_values": [3, False]}, "must be integers"),
        ({"classes": {10: 1}, "ignore_values": [10]}, "both list"),
        ({"classes": {10: 1}, "nodata": 10}, "nodata"),
        ({"classes": {10: 1}, "unmapped": "ignore", "ignore_index": None}, "unmapped"),
        ({"classes": {10: 1}, "resampling": "bilinear"}, "nearest"),
        ({"classes": {10: 1}, "band": 0}, "greater than or equal to 1"),
        ({"classes": {10: 1}, "all_touched": True}, "Extra inputs"),
        ({"classes": {10: 1}, "path": "http://example.com/lc.tif"}, "labels.path"),
        ({"classes": {10: 1}, "path": "https://user:pw@example.com/lc.tif"}, "credentials"),
    ],
)
def test_invalid_raster_label_settings_are_rejected(labels: Dict[str, Any], message: str) -> None:
    with pytest.raises(ValidationError, match=message):
        _labels(**labels)


def _config(labels: Dict[str, Any]) -> MapcvConfig:
    return MapcvConfig.model_validate(
        {
            "region": {"west": 3.0, "south": 48.8, "east": 3.01, "north": 48.81},
            "imagery": {"type": "xyz", "zoom": 15, "source": "esri_satellite"},
            "labels": labels,
            "sampler": {"patch_size": 64},
            "writer": {"staging_dir": "out"},
        }
    )


def test_labels_type_defaults_to_vector_and_explains_rasters() -> None:
    assert isinstance(_config({"path": "a.geojson"}).labels, LabelsConfig)
    assert isinstance(_config({"type": "vector", "path": "a.kml"}).labels, LabelsConfig)
    raster = _config({"type": "raster", "path": "s3://bucket/lc.tif", "classes": {1: 1}}).labels
    assert isinstance(raster, RasterLabelsConfig)
    with pytest.raises(ValidationError, match="labels.type: raster"):
        _config({"path": "landcover.tif"})
    with pytest.raises(ValidationError, match="label_field"):
        _config({"type": "raster", "path": "a.tif", "classes": {1: 1}, "label_field": "x"})


def test_detection_refuses_raster_labels() -> None:
    with pytest.raises(ValidationError, match="detection needs vector labels"):
        MapcvConfig.model_validate(
            {
                "task": "detection",
                "region": {"west": 3.0, "south": 48.8, "east": 3.01, "north": 48.81},
                "imagery": {"type": "xyz", "zoom": 15, "source": "esri_satellite"},
                "labels": {"type": "raster", "path": "lc.tif", "classes": {1: 1}},
                "sampler": {"patch_size": 64},
                "writer": {"staging_dir": "out"},
            }
        )


def test_relative_label_raster_paths_resolve_but_urls_do_not(tmp_path: Path) -> None:
    base = (
        "region: {west: 3.0, south: 48.8, east: 3.01, north: 48.81}\n"
        "imagery: {type: xyz, zoom: 15, source: esri_satellite}\n"
        "sampler: {patch_size: 64}\nwriter: {staging_dir: out}\n"
    )
    config_path = tmp_path / "sub" / "mapcv.yaml"
    config_path.parent.mkdir()
    config_path.write_text(base + "labels: {type: raster, path: lc.tif, classes: {10: 1}}\n")
    labels = MapcvConfig.from_yaml(config_path).labels
    assert isinstance(labels, RasterLabelsConfig)
    assert Path(labels.path) == tmp_path / "sub" / "lc.tif"
    url = "https://example.com/data/lc.tif"
    config_path.write_text(base + f"labels: {{type: raster, path: '{url}', classes: {{10: 1}}}}\n")
    labels = MapcvConfig.from_yaml(config_path).labels
    assert isinstance(labels, RasterLabelsConfig) and labels.path == url


# ── CLI ──────────────────────────────────────────────────────────────────────


def _region_yaml(region: Dict[str, float]) -> str:
    return "region:\n" + "".join(f"  {key}: {value}\n" for key, value in region.items())


def test_cli_validate_plan_generate_info(tmp_path: Path) -> None:
    imagery, _ = _small_case(tmp_path)
    config_path = tmp_path / "mapcv.yaml"
    config_path.write_text(
        _region_yaml(imagery.region()) + "imagery: {type: geotiff, path: scene.tif}\n"
        "labels:\n  type: raster\n  path: labels.tif\n  classes:\n"
        "    10: {id: 1, name: tree_cover}\n    20: {id: 2, name: grassland}\n"
        "sampler: {patch_size: 64, edge_strategy: drop}\n"
        "writer: {staging_dir: out, image_format: png}\n"
        "split: {strategy: spatial}\n"
    )
    result = runner.invoke(app, ["validate", str(config_path)])
    assert result.exit_code == 0, result.output
    assert flat("raster band 1 · 2 class(es)") in flat(result.output)

    result = runner.invoke(app, ["plan", str(config_path)])
    assert result.exit_code == 0, result.output
    assert flat("raster labels.tif · EPSG:32631") in flat(result.output)
    assert flat("grassland → 2, tree_cover → 1") in flat(result.output)

    result = runner.invoke(app, ["generate", str(config_path), "--yes"])
    assert result.exit_code == 0, result.output
    result = runner.invoke(app, ["info", str(tmp_path / "out")])
    assert result.exit_code == 0, result.output
    assert flat("marks pixels without imagery or label") in flat(result.output)
    assert flat("tree_cover") in flat(result.output)
    assert flat("ignored (no imagery or label)") in flat(result.output)

    (tmp_path / "labels.tif").unlink()
    result = runner.invoke(app, ["validate", str(config_path)])
    assert result.exit_code == 0 and flat("labels.path not found") in flat(result.output)


def test_init_template_shows_raster_labels(tmp_path: Path) -> None:
    result = runner.invoke(app, ["init", "--template", "geotiff", "--stdout"])
    assert result.exit_code == 0
    assert "type: raster" in result.output


def test_init_wizard_offers_label_rasters(tmp_path: Path) -> None:
    imagery = make_imagery(tmp_path)
    transform, width, height = labels_around(imagery, 4326, 1.3e-5)
    labels = make_labels(tmp_path, transform, width, height, epsg=4326)
    out = tmp_path / "mapcv.yaml"
    # imagery, path, area (whole file), labels, patch size, output folder, split.
    answers = "\n".join(["geotiff", str(imagery.path), "", str(labels), "64", "./ds", "n"])
    result = runner.invoke(app, ["init", str(out), "--interactive"], input=answers + "\n")
    assert result.exit_code == 0, result.output
    assert flat("A label raster") in flat(result.output)
    assert flat("EPSG:4326") in flat(result.output)
    config = MapcvConfig.from_yaml(out)
    assert isinstance(config.labels, RasterLabelsConfig)
    # Every value under the area except NoData becomes a class; 0 is background.
    assert {value: target.id for value, target in config.labels.classes.items()} == {
        0: 0,
        10: 10,
        20: 20,
        30: 30,
        40: 40,
        99: 99,
    }
    assert config.labels.class_map()["value_99"] == 99
    estimate = plan(config)
    assert estimate.labels is not None and not estimate.warnings


def test_init_wizard_writes_an_editable_config_for_an_unreadable_raster(tmp_path: Path) -> None:
    imagery = make_imagery(tmp_path)
    out = tmp_path / "mapcv.yaml"
    answers = "\n".join(
        ["geotiff", str(imagery.path), "", str(tmp_path / "missing.tif"), "64", "./ds", "n"]
    )
    result = runner.invoke(app, ["init", str(out), "--interactive"], input=answers + "\n")
    assert result.exit_code == 0, result.output
    assert flat("Cannot read that file") in flat(result.output)
    config = MapcvConfig.from_yaml(out)
    assert isinstance(config.labels, RasterLabelsConfig)


# ── Sampler edge cases ───────────────────────────────────────────────────────


def _sampler(
    path: Path, imagery_crs: str = f"EPSG:{IMAGERY_EPSG}", **labels: Any
) -> LabelRasterSampler:
    config = RasterLabelsConfig.model_validate(
        {"type": "raster", "path": str(path), "classes": CLASSES, **labels}
    )
    return LabelRasterSampler(config, imagery_crs)


def test_32_bit_labels_are_classified_without_a_lookup_table(tmp_path: Path) -> None:
    transform = Affine(1.0, 0.0, 500_000.0, 0.0, -1.0, 5_400_000.0)
    cases: Sequence[Tuple[str, Tuple[int, ...], Dict[int, int]]] = [
        ("uint32", (5, 70_000, 9, 123_456), {70_000: 1, 5: 2}),
        ("int32", (-70_000, 3, 9, 8), {-70_000: 1, 3: 2}),
    ]
    for dtype, values, classes in cases:
        path = make_labels(
            tmp_path, transform, 40, 30, dtype=dtype, values=values, nodata=9, name=f"{dtype}.tif"
        )
        sampler = _sampler(path, classes=classes)
        mask = sampler.sample(six(transform), 30, 40)
        with rasterio.open(path) as src:
            raw = src.read(1).astype(np.int64)
        np.testing.assert_array_equal(mask, expected_mask(raw, classes, nodata=9))


def test_windows_off_the_label_raster_are_ignored(tmp_path: Path) -> None:
    transform = Affine(1.0, 0.0, 500_000.0, 0.0, -1.0, 5_400_000.0)
    labels = make_labels(tmp_path, transform, 40, 30)
    far = six(transform * Affine.translation(1_000, 1_000))
    same_grid = _sampler(labels)
    assert same_grid.grid_offset(far) == (1_000, 1_000)
    assert (same_grid.sample(far, 8, 8) == IGNORE).all()
    # Same CRS on another grid: the separable path.
    shifted = six(transform * Affine.translation(1_000.5, 1_000))
    assert same_grid.grid_offset(shifted) is None
    assert (same_grid.sample(shifted, 8, 8) == IGNORE).all()
    # Another CRS: the general path; background instead of ignore without an ignore_index.
    lon, lat = Transformer.from_crs("EPSG:32631", "EPSG:4326", always_xy=True).transform(
        499_000.0, 5_399_000.0
    )
    other = _sampler(labels, "EPSG:4326", ignore_index=None)
    window = (1e-5, 0.0, lon, 0.0, -1e-5, lat)
    assert (other.sample(window, 8, 8) == 0).all()


def test_the_general_path_reads_more_when_the_perimeter_misses_pixels(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    imagery = make_imagery(tmp_path)
    transform, width, height = labels_around(imagery, 4326, 1.3e-5)
    labels = make_labels(tmp_path, transform, width, height, epsg=4326)
    sampler = _sampler(labels)
    window = six(imagery.transform)
    expected = sampler.sample(window, 200, 200)
    # A perimeter estimate that is far too small, or none at all.
    estimates: Sequence[Optional[Tuple[int, int, int, int]]] = [(0, 1, 0, 1), None]
    for estimate in estimates:
        monkeypatch.setattr(sampler, "_perimeter_window", lambda *args, e=estimate: e)
        np.testing.assert_array_equal(sampler.sample(window, 200, 200), expected)


def test_perimeter_window_is_none_without_finite_coordinates(tmp_path: Path) -> None:
    transform = Affine(1.0, 0.0, 500_000.0, 0.0, -1.0, 5_400_000.0)
    sampler = _sampler(make_labels(tmp_path, transform, 4, 4))
    nan = float("nan")
    assert sampler._perimeter_window((nan, 0.0, 0.0, 0.0, nan, 0.0), 3, 3) is None


def test_a_tiny_label_raster_still_counts_as_overlapping(tmp_path: Path) -> None:
    from mapcv.imagery import RasterMetadata

    source = RasterMetadata(
        source_type="geotiff",
        product_id="big",
        width=200_000,
        height=200_000,
        bands=["b1"],
        dtype="uint8",
        crs=f"EPSG:{IMAGERY_EPSG}",
        transform=(1.0, 0.0, 400_000.0, 0.0, -1.0, 5_500_000.0),
        chunk_rows=1024,
    )
    x, y = 500_000.3, 5_400_000.6  # between two of the 64 x 64 sample points
    small = make_labels(tmp_path, Affine(1.0, 0.0, x, 0.0, -1.0, y), 2, 2, name="utm.tif")
    assert _sampler(small).overlaps(source)
    lon, lat = Transformer.from_crs("EPSG:32631", "EPSG:4326", always_xy=True).transform(x, y)
    geographic = make_labels(
        tmp_path, Affine(1e-5, 0.0, lon, 0.0, -1e-5, lat), 2, 2, epsg=4326, name="geo.tif"
    )
    assert _sampler(geographic).overlaps(source)
    far = make_labels(tmp_path, Affine(1.0, 0.0, 10.0, 0.0, -1.0, 10.0), 2, 2, name="far.tif")
    assert not _sampler(far).overlaps(source)


def test_label_rasters_without_a_crs_are_refused(tmp_path: Path) -> None:
    path = tmp_path / "nocrs.tif"
    with rasterio.open(
        path,
        "w",
        driver="GTiff",
        height=4,
        width=4,
        count=1,
        dtype="uint8",
        transform=Affine(2.0, 0.0, 100.0, 0.0, -2.0, 200.0),
    ) as dst:
        dst.write(np.zeros((1, 4, 4), dtype=np.uint8))
    with pytest.raises(ValueError, match="no usable CRS"):
        _sampler(path)


def test_the_target_needs_prepare_first() -> None:
    target = RasterSegmentationTarget(
        RasterLabelsConfig.model_validate({"type": "raster", "path": "x.tif", "classes": {1: 1}})
    )
    with pytest.raises(RuntimeError, match="prepare"):
        target.sampler
    with pytest.raises(RuntimeError, match="prepare"):
        target.record()


# ── init wizard: label raster values ─────────────────────────────────────────


def _bbox(imagery: Imagery) -> Tuple[float, float, float, float]:
    region = imagery.region()
    return region["west"], region["south"], region["east"], region["north"]


def test_wizard_maps_non_identity_values_to_sequential_ids(tmp_path: Path) -> None:
    from mapcv.cli import _ask_label_raster

    imagery = make_imagery(tmp_path)
    labels = make_labels(
        tmp_path, imagery.transform, 320, 288, dtype="uint16", values=VALUES_U16, nodata=NODATA_U16
    )
    lines = _ask_label_raster(str(labels), _bbox(imagery))
    assert "    1000: {id: 1, name: value_1000}" in lines
    assert "    4000: {id: 4, name: value_4000}" in lines
    assert not any("65535" in line for line in lines)


def test_wizard_writes_a_placeholder_when_values_cannot_be_used(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from mapcv import cli

    imagery = make_imagery(tmp_path)
    bbox = _bbox(imagery)
    placeholder = "    1: {id: 1, name: class_1}"
    # Too many distinct values for class IDs (and more than the 20 listed).
    many = make_labels(
        tmp_path,
        imagery.transform,
        320,
        288,
        dtype="uint16",
        values=range(1, 400),
        nodata=None,
        name="many.tif",
    )
    assert placeholder in cli._ask_label_raster(str(many), bbox)
    # Nothing under the area.
    far = make_labels(tmp_path, Affine(1.0, 0.0, 10.0, 0.0, -1.0, 10.0), 8, 8, name="far.tif")
    assert placeholder in cli._ask_label_raster(str(far), bbox)
    # Float values.
    floats = tmp_path / "float.tif"
    with rasterio.open(
        floats,
        "w",
        driver="GTiff",
        height=8,
        width=8,
        count=1,
        dtype="float32",
        crs=CRS.from_epsg(IMAGERY_EPSG),
        transform=imagery.transform,
    ) as dst:
        dst.write(np.zeros((1, 8, 8), dtype=np.float32))
    assert placeholder in cli._ask_label_raster(str(floats), bbox)
    # Reading fails.
    good = make_labels(tmp_path, imagery.transform, 320, 288, name="good.tif")

    def fail(*args: Any) -> Any:
        raise RuntimeError("connection reset")

    monkeypatch.setattr(cli, "_sample_label_values", fail)
    assert placeholder in cli._ask_label_raster(str(good), bbox)


def test_wizard_samples_an_overview_or_the_centre_of_a_large_area(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from mapcv import cli
    from mapcv.geotiff import GeoTiff

    imagery = make_imagery(tmp_path)
    bbox = _bbox(imagery)
    monkeypatch.setattr(cli, "_WIZARD_SAMPLE_PIXELS", 2_500)
    plain = make_labels(tmp_path, imagery.transform, 320, 288, name="plain.tif")
    counts, sampled = cli._sample_label_values(GeoTiff(plain), bbox)
    assert sampled and 0 < sum(counts.values()) <= 2_500
    with_overviews = make_labels(tmp_path, imagery.transform, 320, 288, name="ovr.tif")
    with rasterio.open(with_overviews, "r+") as dst:
        dst.build_overviews([8], Resampling.nearest)
    counts, sampled = cli._sample_label_values(GeoTiff(with_overviews), bbox)
    assert sampled and 0 < sum(counts.values()) <= 2_500
