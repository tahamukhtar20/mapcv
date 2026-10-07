"""GeoTIFF, NPY and world-file outputs, read back with rasterio (an independent reader).

Georeferencing errors are silent: a wrong tie point or a CRS that is off by one
code still gives a file every viewer opens. These tests therefore open each written
``.tif`` with rasterio (GDAL) and compare CRS, transform, dtype, band count, no-data
and pixels with what mapcv holds in memory and records in the manifest.
"""

from __future__ import annotations

import io
import json
import sys
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional, Tuple

import numpy as np
import numpy.typing as npt
import pytest
import yaml
from PIL import Image

from mapcv._mapcv_rs import write_geotiffs
from mapcv.config import MapcvConfig
from mapcv.imagery import RasterMetadata
from mapcv.manifest import (
    Manifest,
    ManifestEntry,
    ManifestMismatchError,
    SourceRecord,
    TargetRecord,
)
from mapcv.pipeline import run_generate, run_split
from mapcv.sampler import PatchMeta
from mapcv.splitter import SplitterConfig
from mapcv.writer import WriterConfig, write_patches
from mapcv.writers import FilesWriter

rasterio = pytest.importorskip("rasterio", reason="GeoTIFF outputs are checked with rasterio")
pyproj = pytest.importorskip("pyproj", reason="footprints are checked against pyproj")

FIXTURE = Path(__file__).parent / "fixtures" / "mapcv-0.2.0"

Transform = Tuple[float, float, float, float, float, float]


# ── helpers ──────────────────────────────────────────────────────────────────


class _NoiseTiles(BaseHTTPRequestHandler):
    """Serves a deterministic noise tile per (z, x, y), so patches have real pixel content."""

    def log_message(self, *args: object) -> None:
        pass

    def do_GET(self) -> None:  # noqa: N802 - http.server API
        z, x, y = (int(part) for part in self.path.strip("/").split(".")[0].split("/"))
        rng = np.random.default_rng([z, x, y])
        pixels = rng.integers(1, 256, size=(256, 256, 3), dtype=np.uint8)
        buffer = io.BytesIO()
        Image.fromarray(pixels).save(buffer, "PNG")
        body = buffer.getvalue()
        self.send_response(200)
        self.send_header("Content-Type", "image/png")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)


@pytest.fixture
def tile_port() -> Iterator[int]:
    server = ThreadingHTTPServer(("127.0.0.1", 0), _NoiseTiles)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield int(server.server_address[1])
    finally:
        server.shutdown()
        thread.join()


def _xyz_config(tmp_path: Path, name: str, port: int, **writer: Any) -> MapcvConfig:
    data = yaml.safe_load((FIXTURE / "mapcv.yaml").read_text(encoding="utf-8"))
    data["imagery"]["url_template"] = f"http://127.0.0.1:{port}/{{z}}/{{x}}/{{y}}.png"
    data["labels"]["path"] = str(FIXTURE / "labels.geojson")
    data["writer"] = {"staging_dir": str(tmp_path / name), **writer}
    return MapcvConfig.model_validate(data)


def _close(actual: Transform, expected: Transform) -> None:
    assert actual == pytest.approx(expected, rel=1e-9, abs=1e-9)


def _check_geotiff(
    path: Path,
    manifest: Manifest,
    entry: ManifestEntry,
    *,
    count: int,
    dtype: str,
    nodata: Optional[float],
) -> npt.NDArray[Any]:
    """Assert a patch file's georeferencing; returns its pixels, ``(bands, rows, cols)``."""
    expected = manifest.patch_transform(entry)
    assert manifest.source.crs is not None
    with rasterio.open(path) as src:
        assert src.crs.to_epsg() == int(manifest.source.crs.split(":")[1])
        _close(tuple(src.transform)[:6], expected)
        assert (src.count, src.dtypes[0]) == (count, dtype)
        assert len(set(src.dtypes)) == 1
        if nodata is None:
            assert src.nodata is None
        elif np.isnan(nodata):
            assert src.nodata is not None and np.isnan(src.nodata)
        else:
            assert src.nodata == nodata
        patch_size = (manifest.sampler or {})["patch_size"]
        assert (src.height, src.width) == (patch_size, patch_size)
        bounds = manifest.patch_bounds(entry)
        assert tuple(src.bounds) == pytest.approx(
            (bounds[0], bounds[1], bounds[2], bounds[3]), rel=1e-9
        )
        return src.read()  # type: ignore[no-any-return]


def _patch_meta(row: int, col: int) -> PatchMeta:
    return PatchMeta(row=row, col=col, padded=False, empty_ratio=0.0)


def _manifest(crs: str, transform: Transform, bands: List[str], patch_size: int) -> Manifest:
    return Manifest(
        sources=[SourceRecord(crs=crs, transform=transform, bands=bands)],
        sampler={"patch_size": patch_size},
    )


# ── XYZ: uint8 RGB in EPSG:3857, masks with an ignore index ─────────────────


def test_xyz_geotiff_images_and_masks_are_georeferenced_and_match_png(
    tmp_path: Path, tile_port: int
) -> None:
    png = run_generate(_xyz_config(tmp_path, "png", tile_port)).manifest
    result = run_generate(
        _xyz_config(tmp_path, "tif", tile_port, image_format="tif", mask_format="tif")
    )
    manifest = result.manifest
    staging = tmp_path / "tif"
    assert len(manifest.patches) == len(png.patches) > 0
    assert manifest.source.crs == "EPSG:3857"
    assert manifest.source.patch_shape == [3, 192, 192]
    assert manifest.writer is not None
    assert (manifest.writer["image_format"], manifest.writer["mask_format"]) == ("tif", "tif")
    assert manifest.ignore_index == 255

    for index, (entry, png_entry) in enumerate(zip(manifest.patches, png.patches)):
        assert entry["files"] == {
            "image": f"Images/patch_{index:07d}.tif",
            "mask": f"Masks/patch_{index:07d}.tif",
        }
        assert entry["summary"] == png_entry["summary"] or (
            entry["summary"]["class_pixels"] == png_entry["summary"]["class_pixels"]
        )
        image = _check_geotiff(
            staging / entry["files"]["image"], manifest, entry, count=3, dtype="uint8", nodata=None
        )
        mask = _check_geotiff(
            staging / entry["files"]["mask"], manifest, entry, count=1, dtype="uint8", nodata=255
        )
        png_image = np.asarray(Image.open(tmp_path / "png" / png_entry["files"]["image"]))
        png_mask = np.asarray(Image.open(tmp_path / "png" / png_entry["files"]["mask"]))
        np.testing.assert_array_equal(np.moveaxis(image, 0, -1), png_image)
        np.testing.assert_array_equal(mask[0], png_mask)

        # Image and mask of one patch open aligned: the same transform, bit for bit.
        with (
            rasterio.open(staging / entry["files"]["image"]) as img,
            rasterio.open(staging / entry["files"]["mask"]) as msk,
        ):
            assert tuple(img.transform) == tuple(msk.transform)
            assert img.crs == msk.crs and img.bounds == msk.bounds
            assert img.descriptions == ("red", "green", "blue")
            assert img.colorinterp[0].name == "red"
            assert msk.nodatavals == (255.0,)

    # The padded edge patches carry the ignore value, declared as no-data.
    padded = [entry for entry in manifest.patches if entry["padded"]]
    assert padded
    with rasterio.open(staging / padded[0]["files"]["mask"]) as src:
        assert (src.read(1) == 255).any()
        assert src.read_masks(1).min() == 0  # rasterio masks them as no-data


def test_png_output_is_unchanged_by_the_new_options(tmp_path: Path, tile_port: int) -> None:
    default = run_generate(_xyz_config(tmp_path, "a", tile_port)).manifest
    explicit = run_generate(
        _xyz_config(tmp_path, "b", tile_port, mask_format="png", footprints=False)
    ).manifest
    for left, right in zip(default.patches, explicit.patches):
        for role in ("image", "mask"):
            assert (tmp_path / "a" / left["files"][role]).read_bytes() == (
                tmp_path / "b" / right["files"][role]
            ).read_bytes()
    assert default.writer == explicit.writer
    assert default.writer == {
        "layout": "files",
        "image_format": "png",
        "jpg_quality": 95,
        "jpg_subsampling": "4:2:0",
        "mask_format": "png",
    }
    assert (tmp_path / "a" / "patches.geojson").is_file()
    assert not (tmp_path / "b" / "patches.geojson").exists()


# ── EOPF-like: float32 multiband in a projected UTM CRS ─────────────────────

UTM = "EPSG:32633"
UTM_TRANSFORM: Transform = (10.0, 0.0, 500000.0, 0.0, -10.0, 4500000.0)


class _FakeEopf:
    """A 4-band float32 raster with NaN holes, in UTM zone 33N."""

    def __init__(self) -> None:
        rng = np.random.default_rng(7)
        self.image = rng.normal(1000, 300, size=(50, 44, 4)).astype(np.float32)
        self.image[5:9, 6:12] = np.nan
        self.valid = np.all(np.isfinite(self.image), axis=-1)
        self.metadata = RasterMetadata(
            source_type="eopf_zarr",
            product_id="S2_TEST.zarr",
            width=44,
            height=50,
            bands=["b04", "b03", "b02", "b08"],
            dtype="float32",
            crs=UTM,
            transform=UTM_TRANSFORM,
            chunk_rows=16,
        )

    def read_window(
        self, row_start: int, row_stop: int, col_start: int, col_stop: int
    ) -> Tuple[npt.NDArray[np.float32], npt.NDArray[np.bool_]]:
        window = (slice(row_start, row_stop), slice(col_start, col_stop))
        return self.image[window], self.valid[window]

    def close(self) -> None:
        pass


_LABELS = {
    "type": "FeatureCollection",
    "features": [
        {
            "type": "Feature",
            "properties": {"kind": "3"},
            # UTM metres: the test replaces the lon/lat reprojection with the identity.
            "geometry": {
                "type": "Polygon",
                "coordinates": [
                    [
                        [500100, 4499600],
                        [500300, 4499600],
                        [500300, 4499900],
                        [500100, 4499900],
                        [500100, 4499600],
                    ]
                ],
            },
        }
    ],
}


def _eopf_config(tmp_path: Path, name: str, **writer: Any) -> MapcvConfig:
    labels = tmp_path / "labels.geojson"
    labels.write_text(json.dumps(_LABELS))
    return MapcvConfig.model_validate(
        {
            "region": {"west": 9.0, "south": 45.0, "east": 9.1, "north": 45.1},
            "imagery": {"type": "eopf_zarr", "path": str(tmp_path / "unused.zarr")},
            "labels": {"path": str(labels), "label_field": "kind"},
            "sampler": {"patch_size": 16, "stride": 16, "edge_strategy": "pad"},
            "writer": {"staging_dir": str(tmp_path / name), **writer},
            "split": {"test_ratio": 0.2, "val_ratio": 0.2, "strategy": "random"},
        }
    )


@pytest.fixture
def fake_eopf(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("mapcv.pipeline.open_raster_source", lambda *a, **k: _FakeEopf())
    monkeypatch.setattr(
        "mapcv.targets.segmentation.transform_geometry_to_crs", lambda geometry, crs: geometry
    )


@pytest.mark.usefixtures("fake_eopf")
def test_eopf_float32_geotiffs_match_the_npy_dataset(tmp_path: Path) -> None:
    npy = run_generate(
        _eopf_config(tmp_path, "npy", image_format="npy", mask_format="npy")
    ).manifest
    manifest = run_generate(
        _eopf_config(tmp_path, "tif", image_format="tif", mask_format="tif")
    ).manifest
    staging = tmp_path / "tif"
    assert manifest.source.patch_shape == [4, 16, 16] == npy.source.patch_shape
    assert manifest.source.crs == UTM
    assert len(manifest.patches) == len(npy.patches) > 4
    assert any(entry["padded"] for entry in manifest.patches)

    saw_nan = False
    for entry, npy_entry in zip(manifest.patches, npy.patches):
        image = _check_geotiff(
            staging / entry["files"]["image"],
            manifest,
            entry,
            count=4,
            dtype="float32",
            nodata=float("nan"),
        )
        mask = _check_geotiff(
            staging / entry["files"]["mask"], manifest, entry, count=1, dtype="uint8", nodata=255
        )
        expected_image = np.load(tmp_path / "npy" / npy_entry["files"]["image"])
        expected_mask = np.load(tmp_path / "npy" / npy_entry["files"]["mask"])
        np.testing.assert_array_equal(image, expected_image)  # NaN positions included
        np.testing.assert_array_equal(mask[0], expected_mask)
        assert expected_mask.shape == (16, 16)  # NPY masks are (H, W)
        assert entry["summary"] == npy_entry["summary"]
        saw_nan = saw_nan or bool(np.isnan(image).any())
        with rasterio.open(staging / entry["files"]["image"]) as src:
            assert src.descriptions == ("b04", "b03", "b02", "b08")
            assert src.profile["compress"].lower() == "deflate"
    assert saw_nan


@pytest.mark.usefixtures("fake_eopf")
def test_a_tif_image_with_an_npy_mask_and_the_other_way_round(tmp_path: Path) -> None:
    mixed = run_generate(
        _eopf_config(tmp_path, "mixed", image_format="tif", mask_format="npy")
    ).manifest
    other = run_generate(
        _eopf_config(tmp_path, "other", image_format="npy", mask_format="tif")
    ).manifest
    assert mixed.patches[0]["files"]["mask"].endswith(".npy")
    assert mixed.patches[0]["files"]["image"].endswith(".tif")
    assert other.patches[0]["files"]["mask"].endswith(".tif")
    for entry in mixed.patches:
        mask = np.load(tmp_path / "mixed" / entry["files"]["mask"])
        assert mask.dtype == np.uint8 and mask.shape == (16, 16)
        assert set(map(str, np.unique(mask))) == set(entry["summary"]["class_pixels"])


# ── footprints: patches.geojson ──────────────────────────────────────────────


def _footprints(staging: Path) -> Dict[str, Any]:
    data: Dict[str, Any] = json.loads((staging / "patches.geojson").read_text(encoding="utf-8"))
    return data


@pytest.mark.usefixtures("fake_eopf")
def test_patches_geojson_matches_the_patch_bounds_and_splits(tmp_path: Path) -> None:
    result = run_generate(_eopf_config(tmp_path, "d", image_format="npy"))
    manifest, staging = result.manifest, tmp_path / "d"
    collection = _footprints(staging)
    assert collection["type"] == "FeatureCollection"
    features = collection["features"]
    assert len(features) == len(manifest.patches)

    splits = {
        name: set((staging / "splits" / f"{name}.txt").read_text().split())
        for name in ("train", "val", "test")
    }
    to_lonlat = pyproj.Transformer.from_crs(UTM, "EPSG:4326", always_xy=True)
    seen = set()
    for feature, entry in zip(features, manifest.patches):
        left, bottom, right, top = manifest.patch_bounds(entry)
        corners = [(left, bottom), (right, bottom), (right, top), (left, top), (left, bottom)]
        expected = [to_lonlat.transform(x, y) for x, y in corners]
        ring = feature["geometry"]["coordinates"][0]
        assert feature["geometry"]["type"] == "Polygon"
        np.testing.assert_allclose(ring, expected, rtol=0, atol=1e-8)
        props = feature["properties"]
        name = manifest.patch_name(entry)
        assert props["filename"] == name
        assert (props["row"], props["col"], props["padded"]) == (
            entry["row"],
            entry["col"],
            entry["padded"],
        )
        assert props["class_pixels"] == entry["summary"]["class_pixels"]
        assert props["empty_ratio"] == entry["summary"]["empty_ratio"]
        in_splits = [split for split, names in splits.items() if name in names]
        assert props["split"] == (in_splits[0] if in_splits else None)
        seen.add(props["split"])
    assert {"train", "val", "test"} <= seen
    # Counter-clockwise exterior ring, as GeoJSON (RFC 7946) asks.
    ring = np.array(features[0]["geometry"]["coordinates"][0])
    area = 0.5 * np.sum(ring[:-1, 0] * ring[1:, 1] - ring[1:, 0] * ring[:-1, 1])
    assert area > 0


def test_patches_geojson_of_xyz_uses_web_mercator_inverse(tmp_path: Path, tile_port: int) -> None:
    manifest = run_generate(_xyz_config(tmp_path, "x", tile_port)).manifest
    to_lonlat = pyproj.Transformer.from_crs("EPSG:3857", "EPSG:4326", always_xy=True)
    for feature, entry in zip(_footprints(tmp_path / "x")["features"], manifest.patches):
        left, bottom, right, top = manifest.patch_bounds(entry)
        expected = [
            to_lonlat.transform(x, y)
            for x, y in [(left, bottom), (right, bottom), (right, top), (left, top), (left, bottom)]
        ]
        np.testing.assert_allclose(
            feature["geometry"]["coordinates"][0], expected, rtol=0, atol=1e-8
        )
    # The fixture config has a split, so every kept patch is assigned or dropped.
    splits = {f["properties"]["split"] for f in _footprints(tmp_path / "x")["features"]}
    assert splits <= {"train", "val", "test", None} and splits - {None}


@pytest.mark.usefixtures("fake_eopf")
def test_a_new_split_refreshes_patches_geojson(tmp_path: Path) -> None:
    config = _eopf_config(tmp_path, "d", image_format="npy")
    run_generate(config)
    before = {
        f["properties"]["filename"]: f["properties"]["split"]
        for f in _footprints(tmp_path / "d")["features"]
    }
    counts = run_split(tmp_path / "d", SplitterConfig(seed=99, strategy="random"))
    after = {
        f["properties"]["filename"]: f["properties"]["split"]
        for f in _footprints(tmp_path / "d")["features"]
    }
    assert before != after
    assert sum(1 for split in after.values() if split == "test") == counts["test"]


@pytest.mark.usefixtures("fake_eopf")
def test_footprints_can_be_turned_off(tmp_path: Path) -> None:
    run_generate(_eopf_config(tmp_path, "d", image_format="npy", footprints=False))
    assert not (tmp_path / "d" / "patches.geojson").exists()


def test_footprints_of_a_manifest_without_pyproj_crs_warn_not_fail(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    manifest = _manifest("EPSG:32633", UTM_TRANSFORM, ["a"], 8)
    config = WriterConfig(staging_dir=tmp_path, image_format="npy")
    patches = np.zeros((1, 8, 8, 1), dtype=np.float32)
    write_patches(patches, None, [_patch_meta(0, 0)], config, manifest)
    monkeypatch.setitem(sys.modules, "pyproj", None)
    with pytest.warns(UserWarning, match="patches.geojson was not written"):
        FilesWriter(config).finalize(manifest, None)
    assert not (tmp_path / "patches.geojson").exists()


# ── mask dtypes and formats ──────────────────────────────────────────────────

MASK_DTYPES = ["uint8", "int8", "uint16", "int16", "uint32", "int32", "float32", "float64"]


def _mask_values(dtype: str, count: int = 3, size: int = 8) -> npt.NDArray[Any]:
    rng = np.random.default_rng(3)
    if np.dtype(dtype).kind == "f":
        return rng.normal(size=(count, size, size)).astype(dtype)
    info = np.iinfo(dtype)
    top = min(int(info.max), 70_000)
    return rng.integers(max(int(info.min), -100), top, size=(count, size, size)).astype(dtype)


@pytest.mark.parametrize("dtype", MASK_DTYPES)
def test_masks_round_trip_as_npy_and_tif(tmp_path: Path, dtype: str) -> None:
    masks = _mask_values(dtype)
    meta = [_patch_meta(0, 0), _patch_meta(0, 8), _patch_meta(8, 0)]
    images = np.zeros((3, 8, 8, 3), dtype=np.uint8)
    for fmt in ("npy", "tif"):
        manifest = _manifest("EPSG:3857", (2.0, 0.0, 100.0, 0.0, -2.0, 900.0), ["r", "g", "b"], 8)
        config = WriterConfig(staging_dir=tmp_path / fmt, mask_format=fmt)
        write_patches(images, masks, meta, config, manifest)
        for entry, mask in zip(manifest.patches, masks):
            path = tmp_path / fmt / entry["files"]["mask"]
            assert path.suffix == f".{fmt}"
            if fmt == "npy":
                loaded = np.load(path)
                assert loaded.dtype == mask.dtype
                np.testing.assert_array_equal(loaded, mask)
            else:
                pixels = _check_geotiff(path, manifest, entry, count=1, dtype=dtype, nodata=None)
                np.testing.assert_array_equal(pixels[0], mask)
            if np.dtype(dtype).kind == "f":
                assert "class_pixels" not in entry["summary"]
            else:
                values, counts = np.unique(mask, return_counts=True)
                assert entry["summary"]["class_pixels"] == {
                    str(int(v)): int(c) for v, c in zip(values, counts)
                }


def test_uint16_masks_can_be_png(tmp_path: Path) -> None:
    masks = _mask_values("uint16")
    manifest = _manifest("EPSG:3857", (2.0, 0.0, 100.0, 0.0, -2.0, 900.0), ["r", "g", "b"], 8)
    config = WriterConfig(staging_dir=tmp_path)
    meta = [_patch_meta(0, 0), _patch_meta(0, 8), _patch_meta(8, 0)]
    write_patches(np.zeros((3, 8, 8, 3), np.uint8), masks, meta, config, manifest)
    for entry, mask in zip(manifest.patches, masks):
        loaded = np.asarray(Image.open(tmp_path / entry["files"]["mask"]))
        assert loaded.dtype == np.uint16
        np.testing.assert_array_equal(loaded, mask)


def test_png_masks_refuse_dtypes_they_cannot_hold(tmp_path: Path) -> None:
    manifest = _manifest("EPSG:3857", (2.0, 0.0, 100.0, 0.0, -2.0, 900.0), ["r", "g", "b"], 8)
    config = WriterConfig(staging_dir=tmp_path)
    with pytest.raises(ValueError, match="uint8 or uint16.*not float32"):
        write_patches(
            np.zeros((1, 8, 8, 3), np.uint8),
            np.zeros((1, 8, 8), np.float32),
            [_patch_meta(0, 0)],
            config,
            manifest,
        )


def test_tif_mask_nodata_is_the_ignore_index(tmp_path: Path) -> None:
    manifest = _manifest("EPSG:3857", (2.0, 0.0, 100.0, 0.0, -2.0, 900.0), ["r", "g", "b"], 8)
    manifest.target = TargetRecord(type="segmentation", ignore_index=65535, dtype="uint16")
    masks = np.full((1, 8, 8), 65535, dtype=np.uint16)
    masks[0, :4] = 3
    config = WriterConfig(staging_dir=tmp_path, mask_format="tif")
    write_patches(np.zeros((1, 8, 8, 3), np.uint8), masks, [_patch_meta(0, 0)], config, manifest)
    path = tmp_path / manifest.patches[0]["files"]["mask"]
    pixels = _check_geotiff(
        path, manifest, manifest.patches[0], count=1, dtype="uint16", nodata=65535
    )
    np.testing.assert_array_equal(pixels[0], masks[0])
    # An ignore value the mask type cannot hold is an error, not a wrong tag.
    manifest.target.ignore_index = 300
    with pytest.raises(ValueError, match="does not fit a uint8"):
        write_patches(
            np.zeros((1, 8, 8, 3), np.uint8),
            masks.astype(np.uint8),
            [_patch_meta(8, 0)],
            WriterConfig(staging_dir=tmp_path, mask_format="tif"),
            manifest,
        )


# ── the Rust writer itself, with rasterio ────────────────────────────────────


def _write_raw(
    tmp_path: Path,
    array: npt.NDArray[Any],
    transform: Transform,
    epsg: int,
    geographic: bool,
    nodata: Optional[float] = None,
    band_names: Optional[List[str]] = None,
) -> Path:
    array = np.ascontiguousarray(array)
    n, h, w = array.shape[:3]
    c = array.shape[3] if array.ndim == 4 else 1
    write_geotiffs(
        array.reshape(-1).view(np.uint8),
        array.dtype.name,
        (n, h, w, c),
        [transform] * n,
        [f"p{i}.tif" for i in range(n)],
        str(tmp_path),
        epsg,
        geographic,
        nodata,
        band_names,
    )
    return tmp_path / "p0.tif"


@pytest.mark.parametrize(
    "transform",
    [
        (10.0, 0.0, 500000.0, 0.0, -10.0, 4500000.0),
        (0.5, 0.0, -1234.5, 0.0, -0.25, 99.75),
        (8.0, 6.0, 500000.0, -3.0, -9.0, 4500000.0),  # rotated / sheared
        (10.0, 0.0, 500000.0, 0.0, 10.0, 4500000.0),  # south-up
    ],
)
def test_any_affine_transform_reads_back_exactly(tmp_path: Path, transform: Transform) -> None:
    pixels = np.arange(2 * 5 * 7, dtype=np.uint16).reshape(2, 5, 7, 1)
    path = _write_raw(tmp_path, pixels, transform, 32633, False)
    with rasterio.open(path) as src:
        assert tuple(src.transform)[:6] == transform
        assert src.crs.to_epsg() == 32633
        np.testing.assert_array_equal(src.read(1), pixels[0, :, :, 0])


def test_geographic_crs_and_float64(tmp_path: Path) -> None:
    pixels = np.random.default_rng(1).normal(size=(1, 9, 11, 2))
    path = _write_raw(
        tmp_path,
        pixels,
        (0.001, 0.0, 10.0, 0.0, -0.001, 45.0),
        4326,
        True,
        nodata=-9999.5,
        band_names=["a & <b>", "c"],
    )
    with rasterio.open(path) as src:
        assert src.crs.to_epsg() == 4326 and src.crs.is_geographic
        assert src.dtypes == ("float64", "float64")
        assert src.nodata == -9999.5
        assert src.descriptions == ("a & <b>", "c")
        np.testing.assert_array_equal(src.read(), np.moveaxis(pixels[0], -1, 0))


@pytest.mark.parametrize("dtype", ["int8", "uint8", "int16", "uint16", "int32", "uint32", "int64"])
def test_integer_dtypes_and_band_counts_read_back(tmp_path: Path, dtype: str) -> None:
    info = np.iinfo(dtype)
    rng = np.random.default_rng(5)
    pixels = rng.integers(info.min, info.max, size=(1, 13, 17, 5), dtype=dtype, endpoint=True)
    path = _write_raw(tmp_path, pixels, (3.0, 0.0, 10.0, 0.0, -3.0, 20.0), 3857, False)
    with rasterio.open(path) as src:
        assert src.count == 5 and src.dtypes[0] == dtype
        np.testing.assert_array_equal(src.read(), np.moveaxis(pixels[0], -1, 0))


def test_a_big_patch_spans_several_strips(tmp_path: Path) -> None:
    pixels = np.random.default_rng(2).integers(0, 255, size=(1, 700, 800, 3), dtype=np.uint8)
    path = _write_raw(tmp_path, pixels, (1.0, 0.0, 0.0, 0.0, -1.0, 700.0), 3857, False)
    with rasterio.open(path) as src:
        assert src.block_shapes[0][0] < 700  # more than one strip
        np.testing.assert_array_equal(src.read(), np.moveaxis(pixels[0], -1, 0))


def test_non_epsg_crs_is_rejected(tmp_path: Path) -> None:
    manifest = _manifest("PROJ:custom", (2.0, 0.0, 100.0, 0.0, -2.0, 900.0), ["r", "g", "b"], 8)
    config = WriterConfig(staging_dir=tmp_path, image_format="tif")
    with pytest.raises(ValueError, match="EPSG"):
        write_patches(np.zeros((1, 8, 8, 3), np.uint8), None, [_patch_meta(0, 0)], config, manifest)


def test_unsupported_dtype_and_bad_arguments_are_errors(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="float16"):
        write_geotiffs(
            np.zeros(8, np.uint8), "float16", (1, 2, 2, 1), [(1, 0, 0, 0, -1, 0)], ["a.tif"],
            str(tmp_path), 3857, False,
        )  # fmt: skip
    with pytest.raises(RuntimeError, match="pixel buffer"):
        write_geotiffs(
            np.zeros(7, np.uint8), "uint8", (1, 2, 2, 2), [(1, 0, 0, 0, -1, 0)], ["a.tif"],
            str(tmp_path), 3857, False,
        )  # fmt: skip
    with pytest.raises(RuntimeError, match="plain file name"):
        write_geotiffs(
            np.zeros(4, np.uint8), "uint8", (1, 2, 2, 1), [(1, 0, 0, 0, -1, 0)], ["../a.tif"],
            str(tmp_path), 3857, False,
        )  # fmt: skip


# ── world files ──────────────────────────────────────────────────────────────


@pytest.mark.parametrize("image_format", ["png", "jpg"])
def test_world_files_georeference_png_and_jpg_for_gdal(
    tmp_path: Path, tile_port: int, image_format: str
) -> None:
    manifest = run_generate(
        _xyz_config(tmp_path, "w", tile_port, image_format=image_format, world_files=True)
    ).manifest
    assert manifest.writer is not None and manifest.writer["world_files"] is True
    suffix = "pgw" if image_format == "png" else "jgw"
    staging = tmp_path / "w"
    for index, entry in enumerate(manifest.patches):
        assert entry["files"]["image_world"] == f"Images/patch_{index:07d}.{suffix}"
        assert entry["files"]["mask_world"] == f"Masks/patch_{index:07d}.pgw"
        for role in ("image", "mask"):
            with rasterio.open(staging / entry["files"][role]) as src:
                # GDAL reads the world file (it holds no CRS): the transform must match.
                _close(tuple(src.transform)[:6], manifest.patch_transform(entry))
        lines = (staging / entry["files"]["image_world"]).read_text().split("\n")
        assert len(lines) == 7 and lines[6] == ""


def test_world_files_are_off_by_default_and_not_in_the_fingerprint(tmp_path: Path) -> None:
    assert WriterConfig(staging_dir=tmp_path).world_files is False
    assert "world_files" not in FilesWriter(WriterConfig(staging_dir=tmp_path)).fingerprint()
    on = FilesWriter(WriterConfig(staging_dir=tmp_path, world_files=True)).fingerprint()
    assert on["world_files"] is True


def test_world_files_need_a_png_or_jpg(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="world_files"):
        WriterConfig(staging_dir=tmp_path, image_format="tif", mask_format="tif", world_files=True)
    with pytest.raises(ValueError, match="world_files"):
        WriterConfig(staging_dir=tmp_path, image_format="npy", mask_format="npy", world_files=True)
    # Masks alone are enough.
    WriterConfig(staging_dir=tmp_path, image_format="tif", mask_format="png", world_files=True)


# ── resuming ─────────────────────────────────────────────────────────────────


@pytest.mark.usefixtures("fake_eopf")
@pytest.mark.parametrize(
    "first, second",
    [
        ({"image_format": "npy"}, {"image_format": "tif"}),
        ({"image_format": "tif"}, {"image_format": "npy"}),
        ({"image_format": "npy"}, {"image_format": "npy", "mask_format": "npy"}),
        ({"image_format": "npy", "mask_format": "tif"}, {"image_format": "npy"}),
        ({"image_format": "npy"}, {"image_format": "npy", "world_files": True}),
    ],
)
def test_resume_refuses_a_changed_output_format(
    tmp_path: Path, first: Dict[str, Any], second: Dict[str, Any]
) -> None:
    run_generate(_eopf_config(tmp_path, "d", **first))
    with pytest.raises(ManifestMismatchError, match="writer"):
        run_generate(_eopf_config(tmp_path, "d", **second))


@pytest.mark.usefixtures("fake_eopf")
def test_resume_with_the_same_formats_adds_nothing_and_keeps_the_footprints(
    tmp_path: Path,
) -> None:
    config = _eopf_config(tmp_path, "d", image_format="tif", mask_format="tif")
    first = run_generate(config)
    again = run_generate(config)
    assert again.new_patches == 0
    assert len(_footprints(tmp_path / "d")["features"]) == len(first.manifest.patches)
    # Changing only the footprint switch does not change the dataset, so it still resumes.
    run_generate(
        _eopf_config(tmp_path, "d", image_format="tif", mask_format="tif", footprints=False)
    )


@pytest.mark.usefixtures("fake_eopf")
def test_interrupted_geotiff_run_overwrites_orphans_and_resumes(tmp_path: Path) -> None:
    config = _eopf_config(tmp_path, "d", image_format="tif", mask_format="tif")
    full = run_generate(config).manifest
    staging = tmp_path / "d"
    raw = json.loads((staging / "manifest.json").read_text())
    kept = len(full.patches) // 2
    raw["patches"] = raw["patches"][:kept]
    (staging / "manifest.json").write_text(json.dumps(raw))
    orphan = staging / full.patches[kept]["files"]["image"]
    orphan.write_bytes(b"garbage")

    resumed = run_generate(config).manifest
    assert resumed.patches == full.patches
    with rasterio.open(orphan) as src:
        assert src.count == 4


# ── empty_ratio comes from the validity mask, in every format ────────────────


class _BlackIsDataRaster:
    """A GeoTIFF-like uint8 RGB raster: a black block is real data, a stripe has no data."""

    def __init__(self) -> None:
        rng = np.random.default_rng(11)
        self.image = rng.integers(1, 256, size=(32, 32, 3), dtype=np.uint8)
        self.image[:16, :16] = 0  # valid black (water, shadow)
        self.valid = np.ones((32, 32), dtype=np.bool_)
        self.valid[:, 28:] = False  # NoData stripe
        self.image[:, 28:] = 0
        self.metadata = RasterMetadata(
            source_type="geotiff",
            product_id="black.tif",
            width=32,
            height=32,
            bands=["red", "green", "blue"],
            dtype="uint8",
            crs=UTM,
            transform=UTM_TRANSFORM,
            chunk_rows=32,
        )

    def read_window(
        self, row_start: int, row_stop: int, col_start: int, col_stop: int
    ) -> Tuple[npt.NDArray[np.uint8], npt.NDArray[np.bool_]]:
        window = (slice(row_start, row_stop), slice(col_start, col_stop))
        return self.image[window], self.valid[window]

    def close(self) -> None:
        pass


@pytest.mark.parametrize("image_format", ["png", "jpg", "tif", "npy"])
def test_empty_ratio_counts_invalid_pixels_not_black_ones(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, image_format: str
) -> None:
    monkeypatch.setattr("mapcv.pipeline.open_raster_source", lambda *a, **k: _BlackIsDataRaster())
    imagery: Dict[str, Any] = (
        {"type": "eopf_zarr", "path": str(tmp_path / "unused.zarr")}
        if image_format == "npy"
        else {"type": "xyz", "zoom": 18, "url_template": "http://127.0.0.1:1/{z}/{x}/{y}.png"}
    )
    config = MapcvConfig.model_validate(
        {
            "region": {"west": 9.0, "south": 45.0, "east": 9.1, "north": 45.1},
            "imagery": imagery,
            "sampler": {"patch_size": 16, "stride": 16, "edge_strategy": "drop"},
            "writer": {"staging_dir": str(tmp_path / "d"), "image_format": image_format},
        }
    )
    manifest = run_generate(config).manifest
    ratios = {(e["row"], e["col"]): e["summary"]["empty_ratio"] for e in manifest.patches}
    # Patch (0, 0) is all black but fully valid; (0, 16) has a 4-column no-data stripe.
    assert ratios == {(0, 0): 0.0, (0, 16): 4 / 16, (16, 0): 0.0, (16, 16): 4 / 16}


# ── GeoTIFF imagery in, GeoTIFF patches out ──────────────────────────────────


def test_geotiff_imagery_keeps_its_dtype_bands_nodata_and_georeferencing(
    tmp_path: Path,
) -> None:
    from rasterio.transform import Affine

    rng = np.random.default_rng(4)
    data = rng.integers(1, 4000, size=(3, 64, 64), dtype=np.uint16)
    data[:, :20, :20] = 0  # 0 is this file's NoData
    transform = Affine(10.0, 0.0, 500000.0, 0.0, -10.0, 4428000.0)
    source = tmp_path / "scene.tif"
    with rasterio.open(
        source,
        "w",
        driver="GTiff",
        width=64,
        height=64,
        count=3,
        dtype="uint16",
        crs="EPSG:32633",
        transform=transform,
        nodata=0,
    ) as dst:
        dst.write(data)
    to_lonlat = pyproj.Transformer.from_crs(UTM, "EPSG:4326", always_xy=True)
    west, south = to_lonlat.transform(500000 + 5, 4428000 - 640 + 5)
    east, north = to_lonlat.transform(500000 + 635, 4428000 - 5)
    config = MapcvConfig.model_validate(
        {
            "region": {"west": west, "south": south, "east": east, "north": north},
            "imagery": {"type": "geotiff", "path": str(source)},
            "sampler": {"patch_size": 32, "stride": 32, "edge_strategy": "drop"},
            "writer": {"staging_dir": str(tmp_path / "d"), "image_format": "tif"},
        }
    )
    manifest = run_generate(config).manifest
    assert manifest.source.patch_shape == [3, 32, 32]
    assert len(manifest.patches) == 4
    for entry in manifest.patches:
        pixels = _check_geotiff(
            tmp_path / "d" / entry["files"]["image"],
            manifest,
            entry,
            count=3,
            dtype="uint16",
            nodata=0,  # the source's NoData, declared in the patch too
        )
        row, col = entry["row"], entry["col"]
        np.testing.assert_array_equal(pixels, data[:, row : row + 32, col : col + 32])
        with rasterio.open(source) as src:  # the patch sits where its source pixels were
            window = rasterio.windows.Window(col, row, 32, 32)
            _close(tuple(src.window_transform(window))[:6], manifest.patch_transform(entry))
