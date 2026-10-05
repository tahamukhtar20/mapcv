"""Tests for the GDAL-free GeoTIFF/COG reader (``mapcv.geotiff``).

Most tests write rasters with rasterio (GDAL) into a temporary directory and
compare the reader's metadata and windows with what rasterio reports for the
same file; they are skipped where rasterio is not installed. The committed
fixtures in ``tests/data/geotiff`` (written by ``generate_geotiff_fixtures.py``)
are checked without rasterio.
"""

from __future__ import annotations

import itertools
import math
import struct
import threading
import zlib
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional, Tuple

import numpy as np
import numpy.typing as npt
import pytest

from mapcv.geotiff import GeoTiff

DATA = Path(__file__).parent / "data" / "geotiff"


# ---- Committed fixtures (no rasterio needed) ---------------------------------


def _u8_pattern(height: int, width: int, bands: int) -> npt.NDArray[np.uint8]:
    r, c, b = np.meshgrid(np.arange(height), np.arange(width), np.arange(bands), indexing="ij")
    return ((b * 37 + r * 3 + c) % 251).astype(np.uint8)


def test_committed_rgb_cog() -> None:
    tif = GeoTiff(DATA / "rgb_u8_deflate_cog_3857.tif")
    info = tif.info
    assert (info.height, info.width, info.count) == (48, 64, 3)
    assert info.dtype == np.uint8
    assert info.epsg == 3857 and tif.epsg == 3857
    assert info.transform == (2.5, 0.0, 1000.0, 0.0, -2.5, 2000.0)
    assert info.raster_type == "area"
    assert info.nodata is None
    assert info.tiled and info.block_size == (16, 16)
    assert info.compression == "Deflate" and info.predictor == 2
    assert info.overviews == ((24, 32),)
    assert info.overview_transform(1) == (5.0, 0.0, 1000.0, 0.0, -5.0, 2000.0)
    data, valid = tif.read_window(0, 48, 0, 64)
    np.testing.assert_array_equal(data, _u8_pattern(48, 64, 3))
    assert valid.all()
    # A window hanging over the bottom-right corner.
    data, valid = tif.read_window(40, 56, 60, 70, bands=[2, 0])
    assert data.shape == (16, 10, 2)
    np.testing.assert_array_equal(valid[:8, :4], True)
    assert not valid[8:].any() and not valid[:, 4:].any()
    np.testing.assert_array_equal(data[:8, :4], _u8_pattern(48, 64, 3)[40:, 60:][:, :, [2, 0]])
    assert (data[~valid] == 0).all()
    ovr, _ = tif.read_window(0, 24, 0, 32, overview=1)
    assert ovr.shape == (24, 32, 3)


def test_committed_float_point_big_endian() -> None:
    tif = GeoTiff(DATA / "f32_point_nan_4326_be.tif")
    info = tif.info
    assert (info.height, info.width, info.count) == (30, 20, 1)
    assert info.dtype == np.float32
    assert info.epsg == 4326
    assert info.raster_type == "point"
    assert info.byte_order == "big"
    assert info.nodata is not None and math.isnan(info.nodata)
    assert info.compression == "LZW" and info.predictor == 3
    assert not info.tiled and info.block_size == (7, 20)
    # Written with a corner transform (10, 50) and 0.5 deg pixels; GDAL stores
    # the tie point at the pixel centre and reads it back shifted.
    assert info.transform == (0.5, 0.0, 10.0, 0.0, -0.5, 50.0)
    data, valid = tif.read_window(-2, 32, -1, 21)
    r, c = np.meshgrid(np.arange(30), np.arange(20), indexing="ij")
    expected = (r + c / 100).astype(np.float32)
    expected[::5, ::3] = np.nan
    np.testing.assert_array_equal(data[2:32, 1:21, 0], expected)
    assert valid[2:32, 1:21].all() and valid.sum() == 600
    assert np.isnan(data[~valid]).all()


def test_committed_int16_planar_rotated() -> None:
    tif = GeoTiff(DATA / "i16_planar_rotated_32633.tif")
    info = tif.info
    assert (info.height, info.width, info.count) == (24, 24, 4)
    assert info.dtype == np.int16
    assert info.epsg == 32633
    assert info.planar
    assert info.compression == "ZSTD"
    assert info.nodata == -9999
    assert info.transform == (10.0, 2.0, 500000.0, 1.5, -10.0, 4500000.0)
    r, c, b = np.meshgrid(np.arange(24), np.arange(24), np.arange(4), indexing="ij")
    expected = (b * 1000 - r * 50 + c).astype(np.int16)
    data, _ = tif.read_window(0, 24, 0, 24)
    np.testing.assert_array_equal(data, expected)
    data, valid = tif.read_window(20, 30, 5, 9, bands=[3, 1, 3])
    np.testing.assert_array_equal(data[:4], expected[20:24, 5:9][:, :, [3, 1, 3]])
    assert (data[4:] == -9999).all() and not valid[4:].any()


def test_errors_are_clear(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        GeoTiff(tmp_path / "missing.tif")
    not_tiff = tmp_path / "x.tif"
    not_tiff.write_bytes(b"\x89PNG\r\n\x1a\n" + b"\0" * 32)
    with pytest.raises(ValueError, match="not a TIFF"):
        GeoTiff(not_tiff)
    tif = GeoTiff(DATA / "rgb_u8_deflate_cog_3857.tif")
    with pytest.raises(ValueError, match="band index 3 is out of range"):
        tif.read_window(0, 1, 0, 1, bands=[3])
    with pytest.raises(ValueError, match="overview 2 does not exist"):
        tif.read_window(0, 1, 0, 1, overview=2)
    with pytest.raises(ValueError, match="empty window"):
        tif.read_window(5, 5, 0, 1)
    with pytest.raises(ValueError, match="credentials"):
        GeoTiff("https://user:secret@example.com/a.tif")
    with pytest.raises(ValueError, match="unsupported URL scheme"):
        GeoTiff("ftp://example.com/a.tif")
    # A window entirely outside the raster is all fill.
    data, valid = tif.read_window(-10, -5, 100, 110)
    assert data.shape == (5, 10, 3) and not valid.any() and (data == 0).all()


def test_unsupported_compression_names_the_codec(tmp_path: Path) -> None:
    data = bytearray((DATA / "rgb_u8_deflate_cog_3857.tif").read_bytes())
    # Patch the Compression tag (259, SHORT, count 1) of every IFD to LERC (34887).
    entry = struct.pack("<HHI", 259, 3, 1) + struct.pack("<H", 8)
    assert data.count(entry) >= 1
    patched = bytes(data).replace(entry, struct.pack("<HHI", 259, 3, 1) + struct.pack("<H", 34887))
    path = tmp_path / "lerc.tif"
    path.write_bytes(patched)
    with pytest.raises(ValueError, match="LERC compression"):
        GeoTiff(path)


# ---- Cross-validation against rasterio ----------------------------------------


@dataclass
class Case:
    """One raster to write with rasterio and read back."""

    name: str
    dtype: str
    count: int
    height: int = 301
    width: int = 257
    profile: Dict[str, Any] = field(default_factory=dict)
    crs: str = "EPSG:32633"
    transform: Tuple[float, ...] = (10.0, 0.0, 500000.0, 0.0, -10.0, 4500000.0)
    nodata: Optional[float] = None
    point: bool = False
    overviews: Tuple[int, ...] = ()
    sparse: bool = False


def _values(case: Case, rng: np.random.Generator) -> npt.NDArray[Any]:
    shape = (case.count, case.height, case.width)
    dtype = np.dtype(case.dtype)
    if case.profile.get("compress") in ("jpeg", "webp"):
        # Smooth content, as in imagery: lossy codecs are compared with a tolerance.
        r, c = np.meshgrid(np.arange(case.height), np.arange(case.width), indexing="ij")
        base = np.stack([(r * (b + 1) + c * 2 + 40 * b) % 256 for b in range(case.count)])
        noise = rng.integers(-8, 9, size=shape)
        return np.clip(base + noise, 0, 255).astype(np.uint8)
    if dtype.kind == "f":
        values = rng.normal(0, 1000, size=shape).astype(dtype)
        values[:, ::7, ::5] = np.nan
        return values
    info = np.iinfo(dtype)
    # Smooth plus noise so predictors and codecs see realistic data, with the
    # extremes of the type present.
    r = np.arange(case.height)[None, :, None]
    c = np.arange(case.width)[None, None, :]
    span = min(int(info.max) - int(info.min), 2**31)
    smooth = (r * 7 + c * 3) % max(span // 4, 1)
    noise = rng.integers(0, max(span // 64, 2), size=shape)
    values = (int(info.min) + (smooth + noise) % span).astype(dtype)
    values[:, 0, 0] = info.min
    values[:, -1, -1] = info.max
    return values


def _source_values(case: Case) -> npt.NDArray[Any]:
    """The (bands, rows, cols) values written for ``case``, the same on every run."""
    return _values(case, np.random.default_rng(zlib.crc32(case.name.encode())))


def _write(case: Case, directory: Path) -> Path:
    rasterio = pytest.importorskip("rasterio")
    from rasterio.crs import CRS
    from rasterio.enums import Resampling
    from rasterio.transform import Affine

    path = directory / f"{case.name}.tif"
    profile: Dict[str, Any] = dict(
        driver="GTiff",
        height=case.height,
        width=case.width,
        count=case.count,
        dtype=case.dtype,
        crs=CRS.from_user_input(case.crs),
        transform=Affine(*case.transform),
        nodata=case.nodata,
    )
    profile.update(case.profile)
    values = _source_values(case)
    with rasterio.open(path, "w", **profile) as dst:
        if case.point:
            dst.update_tags(AREA_OR_POINT="Point")
        if case.sparse:
            # Write only the top-left quarter; with SPARSE_OK the other blocks
            # are never written and have offset 0.
            h, w = case.height // 2, case.width // 2
            dst.write(values[:, :h, :w], window=rasterio.windows.Window(0, 0, w, h))
        else:
            dst.write(values)
        if case.overviews:
            dst.build_overviews(list(case.overviews), Resampling.nearest)
    return path


def _cases() -> List[Case]:
    cases: List[Case] = []
    layouts: List[Tuple[str, Dict[str, Any]]] = [
        ("strip", {}),
        ("strip7", {"blockysize": 7}),
        ("tile16", {"tiled": True, "blockxsize": 16, "blockysize": 16}),
        ("tile64x32", {"tiled": True, "blockxsize": 64, "blockysize": 32}),
        ("tile512", {"tiled": True, "blockxsize": 512, "blockysize": 512}),
    ]
    dtypes = ["uint8", "int8", "uint16", "int16", "uint32", "int32", "float32", "float64"]
    codecs: List[Tuple[str, Dict[str, Any]]] = [
        ("none", {}),
        ("deflate", {"compress": "deflate"}),
        ("lzw", {"compress": "lzw"}),
        ("zstd", {"compress": "zstd"}),
        ("packbits", {"compress": "packbits"}),
    ]
    # Every dtype with every codec, cycling through layouts, band counts,
    # interleaving and predictors.
    for i, (dtype, (codec, codec_profile)) in enumerate(itertools.product(dtypes, codecs)):
        layout, layout_profile = layouts[i % len(layouts)]
        count = (1, 3, 4, 2)[i % 4]
        profile = {**layout_profile, **codec_profile}
        if count > 1 and i % 3 == 0:
            profile["interleave"] = "band"
        if codec in ("deflate", "lzw", "zstd") and i % 2 == 0:
            profile["predictor"] = 3 if dtype.startswith("float") else 2
        name = f"{dtype}_{codec}_{layout}_{count}b_{profile.get('interleave', 'pixel')}"
        name += f"_p{profile.get('predictor', 1)}"
        cases.append(Case(name, dtype, count, profile=profile))
    # Predictors with every layout and interleaving.
    for (layout, layout_profile), interleave, (dtype, predictor) in itertools.product(
        layouts,
        ("pixel", "band"),
        (("uint16", 2), ("int32", 2), ("uint8", 2), ("float32", 3), ("float64", 3)),
    ):
        profile = {
            **layout_profile,
            "compress": "deflate",
            "predictor": predictor,
            "interleave": interleave,
        }
        cases.append(Case(f"pred_{dtype}_{layout}_{interleave}", dtype, 3, profile=profile))
    # 64-bit integers, byte order, BigTIFF, sizes smaller than a block.
    cases += [
        Case("u64_zstd", "uint64", 2, profile={"compress": "zstd", "predictor": 2}),
        Case("i64_tiled", "int64", 1, profile={"tiled": True, "blockxsize": 32, "blockysize": 32}),
        Case(
            "big_endian_u16_lzw_p2",
            "uint16",
            3,
            profile={"compress": "lzw", "predictor": 2, "ENDIANNESS": "BIG"},
        ),
        Case(
            "big_endian_f64_tiled_p3",
            "float64",
            2,
            profile={
                "compress": "deflate",
                "predictor": 3,
                "ENDIANNESS": "BIG",
                "tiled": True,
                "blockxsize": 32,
                "blockysize": 48,
            },
        ),
        Case(
            "big_endian_i16_planar",
            "int16",
            3,
            profile={"ENDIANNESS": "BIG", "interleave": "band"},
        ),
        Case(
            "bigtiff_tiled_zstd",
            "int16",
            2,
            profile={
                "BIGTIFF": "YES",
                "tiled": True,
                "blockxsize": 48,
                "blockysize": 16,
                "compress": "zstd",
            },
        ),
        Case("bigtiff_strip", "float32", 1, profile={"BIGTIFF": "YES", "blockysize": 13}),
        Case("tiny", "uint8", 1, height=3, width=5, profile={"compress": "deflate"}),
        Case(
            "tiny_tiled",
            "int16",
            2,
            height=5,
            width=3,
            profile={"tiled": True, "blockxsize": 16, "blockysize": 16},
        ),
        Case(
            "wide_one_strip",
            "uint8",
            3,
            height=40,
            width=1000,
            profile={"blockysize": 40, "compress": "lzw"},
        ),
        Case(
            "uncompressed_one_strip",
            "uint16",
            2,
            height=120,
            width=90,
            profile={"blockysize": 120},
        ),
    ]
    # Georeferencing: CRSs, PixelIsPoint, rotation, south-up, nodata.
    cases += [
        Case("geo_4326", "uint8", 1, crs="EPSG:4326", transform=(0.01, 0, -10.5, 0, -0.01, 51)),
        Case("geo_3857", "uint8", 1, crs="EPSG:3857", transform=(4.77, 0, 1e6, 0, -4.77, 6e6)),
        Case("geo_point", "uint16", 1, point=True),
        Case(
            "geo_point_rotated",
            "float32",
            1,
            point=True,
            transform=(10.0, 3.0, 500000.0, -2.0, -10.0, 4500000.0),
        ),
        Case("geo_rotated", "int16", 2, transform=(10.0, 2.0, 500000.0, 1.5, -10.0, 4500000.0)),
        Case("geo_south_up", "uint8", 1, transform=(10.0, 0.0, 500000.0, 0.0, 10.0, 4400000.0)),
        Case("geo_fractional", "float64", 1, transform=(0.1, 0, 123.456789, 0, -0.1, 45.678)),
        Case("nodata_u8", "uint8", 3, nodata=255, profile={"compress": "deflate"}),
        Case("nodata_i16", "int16", 1, nodata=-9999, profile={"tiled": True}),
        Case("nodata_f32_nan", "float32", 2, nodata=float("nan"), profile={"compress": "lzw"}),
        Case("nodata_f64_neg", "float64", 1, nodata=-3.4e38),
        Case("nodata_f32_inf", "float32", 1, nodata=float("-inf")),
    ]
    # Overviews and sparse files.
    cases += [
        Case(
            "ovr_tiled",
            "uint8",
            3,
            height=700,
            width=513,
            overviews=(2, 4, 8),
            profile={"tiled": True, "blockxsize": 64, "blockysize": 64, "compress": "deflate"},
        ),
        Case(
            "ovr_strip_planar",
            "int16",
            2,
            overviews=(2, 3),
            profile={"interleave": "band", "compress": "lzw", "predictor": 2},
        ),
        Case(
            "sparse_tiled",
            "uint16",
            2,
            nodata=7,
            sparse=True,
            profile={"tiled": True, "blockxsize": 32, "blockysize": 32, "SPARSE_OK": "TRUE"},
        ),
        Case(
            "sparse_no_nodata",
            "float32",
            1,
            sparse=True,
            profile={"tiled": True, "blockxsize": 32, "blockysize": 32, "SPARSE_OK": "TRUE"},
        ),
    ]
    return cases


CASES = _cases()


@pytest.fixture(scope="module")
def fixture_dir(tmp_path_factory: pytest.TempPathFactory) -> Path:
    return tmp_path_factory.mktemp("geotiff")


def _random_windows(
    height: int, width: int, rng: np.random.Generator, n: int = 12
) -> Iterator[Tuple[int, int, int, int]]:
    yield (0, height, 0, width)
    yield (-3, height + 2, -5, width + 1)  # everything, plus a border
    yield (height - 1, height + 4, width - 1, width + 4)  # the last pixel
    for _ in range(n):
        h = int(rng.integers(1, max(height // 2, 2) + 1))
        w = int(rng.integers(1, max(width // 2, 2) + 1))
        r0 = int(rng.integers(-h + 1, height))
        c0 = int(rng.integers(-w + 1, width))
        yield (r0, r0 + h, c0, c0 + w)


def _fill(dtype: np.dtype[Any], nodata: Optional[float]) -> float:
    if nodata is None:
        return 0.0
    if dtype.kind == "f":
        return float(np.array(nodata).astype(dtype))
    info = np.iinfo(dtype)
    if nodata == int(nodata) and info.min <= nodata <= info.max:
        return nodata
    return 0.0


def _assert_window_matches(
    tif: GeoTiff,
    src: Any,
    window: Tuple[int, int, int, int],
    overview: int = 0,
    bands: Optional[List[int]] = None,
    atol: float = 0,
) -> None:
    from rasterio.windows import Window

    r0, r1, c0, c1 = window
    data, valid = tif.read_window(r0, r1, c0, c1, bands=bands, overview=overview)
    count = len(bands) if bands is not None else src.count
    assert data.shape == (r1 - r0, c1 - c0, count)
    assert data.dtype == np.dtype(src.dtypes[0])
    ra, rb = max(r0, 0), min(r1, src.height)
    ca, cb = max(c0, 0), min(c1, src.width)
    expected_valid = np.zeros((r1 - r0, c1 - c0), dtype=bool)
    if ra < rb and ca < cb:
        expected_valid[ra - r0 : rb - r0, ca - c0 : cb - c0] = True
        indexes = [b + 1 for b in bands] if bands is not None else None
        theirs = src.read(indexes, window=Window(ca, ra, cb - ca, rb - ra)).transpose(1, 2, 0)
        ours = data[ra - r0 : rb - r0, ca - c0 : cb - c0]
        if atol:
            diff = np.abs(ours.astype(np.int32) - theirs.astype(np.int32))
            assert diff.max() <= atol, f"max difference {diff.max()} in {window}"
        else:
            np.testing.assert_array_equal(ours, theirs, err_msg=f"window {window}")
    np.testing.assert_array_equal(valid, expected_valid)
    fill = _fill(data.dtype, src.nodata)
    outside = data[~valid]
    if math.isnan(fill):
        assert np.isnan(outside).all()
    else:
        assert (outside == fill).all()


def _assert_metadata_matches(tif: GeoTiff, src: Any, case: Case) -> None:
    info = tif.info
    assert (info.height, info.width, info.count) == (src.height, src.width, src.count)
    assert info.dtype == np.dtype(src.dtypes[0])
    assert info.epsg == src.crs.to_epsg()
    assert info.transform is not None
    np.testing.assert_allclose(info.transform, tuple(src.transform)[:6], rtol=1e-9, atol=1e-12)
    assert info.raster_type == ("point" if case.point else "area")
    if src.nodata is None:
        assert info.nodata is None
    elif math.isnan(src.nodata):
        assert info.nodata is not None and math.isnan(info.nodata)
    else:
        assert info.nodata == src.nodata
    if info.compression == "none" and not info.tiled and info.block_size[0] == info.height:
        # GDAL presents one uncompressed strip as several smaller virtual blocks.
        assert info.block_size == (src.height, src.width)
    else:
        assert info.block_size == tuple(src.block_shapes[0])
    assert info.tiled == bool(src.profile.get("tiled", False))
    band_interleaved = src.interleaving is not None and src.interleaving.value == "BAND"
    assert info.planar == (src.count > 1 and band_interleaved)
    ovr_factors = src.overviews(1)
    assert len(info.overviews) == len(ovr_factors)


@pytest.mark.parametrize("case", CASES, ids=[c.name for c in CASES])
def test_matches_rasterio(case: Case, fixture_dir: Path) -> None:
    rasterio = pytest.importorskip("rasterio")
    path = _write(case, fixture_dir)
    tif = GeoTiff(path)
    rng = np.random.default_rng(len(case.name))
    with rasterio.open(path) as src:
        _assert_metadata_matches(tif, src, case)
        for window in _random_windows(src.height, src.width, rng):
            _assert_window_matches(tif, src, window)
        if src.count > 1:
            picks = [src.count - 1, 0, src.count - 1]
            _assert_window_matches(tif, src, (5, 60, 3, 90), bands=picks)
            _assert_window_matches(tif, src, (-4, 33, 200, 300), bands=[1])
    for level, (height, width) in enumerate(tif.info.overviews, start=1):
        with rasterio.open(path, overview_level=level - 1) as ovr:
            assert (ovr.height, ovr.width) == (height, width)
            expected = tif.info.overview_transform(level)
            assert expected is not None
            np.testing.assert_allclose(expected, tuple(ovr.transform)[:6], rtol=1e-9)
            for window in _random_windows(height, width, rng, n=4):
                _assert_window_matches(tif, ovr, window, overview=level)


@pytest.mark.parametrize(
    ("photometric", "count", "interleave", "atol"),
    [
        # GDAL decodes with libjpeg-turbo, mapcv with zune-jpeg: the IDCTs
        # round differently, and YCbCr chroma upsampling differs slightly.
        ("minisblack", 1, "pixel", 1),
        ("rgb", 3, "pixel", 1),
        ("ycbcr", 3, "pixel", 4),
        ("minisblack", 3, "band", 1),
    ],
)
def test_jpeg_matches_rasterio_within_tolerance(
    photometric: str, count: int, interleave: str, atol: int, fixture_dir: Path
) -> None:
    rasterio = pytest.importorskip("rasterio")
    profile = {
        "compress": "jpeg",
        "photometric": photometric,
        "interleave": interleave,
        "tiled": True,
        "blockxsize": 64,
        "blockysize": 32,
    }
    case = Case(f"jpeg_{photometric}_{count}_{interleave}", "uint8", count, profile=profile)
    path = _write(case, fixture_dir)
    tif = GeoTiff(path)
    assert tif.info.compression == "JPEG"
    rng = np.random.default_rng(3)
    with rasterio.open(path) as src:
        _assert_metadata_matches(tif, src, case)
        full, _ = tif.read_window(0, src.height, 0, src.width)
        theirs = src.read().transpose(1, 2, 0).astype(np.int32)
        # The decoder-independent check: no further from the encoded image than
        # GDAL's own decode is.
        original = _source_values(case).transpose(1, 2, 0).astype(np.int32)
        ours_error = np.abs(full.astype(np.int32) - original).mean()
        gdal_error = np.abs(theirs - original).mean()
        assert ours_error <= gdal_error + 0.5, (ours_error, gdal_error)
        gdal = tuple(int(v) for v in rasterio.__gdal_version__.split(".")[:2])
        if photometric == "ycbcr" and gdal < (3, 12):
            # The GDAL 3.10 in rasterio 1.4 wheels upsamples YCbCr chroma
            # differently from GDAL 3.12 (up to 90 levels apart on this image);
            # mapcv matches GDAL 3.12 within the tolerance below.
            return
        for window in _random_windows(src.height, src.width, rng):
            _assert_window_matches(tif, src, window, atol=atol)
        mean = np.abs(full.astype(np.int32) - theirs).mean()
        assert mean < 0.5, mean


@pytest.mark.parametrize("count", [3, 4])
def test_webp_matches_rasterio(count: int, fixture_dir: Path) -> None:
    rasterio = pytest.importorskip("rasterio")
    profile = {"compress": "webp", "tiled": True, "blockxsize": 64, "blockysize": 64}
    for lossless in (False, True):
        case = Case(
            f"webp_{count}_{lossless}",
            "uint8",
            count,
            profile={**profile, "WEBP_LOSSLESS": lossless},
        )
        try:
            path = _write(case, fixture_dir)
        except Exception as error:  # GDAL built without WebP
            pytest.skip(f"rasterio cannot write WebP: {error}")
        tif = GeoTiff(path)
        rng = np.random.default_rng(4)
        with rasterio.open(path) as src:
            for window in _random_windows(src.height, src.width, rng):
                _assert_window_matches(tif, src, window)


def test_negative_scale_y_is_read_north_up_like_gdal(fixture_dir: Path) -> None:
    """GDAL reads a negative ModelPixelScale Y as a north-up image too."""
    rasterio = pytest.importorskip("rasterio")
    path = _write(Case("neg_scale_src", "uint8", 1), fixture_dir)
    data = path.read_bytes()
    scale = struct.pack("<3d", 10.0, 10.0, 0.0)
    assert data.count(scale) == 1
    patched = fixture_dir / "neg_scale.tif"
    patched.write_bytes(data.replace(scale, struct.pack("<3d", 10.0, -10.0, 0.0)))
    with rasterio.open(patched) as src:
        expected = tuple(src.transform)[:6]
    assert GeoTiff(patched).info.transform == pytest.approx(expected, rel=1e-12)


def test_user_defined_crs_is_reported(fixture_dir: Path) -> None:
    pytest.importorskip("rasterio")
    crs = "+proj=lcc +lat_1=33 +lat_2=45 +lat_0=39 +lon_0=-96 +x_0=0 +y_0=0 +ellps=GRS80 +units=m"
    path = _write(Case("user_crs", "uint8", 1, crs=crs), fixture_dir)
    tif = GeoTiff(path)
    assert tif.info.epsg is None
    assert tif.info.crs_error is not None and "user-defined" in tif.info.crs_error
    with pytest.raises(ValueError, match="user-defined projected CRS"):
        _ = tif.epsg
    # Pixels and the transform are still available.
    assert tif.info.transform == (10.0, 0.0, 500000.0, 0.0, -10.0, 4500000.0)
    data, _ = tif.read_window(0, 2, 0, 2)
    assert data.shape == (2, 2, 1)


def test_lerc_is_rejected_by_name(fixture_dir: Path) -> None:
    pytest.importorskip("rasterio")
    try:
        path = _write(Case("lerc", "float32", 1, profile={"compress": "lerc"}), fixture_dir)
    except Exception as error:  # GDAL built without LERC
        pytest.skip(f"rasterio cannot write LERC: {error}")
    with pytest.raises(ValueError, match=r"LERC compression \(TIFF compression code 34887\)"):
        GeoTiff(path)


def test_concurrent_reads_share_one_file(fixture_dir: Path) -> None:
    rasterio = pytest.importorskip("rasterio")
    case = Case(
        "threads",
        "uint16",
        3,
        height=512,
        width=512,
        profile={"tiled": True, "blockxsize": 64, "blockysize": 64, "compress": "zstd"},
    )
    path = _write(case, fixture_dir)
    tif = GeoTiff(path)
    with rasterio.open(path) as src:
        expected = src.read().transpose(1, 2, 0)
    errors: List[BaseException] = []

    def worker(seed: int) -> None:
        rng = np.random.default_rng(seed)
        try:
            for _ in range(20):
                r0, c0 = (int(v) for v in rng.integers(0, 400, size=2))
                data, _ = tif.read_window(r0, r0 + 100, c0, c0 + 100)
                np.testing.assert_array_equal(data, expected[r0 : r0 + 100, c0 : c0 + 100])
        except BaseException as error:  # noqa: BLE001 - reported below
            errors.append(error)

    threads = [threading.Thread(target=worker, args=(i,)) for i in range(4)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert not errors, errors[0]


# ---- Remote files over HTTP range requests ------------------------------------


@dataclass
class RangeLog:
    """Ranges requested from the test server."""

    ranges: List[Tuple[int, int]] = field(default_factory=list)

    @property
    def bytes(self) -> int:
        return sum(end - start + 1 for start, end in self.ranges)


def _serve(httpserver: Any, payload: bytes, log: RangeLog, ranges: bool = True) -> str:
    from werkzeug import Request, Response

    def handler(request: Request) -> Response:
        header = request.headers.get("Range")
        if not ranges or header is None:
            return Response(payload, status=200, content_type="image/tiff")
        spec = header.removeprefix("bytes=")
        start_text, end_text = spec.split("-")
        start, end = int(start_text), min(int(end_text), len(payload) - 1)
        log.ranges.append((start, end))
        return Response(
            payload[start : end + 1],
            status=206,
            content_type="image/tiff",
            headers={"Content-Range": f"bytes {start}-{end}/{len(payload)}"},
        )

    httpserver.expect_request("/cog.tif").respond_with_handler(handler)
    return str(httpserver.url_for("/cog.tif"))


def _remote_case() -> Case:
    # 512x512 one-band uint8 in 64x64 uncompressed tiles: 4 KiB per tile, 8
    # tiles (32 KiB) per tile row, so tile rows are 2 cache blocks apart.
    return Case(
        "remote",
        "uint8",
        1,
        height=512,
        width=512,
        overviews=(2, 4),
        profile={"tiled": True, "blockxsize": 64, "blockysize": 64},
    )


def test_remote_reads_fetch_only_intersecting_blocks(httpserver: Any, fixture_dir: Path) -> None:
    rasterio = pytest.importorskip("rasterio")
    path = _write(_remote_case(), fixture_dir)
    payload = path.read_bytes()
    log = RangeLog()
    url = _serve(httpserver, payload, log)
    tif = GeoTiff(url)
    assert tif.info.overviews == ((256, 256), (128, 128))
    opened = len(log.ranges)
    assert log.bytes < len(payload) / 4, "opening fetched too much"

    with rasterio.open(path) as src:
        log.ranges.clear()
        # A window inside tiles (1..2, 1..2): two tile rows, each two adjacent
        # tiles, so one coalesced request per tile row.
        _assert_window_matches(tif, src, (70, 190, 100, 250))
        assert 1 <= len(log.ranges) <= 2, log.ranges
        # 4 tiles of 4 KiB; block alignment adds at most 2 x 16 KiB per run.
        assert log.bytes <= 4 * 4096 + len(log.ranges) * 2 * 16384
        fetched = list(log.ranges)

        # The same window again is served from the block cache.
        log.ranges.clear()
        _assert_window_matches(tif, src, (70, 190, 100, 250))
        assert log.ranges == []

        # A far-away window fetches different bytes only.
        log.ranges.clear()
        _assert_window_matches(tif, src, (450, 512, 450, 512))
        assert len(log.ranges) == 1
        assert all(start > fetched[-1][1] for start, _ in log.ranges)

        # Reading everything is right too (and never refetches cached blocks).
        _assert_window_matches(tif, src, (-1, 513, -1, 513))
    with rasterio.open(path, overview_level=1) as ovr:
        _assert_window_matches(tif, ovr, (0, 128, 0, 128), overview=2)
    assert opened >= 1


def test_remote_server_without_range_support_is_a_clear_error(
    httpserver: Any, fixture_dir: Path
) -> None:
    pytest.importorskip("rasterio")
    payload = _write(_remote_case(), fixture_dir).read_bytes()
    url = _serve(httpserver, payload, RangeLog(), ranges=False)
    with pytest.raises(RuntimeError, match="does not support HTTP range requests"):
        GeoTiff(url)


def test_remote_http_errors(httpserver: Any) -> None:
    httpserver.expect_request("/missing.tif").respond_with_data("nope", status=404)
    with pytest.raises(RuntimeError, match="HTTP 404"):
        GeoTiff(str(httpserver.url_for("/missing.tif")))


def test_remote_committed_fixture_without_rasterio(httpserver: Any) -> None:
    payload = (DATA / "rgb_u8_deflate_cog_3857.tif").read_bytes()
    url = _serve(httpserver, payload, RangeLog())
    remote = GeoTiff(url + "?token=abc")
    local = GeoTiff(DATA / "rgb_u8_deflate_cog_3857.tif")
    assert remote.info == local.info
    for overview in (0, 1):
        a, _ = remote.read_window(-3, 30, 5, 70, overview=overview)
        b, _ = local.read_window(-3, 30, 5, 70, overview=overview)
        np.testing.assert_array_equal(a, b)


def test_fixture_files_are_small() -> None:
    sizes = {p.name: p.stat().st_size for p in DATA.glob("*.tif")}
    assert len(sizes) == 3
    assert all(size < 16 * 1024 for size in sizes.values()), sizes
