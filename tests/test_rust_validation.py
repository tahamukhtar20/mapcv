"""Bad input to the Rust bindings raises ordinary exceptions, never PanicException.

A Rust panic reaches Python as ``pyo3_runtime.PanicException``, a
``BaseException`` that ``except Exception`` does not catch, so every check
here uses ``pytest.raises`` with a concrete ``Exception`` subclass: a panic
would escape it and fail the test.
"""

from __future__ import annotations

import io
import math
from pathlib import Path
from typing import Any, List, Optional, Tuple

import numpy as np
import numpy.typing as npt
import pytest
from PIL import Image

from mapcv._mapcv_rs import (
    PyTileIndex,
    bounds,
    fetch_tiles,
    grid_sample_anchors,
    parse_kml_rs,
    random_sample_anchors,
    rasterize,
    stitch_tiles,
    tile,
    tile_transform,
    write_patches_rs,
    xy,
    xy_bounds,
)

U64_MAX = 2**64 - 1
IDENTITY = (1.0, 0.0, 0.0, 0.0, 1.0, 0.0)


# ---------------------------------------------------------------------------
# write_patches_rs
# ---------------------------------------------------------------------------


def _images(shape: Tuple[int, ...], dtype: Any = np.uint8) -> npt.NDArray[Any]:
    return np.zeros(shape, dtype=dtype)


def _meta(n: int) -> List[Tuple[int, int, bool]]:
    return [(0, i, False) for i in range(n)]


def _write(
    tmp_path: Path,
    images: Any,
    masks: Optional[Any] = None,
    meta: Optional[List[Tuple[int, int, bool]]] = None,
    start_idx: int = 0,
    image_format: str = "png",
    jpg_quality: int = 95,
    jpg_subsampling: str = "4:2:0",
) -> Any:
    n = images.shape[0] if hasattr(images, "shape") and images.ndim else 0
    return write_patches_rs(
        images,
        masks,
        _meta(n) if meta is None else meta,
        start_idx,
        0,
        str(tmp_path),
        str(tmp_path),
        image_format,
        jpg_quality,
        jpg_subsampling,
    )


def test_write_patches_valid_input_still_writes(tmp_path: Path) -> None:
    results = _write(tmp_path, _images((2, 4, 4, 3)), _images((2, 4, 4)))
    assert [r[0] for r in results] == ["patch_0000000.png", "patch_0000001.png"]


@pytest.mark.parametrize(
    "shape",
    [
        (2, 4, 6, 3),  # non-square
        (2, 6, 4, 3),
        (2, 4, 4, 4),  # RGBA
        (2, 4, 4, 1),  # single band
        (1, 0, 0, 3),  # zero patch size
    ],
)
def test_write_patches_rejects_bad_image_shape(tmp_path: Path, shape: Tuple[int, ...]) -> None:
    with pytest.raises(ValueError, match="image_patches"):
        _write(tmp_path, _images(shape))
    assert not any(tmp_path.iterdir())


@pytest.mark.parametrize("shape", [(2, 4, 4), (4, 4, 3), (1, 2, 4, 4, 3)])
def test_write_patches_rejects_wrong_dimensions(tmp_path: Path, shape: Tuple[int, ...]) -> None:
    with pytest.raises(ValueError, match=r"image_patches must be a uint8 array shaped"):
        _write(tmp_path, _images(shape), meta=_meta(shape[0]))


@pytest.mark.parametrize("dtype", [np.float32, np.int64, np.uint16, np.bool_])
def test_write_patches_rejects_wrong_dtype(tmp_path: Path, dtype: Any) -> None:
    with pytest.raises(ValueError, match="uint8"):
        _write(tmp_path, _images((2, 4, 4, 3), dtype))


def test_write_patches_rejects_non_array(tmp_path: Path) -> None:
    with pytest.raises(TypeError, match="numpy array"):
        write_patches_rs([[1, 2, 3]], None, _meta(1), 0, 0, str(tmp_path), str(tmp_path))


@pytest.mark.parametrize("n_meta", [0, 1, 3])
def test_write_patches_rejects_meta_length_mismatch(tmp_path: Path, n_meta: int) -> None:
    with pytest.raises(ValueError, match=f"meta has {n_meta} entries but image_patches holds 2"):
        _write(tmp_path, _images((2, 4, 4, 3)), meta=_meta(n_meta))
    assert not any(tmp_path.iterdir())


@pytest.mark.parametrize(
    "mask_shape",
    [
        (1, 4, 4),  # fewer patches than images
        (3, 4, 4),  # more patches than images
        (2, 4, 5),  # wrong patch size
        (2, 8, 8),
    ],
)
def test_write_patches_rejects_mask_shape_mismatch(
    tmp_path: Path, mask_shape: Tuple[int, ...]
) -> None:
    with pytest.raises(ValueError, match=r"mask_patches must be shaped \(2, 4, 4\)"):
        _write(tmp_path, _images((2, 4, 4, 3)), _images(mask_shape))


@pytest.mark.parametrize(
    "mask",
    [np.zeros((2, 4, 4, 1), np.uint8), np.zeros((2, 4, 4), np.int32), np.zeros(32, np.uint8)],
)
def test_write_patches_rejects_bad_mask_array(tmp_path: Path, mask: npt.NDArray[Any]) -> None:
    with pytest.raises(ValueError, match="mask_patches must be a uint8 array"):
        _write(tmp_path, _images((2, 4, 4, 3)), mask)


def test_write_patches_rejects_non_contiguous_arrays(tmp_path: Path) -> None:
    strided = _images((4, 4, 4, 3))[::2]
    fortran = np.asfortranarray(_images((2, 4, 4, 3)))
    for images in (strided, fortran):
        with pytest.raises(ValueError, match="C-contiguous"):
            _write(tmp_path, images)
    with pytest.raises(ValueError, match="mask_patches must be C-contiguous"):
        _write(tmp_path, _images((2, 4, 4, 3)), _images((2, 4, 8))[:, :, ::2])


@pytest.mark.parametrize("image_format", ["jpeg", "PNG", "tif", ""])
def test_write_patches_rejects_unknown_format(tmp_path: Path, image_format: str) -> None:
    with pytest.raises(ValueError, match="image_format must be 'png' or 'jpg'"):
        _write(tmp_path, _images((1, 4, 4, 3)), image_format=image_format)
    assert not any(tmp_path.iterdir())


@pytest.mark.parametrize("quality", [0, 101, 255])
def test_write_patches_rejects_jpg_quality_out_of_range(tmp_path: Path, quality: int) -> None:
    with pytest.raises(ValueError, match="jpg_quality"):
        _write(tmp_path, _images((1, 4, 4, 3)), image_format="jpg", jpg_quality=quality)


@pytest.mark.parametrize("subsampling", ["4:2:2", "420", "", "4:4:4 "])
def test_write_patches_rejects_unknown_jpg_subsampling(tmp_path: Path, subsampling: str) -> None:
    with pytest.raises(ValueError, match="jpg_subsampling"):
        _write(tmp_path, _images((1, 4, 4, 3)), image_format="jpg", jpg_subsampling=subsampling)
    assert not any(tmp_path.iterdir())


def test_write_patches_rejects_index_overflow(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="overflows"):
        _write(tmp_path, _images((2, 4, 4, 3)), start_idx=U64_MAX)


def test_write_patches_negative_index_is_a_normal_error(tmp_path: Path) -> None:
    # PyO3 rejects negative values for unsigned parameters before Rust runs.
    with pytest.raises(OverflowError):
        _write(tmp_path, _images((1, 4, 4, 3)), start_idx=-1)
    with pytest.raises(OverflowError):
        _write(tmp_path, _images((1, 4, 4, 3)), meta=[(-1, 0, False)])


# ---------------------------------------------------------------------------
# stitch_tiles
# ---------------------------------------------------------------------------


def _png(width: int, height: int) -> bytes:
    buf = io.BytesIO()
    Image.new("RGB", (width, height), (10, 20, 30)).save(buf, format="PNG")
    return buf.getvalue()


@pytest.mark.parametrize("size", [(1, 1), (512, 512), (256, 255), (255, 256)])
def test_stitch_rejects_tiles_that_are_not_256px(size: Tuple[int, int]) -> None:
    with pytest.raises(ValueError, match="every tile must be 256x256"):
        stitch_tiles([(PyTileIndex(0, 0, 1), _png(*size))])


def test_stitch_rejects_odd_tile_among_good_ones() -> None:
    good = _png(256, 256)
    tiles = [(PyTileIndex(0, 0, 2), good), (PyTileIndex(1, 0, 2), _png(1, 1))]
    with pytest.raises(ValueError, match=r"tile 2/1/0 is 1x1 pixels"):
        stitch_tiles(tiles)


def test_stitch_rejects_mixed_zoom_levels() -> None:
    good = _png(256, 256)
    with pytest.raises(ValueError, match="same zoom level"):
        stitch_tiles([(PyTileIndex(0, 0, 1), good), (PyTileIndex(0, 0, 2), good)])


def test_stitch_undecodable_tile_stays_runtime_error() -> None:
    with pytest.raises(RuntimeError, match="decode"):
        stitch_tiles([(PyTileIndex(0, 0, 1), b"\x89PNG\r\n\x1a\nbroken")])


def test_stitch_valid_tile_still_stitches() -> None:
    image, min_x, min_y = stitch_tiles([(PyTileIndex(3, 5, 4), _png(256, 256))])
    assert image.shape == (256, 256, 3)
    assert (min_x, min_y) == (3, 5)
    assert image[0, 0].tolist() == [10, 20, 30]


# ---------------------------------------------------------------------------
# rasterize
# ---------------------------------------------------------------------------


_SQUARE = [[(0.5, 0.5), (2.5, 0.5), (2.5, 2.5), (0.5, 2.5)]]


@pytest.mark.parametrize("height, width", [(0, 4), (4, 0), (0, 0)])
def test_rasterize_rejects_zero_size(height: int, width: int) -> None:
    with pytest.raises(ValueError, match="> 0"):
        rasterize([(_SQUARE, 1)], height, width, IDENTITY)


@pytest.mark.parametrize("height, width", [(2**40, 2**40), (2**62, 2), (2**31, 2**31)])
def test_rasterize_rejects_unallocatable_size(height: int, width: int) -> None:
    with pytest.raises(ValueError, match="overflow|too large"):
        rasterize([], height, width, IDENTITY)


def test_rasterize_rejects_singular_transform() -> None:
    with pytest.raises(ValueError, match="singular"):
        rasterize([(_SQUARE, 1)], 4, 4, (0.0, 0.0, 0.0, 0.0, 0.0, 0.0))


@pytest.mark.parametrize("far", [1e15, 1e19, 1e300, math.inf, -math.inf, math.nan])
@pytest.mark.parametrize("all_touched", [False, True])
def test_rasterize_survives_far_or_non_finite_vertices(far: float, all_touched: bool) -> None:
    rings = [[(0.5, 0.5), (far, 0.5), (far, 2.5), (0.5, 2.5)]]
    mask = rasterize([(rings, 1)], 4, 4, IDENTITY, all_touched)
    assert mask.shape == (4, 4)
    if all_touched and math.isfinite(far):
        assert mask[0].tolist() == [1, 1, 1, 1]


def test_rasterize_negative_size_is_a_normal_error() -> None:
    with pytest.raises(OverflowError):
        rasterize([(_SQUARE, 1)], -1, 4, IDENTITY)


# ---------------------------------------------------------------------------
# grid_sample_anchors / random_sample_anchors
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "args",
    [
        (0, 10, 4, 4),
        (10, 0, 4, 4),
        (10, 10, 0, 4),
        (10, 10, 4, 0),
    ],
)
def test_grid_anchors_reject_zero_sizes(args: Tuple[int, int, int, int]) -> None:
    with pytest.raises(ValueError, match="> 0"):
        grid_sample_anchors(*args)


@pytest.mark.parametrize("args", [(0, 10, 4, 5), (10, 0, 4, 5), (10, 10, 0, 5)])
def test_random_anchors_reject_zero_sizes(args: Tuple[int, int, int, int]) -> None:
    with pytest.raises(ValueError, match="> 0"):
        random_sample_anchors(*args)


@pytest.mark.parametrize("strategy", ["wrap", "PAD", ""])
def test_anchors_reject_unknown_edge_strategy(strategy: str) -> None:
    with pytest.raises(ValueError, match="edge_strategy must be"):
        grid_sample_anchors(10, 10, 4, 4, strategy)
    with pytest.raises(ValueError, match="edge_strategy must be"):
        random_sample_anchors(10, 10, 4, 3, 42, strategy)


def test_anchors_reject_unallocatable_requests() -> None:
    with pytest.raises(ValueError, match="cannot allocate"):
        grid_sample_anchors(2**63, 2**63, 1, 1, "pad")
    # A count beyond the raster's capacity is capped to it (not an allocation request),
    # so only a raster with too many positions to hold can fail.
    assert len(random_sample_anchors(10, 10, 4, U64_MAX, 42, "pad")) == 49
    with pytest.raises(ValueError, match="cannot allocate"):
        random_sample_anchors(2**30, 2**30, 1, U64_MAX, 42, "pad")
    with pytest.raises(ValueError, match="too large"):
        random_sample_anchors(U64_MAX, U64_MAX, 1, 1, 42, "pad")


def test_anchors_with_huge_stride_do_not_overflow() -> None:
    assert grid_sample_anchors(10, 10, 5, U64_MAX, "drop") == [(0, 0)]
    assert grid_sample_anchors(10, 10, 5, U64_MAX, "pad") == [(0, 0)]
    assert grid_sample_anchors(10, 10, 5, U64_MAX, "shift") == [(0, 0), (0, 5), (5, 0), (5, 5)]
    assert grid_sample_anchors(U64_MAX, 1, U64_MAX, U64_MAX, "shift") == [(0, 0)]


def test_anchors_negative_values_are_normal_errors() -> None:
    with pytest.raises(OverflowError):
        grid_sample_anchors(-1, 10, 4, 4)
    with pytest.raises(OverflowError):
        random_sample_anchors(10, 10, 4, -5)


# ---------------------------------------------------------------------------
# fetch_tiles
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("ratio", [-0.1, 1.5, math.nan, math.inf])
def test_fetch_tiles_rejects_ratio_outside_unit_interval(ratio: float) -> None:
    with pytest.raises(ValueError, match="max_failed_ratio"):
        fetch_tiles([], "http://127.0.0.1:9/{z}/{x}/{y}.png", max_failed_ratio=ratio)


def test_fetch_tiles_rejects_zero_connections() -> None:
    with pytest.raises(ValueError, match="max_connections"):
        fetch_tiles([], "http://127.0.0.1:9/{z}/{x}/{y}.png", max_connections=0)


# ---------------------------------------------------------------------------
# Scalar tile helpers and the KML parser
# ---------------------------------------------------------------------------


def test_scalar_tile_helpers_clamp_extreme_values() -> None:
    max_u32 = 2**32 - 1
    assert tile(math.nan, math.nan, 255).z == 32
    assert tile(1e308, -1e308, 0).z == 0
    assert math.isinf(xy(0.0, 90.0)[1])
    assert bounds(max_u32, max_u32, 255).east == pytest.approx(180.0)
    assert xy_bounds(max_u32, max_u32, 0).west == pytest.approx(-20037508.342789244)
    assert len(tile_transform(max_u32, max_u32, 255)) == 6


@pytest.mark.parametrize(
    "data",
    [b"\x00\xff" * 8, b"<kml><Placemark><Polygon>", b"<kml></Placemark>"],
)
def test_parse_kml_rejects_garbage(data: bytes) -> None:
    with pytest.raises(ValueError, match="invalid KML"):
        parse_kml_rs(data)


def test_parse_kml_empty_input_has_no_polygons() -> None:
    assert parse_kml_rs(b"") == ([], 0)
