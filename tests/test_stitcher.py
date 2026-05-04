"""Tests for Rust stitch_tiles and tile_transform bindings."""

from __future__ import annotations

import io

import numpy as np
import pytest
from PIL import Image

from mapcv._mapcv_rs import PyTileIndex, stitch_tiles, tile_transform


def _make_tile(r: int, g: int, b: int) -> bytes:
    """Return PNG bytes for a solid-colour 256×256 RGB tile."""
    img = Image.fromarray(np.full((256, 256, 3), [r, g, b], dtype=np.uint8), mode="RGB")
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


RED = _make_tile(255, 0, 0)
GREEN = _make_tile(0, 255, 0)
BLUE = _make_tile(0, 0, 255)
WHITE = _make_tile(255, 255, 255)


# ---------------------------------------------------------------------------
# stitch_tiles
# ---------------------------------------------------------------------------


def test_empty_returns_empty() -> None:
    arr, min_x, min_y = stitch_tiles([])
    assert arr.shape == (0, 0, 3)
    assert min_x == 0 and min_y == 0


def test_single_tile_shape() -> None:
    t = PyTileIndex(10, 20, 16)
    arr, min_x, min_y = stitch_tiles([(t, RED)])
    assert arr.shape == (256, 256, 3)
    assert min_x == 10 and min_y == 20


def test_single_tile_colour() -> None:
    t = PyTileIndex(0, 0, 0)
    arr, _, _ = stitch_tiles([(t, RED)])
    assert arr[0, 0].tolist() == [255, 0, 0]
    assert arr[255, 255].tolist() == [255, 0, 0]


def test_two_tiles_horizontal() -> None:
    """Two tiles side-by-side: (tx=5,ty=3) left, (tx=6,ty=3) right."""
    tl = PyTileIndex(5, 3, 16)
    tr = PyTileIndex(6, 3, 16)
    arr, min_x, min_y = stitch_tiles([(tl, RED), (tr, GREEN)])
    assert arr.shape == (256, 512, 3)
    assert min_x == 5 and min_y == 3
    assert arr[128, 0].tolist() == [255, 0, 0]  # left half → red
    assert arr[128, 256].tolist() == [0, 255, 0]  # right half → green


def test_two_tiles_vertical() -> None:
    """Two tiles stacked: top (ty=10), bottom (ty=11)."""
    top = PyTileIndex(0, 10, 16)
    bot = PyTileIndex(0, 11, 16)
    arr, min_x, min_y = stitch_tiles([(top, BLUE), (bot, WHITE)])
    assert arr.shape == (512, 256, 3)
    assert min_x == 0 and min_y == 10
    assert arr[0, 128].tolist() == [0, 0, 255]  # top → blue
    assert arr[256, 128].tolist() == [255, 255, 255]  # bottom → white


def test_2x2_grid() -> None:
    """2×2 tile grid → 512×512 canvas with correct quadrant colours."""
    tl = PyTileIndex(0, 0, 1)
    tr = PyTileIndex(1, 0, 1)
    bl = PyTileIndex(0, 1, 1)
    br = PyTileIndex(1, 1, 1)
    arr, min_x, min_y = stitch_tiles(
        [
            (tl, RED),
            (tr, GREEN),
            (bl, BLUE),
            (br, WHITE),
        ]
    )
    assert arr.shape == (512, 512, 3)
    assert arr[0, 0].tolist() == [255, 0, 0]  # top-left  → red
    assert arr[0, 256].tolist() == [0, 255, 0]  # top-right → green
    assert arr[256, 0].tolist() == [0, 0, 255]  # bot-left  → blue
    assert arr[256, 256].tolist() == [255, 255, 255]  # bot-right → white


def test_order_independent() -> None:
    """Tiles supplied in reverse order should produce the same canvas."""
    tl = PyTileIndex(5, 5, 10)
    tr = PyTileIndex(6, 5, 10)
    arr1, _, _ = stitch_tiles([(tl, RED), (tr, GREEN)])
    arr2, _, _ = stitch_tiles([(tr, GREEN), (tl, RED)])
    np.testing.assert_array_equal(arr1, arr2)


def test_dtype_is_uint8() -> None:
    t = PyTileIndex(0, 0, 0)
    arr, _, _ = stitch_tiles([(t, RED)])
    assert arr.dtype == np.uint8


def test_corrupt_png_raises() -> None:
    t = PyTileIndex(0, 0, 0)
    with pytest.raises(BaseException):
        stitch_tiles([(t, b"not a png")])


# ---------------------------------------------------------------------------
# tile_transform
# ---------------------------------------------------------------------------


def test_tile_transform_returns_tuple_of_six() -> None:
    result = tile_transform(0, 0, 0)
    assert len(result) == 6


def test_tile_transform_zero_zoom() -> None:
    """At zoom 0 there is one 256×256 tile covering the whole Web Mercator plane."""
    a, b, c, d, e, f = tile_transform(0, 0, 0)
    assert b == pytest.approx(0.0)
    assert d == pytest.approx(0.0)
    assert a > 0  # positive pixel width in metres
    assert e < 0  # negative pixel height (y decreases downward)


def test_tile_transform_pixel_size_decreases_with_zoom() -> None:
    """Higher zoom → smaller pixel footprint."""
    a0, *_ = tile_transform(0, 0, 0)
    a16, *_ = tile_transform(0, 0, 16)
    assert a16 < a0


def test_tile_transform_origin_is_northwest() -> None:
    """c and f must be the northwest corner of tile (min_x, min_y)."""
    from mapcv._mapcv_rs import xy_bounds

    b = xy_bounds(3, 5, 10)
    _a, _b, c, _d, _e, f = tile_transform(3, 5, 10)
    assert c == pytest.approx(b.west, rel=1e-9)
    assert f == pytest.approx(b.north, rel=1e-9)
