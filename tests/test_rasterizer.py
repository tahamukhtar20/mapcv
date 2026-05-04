"""Tests for the Rust scanline rasterizer."""

from __future__ import annotations

from typing import Tuple

import numpy as np
import pytest
from shapely.geometry import MultiPolygon, Point, Polygon

from mapcv import rasterize

Transform = Tuple[float, float, float, float, float, float]

# Identity: pixel (col, row) maps to world (col, row).
IDENTITY: Transform = (1.0, 0.0, 0.0, 0.0, 1.0, 0.0)


def _square(x0: float, y0: float, side: float) -> Polygon:
    return Polygon([(x0, y0), (x0 + side, y0), (x0 + side, y0 + side), (x0, y0 + side), (x0, y0)])


def test_unit_square_at_origin_fills_one_pixel() -> None:
    mask = rasterize([(_square(0, 0, 1), 1)], (4, 4), IDENTITY)
    assert mask.shape == (4, 4)
    assert mask.dtype == np.uint8
    assert mask[0, 0] == 1
    assert int(np.count_nonzero(mask)) == 1


def test_full_image_square_fills_everything() -> None:
    mask = rasterize([(_square(0, 0, 4), 7)], (4, 4), IDENTITY)
    assert (mask == 7).all()


def test_polygon_with_hole() -> None:
    outer = [(0, 0), (10, 0), (10, 10), (0, 10), (0, 0)]
    hole = [(3, 3), (7, 3), (7, 7), (3, 7), (3, 3)]
    poly = Polygon(outer, holes=[hole])
    mask = rasterize([(poly, 1)], (10, 10), IDENTITY)
    # Pixel (5, 5) center is (5.5, 5.5), inside the hole [3, 7) x [3, 7).
    assert mask[5, 5] == 0
    assert mask[0, 0] == 1
    assert mask[1, 1] == 1


def test_multipolygon_fills_both_parts() -> None:
    mp = MultiPolygon([_square(0, 0, 2), _square(4, 4, 2)])
    mask = rasterize([(mp, 3)], (8, 8), IDENTITY)
    assert (mask[0:2, 0:2] == 3).all()
    assert (mask[4:6, 4:6] == 3).all()
    assert mask[3, 3] == 0


def test_non_polygon_geometries_are_skipped() -> None:
    mask = rasterize([(Point(2, 2), 1)], (4, 4), IDENTITY)
    assert (mask == 0).all()


def test_last_writer_wins_replace_semantics() -> None:
    s1 = (_square(0, 0, 4), 1)
    s2 = (_square(2, 2, 2), 2)
    mask = rasterize([s1, s2], (4, 4), IDENTITY)
    assert mask[3, 3] == 2
    assert mask[0, 0] == 1


def test_multiclass_mapping() -> None:
    geoms = [
        (_square(0, 0, 2), 1),
        (_square(2, 0, 2), 2),
        (_square(0, 2, 2), 3),
    ]
    mask = rasterize(geoms, (4, 4), IDENTITY)
    assert (mask[0:2, 0:2] == 1).all()
    assert (mask[0:2, 2:4] == 2).all()
    assert (mask[2:4, 0:2] == 3).all()
    assert (mask[2:4, 2:4] == 0).all()


def test_class_id_zero_is_rejected() -> None:
    with pytest.raises(ValueError, match="class_id"):
        rasterize([(_square(0, 0, 1), 0)], (4, 4), IDENTITY)


def test_class_id_above_255_is_rejected() -> None:
    with pytest.raises(ValueError, match="class_id"):
        rasterize([(_square(0, 0, 1), 256)], (4, 4), IDENTITY)


def test_singular_transform_raises() -> None:
    bad: Transform = (0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
    with pytest.raises(ValueError, match="singular"):
        rasterize([(_square(0, 0, 1), 1)], (4, 4), bad)


def test_zero_dimensions_rejected() -> None:
    with pytest.raises(ValueError, match=">"):
        rasterize([(_square(0, 0, 1), 1)], (0, 4), IDENTITY)


def test_geographic_transform_north_up() -> None:
    """Realistic transform: 1m pixel size, origin at (1000, 2000), north-up."""
    # x = col*1 + 1000, y = -row*1 + 2000  => transform = (1, 0, 1000, 0, -1, 2000)
    transform: Transform = (1.0, 0.0, 1000.0, 0.0, -1.0, 2000.0)
    # World polygon covering pixels [col=0..2, row=0..2]
    # col=0..2 -> x=1000..1002, row=0..2 -> y=2000..1998
    poly = Polygon([(1000, 2000), (1002, 2000), (1002, 1998), (1000, 1998), (1000, 2000)])
    mask = rasterize([(poly, 1)], (4, 4), transform)
    assert (mask[0:2, 0:2] == 1).all()
    assert (mask[2:, :] == 0).all()
    assert (mask[:, 2:] == 0).all()


def test_all_touched_burns_thin_diagonal() -> None:
    # A thin diagonal strip whose center misses some pixels under the
    # default scanline rule but should be filled with all_touched=True.
    poly = Polygon([(0, 0), (4, 4), (4, 4.01), (0, 0.01), (0, 0)])
    base = rasterize([(poly, 1)], (4, 4), IDENTITY, all_touched=False)
    expanded = rasterize([(poly, 1)], (4, 4), IDENTITY, all_touched=True)
    assert int(np.count_nonzero(expanded)) >= int(np.count_nonzero(base))
    assert expanded[0, 0] == 1
    assert expanded[3, 3] == 1


def test_polygon_outside_image_is_a_noop() -> None:
    poly = _square(100, 100, 5)
    mask = rasterize([(poly, 1)], (4, 4), IDENTITY)
    assert (mask == 0).all()


def test_far_outside_polygon_all_touched_is_fast_noop() -> None:
    """all_touched on a polygon at coords ~1e9 must short-circuit, not loop billions of times."""
    import time

    poly = Polygon([(1e9, 1e9), (1e9 + 10, 1e9), (1e9 + 10, 1e9 + 10), (1e9, 1e9 + 10)])
    start = time.perf_counter()
    mask = rasterize([(poly, 1)], (16, 16), IDENTITY, all_touched=True)
    elapsed = time.perf_counter() - start
    assert (mask == 0).all()
    assert elapsed < 0.1, f"all_touched took {elapsed:.3f}s on a far-outside polygon"


def test_open_ring_via_rust_binding_is_auto_closed() -> None:
    """The Rust binding auto-closes rings whose last vertex != first."""
    from mapcv._mapcv_rs import rasterize as _rs_rasterize

    open_ring = [(0.0, 0.0), (4.0, 0.0), (4.0, 4.0), (0.0, 4.0)]
    closed_ring = open_ring + [(0.0, 0.0)]
    mask_open = _rs_rasterize([([open_ring], 1)], 4, 4, IDENTITY, False)
    mask_closed = _rs_rasterize([([closed_ring], 1)], 4, 4, IDENTITY, False)
    np.testing.assert_array_equal(mask_open, mask_closed)
    assert (mask_open == 1).all()


def test_partial_overlap_clipped_to_bounds() -> None:
    poly = _square(2, 2, 10)
    mask = rasterize([(poly, 1)], (4, 4), IDENTITY)
    assert (mask[2:4, 2:4] == 1).all()
    assert (mask[0:2, :] == 0).all()
    assert (mask[:, 0:2] == 0).all()


def test_cross_validate_against_rasterio() -> None:
    """Pixel-exact agreement with rasterio.features.rasterize on a mixed batch."""
    rasterio_features = pytest.importorskip("rasterio.features")
    from rasterio.transform import Affine as RIOAffine

    geoms = [
        _square(1, 1, 4),
        Polygon(
            [(5, 5), (12, 5), (12, 12), (5, 12), (5, 5)],
            holes=[[(7, 7), (10, 7), (10, 10), (7, 10), (7, 7)]],
        ),
        Polygon([(13, 1), (15, 1), (15, 3), (13, 3), (13, 1)]),
    ]
    shape = (16, 16)
    transform: Transform = IDENTITY
    rio_transform = RIOAffine(*transform)

    ours = rasterize([(g, i + 1) for i, g in enumerate(geoms)], shape, transform)
    theirs = rasterio_features.rasterize(
        [(g, i + 1) for i, g in enumerate(geoms)],
        out_shape=shape,
        transform=rio_transform,
        fill=0,
        all_touched=False,
        dtype=np.uint8,
    )
    np.testing.assert_array_equal(ours, theirs)
