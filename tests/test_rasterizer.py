"""Tests for the Rust scanline rasterizer."""

from __future__ import annotations

import itertools
import math
import platform
import random
from typing import NamedTuple

import numpy as np
import pytest
from shapely.geometry import LineString, MultiLineString, MultiPolygon, Point, Polygon, box

from mapcv import rasterize

Transform = tuple[float, float, float, float, float, float]

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


def test_single_pass_geometry_iterable() -> None:
    geometries = ((_square(float(i), 0, 1), i + 1) for i in range(3))

    mask = rasterize(geometries, (2, 4), IDENTITY)

    assert mask[0, :3].tolist() == [1, 2, 3]


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


# ---- GDAL pixel rules (#70, #71, #124) ---------------------------------------
# Expected masks below were checked against rasterio 1.5 / GDAL 3.12.


def test_all_touched_burns_every_pixel_an_edge_crosses() -> None:
    """#70: the edge (0.1, 0.5) -> (3.1, 3.5) crosses pixel (row 1, col 1)."""
    tri = Polygon([(0.1, 0.5), (3.1, 3.5), (0.0, 4.0)])
    mask = rasterize([(tri, 1)], (4, 4), IDENTITY, all_touched=True)
    assert mask.tolist() == [
        [1, 0, 0, 0],
        [1, 1, 0, 0],
        [1, 1, 1, 0],
        [1, 1, 1, 1],
    ]


@pytest.mark.parametrize("eps", [0.0, 1e-13, 2e-11, 1e-6])
@pytest.mark.parametrize("notch, expected", [(0.6, [1, 1, 1, 1]), (0.4, [1, 0, 1, 1])])
def test_near_horizontal_edges_keep_scanline_parity(
    eps: float, notch: float, expected: list[int]
) -> None:
    """#71: an edge with |dy| < 1e-12 across the row-0 centre line still counts."""
    ring = [(0, 0), (4, 0), (4, 1), (2, notch + eps), (1, notch - eps), (0, 1)]
    mask = rasterize([(Polygon(ring), 1)], (1, 4), IDENTITY)
    assert mask[0].tolist() == expected


def test_duplicate_vertices_and_vertices_on_centre_rows_change_nothing() -> None:
    plain = _square(0, 0, 4)
    noisy = Polygon([(0, 0), (0, 0), (4, 0), (4, 1.5), (4, 1.5), (4, 4), (0, 4), (0, 2.5), (0, 0)])
    for all_touched in (False, True):
        a = rasterize([(plain, 1)], (6, 6), IDENTITY, all_touched=all_touched)
        b = rasterize([(noisy, 1)], (6, 6), IDENTITY, all_touched=all_touched)
        np.testing.assert_array_equal(a, b)


def test_pixel_aligned_square_all_touched_burns_interior_only() -> None:
    """#124: GDAL burns 2x2 for the (1,1)-(3,3) square, not 3x3."""
    mask = rasterize([(_square(1, 1, 2), 1)], (5, 5), IDENTITY, all_touched=True)
    expected = np.zeros((5, 5), dtype=np.uint8)
    expected[1:3, 1:3] = 1
    np.testing.assert_array_equal(mask, expected)


def test_pixel_centres_on_edges_follow_gdal() -> None:
    """#124: centres on a left edge are out, on a right edge in.

    Centres on a horizontal edge are in when the edge runs right to left once
    the ring is clockwise in world coordinates, so the outcome depends on
    which way the transform flips y.
    """
    sq = _square(0.5, 0.5, 2)
    assert rasterize([(sq, 1)], (4, 4), IDENTITY).tolist() == [
        [0, 1, 1, 0],
        [0, 1, 1, 0],
        [0, 0, 0, 0],
        [0, 0, 0, 0],
    ]
    north_up: Transform = (1.0, 0.0, 0.0, 0.0, -1.0, 4.0)
    sq_north_up = _square(0.5, 1.5, 2)  # the same pixels under north_up
    assert rasterize([(sq_north_up, 1)], (4, 4), north_up).tolist() == [
        [0, 1, 1, 0],
        [0, 1, 1, 0],
        [0, 1, 1, 0],
        [0, 0, 0, 0],
    ]


@pytest.mark.parametrize("all_touched", [False, True])
def test_hole_with_edges_through_pixel_centres(all_touched: bool) -> None:
    hole = [(2.5, 2.5), (5.5, 2.5), (5.5, 5.5), (2.5, 5.5)]
    poly = Polygon([(0, 0), (8, 0), (8, 8), (0, 8)], [hole])
    mask = rasterize([(poly, 1)], (8, 8), IDENTITY, all_touched=all_touched)
    # Hole centres on its left/top edges stay burned; all_touched also burns
    # column 5, which the hole's right edge passes through.
    last_col = 5 if all_touched else 6
    expected = np.ones((8, 8), dtype=np.uint8)
    expected[3:5, 3:last_col] = 0
    np.testing.assert_array_equal(mask, expected)


def test_tiny_geographic_pixels_are_not_singular() -> None:
    """#71: 5e-7 degree pixels have |det| = 2.5e-13 but are a valid transform."""
    transform: Transform = (5e-7, 0.0, 13.4, 0.0, -5e-7, 52.5)
    x0, y0 = 13.4 + 1e-6, 52.5 - 1e-6
    poly = Polygon([(x0, y0), (x0 + 2e-6, y0), (x0 + 2e-6, y0 - 2e-6), (x0, y0 - 2e-6)])
    mask = rasterize([(poly, 1)], (8, 8), transform)
    expected = np.zeros((8, 8), dtype=np.uint8)
    expected[2:6, 2:6] = 1
    np.testing.assert_array_equal(mask, expected)


# ---- Property-style comparison against rasterio ------------------------------

_EXACT_PARITY_MACHINES = {"x86_64", "AMD64"}

_RIO_TRANSFORMS: list[Transform] = [
    (1.0, 0.0, 0.0, 0.0, 1.0, 0.0),
    (0.5, 0.0, 1024.0, 0.0, -0.25, 4096.0),
    (0.5971642834779395, 0.0, 1113194.9079327357, 0.0, -0.5971642834779395, 6800125.4543973),
    (5e-7, 0.0, 13.404954, 0.0, -5e-7, 52.520008),
    (0.8660254037844387, -0.5, 100.0, 0.5, 0.8660254037844387, -50.0),
]


def _random_pixel_rings(rng: random.Random, w: int, h: int) -> list[list[tuple[float, float]]]:
    """A random polygon in pixel space, biased towards GDAL's tie cases."""
    kind = rng.choice(["star", "sliver", "near_h", "snapped", "hole", "dup"])
    cx, cy = rng.uniform(-3, w + 3), rng.uniform(-3, h + 3)
    r = rng.uniform(0.3, 25)
    angles = sorted(rng.uniform(0, 2 * math.pi) for _ in range(rng.randint(3, 12)))
    pts = [
        (cx + rng.uniform(0.2, 1) * r * math.cos(a), cy + rng.uniform(0.2, 1) * r * math.sin(a))
        for a in angles
    ]
    if kind == "sliver":
        ang = rng.uniform(0, 2 * math.pi)
        length, width = rng.uniform(1, 50), 10 ** rng.uniform(-3, -0.3)
        dx, dy = math.cos(ang), math.sin(ang)
        x1, y1 = cx + dx * length, cy + dy * length
        return [[(cx, cy), (x1, y1), (x1 - dy * width, y1 + dx * width)]]
    if kind == "near_h":
        base = math.floor(cy) + rng.choice([0.5, 0.0])
        eps = [0.0, 1e-13, -1e-13, 2e-11, -1e-6, 1e-3]
        pts = [(x, base + rng.choice(eps)) if rng.random() < 0.5 else (x, y) for x, y in pts]
    elif kind == "snapped":
        half = rng.random() < 0.5
        pts = [(math.floor(x) + 0.5 * half, math.floor(y) + 0.5 * half) for x, y in pts]
    elif kind == "dup":
        pts = [p for p in pts for _ in range(rng.randint(1, 2))]
    elif kind == "hole":
        poly = Polygon(pts).convex_hull
        inner = poly.buffer(-r / 4, join_style=2)
        if isinstance(inner, Polygon) and not inner.is_empty and isinstance(poly, Polygon):
            hole = [(math.floor(x) + 0.5, math.floor(y) + 0.5) for x, y in inner.exterior.coords]
            return [list(poly.exterior.coords), hole]
    return [pts]


# Pixel-space tolerances for the non-x86 path below. The affine transform of the
# largest-magnitude test transform rounds to about 1e-8 px, so 1e-6 px is
# generously above floating-point noise yet far below any real geometric gap.
_TIE_EPS = 1e-6
# A cascade (see _is_tie) follows an edge for its whole length; the longest
# edge here covers 40 px at a slope of 1e-3, so look 1e-2 px away from it.
_CASCADE_REACH = 1e-2
# Share of burned pixels allowed to differ from rasterio where GDAL is built
# with fused multiply-add. The random corpus is deliberately saturated with
# tie cases; the share observed on macOS arm64 was 0.25% (all_touched=False)
# and 0.12% (all_touched=True).
_MAX_TIE_FRACTION = 5e-3


class _PixelShape(NamedTuple):
    value: int
    boundary: MultiLineString
    # Edges with an end within _TIE_EPS of a pixel grid line (integer x or y).
    ambiguous_edges: MultiLineString


def _pixel_shape(rings: list[list[tuple[float, float]]], value: int) -> _PixelShape:
    boundary, ambiguous = [], []
    for ring in rings:
        closed = ring + ring[:1]
        for start, end in itertools.pairwise(closed):
            if start == end:
                continue
            line = LineString([start, end])
            boundary.append(line)
            if any(abs(v - round(v)) <= _TIE_EPS for point in (start, end) for v in point):
                ambiguous.append(line)
    return _PixelShape(value, MultiLineString(boundary), MultiLineString(ambiguous))


def _is_tie(
    shapes: list[_PixelShape], row: int, col: int, values: tuple[int, int], touched: bool
) -> bool:
    """True if floating-point noise can explain why GDAL and mapcv differ at a pixel.

    Only shapes whose value appears at the pixel in either result are considered.
    The pixel is a tie if, for one of them:

    - a polygon edge passes within ``_TIE_EPS`` of the pixel centre, where the
      scanline fill rule is decided; or, with ``all_touched``,
    - an edge merely grazes the pixel square (a corner, or a side) without
      entering it by more than ``_TIE_EPS``; or
    - the pixel lies next to an edge that starts or ends within ``_TIE_EPS`` of
      a pixel grid line: GDAL's test for axis-aligned edges then flips for the
      whole edge, not only at the vertex.
    """
    centre = Point(col + 0.5, row + 0.5)
    square = box(col, row, col + 1, row + 1)
    inset = box(col + _TIE_EPS, row + _TIE_EPS, col + 1 - _TIE_EPS, row + 1 - _TIE_EPS)
    for shape in shapes:
        if shape.value not in values:
            continue
        if shape.boundary.distance(centre) <= _TIE_EPS:
            return True
        if not touched:
            continue
        if not shape.boundary.intersects(inset) and shape.boundary.distance(square) <= _TIE_EPS:
            return True
        if shape.ambiguous_edges.distance(square) <= _CASCADE_REACH:
            return True
    return False


@pytest.mark.parametrize("all_touched", [False, True])
def test_matches_rasterio_on_random_polygons(all_touched: bool) -> None:
    """Agreement with GDAL on random polygons full of tie cases.

    On x86-64 the result must be pixel-exact: mapcv's port of ``llrasterize``
    does the same IEEE-754 arithmetic as the GDAL inside the x86-64 rasterio
    wheels, which are built without fused multiply-add (FMA).

    On other CPUs (the macOS arm64 runners) clang contracts ``a * b + c`` into a
    single FMA by default, so GDAL's affine transform and edge intersections
    round once instead of twice and can land on the other side of a pixel
    centre (or grid line) that sits within ~1e-8 px of an edge. mapcv itself
    never differs between platforms (Rust does not contract floating-point
    operations), so there the test accepts only such tie pixels (see
    ``_is_tie``) and caps their total at ``_MAX_TIE_FRACTION`` of the burned
    pixels. Any other mismatch is a real algorithmic difference and fails.
    """
    rasterio = pytest.importorskip("rasterio")
    rasterio_features = pytest.importorskip("rasterio.features")
    from rasterio.transform import Affine as RIOAffine

    gdal_version = tuple(int(part) for part in rasterio.__gdal_version__.split(".")[:2])
    if all_touched and gdal_version < (3, 11):
        # Before GDAL 3.11 (OSGeo/gdal commit 58d0299, "fix/simplify
        # vertical/horizontal detection"), the all_touched edge walker drew any
        # edge whose ends share a column (or row) as a vertical (horizontal) run.
        # A diagonal edge from (6.0, 12.0) to (6.99999, 14.0) then counts as
        # pixel-aligned and burns nothing, although it crosses pixel (12, 6).
        # mapcv follows the fixed rule.
        pytest.skip(f"all_touched follows GDAL >= 3.11, rasterio has {rasterio.__gdal_version__}")

    exact = platform.machine() in _EXACT_PARITY_MACHINES
    rng = random.Random(70_71_124)
    burned = 0
    ties: list[str] = []
    not_ties: list[str] = []
    for case in range(300):
        transform = rng.choice(_RIO_TRANSFORMS)
        a, b, c, d, e, f = transform
        h, w = rng.randint(1, 40), rng.randint(1, 40)
        shapes = []
        pixel_shapes: list[_PixelShape] = []
        for _ in range(rng.randint(1, 3)):
            pixel_space = _random_pixel_rings(rng, w, h)
            rings = [
                [(a * x + b * y + c, d * x + e * y + f) for x, y in ring] for ring in pixel_space
            ]
            if len(rings[0]) >= 3:
                value = rng.randint(1, 255)
                shapes.append((Polygon(rings[0], rings[1:]), value))
                pixel_shapes.append(_pixel_shape(pixel_space, value))
        if not shapes:
            continue
        ours = rasterize(shapes, (h, w), transform, all_touched=all_touched)
        theirs = rasterio_features.rasterize(
            shapes,
            out_shape=(h, w),
            transform=RIOAffine(*transform),
            fill=0,
            all_touched=all_touched,
            dtype=np.uint8,
        )
        if exact:
            np.testing.assert_array_equal(ours, theirs, err_msg=f"case {case}")
            continue
        burned += int(np.count_nonzero(theirs))
        for row, col in zip(*np.nonzero(ours != theirs)):
            row, col = int(row), int(col)
            values = (int(ours[row, col]), int(theirs[row, col]))
            description = (
                f"case {case} (row {row}, col {col}): mapcv {values[0]}, rasterio {values[1]}"
            )
            if _is_tie(pixel_shapes, row, col, values, all_touched):
                ties.append(description)
            else:
                not_ties.append(description)
    assert not not_ties, f"mismatches that are not floating-point ties: {not_ties}"
    assert len(ties) <= _MAX_TIE_FRACTION * burned, (
        f"{len(ties)} tie pixels differ from rasterio out of {burned} burned "
        f"(limit {_MAX_TIE_FRACTION:.2%}): {ties[:10]}"
    )
