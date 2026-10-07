import math
from typing import Any

import mercantile
import pytest

from mapcv._mapcv_rs import bounds, snap_bbox, tile, tiles, xy, xy_bounds


def test_xy() -> None:
    lng_lats = [
        (0.0, 0.0),
        (-122.4194, 37.7749),
        (139.6917, 35.6895),
        (-43.1729, -22.9068),
        (179.999, 85.0),
        (-179.999, -85.0),
    ]

    for lng, lat in lng_lats:
        m_x, m_y = mercantile.xy(lng, lat, truncate=False)
        r_x, r_y = xy(lng, lat)
        assert pytest.approx(m_x, abs=1e-5) == r_x
        assert pytest.approx(m_y, abs=1e-5) == r_y


def test_tile() -> None:
    lng_lats = [
        (0.0, 0.0, 0),
        (-122.4194, 37.7749, 14),
        (139.6917, 35.6895, 10),
        (-43.1729, -22.9068, 5),
        (0.0, 85.051129, 2),
        (0.0, -85.051129, 2),
        (200.0, 0.0, 2),
        (-200.0, 0.0, 2),
    ]

    for lng, lat, zoom in lng_lats:
        m_tile = mercantile.tile(lng, lat, zoom, truncate=True)
        r_tile = tile(lng, lat, zoom)

        assert m_tile.x == r_tile.x
        assert m_tile.y == r_tile.y
        assert m_tile.z == r_tile.z

    zoom = 40
    lng, lat = 12.34, 56.78
    m_tile = mercantile.tile(lng, lat, 32, truncate=False)
    r_tile = tile(lng, lat, zoom)
    assert m_tile.x == r_tile.x
    assert m_tile.y == r_tile.y
    assert r_tile.z == 32


def test_xy_bounds() -> None:
    tile_indices = [(0, 0, 0), (2621, 6331, 14), (907, 404, 10)]

    for x, y, z in tile_indices:
        m_bounds = mercantile.xy_bounds(x, y, z)
        r_bounds = xy_bounds(x, y, z)

        assert pytest.approx(m_bounds.left, abs=1e-5) == r_bounds.west
        assert pytest.approx(m_bounds.right, abs=1e-5) == r_bounds.east
        assert pytest.approx(m_bounds.bottom, abs=1e-5) == r_bounds.south
        assert pytest.approx(m_bounds.top, abs=1e-5) == r_bounds.north


def test_tiles() -> None:
    bboxes = [(-122.5, 37.7, -122.4, 37.8), (-0.1, -0.1, 0.1, 0.1), (179.0, 0.0, -179.0, 1.0)]
    zooms = [10, 12, 14]

    for bbox in bboxes:
        for z in zooms:
            m_tiles = list(mercantile.tiles(*bbox, [z]))
            r_tiles = tiles(*bbox, [z])

            assert len(m_tiles) == len(r_tiles)

            m_set = {(t.x, t.y, t.z) for t in m_tiles}
            r_set = {(t.x, t.y, t.z) for t in r_tiles}
            assert m_set == r_set

    tiny = 1e-9
    r_tiles = tiles(0.0, 0.0, tiny, tiny, [40])
    m_tiles = list(mercantile.tiles(0.0, 0.0, tiny, tiny, [32]))
    assert {(t.x, t.y, t.z) for t in r_tiles} == {(t.x, t.y, t.z) for t in m_tiles}


def test_xy_bounds_zoom_clamp() -> None:
    z = 40
    max_index = (1 << 32) - 1
    r_bounds = xy_bounds(max_index, max_index, z)
    m_bounds = mercantile.xy_bounds(max_index, max_index, 32)
    assert pytest.approx(m_bounds.left, abs=1e-5) == r_bounds.west
    assert pytest.approx(m_bounds.right, abs=1e-5) == r_bounds.east
    assert pytest.approx(m_bounds.bottom, abs=1e-5) == r_bounds.south
    assert pytest.approx(m_bounds.top, abs=1e-5) == r_bounds.north


def test_bounds() -> None:
    tile_indices = [(0, 0, 0), (2621, 6331, 14), (907, 404, 10)]
    for x, y, z in tile_indices:
        m_bounds = mercantile.bounds(x, y, z)
        r_bounds = bounds(x, y, z)
        assert pytest.approx(m_bounds.west, abs=1e-6) == r_bounds.west
        assert pytest.approx(m_bounds.east, abs=1e-6) == r_bounds.east
        assert pytest.approx(m_bounds.south, abs=1e-6) == r_bounds.south
        assert pytest.approx(m_bounds.north, abs=1e-6) == r_bounds.north


def test_snap_bbox_matches_tile_bounds() -> None:
    west, south, east, north = -122.42, 37.77, -122.41, 37.78
    zoom = 14
    snapped = snap_bbox(west, south, east, north, zoom)
    snapped_tiles = tiles(snapped.west, snapped.south, snapped.east, snapped.north, [zoom])
    min_x = min(t.x for t in snapped_tiles)
    min_y = min(t.y for t in snapped_tiles)
    max_x = max(t.x for t in snapped_tiles)
    max_y = max(t.y for t in snapped_tiles)
    ul = bounds(min_x, min_y, zoom)
    lr = bounds(max_x, max_y, zoom)
    assert pytest.approx(ul.west, abs=1e-6) == snapped.west
    assert pytest.approx(ul.north, abs=1e-6) == snapped.north
    assert pytest.approx(lr.east, abs=1e-6) == snapped.east
    assert pytest.approx(lr.south, abs=1e-6) == snapped.south


# ---------------------------------------------------------------------------
# Degenerate, inverted and antimeridian boxes (#72)
# ---------------------------------------------------------------------------

_WORLD = (-180.0, -85.051129, 180.0, 85.051129)


def _box(b: Any) -> tuple[float, float, float, float]:
    return (b.west, b.south, b.east, b.north)


@pytest.mark.parametrize(
    "lng, lat", [(0.0, 0.0), (13.4, 52.5), (-122.42, 37.77), (-180.0, 85.0), (180.0, -85.0)]
)
@pytest.mark.parametrize("zoom", [0, 1, 10, 15, 22])
def test_point_snaps_to_the_tile_containing_it(lng: float, lat: float, zoom: int) -> None:
    point_tile = tile(lng, lat, zoom)
    expected = bounds(point_tile.x, point_tile.y, zoom)
    assert _box(snap_bbox(lng, lat, lng, lat, zoom)) == pytest.approx(_box(expected))
    covered = tiles(lng, lat, lng, lat, [zoom])
    assert [(t.x, t.y, t.z) for t in covered] == [(point_tile.x, point_tile.y, zoom)]


@pytest.mark.parametrize(
    "bbox",
    [
        (0.0, 0.0, 0.0, 0.0),  # a point on a tile corner
        (0.0, -1.0, 0.0, 1.0),  # a line on a tile column edge
        (-1.0, 0.0, 1.0, 0.0),  # a line on a tile row edge
        (0.0, 0.0, 1e-13, 1e-13),  # thinner than the corner nudge
    ],
)
@pytest.mark.parametrize("zoom", [1, 10, 15])
def test_degenerate_box_never_snaps_to_the_whole_world(
    bbox: tuple[float, float, float, float], zoom: int
) -> None:
    snapped = _box(snap_bbox(*bbox, zoom))
    assert snapped != pytest.approx(_WORLD)
    covered = tiles(*bbox, [zoom])
    # A point or line covers a single column or row of tiles.
    assert covered
    assert len({t.x for t in covered}) == 1 or len({t.y for t in covered}) == 1
    # The snapped box is exactly the bounds of the covered tiles.
    min_x = min(t.x for t in covered)
    max_y = max(t.y for t in covered)
    assert snapped[0] == pytest.approx(bounds(min_x, 0, zoom).west)
    assert snapped[1] == pytest.approx(bounds(0, max_y, zoom).south)


def test_degenerate_box_tiles_agree_with_mercantile_when_it_finds_any() -> None:
    # Off tile edges mercantile also returns the containing tile; on an edge it
    # returns nothing, where mapcv returns the tile `tile()` assigns the edge to.
    for bbox in [(13.4, 52.5, 13.4, 52.5), (13.4, 52.0, 13.4, 53.0), (13.0, 52.5, 14.0, 52.5)]:
        m_set = {(t.x, t.y, t.z) for t in mercantile.tiles(*bbox, [12])}
        r_set = {(t.x, t.y, t.z) for t in tiles(*bbox, [12])}
        assert m_set and r_set == m_set
    assert not list(mercantile.tiles(0.0, 0.0, 0.0, 0.0, [12]))
    assert [(t.x, t.y) for t in tiles(0.0, 0.0, 0.0, 0.0, [12])] == [(2048, 2048)]


@pytest.mark.parametrize("func", [snap_bbox, lambda w, s, e, n, z: tiles(w, s, e, n, [z])])
def test_inverted_latitudes_are_rejected(func: Any) -> None:
    with pytest.raises(ValueError, match=r"south \(10\) must not be greater than north \(5\)"):
        func(0.0, 10.0, 1.0, 5.0, 12)


@pytest.mark.parametrize(
    "bbox",
    [
        (math.nan, 0.0, 1.0, 1.0),
        (0.0, math.nan, 1.0, 1.0),
        (0.0, 0.0, math.nan, 1.0),
        (0.0, 0.0, 1.0, math.nan),
    ],
)
def test_nan_coordinates_are_rejected(bbox: tuple[float, float, float, float]) -> None:
    with pytest.raises(ValueError, match="must be numbers"):
        snap_bbox(*bbox, 10)
    with pytest.raises(ValueError, match="must be numbers"):
        tiles(*bbox, [10])


def test_snap_rejects_antimeridian_box() -> None:
    with pytest.raises(ValueError, match="antimeridian"):
        snap_bbox(179.9, 0.0, -179.9, 0.1, 16)


def test_tiles_splits_antimeridian_box_like_mercantile() -> None:
    bbox = (179.9, 0.0, -179.9, 0.1)
    r_tiles = tiles(*bbox, [16])
    m_set = {(t.x, t.y, t.z) for t in mercantile.tiles(*bbox, [16])}
    assert {(t.x, t.y, t.z) for t in r_tiles} == m_set
    # Two narrow strips at the edges of the world, not its whole width.
    assert len(r_tiles) < 1000
    assert all(t.x < 20 or t.x > 2**16 - 20 for t in r_tiles)


def test_tiles_rejects_huge_covers_before_allocating() -> None:
    with pytest.raises(ValueError, match="more than the limit"):
        tiles(*_WORLD, [15])
    with pytest.raises(ValueError, match="more than the limit"):
        tiles(*_WORLD, [32])
    # The limit counts every zoom together.
    with pytest.raises(ValueError, match="more than the limit"):
        tiles(*_WORLD, [12, 12])
    assert len(tiles(*_WORLD, [2])) == 16


def test_snap_has_no_tile_count_limit() -> None:
    assert _box(snap_bbox(*_WORLD, 32)) == pytest.approx(_WORLD)


def test_out_of_range_coordinates_are_still_clamped_like_mercantile() -> None:
    for bbox in [(-200.0, -90.0, 200.0, 90.0), (-math.inf, -89.0, math.inf, 89.0)]:
        m_set = {(t.x, t.y, t.z) for t in mercantile.tiles(*bbox, [3])}
        assert {(t.x, t.y, t.z) for t in tiles(*bbox, [3])} == m_set
