"""``labels.buffer`` on features that lie in several UTM zones."""

from __future__ import annotations

from typing import Any

import numpy as np
import pytest
import shapely
from pyproj import Geod
from shapely.geometry import LineString, MultiLineString, MultiPoint, Point, Polygon

from mapcv.labels import _buffer_in_zone, buffer_metres

GEOD = Geod(ellps="WGS84")


def _half_width_north(polygon: Any, lon: float, lat: float) -> float:
    """Distance in metres, along the geodesic going north from (lon, lat), to where the
    polygon ends (found by bisection; the start must be inside it)."""
    low, high = 0.0, 100.0
    for _ in range(50):
        middle = (low + high) / 2
        x, y, _ = GEOD.fwd(lon, lat, 0.0, middle)
        if polygon.contains(Point(x, y)):
            low = middle
        else:
            high = middle
    return low


def _half_width_east(polygon: Any, lon: float, lat: float) -> float:
    low, high = 0.0, 100.0
    for _ in range(50):
        middle = (low + high) / 2
        x, y, _ = GEOD.fwd(lon, lat, 90.0, middle)
        if polygon.contains(Point(x, y)):
            low = middle
        else:
            high = middle
    return low


def test_each_part_of_a_wide_multiline_is_buffered_in_its_own_zone() -> None:
    near = [(74.30, 31.485), (74.31, 31.485)]
    far = [(154.30, 31.485), (154.31, 31.485)]
    line = MultiLineString([near, far])
    result = buffer_metres(line, 5.0)
    assert len(result.geoms) == 2
    for lon in (74.305, 154.305):
        # A geodesic measurement, not a projection: pyproj's ellipsoidal distance.
        width = _half_width_north(result, lon, 31.485)
        assert width == pytest.approx(5.0, rel=0.003), f"half-width at {lon}"
    # What the whole feature in the zone of its centroid used to give: 18% too narrow.
    old = _buffer_in_zone(line, 5.0)
    assert _half_width_north(old, 154.305, 31.485) < 4.5


def test_each_point_of_a_wide_multipoint_gets_the_same_radius() -> None:
    points = MultiPoint([(74.30, 31.485), (154.30, 31.485), (-100.0, -35.0)])
    result = buffer_metres(points, 5.0)
    assert len(result.geoms) == 3
    for lon, lat in ((74.30, 31.485), (154.30, 31.485), (-100.0, -35.0)):
        # Distance to the edge, east and north, measured on the ellipsoid.
        assert _half_width_east(result, lon, lat) == pytest.approx(5.0, rel=0.003)
        assert _half_width_north(result, lon, lat) == pytest.approx(5.0, rel=0.003)


def test_parts_in_one_zone_are_buffered_as_before() -> None:
    # Same zone for every part: exactly the old single buffer (labels stay byte-identical).
    line = MultiLineString([[(74.30, 31.485), (74.31, 31.485)], [(74.40, 31.50), (74.41, 31.50)]])
    assert buffer_metres(line, 5.0).equals_exact(_buffer_in_zone(line, 5.0), 0.0)
    single = line.geoms[0]
    assert buffer_metres(single, 5.0).equals_exact(_buffer_in_zone(single, 5.0), 0.0)


def _parallel(west: float, east: float, lat: float) -> LineString:
    """A line along a parallel, with a vertex every 0.05 degrees so it follows it."""
    return LineString(
        [(lon, lat) for lon in np.linspace(west, east, round((east - west) / 0.05) + 1)]
    )


@pytest.mark.parametrize("lat", [10.0, -35.0])
def test_a_long_line_is_buffered_in_each_zone_it_crosses(lat: float) -> None:
    # 0 to 30 degrees east: zones 31 to 35. In the zone of its centroid (33) the band was
    # 3% too narrow at the ends.
    line = _parallel(0.0, 30.0, lat)
    result = buffer_metres(line, 5.0)
    assert isinstance(result, Polygon), "the pieces' buffers merge into one band"
    for lon in np.arange(0.25, 30.0, 0.5):
        # Measured on the ellipsoid, across the band, all along the line.
        assert _half_width_north(result, lon, lat) == pytest.approx(5.0, rel=0.003), lon
    old = _buffer_in_zone(line, 5.0)
    assert _half_width_north(old, 0.25, lat) < 4.9


def test_a_line_crossing_one_zone_edge_is_buffered_on_both_sides() -> None:
    # From the centre of zone 32 to the centre of zone 33 (12 degrees east is the edge).
    line = _parallel(9.0, 15.0, 45.0)
    result = buffer_metres(line, 50.0)
    assert isinstance(result, Polygon)
    for lon in (9.0, 11.0, 11.999, 12.0, 12.001, 13.0, 15.0):
        assert _half_width_north(result, lon, 45.0) == pytest.approx(50.0, rel=0.003), lon


def test_a_multiline_with_a_part_across_zones_is_cut_there() -> None:
    near = [(74.30, 31.485), (74.31, 31.485)]
    long = list(_parallel(140.0, 160.0, 31.485).coords)
    result = buffer_metres(MultiLineString([near, long]), 5.0)
    assert len(result.geoms) == 2
    for lon in (74.305, 140.25, 150.0, 159.75):
        assert _half_width_north(result, lon, 31.485) == pytest.approx(5.0, rel=0.003), lon


def test_a_point_is_buffered_in_its_own_zone_even_at_a_zone_edge() -> None:
    # A point is never split: the zone of a point on an edge (6 degrees east) is the one
    # east of it, where the scale at the point is within 0.1% of 1.
    point = Point(6.0, 45.0)
    assert buffer_metres(point, 5.0).equals_exact(_buffer_in_zone(point, 5.0), 0.0)


def test_lines_in_one_zone_and_parts_in_separate_zones_are_buffered_as_before() -> None:
    # Byte for byte what mapcv 0.3.0 gave: one buffer in the line's zone, or one buffer per
    # zone of whole parts, merged.
    line = _parallel(74.0, 77.9, 31.485)
    assert buffer_metres(line, 5.0).equals_exact(_buffer_in_zone(line, 5.0), 0.0)
    near = [(74.30, 31.485), (74.31, 31.485)]
    far = [(154.30, 31.485), (154.31, 31.485)]
    multi = MultiLineString([near, far])
    expected = shapely.union_all(
        [_buffer_in_zone(MultiLineString([part]), 5.0) for part in (near, far)]
    )
    assert buffer_metres(multi, 5.0).equals_exact(expected, 0.0)


@pytest.mark.parametrize(("west", "east"), [(170.0, 180.0), (-180.0, -170.0), (170.0, 190.0)])
def test_a_line_reaching_the_antimeridian_is_buffered_as_before(west: float, east: float) -> None:
    # The buffer of the piece at +/-180 degrees comes back wrapped and cannot be merged with
    # the others: the line is buffered in one zone, as before, rather than raising.
    line = _parallel(west, east, 10.0)
    assert buffer_metres(line, 5.0).equals_exact(_buffer_in_zone(line, 5.0), 0.0)
