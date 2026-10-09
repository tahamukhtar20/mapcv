"""``labels.buffer`` on features whose parts lie in different UTM zones."""

from __future__ import annotations

from typing import Any

import pytest
from pyproj import Geod
from shapely.geometry import MultiLineString, MultiPoint, Point

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
