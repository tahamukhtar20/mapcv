"""Tests for label parsing (KML + GeoJSON) and CRS transform."""

from __future__ import annotations

import json
import math
import pytest
from shapely.geometry import MultiPolygon, Polygon

from mapcv._mapcv_rs import xy as rust_xy
from mapcv.labels import parse_geojson, parse_kml, transform_to_mercator

_KML_HEADER = b'<?xml version="1.0" encoding="UTF-8"?><kml xmlns="http://www.opengis.net/kml/2.2">'
_KML_FOOTER = b"</kml>"


def _kml(body: bytes) -> bytes:
    return _KML_HEADER + body + _KML_FOOTER


BINARY_KML = _kml(
    b"""
    <Document>
      <Placemark>
        <name>Field A</name>
        <Polygon>
          <outerBoundaryIs><LinearRing>
            <coordinates>10,20 11,20 11,21 10,21 10,20</coordinates>
          </LinearRing></outerBoundaryIs>
        </Polygon>
      </Placemark>
    </Document>
"""
)

MULTICLASS_KML = _kml(
    b"""
    <Document>
      <Placemark>
        <name>P1</name>
        <ExtendedData>
          <Data name="land_use"><value>residential</value></Data>
        </ExtendedData>
        <Polygon>
          <outerBoundaryIs><LinearRing>
            <coordinates>0,0 1,0 1,1 0,1 0,0</coordinates>
          </LinearRing></outerBoundaryIs>
        </Polygon>
      </Placemark>
      <Placemark>
        <name>P2</name>
        <ExtendedData>
          <Data name="land_use"><value>industrial</value></Data>
        </ExtendedData>
        <Polygon>
          <outerBoundaryIs><LinearRing>
            <coordinates>2,0 3,0 3,1 2,1 2,0</coordinates>
          </LinearRing></outerBoundaryIs>
        </Polygon>
      </Placemark>
      <Placemark>
        <name>P3</name>
        <ExtendedData>
          <Data name="land_use"><value>residential</value></Data>
        </ExtendedData>
        <Polygon>
          <outerBoundaryIs><LinearRing>
            <coordinates>4,0 5,0 5,1 4,1 4,0</coordinates>
          </LinearRing></outerBoundaryIs>
        </Polygon>
      </Placemark>
    </Document>
"""
)

NESTED_FOLDER_KML = _kml(
    b"""
    <Document>
      <Folder>
        <name>Zone A</name>
        <Folder>
          <name>Sub-zone A1</name>
          <Placemark>
            <name>Nested</name>
            <Polygon>
              <outerBoundaryIs><LinearRing>
                <coordinates>5,5 6,5 6,6 5,6 5,5</coordinates>
              </LinearRing></outerBoundaryIs>
            </Polygon>
          </Placemark>
        </Folder>
      </Folder>
    </Document>
"""
)

HOLE_KML = _kml(
    b"""
    <Document>
      <Placemark>
        <Polygon>
          <outerBoundaryIs><LinearRing>
            <coordinates>0,0 10,0 10,10 0,10 0,0</coordinates>
          </LinearRing></outerBoundaryIs>
          <innerBoundaryIs><LinearRing>
            <coordinates>2,2 8,2 8,8 2,8 2,2</coordinates>
          </LinearRing></innerBoundaryIs>
        </Polygon>
      </Placemark>
    </Document>
"""
)

MULTIGEOMETRY_KML = _kml(
    b"""
    <Document>
      <Placemark>
        <MultiGeometry>
          <Polygon>
            <outerBoundaryIs><LinearRing>
              <coordinates>0,0 1,0 1,1 0,1 0,0</coordinates>
            </LinearRing></outerBoundaryIs>
          </Polygon>
          <Polygon>
            <outerBoundaryIs><LinearRing>
              <coordinates>2,0 3,0 3,1 2,1 2,0</coordinates>
            </LinearRing></outerBoundaryIs>
          </Polygon>
        </MultiGeometry>
      </Placemark>
    </Document>
"""
)

NO_GEOMETRY_KML = _kml(
    b"""
    <Document>
      <Placemark><name>Empty</name></Placemark>
      <Placemark>
        <Polygon>
          <outerBoundaryIs><LinearRing>
            <coordinates>0,0 1,0 1,1 0,1 0,0</coordinates>
          </LinearRing></outerBoundaryIs>
        </Polygon>
      </Placemark>
    </Document>
"""
)

MISSING_FIELD_KML = _kml(
    b"""
    <Document>
      <Placemark>
        <name>No ExtendedData</name>
        <Polygon>
          <outerBoundaryIs><LinearRing>
            <coordinates>0,0 1,0 1,1 0,1 0,0</coordinates>
          </LinearRing></outerBoundaryIs>
        </Polygon>
      </Placemark>
      <Placemark>
        <name>Wrong field</name>
        <ExtendedData>
          <Data name="other_field"><value>foo</value></Data>
        </ExtendedData>
        <Polygon>
          <outerBoundaryIs><LinearRing>
            <coordinates>2,0 3,0 3,1 2,1 2,0</coordinates>
          </LinearRing></outerBoundaryIs>
        </Polygon>
      </Placemark>
    </Document>
"""
)

POINT_LINE_KML = _kml(
    b"""
    <Document>
      <Placemark><Point><coordinates>1,2,0</coordinates></Point></Placemark>
      <Placemark>
        <LineString>
          <coordinates>0,0 1,1</coordinates>
        </LineString>
      </Placemark>
      <Placemark>
        <Polygon>
          <outerBoundaryIs><LinearRing>
            <coordinates>0,0 1,0 1,1 0,1 0,0</coordinates>
          </LinearRing></outerBoundaryIs>
        </Polygon>
      </Placemark>
    </Document>
"""
)

_SQUARE = [[0, 0], [1, 0], [1, 1], [0, 1], [0, 0]]
_SQUARE2 = [[2, 0], [3, 0], [3, 1], [2, 1], [2, 0]]


def _fc(*features: object) -> bytes:
    return json.dumps({"type": "FeatureCollection", "features": list(features)}).encode()


def _feat(coords: list[list[int]], props: dict[str, object] | None = None) -> dict[str, object]:
    return {
        "type": "Feature",
        "geometry": {"type": "Polygon", "coordinates": [coords]},
        "properties": props or {},
    }


BINARY_GEOJSON = _fc(_feat(_SQUARE))

MULTICLASS_GEOJSON = _fc(
    _feat(_SQUARE, {"category": "residential"}),
    _feat(_SQUARE2, {"category": "industrial"}),
    _feat([[4, 0], [5, 0], [5, 1], [4, 1], [4, 0]], {"category": "residential"}),
)

SINGLE_FEATURE_GEOJSON = json.dumps(
    {
        "type": "Feature",
        "geometry": {"type": "Polygon", "coordinates": [_SQUARE]},
        "properties": {},
    }
).encode()

NULL_GEOMETRY_GEOJSON = _fc(
    {"type": "Feature", "geometry": None, "properties": {}},
    _feat(_SQUARE),
)

MULTIPOLYGON_GEOJSON = _fc(
    {
        "type": "Feature",
        "geometry": {
            "type": "MultiPolygon",
            "coordinates": [[_SQUARE], [_SQUARE2]],
        },
        "properties": {},
    }
)

MISSING_PROP_GEOJSON = _fc(
    _feat(_SQUARE, {}),  # label_field key absent
    _feat(_SQUARE2, {"category": "industrial"}),
)

POINT_GEOJSON = _fc(
    {"type": "Feature", "geometry": {"type": "Point", "coordinates": [0, 0]}, "properties": {}},
    _feat(_SQUARE),
)

def test_parse_kml_binary_mode() -> None:
    geoms, class_map = parse_kml(BINARY_KML)
    assert len(geoms) == 1
    assert class_map == {}
    geom, cid = geoms[0]
    assert cid == 1
    assert geom.geom_type == "Polygon"


def test_parse_kml_multiclass() -> None:
    geoms, class_map = parse_kml(MULTICLASS_KML, label_field="land_use")
    assert len(geoms) == 3
    assert set(class_map.keys()) == {"residential", "industrial"}
    assert class_map["residential"] != class_map["industrial"]
    assert class_map["residential"] == 1
    assert class_map["industrial"] == 2
    assert geoms[2][1] == class_map["residential"]


def test_parse_kml_nested_folders() -> None:
    geoms, _ = parse_kml(NESTED_FOLDER_KML)
    assert len(geoms) == 1
    assert geoms[0][1] == 1


def test_parse_kml_hole() -> None:
    geoms, _ = parse_kml(HOLE_KML)
    assert len(geoms) == 1
    geom, _ = geoms[0]
    assert geom.geom_type == "Polygon"
    assert not geom.exterior.is_empty
    assert len(list(geom.interiors)) == 1


def test_parse_kml_multigeometry() -> None:
    geoms, _ = parse_kml(MULTIGEOMETRY_KML)
    assert len(geoms) == 1
    geom, cid = geoms[0]
    assert geom.geom_type == "MultiPolygon"
    assert cid == 1


def test_parse_kml_skips_empty_placemarks() -> None:
    geoms, _ = parse_kml(NO_GEOMETRY_KML)
    assert len(geoms) == 1  # only the one with a Polygon


def test_parse_kml_skips_missing_label_field() -> None:
    geoms, class_map = parse_kml(MISSING_FIELD_KML, label_field="land_use")
    assert len(geoms) == 0
    assert class_map == {}


def test_parse_kml_skips_non_polygon_geometries() -> None:
    geoms, _ = parse_kml(POINT_LINE_KML)
    assert len(geoms) == 1  # only the Polygon


# ---------------------------------------------------------------------------
# GeoJSON tests
# ---------------------------------------------------------------------------


def test_parse_geojson_binary_mode() -> None:
    geoms, class_map = parse_geojson(BINARY_GEOJSON)
    assert len(geoms) == 1
    assert class_map == {}
    geom, cid = geoms[0]
    assert cid == 1
    assert geom.geom_type == "Polygon"


def test_parse_geojson_multiclass() -> None:
    geoms, class_map = parse_geojson(MULTICLASS_GEOJSON, label_field="category")
    assert len(geoms) == 3
    assert class_map["residential"] == 1
    assert class_map["industrial"] == 2
    assert geoms[2][1] == class_map["residential"]


def test_parse_geojson_single_feature() -> None:
    geoms, _ = parse_geojson(SINGLE_FEATURE_GEOJSON)
    assert len(geoms) == 1


def test_parse_geojson_null_geometry_skipped() -> None:
    geoms, _ = parse_geojson(NULL_GEOMETRY_GEOJSON)
    assert len(geoms) == 1


def test_parse_geojson_multipolygon() -> None:
    geoms, _ = parse_geojson(MULTIPOLYGON_GEOJSON)
    assert len(geoms) == 1
    assert geoms[0][0].geom_type == "MultiPolygon"


def test_parse_geojson_skips_missing_property() -> None:
    geoms, class_map = parse_geojson(MISSING_PROP_GEOJSON, label_field="category")
    assert len(geoms) == 1
    assert "industrial" in class_map


def test_parse_geojson_skips_non_polygon() -> None:
    geoms, _ = parse_geojson(POINT_GEOJSON)
    assert len(geoms) == 1


def test_parse_geojson_invalid_type_raises() -> None:
    data = json.dumps({"type": "GeometryCollection", "geometries": []}).encode()
    with pytest.raises(ValueError, match="Expected FeatureCollection or Feature"):
        parse_geojson(data)


def test_transform_to_mercator_equator() -> None:
    pt = Polygon([(0, 0), (1, 0), (1, 1), (0, 1), (0, 0)])
    projected = transform_to_mercator(pt)
    x, y = projected.exterior.coords[0][:2]
    assert abs(x) < 1e-6
    assert abs(y) < 1e-6


def test_transform_to_mercator_cross_validate_rust_xy() -> None:
    """transform_to_mercator agrees with Rust xy() to < 0.001 m on 1000+ points."""
    import numpy as np
    from shapely.geometry import Point

    rng = np.random.default_rng(42)
    lngs = rng.uniform(-180.0, 180.0, 1000)
    lats = rng.uniform(-85.0, 85.0, 1000)

    for lng, lat in zip(lngs, lats):
        rust_x, rust_y = rust_xy(float(lng), float(lat))
        proj = transform_to_mercator(Point(float(lng), float(lat)))
        py_x, py_y = proj.x, proj.y

        assert abs(py_x - rust_x) < 1e-3, f"x mismatch at ({lng}, {lat})"
        assert abs(py_y - rust_y) < 1e-3, f"y mismatch at ({lng}, {lat})"


def test_transform_to_mercator_geometry() -> None:
    RE = 6_378_137.0
    lng, lat = 45.0, 30.0
    box = Polygon([(lng, lat), (lng + 0.001, lat), (lng + 0.001, lat + 0.001), (lng, lat + 0.001)])
    proj = transform_to_mercator(box)

    expected_x = RE * math.radians(lng)
    expected_y = RE * math.log(math.tan(math.pi / 4 + math.radians(lat) / 2))
    actual_x, actual_y = proj.exterior.coords[0][:2]

    assert abs(actual_x - expected_x) < 1e-3
    assert abs(actual_y - expected_y) < 1e-3


def test_transform_to_mercator_multipolygon() -> None:
    mp = MultiPolygon(
        [
            ([(0, 0), (1, 0), (1, 1), (0, 1)], []),
            ([(2, 2), (3, 2), (3, 3), (2, 3)], []),
        ]
    )
    proj = transform_to_mercator(mp)
    assert proj.geom_type == "MultiPolygon"
    assert proj.is_valid
