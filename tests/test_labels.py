"""Tests for label parsing (KML + GeoJSON) and CRS transform."""

from __future__ import annotations

import json
import math
import warnings
from pathlib import Path
from typing import Any

import numpy as np
import numpy.typing as npt
import pytest
import shapely
from shapely.geometry import MultiPolygon, Polygon

from mapcv._mapcv_rs import xy as rust_xy
from mapcv.config import LabelsConfig
from mapcv.labels import (
    assign_class_ids,
    load_vector_labels,
    parse_geojson,
    parse_kml,
    transform_to_mercator,
)

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
    # IDs follow sorted label order, not the order features appear in.
    assert class_map == {"industrial": 1, "residential": 2}
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


def test_parse_kml_rejects_a_label_field_no_placemark_has() -> None:
    # A misspelled field used to build an all-background dataset; GeoPackage and
    # Shapefile already refused it.
    with pytest.raises(ValueError, match="'land_use' is not a property.*'other_field'"):
        parse_kml(MISSING_FIELD_KML, label_field="land_use")


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
    assert class_map == {"industrial": 1, "residential": 2}
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


# ---------------------------------------------------------------------------
# Class IDs, label normalization, skipped features
# ---------------------------------------------------------------------------

_UNIT_SQUARE: list[list[list[int]]] = [[[0, 0], [1, 0], [1, 1], [0, 1], [0, 0]]]


def _collection(features: list[dict[str, Any]], **extra: Any) -> bytes:
    return json.dumps({"type": "FeatureCollection", "features": features, **extra}).encode()


def _feature(value: Any, geometry: dict[str, Any] | None = None) -> dict[str, Any]:
    return {
        "type": "Feature",
        "properties": {"cls": value},
        "geometry": geometry or {"type": "Polygon", "coordinates": _UNIT_SQUARE},
    }


def test_class_ids_do_not_depend_on_feature_order() -> None:
    values = ["water", "road", "building"]
    _, forward = parse_geojson(_collection([_feature(v) for v in values]), "cls")
    _, backward = parse_geojson(_collection([_feature(v) for v in reversed(values)]), "cls")
    assert forward == backward == {"building": 1, "road": 2, "water": 3}


def test_integer_labels_are_used_as_class_ids() -> None:
    geoms, class_map = parse_geojson(
        _collection([_feature(3), _feature(3.0), _feature("7")]), "cls"
    )
    assert class_map == {"3": 3, "7": 7}
    assert [cid for _, cid in geoms] == [3, 3, 7]


def test_explicit_classes_map_and_skips_unmapped_labels() -> None:
    data = _collection([_feature("roof"), _feature("tree"), _feature("road")])
    with pytest.warns(UserWarning, match="1 with a label not in labels.classes"):
        geoms, class_map = parse_geojson(data, "cls", {"roof": 1, "road": 1})
    assert class_map == {"roof": 1, "road": 1}
    assert [cid for _, cid in geoms] == [1, 1]


def test_more_than_255_classes_is_an_error() -> None:
    with pytest.raises(ValueError, match="at most 255 classes"):
        assign_class_ids([f"c{i}" for i in range(256)], "cls")


def test_skipped_features_are_reported() -> None:
    data = _collection(
        [
            _feature("a"),
            _feature(None),
            _feature("b", {"type": "LineString", "coordinates": [[0, 0], [1, 1]]}),
        ]
    )
    with pytest.warns(UserWarning, match="1 without polygon geometry.*1 without a label"):
        geoms, _ = parse_geojson(data, "cls")
    assert len(geoms) == 1


def test_geometry_collection_polygons_are_kept() -> None:
    geometry = {
        "type": "GeometryCollection",
        "geometries": [
            {"type": "Polygon", "coordinates": _UNIT_SQUARE},
            {"type": "Point", "coordinates": [0, 0]},
        ],
    }
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        geoms, _ = parse_geojson(_collection([_feature("a", geometry)]))
    assert geoms[0][0].geom_type == "Polygon"


@pytest.mark.parametrize("name", ["urn:ogc:def:crs:OGC:1.3:CRS84", "EPSG:4326"])
def test_geojson_wgs84_crs_member_is_accepted(name: str) -> None:
    crs = {"type": "name", "properties": {"name": name}}
    geoms, _ = parse_geojson(_collection([_feature("a")], crs=crs))
    assert len(geoms) == 1


def test_geojson_projected_crs_member_is_rejected() -> None:
    crs = {"type": "name", "properties": {"name": "urn:ogc:def:crs:EPSG::3857"}}
    with pytest.raises(ValueError, match="Reproject the file to EPSG:4326"):
        parse_geojson(_collection([_feature("a")], crs=crs))


def test_kml_simple_data_labels_are_read() -> None:
    body = (
        b"<Document><Placemark><ExtendedData><SchemaData schemaUrl='#s'>"
        b"<SimpleData name='kind'>roof</SimpleData></SchemaData></ExtendedData>"
        b"<Polygon><outerBoundaryIs><LinearRing><coordinates>0,0 1,0 1,1 0,0"
        b"</coordinates></LinearRing></outerBoundaryIs></Polygon></Placemark></Document>"
    )
    geoms, class_map = parse_kml(_kml(body), label_field="kind")
    assert class_map == {"roof": 1}
    assert len(geoms) == 1


def test_truncated_kml_is_an_error() -> None:
    with pytest.raises(ValueError, match="invalid KML"):
        parse_kml(_kml(b"<Document><Placemark><Polygon>")[:-6])


def test_labels_config_normalizes_classes_and_checks_suffix() -> None:
    def config(**fields: Any) -> LabelsConfig:
        return LabelsConfig.model_validate(fields)

    assert config(path="labels.geojson", label_field="cls", classes={3: 1, "roof": 2}).classes == {
        "3": 1,
        "roof": 2,
    }
    with pytest.raises(ValueError, match="1..255"):
        config(path="labels.geojson", label_field="cls", classes={"a": 256})
    with pytest.raises(ValueError, match="requires labels.label_field"):
        config(path="labels.geojson", classes={"a": 1})
    with pytest.raises(ValueError, match="KMZ is a zipped KML"):
        config(path="labels.kmz")
    # The KMZ hint is for .kmz files only.
    with pytest.raises(ValueError, match=r"got 'labels.fgb' \["):
        config(path="labels.fgb")
    with pytest.raises(ValueError, match="unzip it first"):
        config(path="labels.zip")


def test_transform_all_to_mercator_matches_per_geometry() -> None:
    from shapely.geometry import MultiPolygon, Polygon

    from mapcv.labels import transform_all_to_mercator

    geometries = [
        Polygon([(4.9, 52.3), (5.0, 52.3), (5.0, 52.4), (4.9, 52.3)]),
        Polygon(
            [(-10, -10), (10, -10), (10, 10), (-10, 10)],
            holes=[[(-1, -1), (1, -1), (1, 1), (-1, -1)]],
        ),
        MultiPolygon(
            [Polygon([(170, 60), (179, 60), (179, 70)]), Polygon([(0, 0), (1, 0), (1, 1)])]
        ),
    ]
    vectorized = transform_all_to_mercator(geometries)
    assert [g.geom_type for g in vectorized] == [g.geom_type for g in geometries]
    for fast, slow in zip(vectorized, (transform_to_mercator(g) for g in geometries)):
        assert fast.equals_exact(slow, tolerance=0)
    assert transform_all_to_mercator([]) == []


# ---------------------------------------------------------------------------
# Class IDs of numeric and non-ASCII labels
# ---------------------------------------------------------------------------


def test_numeric_labels_are_numbered_by_value_not_as_text() -> None:
    # 0 is not a class ID, so all labels are numbered in sorted order: numbers by value
    # (as text, "10" sorted before "2" and became ID 2).
    _, class_map = assign_class_ids(["10", "0", "2"], "cls")
    assert class_map == {"0": 1, "2": 2, "10": 3}
    _, class_map = assign_class_ids(["-1", "b", "10", "a", "2.5", "3"], "cls")
    assert class_map == {"-1": 1, "2.5": 2, "3": 3, "10": 4, "a": 5, "b": 6}


def test_only_ascii_digits_are_integer_ids() -> None:
    # "²" passes str.isdigit() and broke int(); a full-width "１" became ID 1 like "1".
    _, class_map = assign_class_ids(["1", "\u00b2"], "cls")
    assert class_map == {"1": 1, "\u00b2": 2}
    _, class_map = assign_class_ids(["1", "\uff11"], "cls")
    assert class_map == {"1": 1, "\uff11": 2}
    geoms, class_map = parse_geojson(
        _fc(_feat(_SQUARE, {"c": "1"}), _feat(_SQUARE2, {"c": "\uff11.\uff10"})), "c"
    )
    assert class_map == {"1": 1, "\uff11.\uff10": 2}
    assert [cid for _, cid in geoms] == [1, 2]


def test_number_spellings_of_one_value_are_one_class() -> None:
    _, class_map = parse_geojson(
        _fc(_feat(_SQUARE, {"c": 3}), _feat(_SQUARE, {"c": "3.0"}), _feat(_SQUARE, {"c": 3.0})),
        "c",
    )
    assert class_map == {"3": 3}


# ---------------------------------------------------------------------------
# Malformed GeoJSON: a message naming the file and the feature
# ---------------------------------------------------------------------------

_POLYGON = {"type": "Polygon", "coordinates": [[[0, 0], [1, 0], [1, 1], [0, 0]]]}


@pytest.mark.parametrize(
    ("document", "message"),
    [
        ([{"type": "Feature", "geometry": _POLYGON}], r"got a list"),
        ({"type": "FeatureCollection", "features": 5}, r"'features' must be a list"),
        ({"type": "FeatureCollection", "features": [5]}, r"feature 0 is not a GeoJSON Feature"),
        (
            {
                "type": "FeatureCollection",
                "features": [
                    _feat(_SQUARE, {}),
                    {"type": "Feature", "properties": [1], "geometry": _POLYGON},
                ],
            },
            r"feature 1: 'properties' must be an object, not a list",
        ),
        (_feat(_SQUARE, {}) | {"crs": "EPSG:3857"}, r"'crs' 'epsg:3857' is not supported"),
        (
            {"type": "Feature", "geometry": {"type": "Banana", "coordinates": []}},
            r"feature 0: the geometry type is 'Banana'",
        ),
        ({"type": "Feature", "geometry": {"coordinates": [0, 0]}}, r"geometry type is missing"),
        ({"type": "Feature", "geometry": "x"}, r"'geometry' must be a GeoJSON geometry object"),
        ({"type": "Feature", "geometry": {"type": "Polygon"}}, r"the Polygon has no 'coordinates'"),
        (
            {"type": "Feature", "geometry": {"type": "Polygon", "coordinates": [[0, 0], [1, 1]]}},
            r"feature 0: the Polygon coordinates are malformed",
        ),
        (
            {"type": "Feature", "geometry": {"type": "Polygon", "coordinates": [[[0, 0], [1, 1]]]}},
            r"feature 0: the Polygon coordinates are malformed",
        ),
        (
            {"type": "Feature", "geometry": {"type": "Point", "coordinates": [0, 0, 0, 0, 0]}},
            r"feature 0: the Point coordinates are malformed",
        ),
        (
            {"type": "Feature", "geometry": {"type": "GeometryCollection", "geometries": None}},
            r"needs a 'geometries' list",
        ),
    ],
)
def test_malformed_geojson_is_a_message_not_a_traceback(
    tmp_path: Path, document: Any, message: str
) -> None:
    path = tmp_path / "labels.geojson"
    path.write_text(json.dumps(document), encoding="utf-8")
    with pytest.raises(ValueError, match=r"^labels\.geojson: .*" + message):
        load_vector_labels(path)


@pytest.mark.parametrize("bad", [math.nan, math.inf, -math.inf, 91.0, -90.5])
def test_features_with_invalid_coordinates_are_skipped_and_counted(bad: float) -> None:
    broken = [[0.0, 0.0], [1.0, 0.0], [1.0, bad], [0.0, 0.0]]
    feature = {"type": "Feature", "geometry": {"type": "Polygon", "coordinates": [broken]}}
    data = _fc(_feat(_SQUARE), feature)
    with pytest.warns(UserWarning, match="skipped 1 with invalid coordinates"):
        geoms, _ = parse_geojson(data)
    # Only the valid polygon is kept (and counted by plan); it used to keep both.
    assert len(geoms) == 1
    # Lines and points to buffer are checked before buffering, which they would break.
    line = {"type": "Feature", "geometry": {"type": "LineString", "coordinates": broken[:3]}}
    with pytest.warns(UserWarning, match="skipped 1 with invalid coordinates"):
        geoms, _ = parse_geojson(_fc(_feat(_SQUARE), line), buffer=(5.0, 5.0))
    assert len(geoms) == 1


def test_polygons_in_nested_geometry_collections_are_kept() -> None:
    nested = {
        "type": "GeometryCollection",
        "geometries": [
            {
                "type": "GeometryCollection",
                "geometries": [{"type": "Polygon", "coordinates": [_SQUARE]}],
            },
            {"type": "Point", "coordinates": [5, 5]},
        ],
    }
    data = json.dumps(
        {
            "type": "FeatureCollection",
            "features": [{"type": "Feature", "properties": {}, "geometry": nested}],
        }
    ).encode()
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        geoms, _ = parse_geojson(data)
    assert len(geoms) == 1 and geoms[0][0].equals(Polygon(_SQUARE))


# ---------------------------------------------------------------------------
# A label_field that no feature has
# ---------------------------------------------------------------------------


def test_a_label_field_no_geojson_feature_has_is_an_error(tmp_path: Path) -> None:
    path = tmp_path / "labels.geojson"
    path.write_bytes(_fc(_feat(_SQUARE, {"kind": "a"}), _feat(_SQUARE2, {"Klass": "b"})))
    with pytest.raises(
        ValueError, match=r"'klass' is not a property.*'kind', 'Klass'.*Did you mean 'Klass'"
    ):
        load_vector_labels(path, "klass")
    # A field that some features have is fine; the others are skipped with a warning.
    with pytest.warns(UserWarning, match="skipped 1 without a label value"):
        geoms, _ = load_vector_labels(path, "kind")
    assert len(geoms) == 1


def test_a_kml_label_field_that_is_the_placemark_name_says_where_labels_come_from() -> None:
    with pytest.raises(
        ValueError, match=r"'name' is not a property.*not from the placemark's <name>"
    ):
        parse_kml(MISSING_FIELD_KML, label_field="name")


# ---------------------------------------------------------------------------
# Web Mercator ends at +/-85.0511 degrees
# ---------------------------------------------------------------------------


def test_polygons_reaching_the_poles_are_cut_at_the_mercator_limit() -> None:
    from pyproj import Transformer

    limit = math.degrees(math.atan(math.sinh(math.pi)))
    to_mercator = Transformer.from_crs("EPSG:4326", "EPSG:3857", always_xy=True)
    world = Polygon([(-180, -90), (180, -90), (180, 90), (-180, 90)])
    strip = Polygon([(74.305, 31.485), (74.309, 31.485), (74.309, 90), (74.305, 90)])
    for geometry in (world, strip):
        west, south, east, north = geometry.bounds
        # The reference: the polygon cut at the limit, projected by PROJ.
        x0, y0 = to_mercator.transform(west, max(south, -limit))
        x1, y1 = to_mercator.transform(east, min(north, limit))
        projected = transform_to_mercator(geometry)
        assert projected.is_valid and not projected.is_empty
        assert projected.bounds == pytest.approx((x0, y0, x1, y1), rel=1e-12, abs=1e-6)
        # It used to have infinite coordinates, and the rasterizer dropped it.
        assert all(math.isfinite(value) for value in projected.bounds)


def test_polygons_within_the_mercator_limit_are_projected_as_before() -> None:
    inside = Polygon([(10, -85), (11, -85), (11, 85), (10, 85)])
    raw = shapely.transform(inside, _mercator)
    assert transform_to_mercator(inside).equals_exact(raw, tolerance=0)


def _mercator(coords: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
    """Web Mercator by its formula, written independently of mapcv's."""
    x = 6_378_137.0 * np.radians(coords[:, 0])
    y = 6_378_137.0 * np.log(np.tan(np.pi / 4 + np.radians(coords[:, 1]) / 2))
    return np.column_stack((x, y))


def test_a_polygon_reaching_a_pole_is_burned_into_the_mask_like_rasterio() -> None:
    rasterio_features = pytest.importorskip("rasterio.features")
    from rasterio.transform import Affine

    from mapcv.rasterizer import rasterize

    # The region of the report (74.30, 31.48 -> 74.31, 31.49) at about zoom 17 pixels.
    origin = transform_to_mercator(Polygon([(74.30, 31.49), (74.31, 31.49), (74.31, 31.48)]))
    west, _, _, north = origin.bounds
    transform = (1.19, 0.0, west, 0.0, -1.19, north)
    strip = Polygon([(74.305, 31.485), (74.309, 31.485), (74.309, 90), (74.305, 90)])
    projected = transform_to_mercator(strip)
    mask = rasterize([(projected, 1)], (1000, 1000), transform)
    reference = rasterio_features.rasterize(
        [(projected, 1)], out_shape=(1000, 1000), transform=Affine(*transform), dtype="uint8"
    )
    # Masks used to be all background: the polygon had infinite coordinates.
    assert mask.any()
    np.testing.assert_array_equal(mask, reference)
