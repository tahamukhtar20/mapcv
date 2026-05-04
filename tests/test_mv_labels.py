"""MV: vertex-level validation of KML parser against geopandas golden fixtures."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Tuple

import pytest
from shapely.geometry import MultiPolygon, Polygon
from shapely.geometry.base import BaseGeometry

from mapcv.labels import parse_kml

_GOLDEN = Path(__file__).parent / "golden" / "label_parse_golden.json"

if not _GOLDEN.exists():
    pytest.skip(
        "label_parse_golden.json not found - run tests/generate_golden.py first",
        allow_module_level=True,
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

# nested_folder excluded: geopandas treats each KML Folder as a separate layer
# and cannot serve as a clean oracle. Nested folder parsing is covered in test_labels.py.
_KML_CASES = [
    ("binary", BINARY_KML, None),
    ("multiclass", MULTICLASS_KML, "land_use"),
    ("hole", HOLE_KML, None),
    ("multigeometry", MULTIGEOMETRY_KML, None),
]

_DATA: List[Dict[str, Any]] = json.loads(_GOLDEN.read_text())
_BY_ID = {c["kml_id"]: c for c in _DATA}

_COORD_TOL = 1e-10


def _exterior_coords(geom: BaseGeometry) -> List[Tuple[float, float]]:
    if isinstance(geom, Polygon):
        return [(c[0], c[1]) for c in geom.exterior.coords]
    if isinstance(geom, MultiPolygon):
        return [(c[0], c[1]) for c in list(geom.geoms)[0].exterior.coords]
    return []


@pytest.mark.parametrize("kml_id,kml_bytes,label_field", _KML_CASES)
def test_kml_feature_count_matches_geopandas(
    kml_id: str, kml_bytes: bytes, label_field: Any
) -> None:
    """Feature count from our parser must match geopandas."""
    geoms, _ = parse_kml(kml_bytes, label_field)
    ref = _BY_ID[kml_id]["features"]
    assert len(geoms) == len(ref), f"{kml_id}: got {len(geoms)} features, golden has {len(ref)}"


@pytest.mark.parametrize("kml_id,kml_bytes,label_field", _KML_CASES)
def test_kml_geometry_types_match_geopandas(
    kml_id: str, kml_bytes: bytes, label_field: Any
) -> None:
    """Geometry types from our parser must match geopandas."""
    geoms, _ = parse_kml(kml_bytes, label_field)
    ref = _BY_ID[kml_id]["features"]
    for i, ((geom, _cid), ref_feat) in enumerate(zip(geoms, ref)):
        assert geom.geom_type == ref_feat["type"], (
            f"{kml_id}[{i}]: type={geom.geom_type!r}, expected={ref_feat['type']!r}"
        )


@pytest.mark.parametrize("kml_id,kml_bytes,label_field", _KML_CASES)
def test_kml_vertex_coords_match_geopandas(kml_id: str, kml_bytes: bytes, label_field: Any) -> None:
    """Exterior ring vertex coordinates must agree within 1e-10 degrees."""
    geoms, _ = parse_kml(kml_bytes, label_field)
    ref = _BY_ID[kml_id]["features"]
    for i, ((geom, _cid), ref_feat) in enumerate(zip(geoms, ref)):
        our_coords = _exterior_coords(geom)
        ref_coords = ref_feat["exterior_coords"]
        assert len(our_coords) == len(ref_coords), (
            f"{kml_id}[{i}]: our vertex count={len(our_coords)}, ref={len(ref_coords)}"
        )
        for j, (oc, rc) in enumerate(zip(our_coords, ref_coords)):
            dx = abs(oc[0] - rc[0])
            dy = abs(oc[1] - rc[1])
            assert dx <= _COORD_TOL and dy <= _COORD_TOL, (
                f"{kml_id}[{i}] vertex[{j}]: ours={oc}, ref={rc}, dx={dx:.2e} dy={dy:.2e}"
            )
