"""GeoPackage, Shapefile and GeoParquet labels, read without GDAL.

The fixtures in ``tests/data/vector`` are written by GDAL (pyogrio) and geopandas, an
implementation independent of mapcv's readers (``tests/generate_vector_fixtures.py``).
``labels.geojson`` is the reference: every other file must give the same geometries
(coordinates within 1e-9 degrees, ring order and start vertex aside), the same class IDs and
the same class map, with the same skipped-feature counts. Projected files are checked
against pyproj, so a swapped axis or a wrong CRS shows as a test failure and not as a
shifted mask.
"""

from __future__ import annotations

import hashlib
import json
import random
import re
import shutil
import sqlite3
import struct
import sys
import threading
import warnings
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from io import BytesIO
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict, Iterator, List, Optional, Sequence, Tuple

import numpy as np
import pytest
import shapely
import shapefile
import yaml
from PIL import Image
from pydantic import ValidationError
from pyproj import CRS, Transformer
from shapely.geometry import Point, Polygon, box
from typer.testing import CliRunner

from mapcv import vector_files
from mapcv.cli import _ask_bbox_or_file, app, label_fields
from mapcv.config import LabelsConfig, MapcvConfig
from mapcv.labels import (
    GeomWithClass,
    label_file_sha256,
    load_vector_labels,
    parse_geojson,
    vector_attributes,
    vector_layers,
)
from mapcv.pipeline import run_generate

DATA = Path(__file__).parent / "data" / "vector"
GEOJSON = DATA / "labels.geojson"
TOLERANCE = 1e-9  # degrees: 0.1 mm
runner = CliRunner()


def flat(text: str) -> str:
    """Console output without line wrapping."""
    return "".join(text.split())


def load(
    path: Path, field: Optional[str] = "class", **kwargs: Any
) -> Tuple[List[GeomWithClass], Dict[str, int], List[str]]:
    """``load_vector_labels`` plus the text of the warnings it raised."""
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        geometries, class_map = load_vector_labels(path, field, **kwargs)
    return geometries, class_map, [str(item.message) for item in caught]


def _unwrapped(geometry: Any) -> Any:
    """A one-part multipolygon as its polygon: GDAL writes a layer of mixed Polygon and
    MultiPolygon features as all-MultiPolygon, which is the same area."""
    if geometry.geom_type == "MultiPolygon" and len(geometry.geoms) == 1:
        return geometry.geoms[0]
    return geometry


def assert_same(got: Sequence[GeomWithClass], expected: Sequence[GeomWithClass]) -> None:
    assert len(got) == len(expected)
    for (geometry, class_id), (reference, reference_id) in zip(got, expected):
        assert class_id == reference_id
        geometry, reference = _unwrapped(geometry), _unwrapped(reference)
        assert geometry.geom_type == reference.geom_type
        assert geometry.normalize().equals_exact(reference.normalize(), TOLERANCE), (
            geometry.wkt,
            reference.wkt,
        )


def reference(field: Optional[str] = "class", points: bool = False, path: Path = GEOJSON) -> Any:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return parse_geojson(path.read_bytes(), field, None, points=points)


def skipped(messages: List[str]) -> str:
    """The text of the skipped-feature warning, without the format name."""
    assert len(messages) == 1
    return messages[0].split(": ", 1)[1]


# ── The same features in every format ────────────────────────────────────────

MIXED = ["labels.gpkg", "labels_32633.gpkg", "labels_4269.gpkg", "labels.parquet"]
MIXED += ["labels_32633.parquet"]


@pytest.mark.parametrize("name", MIXED)
@pytest.mark.parametrize("field", ["class", "rank", "score", None])
@pytest.mark.parametrize("points", [False, True])
def test_mixed_formats_match_the_geojson(name: str, field: Optional[str], points: bool) -> None:
    if name.endswith(".parquet"):
        pytest.importorskip("pyarrow")
    expected, expected_map = reference(field, points)
    got, class_map, messages = load(DATA / name, field, points=points)
    assert_same(got, expected)
    assert class_map == expected_map
    _, _, reference_messages = load(GEOJSON, field, points=points)
    assert [skipped(messages)] == [skipped(reference_messages)]


def test_fixture_covers_the_hard_cases() -> None:
    """The reference holds what the issue asks for, so the comparisons above mean something."""
    geometries, class_map = reference("class", points=True)
    kinds = sorted(geometry.geom_type for geometry, _ in geometries)
    assert kinds == ["MultiPolygon", "MultiPolygon", "Point", "Polygon", "Polygon"]
    assert class_map == {"building": 1, "farmland": 2, "forêt": 3}
    holes = [len(polygon.interiors) for g, _ in geometries for polygon in _polygons(g)]
    assert max(holes) >= 1
    # A polygon lying inside a hole of its own multipolygon (the hard case for Shapefiles).
    nested = [g for g, _ in geometries if g.geom_type == "MultiPolygon" and len(g.geoms) == 2][0]
    assert (
        any(nested.geoms[0].interiors[0].coords)
        and nested.geoms[0].contains(nested.geoms[1]) is False
    )
    assert nested.geoms[1].within(Polygon(nested.geoms[0].interiors[0]))


def _polygons(geometry: Any) -> List[Polygon]:
    if geometry.geom_type == "Polygon":
        return [geometry]
    if geometry.geom_type == "MultiPolygon":
        return list(geometry.geoms)
    return []


def test_geopackage_layers_are_discoverable_and_selectable() -> None:
    path = DATA / "labels_2layers.gpkg"
    assert vector_layers(path) == ["buildings", "landuse"]
    for layer in ("buildings", "landuse"):
        expected, expected_map = reference(path=DATA / f"layer_{layer}.geojson")
        got, class_map, _ = load(path, layer=layer)
        assert_same(got, expected)
        assert class_map == expected_map


def test_a_single_layer_needs_no_layer_name() -> None:
    assert vector_layers(DATA / "labels.gpkg") == ["labels"]
    assert load(DATA / "labels.gpkg")[0]  # no layer given
    assert load(DATA / "labels.gpkg", layer="labels")[0]


def test_two_layers_need_layer_and_the_error_lists_them() -> None:
    with pytest.raises(ValueError, match=r"2 layers: buildings, landuse.*labels\.layer"):
        load_vector_labels(DATA / "labels_2layers.gpkg", "class")
    with pytest.raises(ValueError, match=r"no layer 'roads'.*buildings, landuse"):
        load_vector_labels(DATA / "labels_2layers.gpkg", "class", layer="roads")
    with pytest.raises(ValueError, match="no layer 'roads'"):
        load_vector_labels(DATA / "labels.gpkg", "class", layer="roads")


@pytest.mark.parametrize("name", ["labels.geojson", "polygons.shp"])
def test_layer_is_for_geopackages_only(name: str) -> None:
    with pytest.raises(ValueError, match="GeoPackage files only"):
        load_vector_labels(DATA / name, "class", layer="x")


@pytest.mark.parametrize("name", ["labels.gpkg", "polygons.shp", "labels.parquet"])
def test_an_unknown_label_field_is_an_error_listing_the_columns(name: str) -> None:
    if name.endswith(".parquet"):
        pytest.importorskip("pyarrow")
    with pytest.raises(ValueError, match=r"'clas' is not a column.*'class'.*'rank'"):
        load_vector_labels(DATA / name, "clas")


# ── Shapefile ────────────────────────────────────────────────────────────────


@pytest.mark.parametrize("name", ["polygons.shp", "polygons_32633.shp", "polygons_latin1.shp"])
@pytest.mark.parametrize("field", ["class", "rank", "score", None])
def test_shapefile_polygons_match_the_geojson(name: str, field: Optional[str]) -> None:
    expected, expected_map = reference(field)
    got, class_map, messages = load(DATA / name, field)
    assert_same(got, expected)
    assert class_map == expected_map
    if field == "class":
        assert skipped(messages) == "skipped 1 without a label value."


def test_shapefile_points_and_lines_follow_the_geojson_rules() -> None:
    expected, expected_map = reference("class", points=True, path=DATA / "points.geojson")
    got, class_map, messages = load(DATA / "points.shp", points=True)
    assert_same(got, expected)
    assert class_map == expected_map == {"building": 1}
    assert messages == []

    got, class_map, messages = load(DATA / "points.shp", points=False)
    assert got == [] and class_map == {}
    assert skipped(messages) == "skipped 1 without polygon geometry (points/lines)."

    got, _, messages = load(DATA / "lines.shp", points=True)
    assert got == []
    assert skipped(messages) == "skipped 1 without polygon geometry (lines)."


def test_projected_files_hold_projected_coordinates_that_come_back_as_lonlat() -> None:
    """The 32633 fixtures really are in metres, and read back as the lon/lat they came from."""
    forward = Transformer.from_crs("EPSG:4326", "EPSG:32633", always_xy=True)
    first = reference("class")[0][0][0]  # the polygon with a hole
    expected = {
        (round(x, 3), round(y, 3))
        for x, y in zip(*forward.transform(*np.asarray(first.exterior.coords).T))
    }
    raw = shapefile.Reader(str(DATA / "polygons_32633.shp")).shape(0)
    assert {
        (round(x, 3), round(y, 3)) for x, y in raw.points[: len(first.exterior.coords) - 1]
    } <= (expected)
    assert min(x for x, _ in raw.points) > 100_000  # metres, not degrees

    # And the loader's result is the inverse of the independent forward projection.
    got = load(DATA / "polygons_32633.shp")[0][0][0]
    assert got.normalize().equals_exact(first.normalize(), TOLERANCE)


def test_nad83_is_read_with_longitude_first() -> None:
    """EPSG:4269 lists latitude first; the data is still x = lon, y = lat (``always_xy``)."""
    got = load(DATA / "labels_4269.gpkg")[0][0][0]
    back = Transformer.from_crs("EPSG:4269", "EPSG:4326", always_xy=True)
    raw = reference("class")[0][0][0]
    x, y = back.transform(*np.asarray(raw.exterior.coords).T)
    assert np.allclose(np.asarray(got.exterior.coords), np.column_stack((x, y)), atol=TOLERANCE)
    # Swapped axes would put the polygon in the sea at 16 N, 48 E.
    assert 16.0 < got.bounds[0] < 17.0 and 48.0 < got.bounds[1] < 49.0


def copy_shapefile(tmp_path: Path, stem: str = "polygons", skip: Sequence[str] = ()) -> Path:
    for source in DATA.glob(f"{stem}.*"):
        if source.suffix not in (".geojson", *skip):
            shutil.copy(source, tmp_path / source.name)
    return tmp_path / f"{stem}.shp"


def test_a_shapefile_without_prj_is_an_error_not_a_guess(tmp_path: Path) -> None:
    path = copy_shapefile(tmp_path, skip=[".prj"])
    with pytest.raises(ValueError) as error:
        load_vector_labels(path, "class")
    text = str(error.value)
    assert "no .prj file" in text and "CRS is unknown" in text
    assert "will not guess" in text and "polygons.prj" in text and "-a_srs" in text


def test_an_unreadable_prj_is_an_error(tmp_path: Path) -> None:
    path = copy_shapefile(tmp_path)
    path.with_suffix(".prj").write_text("not a crs", encoding="utf-8")
    with pytest.raises(ValueError, match=r"polygons\.prj.*CRS definition cannot be understood"):
        load_vector_labels(path, "class")


def test_sidecar_files_are_found_whatever_their_case(tmp_path: Path) -> None:
    path = copy_shapefile(tmp_path)
    for suffix in (".prj", ".dbf", ".cpg", ".shx"):
        path.with_suffix(suffix).rename(path.with_suffix(suffix.upper()))
    got, _, _ = load(path)
    assert_same(got, reference()[0])


def test_cpg_sets_the_attribute_encoding(tmp_path: Path) -> None:
    path = copy_shapefile(tmp_path, "polygons_latin1")
    assert path.with_suffix(".cpg").read_text().strip() == "ISO-8859-1"
    assert "forêt" in load(path)[1]
    # The same bytes read as UTF-8 do not decode: an error, not "forÃªt" as a class name.
    path.with_suffix(".cpg").write_text("UTF-8")
    with pytest.raises(ValueError, match=r"not valid utf-8|do not decode as utf-8") as error:
        load_vector_labels(path, "class")
    assert ".cpg" in str(error.value)
    path.with_suffix(".cpg").unlink()
    with pytest.raises(ValueError, match=r"\.cpg"):
        load_vector_labels(path, "class")


@pytest.mark.parametrize("text", ["windows-1252", "ANSI 1252", "1252", "latin-1"])
def test_cpg_names_are_understood(tmp_path: Path, text: str) -> None:
    path = copy_shapefile(tmp_path, "polygons_latin1")
    path.with_suffix(".cpg").write_text(text)
    assert "forêt" in load(path)[1]


def test_an_unknown_cpg_encoding_is_an_error(tmp_path: Path) -> None:
    path = copy_shapefile(tmp_path)
    path.with_suffix(".cpg").write_text("klingon")
    with pytest.raises(ValueError, match=r"polygons\.cpg names the encoding 'klingon'"):
        load_vector_labels(path, "class")


def test_a_shapefile_without_dbf_works_without_a_label_field(tmp_path: Path) -> None:
    path = copy_shapefile(tmp_path, skip=[".dbf"])
    got, class_map, _ = load(path, None)
    assert len(got) == 5 and class_map == {}
    with pytest.raises(ValueError, match="'class' is not a column"):
        load_vector_labels(path, "class")


def test_sidecar_suffixes_of_any_case_mix_are_found(tmp_path: Path) -> None:
    path = copy_shapefile(tmp_path)
    for suffix, mixed in ((".dbf", ".Dbf"), (".prj", ".PrJ"), (".cpg", ".cPg")):
        path.with_suffix(suffix).rename(path.with_suffix(mixed))
    assert_same(load(path)[0], reference()[0])
    assert label_file_sha256(path)


def test_the_hash_of_a_lone_shp_covers_just_that_file(tmp_path: Path) -> None:
    path = copy_shapefile(tmp_path, skip=[".shx", ".dbf", ".prj", ".cpg"])
    assert label_file_sha256(path) == hashlib.sha256(path.read_bytes()).hexdigest()


def shape(shape_type: int, points: Any, parts: Any = (0,)) -> Any:
    """A stand-in for a pyshp shape: ``_shape_geometry`` only reads these three fields."""
    return SimpleNamespace(shapeType=shape_type, points=list(points), parts=list(parts))


def geometry_of(record: Any) -> Any:
    return vector_files._shape_geometry(record)


def test_shape_geometry_kinds() -> None:
    square = [(0.0, 0.0), (0.0, 1.0), (1.0, 1.0), (1.0, 0.0), (0.0, 0.0)]
    assert geometry_of(shape(0, [])) is None
    assert geometry_of(shape(1, [(1.0, 2.0)])).equals(Point(1, 2))
    assert geometry_of(shape(18, [(1.0, 2.0), (3.0, 4.0)])).geom_type == "MultiPoint"
    assert geometry_of(shape(3, square, [0])).geom_type == "LineString"
    assert geometry_of(shape(13, square + square, [0, 5])).geom_type == "MultiLineString"
    # A polygon of one ring is returned as a ring, for the caller to build in bulk.
    ring = geometry_of(shape(5, square, [0]))
    assert isinstance(ring, vector_files._SingleRing) and len(ring.ring) == 5
    inner = [(0.2, 0.2), (0.8, 0.2), (0.8, 0.8), (0.2, 0.8), (0.2, 0.2)]
    holed = geometry_of(shape(5, square + inner, [0, 5]))
    assert holed.geom_type == "Polygon" and holed.area == pytest.approx(0.64)
    two = [(5.0, 5.0), (5.0, 6.0), (6.0, 6.0), (6.0, 5.0), (5.0, 5.0)]
    assert geometry_of(shape(5, square + two, [0, 5])).geom_type == "MultiPolygon"
    # A ring of fewer than three points is dropped: a real ring and a stray pair make one polygon.
    assert geometry_of(shape(5, square + [(9.0, 9.0), (9.5, 9.5)], [0, 5])).geom_type == "Polygon"


@pytest.mark.parametrize(
    "record",
    [
        shape(1, []),  # a point without coordinates
        shape(3, [(0.0, 0.0)], [0]),  # a line of one point
        shape(5, [(0.0, 0.0), (1.0, 1.0)], [0]),  # a polygon without a ring
        shape(31, [(0.0, 0.0)] * 4, [0]),  # a MultiPatch
        shape(77, [(0.0, 0.0)] * 4, [0]),  # an unknown type
    ],
)
def test_shapes_without_usable_geometry_raise_type_error(record: Any) -> None:
    with pytest.raises(TypeError):
        geometry_of(record)


def test_shape_parts_out_of_order_are_a_value_error() -> None:
    with pytest.raises(ValueError, match="part offsets"):
        geometry_of(shape(5, [(0.0, 0.0)] * 10, [5, 0]))
    with pytest.raises(ValueError, match="part offsets"):
        geometry_of(shape(5, [(0.0, 0.0)] * 4, [0, 9]))


def test_a_shapefile_without_shx_is_read_in_order(tmp_path: Path) -> None:
    path = copy_shapefile(tmp_path, skip=[".shx"])
    assert_same(load(path)[0], reference()[0])


def test_records_deleted_in_the_dbf_are_dropped_with_their_shape(tmp_path: Path) -> None:
    """As GDAL does (checked with pyogrio): the dbf deletion flag removes the feature."""
    path = copy_shapefile(tmp_path)
    dbf = path.with_suffix(".dbf")
    data = bytearray(dbf.read_bytes())
    header_length, record_length = struct.unpack("<HH", data[8:12])
    data[header_length + 1 * record_length] = 0x2A  # delete the second feature (farmland)
    dbf.write_bytes(bytes(data))
    got, class_map, _ = load(path)
    expected = reference()[0]
    kept = [expected[0], expected[2], expected[3]]
    assert [g.normalize().wkt for g, _ in got] == [g.normalize().wkt for g, _ in kept]
    assert class_map == {"building": 1, "forêt": 2}


def test_shape_and_record_counts_must_agree(tmp_path: Path) -> None:
    path = copy_shapefile(tmp_path)
    dbf = path.with_suffix(".dbf")
    data = bytearray(dbf.read_bytes())
    struct.pack_into("<I", data, 4, 3)  # claim 3 records where the .shp holds 5 shapes
    dbf.write_bytes(bytes(data))
    with pytest.raises(ValueError, match="different numbers of features"):
        load_vector_labels(path, "class")


def write_z_shapefile(directory: Path) -> Path:
    """A PolygonZ shapefile written with pyshp, with a WGS-84 .prj."""
    path = directory / "z.shp"
    with shapefile.Writer(str(path), shapeType=shapefile.POLYGONZ) as writer:
        writer.field("class", "C")
        ring = [(16.36, 48.19, 5.0), (16.36, 48.20, 6.0), (16.37, 48.20, 7.0), (16.36, 48.19, 5.0)]
        writer.polyz([ring])
        writer.record("a")
    path.with_suffix(".prj").write_text(CRS.from_epsg(4326).to_wkt(), encoding="utf-8")
    return path


def test_z_coordinates_are_dropped(tmp_path: Path) -> None:
    got, class_map, _ = load(write_z_shapefile(tmp_path))
    assert class_map == {"a": 1}
    polygon = got[0][0]
    assert not polygon.has_z
    expected = Polygon([(16.36, 48.19), (16.36, 48.20), (16.37, 48.20)])
    assert polygon.normalize().equals_exact(expected.normalize(), TOLERANCE)


def rewrite_rings(source: Path, target: Path, reverse: bool) -> Path:
    """The polygons of ``source`` written again with pyshp, every ring as it is or reversed."""
    with shapefile.Reader(str(source)) as reader:
        shapes = reader.shapes()
        records = reader.records()
    with shapefile.Writer(str(target), shapeType=shapefile.POLYGON) as writer:
        for name, kind, size, decimals in reader.fields[1:]:
            writer.field(name, kind, size, decimals)
        for shape, record in zip(shapes, records):
            bounds = [*shape.parts, len(shape.points)]
            rings = [shape.points[a:b] for a, b in zip(bounds, bounds[1:])]
            writer.poly([ring[::-1] if reverse else ring for ring in rings])
            writer.record(*record)
    shutil.copy(source.with_suffix(".prj"), target.with_suffix(".prj"))
    shutil.copy(source.with_suffix(".cpg"), target.with_suffix(".cpg"))
    return target


def test_ring_winding_does_not_decide_what_is_a_hole(tmp_path: Path) -> None:
    """Holes and islands come from containment, as in GDAL, not from clockwise or not."""
    source = copy_shapefile(tmp_path)
    expected = reference()[0]
    # As GDAL wrote them (exteriors clockwise), and with every ring reversed (counter-clockwise
    # exteriors, clockwise holes), which some writers produce.
    for name, reverse in (("same", False), ("reversed", True)):
        directory = tmp_path / name
        directory.mkdir()
        path = rewrite_rings(source, directory / "polygons.shp", reverse)
        assert_same(load(path)[0], expected)


def test_null_and_zero_area_shapes(tmp_path: Path) -> None:
    """A null shape is ignored (as a null GeoJSON geometry); a ring of one repeated point is
    kept with no area, as shapely builds it from GeoJSON."""
    path = tmp_path / "odd.shp"
    good = [(16.36, 48.19), (16.36, 48.20), (16.37, 48.20), (16.36, 48.19)]
    with shapefile.Writer(str(path), shapeType=shapefile.POLYGON) as writer:
        writer.field("class", "C")
        writer.poly([good])
        writer.record("ok")
        writer.null()
        writer.record("null")
        writer.poly([[(16.4, 48.2), (16.4, 48.2), (16.4, 48.2), (16.4, 48.2)]])
        writer.record("flat")
        writer.poly([good, [(16.5, 48.2), (16.5, 48.2), (16.5, 48.2), (16.5, 48.2)]])
        writer.record("mixed")
    path.with_suffix(".prj").write_text(CRS.from_epsg(4326).to_wkt(), encoding="utf-8")
    got, class_map, messages = load(path)
    assert class_map == {"flat": 1, "mixed": 2, "ok": 3}
    areas = [round(g.area, 9) for g, _ in got]
    assert areas == [round(Polygon(good).area, 9), 0.0, round(Polygon(good).area, 9)]
    assert messages == []


def test_unreadable_shape_types_are_counted_as_skipped(tmp_path: Path) -> None:
    """A MultiPatch has no polygon geometry mapcv can read."""
    path = tmp_path / "patch.shp"
    with shapefile.Writer(str(path), shapeType=shapefile.MULTIPATCH) as writer:
        writer.field("class", "C")
        ring = [(16.36, 48.19, 0.0), (16.36, 48.20, 0.0), (16.37, 48.20, 0.0), (16.36, 48.19, 0.0)]
        writer.multipatch([ring], partTypes=[shapefile.RING])
        writer.record("a")
    path.with_suffix(".prj").write_text(CRS.from_epsg(4326).to_wkt(), encoding="utf-8")
    got, _, messages = load(path)
    assert got == []
    assert skipped(messages) == "skipped 1 without polygon geometry (points/lines)."


# ── GeoPackage by hand: the binary header and the CRS table ──────────────────


def gpkg_blob(
    geometry: Any,
    srs_id: int = 4326,
    envelope: int = 0,
    little: bool = True,
    empty: bool = False,
    extended: bool = False,
    wkb_little: bool = True,
) -> bytes:
    """A GeoPackage geometry blob (spec 2.1.3) written from the spec, field by field."""
    flags = int(little) | (envelope << 1) | (0x10 if empty else 0) | (0x20 if extended else 0)
    order = "<" if little else ">"
    sizes = {0: 0, 1: 4, 2: 6, 3: 6, 4: 8}[envelope]
    minx, miny, maxx, maxy = geometry.bounds if not geometry.is_empty else (0, 0, 0, 0)
    values = [minx, maxx, miny, maxy, 0.0, 0.0, 0.0, 0.0][:sizes]
    header = b"GP" + bytes([0, flags]) + struct.pack(f"{order}i", srs_id)
    header += struct.pack(f"{order}{sizes}d", *values)
    wkb: bytes = shapely.to_wkb(geometry, byte_order=1 if wkb_little else 0)
    return header + wkb


def make_gpkg(
    path: Path,
    rows: Sequence[Tuple[Optional[bytes], Optional[str]]],
    srs: Sequence[Tuple[int, str, str, int, str]] = (
        (4326, "WGS 84", "EPSG", 4326, "undefined"),
        (-1, "Undefined Cartesian SRS", "NONE", -1, "undefined"),
        (0, "Undefined geographic SRS", "NONE", 0, "undefined"),
    ),
    srs_id: int = 4326,
) -> Path:
    """A minimal GeoPackage written with ``sqlite3`` only: one table ``feat`` (geom, label)."""
    connection = sqlite3.connect(path)
    connection.executescript(
        """
        CREATE TABLE gpkg_spatial_ref_sys (srs_name TEXT, srs_id INTEGER PRIMARY KEY,
            organization TEXT, organization_coordsys_id INTEGER, definition TEXT);
        CREATE TABLE gpkg_contents (table_name TEXT PRIMARY KEY, data_type TEXT, identifier TEXT);
        CREATE TABLE gpkg_geometry_columns (table_name TEXT, column_name TEXT,
            geometry_type_name TEXT, srs_id INTEGER, z INTEGER, m INTEGER);
        CREATE TABLE feat (fid INTEGER PRIMARY KEY AUTOINCREMENT, geom BLOB, label TEXT);
        INSERT INTO gpkg_contents VALUES ('feat', 'features', 'feat');
        """
    )
    connection.execute(
        "INSERT INTO gpkg_geometry_columns VALUES ('feat', 'geom', 'GEOMETRY', ?, 0, 0)", (srs_id,)
    )
    for number, name, org, code, definition in srs:
        connection.execute(
            "INSERT INTO gpkg_spatial_ref_sys VALUES (?, ?, ?, ?, ?)",
            (name, number, org, code, definition),
        )
    connection.executemany("INSERT INTO feat (geom, label) VALUES (?, ?)", rows)
    connection.commit()
    connection.close()
    return path


SQUARE = box(16.36, 48.19, 16.37, 48.20)
HOLED = Polygon(
    box(16.30, 48.10, 16.40, 48.20).exterior.coords,
    [box(16.33, 48.13, 16.37, 48.17).exterior.coords],
)


@pytest.mark.parametrize("envelope", [0, 1, 2, 3, 4])
@pytest.mark.parametrize("little", [True, False])
@pytest.mark.parametrize("wkb_little", [True, False])
def test_geopackage_header_variants(
    tmp_path: Path, envelope: int, little: bool, wkb_little: bool
) -> None:
    blob = gpkg_blob(HOLED, envelope=envelope, little=little, wkb_little=wkb_little)
    path = make_gpkg(tmp_path / "a.gpkg", [(blob, "x")])
    got, class_map, _ = load(path, "label")
    assert class_map == {"x": 1}
    assert got[0][0].equals_exact(HOLED, TOLERANCE)


def test_empty_and_null_geometries_are_ignored(tmp_path: Path) -> None:
    rows: List[Tuple[Optional[bytes], Optional[str]]] = [
        (gpkg_blob(SQUARE), "a"),
        (None, "b"),
        (gpkg_blob(Polygon(), empty=True), "c"),
        (gpkg_blob(Point(16.365, 48.195), envelope=1), "d"),
    ]
    got, class_map, messages = load(make_gpkg(tmp_path / "a.gpkg", rows), "label")
    assert [class_id for _, class_id in got] == [1] and class_map == {"a": 1}
    assert skipped(messages) == "skipped 1 without polygon geometry (points/lines)."


@pytest.mark.parametrize(
    "blob, message",
    [
        (b"XX\x00\x01" + bytes(20), "must start with 'GP'"),
        (b"GP", "must start with 'GP'"),
        (b"GP\x01\x01" + bytes(20), "version 1 is not supported"),
        (b"GP\x00\x21" + bytes(20), "ExtendedGeoPackageBinary"),
        (b"GP\x00\x0f" + bytes(20), "invalid envelope indicator"),
        (b"GP\x00\x01\xe6\x10\x00\x00", "truncated"),
        (b"GP\x00\x01\xe6\x10\x00\x00\x01\x02\x03", "not valid WKB"),
    ],
)
def test_corrupt_geopackage_geometries_are_value_errors(
    tmp_path: Path, blob: bytes, message: str
) -> None:
    path = make_gpkg(tmp_path / "a.gpkg", [(blob, "x")])
    with pytest.raises(ValueError, match=message):
        load_vector_labels(path, "label")


def test_geopackage_crs_from_the_authority_code(tmp_path: Path) -> None:
    utm = Transformer.from_crs("EPSG:4326", "EPSG:32633", always_xy=True)
    projected = shapely.transform(
        SQUARE, lambda c: np.column_stack(utm.transform(c[:, 0], c[:, 1]))
    )
    srs = [(32633, "UTM 33N", "EPSG", 32633, "undefined")]
    path = make_gpkg(tmp_path / "a.gpkg", [(gpkg_blob(projected, 32633), "x")], srs, 32633)
    assert_same(load(path, "label")[0], [(SQUARE, 1)])


def test_geopackage_crs_from_the_wkt_definition(tmp_path: Path) -> None:
    """An organization pyproj does not know: the stored WKT definition is used."""
    utm = Transformer.from_crs("EPSG:4326", "EPSG:32633", always_xy=True)
    projected = shapely.transform(
        SQUARE, lambda c: np.column_stack(utm.transform(c[:, 0], c[:, 1]))
    )
    wkt = CRS.from_epsg(32633).to_wkt()
    srs = [(100001, "my utm", "my-company", 7, wkt)]
    path = make_gpkg(tmp_path / "a.gpkg", [(gpkg_blob(projected, 100001), "x")], srs, 100001)
    assert_same(load(path, "label")[0], [(SQUARE, 1)])


@pytest.mark.parametrize(
    "srs_id, srs, message",
    [
        (0, None, r"no CRS \(srs_id 0"),
        (-1, None, r"no CRS \(srs_id -1"),
        (777, None, r"srs_id 777 is not in gpkg_spatial_ref_sys"),
        (555, [(555, "x", "my-company", 1, "undefined")], "not one mapcv can interpret"),
        (556, [(556, "x", "NONE", 1, "garbage")], "not one mapcv can interpret"),
    ],
)
def test_geopackage_with_an_unknown_crs_is_an_error(
    tmp_path: Path, srs_id: int, srs: Optional[List[Tuple[int, str, str, int, str]]], message: str
) -> None:
    extra: Tuple[Tuple[int, str, str, int, str], ...] = (
        (4326, "WGS 84", "EPSG", 4326, "undefined"),
        (0, "u", "NONE", 0, "undefined"),
        (-1, "u", "NONE", -1, "undefined"),
        *(srs or []),
    )
    path = make_gpkg(tmp_path / "a.gpkg", [(gpkg_blob(SQUARE, srs_id), "x")], extra, srs_id)
    with pytest.raises(ValueError, match=message):
        load_vector_labels(path, "label")


def test_coordinates_outside_a_crs_area_of_use_are_an_error(tmp_path: Path) -> None:
    far = box(1e9, 1e9, 1e9 + 1, 1e9 + 1)
    srs = [(32633, "UTM 33N", "EPSG", 32633, "undefined")]
    path = make_gpkg(tmp_path / "a.gpkg", [(gpkg_blob(far, 32633), "x")], srs, 32633)
    with pytest.raises(ValueError, match="cannot be reprojected"):
        load_vector_labels(path, "label")


def test_not_a_geopackage_and_missing_files(tmp_path: Path) -> None:
    plain = tmp_path / "plain.gpkg"
    connection = sqlite3.connect(plain)
    connection.execute("CREATE TABLE t (a INTEGER)")
    connection.commit()
    connection.close()
    with pytest.raises(ValueError, match="not a readable GeoPackage"):
        load_vector_labels(plain, "a")
    text = tmp_path / "text.gpkg"
    text.write_text("this is not sqlite")
    with pytest.raises(ValueError, match="not a readable GeoPackage"):
        load_vector_labels(text, "a")
    for name in ("none.gpkg", "none.shp", "none.parquet", "none.geojson", "none.kml"):
        with pytest.raises(FileNotFoundError):
            load_vector_labels(tmp_path / name, None)


def test_a_layer_whose_geometries_are_all_null_is_empty_not_an_error(tmp_path: Path) -> None:
    srs = [(32633, "UTM 33N", "EPSG", 32633, "undefined")]
    path = make_gpkg(tmp_path / "a.gpkg", [(None, "x"), (None, "y")], srs, 32633)
    got, class_map, messages = load(path, "label")
    assert got == [] and class_map == {} and messages == []


def test_a_crs_that_cannot_be_transformed_is_an_error(tmp_path: Path) -> None:
    """A local engineering CRS has no relation to WGS-84."""
    wkt = (
        'ENGCRS["local",EDATUM["x"],CS[Cartesian,2],AXIS["x",east],AXIS["y",north],'
        'LENGTHUNIT["metre",1]]'
    )
    srs = [(100002, "local", "my-company", 1, wkt)]
    path = make_gpkg(tmp_path / "a.gpkg", [(gpkg_blob(SQUARE, 100002), "x")], srs, 100002)
    with pytest.raises(ValueError, match="cannot reproject from its CRS"):
        load_vector_labels(path, "label")


def test_damaged_geopackage_tables_are_value_errors(tmp_path: Path) -> None:
    no_srs = make_gpkg(tmp_path / "no_srs.gpkg", [(gpkg_blob(SQUARE), "x")])
    connection = sqlite3.connect(no_srs)
    connection.execute("DROP TABLE gpkg_spatial_ref_sys")
    connection.commit()
    connection.close()
    with pytest.raises(ValueError, match="cannot read gpkg_spatial_ref_sys"):
        load_vector_labels(no_srs, "label")

    no_table = make_gpkg(tmp_path / "no_table.gpkg", [(gpkg_blob(SQUARE), "x")])
    connection = sqlite3.connect(no_table)
    connection.execute("DROP TABLE feat")
    connection.commit()
    connection.close()
    with pytest.raises(ValueError, match="cannot read the features"):
        load_vector_labels(no_table, None)


def test_geopackage_features_come_back_in_fid_order(tmp_path: Path) -> None:
    """Feature order decides overlaps and instance IDs, so it must not depend on how the
    rows are stored. GDAL (pyogrio) returns the same file in fid order, checked by hand."""
    path = make_gpkg(tmp_path / "a.gpkg", [])
    connection = sqlite3.connect(path)
    for fid, label in ((3, "c"), (1, "a"), (2, "b")):  # physical order 3, 1, 2
        blob = gpkg_blob(box(16.0 + (4 - fid), 48.0, 16.5 + (4 - fid), 48.5))
        connection.execute(
            "INSERT INTO feat (fid, geom, label) VALUES (?, ?, ?)", (fid, blob, label)
        )
    # With only the geometry selected, an index on it covers the query: SQLite then scans the
    # index (in geometry order, here 3, 2, 1) unless the query says ORDER BY.
    connection.execute("CREATE INDEX feat_scan ON feat (geom)")
    connection.commit()
    connection.close()
    for field in (None, "label"):
        got = load(path, field)[0]
        assert [round(g.bounds[0]) for g, _ in got] == [19, 18, 17], field  # fid 1, 2, 3
    assert [c for _, c in load(path, "label")[0]] == [1, 2, 3]  # a, b, c


def test_geopackage_order_without_an_integer_key(tmp_path: Path) -> None:
    path = tmp_path / "a.gpkg"
    make_gpkg(path, [(gpkg_blob(box(1, 1, 2, 2)), "a"), (gpkg_blob(box(3, 3, 4, 4)), "b")])
    connection = sqlite3.connect(path)
    # No primary key at all: ordered by rowid (insertion order).
    connection.executescript("DROP TABLE feat; CREATE TABLE feat (geom BLOB, label TEXT);")
    for x, label in ((5, "z"), (1, "y")):
        connection.execute(
            "INSERT INTO feat VALUES (?, ?)", (gpkg_blob(box(x, 0, x + 1, 1)), label)
        )
    connection.commit()
    connection.close()
    assert [g.bounds[0] for g, _ in load(path, "label")[0]] == [5.0, 1.0]
    # WITHOUT ROWID with a text key: no rowid to order by, still readable.
    connection = sqlite3.connect(path)
    connection.executescript(
        "DROP TABLE feat;"
        "CREATE TABLE feat (name TEXT PRIMARY KEY, geom BLOB, label TEXT) WITHOUT ROWID;"
    )
    connection.execute("INSERT INTO feat VALUES ('k', ?, 'q')", (gpkg_blob(box(0, 0, 1, 1)),))
    connection.commit()
    connection.close()
    assert len(load(path, "label")[0]) == 1


def test_a_geopackage_with_no_features_table(tmp_path: Path) -> None:
    path = make_gpkg(tmp_path / "a.gpkg", [])
    connection = sqlite3.connect(path)
    connection.execute("UPDATE gpkg_contents SET data_type = 'attributes'")
    connection.commit()
    connection.close()
    with pytest.raises(ValueError, match="no feature tables"):
        load_vector_labels(path, "label")


# ── GeoParquet ───────────────────────────────────────────────────────────────


@pytest.fixture
def pq() -> Any:
    return pytest.importorskip("pyarrow.parquet")


def rewrite_geo(source: Path, target: Path, change: Any) -> Path:
    """Copy a GeoParquet file with its ``geo`` metadata changed by ``change(geo_dict)``."""
    import pyarrow.parquet as parquet

    table = parquet.read_table(source)
    geo = json.loads(table.schema.metadata[b"geo"])
    change(geo)
    metadata = {**table.schema.metadata, b"geo": json.dumps(geo).encode()}
    parquet.write_table(table.replace_schema_metadata(metadata), target)
    return target


def test_geoparquet_without_pyarrow_says_how_to_install_it(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setitem(sys.modules, "pyarrow", None)
    monkeypatch.setitem(sys.modules, "pyarrow.parquet", None)
    with pytest.raises(ValueError) as error:
        load_vector_labels(DATA / "labels.parquet", "class")
    assert "pip install 'mapcv[parquet]'" in str(error.value)


def test_geoparquet_that_is_not_wkb_is_an_error(pq: Any) -> None:
    with pytest.raises(ValueError, match=r"'multipolygon' encoding; mapcv reads WKB.*WKB"):
        load_vector_labels(DATA / "labels_geoarrow.parquet", "class")


def test_geoparquet_crs_null_is_unknown_and_missing_means_crs84(pq: Any, tmp_path: Path) -> None:
    unknown = rewrite_geo(
        DATA / "labels.parquet", tmp_path / "null.parquet", lambda geo: _set_crs(geo, None)
    )
    with pytest.raises(ValueError, match=r"sets crs to null.*will not guess"):
        load_vector_labels(unknown, "class")
    default = rewrite_geo(
        DATA / "labels.parquet", tmp_path / "absent.parquet", lambda geo: _set_crs(geo, ...)
    )
    expected, expected_map = reference("class")
    got, class_map, _ = load(default)
    assert_same(got, expected)
    assert class_map == expected_map


def _set_crs(geo: Dict[str, Any], crs: Any) -> None:
    column = geo["columns"][geo["primary_column"]]
    if crs is ...:
        column.pop("crs", None)
    else:
        column["crs"] = crs


def test_geoparquet_crs_is_projjson_with_any_axis_order(pq: Any) -> None:
    """geopandas writes EPSG:4326 PROJJSON (latitude first); coordinates are still lon/lat."""
    geo = json.loads(pq.read_table(DATA / "labels.parquet").schema.metadata[b"geo"])
    crs = geo["columns"]["geometry"]["crs"]
    assert crs["id"] == {"authority": "EPSG", "code": 4326}
    assert crs["coordinate_system"]["axis"][0]["direction"] == "north"
    assert load(DATA / "labels.parquet")[0][0][0].bounds[0] == pytest.approx(16.36)


@pytest.mark.parametrize(
    "change, message",
    [
        (lambda geo: geo.pop("primary_column"), "'geo' metadata is malformed"),
        (lambda geo: geo["columns"].clear(), "'geo' metadata is malformed"),
        (lambda geo: geo["columns"]["geometry"].update(edges="spherical"), "'spherical' edges"),
        (lambda geo: geo.update(primary_column="nope", columns={"nope": {}}), "'nope'"),
    ],
)
def test_malformed_geoparquet_metadata(pq: Any, tmp_path: Path, change: Any, message: str) -> None:
    path = rewrite_geo(DATA / "labels.parquet", tmp_path / "bad.parquet", change)
    with pytest.raises(ValueError, match=message):
        load_vector_labels(path, "class")


def test_geoparquet_primary_column_must_exist(pq: Any, tmp_path: Path) -> None:
    def change(geo: Dict[str, Any]) -> None:
        geo["columns"]["nope"] = geo["columns"].pop("geometry")
        geo["primary_column"] = "nope"

    path = rewrite_geo(DATA / "labels.parquet", tmp_path / "bad.parquet", change)
    with pytest.raises(ValueError, match="geometry column 'nope' is not in the file"):
        load_vector_labels(path, "class")


def test_geoparquet_column_metadata_must_be_an_object(pq: Any, tmp_path: Path) -> None:
    path = rewrite_geo(
        DATA / "labels.parquet",
        tmp_path / "bad.parquet",
        lambda geo: geo["columns"].update(geometry="WKB"),
    )
    with pytest.raises(ValueError, match="'geo' metadata is malformed"):
        load_vector_labels(path, "class")


def test_parquet_without_geo_metadata_and_corrupt_parquet(pq: Any, tmp_path: Path) -> None:
    import pyarrow as pa

    plain = tmp_path / "plain.parquet"
    pq.write_table(pa.table({"a": [1, 2]}), plain)
    with pytest.raises(ValueError, match="without GeoParquet 'geo' metadata"):
        load_vector_labels(plain, "a")
    data = (DATA / "labels.parquet").read_bytes()
    broken = tmp_path / "broken.parquet"
    broken.write_bytes(data[: len(data) // 2])
    with pytest.raises(ValueError, match="cannot read it as a Parquet file"):
        load_vector_labels(broken, "class")
    broken.write_bytes(b"PAR1 definitely not parquet PAR1")
    with pytest.raises(ValueError, match="Parquet"):
        load_vector_labels(broken, "class")


def test_geoparquet_suffixes(pq: Any, tmp_path: Path) -> None:
    copy = tmp_path / "labels.geoparquet"
    shutil.copy(DATA / "labels.parquet", copy)
    assert_same(load(copy)[0], reference()[0])


# ── Dispatcher, hashes, listing ──────────────────────────────────────────────


def test_suffixes_are_case_insensitive_and_unknown_ones_are_errors(tmp_path: Path) -> None:
    upper = tmp_path / "LABELS.GPKG"
    shutil.copy(DATA / "labels.gpkg", upper)
    assert_same(load(upper)[0], reference()[0])
    json_copy = tmp_path / "labels.json"
    shutil.copy(GEOJSON, json_copy)
    assert_same(load(json_copy)[0], reference()[0])
    with pytest.raises(ValueError, match=r"file type must be one of .*\.gpkg.*\.shp"):
        load_vector_labels(tmp_path / "labels.kmz", None)


def test_geojson_and_kml_results_are_what_they_were() -> None:
    """The dispatcher hands GeoJSON to ``parse_geojson`` unchanged."""
    direct = parse_geojson(GEOJSON.read_bytes(), "class", None, points=True)
    via = load_vector_labels(GEOJSON, "class", points=True)
    assert [(g.wkb, c) for g, c in direct[0]] == [(g.wkb, c) for g, c in via[0]]
    assert direct[1] == via[1]
    broken = GEOJSON.parent / "missing.geojson"
    with pytest.raises(FileNotFoundError):
        load_vector_labels(broken, None)


def test_invalid_geojson_is_a_value_error(tmp_path: Path) -> None:
    bad = tmp_path / "bad.geojson"
    bad.write_text("{not json")
    with pytest.raises(ValueError, match="not valid GeoJSON"):
        load_vector_labels(bad, None)


def test_invalid_utf8_geojson_is_a_value_error(tmp_path: Path) -> None:
    bad = tmp_path / "bad.geojson"
    bad.write_bytes(b'{"type": "FeatureCollection", "features": [], "x": "\xff\xfe"}')
    with pytest.raises(ValueError, match="not valid UTF-8"):
        load_vector_labels(bad, None)


def test_the_wizard_lists_attributes_of_vector_tables_only() -> None:
    with pytest.raises(ValueError, match="listed by the caller"):
        vector_attributes(GEOJSON)
    assert vector_layers(GEOJSON) == [] and vector_layers(DATA / "polygons.shp") == []


def test_label_file_hash(tmp_path: Path) -> None:
    assert label_file_sha256(GEOJSON) == hashlib.sha256(GEOJSON.read_bytes()).hexdigest()
    assert (
        label_file_sha256(DATA / "labels.gpkg")
        == hashlib.sha256((DATA / "labels.gpkg").read_bytes()).hexdigest()
    )
    path = copy_shapefile(tmp_path)
    before = label_file_sha256(path)
    assert before == label_file_sha256(path)
    for suffix in (".dbf", ".prj", ".cpg", ".shx", ".shp"):
        edited = tmp_path / "edited"
        shutil.rmtree(edited, ignore_errors=True)
        edited.mkdir()
        target = copy_shapefile(edited)
        file = target.with_suffix(suffix)
        file.write_bytes(file.read_bytes() + b" ")
        assert label_file_sha256(target) != before, suffix


def test_attribute_listing_for_the_wizard(pq: Any) -> None:
    for name in ("labels.gpkg", "polygons.shp", "labels.parquet"):
        columns = vector_attributes(DATA / name)
        assert set(columns) == {"class", "rank", "score"}, name
    assert set(label_fields(DATA / "labels.gpkg")) == {"class", "rank", "score"}
    fields = label_fields(DATA / "polygons.shp")
    assert fields["class"][0] in {"building", "farmland", "forêt"}
    assert set(label_fields(DATA / "labels.parquet")) == {"class", "rank", "score"}
    assert set(label_fields(DATA / "labels_2layers.gpkg", layer="landuse")) == {
        "class",
        "rank",
        "score",
    }
    with pytest.raises(ValueError, match="2 layers"):
        label_fields(DATA / "labels_2layers.gpkg")


# ── Config ───────────────────────────────────────────────────────────────────


@pytest.mark.parametrize(
    "name",
    ["a.geojson", "a.json", "a.kml", "a.gpkg", "a.shp", "a.parquet", "a.geoparquet", "A.GPKG"],
)
def test_config_accepts_every_vector_suffix(name: str) -> None:
    assert LabelsConfig.model_validate({"path": name}).path == Path(name)


def test_config_rejects_other_suffixes_with_the_supported_list() -> None:
    with pytest.raises(ValidationError, match=r"\.gpkg, \.shp, \.parquet or \.geoparquet.*KMZ"):
        LabelsConfig.model_validate({"path": "a.kmz"})
    with pytest.raises(ValidationError, match="is a raster"):
        LabelsConfig.model_validate({"path": "a.tif"})


def test_layer_setting() -> None:
    config = LabelsConfig.model_validate({"path": "a.gpkg", "layer": "roads"})
    assert config.layer == "roads"
    assert config.model_dump(mode="json")["layer"] == "roads"
    # Unset, it is left out, so manifests and resume checks of other formats do not change.
    assert "layer" not in LabelsConfig.model_validate({"path": "a.gpkg"}).model_dump(mode="json")
    assert "layer" not in LabelsConfig.model_validate({"path": "a.geojson"}).model_dump()
    assert "layer" not in config.model_dump(mode="json", exclude={"layer"})
    with pytest.raises(ValidationError, match="picks a table of a GeoPackage"):
        LabelsConfig.model_validate({"path": "a.shp", "layer": "roads"})
    with pytest.raises(ValidationError, match="must not be empty"):
        LabelsConfig.model_validate({"path": "a.gpkg", "layer": " "})


# ── CLI ──────────────────────────────────────────────────────────────────────

REGION = {"west": 16.358, "south": 48.188, "east": 16.392, "north": 48.206}


def write_config(
    tmp_path: Path,
    labels: Dict[str, Any],
    url: str = "http://127.0.0.1:9/{z}/{x}/{y}.png",
    **extra: Any,
) -> Path:
    config = {
        "region": REGION,
        "imagery": {"type": "xyz", "zoom": 16, "url_template": url},
        "labels": labels,
        "sampler": {"patch_size": 96, "stride": 96, "edge_strategy": "pad"},
        "writer": {"staging_dir": str(tmp_path / "out")},
        **extra,
    }
    path = tmp_path / "mapcv.yaml"
    path.write_text(yaml.safe_dump(config), encoding="utf-8")
    return path


def test_plan_reports_a_corrupt_label_file_without_a_traceback(tmp_path: Path) -> None:
    bad = tmp_path / "bad.gpkg"
    bad.write_bytes(b"SQLite format 3\x00" + bytes(200))
    result = runner.invoke(app, ["plan", str(write_config(tmp_path, {"path": str(bad)}))])
    assert result.exit_code == 1
    assert "Cannot plan this config" in result.output and "GeoPackage" in result.output
    assert "Traceback" not in result.output


def test_cli_errors_name_the_fix_for_each_problem(tmp_path: Path) -> None:
    layers = write_config(tmp_path, {"path": str(DATA / "labels_2layers.gpkg")})
    result = runner.invoke(app, ["plan", str(layers)])
    assert result.exit_code == 1
    assert "buildings, landuse" in flat(result.output) or "buildings,landuse" in flat(result.output)
    assert "labels.layer" in flat(result.output) and "Traceback" not in result.output

    noprj = copy_shapefile(tmp_path, skip=[".prj"])
    result = runner.invoke(app, ["generate", str(write_config(tmp_path, {"path": str(noprj)}))])
    assert result.exit_code == 1
    assert "CRS is unknown" in flat(result.output).replace("CRSisunknown", "CRS is unknown")
    assert "Traceback" not in result.output


def test_missing_pyarrow_in_the_cli_shows_the_install_command(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setitem(sys.modules, "pyarrow", None)
    monkeypatch.setitem(sys.modules, "pyarrow.parquet", None)
    config = write_config(tmp_path, {"path": str(DATA / "labels.parquet")})
    result = runner.invoke(app, ["plan", str(config)])
    assert result.exit_code == 1
    assert "pipinstall'mapcv[parquet]'" in flat(result.output)  # not eaten as a markup tag
    assert "Traceback" not in result.output


def test_validate_and_plan_accept_the_new_formats(pq: Any, tmp_path: Path) -> None:
    for name, layer in (
        ("labels.gpkg", None),
        ("labels_2layers.gpkg", "buildings"),
        ("labels.parquet", None),
        ("labels_32633.parquet", None),
    ):
        labels: Dict[str, Any] = {"path": str(DATA / name), "label_field": "class"}
        if layer:
            labels["layer"] = layer
        config = write_config(tmp_path, labels)
        assert runner.invoke(app, ["validate", str(config)]).exit_code == 0, name
        result = runner.invoke(app, ["plan", str(config)])
        assert result.exit_code == 0, (name, result.output)
        assert "polygon(s)" in flat(result.output)
    shp = write_config(tmp_path, {"path": str(DATA / "polygons.shp"), "label_field": "class"})
    result = runner.invoke(app, ["plan", str(shp)])
    # The table wraps lines at the terminal's width: drop its borders and spaces first.
    text = re.sub(r"[│╭╮╰╯─\s]", "", result.output)
    assert result.exit_code == 0 and re.search(r"(?<!\d)4polygon\(s\)", text), result.output


def test_validate_shows_the_layer(tmp_path: Path) -> None:
    config = write_config(tmp_path, {"path": str(DATA / "labels_2layers.gpkg"), "layer": "landuse"})
    result = runner.invoke(app, ["validate", str(config)])
    assert result.exit_code == 0 and "layer:landuse" in flat(result.output)


def test_wizard_reads_a_geopackage_and_asks_for_its_layer(tmp_path: Path) -> None:
    out = tmp_path / "mapcv.yaml"
    path = str(DATA / "labels_2layers.gpkg")
    answers = (
        "\n".join(["esri", path, "landuse", "17", "y", "class", "", "256", "./ds", "y"]) + "\n"
    )
    result = runner.invoke(app, ["init", str(out), "--interactive"], input=answers)
    assert result.exit_code == 0, result.output
    config = MapcvConfig.from_yaml(out)
    assert isinstance(config.labels, LabelsConfig)
    assert (config.labels.layer, config.labels.label_field) == ("landuse", "class")
    assert config.labels.first_path is not None
    assert config.labels.first_path.name == "labels_2layers.gpkg"
    assert config.region.west == pytest.approx(16.36, abs=1e-4)


def test_wizard_area_from_a_shapefile_and_parquet(pq: Any, tmp_path: Path) -> None:
    for name in ("polygons.shp", "labels.parquet", "polygons_32633.shp"):
        out = tmp_path / f"{name}.yaml"
        answers = "\n".join(["esri", str(DATA / name), "17", "y", "class", "", "256", "./ds", "y"])
        result = runner.invoke(app, ["init", str(out), "--interactive"], input=answers + "\n")
        assert result.exit_code == 0, (name, result.output)
        config = MapcvConfig.from_yaml(out)
        assert config.region.west == pytest.approx(16.36, abs=1e-4), name
        assert isinstance(config.labels, LabelsConfig) and config.labels.label_field == "class"


def test_wizard_asks_for_the_layer_of_a_separate_label_file(tmp_path: Path) -> None:
    out = tmp_path / "mapcv.yaml"
    path = str(DATA / "labels_2layers.gpkg")
    area = "16.358,48.188,16.392,48.206"
    steps = ["esri", area, "17", path, "buildings", "class", "", "256", "./ds", "y"]
    result = runner.invoke(app, ["init", str(out), "--interactive"], input="\n".join(steps) + "\n")
    assert result.exit_code == 0, result.output
    config = MapcvConfig.from_yaml(out)
    assert isinstance(config.labels, LabelsConfig) and config.labels.layer == "buildings"


def test_wizard_helpers_survive_unreadable_label_files(tmp_path: Path) -> None:
    from mapcv.cli import _ask_label_field, _ask_layer

    bad = tmp_path / "bad.gpkg"
    bad.write_text("not sqlite")
    assert _ask_layer(bad) is None  # the file is read next, which says why it cannot be
    assert _ask_layer(DATA / "labels.gpkg") is None  # one layer: nothing to ask
    noprj = copy_shapefile(tmp_path, skip=[".prj"])
    assert _ask_label_field(noprj) is None  # reported, and the config is written for editing


def test_label_fields_of_a_kml_file(tmp_path: Path) -> None:
    kml = tmp_path / "a.kml"
    kml.write_text(
        '<?xml version="1.0"?><kml xmlns="http://www.opengis.net/kml/2.2"><Document><Placemark>'
        '<ExtendedData><SchemaData><SimpleData name="kind">roof</SimpleData></SchemaData>'
        "</ExtendedData><Polygon><outerBoundaryIs><LinearRing><coordinates>"
        "0,0,0 0,1,0 1,1,0 0,0,0</coordinates></LinearRing></outerBoundaryIs></Polygon>"
        "</Placemark></Document></kml>",
        encoding="utf-8",
    )
    assert label_fields(kml) == {"kind": ["roof"]}


def test_wizard_area_prompt_reports_an_unreadable_file_and_asks_again(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    noprj = copy_shapefile(tmp_path, skip=[".prj"])
    answers = iter([str(noprj), "16.0,48.0,16.1,48.1"])
    monkeypatch.setattr("mapcv.cli.Prompt.ask", lambda *a, **k: next(answers))
    box_, path, layer = _ask_bbox_or_file()
    assert box_ == (16.0, 48.0, 16.1, 48.1) and path is None and layer is None


# ── Fuzzing: damaged files error, they do not crash or hang ──────────────────


def damaged(data: bytes, rng: random.Random) -> bytes:
    mode = rng.randrange(3)
    if mode == 0:
        return data[: rng.randrange(len(data))]
    mutable = bytearray(data)
    for _ in range(rng.randrange(1, 8)):
        mutable[rng.randrange(len(mutable))] = rng.randrange(256)
    return bytes(mutable) if mode == 1 else bytes(mutable[: rng.randrange(1, len(mutable))])


@pytest.mark.parametrize("suffix", [".shp", ".shx", ".dbf", ".prj", ".cpg"])
def test_damaged_shapefiles_never_crash(tmp_path: Path, suffix: str) -> None:
    rng = random.Random(1234)
    for trial in range(60):
        directory = tmp_path / f"trial{trial}"
        directory.mkdir()
        path = copy_shapefile(directory)
        victim = path.with_suffix(suffix)
        victim.write_bytes(damaged(victim.read_bytes(), rng))
        try:
            load_vector_labels(path, "class")
        except ValueError:
            pass


def test_damaged_geopackages_never_crash(tmp_path: Path) -> None:
    rng = random.Random(99)
    source = (DATA / "labels.gpkg").read_bytes()
    for trial in range(60):
        path = tmp_path / f"d{trial}.gpkg"
        path.write_bytes(damaged(source, rng))
        try:
            load_vector_labels(path, "class")
        except ValueError:
            pass


def test_damaged_geoparquet_never_crashes(pq: Any, tmp_path: Path) -> None:
    rng = random.Random(7)
    source = (DATA / "labels.parquet").read_bytes()
    for trial in range(60):
        path = tmp_path / f"d{trial}.parquet"
        path.write_bytes(damaged(source, rng))
        try:
            load_vector_labels(path, "class")
        except ValueError:
            pass


# ── End to end: the same dataset from every format ───────────────────────────


class _NoiseTiles(BaseHTTPRequestHandler):
    """A deterministic noise tile per (z, x, y), so patches have pixel content."""

    def log_message(self, *args: object) -> None:
        pass

    def do_GET(self) -> None:  # noqa: N802 - http.server API
        z, x, y = (int(part) for part in self.path.strip("/").split(".")[0].split("/"))
        pixels = np.random.default_rng([z, x, y]).integers(1, 256, (256, 256, 3), dtype=np.uint8)
        buffer = BytesIO()
        Image.fromarray(pixels).save(buffer, "PNG")
        body = buffer.getvalue()
        self.send_response(200)
        self.send_header("Content-Type", "image/png")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)


@pytest.fixture(scope="module")
def tiles() -> Iterator[str]:
    server = ThreadingHTTPServer(("127.0.0.1", 0), _NoiseTiles)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield f"http://127.0.0.1:{server.server_address[1]}/{{z}}/{{x}}/{{y}}.png"
    finally:
        server.shutdown()
        thread.join()


class Runs:
    """Generates datasets from a local tile server, once per distinct configuration."""

    def __init__(self, directory: Path, url: str) -> None:
        self.directory = directory
        self.url = url
        self.done: Dict[str, Tuple[Path, Dict[str, Any]]] = {}

    def generate(self, labels: Dict[str, Any], **extra: Any) -> Tuple[Path, Dict[str, Any]]:
        key = json.dumps([labels, extra], sort_keys=True)
        if key not in self.done:
            directory = self.directory / f"run{len(self.done)}"
            directory.mkdir()
            config = write_config(directory, labels, self.url, **extra)
            data = yaml.safe_load(config.read_text(encoding="utf-8"))
            data["split"] = {"test_ratio": 0.25, "val_ratio": 0.25}
            run_generate(MapcvConfig.model_validate(data))
            staging = directory / "out"
            manifest = json.loads((staging / "manifest.json").read_text(encoding="utf-8"))
            self.done[key] = (staging, manifest)
        staging, manifest = self.done[key]
        return staging, json.loads(json.dumps(manifest))


@pytest.fixture(scope="module")
def runs(tmp_path_factory: pytest.TempPathFactory, tiles: str) -> Runs:
    return Runs(tmp_path_factory.mktemp("runs"), tiles)


def tree(staging: Path) -> Dict[str, bytes]:
    """Every file of a dataset but the manifest, by relative path."""
    return {
        path.relative_to(staging).as_posix(): path.read_bytes()
        for path in sorted(staging.rglob("*"))
        if path.is_file() and path.name != "manifest.json"
    }


E2E = [
    ("labels.gpkg", "labels.geojson", None),
    ("labels_32633.gpkg", "labels.geojson", None),
    ("labels_2layers.gpkg", "layer_buildings.geojson", "buildings"),
    ("labels.parquet", "labels.geojson", None),
    ("polygons.shp", "polygons.geojson", None),
    ("polygons_32633.shp", "polygons.geojson", None),
]


@pytest.mark.parametrize("task", ["segmentation", "detection", "instance"])
@pytest.mark.parametrize("name, equivalent, layer", E2E)
def test_generate_is_byte_identical_to_the_geojson_run(
    runs: Runs, name: str, equivalent: str, layer: Optional[str], task: str
) -> None:
    """Images, masks, boxes and splits equal those of the equivalent GeoJSON, byte for byte."""
    if name.endswith(".parquet"):
        pytest.importorskip("pyarrow")
    extra: Dict[str, Any] = {"task": task}
    if task == "detection":
        extra["detection"] = {"point_box_size": 24, "min_visible": 0.0}
    if task == "instance":
        extra["instance"] = {"min_visible": 0.0}
    base: Dict[str, Any] = {"label_field": "class"}
    own: Dict[str, Any] = {"path": str(DATA / name), **base}
    if layer:
        own["layer"] = layer
    expected, expected_manifest = runs.generate({"path": str(DATA / equivalent), **base}, **extra)
    got, manifest = runs.generate(own, **extra)

    assert expected_manifest["patches"], "the region must produce patches"
    assert tree(got) == tree(expected)
    files = tree(got)
    if task == "segmentation":  # the comparison is not vacuous: some masks hold classes
        masks = [Image.open(BytesIO(d)) for n, d in files.items() if n.startswith("Masks/")]
        classes = {int(v) for m in masks for v in np.unique(np.asarray(m))}
        assert classes - {0, 255}
    elif task == "detection":
        boxes = [data for name, data in files.items() if name.startswith("labels/")]
        assert any(boxes)
    else:  # instance: COCO annotations with masks
        coco = [d for n, d in files.items() if n.startswith("annotations/") and n.endswith(".json")]
        assert any(b'"iscrowd"' in data for data in coco)
    # The whole manifest agrees but the label file's hash (and the layer, when one is set).
    labels = manifest["target"]["labels"]
    assert labels.pop("sha256") != expected_manifest["target"]["labels"].pop("sha256")
    assert labels.pop("layer", None) == layer
    assert manifest == expected_manifest


def test_manifest_for_geojson_labels_has_no_layer_key(runs: Runs) -> None:
    _, manifest = runs.generate({"path": str(GEOJSON), "label_field": "class"})
    labels = manifest["target"]["labels"]
    assert set(labels) == {"label_field", "classes", "all_touched", "sha256"}
    assert labels["sha256"] == hashlib.sha256(GEOJSON.read_bytes()).hexdigest()
