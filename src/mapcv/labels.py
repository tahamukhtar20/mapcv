"""Label parsing: vector files (GeoJSON, KML, GeoPackage, Shapefile, GeoParquet) ->
shapely geometries with class IDs."""

from __future__ import annotations

import hashlib
import html
import json
import math
import re
import warnings
from collections.abc import Sequence
from functools import lru_cache
from math import pi
from pathlib import Path
from typing import Any, cast

import numpy as np
import numpy.typing as npt
import shapely
from shapely.geometry import MultiPolygon, shape
from shapely.geometry import Polygon as ShapelyPolygon
from shapely.geometry.base import BaseGeometry

from mapcv import vector_files
from mapcv._mapcv_rs import parse_kml as _parse_kml_bytes

_RE: float = 6_378_137.0

GeomWithClass = tuple[BaseGeometry, int]
ClassMap = dict[str, int]

_POLYGON_TYPES = frozenset({"Polygon", "MultiPolygon"})
_POINT_TYPES = frozenset({"Point", "MultiPoint"})
_LINE_TYPES = frozenset({"LineString", "MultiLineString", "LinearRing"})

#: Buffer distances in metres: ``(line, point)``, either ``None`` for "do not buffer".
BufferDistances = tuple[float | None, float | None]


def _utm_epsg(lon: float, lat: float) -> int:
    """The WGS-84 UTM zone (EPSG code) holding a lon/lat position."""
    zone = min(60, max(1, int((lon + 180.0) // 6.0) + 1))
    return (32600 if lat >= 0.0 else 32700) + zone


@lru_cache(maxsize=128)
def _utm_transformers(epsg: int) -> tuple[Any, Any]:
    from pyproj import Transformer

    return (
        Transformer.from_crs("EPSG:4326", f"EPSG:{epsg}", always_xy=True),
        Transformer.from_crs(f"EPSG:{epsg}", "EPSG:4326", always_xy=True),
    )


def buffer_metres(geometry: BaseGeometry, distance: float) -> BaseGeometry:
    """``geometry`` (WGS-84 lon/lat) buffered by ``distance`` metres on the ground.

    The buffer is made in the UTM zone of the geometry's centroid, where a metre is a
    metre to within 0.1%, and the polygon is brought back to lon/lat.
    """
    centroid = geometry.centroid
    to_utm, to_lonlat = _utm_transformers(_utm_epsg(centroid.x, centroid.y))

    def project(transformer: Any) -> Any:
        def apply(coords: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
            x, y = transformer.transform(coords[:, 0], coords[:, 1])
            return np.column_stack((x, y))

        return apply

    projected = shapely.transform(geometry, project(to_utm))
    return shapely.transform(projected.buffer(distance, quad_segs=8), project(to_lonlat))


def _to_mercator(
    x: npt.NDArray[np.float64], y: npt.NDArray[np.float64]
) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
    # lat >= +/-90 clamps to +/-inf - matches Rust xy() guard
    mx: npt.NDArray[np.float64] = _RE * np.radians(x)
    raw: npt.NDArray[np.float64] = _RE * np.log(np.tan(pi / 4.0 + np.radians(y) / 2.0))
    my: npt.NDArray[np.float64] = np.where(y >= 90.0, np.inf, np.where(y <= -90.0, -np.inf, raw))
    return mx, my


def _mercator_coords(coords: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
    mx, my = _to_mercator(coords[:, 0], coords[:, 1])
    return np.column_stack((mx, my))


#: Latitude limit of Web Mercator, where the square world map ends (``atan(sinh(pi))``).
MERCATOR_MAX_LATITUDE: float = math.degrees(math.atan(math.sinh(math.pi)))


def _clip_to_mercator(array: npt.NDArray[np.object_]) -> npt.NDArray[np.object_]:
    """The geometries cut to latitudes +/-:data:`MERCATOR_MAX_LATITUDE`, beyond which
    Web Mercator is infinite (at the poles) and a geometry would not project. Geometries
    within the limit (nearly all) are returned as they are."""
    bounds = shapely.bounds(array)
    beyond = (bounds[:, 1] < -MERCATOR_MAX_LATITUDE) | (bounds[:, 3] > MERCATOR_MAX_LATITUDE)
    if not beyond.any():
        return array
    clipped = array.copy()
    for index in np.flatnonzero(beyond):
        west, _, east, _ = bounds[index]
        clipped[index] = shapely.clip_by_rect(
            array[index], west - 1.0, -MERCATOR_MAX_LATITUDE, east + 1.0, MERCATOR_MAX_LATITUDE
        )
    return clipped


def transform_to_mercator(geom: BaseGeometry) -> BaseGeometry:
    """Reproject a shapely geometry from WGS-84 to Web Mercator (EPSG:3857); Z is dropped.

    The part beyond latitude +/-85.0511 (where Web Mercator ends) is cut off first.
    """
    return transform_all_to_mercator([geom])[0]


def transform_all_to_mercator(geometries: Sequence[BaseGeometry]) -> list[BaseGeometry]:
    """Reproject many WGS-84 geometries to Web Mercator in one vectorized pass.

    Same values as :func:`transform_to_mercator` per geometry, but one shapely call
    for the whole list instead of one per geometry. Z coordinates are dropped.
    """
    if not geometries:
        return []

    array = np.empty(len(geometries), dtype=object)
    array[:] = list(geometries)
    return list(shapely.transform(_clip_to_mercator(array), _mercator_coords))


MAX_CLASS_ID = 255

# RFC 7946 GeoJSON is always WGS-84 lon/lat; these legacy `crs` names mean the same.
_WGS84_CRS_NAMES = frozenset(
    {
        "urn:ogc:def:crs:ogc:1.3:crs84",
        "urn:ogc:def:crs:epsg::4326",
        "epsg:4326",
        "crs84",
    }
)


# A label written as a number in ASCII digits ("3", "-2", "3.0", "1e3"); Python's float()
# and int() also take full-width and other Unicode digits and underscores, which a label
# file means as different text.
_NUMBER = re.compile(r"[+-]?(?:[0-9]+\.?[0-9]*|\.[0-9]+)(?:[eE][+-]?[0-9]+)?")
_INTEGER = re.compile(r"[0-9]+")


def _normalize_label(value: Any) -> str | None:
    """Return a stable string form of a label value, or ``None`` when missing."""
    if value is None:
        return None
    if isinstance(value, bool):
        return str(value).lower()
    if isinstance(value, float) and value.is_integer():
        return str(int(value))
    text = str(value).strip()
    if not text:
        return None
    if not _NUMBER.fullmatch(text):
        return text
    number = float(text)
    return str(int(number)) if number.is_integer() and "." in text else text


def _label_order(label: str) -> tuple[int, float, str]:
    """Sort key of a label: numbers by value, before every other label (by text)."""
    if _NUMBER.fullmatch(label):
        number = float(label)
        if math.isfinite(number):
            return (0, number, label)
    return (1, 0.0, label)


def assign_class_ids(
    labels: list[str | None],
    label_field: str | None,
    classes: ClassMap | None = None,
) -> tuple[list[int], ClassMap]:
    """Map raw label values to mask class IDs.

    Without ``label_field`` every geometry is class 1 and the class map is
    empty. With an explicit ``classes`` map, labels missing from it get 0
    (skipped). Otherwise labels that are all integers in 1..255 (in ASCII
    digits) are used as their own IDs, and any other labels get IDs 1..N in
    sorted order: numbers by value, then text. IDs do not depend on the order
    of features in the file.

    Returns one class ID per label (0 = skip) and the class map.

    Raises:
        ValueError: More distinct labels than fit in a ``uint8`` mask.
    """
    if label_field is None:
        return [1] * len(labels), {}

    present = sorted({label for label in labels if label is not None}, key=_label_order)
    if classes is not None:
        class_map = dict(classes)
    elif present and all(
        _INTEGER.fullmatch(label) and 1 <= int(label) <= MAX_CLASS_ID for label in present
    ):
        class_map = {label: int(label) for label in present}
    else:
        if len(present) > MAX_CLASS_ID:
            raise ValueError(
                f"labels.label_field '{label_field}' has {len(present)} distinct values; "
                f"masks support at most {MAX_CLASS_ID} classes. Map them with labels.classes."
            )
        class_map = {label: index for index, label in enumerate(present, start=1)}

    ids = [class_map.get(label, 0) if label is not None else 0 for label in labels]
    return ids, class_map


def _warn_skipped(
    source: str,
    unlabeled: int,
    unmapped: int,
    non_polygon: int,
    points: bool = False,
    invalid: int = 0,
) -> None:
    reasons = []
    if invalid:
        reasons.append(
            f"{invalid} with invalid coordinates (not a finite number, or a latitude beyond "
            "±90: the file must hold longitude, latitude)"
        )
    if non_polygon:
        kinds = "lines" if points else "points/lines"
        reasons.append(f"{non_polygon} without polygon geometry ({kinds})")
    if unlabeled:
        reasons.append(f"{unlabeled} without a label value")
    if unmapped:
        reasons.append(f"{unmapped} with a label not in labels.classes")
    if reasons:
        warnings.warn(f"{source}: skipped {', '.join(reasons)}.", UserWarning, stacklevel=4)


def _with_class_ids(
    source: str,
    geometries: list[BaseGeometry],
    labels: list[str | None],
    label_field: str | None,
    classes: ClassMap | None,
    non_polygon: int,
    points: bool = False,
    invalid: int = 0,
) -> tuple[list[GeomWithClass], ClassMap]:
    ids, class_map = assign_class_ids(labels, label_field, classes)
    result = [(geom, class_id) for geom, class_id in zip(geometries, ids) if class_id != 0]
    unlabeled = sum(1 for label in labels if label is None) if label_field is not None else 0
    unmapped = sum(1 for label, class_id in zip(labels, ids) if label is not None and class_id == 0)
    _warn_skipped(source, unlabeled, unmapped, non_polygon, points, invalid)
    return result, class_map


def _invalid_coordinates(geometries: Sequence[BaseGeometry | None]) -> set[int]:
    """The indices of the geometries with a vertex that is not a finite longitude and
    latitude: a NaN or infinite coordinate, or a latitude beyond +/-90. Such a geometry
    cannot be projected (it would vanish from the masks, or break buffering), so it is
    skipped and counted."""
    if not geometries:
        return set()
    array = np.empty(len(geometries), dtype=object)
    array[:] = list(geometries)
    coords, owner = shapely.get_coordinates(array, return_index=True)
    bad_vertex = ~np.isfinite(coords).all(axis=1) | (np.abs(coords[:, 1]) > 90.0)
    return set(np.unique(owner[bad_vertex]).tolist())


def _drop_invalid(
    geometries: list[BaseGeometry], labels: list[str | None]
) -> tuple[list[BaseGeometry], list[str | None], int]:
    """The geometries (and their labels) without :func:`_invalid_coordinates`, and how
    many were dropped."""
    bad = _invalid_coordinates(geometries)
    if not bad:
        return geometries, labels, 0
    kept = [index for index in range(len(geometries)) if index not in bad]
    return [geometries[i] for i in kept], [labels[i] for i in kept], len(bad)


def parse_kml(
    data: bytes,
    label_field: str | None = None,
    classes: ClassMap | None = None,
) -> tuple[list[GeomWithClass], ClassMap]:
    """Parse KML bytes into (geometry, class_id) pairs.

    Labels are read from ``<Data>`` or ``<SimpleData>`` fields named
    ``label_field``; see :func:`assign_class_ids` for how IDs are chosen.
    Points, lines, and unlabeled placemarks are skipped with a warning.
    Returns (geometries, class_map).
    """
    raw_polys, non_polygon = _parse_kml_bytes(data, label_field)
    geometries: list[BaseGeometry] = []
    labels: list[str | None] = []
    for poly_group, label in raw_polys:
        parts = [ShapelyPolygon(rings[0], rings[1:]) for rings in poly_group]
        geometries.append(parts[0] if len(parts) == 1 else MultiPolygon(parts))
        labels.append(_normalize_label(label))
    if label_field and raw_polys and all(label is None for _, label in raw_polys):
        fields = _kml_field_names(data)
        if label_field not in fields:
            raise ValueError(
                _missing_field_message(label_field, fields, "in the file")
                + " KML labels are read from <ExtendedData> <Data name=...> or <SimpleData "
                "name=...> fields, not from the placemark's <name>."
            )
    geometries, labels, invalid = _drop_invalid(geometries, labels)
    return _with_class_ids(
        "KML", geometries, labels, label_field, classes, non_polygon, invalid=invalid
    )


_KML_FIELD = re.compile(rb"<(?:[\w.-]+:)?(?:Simple)?Data\b[^>]*?\bname\s*=\s*(\"[^\"]*\"|'[^']*')")


def _kml_field_names(data: bytes) -> list[str]:
    """The names of the ``<Data>`` and ``<SimpleData>`` fields of a KML file, in order of
    first appearance (for the message about a ``label_field`` that is not one of them)."""
    names: dict[str, None] = {}
    for match in _KML_FIELD.finditer(data):
        raw = match.group(1)[1:-1].decode("utf-8", errors="replace")
        names[html.unescape(raw)] = None
    return list(names)


def _check_geojson_crs(obj: dict[str, Any]) -> None:
    crs = obj.get("crs")
    if not crs:
        return
    if isinstance(crs, dict):
        properties = crs.get("properties")
        name = str(properties.get("name", "") if isinstance(properties, dict) else "").lower()
    else:  # some tools write the name alone, such as "EPSG:4326"
        name = str(crs).lower()
    if name not in _WGS84_CRS_NAMES:
        raise ValueError(
            f"GeoJSON 'crs' {name or crs!r} is not supported; mapcv reads GeoJSON as WGS-84 "
            "longitude/latitude (RFC 7946). Reproject the file to EPSG:4326 first."
        )


def _collect_polygons(geom: BaseGeometry, parts: list[ShapelyPolygon]) -> None:
    if geom.geom_type == "Polygon":
        parts.append(cast(ShapelyPolygon, geom))
    elif geom.geom_type in ("MultiPolygon", "GeometryCollection"):
        for part in getattr(geom, "geoms", []):
            _collect_polygons(part, parts)


def _polygon_parts(geom: BaseGeometry) -> BaseGeometry | None:
    """The polygon(s) of a geometry, also from (nested) GeometryCollections, or ``None``."""
    if geom.geom_type in _POLYGON_TYPES:
        return geom
    if geom.geom_type == "GeometryCollection":
        parts: list[ShapelyPolygon] = []
        _collect_polygons(geom, parts)
        if parts:
            return parts[0] if len(parts) == 1 else MultiPolygon(parts)
    return None


_GEOJSON_GEOMETRY_TYPES = frozenset(
    {
        "Point",
        "MultiPoint",
        "LineString",
        "MultiLineString",
        "Polygon",
        "MultiPolygon",
        "GeometryCollection",
    }
)


def _json_kind(value: Any) -> str:
    """How a JSON value reads in a message: ``a list``, ``the number 5``, ..."""
    if isinstance(value, list):
        return "a list"
    if isinstance(value, str):
        return f"the text {value[:40]!r}"
    if isinstance(value, bool) or value is None:
        return json.dumps(value)
    if isinstance(value, (int, float)):
        return f"the number {value!r}"
    return "an object"


def _geojson_geometry(value: Any, where: str) -> BaseGeometry:
    """A shapely geometry from a GeoJSON geometry object, or a ``ValueError`` that says
    what is wrong with it."""
    if not isinstance(value, dict):
        raise ValueError(
            f"{where}: 'geometry' must be a GeoJSON geometry object or null, not "
            f"{_json_kind(value)}"
        )
    kind = value.get("type")
    if not isinstance(kind, str) or kind not in _GEOJSON_GEOMETRY_TYPES:
        shown = repr(kind) if isinstance(kind, str) else "missing"
        raise ValueError(
            f"{where}: the geometry type is {shown}; GeoJSON geometries are "
            f"{', '.join(sorted(_GEOJSON_GEOMETRY_TYPES))}"
        )
    if kind == "GeometryCollection":
        members = value.get("geometries")
        if not isinstance(members, list):
            raise ValueError(f"{where}: a GeometryCollection needs a 'geometries' list")
        parts = [_geojson_geometry(member, where) for member in members]
        return shapely.GeometryCollection(parts)
    if "coordinates" not in value:
        raise ValueError(f"{where}: the {kind} has no 'coordinates'")
    try:
        geometry: BaseGeometry = shape(value)
    except (TypeError, ValueError, KeyError, IndexError, AttributeError) as exc:
        raise ValueError(f"{where}: the {kind} coordinates are malformed ({exc})") from None
    except shapely.errors.ShapelyError as exc:
        raise ValueError(f"{where}: the {kind} coordinates are malformed ({exc})") from None
    return geometry


def _missing_field_message(label_field: str, fields: Sequence[str], where: str) -> str:
    """The error for a ``label_field`` that no feature has, with the fields there are."""
    shown = ", ".join(repr(name) for name in fields) or "none"
    return (
        f"labels.label_field '{label_field}' is not a property of any feature {where}. "
        f"The features have these properties: {shown}.{vector_files.did_you_mean(label_field, fields)}"
    )


def parse_geojson(
    data: bytes,
    label_field: str | None = None,
    classes: ClassMap | None = None,
    points: bool = False,
    buffer: BufferDistances | None = None,
) -> tuple[list[GeomWithClass], ClassMap]:
    """Parse GeoJSON bytes into (geometry, class_id) pairs.

    Accepts a FeatureCollection or a single Feature in WGS-84 lon/lat.
    Polygons inside GeometryCollections are kept; points and lines are
    skipped with a warning, except that ``points=True`` keeps Point and
    MultiPoint features (detection draws a box around them). See
    :func:`assign_class_ids` for class IDs. Returns (geometries, class_map).
    """
    obj: Any = json.loads(data.decode("utf-8"))
    if not isinstance(obj, dict):
        raise ValueError(
            f"Expected FeatureCollection or Feature, got {_json_kind(obj)}: a GeoJSON file "
            "holds one JSON object"
        )
    top_type = obj.get("type", "")
    if top_type == "FeatureCollection":
        features: Any = obj.get("features") or []
        if not isinstance(features, list):
            raise ValueError(
                f"the FeatureCollection's 'features' must be a list, not {_json_kind(features)}"
            )
    elif top_type == "Feature":
        features = [obj]
    else:
        raise ValueError(f"Expected FeatureCollection or Feature, got: {top_type!r}")
    _check_geojson_crs(obj)

    geometries: list[BaseGeometry | None] = []
    raw_labels: list[Any] = []
    fields: dict[str, None] = {}
    for index, feat in enumerate(features):
        where = f"feature {index}"
        if not isinstance(feat, dict):
            raise ValueError(f"{where} is not a GeoJSON Feature object but {_json_kind(feat)}")
        geom_dict: Any = feat.get("geometry")
        geometries.append(_geojson_geometry(geom_dict, where) if geom_dict is not None else None)
        props: Any = feat.get("properties")
        if props is None:
            props = {}
        elif not isinstance(props, dict):
            raise ValueError(f"{where}: 'properties' must be an object, not {_json_kind(props)}")
        fields.update(dict.fromkeys(props))
        raw_labels.append(props.get(label_field) if label_field else None)
    if label_field and features and label_field not in fields:
        raise ValueError(_missing_field_message(label_field, list(fields), "in the file"))
    return _polygon_features(
        "GeoJSON", geometries, raw_labels, label_field, classes, points, buffer=buffer
    )


def _polygon_features(
    source: str,
    geometries: Sequence[BaseGeometry | None],
    raw_labels: Sequence[Any],
    label_field: str | None,
    classes: ClassMap | None,
    points: bool,
    unreadable: set[int] | None = None,
    buffer: BufferDistances | None = None,
) -> tuple[list[GeomWithClass], ClassMap]:
    """Keep the polygon (and, with ``points``, point) features and give them class IDs.

    The one place where every vector format gets the same rules: a feature without a
    geometry is ignored; polygons inside a GeometryCollection are kept; with ``buffer``,
    lines and points become polygons (see :func:`buffer_metres`); other geometries are
    skipped and counted. ``unreadable`` lists features (by index) that are counted as
    having no polygon geometry because the format stored a kind mapcv cannot read.
    """
    line_m, point_m = buffer if buffer is not None else (None, None)
    kept: list[BaseGeometry] = []
    labels: list[str | None] = []
    non_polygon = 0
    invalid = _invalid_coordinates(geometries)
    for index, parsed in enumerate(geometries):
        if parsed is None:
            if unreadable is not None and index in unreadable:
                non_polygon += 1
            continue
        if index in invalid:
            continue
        geom = _polygon_parts(parsed)
        if geom is None and not parsed.is_empty:
            if line_m is not None and parsed.geom_type in _LINE_TYPES:
                geom = buffer_metres(parsed, line_m / 2.0)
            elif point_m is not None and parsed.geom_type in _POINT_TYPES:
                geom = buffer_metres(parsed, point_m / 2.0)
            elif points and parsed.geom_type in _POINT_TYPES:
                geom = parsed
        if geom is None:
            non_polygon += 1
            continue
        kept.append(geom)
        labels.append(_normalize_label(raw_labels[index]) if label_field else None)
    return _with_class_ids(
        source, kept, labels, label_field, classes, non_polygon, points, len(invalid)
    )


#: File suffixes (lower case) of the vector label formats, by format.
VECTOR_LABEL_SUFFIXES = frozenset(
    {".geojson", ".json", ".kml", ".gpkg", ".shp", ".parquet", ".geoparquet"}
)
_GEOJSON_SUFFIXES = frozenset({".geojson", ".json"})
_PARQUET_SUFFIXES = frozenset({".parquet", ".geoparquet"})


def label_suffix_hint(suffix: str) -> str:
    """A next step for a label file of an unsupported type, by its (lower-case) suffix."""
    if suffix == ".kmz":
        return " (a KMZ is a zipped KML: unzip it and use the .kml inside)"
    if suffix == ".zip":
        return " (unzip it first, then point at the .shp, .gpkg or .geojson inside)"
    return ""


def _format_name(path: Path) -> str:
    suffix = path.suffix.lower()
    if suffix not in VECTOR_LABEL_SUFFIXES:
        raise ValueError(
            f"Cannot read '{path.name}' as vector labels: the file type must be one of "
            f"{', '.join(sorted(VECTOR_LABEL_SUFFIXES))}{label_suffix_hint(suffix)}."
        )
    if suffix in _GEOJSON_SUFFIXES:
        return "GeoJSON"
    if suffix in _PARQUET_SUFFIXES:
        return "GeoParquet"
    return {".kml": "KML", ".gpkg": "GeoPackage", ".shp": "Shapefile"}[suffix]


def _read_table(
    path: Path, kind: str, layer: str | None, fields: Sequence[str] | None
) -> vector_files.VectorTable:
    if kind == "GeoPackage":
        return vector_files.read_gpkg(path, layer, fields)
    if kind == "Shapefile":
        return vector_files.read_shapefile(path, fields)
    return vector_files.read_geoparquet(path, fields)


def _check_layer(path: Path, kind: str, layer: str | None) -> None:
    if layer is not None and kind != "GeoPackage":
        raise ValueError(
            f"labels.layer applies to GeoPackage files only, not '{path.name}'; remove it."
        )


def load_vector_labels(
    path: Path,
    label_field: str | None = None,
    classes: ClassMap | None = None,
    points: bool = False,
    layer: str | None = None,
    buffer: BufferDistances | None = None,
) -> tuple[list[GeomWithClass], ClassMap]:
    """Read a vector label file of any supported format into ``(geometry, class_id)`` pairs.

    The format follows the file suffix: ``.geojson``/``.json``, ``.kml``, ``.gpkg``
    (GeoPackage), ``.shp`` (Shapefile) or ``.parquet``/``.geoparquet`` (GeoParquet, needs
    the ``mapcv[parquet]`` extra). Geometries are returned in WGS-84 longitude/latitude;
    GeoPackage, Shapefile and GeoParquet files in another CRS are reprojected with pyproj,
    and a file whose CRS is not stated is an error. Every format follows the same rules:

    * ``label_field`` names the attribute that holds the class (``None``: every feature
      is class 1); ``classes`` pins IDs; see :func:`assign_class_ids`.
    * Polygons and multipolygons are kept, including the polygons of a
      GeometryCollection. With ``points=True`` points and multipoints are kept too
      (detection draws a box around them); everything else is skipped, and counted in a
      ``UserWarning`` together with features that have no label or an unmapped one.
    * ``layer`` picks the table of a GeoPackage with more than one; it is an error for the
      other formats.
    * ``buffer`` ``(line, point)`` widths in metres turn lines into polygons that wide and
      points into discs that wide (not for KML, whose lines and points are not read).

    Args:
        path: The label file. Shapefiles are found with their ``.dbf``, ``.prj`` and
            ``.cpg`` next to them.
        label_field: Attribute holding the class, or ``None``.
        classes: Optional label-to-ID map.
        points: Keep point features (KML points are never read).
        layer: GeoPackage table name.

    Returns:
        ``(geometries, class_map)``.

    Raises:
        ValueError: An unsupported or corrupt file, an unknown ``layer`` or
            ``label_field``, an unknown CRS, or a missing optional dependency.
        FileNotFoundError: The file does not exist.
    """
    kind = _format_name(path)
    _check_layer(path, kind, layer)
    if kind in ("GeoJSON", "KML"):
        data = path.read_bytes()
        try:
            if kind == "KML":
                if buffer is not None:
                    raise ValueError(
                        f"{path.name}: buffering needs line and point features, which mapcv "
                        "does not read from KML; convert the file to GeoJSON or GeoPackage"
                    )
                return parse_kml(data, label_field, classes)
            return parse_geojson(data, label_field, classes, points=points, buffer=buffer)
        except UnicodeDecodeError as exc:
            raise ValueError(f"{path.name} is not valid UTF-8 text ({exc}).") from exc
        except json.JSONDecodeError as exc:
            raise ValueError(
                f"{path.name} is not valid GeoJSON ({exc}). Check that the file is complete."
            ) from exc
        except ValueError as exc:
            if str(exc).startswith(path.name):
                raise
            raise ValueError(f"{path.name}: {exc}") from None
    fields = [label_field] if label_field else []
    table = _read_table(path, kind, layer, fields)
    raw_labels: list[Any] = table.columns[label_field] if label_field else []
    if not label_field:
        raw_labels = [None] * len(table.geometries)
    return _polygon_features(
        kind, table.geometries, raw_labels, label_field, classes, points, table.unreadable, buffer
    )


def vector_layers(path: Path) -> list[str]:
    """Layer names of a GeoPackage; an empty list for every other format.

    Raises:
        ValueError: A ``.gpkg`` that cannot be read.
    """
    if path.suffix.lower() == ".gpkg":
        return vector_files.gpkg_layer_names(path)
    return []


def vector_attributes(path: Path, layer: str | None = None) -> dict[str, list[Any]]:
    """Every attribute column of a GeoPackage, Shapefile or GeoParquet file, by name.

    For the wizard's field listing: the raw values of each column, in feature order.

    Raises:
        ValueError: As :func:`load_vector_labels`; also for GeoJSON and KML, which the
            caller reads itself.
    """
    kind = _format_name(path)
    _check_layer(path, kind, layer)
    if kind in ("GeoJSON", "KML"):
        raise ValueError(f"{kind} attributes are listed by the caller.")
    return _read_table(path, kind, layer, None).columns


def label_file_sha256(path: Path) -> str:
    """SHA-256 of a label file, to tell a resumed run that the labels changed.

    For a Shapefile the hash also covers the ``.dbf``, ``.prj`` and ``.cpg`` next to it
    (each prefixed by its suffix), since editing any of them changes the labels. For the
    other formats it is the hash of the file's bytes.
    """
    digest = hashlib.sha256()
    files = vector_files.shapefile_files(path) if path.suffix.lower() == ".shp" else [path]
    for index, file in enumerate(files):
        if index:
            digest.update(file.suffix.lower().encode("ascii", errors="replace"))
        with file.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1 << 20), b""):
                digest.update(chunk)
    return digest.hexdigest()
