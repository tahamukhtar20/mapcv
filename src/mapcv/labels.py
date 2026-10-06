"""Label parsing: vector files (GeoJSON, KML, GeoPackage, Shapefile, GeoParquet) ->
shapely geometries with class IDs."""

from __future__ import annotations

import hashlib
import json
import warnings
from functools import lru_cache
from math import pi
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple

import numpy as np
import numpy.typing as npt
import shapely
from shapely.geometry import MultiPolygon, Polygon as ShapelyPolygon
from shapely.geometry import shape
from shapely.geometry.base import BaseGeometry
from shapely.ops import transform as shapely_transform

from mapcv import vector_files
from mapcv._mapcv_rs import parse_kml_rs

_RE: float = 6_378_137.0

GeomWithClass = Tuple[BaseGeometry, int]
ClassMap = Dict[str, int]

_POLYGON_TYPES = frozenset({"Polygon", "MultiPolygon"})
_POINT_TYPES = frozenset({"Point", "MultiPoint"})
_LINE_TYPES = frozenset({"LineString", "MultiLineString", "LinearRing"})

#: Buffer distances in metres: ``(line, point)``, either ``None`` for "do not buffer".
BufferDistances = Tuple[Optional[float], Optional[float]]


def _utm_epsg(lon: float, lat: float) -> int:
    """The WGS-84 UTM zone (EPSG code) holding a lon/lat position."""
    zone = min(60, max(1, int((lon + 180.0) // 6.0) + 1))
    return (32600 if lat >= 0.0 else 32700) + zone


@lru_cache(maxsize=128)
def _utm_transformers(epsg: int) -> Tuple[Any, Any]:
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
    x: npt.NDArray[np.float64],
    y: npt.NDArray[np.float64],
    z: Optional[npt.NDArray[np.float64]] = None,
) -> Tuple[npt.NDArray[np.float64], ...]:
    # lat >= +/-90 clamps to +/-inf - matches Rust xy() guard
    # z is passed by shapely.ops.transform for 3D geometries and returned unchanged
    mx: npt.NDArray[np.float64] = _RE * np.radians(x)
    raw: npt.NDArray[np.float64] = _RE * np.log(np.tan(pi / 4.0 + np.radians(y) / 2.0))
    my: npt.NDArray[np.float64] = np.where(y >= 90.0, np.inf, np.where(y <= -90.0, -np.inf, raw))
    if z is not None:
        return mx, my, z
    return mx, my


def transform_to_mercator(geom: BaseGeometry) -> BaseGeometry:
    """Reproject a shapely geometry from WGS-84 to Web Mercator (EPSG:3857)."""
    result: BaseGeometry = shapely_transform(_to_mercator, geom)
    return result


def transform_all_to_mercator(geometries: Sequence[BaseGeometry]) -> List[BaseGeometry]:
    """Reproject many WGS-84 geometries to Web Mercator in one vectorized pass.

    Same values as :func:`transform_to_mercator` per geometry, but one shapely call
    for the whole list instead of one per geometry. Z coordinates are dropped.
    """
    if not geometries:
        return []

    def project(coords: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
        mx, my = _to_mercator(coords[:, 0], coords[:, 1])
        return np.column_stack((mx, my))

    array = np.empty(len(geometries), dtype=object)
    array[:] = list(geometries)
    return list(shapely.transform(array, project))


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


def _normalize_label(value: Any) -> Optional[str]:
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
    try:
        number = float(text)
    except ValueError:
        return text
    return str(int(number)) if number.is_integer() and "." in text else text


def assign_class_ids(
    labels: List[Optional[str]],
    label_field: Optional[str],
    classes: Optional[ClassMap] = None,
) -> Tuple[List[int], ClassMap]:
    """Map raw label values to mask class IDs.

    Without ``label_field`` every geometry is class 1 and the class map is
    empty. With an explicit ``classes`` map, labels missing from it get 0
    (skipped). Otherwise labels that are all integers in 1..255 are used as
    their own IDs, and any other labels get IDs 1..N in sorted order, so IDs
    do not depend on the order of features in the file.

    Returns one class ID per label (0 = skip) and the class map.

    Raises:
        ValueError: More distinct labels than fit in a ``uint8`` mask.
    """
    if label_field is None:
        return [1] * len(labels), {}

    present = sorted({label for label in labels if label is not None})
    if classes is not None:
        class_map = dict(classes)
    elif present and all(label.isdigit() and 1 <= int(label) <= MAX_CLASS_ID for label in present):
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
    source: str, unlabeled: int, unmapped: int, non_polygon: int, points: bool = False
) -> None:
    reasons = []
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
    geometries: List[BaseGeometry],
    labels: List[Optional[str]],
    label_field: Optional[str],
    classes: Optional[ClassMap],
    non_polygon: int,
    points: bool = False,
) -> Tuple[List[GeomWithClass], ClassMap]:
    ids, class_map = assign_class_ids(labels, label_field, classes)
    result = [(geom, class_id) for geom, class_id in zip(geometries, ids) if class_id != 0]
    unlabeled = sum(1 for label in labels if label is None) if label_field is not None else 0
    unmapped = sum(1 for label, class_id in zip(labels, ids) if label is not None and class_id == 0)
    _warn_skipped(source, unlabeled, unmapped, non_polygon, points)
    return result, class_map


def parse_kml(
    data: bytes,
    label_field: Optional[str] = None,
    classes: Optional[ClassMap] = None,
) -> Tuple[List[GeomWithClass], ClassMap]:
    """Parse KML bytes into (geometry, class_id) pairs.

    Labels are read from ``<Data>`` or ``<SimpleData>`` fields named
    ``label_field``; see :func:`assign_class_ids` for how IDs are chosen.
    Points, lines, and unlabeled placemarks are skipped with a warning.
    Returns (geometries, class_map).
    """
    raw_polys, non_polygon = parse_kml_rs(data, label_field)
    geometries: List[BaseGeometry] = []
    labels: List[Optional[str]] = []
    for poly_group, label in raw_polys:
        parts = [ShapelyPolygon(rings[0], rings[1:]) for rings in poly_group]
        geometries.append(parts[0] if len(parts) == 1 else MultiPolygon(parts))
        labels.append(_normalize_label(label))
    return _with_class_ids("KML", geometries, labels, label_field, classes, non_polygon)


def _check_geojson_crs(obj: Any) -> None:
    crs = obj.get("crs")
    if not crs:
        return
    name = str((crs.get("properties") or {}).get("name", "")).lower()
    if name not in _WGS84_CRS_NAMES:
        raise ValueError(
            f"GeoJSON 'crs' {name or crs!r} is not supported; mapcv reads GeoJSON as WGS-84 "
            "longitude/latitude (RFC 7946). Reproject the file to EPSG:4326 first."
        )


def _polygon_parts(geom: BaseGeometry) -> Optional[BaseGeometry]:
    if geom.geom_type in _POLYGON_TYPES:
        return geom
    if geom.geom_type == "GeometryCollection":
        parts: List[ShapelyPolygon] = []
        for part in getattr(geom, "geoms", []):
            if part.geom_type == "Polygon":
                parts.append(part)
            elif part.geom_type == "MultiPolygon":
                parts.extend(part.geoms)
        if parts:
            return parts[0] if len(parts) == 1 else MultiPolygon(parts)
    return None


def parse_geojson(
    data: bytes,
    label_field: Optional[str] = None,
    classes: Optional[ClassMap] = None,
    points: bool = False,
    buffer: Optional[BufferDistances] = None,
) -> Tuple[List[GeomWithClass], ClassMap]:
    """Parse GeoJSON bytes into (geometry, class_id) pairs.

    Accepts a FeatureCollection or a single Feature in WGS-84 lon/lat.
    Polygons inside GeometryCollections are kept; points and lines are
    skipped with a warning, except that ``points=True`` keeps Point and
    MultiPoint features (detection draws a box around them). See
    :func:`assign_class_ids` for class IDs. Returns (geometries, class_map).
    """
    obj: Any = json.loads(data.decode("utf-8"))
    top_type: str = obj.get("type", "")
    if top_type == "FeatureCollection":
        features: List[Any] = obj.get("features") or []
    elif top_type == "Feature":
        features = [obj]
    else:
        raise ValueError(f"Expected FeatureCollection or Feature, got: {top_type!r}")
    _check_geojson_crs(obj)

    geometries: List[Optional[BaseGeometry]] = []
    raw_labels: List[Any] = []
    for feat in features:
        geom_dict: Any = feat.get("geometry")
        geometries.append(shape(geom_dict) if geom_dict is not None else None)
        props: Dict[str, Any] = feat.get("properties") or {}
        raw_labels.append(props.get(label_field) if label_field else None)
    return _polygon_features(
        "GeoJSON", geometries, raw_labels, label_field, classes, points, buffer=buffer
    )


def _polygon_features(
    source: str,
    geometries: Sequence[Optional[BaseGeometry]],
    raw_labels: Sequence[Any],
    label_field: Optional[str],
    classes: Optional[ClassMap],
    points: bool,
    unreadable: Optional[Set[int]] = None,
    buffer: Optional[BufferDistances] = None,
) -> Tuple[List[GeomWithClass], ClassMap]:
    """Keep the polygon (and, with ``points``, point) features and give them class IDs.

    The one place where every vector format gets the same rules: a feature without a
    geometry is ignored; polygons inside a GeometryCollection are kept; with ``buffer``,
    lines and points become polygons (see :func:`buffer_metres`); other geometries are
    skipped and counted. ``unreadable`` lists features (by index) that are counted as
    having no polygon geometry because the format stored a kind mapcv cannot read.
    """
    line_m, point_m = buffer if buffer is not None else (None, None)
    kept: List[BaseGeometry] = []
    labels: List[Optional[str]] = []
    non_polygon = 0
    for index, parsed in enumerate(geometries):
        if parsed is None:
            if unreadable is not None and index in unreadable:
                non_polygon += 1
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
    return _with_class_ids(source, kept, labels, label_field, classes, non_polygon, points)


#: File suffixes (lower case) of the vector label formats, by format.
VECTOR_LABEL_SUFFIXES = frozenset(
    {".geojson", ".json", ".kml", ".gpkg", ".shp", ".parquet", ".geoparquet"}
)
_GEOJSON_SUFFIXES = frozenset({".geojson", ".json"})
_PARQUET_SUFFIXES = frozenset({".parquet", ".geoparquet"})


def _format_name(path: Path) -> str:
    suffix = path.suffix.lower()
    if suffix not in VECTOR_LABEL_SUFFIXES:
        raise ValueError(
            f"Cannot read '{path.name}' as vector labels: the file type must be one of "
            f"{', '.join(sorted(VECTOR_LABEL_SUFFIXES))} (convert KMZ to KML first)."
        )
    if suffix in _GEOJSON_SUFFIXES:
        return "GeoJSON"
    if suffix in _PARQUET_SUFFIXES:
        return "GeoParquet"
    return {".kml": "KML", ".gpkg": "GeoPackage", ".shp": "Shapefile"}[suffix]


def _read_table(
    path: Path, kind: str, layer: Optional[str], fields: Optional[Sequence[str]]
) -> vector_files.VectorTable:
    if kind == "GeoPackage":
        return vector_files.read_gpkg(path, layer, fields)
    if kind == "Shapefile":
        return vector_files.read_shapefile(path, fields)
    return vector_files.read_geoparquet(path, fields)


def _check_layer(path: Path, kind: str, layer: Optional[str]) -> None:
    if layer is not None and kind != "GeoPackage":
        raise ValueError(
            f"labels.layer applies to GeoPackage files only, not '{path.name}'; remove it."
        )


def load_vector_labels(
    path: Path,
    label_field: Optional[str] = None,
    classes: Optional[ClassMap] = None,
    points: bool = False,
    layer: Optional[str] = None,
    buffer: Optional[BufferDistances] = None,
) -> Tuple[List[GeomWithClass], ClassMap]:
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
    fields = [label_field] if label_field else []
    table = _read_table(path, kind, layer, fields)
    raw_labels: List[Any] = table.columns[label_field] if label_field else []
    if not label_field:
        raw_labels = [None] * len(table.geometries)
    return _polygon_features(
        kind, table.geometries, raw_labels, label_field, classes, points, table.unreadable, buffer
    )


def vector_layers(path: Path) -> List[str]:
    """Layer names of a GeoPackage; an empty list for every other format.

    Raises:
        ValueError: A ``.gpkg`` that cannot be read.
    """
    if path.suffix.lower() == ".gpkg":
        return vector_files.gpkg_layer_names(path)
    return []


def vector_attributes(path: Path, layer: Optional[str] = None) -> Dict[str, List[Any]]:
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
