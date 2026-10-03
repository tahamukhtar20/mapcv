"""Label parsing: KML and GeoJSON -> shapely geometries with class IDs."""

from __future__ import annotations

import json
import warnings
from math import pi
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import numpy.typing as npt
from shapely.geometry import MultiPolygon, Polygon as ShapelyPolygon
from shapely.geometry import shape
from shapely.geometry.base import BaseGeometry
from shapely.ops import transform as shapely_transform

from mapcv._mapcv_rs import parse_kml_rs

_RE: float = 6_378_137.0

GeomWithClass = Tuple[BaseGeometry, int]
ClassMap = Dict[str, int]

_POLYGON_TYPES = frozenset({"Polygon", "MultiPolygon"})


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


def _warn_skipped(source: str, unlabeled: int, unmapped: int, non_polygon: int) -> None:
    reasons = []
    if non_polygon:
        reasons.append(f"{non_polygon} without polygon geometry (points/lines)")
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
) -> Tuple[List[GeomWithClass], ClassMap]:
    ids, class_map = assign_class_ids(labels, label_field, classes)
    result = [(geom, class_id) for geom, class_id in zip(geometries, ids) if class_id != 0]
    unlabeled = sum(1 for label in labels if label is None) if label_field is not None else 0
    unmapped = sum(1 for label, class_id in zip(labels, ids) if label is not None and class_id == 0)
    _warn_skipped(source, unlabeled, unmapped, non_polygon)
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
) -> Tuple[List[GeomWithClass], ClassMap]:
    """Parse GeoJSON bytes into (geometry, class_id) pairs.

    Accepts a FeatureCollection or a single Feature in WGS-84 lon/lat.
    Polygons inside GeometryCollections are kept; points and lines are
    skipped with a warning. See :func:`assign_class_ids` for class IDs.
    Returns (geometries, class_map).
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

    geometries: List[BaseGeometry] = []
    labels: List[Optional[str]] = []
    non_polygon = 0
    for feat in features:
        geom_dict: Any = feat.get("geometry")
        if geom_dict is None:
            continue
        geom = _polygon_parts(shape(geom_dict))
        if geom is None:
            non_polygon += 1
            continue
        geometries.append(geom)
        props: Dict[str, Any] = feat.get("properties") or {}
        labels.append(_normalize_label(props.get(label_field)) if label_field else None)
    return _with_class_ids("GeoJSON", geometries, labels, label_field, classes, non_polygon)
