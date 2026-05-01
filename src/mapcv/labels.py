"""Label parsing: KML and GeoJSON -> shapely geometries with class IDs."""

from __future__ import annotations

import json
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
    result: BaseGeometry = shapely_transform(_to_mercator, geom)
    return result


def parse_kml(
    data: bytes,
    label_field: Optional[str] = None,
) -> Tuple[List[GeomWithClass], ClassMap]:
    """Parse KML bytes into (geometry, class_id) pairs.

    Points, lines, and empty placemarks are skipped. If label_field is None
    all polygons get class 1. Returns (geometries, class_map).
    """
    raw_polys, raw_class_map = parse_kml_rs(data, label_field)
    class_map: ClassMap = {k: int(v) for k, v in raw_class_map.items()}
    result: List[GeomWithClass] = []
    for poly_group, class_id in raw_polys:
        if class_id == 0:
            continue
        if len(poly_group) == 1:
            rings = poly_group[0]
            geom: BaseGeometry = ShapelyPolygon(rings[0], rings[1:])
        else:
            parts = [ShapelyPolygon(rings[0], rings[1:]) for rings in poly_group]
            geom = MultiPolygon(parts)
        result.append((geom, int(class_id)))
    return result, class_map


def parse_geojson(
    data: bytes,
    label_field: Optional[str] = None,
) -> Tuple[List[GeomWithClass], ClassMap]:
    """Parse GeoJSON bytes into (geometry, class_id) pairs.

    Accepts FeatureCollection or a single Feature. If label_field is None
    all polygons get class 1. Returns (geometries, class_map).
    """
    obj: Any = json.loads(data.decode("utf-8"))
    top_type: str = obj.get("type", "")
    if top_type == "FeatureCollection":
        features: List[Any] = obj.get("features") or []
    elif top_type == "Feature":
        features = [obj]
    else:
        raise ValueError(f"Expected FeatureCollection or Feature, got: {top_type!r}")

    class_map: ClassMap = {}
    result: List[GeomWithClass] = []

    for feat in features:
        geom_dict: Any = feat.get("geometry")
        if geom_dict is None:
            continue
        geom: BaseGeometry = shape(geom_dict)
        if geom.geom_type not in _POLYGON_TYPES:
            continue
        if label_field is None:
            result.append((geom, 1))
        else:
            props: Dict[str, Any] = feat.get("properties") or {}
            label_val: Any = props.get(label_field)
            if label_val is None:
                continue
            label = str(label_val)
            if label not in class_map:
                class_map[label] = len(class_map) + 1
            result.append((geom, class_map[label]))

    return result, class_map
