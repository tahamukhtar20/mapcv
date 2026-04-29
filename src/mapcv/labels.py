"""Label parsing: KML and GeoJSON → shapely geometries with class IDs."""

from __future__ import annotations

import io
import json
from math import pi
from typing import Any, Dict, Iterator, List, Optional, Tuple

import fastkml
import fastkml.data
import fastkml.features
import numpy as np
import numpy.typing as npt
from shapely.geometry import shape
from shapely.geometry.base import BaseGeometry
from shapely.ops import transform as shapely_transform

_RE: float = 6_378_137.0

GeomWithClass = Tuple[BaseGeometry, int]
ClassMap = Dict[str, int]

_POLYGON_TYPES = frozenset({"Polygon", "MultiPolygon"})


def _to_mercator(
    x: npt.NDArray[np.float64],
    y: npt.NDArray[np.float64],
) -> Tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
    # lat ≥ ±90 clamps to ±∞ — matches Rust xy() guard
    mx: npt.NDArray[np.float64] = _RE * np.radians(x)
    raw: npt.NDArray[np.float64] = _RE * np.log(np.tan(pi / 4.0 + np.radians(y) / 2.0))
    my: npt.NDArray[np.float64] = np.where(y >= 90.0, np.inf, np.where(y <= -90.0, -np.inf, raw))
    return mx, my


def transform_to_mercator(geom: BaseGeometry) -> BaseGeometry:
    result: BaseGeometry = shapely_transform(_to_mercator, geom)
    return result


def _label_from_placemark(pm: Any, label_field: str) -> Optional[str]:
    ext = pm.extended_data
    if ext is None:
        return None
    for el in ext.elements:
        if isinstance(el, fastkml.data.Data) and el.name == label_field:
            return str(el.value) if el.value is not None else None
    return None


def _iter_placemarks(features: Any) -> Iterator[Any]:
    for feat in features:
        if isinstance(feat, fastkml.features.Placemark):
            yield feat
        elif hasattr(feat, "features"):
            yield from _iter_placemarks(feat.features)


def parse_kml(
    data: bytes,
    label_field: Optional[str] = None,
) -> Tuple[List[GeomWithClass], ClassMap]:
    """Parse KML bytes into (geometry, class_id) pairs.

    Points, lines, and empty placemarks are skipped. If label_field is None
    all polygons get class 1. Returns (geometries, class_map).
    """
    k: Any = fastkml.KML.parse(io.BytesIO(data))
    class_map: ClassMap = {}
    result: List[GeomWithClass] = []

    for pm in _iter_placemarks(k.features):
        if pm.kml_geometry is None or pm.kml_geometry.geometry is None:
            continue
        geom: BaseGeometry = pm.kml_geometry.geometry
        if geom.geom_type not in _POLYGON_TYPES:
            continue
        if label_field is None:
            result.append((geom, 1))
        else:
            label = _label_from_placemark(pm, label_field)
            if label is None:
                continue
            if label not in class_map:
                class_map[label] = len(class_map) + 1
            result.append((geom, class_map[label]))

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
