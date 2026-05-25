"""Rasterize shapely polygons into a uint8 class mask."""

from __future__ import annotations

from typing import Iterable, List, Sequence, Tuple, cast

import numpy as np
import numpy.typing as npt
from shapely.geometry import MultiPolygon, Polygon
from shapely.geometry.base import BaseGeometry

from mapcv._mapcv_rs import rasterize as _rasterize_rs

Transform = Tuple[float, float, float, float, float, float]
Ring = List[Tuple[float, float]]
RingSet = List[Ring]


def _ring_coords(coords: Iterable[Sequence[float]]) -> Ring:
    return [(float(p[0]), float(p[1])) for p in coords]


def _polygon_to_rings(poly: Polygon) -> RingSet:
    return [_ring_coords(poly.exterior.coords)] + [
        _ring_coords(interior.coords) for interior in poly.interiors
    ]


def _flatten(geom: BaseGeometry, class_id: int) -> List[Tuple[RingSet, int]]:
    if isinstance(geom, Polygon):
        return [(_polygon_to_rings(geom), class_id)]
    if isinstance(geom, MultiPolygon):
        return [(_polygon_to_rings(p), class_id) for p in geom.geoms]
    return []


def rasterize(
    geometries: Sequence[Tuple[BaseGeometry, int]],
    out_shape: Tuple[int, int],
    transform: Transform,
    all_touched: bool = False,
) -> npt.NDArray[np.uint8]:
    """Burn (geometry, class_id) pairs into a uint8 mask.

    out_shape is (height, width). transform is a 6-tuple (a, b, c, d, e, f)
    mapping pixel (col, row) -> world (x, y), matching rasterio's Affine.
    Background pixels are 0; polygons in later list positions overwrite
    earlier ones. Non-polygon geometries are silently skipped.
    """
    height, width = out_shape
    polygons: List[Tuple[RingSet, int]] = []
    for geom, cid in geometries:
        cid_int = int(cid)
        if cid_int <= 0 or cid_int > 255:
            raise ValueError("class_id must be in 1..=255 (0 is reserved for background)")
        polygons.extend(_flatten(geom, cid_int))

    return cast(
        npt.NDArray[np.uint8],
        _rasterize_rs(polygons, height, width, transform, all_touched),
    )
