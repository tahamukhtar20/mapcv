"""Areas of interest (``region.path``): restrict patches to polygons and name regions.

The polygons are brought into the imagery's pixel grid once. A patch is kept when its
square overlaps a polygon with a positive area (touching an edge is not enough), and
its region is the one whose polygons cover most of it (ties go to the region listed
first in the file).
"""

from __future__ import annotations

from typing import Any, Dict, List, Sequence, Tuple, cast

import numpy as np
import numpy.typing as npt
import shapely

from mapcv.config import RegionConfig, area_polygons
from mapcv.imagery import Transform, transform_geometry_to_crs
from mapcv.labels import label_file_sha256, transform_all_to_mercator


def _to_pixels(
    geometries: npt.NDArray[np.object_], transform: Transform
) -> npt.NDArray[np.object_]:
    a, b, c, d, e, f = transform
    det = a * e - b * d
    if det == 0:
        raise ValueError("the imagery transform is not invertible")

    def inverse(coords: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
        x = coords[:, 0] - c
        y = coords[:, 1] - f
        return np.column_stack(((e * x - b * y) / det, (a * y - d * x) / det))

    return cast(npt.NDArray[np.object_], shapely.transform(geometries, inverse))


class AreaOfInterest:
    """The polygons of ``region.path`` on an imagery grid (``crs``, ``transform``)."""

    def __init__(self, region: RegionConfig, crs: str, transform: Transform) -> None:
        if region.path is None:
            raise ValueError("an area of interest needs region.path")
        self._region = region
        polygons = area_polygons(region.path, region.name_field, region.layer)
        lonlat = [geometry for geometry, _ in polygons]
        if crs.upper() == "EPSG:3857":
            world = transform_all_to_mercator(lonlat)
        else:
            world = [transform_geometry_to_crs(geometry, crs) for geometry in lonlat]
        array = np.empty(len(world), dtype=object)
        array[:] = world
        self._pixels = _to_pixels(array, transform)
        self._names = [name for _, name in polygons]
        # First appearance of each name: ties between regions go to the earlier one.
        self._order = {name: index for index, name in reversed(list(enumerate(self._names)))}
        self._tree = shapely.STRtree(self._pixels)

    def _boxes(self, anchors: Sequence[Tuple[int, int]], size: int) -> npt.NDArray[np.object_]:
        rows = np.fromiter((row for row, _ in anchors), dtype=np.float64, count=len(anchors))
        cols = np.fromiter((col for _, col in anchors), dtype=np.float64, count=len(anchors))
        return cast(npt.NDArray[np.object_], shapely.box(cols, rows, cols + size, rows + size))

    def keep(self, anchors: List[Tuple[int, int]], size: int) -> List[Tuple[int, int]]:
        """The anchors whose ``size`` x ``size`` patch overlaps a polygon, in order."""
        if not anchors:
            return []
        boxes = self._boxes(anchors, size)
        box_index, polygon_index = self._tree.query(boxes, predicate="intersects")
        overlap = shapely.area(shapely.intersection(boxes[box_index], self._pixels[polygon_index]))
        kept = set(box_index[overlap > 0].tolist())
        return [anchor for index, anchor in enumerate(anchors) if index in kept]

    def region_of(self, row: int, col: int, size: int) -> str:
        """The name of the region covering most of the patch at ``(row, col)``."""
        patch = shapely.box(col, row, col + size, row + size)
        areas: Dict[str, float] = {}
        for index in self._tree.query(patch, predicate="intersects"):
            area = float(shapely.area(shapely.intersection(patch, self._pixels[index])))
            name = self._names[int(index)]
            areas[name] = areas.get(name, 0.0) + area
        if not areas:
            return ""
        return max(areas, key=lambda name: (areas[name], -self._order[name]))

    def record(self) -> Dict[str, Any]:
        """The manifest's ``region`` record: the file's hash and how regions are named."""
        assert self._region.path is not None
        record: Dict[str, Any] = {"aoi_sha256": label_file_sha256(self._region.path)}
        if self._region.name_field is not None:
            record["name_field"] = self._region.name_field
        if self._region.layer is not None:
            record["layer"] = self._region.layer
        return record


def column_clusters(group: List[Tuple[int, int]], patch_size: int) -> List[List[Tuple[int, int]]]:
    """``group`` split where its columns leave a gap wider than two patches, so a chunk
    of far-apart polygons reads several small windows instead of one wide one. The
    anchors keep their order within each cluster; clusters go left to right."""
    columns = sorted({col for _, col in group})
    cluster_of: Dict[int, int] = {}
    cluster = 0
    for previous, column in zip([None, *columns], columns):
        if previous is not None and column - previous > 2 * patch_size:
            cluster += 1
        cluster_of[column] = cluster
    clusters: List[List[Tuple[int, int]]] = [[] for _ in range(cluster + 1)]
    for anchor in group:
        clusters[cluster_of[anchor[1]]].append(anchor)
    return clusters
