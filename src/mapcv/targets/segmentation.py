"""Semantic segmentation: polygon labels burned into per-patch class masks."""

from __future__ import annotations

import hashlib
import warnings
from typing import List, Optional, Tuple, cast

import numpy as np
import numpy.typing as npt
import shapely
from shapely.geometry import box

from mapcv._patching import MaskWindow, NullWindow
from mapcv.config import LabelsConfig
from mapcv.imagery import RasterMetadata, transform_geometry_to_crs
from mapcv.labels import (
    ClassMap,
    GeomWithClass,
    parse_geojson,
    parse_kml,
    transform_all_to_mercator,
)
from mapcv.manifest import TargetRecord
from mapcv.rasterizer import rasterize
from mapcv.targets.base import Transform, WindowTarget

LABELS_MISS_MESSAGE = (
    "no label polygon intersects the imagery extent, so every mask will be background. "
    "Check that labels are longitude/latitude (not swapped) and cover the configured region."
)


def _parse_labels(
    labels: LabelsConfig, data: bytes, destination_crs: str, points: bool = False
) -> Tuple[List[GeomWithClass], ClassMap]:
    """Label geometries in ``destination_crs`` with their class IDs, and the class map.

    ``points=True`` also keeps GeoJSON point features.
    """
    if labels.path.suffix.lower() == ".kml":
        raw, class_map = parse_kml(data, labels.label_field, labels.classes)
    else:
        raw, class_map = parse_geojson(data, labels.label_field, labels.classes, points=points)

    if destination_crs.upper() == "EPSG:3857":
        projected = transform_all_to_mercator([geometry for geometry, _ in raw])
        transformed = [(geometry, class_id) for geometry, (_, class_id) in zip(projected, raw)]
    else:
        transformed = [
            (transform_geometry_to_crs(geometry, destination_crs), class_id)
            for geometry, class_id in raw
        ]
    return transformed, class_map


def _check_ignore_index(ignore: Optional[int], class_map: ClassMap) -> None:
    """Fail when a class would get the mask value reserved for ignored pixels."""
    clashing = sorted(name for name, cid in class_map.items() if cid == ignore)
    if clashing:
        raise ValueError(
            f"class {clashing[0]!r} gets mask value {ignore}, which labels.ignore_index "
            "reserves for pixels without imagery; map it to another ID with labels.classes "
            "or set labels.ignore_index to a free value (or null)"
        )


def _raster_bounds(source: RasterMetadata) -> Tuple[float, float, float, float]:
    a, _, c, _, e, f = source.transform
    xs = (c, c + a * source.width)
    ys = (f, f + e * source.height)
    return min(xs), min(ys), max(xs), max(ys)


def _warn_if_labels_miss_raster(
    geometries: List[GeomWithClass], source: RasterMetadata, message: str = LABELS_MISS_MESSAGE
) -> None:
    if not geometries:
        return
    extent = box(*_raster_bounds(source))
    if not any(geometry.intersects(extent) for geometry, _ in geometries):
        # Attribute the warning to the caller of the target's prepare().
        warnings.warn(message, UserWarning, stacklevel=3)


def _label_bounds(geometries: List[GeomWithClass]) -> npt.NDArray[np.float64]:
    """(N, 4) minx, miny, maxx, maxy per label geometry, computed once per run."""
    if not geometries:
        return np.empty((0, 4), dtype=np.float64)
    array = np.empty(len(geometries), dtype=object)
    array[:] = [geometry for geometry, _ in geometries]
    return cast(npt.NDArray[np.float64], shapely.bounds(array))


def _geometries_in_window(
    geometries: List[GeomWithClass],
    bounds: npt.NDArray[np.float64],
    transform: Transform,
    height: int,
    width: int,
) -> List[GeomWithClass]:
    """Label geometries whose bounding box touches the window, in their original order.

    Order matters: later polygons overwrite earlier ones when rasterized.
    """
    a, b, c, d, e, f = transform
    if b or d:  # rotated grid: no cheap window bounds, keep everything
        return geometries
    xs = sorted((c, c + a * width))
    ys = sorted((f, f + e * height))
    pad = abs(a) + abs(e)  # one pixel of slack for all_touched edges
    hit = (
        (bounds[:, 0] <= xs[1] + pad)
        & (bounds[:, 2] >= xs[0] - pad)
        & (bounds[:, 1] <= ys[1] + pad)
        & (bounds[:, 3] >= ys[0] - pad)
    )
    return [geometries[index] for index in np.flatnonzero(hit)]


class SegmentationTarget:
    """Polygon labels as class masks, one pixel value per class.

    Each window rasterizes the labels that touch it once; patches are then cut out
    of that mask, with ``labels.ignore_index`` marking pixels without imagery.
    """

    def __init__(self, labels: LabelsConfig) -> None:
        self._labels = labels
        self._geometries: List[GeomWithClass] = []
        self._bounds: npt.NDArray[np.float64] = np.empty((0, 4), dtype=np.float64)
        self._class_map: Optional[ClassMap] = None
        self._sha256: Optional[str] = None

    @property
    def type(self) -> Optional[str]:
        return "segmentation"

    @property
    def class_map(self) -> ClassMap:
        if self._class_map is None:
            raise RuntimeError("SegmentationTarget.prepare() must run first")
        return self._class_map

    def prepare(self, source: RasterMetadata) -> None:
        data = self._labels.path.read_bytes()
        self._sha256 = hashlib.sha256(data).hexdigest()
        self._geometries, self._class_map = _parse_labels(self._labels, data, source.crs)
        _check_ignore_index(self._labels.ignore_index, self._class_map)
        _warn_if_labels_miss_raster(self._geometries, source)
        self._bounds = _label_bounds(self._geometries)

    def record(self) -> Optional[TargetRecord]:
        """Class map and ignore value, plus the label settings and a hash of the label
        file, so a resumed run notices edits."""
        if self._sha256 is None:
            raise RuntimeError("SegmentationTarget.prepare() must run first")
        settings = self._labels.model_dump(mode="json", exclude={"path", "ignore_index"})
        settings["sha256"] = self._sha256
        return TargetRecord(
            type="segmentation",
            class_map=self.class_map,
            ignore_index=self._labels.ignore_index,
            dtype="uint8",
            labels=settings,
            options={},
        )

    def window(
        self,
        transform: Transform,
        height: int,
        width: int,
        valid_mask: Optional[npt.NDArray[np.bool_]],
    ) -> WindowTarget:
        if not self._geometries:
            return NullWindow()
        nearby = _geometries_in_window(self._geometries, self._bounds, transform, height, width)
        mask = rasterize(nearby, (height, width), transform, self._labels.all_touched)
        return MaskWindow(mask, self._labels.ignore_index)
