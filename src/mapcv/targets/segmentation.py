"""Semantic segmentation: polygon labels burned into per-patch class masks."""

from __future__ import annotations

import hashlib
import warnings
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple, cast

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
    assign_class_ids,
    label_file_sha256,
    load_vector_labels,
    transform_all_to_mercator,
)
from mapcv.manifest import TargetRecord
from mapcv.rasterizer import rasterize
from mapcv.targets.base import Transform, WindowTarget


def labels_miss_message(what: str, outcome: str) -> str:
    """The warning for label features that all fall outside the imagery."""
    return (
        f"no label {what} intersects the imagery extent, so {outcome}. Check that the region "
        "and the labels are longitude/latitude (not swapped) and that the labels cover the "
        "region."
    )


def labels_empty_message(what: str, outcome: str) -> str:
    """The warning for a label file without a feature mapcv can use."""
    return (
        f"no usable label {what} in the label file, so {outcome}. Check labels.path, "
        "labels.label_field and labels.classes (and labels.layer for a GeoPackage with "
        "several layers)."
    )


def label_buffer(labels: Any) -> Optional[Tuple[Optional[float], Optional[float]]]:
    """The ``(line, point)`` buffer widths of a labels block or label file, or ``None``."""
    if labels.buffer is None:
        return None
    return labels.buffer.line, labels.buffer.point


def load_labels(labels: LabelsConfig, points: bool = False) -> Tuple[List[GeomWithClass], ClassMap]:
    """Every label file's features in WGS-84 lon/lat with class IDs, and the class map.

    One ``labels.path`` is read as :func:`mapcv.labels.load_vector_labels` reads it. With
    ``labels.files`` each file gives its features a class name (its ``label_field`` value,
    or its ``class``), and IDs are assigned once over the names of all files (with
    ``labels.classes`` when given), so a name has one ID whichever file it comes from.
    Features keep the file order: later files are rasterized later and win overlaps.
    """
    if labels.osm is not None:
        from mapcv.osm import default_class_ids, osm_labels_file

        return load_vector_labels(
            osm_labels_file(labels.osm),
            "class",
            labels.classes or default_class_ids(labels.osm),
            points=points,
            buffer=label_buffer(labels),
        )
    if labels.files is None:
        assert labels.path is not None  # the config requires path, files or osm
        return load_vector_labels(
            labels.path,
            labels.label_field,
            labels.classes,
            points=points,
            layer=labels.layer,
            buffer=label_buffer(labels),
        )
    geometries: List[Any] = []
    names: List[Optional[str]] = []
    for file in labels.files:
        raw, file_map = load_vector_labels(
            file.path,
            file.label_field,
            None,
            points=points,
            layer=file.layer,
            buffer=label_buffer(file),
        )
        if file.class_name is not None:
            names.extend(file.class_name for _ in raw)
        else:
            by_id = {class_id: name for name, class_id in file_map.items()}
            names.extend(by_id[class_id] for _, class_id in raw)
        geometries.extend(geometry for geometry, _ in raw)
    ids, class_map = assign_class_ids(names, "of labels.files", labels.classes)
    skipped = ids.count(0)
    if skipped:
        warnings.warn(
            f"labels.files: skipped {skipped} feature(s) with a class not in labels.classes.",
            UserWarning,
            stacklevel=3,
        )
    return [(geometry, cid) for geometry, cid in zip(geometries, ids) if cid], class_map


def label_settings(labels: LabelsConfig, exclude: Set[str]) -> Dict[str, Any]:
    """A labels block as a target record stores it: without paths (machine-specific; the
    files are identified by their hash) and without the ``exclude`` keys. Several files
    keep their settings in order, with ``class`` as the config writes it."""
    settings: Dict[str, Any] = labels.model_dump(mode="json", exclude=exclude | {"path", "files"})
    if labels.files is not None:
        settings["files"] = [
            file.model_dump(mode="json", by_alias=True, exclude={"path"}) for file in labels.files
        ]
    return settings


def labels_sha256(labels: LabelsConfig) -> str:
    """SHA-256 of the label file(s), recorded so a resumed run notices edited labels."""
    if labels.osm is not None:
        from mapcv.osm import osm_labels_file

        return label_file_sha256(osm_labels_file(labels.osm))
    if labels.files is None:
        assert labels.path is not None
        return label_file_sha256(labels.path)
    digest = hashlib.sha256()
    for file in labels.files:
        digest.update(label_file_sha256(file.path).encode("ascii") + b"\n")
    return digest.hexdigest()


def _area_geometries(path: Path, destination_crs: str) -> List[GeomWithClass]:
    """The polygons of an annotated-area file in ``destination_crs`` (classes ignored)."""
    raw, _ = load_vector_labels(path)
    return [(transform_geometry_to_crs(geometry, destination_crs), 1) for geometry, _ in raw]


def _parse_labels(
    labels: LabelsConfig, destination_crs: str, points: bool = False
) -> Tuple[List[GeomWithClass], ClassMap]:
    """Label geometries in ``destination_crs`` with their class IDs, and the class map.

    The file is read by :func:`mapcv.labels.load_vector_labels`, whatever its format.
    ``points=True`` also keeps point features.
    """
    raw, class_map = load_labels(labels, points)

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
    geometries: List[GeomWithClass],
    source: RasterMetadata,
    what: str = "polygon",
    outcome: str = "every mask will be background",
) -> None:
    # Attribute the warnings to the caller of the target's prepare().
    if not geometries:
        warnings.warn(labels_empty_message(what, outcome), UserWarning, stacklevel=3)
        return
    extent = box(*_raster_bounds(source))
    if not any(geometry.intersects(extent) for geometry, _ in geometries):
        warnings.warn(labels_miss_message(what, outcome), UserWarning, stacklevel=3)


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
        # labels.annotated_area: its polygons in the raster CRS, and the file's hash.
        self._area: Optional[List[GeomWithClass]] = None
        self._area_bounds: npt.NDArray[np.float64] = np.empty((0, 4), dtype=np.float64)
        self._area_sha256: Optional[str] = None

    @property
    def type(self) -> Optional[str]:
        return "segmentation"

    @property
    def class_map(self) -> ClassMap:
        if self._class_map is None:
            raise RuntimeError("SegmentationTarget.prepare() must run first")
        return self._class_map

    def prepare(self, source: RasterMetadata) -> None:
        self._sha256 = labels_sha256(self._labels)
        self._geometries, self._class_map = _parse_labels(self._labels, source.crs)
        _check_ignore_index(self._labels.ignore_index, self._class_map)
        _warn_if_labels_miss_raster(self._geometries, source)
        self._bounds = _label_bounds(self._geometries)
        area = self._labels.annotated_area
        if area is not None:
            self._area_sha256 = label_file_sha256(area)
            self._area = _area_geometries(area, source.crs)
            self._area_bounds = _label_bounds(self._area)

    def record(self) -> Optional[TargetRecord]:
        """Class map and ignore value, plus the label settings and a hash of the label
        file, so a resumed run notices edits."""
        if self._sha256 is None:
            raise RuntimeError("SegmentationTarget.prepare() must run first")
        # labels.type is left out: manifests from before raster labels record no type.
        settings = label_settings(self._labels, {"ignore_index", "type", "annotated_area"})
        settings["sha256"] = self._sha256
        if self._area_sha256 is not None:
            # The area's file by its contents, not its path (as for the label file).
            settings["annotated_area_sha256"] = self._area_sha256
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
        if not self._geometries and self._area is None:
            return NullWindow()
        nearby = _geometries_in_window(self._geometries, self._bounds, transform, height, width)
        mask = rasterize(nearby, (height, width), transform, self._labels.all_touched)
        ignore = self._labels.ignore_index
        if self._area is not None and ignore is not None:
            # Outside the annotated area nothing was labeled: not background, ignored.
            parts = _geometries_in_window(self._area, self._area_bounds, transform, height, width)
            inside = rasterize(parts, (height, width), transform, False)
            mask[inside == 0] = ignore
        return MaskWindow(mask, ignore)
