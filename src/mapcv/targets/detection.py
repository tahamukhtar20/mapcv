"""Object detection: each label feature as an axis-aligned box in the patches it shows in.

Conventions (shared with the COCO and YOLO writers):

* **One object per feature.** A MultiPolygon feature is one object, and so is a
  feature that a patch edge cuts into several pieces: its box covers every visible
  piece. Holes do not change a box.
* **Visible part.** An object's box is the bounding box of the part of the feature
  that lies inside the patch, inside the raster (not in padding) and over pixels
  that have imagery (the source's validity mask: NoData, failed tiles). The visible
  area decides ``min_visible`` and is the COCO ``area``.
* **Pixel coordinates.** Boxes are in the patch's pixel grid: ``x`` to the right and
  ``y`` down from the patch's top-left corner, continuous, so pixel ``(col, row)``
  covers ``[col, col + 1] x [row, row + 1]`` and a box can span ``[0, patch_size]``.
  Values are rounded to ``BOX_DECIMALS`` decimals so runs are reproducible.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple, cast

import numpy as np
import numpy.typing as npt
import shapely
from shapely.geometry.base import BaseGeometry

from mapcv._patching import PadMode
from mapcv.config import DetectionOptions, LabelsConfig
from mapcv.imagery import RasterMetadata
from mapcv.labels import ClassMap, GeomWithClass
from mapcv.manifest import TargetRecord
from mapcv.targets.base import Transform, WindowTarget
from mapcv.targets.segmentation import (
    _label_bounds,
    _parse_labels,
    _raster_bounds,
    _warn_if_labels_miss_raster,
    label_settings,
    labels_sha256,
)

# Box coordinates and areas are rounded to 1/10,000 of a pixel.
BOX_DECIMALS = 4
# Relative slack for comparing areas computed by different geometry operations.
_AREA_TOLERANCE = 1e-9


_POINT_TYPES = frozenset({"Point", "MultiPoint"})
_POLYGONAL_TYPES = frozenset({"Polygon", "MultiPolygon"})


@dataclass(frozen=True)
class DetectedObject:
    """One object in one patch, in the patch's pixel coordinates (see the module docs).

    ``bbox`` is ``(x, y, width, height)`` of the visible part, ``area`` the visible
    area in square pixels, and ``truncated`` whether part of the object is not
    visible in this patch (cut by the patch edge, the raster edge or pixels
    without imagery).
    """

    category_id: int
    bbox: Tuple[float, float, float, float]
    area: float
    truncated: bool


@dataclass(frozen=True)
class PatchObjects:
    """A patch's annotation: its objects in feature order.

    ``visible`` holds the visible geometries (window pixels) for
    ``sampler.min_label_ratio``; the writer only gets ``objects``.
    """

    objects: Tuple[DetectedObject, ...]
    visible: Tuple[BaseGeometry, ...]
    patch_size: int

    @property
    def class_counts(self) -> Dict[str, int]:
        """Objects per class ID (string keys, ascending), as in the manifest summary."""
        counts: Dict[int, int] = {}
        for item in self.objects:
            counts[item.category_id] = counts.get(item.category_id, 0) + 1
        return {str(cid): counts[cid] for cid in sorted(counts)}


def _polygonal(geometry: BaseGeometry) -> BaseGeometry:
    """``geometry`` repaired with ``make_valid`` when invalid, keeping its polygonal parts."""
    if geometry.geom_type in _POINT_TYPES or geometry.is_valid:
        return geometry
    repaired = shapely.make_valid(geometry)
    if repaired.geom_type in _POLYGONAL_TYPES:
        return cast(BaseGeometry, repaired)
    parts = [
        part
        for part in getattr(repaired, "geoms", [])
        if part.geom_type in _POLYGONAL_TYPES and not part.is_empty
    ]
    if not parts:
        return cast(BaseGeometry, shapely.Polygon())
    return cast(BaseGeometry, shapely.union_all(parts))


def to_pixels(geometries: npt.NDArray[np.object_], transform: Transform) -> npt.NDArray[np.object_]:
    """Geometries from the raster CRS into the pixel grid of ``transform`` (inverse affine)."""
    a, b, c, d, e, f = transform
    det = a * e - b * d
    if det == 0:
        raise ValueError("the imagery transform is not invertible")

    def inverse(coords: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
        x = coords[:, 0] - c
        y = coords[:, 1] - f
        return np.column_stack(((e * x - b * y) / det, (a * y - d * x) / det))

    return cast(npt.NDArray[np.object_], shapely.transform(geometries, inverse))


def _point_boxes(geometries: npt.NDArray[np.object_], size: float) -> npt.NDArray[np.object_]:
    """Point features (in pixels) as ``size`` x ``size`` squares centred on each point."""
    out = geometries.copy()
    for index, geometry in enumerate(geometries):
        if geometry.geom_type in _POINT_TYPES:
            out[index] = geometry.buffer(size / 2, cap_style="square", join_style="mitre")
    return out


def _runs(mask: npt.NDArray[np.bool_]) -> Tuple[npt.NDArray[np.intp], ...]:
    """``(rows, starts, ends)`` of the runs of true pixels in each row of ``mask``."""
    padded = np.zeros((mask.shape[0], mask.shape[1] + 2), dtype=np.int8)
    padded[:, 1:-1] = mask
    steps = np.diff(padded, axis=1)
    starts = np.argwhere(steps == 1)
    ends = np.argwhere(steps == -1)
    return starts[:, 0], starts[:, 1], ends[:, 1]


def _rectangles(mask: npt.NDArray[np.bool_]) -> List[Tuple[int, int, int, int]]:
    """Disjoint rectangles ``(x0, y0, x1, y1)`` covering the true pixels of ``mask``.

    Runs of true pixels in each row are merged with identical runs in the rows below.
    """
    rows, starts, ends = _runs(mask)
    row_runs: Dict[int, List[Tuple[int, int]]] = {}
    for row, start, end in zip(rows.tolist(), starts.tolist(), ends.tolist()):
        row_runs.setdefault(row, []).append((start, end))
    rectangles: List[Tuple[int, int, int, int]] = []
    open_runs: Dict[Tuple[int, int], int] = {}
    for row in range(mask.shape[0] + 1):
        runs = set(row_runs.get(row, ()))
        for run in [run for run in open_runs if run not in runs]:
            rectangles.append((run[0], open_runs.pop(run), run[1], row))
        for run in sorted(runs):
            open_runs.setdefault(run, row)
    return rectangles


def mask_region(
    mask: npt.NDArray[np.bool_], x_offset: int, y_offset: int
) -> Optional[BaseGeometry]:
    """The true pixels of ``mask`` as one geometry in pixel coordinates, or ``None``."""
    rectangles = _rectangles(mask)
    if not rectangles:
        return None
    corners = np.array(rectangles, dtype=np.float64) + (x_offset, y_offset, x_offset, y_offset)
    boxes = shapely.box(corners[:, 0], corners[:, 1], corners[:, 2], corners[:, 3])
    return cast(BaseGeometry, shapely.union_all(boxes))


def _areal(geometries: npt.NDArray[np.object_]) -> npt.NDArray[np.object_]:
    """Only the polygonal parts of each geometry.

    An intersection of polygons that also touch along an edge or at a point keeps
    that line or point; it has no area but would widen the bounds.
    """
    out = geometries.copy()
    types = shapely.get_type_id(geometries)
    for index in np.flatnonzero((types != 3) & (types != 6)):  # not (Multi)Polygon
        parts = [
            part
            for part in shapely.get_parts(geometries[index])
            if part.geom_type in _POLYGONAL_TYPES and not part.is_empty
        ]
        out[index] = shapely.multipolygons(parts) if parts else shapely.Polygon()
    return out


# Above this many no-imagery rectangles (noisy masks: scattered black pixels in XYZ
# imagery), objects are cut by the runs of valid pixels instead of by the union of
# the invalid ones, whose cost grows much faster.
_MAX_UNION_RECTANGLES = 64


def _over_valid_runs(
    geometry: BaseGeometry, valid: npt.NDArray[np.bool_], x_offset: int, y_offset: int
) -> BaseGeometry:
    """The part of ``geometry`` (pixels) over the true pixels of ``valid``.

    ``valid[0, 0]`` is the pixel at ``(x_offset, y_offset)``. The geometry is cut into
    one-pixel rows and each row into the runs of valid pixels; the pieces are disjoint.
    """
    if geometry.is_empty:
        return geometry
    gx0, gy0, gx1, gy1 = geometry.bounds
    c0 = max(int(np.floor(gx0)) - x_offset, 0)
    r0 = max(int(np.floor(gy0)) - y_offset, 0)
    c1 = min(int(np.ceil(gx1)) - x_offset, valid.shape[1])
    r1 = min(int(np.ceil(gy1)) - y_offset, valid.shape[0])
    sub = valid[r0:r1, c0:c1]
    if sub.all():
        return geometry
    rows, starts, ends = _runs(sub)
    if not len(rows):
        return cast(BaseGeometry, shapely.Polygon())
    left, right = c0 + x_offset, c1 + x_offset
    strip_rows = np.arange(r1 - r0)
    tops = (strip_rows + r0 + y_offset).astype(np.float64)
    strips = shapely.intersection(geometry, shapely.box(left, tops, right, tops + 1))
    run_tops = (rows + r0 + y_offset).astype(np.float64)
    pieces = shapely.intersection(
        strips[rows],
        shapely.box(starts + left, run_tops, ends + left, run_tops + 1),
    )
    pieces = _areal(pieces)
    pieces = pieces[shapely.area(pieces) > 0]
    parts = shapely.get_parts(pieces)
    return cast(BaseGeometry, shapely.multipolygons(parts) if len(parts) else shapely.Polygon())


def visible_parts(
    geometries: npt.NDArray[np.object_],
    valid: npt.NDArray[np.bool_],
    x_offset: int,
    y_offset: int,
) -> npt.NDArray[np.object_]:
    """Each geometry (pixels) without the pixels where ``valid`` is false.

    ``valid[0, 0]`` is the pixel at ``(x_offset, y_offset)`` and the geometries
    must lie inside the pixels ``valid`` covers. Returns polygonal
    geometries; the pieces of a geometry may share edges (only their areas and
    bounds are used).
    """
    if valid.all():
        return geometries
    rectangles = _rectangles(~valid)
    if len(rectangles) <= _MAX_UNION_RECTANGLES:
        invalid = mask_region(~valid, x_offset, y_offset)
        return _areal(shapely.difference(geometries, invalid))
    out = geometries.copy()
    for index, geometry in enumerate(geometries):
        out[index] = _over_valid_runs(geometry, valid, x_offset, y_offset)
    return out


def clip_to_frame(
    geometries: npt.NDArray[np.object_],
    bounds: npt.NDArray[np.float64],
    frame: Tuple[int, int, int, int],
    row: int,
    col: int,
    valid_patch: Optional[npt.NDArray[np.bool_]],
) -> Optional[Tuple[npt.NDArray[np.intp], npt.NDArray[np.object_]]]:
    """The parts of the geometries (window pixels) that a patch shows, or ``None`` for none.

    ``frame`` is ``(x0, y0, x1, y1)``, the patch inside its window (beyond it is
    padding); ``(row, col)`` is the patch's top-left pixel in the window and
    ``valid_patch`` marks the patch pixels that have imagery. Only geometries whose
    bounds touch the frame are cut. Returns their indices (in order) and their
    polygonal parts inside the frame and over pixels with imagery; a part may be empty.
    """
    x0, y0, x1, y1 = frame
    hit = np.flatnonzero(
        (bounds[:, 0] < x1) & (bounds[:, 2] > x0) & (bounds[:, 1] < y1) & (bounds[:, 3] > y0)
    )
    if not len(hit):
        return None
    clipped = _areal(shapely.intersection(geometries[hit], shapely.box(x0, y0, x1, y1)))
    if valid_patch is not None:
        # Only the pixels under the candidates matter; cut the mask down to them.
        cx0 = max(x0, int(np.floor(bounds[hit, 0].min())))
        cy0 = max(y0, int(np.floor(bounds[hit, 1].min())))
        cx1 = min(x1, int(np.ceil(bounds[hit, 2].max())))
        cy1 = min(y1, int(np.ceil(bounds[hit, 3].max())))
        sub = valid_patch[cy0 - row : cy1 - row, cx0 - col : cx1 - col]
        if sub.size:
            clipped = visible_parts(clipped, sub, cx0, cy0)
    return hit, clipped


class DetectionWindow:
    """Window of a detection target: the nearby features in window pixel coordinates."""

    def __init__(
        self,
        geometries: npt.NDArray[np.object_],
        class_ids: npt.NDArray[np.int64],
        height: int,
        width: int,
        options: DetectionOptions,
    ) -> None:
        self._geometries = geometries
        self._class_ids = class_ids
        self._areas = cast(npt.NDArray[np.float64], shapely.area(geometries))
        self._bounds = (
            cast(npt.NDArray[np.float64], shapely.bounds(geometries))
            if len(geometries)
            else np.empty((0, 4), dtype=np.float64)
        )
        self._height = height
        self._width = width
        self._options = options

    def annotate(
        self,
        row: int,
        col: int,
        patch_size: int,
        pad_mode: PadMode,
        valid_patch: Optional[npt.NDArray[np.bool_]],
    ) -> PatchObjects:
        """The objects visible in the patch at window pixel ``(row, col)``."""
        empty = PatchObjects((), (), patch_size)
        # The visible frame: the patch inside the window (beyond it is padding).
        x0, y0 = max(col, 0), max(row, 0)
        x1, y1 = min(col + patch_size, self._width), min(row + patch_size, self._height)
        if x1 <= x0 or y1 <= y0 or not len(self._geometries):
            return empty
        candidates = clip_to_frame(
            self._geometries, self._bounds, (x0, y0, x1, y1), row, col, valid_patch
        )
        if candidates is None:
            return empty
        hit, clipped = candidates
        visible_areas = shapely.area(clipped)
        boxes = shapely.bounds(clipped)
        options = self._options
        objects: List[DetectedObject] = []
        visible: List[BaseGeometry] = []
        for position, index in enumerate(hit):
            area = float(visible_areas[position])
            full = float(self._areas[index])
            if area <= 0.0 or full <= 0.0:
                continue
            fraction = area / full
            if fraction < options.min_visible - _AREA_TOLERANCE:
                continue
            bx0, by0, bx1, by1 = (float(value) for value in boxes[position])
            if bx1 - bx0 < options.min_box_pixels or by1 - by0 < options.min_box_pixels:
                continue
            left = round(bx0 - col, BOX_DECIMALS)
            top = round(by0 - row, BOX_DECIMALS)
            right = round(bx1 - col, BOX_DECIMALS)
            bottom = round(by1 - row, BOX_DECIMALS)
            objects.append(
                DetectedObject(
                    category_id=int(self._class_ids[index]),
                    bbox=(
                        left,
                        top,
                        round(right - left, BOX_DECIMALS),
                        round(bottom - top, BOX_DECIMALS),
                    ),
                    area=round(area, BOX_DECIMALS),
                    truncated=fraction < 1.0 - _AREA_TOLERANCE,
                )
            )
            visible.append(clipped[position])
        return PatchObjects(tuple(objects), tuple(visible), patch_size)

    def accepts(self, annotation: PatchObjects, min_label_ratio: float) -> bool:
        """Whether the kept objects cover enough of the patch (overlaps counted once)."""
        if min_label_ratio <= 0.0:
            return True
        if not annotation.visible:
            return False
        covered = float(shapely.union_all(list(annotation.visible)).area)
        return covered / annotation.patch_size**2 >= min_label_ratio

    def collate(self, annotations: Sequence[PatchObjects], patch_size: int) -> List[PatchObjects]:
        """The kept patches' annotations, in patch order."""
        return [PatchObjects(item.objects, (), item.patch_size) for item in annotations]


class DetectionTarget:
    """Label features as boxes (``task: detection``); see the module docs."""

    def __init__(self, labels: LabelsConfig, options: DetectionOptions, patch_size: int) -> None:
        self._labels = labels
        self._options = options
        self._patch_size = patch_size
        self._geometries: npt.NDArray[np.object_] = np.empty(0, dtype=object)
        self._class_ids: npt.NDArray[np.int64] = np.empty(0, dtype=np.int64)
        self._bounds: npt.NDArray[np.float64] = np.empty((0, 4), dtype=np.float64)
        self._class_map: Optional[ClassMap] = None
        self._sha256: Optional[str] = None

    @property
    def type(self) -> Optional[str]:
        return "detection"

    @property
    def options(self) -> DetectionOptions:
        """The ``detection`` options this target was made with."""
        return self._options

    @property
    def class_map(self) -> ClassMap:
        if self._class_map is None:
            raise RuntimeError("DetectionTarget.prepare() must run first")
        return self._class_map

    def prepare(self, source: RasterMetadata) -> None:
        self._sha256 = labels_sha256(self._labels)
        points = self._options.point_box_size is not None
        parsed, self._class_map = _parse_labels(self._labels, source.crs, points=points)
        features: List[GeomWithClass] = [
            (_polygonal(geometry), class_id) for geometry, class_id in parsed
        ]
        features = [(geometry, cid) for geometry, cid in features if not geometry.is_empty]
        _warn_if_labels_miss_raster(features, source, "feature", "no patch will have objects")
        geometries = np.empty(len(features), dtype=object)
        geometries[:] = [geometry for geometry, _ in features]
        self._geometries = geometries
        self._class_ids = np.array([cid for _, cid in features], dtype=np.int64)
        self._bounds = _label_bounds(features)
        self._warn_if_too_large(source)

    def _warn_if_too_large(self, source: RasterMetadata) -> None:
        """Warn about objects whose visible fraction can never reach ``min_visible``."""
        min_visible = self._options.min_visible
        if min_visible <= 0.0 or not len(self._geometries):
            return
        a, b, _, d, e, _ = source.transform
        pixel_area = abs(a * e - b * d)
        left, bottom, right, top = _raster_bounds(source)
        bounds = self._bounds
        inside = (
            (bounds[:, 0] < right)
            & (bounds[:, 2] > left)
            & (bounds[:, 1] < top)
            & (bounds[:, 3] > bottom)
        )
        areas = shapely.area(self._geometries[inside]) / pixel_area
        too_large = int(np.count_nonzero(areas * min_visible > self._patch_size**2))
        if too_large:
            warnings.warn(
                f"{too_large} label feature(s) are so large that less than "
                f"detection.min_visible ({min_visible:g}) of them fits in a "
                f"{self._patch_size} px patch, so they never get a box; lower "
                "detection.min_visible or use larger patches",
                UserWarning,
                stacklevel=3,
            )

    def record(self) -> Optional[TargetRecord]:
        """Class map, the label settings with a hash of the label file, and the options."""
        if self._sha256 is None:
            raise RuntimeError("DetectionTarget.prepare() must run first")
        # labels.type is left out, as for segmentation: polygon-label records predate it.
        settings = label_settings(self._labels, {"ignore_index", "all_touched", "type"})
        settings["sha256"] = self._sha256
        return TargetRecord(
            type="detection",
            class_map=self.class_map,
            ignore_index=None,
            dtype=None,
            labels=settings,
            options=self._options.model_dump(mode="json"),
        )

    def window(
        self,
        transform: Transform,
        height: int,
        width: int,
        valid_mask: Optional[npt.NDArray[np.bool_]],
    ) -> WindowTarget:
        nearby = self._nearby(transform, height, width)
        geometries = to_pixels(self._geometries[nearby], transform)
        if self._options.point_box_size is not None:
            geometries = _point_boxes(geometries, self._options.point_box_size)
        return DetectionWindow(geometries, self._class_ids[nearby], height, width, self._options)

    def _nearby(self, transform: Transform, height: int, width: int) -> npt.NDArray[np.intp]:
        """Indices of the features whose bounds touch the window, in feature order."""
        a, b, c, d, e, f = transform
        count = len(self._geometries)
        if b or d:  # rotated grid: no cheap window bounds, keep everything
            return np.arange(count)
        xs = sorted((c, c + a * width))
        ys = sorted((f, f + e * height))
        # One pixel of slack, plus half a point box around point features.
        slack = 1.0 + (self._options.point_box_size or 0.0) / 2
        pad_x, pad_y = abs(a) * slack, abs(e) * slack
        bounds = self._bounds
        hit = (
            (bounds[:, 0] <= xs[1] + pad_x)
            & (bounds[:, 2] >= xs[0] - pad_x)
            & (bounds[:, 1] <= ys[1] + pad_y)
            & (bounds[:, 3] >= ys[0] - pad_y)
        )
        return np.flatnonzero(hit)
