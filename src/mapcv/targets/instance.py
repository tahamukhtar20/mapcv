"""Instance segmentation: one binary mask per label feature in the patches it shows in.

Conventions (shared with the COCO writer):

* **One instance per feature.** A MultiPolygon feature is one instance, and so is a
  feature that a patch edge cuts into several pieces: its mask covers every visible
  piece. Holes are holes.
* **Exact masks.** A feature's mask is burned with the same rasterizer and the same
  pixel rule as semantic segmentation masks (GDAL's: a pixel is burned when its centre
  is inside the polygon, or with ``labels.all_touched`` when an edge passes through
  it), on the patch's pixel grid. Every instance is rasterized on its own, so
  overlapping instances each keep their exact mask; nothing overwrites anything.
* **Visible part.** An instance's mask only holds pixels that are inside the patch,
  inside the raster (not in padding) and over pixels that have imagery (the source's
  validity mask: NoData, failed tiles). The visible area (computed as for detection)
  decides ``instance.min_visible``; the mask's pixel count decides
  ``instance.min_area`` and is the COCO ``area``.
* **Truncated.** An instance is truncated when part of the feature is not visible in
  this patch (cut by the patch edge, the raster edge or pixels without imagery).
* **Instance IDs.** The optional instance-ID mask numbers a patch's instances 1..N in
  feature order, the order of the patch's COCO annotations. Where instances overlap,
  the later instance wins (as later polygons overwrite earlier ones in semantic
  segmentation masks); the COCO masks stay exact.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple, cast

import numpy as np
import numpy.typing as npt
import shapely

from mapcv._patching import PadMode
from mapcv._rle import encode_part, mask_bbox
from mapcv.config import InstanceOptions, LabelsConfig
from mapcv.imagery import RasterMetadata
from mapcv.labels import ClassMap, GeomWithClass, label_file_sha256
from mapcv.manifest import TargetRecord
from mapcv.rasterizer import rasterize
from mapcv.targets.base import Transform, WindowTarget
from mapcv.targets.detection import _polygonal, clip_to_frame, to_pixels
from mapcv.targets.segmentation import (
    _label_bounds,
    _parse_labels,
    _raster_bounds,
    _warn_if_labels_miss_raster,
)

# A 16-bit instance-ID mask numbers at most this many instances per patch.
MAX_INSTANCE_ID = 65535
# Relative slack for comparing areas computed by different geometry operations.
_AREA_TOLERANCE = 1e-9

LABELS_MISS_MESSAGE = (
    "no label feature intersects the imagery extent, so no patch will have instances. "
    "Check that labels are longitude/latitude (not swapped) and cover the configured region."
)


@dataclass(frozen=True)
class Instance:
    """One instance in one patch, in the patch's pixel grid (see the module docs).

    ``bbox`` is the integer ``(x, y, width, height)`` of the mask's pixels, ``area``
    their count, ``counts`` the mask as a compressed COCO RLE string (column-major,
    background first) and ``truncated`` whether part of the feature is not visible.
    """

    category_id: int
    bbox: Tuple[int, int, int, int]
    area: int
    truncated: bool
    counts: str


@dataclass(frozen=True)
class PatchInstances:
    """A patch's annotation: its instances in feature order.

    ``covered`` is the number of patch pixels inside any instance (for
    ``sampler.min_label_ratio``). ``id_mask`` is the ``(patch_size, patch_size)``
    uint16 instance-ID mask with ``instance.id_mask``, else ``None``; the writer
    stores it next to the image.
    """

    instances: Tuple[Instance, ...]
    patch_size: int
    covered: int = 0
    id_mask: Optional[npt.NDArray[np.uint16]] = None

    @property
    def class_counts(self) -> Dict[str, int]:
        """Instances per class ID (string keys, ascending), as in the manifest summary."""
        counts: Dict[int, int] = {}
        for item in self.instances:
            counts[item.category_id] = counts.get(item.category_id, 0) + 1
        return {str(cid): counts[cid] for cid in sorted(counts)}


class InstanceWindow:
    """Window of an instance target: the nearby features, in window pixels and in the CRS."""

    def __init__(
        self,
        geometries: npt.NDArray[np.object_],
        world: npt.NDArray[np.object_],
        class_ids: npt.NDArray[np.int64],
        transform: Transform,
        height: int,
        width: int,
        options: InstanceOptions,
        all_touched: bool,
    ) -> None:
        self._geometries = geometries  # window pixels, for the visible fractions
        self._world = world  # the raster CRS, for the masks (as segmentation burns them)
        self._class_ids = class_ids
        self._areas = cast(npt.NDArray[np.float64], shapely.area(geometries))
        self._bounds = (
            cast(npt.NDArray[np.float64], shapely.bounds(geometries))
            if len(geometries)
            else np.empty((0, 4), dtype=np.float64)
        )
        self._transform = transform
        self._height = height
        self._width = width
        self._options = options
        self._all_touched = all_touched

    def _burn(self, index: int, row: int, col: int, patch_size: int) -> npt.NDArray[np.uint8]:
        """Feature ``index`` burned as 1 on the patch's whole pixel grid, padding included.

        Burned by the semantic segmentation rasterizer on the grid of the patch at window
        pixel ``(row, col)``, so it equals ``rasterio.features.rasterize`` of the feature
        on that grid.
        """
        a, b, c, d, e, f = self._transform
        shifted = (a, b, c + a * col + b * row, d, e, f + d * col + e * row)
        burned = rasterize(
            [(self._world[index], 1)], (patch_size, patch_size), shifted, self._all_touched
        )
        return burned

    def annotate(
        self,
        row: int,
        col: int,
        patch_size: int,
        pad_mode: PadMode,
        valid_patch: Optional[npt.NDArray[np.bool_]],
    ) -> PatchInstances:
        """The instances visible in the patch at window pixel ``(row, col)``."""
        options = self._options
        id_mask = np.zeros((patch_size, patch_size), dtype=np.uint16) if options.id_mask else None
        empty = PatchInstances((), patch_size, 0, id_mask)
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
        instances: List[Instance] = []
        covered = np.zeros((patch_size, patch_size), dtype=np.bool_)
        for position, index in enumerate(hit):
            area = float(visible_areas[position])
            full = float(self._areas[index])
            if area <= 0.0 or full <= 0.0:
                continue
            fraction = area / full
            if fraction < options.min_visible - _AREA_TOLERANCE:
                continue
            # Burned pixels lie within a pixel of the feature's bounds, and never in padding:
            # everything below works on that part of the patch only.
            bx0, by0, bx1, by1 = self._bounds[index]
            left = max(x0, int(np.floor(bx0)) - 1) - col
            top = max(y0, int(np.floor(by0)) - 1) - row
            right = min(x1, int(np.ceil(bx1)) + 1) - col
            bottom = min(y1, int(np.ceil(by1)) + 1) - row
            if right <= left or bottom <= top:
                continue
            around = (slice(top, bottom), slice(left, right))
            part = self._burn(int(index), row, col, patch_size)[around] != 0
            if valid_patch is not None:
                part &= valid_patch[around]
            pixels = int(np.count_nonzero(part))
            if pixels < options.min_area:
                continue
            x, y, width, height = mask_bbox(part)
            tight = part[y : y + height, x : x + width]
            x, y = x + left, y + top
            covered[around] |= part
            instances.append(
                Instance(
                    category_id=int(self._class_ids[index]),
                    bbox=(x, y, width, height),
                    area=pixels,
                    truncated=fraction < 1.0 - _AREA_TOLERANCE,
                    counts=encode_part(tight, x, y, patch_size, patch_size),
                )
            )
            if id_mask is not None:
                if len(instances) > MAX_INSTANCE_ID:
                    raise ValueError(
                        f"the patch at row {row}, column {col} has more than {MAX_INSTANCE_ID} "
                        "instances, more than a 16-bit instance-ID mask holds; use a smaller "
                        "sampler.patch_size, raise instance.min_area, or set "
                        "instance.id_mask: false"
                    )
                id_mask[around][part] = len(instances)
        return PatchInstances(tuple(instances), patch_size, int(np.count_nonzero(covered)), id_mask)

    def accepts(self, annotation: PatchInstances, min_label_ratio: float) -> bool:
        """Whether the instances cover enough of the patch (overlaps counted once)."""
        if min_label_ratio <= 0.0:
            return True
        return annotation.covered / annotation.patch_size**2 >= min_label_ratio

    def collate(
        self, annotations: Sequence[PatchInstances], patch_size: int
    ) -> List[PatchInstances]:
        """The kept patches' annotations, in patch order."""
        return list(annotations)


class InstanceTarget:
    """Label features as instance masks (``task: instance``); see the module docs."""

    def __init__(self, labels: LabelsConfig, options: InstanceOptions, patch_size: int) -> None:
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
        return "instance"

    @property
    def options(self) -> InstanceOptions:
        """The ``instance`` options this target was made with."""
        return self._options

    @property
    def class_map(self) -> ClassMap:
        if self._class_map is None:
            raise RuntimeError("InstanceTarget.prepare() must run first")
        return self._class_map

    def prepare(self, source: RasterMetadata) -> None:
        self._sha256 = label_file_sha256(self._labels.path)
        parsed, self._class_map = _parse_labels(self._labels, source.crs)
        features: List[GeomWithClass] = [
            (_polygonal(geometry), class_id) for geometry, class_id in parsed
        ]
        features = [(geometry, cid) for geometry, cid in features if not geometry.is_empty]
        _warn_if_labels_miss_raster(features, source, LABELS_MISS_MESSAGE)
        geometries = np.empty(len(features), dtype=object)
        geometries[:] = [geometry for geometry, _ in features]
        self._geometries = geometries
        self._class_ids = np.array([cid for _, cid in features], dtype=np.int64)
        self._bounds = _label_bounds(features)
        self._warn_if_too_large(source)

    def _warn_if_too_large(self, source: RasterMetadata) -> None:
        """Warn about features whose visible fraction can never reach ``min_visible``."""
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
                f"instance.min_visible ({min_visible:g}) of them fits in a "
                f"{self._patch_size} px patch, so they never get a mask; lower "
                "instance.min_visible or use larger patches",
                UserWarning,
                stacklevel=3,
            )

    def record(self) -> Optional[TargetRecord]:
        """Class map, the label settings with a hash of the label file, and the options."""
        if self._sha256 is None:
            raise RuntimeError("InstanceTarget.prepare() must run first")
        # labels.type is left out, as for segmentation: polygon-label records predate it.
        settings = self._labels.model_dump(mode="json", exclude={"path", "ignore_index", "type"})
        settings["sha256"] = self._sha256
        return TargetRecord(
            type="instance",
            class_map=self.class_map,
            ignore_index=None,
            dtype="uint16" if self._options.id_mask else None,
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
        world = self._geometries[nearby]
        return InstanceWindow(
            to_pixels(world, transform),
            world,
            self._class_ids[nearby],
            transform,
            height,
            width,
            self._options,
            self._labels.all_touched,
        )

    def _nearby(self, transform: Transform, height: int, width: int) -> npt.NDArray[np.intp]:
        """Indices of the features whose bounds touch the window, in feature order."""
        a, b, c, d, e, f = transform
        count = len(self._geometries)
        if b or d:  # rotated grid: no cheap window bounds, keep everything
            return np.arange(count)
        xs = sorted((c, c + a * width))
        ys = sorted((f, f + e * height))
        pad_x, pad_y = abs(a), abs(e)  # one pixel of slack for all_touched edges
        bounds = self._bounds
        hit = (
            (bounds[:, 0] <= xs[1] + pad_x)
            & (bounds[:, 2] >= xs[0] - pad_x)
            & (bounds[:, 1] <= ys[1] + pad_y)
            & (bounds[:, 3] >= ys[0] - pad_y)
        )
        return np.flatnonzero(hit)
