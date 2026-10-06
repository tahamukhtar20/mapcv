"""Patch classification: one label, or a set of labels, per patch.

A patch's labels come from how much of it each class covers, on the very mask that
semantic segmentation would write for the same labels (this target wraps the
segmentation target, vector or label raster, so there is no second rasterization rule):

* **Coverage.** A class's coverage is its pixel count divided by the patch's *valid*
  pixels (``count / valid``, in float64). A pixel is valid when it has imagery (inside
  the raster and the imagery's validity mask: NoData, failed tiles) and is not
  ``labels.ignore_index`` (for a label raster, that includes pixels outside the label
  raster, on its NoData and on ``labels.ignore_values``). Padding is never valid.
  Background (mask value 0) is not a class and has no coverage. Where vector polygons
  overlap, the mask's later-wins rule applies, so the later class takes the pixels.
* **Qualifying.** A class qualifies when its coverage is at least
  ``classification.min_fraction`` and above zero (the default 0 means "any labeled
  pixel").
* **single.** The qualifying class with the largest coverage; ties go to the lowest
  class ID.
* **multi.** Every qualifying class, in ascending class ID order.
* **No class qualifies.** The patch is dropped (``empty: skip``) or kept with the label
  ``background``, recorded as class ID 0 (``empty: background``). A patch with no valid
  pixel at all has no coverage to judge and is dropped either way; ``sampler.max_empty_ratio``
  drops patches that are mostly without imagery.
* **sampler.min_label_ratio** keeps its segmentation meaning: the labeled (not
  background, not ignored) share of the whole patch must reach it.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
import numpy.typing as npt

from mapcv._patching import MaskWindow, PadMode, extract_array_patch
from mapcv.config import (
    BACKGROUND_LABEL,
    ClassificationOptions,
    LabelsConfig,
    RasterLabelsConfig,
    classification_name_problem,
)
from mapcv.imagery import RasterMetadata
from mapcv.labels import ClassMap
from mapcv.manifest import TargetRecord
from mapcv.targets.base import Transform, WindowTarget
from mapcv.targets.raster_labels import RasterSegmentationTarget
from mapcv.targets.segmentation import SegmentationTarget

# The class name when the labels have no label_field (every feature is class 1).
DEFAULT_CLASS = "object"


def class_names(class_map: ClassMap) -> Dict[int, str]:
    """Class ID to name, ascending by ID; ``{1: "object"}`` without a class map."""
    if not class_map:
        return {1: DEFAULT_CLASS}
    return {cid: name for name, cid in sorted(class_map.items(), key=lambda item: item[1])}


@dataclass(frozen=True)
class PatchLabels:
    """A patch's annotation: its coverage per class and the labels it was given.

    ``coverage`` maps class ID to the share of the patch's valid pixels (classes with
    pixels only, ascending). ``labels`` holds the assigned class IDs, ascending (``0``
    is the ``background`` label); it is empty for a patch that no class qualifies for.
    ``labeled_pixels`` counts the valid pixels of any class and ``patch_size`` is the
    patch's side (for ``sampler.min_label_ratio``).
    """

    coverage: Dict[int, float]
    labels: Tuple[int, ...]
    labeled_pixels: int
    patch_size: int


class ClassificationWindow:
    """Window of a classification target: patches are labeled from a class mask.

    ``mask`` covers the same pixels as the image window and holds class IDs, 0 for
    background and (with ``ignore_index``) the ignore value.
    """

    def __init__(
        self,
        mask: npt.NDArray[np.uint8],
        ignore_index: Optional[int],
        options: ClassificationOptions,
    ) -> None:
        self._mask = mask
        self._ignore_index = ignore_index
        self._options = options

    def annotate(
        self,
        row: int,
        col: int,
        patch_size: int,
        pad_mode: PadMode,
        valid_patch: Optional[npt.NDArray[np.bool_]],
    ) -> PatchLabels:
        """The labels of the patch whose top-left pixel is window pixel ``(row, col)``."""
        height, width = self._mask.shape
        patch, _ = extract_array_patch(self._mask, row, col, patch_size, "zero")
        # Padding (beyond the window) has no imagery, whatever pad_mode puts in the image.
        valid = np.zeros((patch_size, patch_size), dtype=np.bool_)
        top, left = max(0, -row), max(0, -col)
        bottom, right = min(patch_size, height - row), min(patch_size, width - col)
        if bottom > top and right > left:
            valid[top:bottom, left:right] = True
        if valid_patch is not None:
            valid &= valid_patch
        if self._ignore_index is not None:
            valid &= patch != self._ignore_index
        valid_pixels = int(np.count_nonzero(valid))
        counts = np.bincount(patch[valid], minlength=256)
        counts[0] = 0  # background is not a class
        present = np.flatnonzero(counts)
        coverage = {int(cid): float(counts[cid]) / valid_pixels for cid in present}
        labeled = int(counts.sum())

        options = self._options
        qualifying = [cid for cid, share in coverage.items() if share >= options.min_fraction]
        labels: Tuple[int, ...]
        if options.mode == "single":
            # ``present`` ascends, so max() keeps the lowest class ID among equal counts.
            best = max(qualifying, key=lambda cid: counts[cid], default=None)
            labels = () if best is None else (best,)
        else:
            labels = tuple(qualifying)
        # A patch without a single valid pixel cannot be judged (coverage is undefined), so
        # it is never background: it is dropped. sampler.max_empty_ratio drops mostly empty ones.
        if not labels and options.empty == "background" and valid_pixels:
            labels = (0,)
        return PatchLabels(coverage, labels, labeled, patch_size)

    def accepts(self, annotation: PatchLabels, min_label_ratio: float) -> bool:
        """Whether the patch has a label and enough labeled area (``sampler.min_label_ratio``)."""
        if (
            min_label_ratio > 0.0
            and annotation.labeled_pixels / annotation.patch_size**2 < min_label_ratio
        ):
            return False
        return bool(annotation.labels)

    def collate(self, annotations: Sequence[PatchLabels], patch_size: int) -> List[PatchLabels]:
        """The kept patches' annotations, in patch order."""
        return list(annotations)


class ClassificationTarget:
    """Patch labels from label coverage (``task: classification``); see the module docs."""

    def __init__(
        self, labels: Union[LabelsConfig, RasterLabelsConfig], options: ClassificationOptions
    ) -> None:
        self._labels = labels
        self._options = options
        self._inner: Union[SegmentationTarget, RasterSegmentationTarget] = (
            RasterSegmentationTarget(labels)
            if isinstance(labels, RasterLabelsConfig)
            else SegmentationTarget(labels)
        )

    @property
    def type(self) -> Optional[str]:
        return "classification"

    @property
    def options(self) -> ClassificationOptions:
        """The ``classification`` options this target was made with."""
        return self._options

    @property
    def class_map(self) -> ClassMap:
        return self._inner.class_map

    def prepare(self, source: RasterMetadata) -> None:
        self._inner.prepare(source)
        problem = classification_name_problem(class_names(self.class_map).values(), self._options)
        if problem is not None:
            raise ValueError(problem)

    def record(self) -> Optional[TargetRecord]:
        """The segmentation target's record (class map, ignore value, label settings and a
        hash or fingerprint of the label file) as a classification record with its options."""
        inner = self._inner.record()
        if inner is None:  # a segmentation target with labels always has a record
            raise RuntimeError("ClassificationTarget.prepare() must run first")
        return inner.model_copy(
            update={
                "type": "classification",
                "dtype": None,
                "options": self._options.model_dump(mode="json"),
            }
        )

    def window(
        self,
        transform: Transform,
        height: int,
        width: int,
        valid_mask: Optional[npt.NDArray[np.bool_]],
    ) -> WindowTarget:
        inner = self._inner.window(transform, height, width, valid_mask)
        if isinstance(inner, MaskWindow):
            return ClassificationWindow(inner.mask, inner.ignore_index, self._options)
        # No label feature near the raster: every mask pixel is background.
        return ClassificationWindow(
            np.zeros((height, width), dtype=np.uint8), self._labels.ignore_index, self._options
        )


__all__ = [
    "BACKGROUND_LABEL",
    "ClassificationTarget",
    "ClassificationWindow",
    "DEFAULT_CLASS",
    "PatchLabels",
    "class_names",
]
