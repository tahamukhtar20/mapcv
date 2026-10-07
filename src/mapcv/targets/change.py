"""Change detection: a binary change mask per patch, for a before and an after image.

The mask comes from labels that mark what changed (vector features or a label
raster: every labeled pixel is change) or from two vector label sets, before and
after: a pixel changed where their masks differ. Changed pixels get
``change.change_value``, unchanged ones 0, and pixels without imagery in either
image (or, for a label raster, without a label, or outside either label set's
``annotated_area``) the ignore value. The masks are
cut out of each window like segmentation masks (:class:`~mapcv._patching.MaskWindow`),
on the grid of the first (before) image.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import numpy.typing as npt

from mapcv._patching import MaskWindow
from mapcv.config import ChangeOptions, LabelsConfig, RasterLabelsConfig
from mapcv.imagery import RasterMetadata
from mapcv.labels import ClassMap
from mapcv.manifest import TargetRecord
from mapcv.targets.base import Transform, WindowTarget
from mapcv.targets.raster_labels import RasterSegmentationTarget
from mapcv.targets.segmentation import SegmentationTarget

# The class name of changed pixels in the manifest's class map.
CHANGE_CLASS = "change"

_LabelTarget = SegmentationTarget | RasterSegmentationTarget


def _window_mask(
    target: _LabelTarget,
    transform: Transform,
    height: int,
    width: int,
    valid_mask: npt.NDArray[np.bool_] | None,
) -> npt.NDArray[Any]:
    """The label target's class mask of a window (all background without labels nearby)."""
    window = target.window(transform, height, width, valid_mask)
    if isinstance(window, MaskWindow):
        return window.mask
    return np.zeros((height, width), dtype=np.uint8)


class ChangeTarget:
    """Change masks (``task: change``); see the module docs."""

    def __init__(
        self,
        options: ChangeOptions,
        labels: LabelsConfig | RasterLabelsConfig | None = None,
    ) -> None:
        self._options = options
        self._changed: _LabelTarget | None = None
        self._before: SegmentationTarget | None = None
        self._after: SegmentationTarget | None = None
        self._annotated_areas = False
        if options.before is not None and options.after is not None:
            self._before = SegmentationTarget(options.before)
            self._after = SegmentationTarget(options.after)
            self._ignore_index = options.before.ignore_index
            self._annotated_areas = (
                options.before.annotated_area is not None
                or options.after.annotated_area is not None
            )
        elif isinstance(labels, RasterLabelsConfig):
            self._changed = RasterSegmentationTarget(labels)
            self._ignore_index = labels.ignore_index
        elif labels is not None:
            self._changed = SegmentationTarget(labels)
            self._ignore_index = labels.ignore_index
        else:  # pragma: no cover - the config refuses this
            raise ValueError("task: change needs labels, or change.before and change.after")

    @property
    def type(self) -> str | None:
        return "change"

    @property
    def options(self) -> ChangeOptions:
        """The ``change`` options this target was made with."""
        return self._options

    @property
    def class_map(self) -> ClassMap:
        return {CHANGE_CLASS: self._options.change_value}

    def prepare(self, source: RasterMetadata) -> None:
        for target in (self._changed, self._before, self._after):
            if target is not None:
                target.prepare(source)

    def record(self) -> TargetRecord | None:
        """The change class, ignore value and change value, and the label settings with
        their files' hashes (both sets', for a before/after comparison)."""
        labels: dict[str, Any] | None
        if self._changed is not None:
            changed = self._changed.record()
            labels = changed.labels if changed is not None else None
        else:
            assert self._before is not None and self._after is not None
            before, after = self._before.record(), self._after.record()
            labels = {
                "before": before.labels if before is not None else None,
                "after": after.labels if after is not None else None,
            }
        return TargetRecord(
            type="change",
            class_map=self.class_map,
            ignore_index=self._ignore_index,
            dtype="uint8",
            labels=labels,
            options={"change_value": self._options.change_value},
        )

    def window(
        self,
        transform: Transform,
        height: int,
        width: int,
        valid_mask: npt.NDArray[np.bool_] | None,
    ) -> WindowTarget:
        value = self._options.change_value
        ignore = self._ignore_index
        if self._changed is not None:
            mask = _window_mask(self._changed, transform, height, width, valid_mask)
            change = np.where(mask == 0, 0, value).astype(np.uint8)
            if ignore is not None:
                # A label raster marks pixels without a label with the ignore value: keep it.
                change[mask == ignore] = ignore
        else:
            assert self._before is not None and self._after is not None
            before = _window_mask(self._before, transform, height, width, valid_mask)
            after = _window_mask(self._after, transform, height, width, valid_mask)
            change = np.where(before != after, value, 0).astype(np.uint8)
            if ignore is not None and self._annotated_areas:
                # Outside a set's annotated_area nothing was labeled (its mask holds the
                # ignore value there), so whether the pixel changed is unknown.
                change[(before == ignore) | (after == ignore)] = ignore
        return MaskWindow(change, ignore)
