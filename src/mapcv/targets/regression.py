"""Regression: per-pixel float targets read from a continuous raster (``task: regression``).

Every imagery pixel takes the raster's value at its centre (nearest neighbour, the
lookup of classified label rasters; see :mod:`mapcv.targets.raster_labels`), scaled
as ``value * labels.scale + labels.offset``. Target patches are float32 with ``NaN``
where there is no value: outside the raster, its NoData and invalid values, and
pixels without imagery (padding included).
"""

from __future__ import annotations

import warnings
from collections.abc import Sequence
from typing import Any

import numpy as np
import numpy.typing as npt

from mapcv._patching import PadMode, extract_array_patch
from mapcv.config import ContinuousLabelsConfig
from mapcv.imagery import RasterMetadata, geotiff_fingerprint
from mapcv.labels import ClassMap
from mapcv.manifest import TargetRecord
from mapcv.targets.base import Transform, WindowTarget
from mapcv.targets.raster_labels import LABEL_RASTER_MISS_MESSAGE, ValueRasterSampler


class ValueWindow:
    """Window of a regression target: target patches are cut out of a float32 raster."""

    def __init__(self, values: npt.NDArray[np.float32]) -> None:
        self._values = values

    def annotate(
        self,
        row: int,
        col: int,
        patch_size: int,
        pad_mode: PadMode,
        valid_patch: npt.NDArray[np.bool_] | None,
    ) -> npt.NDArray[np.float32]:
        # Padding has no value whatever pad_mode does to the image: never mirror targets.
        extracted, _ = extract_array_patch(
            self._values, row, col, patch_size, "zero", fill=float("nan")
        )
        patch = extracted.astype(np.float32, copy=True)
        if valid_patch is not None:
            patch[~valid_patch] = np.nan
        return patch

    def accepts(self, annotation: npt.NDArray[np.float32], min_label_ratio: float) -> bool:
        """Whether enough of the patch has a target value (is not ``NaN``)."""
        if min_label_ratio <= 0.0:
            return True
        return float(np.count_nonzero(np.isfinite(annotation)) / annotation.size) >= (
            min_label_ratio
        )

    def collate(
        self, annotations: Sequence[npt.NDArray[np.float32]], patch_size: int
    ) -> npt.NDArray[np.float32]:
        """Stack the kept patches into an ``(N, ps, ps)`` float32 array."""
        if not annotations:
            return np.empty((0, patch_size, patch_size), dtype=np.float32)
        return np.stack(annotations, axis=0)


class RegressionTarget:
    """Float targets read from a continuous raster; see the module docs."""

    def __init__(self, labels: ContinuousLabelsConfig) -> None:
        self._labels = labels
        self._sampler: ValueRasterSampler | None = None
        self._fingerprint: dict[str, Any] | None = None

    @property
    def type(self) -> str | None:
        return "regression"

    @property
    def class_map(self) -> ClassMap:
        return {}

    @property
    def sampler(self) -> ValueRasterSampler:
        """The raster reader; valid after :meth:`prepare`."""
        if self._sampler is None:
            raise RuntimeError("RegressionTarget.prepare() must run first")
        return self._sampler

    def prepare(self, source: RasterMetadata) -> None:
        sampler = ValueRasterSampler(self._labels, source.crs)
        self._sampler = sampler
        self._fingerprint = geotiff_fingerprint(sampler.location)
        if not sampler.overlaps(source):
            warnings.warn(
                LABEL_RASTER_MISS_MESSAGE.format(value="NaN (no target)"),
                UserWarning,
                stacklevel=3,
            )

    def record(self) -> TargetRecord | None:
        """The target settings and a fingerprint of the raster, so a resumed run notices
        another file or other scaling."""
        if self._fingerprint is None:
            raise RuntimeError("RegressionTarget.prepare() must run first")
        settings = self._labels.model_dump(mode="json", exclude={"path"})
        settings["fingerprint"] = self._fingerprint
        return TargetRecord(
            type="regression",
            class_map={},
            ignore_index=None,
            dtype="float32",
            labels=settings,
            options={},
        )

    def window(
        self,
        transform: Transform,
        height: int,
        width: int,
        valid_mask: npt.NDArray[np.bool_] | None,
    ) -> WindowTarget:
        return ValueWindow(self.sampler.sample(transform, height, width))
