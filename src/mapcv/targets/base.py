"""The target contract: what a dataset task attaches to each image patch."""

from __future__ import annotations

from typing import Any, Optional, Protocol, Sequence, Tuple

import numpy as np
import numpy.typing as npt

from mapcv._patching import PadMode
from mapcv.imagery import RasterMetadata
from mapcv.labels import ClassMap
from mapcv.manifest import TargetRecord

Transform = Tuple[float, float, float, float, float, float]

# One patch's target (a mask patch for segmentation, a list of boxes for detection, ...),
# and all kept patches of a chunk in the form the writer consumes. Only the target and
# the writer that supports it know what these are; the pipeline just hands them over.
Annotation = Any
AnnotationBatch = Any


class WindowTarget(Protocol):
    """A target bound to one imagery window (one chunk of the raster).

    Anchors passed to :meth:`annotate` are local to the window, in pixels.
    """

    def annotate(
        self,
        row: int,
        col: int,
        patch_size: int,
        pad_mode: PadMode,
        valid_patch: Optional[npt.NDArray[np.bool_]],
    ) -> Annotation:
        """The target of the ``patch_size`` square whose top-left pixel is ``(row, col)``.

        ``valid_patch`` marks the patch pixels that have imagery (``None`` when the
        source has no validity mask). Pixels outside the window are padding.
        """

    def accepts(self, annotation: Annotation, min_label_ratio: float) -> bool:
        """Whether a patch holds enough target to be kept (``sampler.min_label_ratio``)."""

    def collate(self, annotations: Sequence[Annotation], patch_size: int) -> AnnotationBatch:
        """Combine the annotations of the kept patches, in patch order, for the writer."""


class Target(Protocol):
    """What the dataset learns: parses its inputs once, then annotates patches per window."""

    @property
    def type(self) -> Optional[str]:
        """The manifest's ``target.type`` (``"segmentation"``), or ``None`` for no target.

        Known before :meth:`prepare`, so writers can be checked up front.
        """

    @property
    def class_map(self) -> ClassMap:
        """Class name to ID; valid after :meth:`prepare`."""

    def prepare(self, source: RasterMetadata) -> None:
        """Read and project the target inputs for the imagery described by ``source``.

        Raises on invalid inputs and warns when they cannot match the imagery.
        """

    def record(self) -> Optional[TargetRecord]:
        """The manifest's ``target`` block, or ``None``; a resumed run must reproduce it.

        Valid after :meth:`prepare`.
        """

    def window(
        self,
        transform: Transform,
        height: int,
        width: int,
        valid_mask: Optional[npt.NDArray[np.bool_]],
    ) -> WindowTarget:
        """Bind the target to a ``height`` x ``width`` window.

        ``transform`` maps the window's pixels to the raster CRS (it is already offset
        to the window origin) and ``valid_mask`` marks its pixels that have imagery.
        """
