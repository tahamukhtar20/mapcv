"""The writer contract: how patches and their annotations reach disk."""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Protocol

import numpy as np
import numpy.typing as npt

from mapcv.imagery import RasterMetadata
from mapcv.sampler import PatchMeta
from mapcv.splitter import SplitLists
from mapcv.targets.base import AnnotationBatch
from mapcv.writer import Manifest


class Writer(Protocol):
    """An output layout. It owns the files and appends one manifest row per patch."""

    @property
    def layout(self) -> str:
        """The layout's name, recorded as the manifest's ``writer.layout``."""

    def supports(self, target_type: Optional[str]) -> bool:
        """Whether this layout can store a target's annotations (``None``: no target)."""

    def fingerprint(self) -> Dict[str, Any]:
        """The manifest's ``writer`` block (``layout`` first); a resumed run must reproduce it."""

    def patch_shape(self, source: RasterMetadata, patch_size: int) -> List[int]:
        """Shape of one stored image patch (the manifest's ``patch_shape``)."""

    def write(
        self,
        images: npt.NDArray[np.generic],
        annotations: AnnotationBatch,
        metadata: List[PatchMeta],
        manifest: Manifest,
        chunk_index: int,
    ) -> None:
        """Write one chunk's kept patches and append their entries to ``manifest.patches``.

        ``annotations`` is whatever the target's ``collate`` returned for these
        patches. Each entry records the patch's ``files`` (paths relative to the
        dataset folder) and a ``summary`` of its annotation. The first new patch is
        numbered ``len(manifest.patches)``, so files of an interrupted run are
        overwritten. The pipeline saves the manifest after each call.
        """

    def finalize(self, manifest: Manifest, split_lists: Optional[SplitLists]) -> None:
        """Called once after the last chunk and the split (``None`` without a split)."""
