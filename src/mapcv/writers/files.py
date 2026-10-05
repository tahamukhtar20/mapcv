"""One file per patch: PNG/JPG/NPY images in ``Images/`` and PNG masks in ``Masks/``."""

from __future__ import annotations

from typing import Any, Dict, List, Optional

import numpy as np
import numpy.typing as npt

from mapcv.imagery import RasterMetadata
from mapcv.sampler import PatchMeta
from mapcv.splitter import SplitLists
from mapcv.writer import Manifest, WriterConfig, write_patches


class FilesWriter:
    """The classic layout, byte-for-byte what :func:`mapcv.writer.write_patches` writes.

    Annotations must be ``None`` (image-only) or a ``(N, ps, ps)`` uint8 mask array.
    PNG/JPG patches are encoded by the Rust writer, NPY patches are stored bands-first.
    """

    def __init__(self, config: WriterConfig) -> None:
        self._config = config

    def fingerprint(self) -> Dict[str, Any]:
        return self._config.model_dump(mode="json", exclude={"staging_dir"})

    def patch_shape(self, source: RasterMetadata, patch_size: int) -> List[int]:
        if self._config.image_format == "npy":
            return [len(source.bands), patch_size, patch_size]
        return [patch_size, patch_size, 3]

    def write(
        self,
        images: npt.NDArray[np.generic],
        annotations: Optional[npt.NDArray[np.uint8]],
        metadata: List[PatchMeta],
        manifest: Manifest,
        chunk_index: int,
    ) -> None:
        if annotations is not None and (
            not isinstance(annotations, np.ndarray)
            or annotations.dtype != np.uint8
            or annotations.ndim != 3
        ):
            raise TypeError(
                "FilesWriter writes segmentation masks: annotations must be None or a "
                "(N, H, W) uint8 array"
            )
        write_patches(
            images, annotations, metadata, self._config, manifest, strip_index=chunk_index
        )

    def finalize(self, manifest: Manifest, split_lists: Optional[SplitLists]) -> None:
        return None
