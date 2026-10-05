"""One file per patch: PNG/JPG/NPY/GeoTIFF images in ``Images/``, PNG/NPY/GeoTIFF masks in ``Masks/``."""

from __future__ import annotations

import warnings
from typing import Any, Dict, FrozenSet, List, Optional

import numpy as np
import numpy.typing as npt

from mapcv.footprints import FOOTPRINTS_FILENAME, write_footprints
from mapcv.imagery import RasterMetadata
from mapcv.sampler import PatchMeta
from mapcv.splitter import SplitLists
from mapcv.writer import Manifest, WriterConfig, write_patches


class FilesWriter:
    """The classic layout, byte-for-byte what :func:`mapcv.writer.write_patches` writes.

    Annotations must be ``None`` (image-only) or an ``(N, ps, ps)`` mask array: uint8
    (the segmentation target today), or uint16 and wider for the formats that hold them.
    PNG/JPG patches are encoded by the Rust writer, NPY and GeoTIFF patches are
    stored bands-first; masks follow ``writer.mask_format`` (PNG, NPY or GeoTIFF).
    GeoTIFF patches and masks of one patch share their georeferencing, and
    ``finalize`` writes ``patches.geojson``.
    """

    TARGET_TYPES: FrozenSet[Optional[str]] = frozenset({None, "segmentation"})

    def __init__(self, config: WriterConfig) -> None:
        self._config = config
        self._wrote = False

    @property
    def layout(self) -> str:
        return "files"

    def supports(self, target_type: Optional[str]) -> bool:
        return target_type in self.TARGET_TYPES

    def fingerprint(self) -> Dict[str, Any]:
        # Only what changes the files of a patch: a resumed run must write the same kind.
        # ``world_files`` appears only when on, so datasets from before it existed resume.
        block = {
            "layout": self.layout,
            **self._config.model_dump(
                mode="json", exclude={"staging_dir", "world_files", "footprints"}
            ),
        }
        if self._config.world_files:
            block["world_files"] = True
        return block

    def patch_shape(self, source: RasterMetadata, patch_size: int) -> List[int]:
        if self._config.image_format in ("npy", "tif"):
            return [len(source.bands), patch_size, patch_size]
        return [patch_size, patch_size, 3]

    def write(
        self,
        images: npt.NDArray[np.generic],
        annotations: Optional[npt.NDArray[Any]],
        metadata: List[PatchMeta],
        manifest: Manifest,
        chunk_index: int,
    ) -> None:
        if annotations is not None and (
            not isinstance(annotations, np.ndarray) or annotations.ndim != 3
        ):
            raise TypeError(
                "FilesWriter writes segmentation masks: annotations must be None or an "
                "(N, H, W) array"
            )
        write_patches(
            images, annotations, metadata, self._config, manifest, chunk_index=chunk_index
        )
        self._wrote = self._wrote or len(metadata) > 0

    def finalize(self, manifest: Manifest, split_lists: Optional[SplitLists]) -> None:
        if not self._config.footprints:
            return
        if manifest.upgraded_from is not None and not self._wrote:
            # A finished mapcv 0.1/0.2 dataset is left untouched by a resume that has nothing to do.
            return
        path = self._config.staging_dir / FOOTPRINTS_FILENAME
        try:
            write_footprints(manifest, split_lists, path)
        except (RuntimeError, ValueError) as exc:
            warnings.warn(
                f"{FOOTPRINTS_FILENAME} was not written: {exc}", UserWarning, stacklevel=2
            )
