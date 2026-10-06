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
from mapcv.manifest import IMAGES_DIR, MASKS_DIR
from mapcv.writer import Manifest, WriterConfig, write_patches, write_source_images


class FilesWriter:
    """The classic layout, byte-for-byte what :func:`mapcv.writer.write_patches` writes.

    Annotations must be ``None`` (image-only) or an ``(N, ps, ps)`` mask array: uint8
    (the segmentation target today), or uint16 and wider for the formats that hold them.
    PNG/JPG patches are encoded by the Rust writer, NPY and GeoTIFF patches are
    stored bands-first; masks follow ``writer.mask_format`` (PNG, NPY or GeoTIFF).
    GeoTIFF patches and masks of one patch share their georeferencing, and
    ``finalize`` writes ``patches.geojson``.

    With ``sources`` (the names of a multi-source dataset's imagery, ``imagery`` as a
    list), each source's patches go to ``Images/<name>/`` under the same file names,
    and each entry's ``files`` has one key per source name.
    """

    TARGET_TYPES: FrozenSet[Optional[str]] = frozenset({None, "segmentation"})

    def __init__(self, config: WriterConfig, sources: Optional[List[str]] = None) -> None:
        self._config = config
        self._sources = sources
        self._wrote = False

    def _images_dir(self, name: str) -> str:
        """The folder of source ``name``'s patches (several sources)."""
        return f"{IMAGES_DIR}/{name}"

    @property
    def _masks_dir(self) -> str:
        return MASKS_DIR

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
                mode="json", exclude={"staging_dir", "world_files", "footprints", "stack_sources"}
            ),
        }
        if self._config.world_files:
            block["world_files"] = True
        if self._config.stack_sources:
            block["stack_sources"] = True
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
        others: Optional[Dict[str, npt.NDArray[np.generic]]] = None,
    ) -> None:
        """Write one chunk; ``others`` holds the further sources' patches by name."""
        if annotations is not None and (
            not isinstance(annotations, np.ndarray) or annotations.ndim != 3
        ):
            raise TypeError(
                "FilesWriter writes segmentation masks: annotations must be None or an "
                "(N, H, W) array"
            )
        if self._sources is None:
            if others:
                raise ValueError("this writer was made for one imagery source")
            write_patches(
                images, annotations, metadata, self._config, manifest, chunk_index=chunk_index
            )
        else:
            self._write_sources(images, annotations, metadata, manifest, chunk_index, others or {})
        self._wrote = self._wrote or len(metadata) > 0

    def _write_sources(
        self,
        images: npt.NDArray[np.generic],
        annotations: Optional[npt.NDArray[Any]],
        metadata: List[PatchMeta],
        manifest: Manifest,
        chunk_index: int,
        others: Dict[str, npt.NDArray[np.generic]],
    ) -> None:
        sources = self._sources or []
        first, rest = sources[0], sources[1:]
        if sorted(others) != sorted(rest):
            raise ValueError(f"expected patches of imagery {rest}, got {sorted(others)}")
        records = {record.name: record for record in manifest.sources}
        if self._config.stack_sources:
            self._write_stack(images, annotations, metadata, manifest, chunk_index, others)
            return
        start = len(manifest.patches)
        write_patches(
            images,
            annotations,
            metadata,
            self._config,
            manifest,
            chunk_index=chunk_index,
            images_dir=self._images_dir(first),
            masks_dir=self._masks_dir,
            image_key=first,
            source=records[first],
        )
        entries = manifest.patches[start:]
        for name in rest:
            files = write_source_images(
                others[name],
                metadata,
                self._config,
                manifest,
                start,
                chunk_index,
                source=records[name],
                images_dir=self._images_dir(name),
            )
            for entry, extra in zip(entries, files):
                entry["files"].update(extra)
        for entry in entries:
            # Sources first, in imagery order, then the mask and the world files.
            files_in_order = {name: entry["files"][name] for name in sources}
            files_in_order.update(entry["files"])
            entry["files"] = files_in_order

    def _write_stack(
        self,
        images: npt.NDArray[np.generic],
        annotations: Optional[npt.NDArray[Any]],
        metadata: List[PatchMeta],
        manifest: Manifest,
        chunk_index: int,
        others: Dict[str, npt.NDArray[np.generic]],
    ) -> None:
        """One file per patch holding every source (``writer.stack_sources``).

        The sources' bands are concatenated in source order, so channel ``t * C + c`` is
        band ``c`` of source ``t``: NPY files are reshaped to ``(T, C, H, W)``, GeoTIFFs
        hold the ``T * C`` bands named ``<source>_<band>``.
        """
        sources = self._sources or []
        records = {record.name: record for record in manifest.sources}
        stacked = np.concatenate([images, *(others[name] for name in sources[1:])], axis=-1)
        write_patches(
            stacked,
            annotations,
            metadata,
            self._config,
            manifest,
            chunk_index=chunk_index,
            band_names=[f"{name}_{band}" for name in sources for band in records[name].bands],
            time_steps=len(sources),
        )

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
