"""Writing patches as files (``Images/``, ``Masks/``) and their manifest entries."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Literal, Optional

import numpy as np
import numpy.typing as npt
from PIL import Image
from pydantic import BaseModel, ConfigDict, Field

from mapcv._mapcv_rs import write_patches_rs
from mapcv.manifest import (
    IMAGES_DIR,
    MANIFEST_VERSION,
    MASKS_DIR,
    Manifest,
    ManifestEntry,
    ManifestMismatchError,
    PatchSummary,
    Transform,
    load_or_create_manifest,
)
from mapcv.sampler import PatchMeta

__all__ = [
    "MANIFEST_VERSION",
    "Manifest",
    "ManifestEntry",
    "ManifestMismatchError",
    "PatchSummary",
    "Transform",
    "WriterConfig",
    "load_or_create_manifest",
    "write_patches",
]


class WriterConfig(BaseModel):
    """Configuration for writing patches to disk."""

    # Unknown keys are errors, so typos and newer-version options are not silently ignored.
    model_config = ConfigDict(extra="forbid")

    staging_dir: Path
    image_format: Literal["png", "jpg", "npy"] = "png"
    jpg_quality: int = Field(default=95, ge=1, le=100)
    # 4:2:0 (Pillow's default) halves the chroma resolution for a much smaller file; 4:4:4 keeps it.
    jpg_subsampling: Literal["4:2:0", "4:4:4"] = "4:2:0"


def _entry(
    image_name: str,
    mask_name: Optional[str],
    row: int,
    col: int,
    padded: bool,
    chunk: int,
    counts: Optional[Dict[str, int]],
    empty_ratio: float,
) -> ManifestEntry:
    files = {"image": f"{IMAGES_DIR}/{image_name}"}
    summary = PatchSummary(empty_ratio=empty_ratio)
    if mask_name is not None:
        files["mask"] = f"{MASKS_DIR}/{mask_name}"
    if counts is not None:
        summary = PatchSummary(class_pixels=counts, empty_ratio=empty_ratio)
    return ManifestEntry(row=row, col=col, padded=padded, chunk=chunk, files=files, summary=summary)


def _class_counts(mask: npt.NDArray[np.uint8]) -> Dict[str, int]:
    # ``np.unique`` returns the values sorted ascending, so keys are numerically
    # ordered ("2" before "10") and identical runs serialise identically.
    values, counts = np.unique(mask, return_counts=True)
    return {str(int(value)): int(count) for value, count in zip(values, counts)}


def _empty_ratio(image: npt.NDArray[Any]) -> float:
    if image.ndim == 2:
        empty = ~np.isfinite(image) | (image == 0)
    else:
        empty = np.all(~np.isfinite(image), axis=-1)
    return float(np.count_nonzero(empty) / empty.size) if empty.size else 1.0


def _write_npy_patches(
    image_patches: npt.NDArray[Any],
    mask_patches: Optional[npt.NDArray[np.uint8]],
    meta: List[PatchMeta],
    config: WriterConfig,
    manifest: Manifest,
    chunk_index: int,
) -> None:
    images_dir = config.staging_dir / IMAGES_DIR
    masks_dir = config.staging_dir / MASKS_DIR
    images_dir.mkdir(parents=True, exist_ok=True)
    if mask_patches is not None:
        masks_dir.mkdir(parents=True, exist_ok=True)

    start_index = len(manifest.patches)
    for local_index, patch_meta in enumerate(meta):
        global_index = start_index + local_index
        filename = f"patch_{global_index:07d}.npy"
        image = image_patches[local_index]
        channels_first = image[np.newaxis, ...] if image.ndim == 2 else np.moveaxis(image, -1, 0)
        np.save(images_dir / filename, np.ascontiguousarray(channels_first), allow_pickle=False)

        mask_filename: Optional[str] = None
        counts: Optional[Dict[str, int]] = None
        if mask_patches is not None:
            mask = mask_patches[local_index]
            mask_filename = f"patch_{global_index:07d}.png"
            Image.fromarray(mask, mode="L").save(masks_dir / mask_filename, format="PNG")
            counts = _class_counts(mask)

        empty_ratio = (
            patch_meta["empty_ratio"] if "empty_ratio" in patch_meta else _empty_ratio(image)
        )
        manifest.patches.append(
            _entry(
                filename,
                mask_filename,
                patch_meta["row"],
                patch_meta["col"],
                patch_meta["padded"],
                chunk_index,
                counts,
                empty_ratio,
            )
        )


def write_patches(
    image_patches: npt.NDArray[Any],
    mask_patches: Optional[npt.NDArray[np.uint8]],
    meta: List[PatchMeta],
    config: WriterConfig,
    manifest: Manifest,
    chunk_index: int = 0,
) -> None:
    """Write patches to ``Images/`` and ``Masks/`` and append their entries to ``manifest``.

    PNG/JPG images retain the Rust-backed RGB writer. NPY images preserve an
    arbitrary channel count and dtype and are stored bands-first on disk. Masks
    are single-channel PNGs. Files are numbered from ``len(manifest.patches)``,
    so any existing file at those numbers is an orphan from an interrupted run
    and is overwritten.
    """
    if len(meta) == 0:
        return

    if config.image_format == "npy":
        _write_npy_patches(image_patches, mask_patches, meta, config, manifest, chunk_index)
        return

    if image_patches.dtype != np.uint8 or image_patches.ndim != 4 or image_patches.shape[-1] != 3:
        raise ValueError("PNG/JPG output requires uint8 image patches shaped (N, H, W, 3)")

    images_dir = config.staging_dir / IMAGES_DIR
    masks_dir = config.staging_dir / MASKS_DIR
    images_dir.mkdir(parents=True, exist_ok=True)
    if mask_patches is not None:
        masks_dir.mkdir(parents=True, exist_ok=True)

    meta_tuples = [(item["row"], item["col"], item["padded"]) for item in meta]
    results = write_patches_rs(
        np.ascontiguousarray(image_patches),
        np.ascontiguousarray(mask_patches) if mask_patches is not None else None,
        meta_tuples,
        len(manifest.patches),
        chunk_index,
        str(images_dir),
        str(masks_dir),
        config.image_format,
        config.jpg_quality,
        config.jpg_subsampling,
    )

    for fname, mask_fname, row, col, padded, chunk, counts, empty_ratio in results:
        manifest.patches.append(
            _entry(
                fname,
                mask_fname,
                row,
                col,
                padded,
                chunk,
                (
                    {str(class_id): int(count) for class_id, count in counts}
                    if mask_patches is not None
                    else None
                ),
                empty_ratio,
            )
        )
