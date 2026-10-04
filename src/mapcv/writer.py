"""Patch writers and versioned dataset manifests."""

from __future__ import annotations

import math
import os
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional, Tuple

import numpy as np
import numpy.typing as npt
from PIL import Image
from pydantic import BaseModel, Field
from typing_extensions import TypedDict

from mapcv._mapcv_rs import write_patches_rs
from mapcv.labels import ClassMap
from mapcv.sampler import PatchMeta


MANIFEST_VERSION: int = 2
_IMAGES_SUBDIR = "Images"
_MASKS_SUBDIR = "Masks"
Transform = Tuple[float, float, float, float, float, float]


class WriterConfig(BaseModel):
    """Configuration for writing patches to disk."""

    staging_dir: Path
    image_format: Literal["png", "jpg", "npy"] = "png"
    jpg_quality: int = Field(default=95, ge=1, le=100)


class ManifestEntry(TypedDict):
    """Per-patch record stored in the manifest."""

    filename: str
    mask_filename: Optional[str]
    row: int
    col: int
    padded: bool
    strip_index: int
    per_class_pixel_counts: Dict[str, int]
    empty_ratio: float


class Manifest(BaseModel):
    """Dataset manifest metadata and patch records.

    Version-1 manifests remain valid because all version-2 metadata fields have
    backward-compatible defaults.
    """

    version: int = MANIFEST_VERSION
    class_map: ClassMap
    source_type: str = "xyz"
    product_id: Optional[str] = None
    bands: List[str] = Field(default_factory=list)
    dtype: Optional[str] = None
    patch_shape: List[int] = Field(default_factory=list)
    crs: Optional[str] = None
    transform: Optional[Transform] = None
    sampler: Optional[Dict[str, Any]] = None
    patches: List[ManifestEntry] = Field(default_factory=list)

    @classmethod
    def load(cls, path: Path) -> "Manifest":
        """Deserialize a version-1 or version-2 manifest from JSON."""
        return cls.model_validate_json(path.read_text(encoding="utf-8"))

    def save(self, path: Path) -> None:
        """Atomically serialize the manifest to indented JSON."""
        tmp_path = path.with_name(path.name + ".tmp")
        tmp_path.write_text(self.model_dump_json(indent=2), encoding="utf-8")
        os.replace(tmp_path, path)


class ManifestMismatchError(ValueError):
    """An existing manifest cannot be resumed with the current configuration."""


def _resume_mismatches(manifest: Manifest, expected: Manifest) -> List[str]:
    fields = (
        "class_map",
        "source_type",
        "product_id",
        "bands",
        "dtype",
        "patch_shape",
        "crs",
        "sampler",
    )
    mismatches = [name for name in fields if getattr(manifest, name) != getattr(expected, name)]
    if manifest.transform is None or expected.transform is None:
        if manifest.transform != expected.transform:
            mismatches.append("transform")
    elif not all(
        math.isclose(a, b, rel_tol=1e-9, abs_tol=1e-9)
        for a, b in zip(manifest.transform, expected.transform)
    ):
        mismatches.append("transform")
    return mismatches


def load_or_create_manifest(
    path: Path,
    class_map: ClassMap,
    *,
    source_type: str = "xyz",
    product_id: Optional[str] = None,
    bands: Optional[List[str]] = None,
    dtype: Optional[str] = None,
    patch_shape: Optional[List[int]] = None,
    crs: Optional[str] = None,
    transform: Optional[Transform] = None,
    sampler: Optional[Dict[str, Any]] = None,
) -> Manifest:
    """Load a resumable manifest from ``path`` or create a version-2 manifest.

    Raises:
        ManifestMismatchError: The existing manifest is version 1, or was
            generated with different imagery, labels, or sampler settings.
    """
    expected = Manifest(
        class_map=class_map,
        source_type=source_type,
        product_id=product_id,
        bands=bands or [],
        dtype=dtype,
        patch_shape=patch_shape or [],
        crs=crs,
        transform=transform,
        sampler=sampler,
    )
    if not path.exists():
        return expected

    manifest = Manifest.load(path)
    if manifest.version < MANIFEST_VERSION:
        raise ManifestMismatchError(
            f"{path} is a version-{manifest.version} manifest from mapcv 0.1.x and cannot be "
            "resumed; generate into a new writer.staging_dir ('mapcv split' still reads it)"
        )
    mismatches = _resume_mismatches(manifest, expected)
    if mismatches:
        raise ManifestMismatchError(
            f"{path} was generated with a different configuration "
            f"({', '.join(mismatches)}); use a new writer.staging_dir or remove the old dataset"
        )
    return manifest


def _class_counts(mask: npt.NDArray[np.uint8]) -> Dict[str, int]:
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
    strip_index: int,
) -> None:
    images_dir = config.staging_dir / _IMAGES_SUBDIR
    masks_dir = config.staging_dir / _MASKS_SUBDIR
    images_dir.mkdir(parents=True, exist_ok=True)
    if mask_patches is not None:
        masks_dir.mkdir(parents=True, exist_ok=True)

    start_index = len(manifest.patches)
    for local_index, patch_meta in enumerate(meta):
        global_index = start_index + local_index
        filename = f"patch_{global_index:07d}.npy"
        image_path = images_dir / filename
        image = image_patches[local_index]
        channels_first = image[np.newaxis, ...] if image.ndim == 2 else np.moveaxis(image, -1, 0)
        np.save(image_path, np.ascontiguousarray(channels_first), allow_pickle=False)

        mask_filename: Optional[str] = None
        counts: Dict[str, int] = {}
        if mask_patches is not None:
            mask = mask_patches[local_index]
            mask_filename = f"patch_{global_index:07d}.png"
            mask_path = masks_dir / mask_filename
            Image.fromarray(mask, mode="L").save(mask_path, format="PNG")
            counts = _class_counts(mask)

        manifest.patches.append(
            ManifestEntry(
                filename=filename,
                mask_filename=mask_filename,
                row=patch_meta["row"],
                col=patch_meta["col"],
                padded=patch_meta["padded"],
                strip_index=strip_index,
                per_class_pixel_counts=counts,
                empty_ratio=(
                    patch_meta["empty_ratio"]
                    if "empty_ratio" in patch_meta
                    else _empty_ratio(image)
                ),
            )
        )


def write_patches(
    image_patches: npt.NDArray[Any],
    mask_patches: Optional[npt.NDArray[np.uint8]],
    meta: List[PatchMeta],
    config: WriterConfig,
    manifest: Manifest,
    strip_index: int = 0,
) -> None:
    """Write patches to staging directories and extend ``manifest`` in place.

    PNG/JPG images retain the Rust-backed RGB writer. NPY images preserve an
    arbitrary channel count and dtype and are stored bands-first on disk.
    Files are indexed from ``len(manifest.patches)``, so any existing file at
    those indices is an orphan from an interrupted run and is overwritten.
    """
    if len(meta) == 0:
        return

    if config.image_format == "npy":
        _write_npy_patches(image_patches, mask_patches, meta, config, manifest, strip_index)
        return

    if image_patches.dtype != np.uint8 or image_patches.ndim != 4 or image_patches.shape[-1] != 3:
        raise ValueError("PNG/JPG output requires uint8 image patches shaped (N, H, W, 3)")

    images_dir = config.staging_dir / _IMAGES_SUBDIR
    masks_dir = config.staging_dir / _MASKS_SUBDIR
    images_dir.mkdir(parents=True, exist_ok=True)
    if mask_patches is not None:
        masks_dir.mkdir(parents=True, exist_ok=True)

    meta_tuples = [(item["row"], item["col"], item["padded"]) for item in meta]
    results = write_patches_rs(
        np.ascontiguousarray(image_patches),
        np.ascontiguousarray(mask_patches) if mask_patches is not None else None,
        meta_tuples,
        len(manifest.patches),
        strip_index,
        str(images_dir),
        str(masks_dir),
        config.image_format,
        config.jpg_quality,
    )

    for fname, mask_fname, row, col, padded, chunk, counts, empty_ratio in results:
        manifest.patches.append(
            ManifestEntry(
                filename=fname,
                mask_filename=mask_fname,
                row=row,
                col=col,
                padded=padded,
                strip_index=chunk,
                per_class_pixel_counts={key: int(value) for key, value in counts.items()},
                empty_ratio=empty_ratio,
            )
        )
