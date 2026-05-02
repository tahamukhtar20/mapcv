"""Patch writer: saves image/mask patches to disk and maintains a manifest."""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Literal, Optional, TypedDict

import numpy as np
import numpy.typing as npt
from pydantic import BaseModel, Field

from mapcv._mapcv_rs import write_patches_rs
from mapcv.labels import ClassMap
from mapcv.sampler import PatchMeta


MANIFEST_VERSION: int = 1
_IMAGES_SUBDIR = "Images"
_MASKS_SUBDIR = "Masks"


class WriterConfig(BaseModel):
    """Configuration for writing patches to disk."""

    staging_dir: Path
    image_format: Literal["png", "jpg"] = "png"
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
    """Dataset manifest: header + per-patch records."""

    version: int = MANIFEST_VERSION
    class_map: ClassMap
    patches: List[ManifestEntry] = Field(default_factory=list)

    @classmethod
    def load(cls, path: Path) -> "Manifest":
        """Deserialize a manifest from JSON at *path*."""
        return cls.model_validate_json(path.read_text())

    def save(self, path: Path) -> None:
        """Serialize the manifest to indented JSON at *path*."""
        path.write_text(self.model_dump_json(indent=2))


def load_or_create_manifest(path: Path, class_map: ClassMap) -> Manifest:
    """Load an existing manifest or create a fresh one."""
    if path.exists():
        return Manifest.load(path)
    return Manifest(class_map=class_map)


def write_patches(
    image_patches: npt.NDArray[np.uint8],
    mask_patches: Optional[npt.NDArray[np.uint8]],
    meta: List[PatchMeta],
    config: WriterConfig,
    manifest: Manifest,
    strip_index: int = 0,
) -> None:
    """Write patches to staging dirs and extend *manifest* in-place.

    Files are named ``patch_{global_idx:07d}.{ext}`` where ``global_idx``
    starts at ``len(manifest.patches)`` so successive calls across strips
    produce a flat, collision-free namespace.

    If an image file already exists it is skipped (resume support). The
    corresponding manifest entry is still appended so the manifest stays
    consistent with what is on disk.
    """
    if len(meta) == 0:
        return

    images_dir = config.staging_dir / _IMAGES_SUBDIR
    masks_dir = config.staging_dir / _MASKS_SUBDIR
    images_dir.mkdir(parents=True, exist_ok=True)
    if mask_patches is not None:
        masks_dir.mkdir(parents=True, exist_ok=True)

    meta_tuples = [(m["row"], m["col"], m["padded"]) for m in meta]

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

    for fname, msk_fname, row, col, padded, si, class_counts, empty_ratio in results:
        manifest.patches.append(
            ManifestEntry(
                filename=fname,
                mask_filename=msk_fname,
                row=row,
                col=col,
                padded=padded,
                strip_index=si,
                per_class_pixel_counts={k: int(v) for k, v in class_counts.items()},
                empty_ratio=empty_ratio,
            )
        )
