"""Patch writer: saves image/mask patches to disk and maintains a manifest."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Dict, List, Literal, Optional, Tuple, TypedDict

import numpy as np
import numpy.typing as npt
from PIL import Image
from pydantic import BaseModel, Field

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
    num_workers: int = Field(default=4, ge=1)


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
        return cls.model_validate_json(path.read_text())

    def save(self, path: Path) -> None:
        path.write_text(self.model_dump_json(indent=2))


def load_or_create_manifest(path: Path, class_map: ClassMap) -> Manifest:
    """Load an existing manifest or create a fresh one."""
    if path.exists():
        return Manifest.load(path)
    return Manifest(class_map=class_map)


def _class_counts(mask: npt.NDArray[np.uint8]) -> Dict[str, int]:
    unique, counts = np.unique(mask, return_counts=True)
    return {str(int(u)): int(c) for u, c in zip(unique, counts)}


def _empty_ratio(img: npt.NDArray[np.uint8]) -> float:
    total: int = img.shape[0] * img.shape[1]
    if total == 0:
        return 0.0
    if img.ndim == 3:
        empty = int(np.count_nonzero(np.all(img == 0, axis=-1)))
    else:
        empty = int(np.count_nonzero(img == 0))
    return float(empty) / float(total)


def _save_image(
    arr: npt.NDArray[np.uint8],
    path: Path,
    fmt: str,
    jpg_quality: int,
) -> None:
    img = Image.fromarray(arr)
    if fmt == "jpg":
        img.save(path, quality=jpg_quality)
    else:
        img.save(path)


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
    has_masks = mask_patches is not None
    if has_masks:
        masks_dir.mkdir(parents=True, exist_ok=True)

    ext = config.image_format
    start_idx = len(manifest.patches)

    # Build the work list: one item per patch
    WorkItem = Tuple[int, int, str, Path, Optional[Path], PatchMeta]
    items: List[WorkItem] = []
    for local_i, pm in enumerate(meta):
        global_idx = start_idx + local_i
        fname = f"patch_{global_idx:07d}.{ext}"
        img_path = images_dir / fname
        msk_path = (masks_dir / fname) if has_masks else None
        items.append((local_i, global_idx, fname, img_path, msk_path, pm))

    def _process(item: WorkItem) -> Tuple[int, ManifestEntry]:
        local_i, global_idx, fname, img_path, msk_path, pm = item
        img_arr = image_patches[local_i]

        if not img_path.exists():
            _save_image(img_arr, img_path, ext, config.jpg_quality)

        mask_filename: Optional[str] = None
        if msk_path is not None and mask_patches is not None:
            mask_filename = fname
            if not msk_path.exists():
                _save_image(mask_patches[local_i], msk_path, ext, config.jpg_quality)

        counts: Dict[str, int] = {}
        if mask_patches is not None:
            counts = _class_counts(mask_patches[local_i])

        return global_idx, ManifestEntry(
            filename=fname,
            mask_filename=mask_filename,
            row=pm["row"],
            col=pm["col"],
            padded=pm["padded"],
            strip_index=strip_index,
            per_class_pixel_counts=counts,
            empty_ratio=_empty_ratio(img_arr),
        )

    results: Dict[int, ManifestEntry] = {}
    with ThreadPoolExecutor(max_workers=config.num_workers) as pool:
        futures = {pool.submit(_process, item): item[1] for item in items}
        for fut in as_completed(futures):
            global_idx, entry = fut.result()
            results[global_idx] = entry

    # Append in deterministic order regardless of thread completion order.
    for _, global_idx, *_ in items:
        manifest.patches.append(results[global_idx])
