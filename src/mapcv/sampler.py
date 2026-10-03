"""Patch sampler: extracts fixed-size patches from image strips."""

from __future__ import annotations

from typing import Any, List, Literal, Optional, Sequence, Tuple

from typing_extensions import NotRequired, TypedDict

import numpy as np
import numpy.typing as npt
from pydantic import BaseModel, Field, model_validator

from mapcv._mapcv_rs import grid_sample_anchors, random_sample_anchors


class SamplerConfig(BaseModel):
    """Configuration for patch sampling from an image strip."""

    patch_size: int = Field(gt=0)
    stride: int = Field(default=0, ge=0)
    mode: Literal["grid", "random"] = "grid"
    edge_strategy: Literal["pad", "drop", "shift"] = "pad"
    pad_mode: Literal["zero", "reflect"] = "zero"
    max_empty_ratio: float = Field(default=1.0, ge=0.0, le=1.0)
    min_label_ratio: float = Field(default=0.0, ge=0.0, le=1.0)
    random_seed: int = Field(default=42, ge=0)
    random_count: int = Field(default=100, ge=0)

    @model_validator(mode="after")
    def _default_stride(self) -> "SamplerConfig":
        # stride=0 is the sentinel for "use patch_size" (non-overlapping grid).
        if self.stride == 0:
            self.stride = self.patch_size
        return self


class PatchMeta(TypedDict):
    """Per-patch metadata returned by sample_patches."""

    row: int
    col: int
    padded: bool
    empty_ratio: NotRequired[float]


def sample_patches(
    image: npt.NDArray[Any],
    mask: Optional[npt.NDArray[np.uint8]],
    config: SamplerConfig,
) -> Tuple[npt.NDArray[Any], Optional[npt.NDArray[np.uint8]], List[PatchMeta]]:
    """Sample fixed-size patches from an in-memory image.

    Anchors follow ``config`` (grid or random) over the whole image; emptiness
    is measured as all-zero pixels. Returns ``(image_patches, mask_patches,
    metadata)`` where image patches have shape ``(N, ps, ps)`` or
    ``(N, ps, ps, C)`` and mask patches ``(N, ps, ps)`` or ``None``.
    """
    height, width = image.shape[:2]
    ps = config.patch_size
    if config.mode == "random":
        anchors: List[Tuple[int, int]] = list(
            random_sample_anchors(
                height, width, ps, config.random_count, config.random_seed, config.edge_strategy
            )
        )
    else:
        anchors = list(grid_sample_anchors(height, width, ps, config.stride, config.edge_strategy))
    return sample_patches_at_anchors(image, mask, anchors, config)


def _extract_array_patch(
    array: npt.NDArray[Any],
    row: int,
    col: int,
    patch_size: int,
    pad_mode: Literal["zero", "reflect"],
) -> Tuple[npt.NDArray[Any], bool]:
    height, width = array.shape[:2]
    chunk = array[
        max(row, 0) : min(row + patch_size, height), max(col, 0) : min(col + patch_size, width)
    ]
    pad_top = max(0, -row)
    pad_left = max(0, -col)
    pad_bottom = max(0, row + patch_size - height)
    pad_right = max(0, col + patch_size - width)
    if pad_top == 0 and pad_left == 0 and pad_bottom == 0 and pad_right == 0:
        return chunk, False

    pad_spec = [(pad_top, pad_bottom), (pad_left, pad_right)]
    if array.ndim == 3:
        pad_spec.append((0, 0))
    use_reflect = pad_mode == "reflect" and chunk.shape[0] >= 2 and chunk.shape[1] >= 2
    if use_reflect:
        return np.pad(chunk, pad_spec, mode="reflect"), True
    return np.pad(chunk, pad_spec, mode="constant", constant_values=0), True


def sample_patches_at_anchors(
    image: npt.NDArray[Any],
    mask: Optional[npt.NDArray[np.uint8]],
    anchors: Sequence[Tuple[int, int]],
    config: SamplerConfig,
    *,
    row_offset: int = 0,
    col_offset: int = 0,
    valid_mask: Optional[npt.NDArray[np.bool_]] = None,
) -> Tuple[npt.NDArray[Any], Optional[npt.NDArray[np.uint8]], List[PatchMeta]]:
    """Extract configured patches at explicit local anchors.

    Metadata coordinates are translated to the global raster using the supplied
    offsets. When ``valid_mask`` is provided, its false pixels define imagery
    emptiness instead of treating numeric zero as NoData.
    """
    image_patches: List[npt.NDArray[Any]] = []
    mask_patches: List[npt.NDArray[np.uint8]] = []
    metadata: List[PatchMeta] = []
    patch_size = config.patch_size

    for row, col in anchors:
        image_patch, padded = _extract_array_patch(image, row, col, patch_size, config.pad_mode)
        mask_patch: Optional[npt.NDArray[np.uint8]] = None
        if mask is not None:
            extracted_mask, _ = _extract_array_patch(mask, row, col, patch_size, config.pad_mode)
            mask_patch = extracted_mask.astype(np.uint8, copy=False)

        if valid_mask is not None:
            valid_patch, _ = _extract_array_patch(valid_mask, row, col, patch_size, "zero")
            empty_ratio = float(1.0 - np.count_nonzero(valid_patch) / valid_patch.size)
        elif image_patch.ndim == 3:
            empty = np.all(image_patch == 0, axis=-1)
            empty_ratio = float(np.count_nonzero(empty) / empty.size)
        else:
            empty_ratio = float(np.count_nonzero(image_patch == 0) / image_patch.size)

        if empty_ratio > config.max_empty_ratio:
            continue
        if config.min_label_ratio > 0.0 and mask_patch is not None:
            labeled_ratio = float(np.count_nonzero(mask_patch) / mask_patch.size)
            if labeled_ratio < config.min_label_ratio:
                continue

        image_patches.append(image_patch)
        if mask_patch is not None:
            mask_patches.append(mask_patch)
        metadata.append(
            PatchMeta(
                row=row + row_offset,
                col=col + col_offset,
                padded=padded,
                empty_ratio=empty_ratio,
            )
        )

    if not image_patches:
        trailing_shape = image.shape[2:]
        empty_images = np.empty((0, patch_size, patch_size, *trailing_shape), dtype=image.dtype)
        empty_masks = (
            np.empty((0, patch_size, patch_size), dtype=np.uint8) if mask is not None else None
        )
        return empty_images, empty_masks, metadata

    stacked_images = np.stack(image_patches, axis=0)
    stacked_masks = np.stack(mask_patches, axis=0) if mask_patches else None
    return stacked_images, stacked_masks, metadata
