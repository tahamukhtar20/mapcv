"""Patch sampler: extracts fixed-size patches from image strips."""

from __future__ import annotations

from typing import List, Literal, Optional, Tuple

from typing_extensions import TypedDict

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


def _extract_patch(
    arr: npt.NDArray[np.uint8],
    row: int,
    col: int,
    patch_size: int,
    pad_mode: Literal["zero", "reflect"],
) -> Tuple[npt.NDArray[np.uint8], bool]:
    h, w = arr.shape[:2]
    r0, r1 = max(row, 0), min(row + patch_size, h)
    c0, c1 = max(col, 0), min(col + patch_size, w)
    chunk = arr[r0:r1, c0:c1]

    pad_top = max(0, -row)
    pad_bottom = max(0, row + patch_size - h)
    pad_left = max(0, -col)
    pad_right = max(0, col + patch_size - w)

    if pad_top == 0 and pad_bottom == 0 and pad_left == 0 and pad_right == 0:
        return chunk, False

    if arr.ndim == 3:
        pad_spec = [(pad_top, pad_bottom), (pad_left, pad_right), (0, 0)]
    else:
        pad_spec = [(pad_top, pad_bottom), (pad_left, pad_right)]

    # Fall back to zero-padding when chunk is too small for reflect.
    use_reflect = pad_mode == "reflect" and chunk.shape[0] >= 2 and chunk.shape[1] >= 2
    if use_reflect:
        result = np.pad(chunk, pad_spec, mode="reflect")
    else:
        result = np.pad(chunk, pad_spec, mode="constant", constant_values=0)

    return result, True


def _passes_filters(
    img: npt.NDArray[np.uint8],
    mask: Optional[npt.NDArray[np.uint8]],
    config: SamplerConfig,
) -> bool:
    total = img.shape[0] * img.shape[1]
    if total == 0:
        return False
    if config.max_empty_ratio < 1.0:
        if img.ndim == 3:
            empty = int(np.count_nonzero(np.all(img == 0, axis=-1)))
        else:
            empty = int(np.count_nonzero(img == 0))
        if empty / total > config.max_empty_ratio:
            return False
    if config.min_label_ratio > 0.0 and mask is not None:
        if int(np.count_nonzero(mask)) / total < config.min_label_ratio:
            return False
    return True


def sample_patches(
    strip_image: npt.NDArray[np.uint8],
    strip_mask: Optional[npt.NDArray[np.uint8]],
    config: SamplerConfig,
) -> Tuple[
    npt.NDArray[np.uint8],
    Optional[npt.NDArray[np.uint8]],
    List[PatchMeta],
]:
    """Sample fixed-size patches from a strip image.

    Returns ``(image_patches, mask_patches, metadata)``.

    ``image_patches`` has shape ``(N, ps, ps)`` or ``(N, ps, ps, C)``.
    ``mask_patches`` has shape ``(N, ps, ps)`` when *strip_mask* is given,
    otherwise ``None``.
    """
    h, w = strip_image.shape[:2]
    ps = config.patch_size

    if config.mode == "random":
        raw_anchors: List[Tuple[int, int]] = list(
            random_sample_anchors(
                h, w, ps, config.random_count, config.random_seed, config.edge_strategy
            )
        )
    else:
        raw_anchors = list(
            grid_sample_anchors(h, w, ps, config.stride, config.edge_strategy)
        )

    img_list: List[npt.NDArray[np.uint8]] = []
    msk_list: List[npt.NDArray[np.uint8]] = []
    meta_list: List[PatchMeta] = []

    for row, col in raw_anchors:
        img_patch, padded = _extract_patch(strip_image, row, col, ps, config.pad_mode)
        msk_patch: Optional[npt.NDArray[np.uint8]] = None
        if strip_mask is not None:
            msk_patch, _ = _extract_patch(strip_mask, row, col, ps, config.pad_mode)

        if not _passes_filters(img_patch, msk_patch, config):
            continue

        img_list.append(img_patch)
        if msk_patch is not None:
            msk_list.append(msk_patch)
        meta_list.append(PatchMeta(row=row, col=col, padded=padded))

    if not img_list:
        empty_img: npt.NDArray[np.uint8]
        if strip_image.ndim == 3:
            empty_img = np.zeros((0, ps, ps, strip_image.shape[2]), dtype=np.uint8)
        else:
            empty_img = np.zeros((0, ps, ps), dtype=np.uint8)
        empty_msk = np.zeros((0, ps, ps), dtype=np.uint8) if strip_mask is not None else None
        return empty_img, empty_msk, meta_list

    stacked_img = np.stack(img_list, axis=0)
    stacked_msk: Optional[npt.NDArray[np.uint8]] = (
        np.stack(msk_list, axis=0) if msk_list else None
    )
    return stacked_img, stacked_msk, meta_list
