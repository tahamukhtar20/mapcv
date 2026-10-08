"""Patch sampler: extracts fixed-size patches from image strips."""

from __future__ import annotations

import warnings
from collections import Counter
from collections.abc import Sequence
from typing import TYPE_CHECKING, Any, Literal

import numpy as np
import numpy.typing as npt
from pydantic import BaseModel, ConfigDict, Field, model_validator
from typing_extensions import NotRequired, TypedDict

from mapcv._mapcv_rs import grid_sample_anchors, random_anchor_capacity, random_sample_anchors
from mapcv._patching import MaskWindow, NullWindow, extract_array_patch

if TYPE_CHECKING:  # the targets package imports the config, which imports this module
    from mapcv.targets.base import Annotation, WindowTarget


class SamplerConfig(BaseModel):
    """Configuration for cutting patches from the raster (``sampler`` in the config)."""

    # Unknown keys are errors, so typos and newer-version options are not silently ignored.
    model_config = ConfigDict(extra="forbid")

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
    def _default_stride(self) -> SamplerConfig:
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


def random_anchors_for(height: int, width: int, config: SamplerConfig) -> list[tuple[int, int]]:
    """Distinct random anchors for ``config``, warning when fewer than requested exist.

    Anchors are sampled without replacement, so ``random_count`` is capped at
    the number of distinct positions on the raster; asking for more returns
    all of them and emits a ``UserWarning`` saying so.
    """
    anchors: list[tuple[int, int]] = list(
        random_sample_anchors(
            height,
            width,
            config.patch_size,
            config.random_count,
            config.random_seed,
            config.edge_strategy,
        )
    )
    if len(anchors) < config.random_count:
        warnings.warn(
            f"sampler.random_count is {config.random_count} but only {len(anchors)} distinct "
            f"patch position(s) exist on a {height}x{width} raster with patch_size "
            f"{config.patch_size} and edge_strategy {config.edge_strategy!r}; "
            f"using {'all of them' if anchors else 'none'}. Lower random_count or enlarge the "
            "region" + ("." if anchors else " (or use edge_strategy: pad)."),
            UserWarning,
            stacklevel=3,
        )
    return anchors


def random_patch_capacity(height: int, width: int, config: SamplerConfig) -> int:
    """Number of patches ``random_count`` can actually yield on this raster."""
    capacity = int(random_anchor_capacity(height, width, config.patch_size, config.edge_strategy))
    return min(config.random_count, capacity)


def sample_patches(
    image: npt.NDArray[Any],
    mask: npt.NDArray[np.uint8] | None,
    config: SamplerConfig,
) -> tuple[npt.NDArray[Any], npt.NDArray[np.uint8] | None, list[PatchMeta]]:
    """Sample fixed-size patches from an in-memory image.

    Anchors follow ``config`` (grid or random) over the whole image; emptiness
    is measured as all-zero pixels. Returns ``(image_patches, mask_patches,
    metadata)`` where image patches have shape ``(N, ps, ps)`` or
    ``(N, ps, ps, C)`` and mask patches ``(N, ps, ps)`` or ``None``.
    """
    height, width = image.shape[:2]
    ps = config.patch_size
    if config.mode == "random":
        anchors: list[tuple[int, int]] = random_anchors_for(height, width, config)
    else:
        anchors = list(grid_sample_anchors(height, width, ps, config.stride, config.edge_strategy))
    return sample_patches_at_anchors(image, mask, anchors, config)


def sample_annotated_patches(
    image: npt.NDArray[Any],
    anchors: Sequence[tuple[int, int]],
    config: SamplerConfig,
    window: WindowTarget,
    *,
    row_offset: int = 0,
    col_offset: int = 0,
    valid_mask: npt.NDArray[np.bool_] | None = None,
    counts: Counter[str] | None = None,
) -> tuple[npt.NDArray[Any], list[Annotation], list[PatchMeta]]:
    """Extract configured patches at explicit local anchors, with their annotations.

    Each kept patch gets ``window.annotate(...)``; ``sampler.max_empty_ratio`` and
    ``window.accepts(..., sampler.min_label_ratio)`` decide which patches are kept.
    With ``valid_mask``, a patch without a single pixel of imagery is never kept,
    whatever ``max_empty_ratio`` is; ``counts["no_imagery"]`` counts them.
    Returns ``(image_patches, annotations, metadata)`` in anchor order; the
    annotations are what the window made of each kept patch, not yet collated.

    Metadata coordinates are translated to the global raster using the supplied
    offsets. When ``valid_mask`` is provided, its false pixels define imagery
    emptiness instead of treating numeric zero as NoData.
    """
    image_patches: list[npt.NDArray[Any]] = []
    annotations: list[Annotation] = []
    metadata: list[PatchMeta] = []
    patch_size = config.patch_size

    for row, col in anchors:
        image_patch, padded = extract_array_patch(image, row, col, patch_size, config.pad_mode)
        valid_patch: npt.NDArray[np.bool_] | None = None
        if valid_mask is not None:
            valid_patch, _ = extract_array_patch(valid_mask, row, col, patch_size, "zero")

        if valid_patch is not None:
            # empty / size, not 1 - valid / size: the same division the Rust PNG writer made.
            empty_ratio = float(np.count_nonzero(~valid_patch) / valid_patch.size)
        elif image_patch.ndim == 3:
            empty = np.all(image_patch == 0, axis=-1)
            empty_ratio = float(np.count_nonzero(empty) / empty.size)
        else:
            empty_ratio = float(np.count_nonzero(image_patch == 0) / image_patch.size)

        if valid_patch is not None and empty_ratio >= 1.0:
            # Nothing to learn from: all failed tiles, NoData or outside the imagery.
            if counts is not None:
                counts["no_imagery"] += 1
            continue
        if empty_ratio > config.max_empty_ratio:
            continue
        annotation = window.annotate(row, col, patch_size, config.pad_mode, valid_patch)
        if not window.accepts(annotation, config.min_label_ratio):
            continue

        image_patches.append(image_patch)
        annotations.append(annotation)
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
        return empty_images, annotations, metadata
    return np.stack(image_patches, axis=0), annotations, metadata


def sample_patches_at_anchors(
    image: npt.NDArray[Any],
    mask: npt.NDArray[np.uint8] | None,
    anchors: Sequence[tuple[int, int]],
    config: SamplerConfig,
    *,
    row_offset: int = 0,
    col_offset: int = 0,
    valid_mask: npt.NDArray[np.bool_] | None = None,
    ignore_index: int | None = None,
) -> tuple[npt.NDArray[Any], npt.NDArray[np.uint8] | None, list[PatchMeta]]:
    """Extract configured patches at explicit local anchors.

    Metadata coordinates are translated to the global raster using the supplied
    offsets. When ``valid_mask`` is provided, its false pixels define imagery
    emptiness instead of treating numeric zero as NoData. With ``ignore_index``,
    mask pixels without imagery (padding, and invalid pixels when ``valid_mask``
    is given) get that value instead of a class, so losses can skip them.

    This is :func:`sample_annotated_patches` with a segmentation mask as the target.
    """
    window: WindowTarget = NullWindow() if mask is None else MaskWindow(mask, ignore_index)
    images, annotations, metadata = sample_annotated_patches(
        image,
        anchors,
        config,
        window,
        row_offset=row_offset,
        col_offset=col_offset,
        valid_mask=valid_mask,
    )
    return images, window.collate(annotations, config.patch_size), metadata
