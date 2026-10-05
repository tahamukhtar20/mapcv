"""Array helpers shared by the sampler and the targets (a leaf module: numpy only)."""

from __future__ import annotations

from typing import Any, Literal, Optional, Sequence, Tuple

import numpy as np
import numpy.typing as npt

PadMode = Literal["zero", "reflect"]


def extract_array_patch(
    array: npt.NDArray[Any],
    row: int,
    col: int,
    patch_size: int,
    pad_mode: PadMode,
    fill: int = 0,
) -> Tuple[npt.NDArray[Any], bool]:
    """Cut a ``patch_size`` square at ``(row, col)``, padding where it leaves ``array``.

    Returns the patch and whether any padding was needed.
    """
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
    return np.pad(chunk, pad_spec, mode="constant", constant_values=fill), True


class NullWindow:
    """Window of a dataset without targets: patches carry no annotation."""

    def annotate(
        self,
        row: int,
        col: int,
        patch_size: int,
        pad_mode: PadMode,
        valid_patch: Optional[npt.NDArray[np.bool_]],
    ) -> None:
        return None

    def accepts(self, annotation: Any, min_label_ratio: float) -> bool:
        return True

    def collate(self, annotations: Sequence[Any], patch_size: int) -> None:
        return None


class MaskWindow:
    """Window of a segmentation target: patches are cut out of a class mask.

    ``mask`` covers the same pixels as the image window. With ``ignore_index``,
    mask pixels without imagery (padding, and pixels outside ``valid_patch``) get
    that value instead of a class, and padding is never mirrored into labels.
    """

    def __init__(self, mask: npt.NDArray[np.uint8], ignore_index: Optional[int]) -> None:
        self._mask = mask
        self._ignore_index = ignore_index

    def annotate(
        self,
        row: int,
        col: int,
        patch_size: int,
        pad_mode: PadMode,
        valid_patch: Optional[npt.NDArray[np.bool_]],
    ) -> npt.NDArray[np.uint8]:
        ignore_index = self._ignore_index
        if ignore_index is None:
            extracted, _ = extract_array_patch(self._mask, row, col, patch_size, pad_mode)
            return extracted.astype(np.uint8, copy=False)
        # Never mirror labels into padding: pad with the ignore value.
        extracted, _ = extract_array_patch(
            self._mask, row, col, patch_size, "zero", fill=ignore_index
        )
        patch = extracted.astype(np.uint8, copy=True)
        if valid_patch is not None:
            patch[~valid_patch] = ignore_index
        return patch

    def accepts(self, annotation: npt.NDArray[np.uint8], min_label_ratio: float) -> bool:
        """Whether enough of the patch is labelled (not background, not ignored)."""
        if min_label_ratio <= 0.0:
            return True
        labeled = annotation != 0
        if self._ignore_index is not None:
            labeled &= annotation != self._ignore_index
        return float(np.count_nonzero(labeled) / annotation.size) >= min_label_ratio

    def collate(
        self, annotations: Sequence[npt.NDArray[np.uint8]], patch_size: int
    ) -> npt.NDArray[np.uint8]:
        """Stack the kept patches into an ``(N, ps, ps)`` array."""
        if not annotations:
            return np.empty((0, patch_size, patch_size), dtype=np.uint8)
        return np.stack(annotations, axis=0)
