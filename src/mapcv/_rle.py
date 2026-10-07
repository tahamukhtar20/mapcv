"""COCO run-length encoding of binary masks (the format of pycocotools' ``mask.encode``).

A mask is read in column-major (Fortran) order, down each column and then to the next
one. Its run lengths alternate between background and foreground and always start with
background, so a mask that starts with a foreground pixel has a zero-length first run.
The counts are then written as the compressed string COCO files hold: every count after
the third is stored as the difference to the count two places before it (runs of equal
size cancel), least significant 5-bit group first, one printable character per group
(48 + the group, plus 32 when more groups follow); the top bit of the last group is the
sign.

The string encoding (:func:`counts_to_string`) follows ``rleToString`` of the COCO API's
``maskApi.c`` (Simplified BSD licence, see ``THIRD_PARTY_NOTICES.md``); the rest is an
independent numpy implementation. Tests check every output against pycocotools.
"""

from __future__ import annotations

import numpy as np
import numpy.typing as npt


def rle_counts_at(
    part: npt.NDArray[np.bool_], x: int, y: int, height: int, width: int
) -> list[int]:
    """Run lengths of a ``height`` x ``width`` mask that is ``part`` placed at ``(x, y)``.

    ``part`` is a boolean array whose top-left pixel is at column ``x``, row ``y`` of the
    mask; every other pixel of the mask is background, and ``part`` lies inside it. Only
    ``part`` is scanned, so the cost does not grow with the size of the mask.
    """
    rows, columns = part.shape
    if not rows or not columns or not part.any():
        return [height * width]
    # One background pixel below each column keeps the runs of different columns apart.
    guarded = np.zeros((columns, rows + 1), dtype=np.int8)
    guarded[:, :rows] = part.T
    steps = np.diff(guarded.reshape(-1), prepend=0, append=0)
    first = np.flatnonzero(steps == 1)
    last = np.flatnonzero(steps == -1) - 1  # the last pixel of each run
    column, row = np.divmod(first, rows + 1)
    starts = (x + column) * height + y + row
    column, row = np.divmod(last, rows + 1)
    ends = (x + column) * height + y + row + 1  # exclusive
    # The pixel below the last row of a column is the first of the next column when the
    # part spans the whole height: join the runs that touch.
    joined = np.concatenate(([True], starts[1:] != ends[:-1]))
    starts = starts[joined]
    ends = ends[np.concatenate((joined[1:], [True]))]
    counts = np.empty(2 * len(starts) + 1, dtype=np.int64)
    counts[0] = starts[0]
    counts[1::2] = ends - starts
    counts[2:-1:2] = starts[1:] - ends[:-1]
    counts[-1] = height * width - ends[-1]
    runs: list[int] = counts.tolist()
    if runs[-1] == 0:  # a mask that ends with foreground has no trailing background run
        runs.pop()
    return runs


def rle_counts(mask: npt.NDArray[np.bool_]) -> list[int]:
    """Run lengths of ``mask`` (``(height, width)``) in column-major order, background first."""
    if mask.ndim != 2:
        raise ValueError("an RLE mask must be a 2-D array")
    height, width = mask.shape
    if not mask.any():
        return [height * width] if height * width else []
    x, y, box_width, box_height = mask_bbox(mask)
    return rle_counts_at(mask[y : y + box_height, x : x + box_width], x, y, height, width)


def counts_to_string(counts: list[int]) -> str:
    """The compressed COCO string of run lengths ``counts``."""
    out: list[str] = []
    for index, count in enumerate(counts):
        value = count - counts[index - 2] if index > 2 else count
        more = True
        while more:
            group = value & 0x1F
            value >>= 5  # arithmetic shift, as in the reference implementation
            more = value != -1 if group & 0x10 else value != 0
            if more:
                group |= 0x20
            out.append(chr(group + 48))
    return "".join(out)


def encode_mask(mask: npt.NDArray[np.bool_]) -> str:
    """The compressed COCO RLE ``counts`` string of a binary ``(height, width)`` mask."""
    return counts_to_string(rle_counts(mask))


def encode_part(part: npt.NDArray[np.bool_], x: int, y: int, height: int, width: int) -> str:
    """The ``counts`` string of the ``height`` x ``width`` mask that is ``part`` at ``(x, y)``."""
    return counts_to_string(rle_counts_at(part, x, y, height, width))


def mask_bbox(mask: npt.NDArray[np.bool_]) -> tuple[int, int, int, int]:
    """Tight ``(x, y, width, height)`` box of the true pixels of ``mask``, in pixels.

    Raises:
        ValueError: The mask has no true pixel.
    """
    rows = np.flatnonzero(mask.any(axis=1))
    cols = np.flatnonzero(mask.any(axis=0))
    if not len(rows):
        raise ValueError("an empty mask has no bounding box")
    x0, x1 = int(cols[0]), int(cols[-1])
    y0, y1 = int(rows[0]), int(rows[-1])
    return x0, y0, x1 - x0 + 1, y1 - y0 + 1
