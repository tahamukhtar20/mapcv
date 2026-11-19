"""The Rust extension in a child made by ``fork`` (a PyTorch ``DataLoader`` worker on
Linux before Python 3.14) after the parent already used it: the child reads and writes
GeoTIFF patches and decodes tiles on its own threads instead of waiting forever for the
parent's, which ``fork`` does not copy."""

from __future__ import annotations

import io
import multiprocessing
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import numpy.typing as npt
import pytest
from PIL import Image

from mapcv._mapcv_rs import decode_tile_window, write_geotiffs
from mapcv.data import read_image, read_mask

pytestmark = pytest.mark.skipif(
    sys.platform != "linux", reason="fork is the default start method only on Linux"
)

N, SIZE = 4, 64
TRANSFORM = (0.5, 0.0, 1000.0, 0.0, -0.5, 2000.0)


def _write(directory: Path, prefix: str, array: npt.NDArray[Any]) -> list[Path]:
    n, h, w, c = array.shape
    names = [f"{prefix}_{i}.tif" for i in range(n)]
    write_geotiffs(
        np.ascontiguousarray(array).reshape(-1).view(np.uint8),
        array.dtype.name,
        (n, h, w, c),
        [TRANSFORM] * n,
        names,
        str(directory),
        32631,
        False,
    )
    return [directory / name for name in names]


def _tile() -> bytes:
    buffer = io.BytesIO()
    pixels = np.random.default_rng(2).integers(0, 255, (256, 256, 3), dtype=np.uint8)
    Image.fromarray(pixels).save(buffer, "PNG")
    return buffer.getvalue()


def _work(directory: Path, images: list[Path], masks: list[Path]) -> list[float]:
    """Everything a worker might do: read image and mask patches, write and read back a
    GeoTIFF, decode a window of tiles. Returns one checksum per step."""
    sums = [float(read_image(path).astype(np.float64).sum()) for path in images]
    sums += [float(np.nansum(read_mask(path).astype(np.float64))) for path in masks]
    written = _write(directory, f"again_{multiprocessing.current_process().pid}", _images())
    sums += [float(read_image(path).astype(np.float64).sum()) for path in written]
    window, _, undecoded = decode_tile_window([(0, 0, _tile())], 0, 0, 0, 256, 0, 256)
    sums.append(float(window.astype(np.float64).sum()) + len(undecoded))
    return sums


def _images() -> npt.NDArray[np.uint8]:
    return np.random.default_rng(0).integers(0, 255, (N, SIZE, SIZE, 3), dtype=np.uint8)


def _child(directory: Path, images: list[Path], masks: list[Path], queue: Any) -> None:
    queue.put(_work(directory, images, masks))


@pytest.mark.filterwarnings("ignore:This process .* is multi-threaded:DeprecationWarning")
def test_a_forked_child_reads_and_writes_after_the_parent_did(tmp_path: Path) -> None:
    images = _write(tmp_path, "image", _images())
    targets = np.random.default_rng(1).uniform(0, 40, (N, SIZE, SIZE, 1)).astype(np.float32)
    targets[0, :8] = np.nan
    # A raster of several strips, which is decoded on the thread pool (a one-strip
    # patch is decoded on the calling thread).
    large = np.random.default_rng(3).uniform(0, 1, (1, 1024, 1024, 1)).astype(np.float32)
    masks = _write(tmp_path, "target", targets) + _write(tmp_path, "large", large)
    # The parent's threads exist before the fork, as when `train_ds[0]` or the
    # in-memory normalisation statistics read patches before the DataLoader starts.
    expected = _work(tmp_path, images, masks)
    context = multiprocessing.get_context("fork")
    queue = context.Queue()
    children = [
        context.Process(target=_child, args=(tmp_path, images, masks, queue)) for _ in range(2)
    ]
    for child in children:
        child.start()
    deadline = time.monotonic() + 30
    for child in children:
        child.join(timeout=max(0.0, deadline - time.monotonic()))
    hung = [child.pid for child in children if child.is_alive()]
    for child in children:
        if child.is_alive():
            child.kill()
    assert not hung, "a forked child hung on the parent's worker threads"
    assert [child.exitcode for child in children] == [0, 0]
    assert [queue.get(timeout=5) for _ in children] == [expected, expected]
