"""Finding what a resumed run has written costs the same however much it has written.

A resume looks up every chunk's anchors among the patches already in the manifest.
That lookup used to sort all the written positions once per chunk, so a run with
millions of patches written spent most of its restart on it before writing anything.
"""

from __future__ import annotations

import tracemalloc

import numpy as np
import numpy.typing as npt
import pytest

from mapcv.pipeline import _completed_positions, _GridBlock, _Positions, _remaining

WIDTH = 1_000


def _reference(chunk: _GridBlock | _Positions, written: set[int]) -> list[int]:
    """The positions of ``chunk`` not in ``written``, in order (a set lookup)."""
    return [int(value) for value in chunk.positions() if int(value) not in written]


def _array(values: list[int]) -> npt.NDArray[np.uint64]:
    return np.array(values, dtype=np.uint64)


def test_the_completed_positions_are_sorted_and_unique() -> None:
    patches = [
        {"row": 3, "col": 4},
        {"row": 0, "col": 9},
        {"row": 3, "col": 4},
        {"row": 1, "col": 0},
    ]
    done = _completed_positions(patches, WIDTH)
    assert done.dtype == np.uint64
    assert done.tolist() == [9, 1_000, 3_004]
    assert len(_completed_positions([], WIDTH)) == 0


@pytest.mark.parametrize("seed", range(8))
def test_the_remaining_anchors_are_those_a_set_lookup_leaves(seed: int) -> None:
    rng = np.random.default_rng(seed)
    rows = np.sort(rng.choice(200, size=12, replace=False)).astype(np.uint64)
    cols = np.sort(rng.choice(WIDTH, size=30, replace=False)).astype(np.uint64)
    block = _GridBlock(rows, cols, WIDTH)
    # Random positions come in the sampled order, not sorted.
    sampled = rng.choice(200 * WIDTH, size=400, replace=False).astype(np.uint64)
    chunks: list[_GridBlock | _Positions] = [block, _Positions(sampled, WIDTH)]
    for share in (0.0, 0.1, 0.5, 1.0):
        pool = np.concatenate(
            [block.positions(), sampled, rng.integers(0, 300 * WIDTH, 50, dtype=np.uint64)]
        )
        written = rng.choice(pool, size=int(len(pool) * share), replace=False)
        # The manifest may list a position twice or hold ones the chunk never had.
        patches = [
            {"row": int(value) // WIDTH, "col": int(value) % WIDTH}
            for value in np.concatenate([written, written[:5]])
        ]
        done = _completed_positions(patches, WIDTH)
        for chunk in chunks:
            expected = _reference(chunk, {int(value) for value in written})
            left = _remaining(chunk, done)
            assert left.positions().tolist() == expected
            assert left.anchors() == [(value // WIDTH, value % WIDTH) for value in expected]


def test_a_chunk_with_nothing_written_in_its_range_is_returned_whole() -> None:
    block = _GridBlock(_array([10, 11]), _array([5, 6, 7]), WIDTH)
    done = _array([3, 9_999, 12_000, 50_000])  # all below or above the block's range
    assert _remaining(block, done) is block
    assert _remaining(block, _array([])) is block
    inside = _array([10_100])  # in range, not an anchor of the block
    assert _remaining(block, inside) is block


def test_a_chunk_out_of_range_of_what_is_written_is_not_built(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def built(self: _GridBlock) -> npt.NDArray[np.uint64]:
        raise AssertionError("the positions of the chunk were built")

    monkeypatch.setattr(_GridBlock, "positions", built)
    block = _GridBlock(_array([10, 11]), _array([5, 6, 7]), WIDTH)
    assert _remaining(block, _array([3, 9_999, 12_000])) is block


def test_a_chunk_that_is_all_written_has_nothing_left() -> None:
    block = _GridBlock(_array([10, 11]), _array([5, 6, 7]), WIDTH)
    assert len(_remaining(block, np.sort(block.positions()))) == 0


@pytest.mark.parametrize("shape", ["bands", "scattered"])
def test_the_lookup_does_not_grow_with_what_is_written(shape: str) -> None:
    """Looking up 1,000 chunks of 5,000 anchors among 1,000,000 written patches allocates
    memory in proportion to a chunk, never to what is written.

    Sorting all the written positions again for every chunk (the old lookup) copies them
    each time, 8 MB here; the bisection allocates a few arrays the size of the chunk. The
    memory is counted rather than the time taken, so a slow machine cannot fail the test."""
    rng = np.random.default_rng(0)
    written = np.arange(1_000_000, dtype=np.uint64) * np.uint64(2)
    span = int(written[-1]) + 1
    edges = np.linspace(0, span * 1.2, 1_001).astype(np.uint64)
    chunks = []
    for index in range(1_000):
        if shape == "bands":  # a chunk is a band of rows
            values = np.sort(rng.integers(edges[index], edges[index + 1], 5_000, dtype=np.uint64))
        else:  # random anchors can land anywhere
            values = rng.integers(0, int(span * 1.2), 5_000, dtype=np.uint64)
        chunks.append(_Positions(values, WIDTH))
    done = np.unique(written)  # built once, as run_generate does
    # Shared by every chunk, so sorting it in place is not a way around the copy either.
    done.flags.writeable = False
    was_tracing = tracemalloc.is_tracing()
    if not was_tracing:
        tracemalloc.start()
    kept = 0
    peak = 0
    try:
        for chunk in chunks:
            tracemalloc.reset_peak()
            before = tracemalloc.get_traced_memory()[0]
            left = _remaining(chunk, done)
            peak = max(peak, tracemalloc.get_traced_memory()[1] - before)
            kept += len(left)
            del left
    finally:
        if not was_tracing:
            tracemalloc.stop()
    chunk_bytes = 5_000 * np.dtype(np.uint64).itemsize
    # About 5 chunk-sized arrays today; a copy of what is written would be 200 of them.
    assert peak < 16 * chunk_bytes, f"{peak:,} bytes for a chunk of {chunk_bytes:,}"
    # Even positions are written, so about half the anchors are left.
    assert 0.4 < kept / 5_000_000 < 0.6
