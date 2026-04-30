"""Tests for the M5 patch sampler (Rust anchors + Python extraction)."""

from __future__ import annotations

import numpy as np
import pytest

from mapcv import SamplerConfig, sample_patches
from mapcv._mapcv_rs import grid_sample_anchors, random_sample_anchors


def test_grid_drop_only_full_patches() -> None:
    anchors = grid_sample_anchors(10, 10, 4, 4, "drop")
    assert len(anchors) == 4  # 2 rows x 2 cols
    for r, c in anchors:
        assert r + 4 <= 10 and c + 4 <= 10


def test_grid_pad_includes_partial_edge_anchors() -> None:
    anchors = grid_sample_anchors(10, 10, 4, 4, "pad")
    # rows: 0, 4, 8 -- cols: 0, 4, 8 -> 9 patches
    assert len(anchors) == 9


def test_grid_shift_all_in_bounds() -> None:
    anchors = grid_sample_anchors(10, 10, 4, 4, "shift")
    for r, c in anchors:
        assert r + 4 <= 10 and c + 4 <= 10
    # Last anchor must reach the last pixels.
    max_row = max(r for r, _ in anchors)
    assert max_row == 6  # 10 - 4


def test_grid_exact_fit_no_extra_shift_anchor() -> None:
    # 8x8, patch=4, stride=4: anchors at 0 and 4 only -- no shift needed.
    anchors = grid_sample_anchors(8, 8, 4, 4, "shift")
    row_vals = sorted({r for r, _ in anchors})
    assert row_vals == [0, 4]


def test_grid_sliding_window_overlap() -> None:
    # stride < patch_size produces overlapping windows.
    anchors = grid_sample_anchors(8, 8, 4, 2, "drop")
    rows = sorted({r for r, _ in anchors})
    assert rows == [0, 2, 4]  # 0+4=4<=8, 2+4=6<=8, 4+4=8<=8; 6+4=10>8


def test_grid_zero_patch_size_raises() -> None:
    with pytest.raises(ValueError):
        grid_sample_anchors(10, 10, 0, 4, "pad")


def test_grid_zero_stride_raises() -> None:
    with pytest.raises(ValueError):
        grid_sample_anchors(10, 10, 4, 0, "pad")


def test_random_drop_anchors_in_bounds() -> None:
    anchors = random_sample_anchors(100, 100, 32, 50, 42, "drop")
    assert len(anchors) == 50
    for r, c in anchors:
        assert r + 32 <= 100 and c + 32 <= 100


def test_random_pad_anchors_any_position() -> None:
    anchors = random_sample_anchors(20, 20, 16, 200, 0, "pad")
    assert len(anchors) == 200
    for r, c in anchors:
        assert 0 <= r < 20 and 0 <= c < 20


def test_random_reproducible_with_same_seed() -> None:
    a = random_sample_anchors(100, 100, 32, 20, 7, "pad")
    b = random_sample_anchors(100, 100, 32, 20, 7, "pad")
    assert a == b


def test_random_different_seeds_differ() -> None:
    a = random_sample_anchors(100, 100, 32, 20, 1, "pad")
    b = random_sample_anchors(100, 100, 32, 20, 2, "pad")
    assert a != b


def test_random_zero_count_empty() -> None:
    assert random_sample_anchors(100, 100, 32, 0, 42, "drop") == []


def test_random_zero_patch_size_raises() -> None:
    with pytest.raises(ValueError):
        random_sample_anchors(10, 10, 0, 5, 42, "pad")


def test_config_default_stride_equals_patch_size() -> None:
    cfg = SamplerConfig(patch_size=64)
    assert cfg.stride == 64


def test_config_explicit_stride_preserved() -> None:
    cfg = SamplerConfig(patch_size=64, stride=32)
    assert cfg.stride == 32


def test_config_defaults() -> None:
    cfg = SamplerConfig(patch_size=32)
    assert cfg.mode == "grid"
    assert cfg.edge_strategy == "pad"
    assert cfg.pad_mode == "zero"
    assert cfg.max_empty_ratio == 1.0
    assert cfg.min_label_ratio == 0.0


def _solid_strip(h: int, w: int, c: int = 3, value: int = 128) -> np.ndarray:
    arr = np.full((h, w, c), value, dtype=np.uint8)
    return arr


def test_basic_grid_shape_3channel() -> None:
    img = _solid_strip(64, 64, 3)
    cfg = SamplerConfig(patch_size=16, edge_strategy="drop")
    patches, masks, meta = sample_patches(img, None, cfg)
    assert patches.shape == (16, 16, 16, 3)
    assert masks is None
    assert len(meta) == 16


def test_basic_grid_shape_single_channel() -> None:
    img = np.full((32, 32), 200, dtype=np.uint8)
    cfg = SamplerConfig(patch_size=8, edge_strategy="drop")
    patches, masks, meta = sample_patches(img, None, cfg)
    assert patches.shape == (16, 8, 8)


def test_mask_returned_when_provided() -> None:
    img = _solid_strip(32, 32, 1)
    msk = np.ones((32, 32), dtype=np.uint8)
    cfg = SamplerConfig(patch_size=8, edge_strategy="drop")
    patches, masks, meta = sample_patches(img, msk, cfg)
    assert masks is not None
    assert masks.shape == (patches.shape[0], 8, 8)


def test_pad_strategy_patches_correct_size() -> None:
    # 10x10 image, patch=8, stride=8 with pad gives 4 patches.
    img = _solid_strip(10, 10, 1)
    cfg = SamplerConfig(patch_size=8, edge_strategy="pad", pad_mode="zero")
    patches, _, meta = sample_patches(img, None, cfg)
    assert patches.shape[1:] == (8, 8, 1)
    padded_count = sum(1 for m in meta if m["padded"])
    assert padded_count > 0  # edge patches must have been padded


def test_pad_mode_zero_fills_with_zeros() -> None:
    img = np.full((6, 6, 1), 255, dtype=np.uint8)
    cfg = SamplerConfig(patch_size=8, edge_strategy="pad", pad_mode="zero")
    patches, _, _ = sample_patches(img, None, cfg)
    # Single anchor at (0, 0); last 2 rows and 2 cols must be zero.
    assert patches.shape == (1, 8, 8, 1)
    assert int(patches[0, 6:, :, 0].sum()) == 0
    assert int(patches[0, :, 6:, 0].sum()) == 0


def test_drop_strategy_no_partial_patches() -> None:
    img = _solid_strip(10, 10, 1)
    cfg = SamplerConfig(patch_size=8, edge_strategy="drop")
    patches, _, meta = sample_patches(img, None, cfg)
    for m in meta:
        assert not m["padded"]


def test_shift_strategy_all_in_bounds() -> None:
    img = _solid_strip(10, 10, 1)
    cfg = SamplerConfig(patch_size=8, edge_strategy="shift")
    patches, _, meta = sample_patches(img, None, cfg)
    for m in meta:
        assert m["row"] + 8 <= 10 and m["col"] + 8 <= 10
        assert not m["padded"]


def test_max_empty_ratio_filters_black_patches() -> None:
    # Strip with left half all-black and right half non-zero.
    img = np.zeros((8, 16, 3), dtype=np.uint8)
    img[:, 8:, :] = 128
    cfg = SamplerConfig(patch_size=8, edge_strategy="drop", max_empty_ratio=0.5)
    patches, _, meta = sample_patches(img, None, cfg)
    # Patch at col=0 is all black -> filtered out.
    assert len(meta) == 1
    assert meta[0]["col"] == 8


def test_min_label_ratio_filters_unlabeled_patches() -> None:
    img = _solid_strip(8, 16, 1)
    msk = np.zeros((8, 16), dtype=np.uint8)
    msk[:, 8:] = 1  # only right half labeled
    cfg = SamplerConfig(patch_size=8, edge_strategy="drop", min_label_ratio=0.5)
    patches, masks, meta = sample_patches(img, msk, cfg)
    assert len(meta) == 1
    assert meta[0]["col"] == 8


def test_no_filters_keeps_all_patches() -> None:
    img = np.zeros((8, 16, 1), dtype=np.uint8)
    cfg = SamplerConfig(patch_size=8, edge_strategy="drop")
    patches, _, meta = sample_patches(img, None, cfg)
    # max_empty_ratio=1.0 by default: black patches are not filtered.
    assert len(meta) == 2


def test_random_mode_returns_count_patches() -> None:
    img = _solid_strip(64, 64, 3)
    cfg = SamplerConfig(patch_size=16, mode="random", random_count=10, random_seed=0)
    patches, _, meta = sample_patches(img, None, cfg)
    assert len(meta) == 10
    assert patches.shape == (10, 16, 16, 3)


def test_random_mode_reproducible() -> None:
    img = _solid_strip(64, 64, 3)
    cfg = SamplerConfig(patch_size=16, mode="random", random_count=5, random_seed=99)
    _, _, meta_a = sample_patches(img, None, cfg)
    _, _, meta_b = sample_patches(img, None, cfg)
    assert meta_a == meta_b


def test_empty_result_correct_shape_3channel() -> None:
    # All-black image with max_empty_ratio=0 -> all patches filtered.
    img = np.zeros((8, 8, 3), dtype=np.uint8)
    cfg = SamplerConfig(patch_size=4, edge_strategy="drop", max_empty_ratio=0.0)
    patches, masks, meta = sample_patches(img, None, cfg)
    assert patches.shape == (0, 4, 4, 3)
    assert masks is None
    assert meta == []


def test_empty_result_with_mask() -> None:
    img = np.zeros((8, 8), dtype=np.uint8)
    msk = np.zeros((8, 8), dtype=np.uint8)
    cfg = SamplerConfig(patch_size=4, edge_strategy="drop", max_empty_ratio=0.0)
    patches, masks, meta = sample_patches(img, msk, cfg)
    assert patches.shape == (0, 4, 4)
    assert masks is not None and masks.shape == (0, 4, 4)


def test_metadata_row_col_correct() -> None:
    img = _solid_strip(16, 16, 1)
    cfg = SamplerConfig(patch_size=8, edge_strategy="drop")
    _, _, meta = sample_patches(img, None, cfg)
    positions = {(m["row"], m["col"]) for m in meta}
    assert positions == {(0, 0), (0, 8), (8, 0), (8, 8)}
