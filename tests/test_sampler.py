"""Tests for the M5 patch sampler (Rust anchors + Python extraction)."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Literal

import numpy as np
import numpy.typing as npt
import pytest

from mapcv import SamplerConfig, sample_patches, sample_patches_at_anchors
from mapcv._mapcv_rs import grid_sample_anchors, random_anchor_capacity, random_sample_anchors
from mapcv._patching import MaskWindow, NullWindow
from mapcv.sampler import sample_annotated_patches


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


def test_random_pad_anchors_fit_inside_the_raster() -> None:
    # 20x20 with 16px patches has 5x5 positions; asking for 200 returns all 25.
    anchors = random_sample_anchors(20, 20, 16, 200, 0, "pad")
    assert len(anchors) == 25
    for r, c in anchors:
        assert 0 <= r <= 4 and 0 <= c <= 4


@pytest.mark.parametrize("strategy", ["drop", "pad", "shift"])
def test_random_anchors_never_repeat(strategy: str) -> None:
    for seed in range(20):
        anchors = random_sample_anchors(300, 300, 256, 100, seed, strategy)
        assert len(anchors) == 100
        assert len(set(anchors)) == 100


@pytest.mark.parametrize("strategy", ["drop", "pad", "shift"])
def test_random_count_is_capped_at_the_distinct_anchors(strategy: str) -> None:
    capacity = random_anchor_capacity(300, 300, 256, strategy)
    assert capacity == 45 * 45
    anchors = random_sample_anchors(300, 300, 256, 10_000, 3, strategy)
    assert len(anchors) == capacity
    assert set(anchors) == {(r, c) for r in range(45) for c in range(45)}


def test_random_raster_smaller_than_patch() -> None:
    assert random_sample_anchors(100, 100, 256, 5, 1, "drop") == []
    assert random_anchor_capacity(100, 100, 256, "drop") == 0
    assert random_sample_anchors(100, 100, 256, 5, 1, "pad") == [(0, 0)]
    assert random_sample_anchors(100, 100, 256, 5, 1, "shift") == [(0, 0)]
    assert random_anchor_capacity(100, 100, 256, "pad") == 1


@pytest.mark.parametrize("strategy", ["drop", "pad", "shift"])
def test_random_anchors_cover_the_edges_uniformly(strategy: str) -> None:
    # 100x100 raster, 20px patches -> 81 positions per axis. The last 20 are
    # anchors in rows/columns 61..80 form the band next to the far edge.
    positions, band, seeds, count = 81, 20, 300, 100
    total = seeds * count
    last_row_band = last_col_band = first_row = last_row = first_col = last_col = 0
    for seed in range(seeds):
        for r, c in random_sample_anchors(100, 100, 20, count, seed, strategy):
            assert r + 20 <= 100 and c + 20 <= 100
            last_row_band += r >= positions - band
            last_col_band += c >= positions - band
            first_row += r == 0
            last_row += r == positions - 1
            first_col += c == 0
            last_col += c == positions - 1
    expected = band / positions
    # Binomial standard deviation of each share is ~0.0025 here.
    assert last_row_band / total == pytest.approx(expected, abs=0.012)
    assert last_col_band / total == pytest.approx(expected, abs=0.012)
    # The top-left and bottom-right anchors are drawn as often as any other.
    for hits in (first_row, last_row, first_col, last_col):
        assert hits / total == pytest.approx(1 / positions, abs=0.006)


def test_random_anchors_are_deterministic_per_seed() -> None:
    for strategy in ("drop", "pad", "shift"):
        a = random_sample_anchors(500, 400, 64, 50, 11, strategy)
        assert a == random_sample_anchors(500, 400, 64, 50, 11, strategy)
        assert a != random_sample_anchors(500, 400, 64, 50, 12, strategy)
    # Seed 0 works and differs from its neighbour.
    assert random_sample_anchors(500, 400, 64, 50, 0, "drop") != random_sample_anchors(
        500, 400, 64, 50, 1, "drop"
    )


def test_random_zero_dimension_raises() -> None:
    with pytest.raises(ValueError):
        random_sample_anchors(0, 10, 4, 5, 42, "pad")
    with pytest.raises(ValueError):
        random_anchor_capacity(10, 10, 0, "pad")


def test_sample_patches_random_warns_when_count_exceeds_the_positions() -> None:
    img = _solid_strip(64, 64, 3)
    cfg = SamplerConfig(
        patch_size=56, mode="random", random_count=100, random_seed=0, edge_strategy="drop"
    )
    with pytest.warns(UserWarning, match=r"only 81 distinct patch position"):
        patches, _, meta = sample_patches(img, None, cfg)
    assert len(meta) == 81
    assert len({(m["row"], m["col"]) for m in meta}) == 81
    assert patches.shape == (81, 56, 56, 3)


def test_sample_patches_random_does_not_warn_when_count_is_met(
    recwarn: pytest.WarningsRecorder,
) -> None:
    img = _solid_strip(64, 64, 3)
    cfg = SamplerConfig(patch_size=16, mode="random", random_count=10, random_seed=0)
    sample_patches(img, None, cfg)
    assert not [w for w in recwarn if issubclass(w.category, UserWarning)]


def test_sample_patches_random_drop_on_a_tiny_raster_says_nothing_fits() -> None:
    img = _solid_strip(8, 8, 3)
    cfg = SamplerConfig(
        patch_size=16, mode="random", random_count=4, random_seed=0, edge_strategy="drop"
    )
    with pytest.warns(UserWarning, match=r"none.*edge_strategy: pad"):
        _, _, meta = sample_patches(img, None, cfg)
    assert meta == []


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


def _solid_strip(h: int, w: int, c: int = 3, value: int = 128) -> npt.NDArray[np.uint8]:
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
    patches, _masks, _meta = sample_patches(img, None, cfg)
    assert patches.shape == (16, 8, 8)


def test_mask_returned_when_provided() -> None:
    img = _solid_strip(32, 32, 1)
    msk = np.ones((32, 32), dtype=np.uint8)
    cfg = SamplerConfig(patch_size=8, edge_strategy="drop")
    patches, masks, _meta = sample_patches(img, msk, cfg)
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
    _patches, _, meta = sample_patches(img, None, cfg)
    for m in meta:
        assert not m["padded"]


def test_shift_strategy_all_in_bounds() -> None:
    img = _solid_strip(10, 10, 1)
    cfg = SamplerConfig(patch_size=8, edge_strategy="shift")
    _patches, _, meta = sample_patches(img, None, cfg)
    for m in meta:
        assert m["row"] + 8 <= 10 and m["col"] + 8 <= 10
        assert not m["padded"]


def test_max_empty_ratio_filters_black_patches() -> None:
    # Strip with left half all-black and right half non-zero.
    img = np.zeros((8, 16, 3), dtype=np.uint8)
    img[:, 8:, :] = 128
    cfg = SamplerConfig(patch_size=8, edge_strategy="drop", max_empty_ratio=0.5)
    _patches, _, meta = sample_patches(img, None, cfg)
    # Patch at col=0 is all black -> filtered out.
    assert len(meta) == 1
    assert meta[0]["col"] == 8


def test_min_label_ratio_filters_unlabeled_patches() -> None:
    img = _solid_strip(8, 16, 1)
    msk = np.zeros((8, 16), dtype=np.uint8)
    msk[:, 8:] = 1  # only right half labeled
    cfg = SamplerConfig(patch_size=8, edge_strategy="drop", min_label_ratio=0.5)
    _patches, _masks, meta = sample_patches(img, msk, cfg)
    assert len(meta) == 1
    assert meta[0]["col"] == 8


def test_no_filters_keeps_all_patches() -> None:
    img = np.zeros((8, 16, 1), dtype=np.uint8)
    cfg = SamplerConfig(patch_size=8, edge_strategy="drop")
    _patches, _, meta = sample_patches(img, None, cfg)
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
    patches, masks, _meta = sample_patches(img, msk, cfg)
    assert patches.shape == (0, 4, 4)
    assert masks is not None and masks.shape == (0, 4, 4)


def test_metadata_row_col_correct() -> None:
    img = _solid_strip(16, 16, 1)
    cfg = SamplerConfig(patch_size=8, edge_strategy="drop")
    _, _, meta = sample_patches(img, None, cfg)
    positions = {(m["row"], m["col"]) for m in meta}
    assert positions == {(0, 0), (0, 8), (8, 0), (8, 8)}


def test_explicit_anchors_preserve_float_values_and_global_offsets() -> None:
    image = np.arange(6 * 6 * 4, dtype=np.float32).reshape(6, 6, 4)
    valid = np.ones((6, 6), dtype=bool)
    config = SamplerConfig(patch_size=3, edge_strategy="drop")

    patches, masks, metadata = sample_patches_at_anchors(
        image,
        None,
        [(1, 2)],
        config,
        row_offset=100,
        col_offset=200,
        valid_mask=valid,
    )

    assert patches.dtype == np.float32
    assert patches.shape == (1, 3, 3, 4)
    assert masks is None
    assert metadata[0]["row"] == 101
    assert metadata[0]["col"] == 202


def test_valid_mask_not_numeric_zero_controls_empty_filter() -> None:
    image = np.zeros((4, 4, 6), dtype=np.float32)
    config = SamplerConfig(patch_size=4, max_empty_ratio=0.0)

    patches, _, metadata = sample_patches_at_anchors(
        image,
        None,
        [(0, 0)],
        config,
        valid_mask=np.ones((4, 4), dtype=bool),
    )

    assert patches.shape[0] == 1
    assert metadata[0]["empty_ratio"] == 0.0


def test_invalid_pixels_and_padding_are_counted_as_empty() -> None:
    image = np.full((3, 3, 2), np.nan, dtype=np.float32)
    valid = np.zeros((3, 3), dtype=bool)
    config = SamplerConfig(patch_size=4, edge_strategy="pad", max_empty_ratio=0.5)

    patches, _, metadata = sample_patches_at_anchors(
        image, None, [(0, 0)], config, valid_mask=valid
    )

    assert patches.shape[0] == 0
    assert metadata == []


# ---------------------------------------------------------------------------
# ignore_index: no labels invented where there is no imagery
# ---------------------------------------------------------------------------


def _labeled_corner() -> tuple[npt.NDArray[np.uint8], npt.NDArray[np.uint8]]:
    image = np.full((6, 6, 3), 100, dtype=np.uint8)
    mask = np.zeros((6, 6), dtype=np.uint8)
    mask[4:, 4:] = 2  # a class touching the bottom-right edge
    return image, mask


@pytest.mark.parametrize("pad_mode", ["zero", "reflect"])
def test_padding_gets_the_ignore_value_not_mirrored_or_background(
    pad_mode: Literal["zero", "reflect"],
) -> None:
    image, mask = _labeled_corner()
    config = SamplerConfig(patch_size=4, edge_strategy="pad", pad_mode=pad_mode)
    _, masks, meta = sample_patches_at_anchors(image, mask, [(4, 4)], config, ignore_index=255)
    assert masks is not None and meta[0]["padded"]
    patch = masks[0]
    assert (patch[:2, :2] == 2).all()  # real labels kept
    assert (patch[2:, :] == 255).all() and (patch[:, 2:] == 255).all()  # padding ignored
    assert (mask[4:, 4:] == 2).all()  # the source mask is not modified


def test_invalid_imagery_pixels_get_the_ignore_value() -> None:
    image, mask = _labeled_corner()
    valid = np.ones((6, 6), dtype=bool)
    valid[0, :] = False  # e.g. a failed tile row or NaN in a band
    config = SamplerConfig(patch_size=6)
    _, masks, _ = sample_patches_at_anchors(
        image, mask, [(0, 0)], config, valid_mask=valid, ignore_index=255
    )
    assert masks is not None
    assert (masks[0][0] == 255).all() and (masks[0][1:4] == 0).all()


def test_without_ignore_index_padding_follows_pad_mode_as_before() -> None:
    image, mask = _labeled_corner()
    config = SamplerConfig(patch_size=4, edge_strategy="pad", pad_mode="zero")
    _, masks, _ = sample_patches_at_anchors(image, mask, [(4, 4)], config)
    assert masks is not None and (masks[0][2:, :] == 0).all()


def test_min_label_ratio_does_not_count_ignored_pixels() -> None:
    image = np.full((4, 4, 3), 100, dtype=np.uint8)
    mask = np.zeros((4, 4), dtype=np.uint8)
    valid = np.zeros((4, 4), dtype=bool)  # nothing valid: every mask pixel becomes 255
    config = SamplerConfig(patch_size=4, min_label_ratio=0.1)
    _, _, meta = sample_patches_at_anchors(
        image, mask, [(0, 0)], config, valid_mask=valid, ignore_index=255
    )
    assert meta == []


class _RecordingWindow:
    """Annotates a patch with its anchor and records what the sampler passes in."""

    def __init__(self, reject: tuple[int, int] = (-1, -1)) -> None:
        self.reject = reject
        self.calls: list[tuple[int, int, int, str, tuple[int, ...] | None, float]] = []

    def annotate(
        self,
        row: int,
        col: int,
        patch_size: int,
        pad_mode: str,
        valid_patch: npt.NDArray[np.bool_] | None,
    ) -> tuple[int, int]:
        shape = None if valid_patch is None else tuple(valid_patch.shape)
        self.calls.append((row, col, patch_size, pad_mode, shape, -1.0))
        return (row, col)

    def accepts(self, annotation: tuple[int, int], min_label_ratio: float) -> bool:
        assert min_label_ratio == 0.25
        return annotation != self.reject

    def collate(self, annotations: Sequence[tuple[int, int]], patch_size: int) -> None:
        raise AssertionError("the sampler does not collate")


def test_annotated_sampling_hands_each_kept_patch_to_the_window() -> None:
    image = np.ones((10, 10, 3), dtype=np.uint8)
    valid = np.ones((10, 10), dtype=bool)
    valid[:4, :4] = False  # the patch at (0, 0) has no imagery at all
    config = SamplerConfig(
        patch_size=4,
        stride=4,
        edge_strategy="pad",
        pad_mode="reflect",
        max_empty_ratio=0.8,
        min_label_ratio=0.25,
    )
    window = _RecordingWindow(reject=(4, 4))

    images, annotations, metadata = sample_annotated_patches(
        image,
        [(0, 0), (0, 4), (4, 4), (8, 8)],
        config,
        window,
        row_offset=100,
        col_offset=200,
        valid_mask=valid,
    )

    # (0, 0) is too empty, (4, 4) is rejected by the window; (8, 8) is padded.
    assert [(m["row"], m["col"], m["padded"]) for m in metadata] == [
        (100, 204, False),
        (108, 208, True),
    ]
    assert annotations == [(0, 4), (8, 8)]
    assert images.shape == (2, 4, 4, 3)
    # Too-empty patches are never annotated; annotate sees the window-local anchor,
    # the pad mode and the patch's validity.
    assert [call[:5] for call in window.calls] == [
        (0, 4, 4, "reflect", (4, 4)),
        (4, 4, 4, "reflect", (4, 4)),
        (8, 8, 4, "reflect", (4, 4)),
    ]


def test_annotated_sampling_without_patches_returns_empty_arrays() -> None:
    image = np.zeros((6, 6, 2), dtype=np.float32)
    config = SamplerConfig(patch_size=3, max_empty_ratio=0.0)

    images, annotations, metadata = sample_annotated_patches(
        image, [(0, 0)], config, NullWindow(), valid_mask=np.zeros((6, 6), dtype=bool)
    )

    assert images.shape == (0, 3, 3, 2) and images.dtype == np.float32
    assert annotations == [] and metadata == []


@pytest.mark.parametrize("ignore_index", [None, 255])
@pytest.mark.parametrize("min_label_ratio", [0.0, 0.3])
def test_mask_sampling_is_the_annotated_sampler_with_a_mask_window(
    ignore_index: int | None, min_label_ratio: float
) -> None:
    rng = np.random.default_rng(5)
    image = rng.integers(0, 256, size=(13, 11, 3), dtype=np.uint8)
    mask = rng.choice(np.array([0, 0, 0, 1, 2], dtype=np.uint8), size=(13, 11))
    valid = rng.random((13, 11)) > 0.1
    config = SamplerConfig(
        patch_size=4,
        stride=3,
        edge_strategy="pad",
        pad_mode="reflect",
        min_label_ratio=min_label_ratio,
        max_empty_ratio=0.4,
    )
    anchors = [(int(r), int(c)) for r, c in grid_sample_anchors(13, 11, 4, 3, "pad")]

    images, masks, metadata = sample_patches_at_anchors(
        image, mask, anchors, config, valid_mask=valid, ignore_index=ignore_index
    )
    window = MaskWindow(mask, ignore_index)
    images2, annotations, metadata2 = sample_annotated_patches(
        image, anchors, config, window, valid_mask=valid
    )

    assert masks is not None and len(masks) > 0
    np.testing.assert_array_equal(images, images2)
    np.testing.assert_array_equal(masks, window.collate(annotations, 4))
    assert metadata == metadata2
