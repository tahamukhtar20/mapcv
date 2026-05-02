"""Tests for M7 dataset splitter."""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional

import pytest

from mapcv.splitter import SplitterConfig, _classify_entry, split_dataset
from mapcv.writer import Manifest, ManifestEntry


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _entry(
    idx: int,
    *,
    empty_ratio: float = 0.0,
    counts: Optional[Dict[str, int]] = None,
    mask: bool = False,
) -> ManifestEntry:
    fname = f"patch_{idx:07d}.png"
    return ManifestEntry(
        filename=fname,
        mask_filename=fname if mask else None,
        row=idx,
        col=0,
        padded=False,
        strip_index=0,
        per_class_pixel_counts=counts if counts is not None else {},
        empty_ratio=empty_ratio,
    )


def _manifest(entries: List[ManifestEntry]) -> Manifest:
    m = Manifest(class_map={"bg": 0, "obj": 1})
    m.patches = list(entries)
    return m


def _make_manifest(n: int, *, empty_ratio: float = 0.0) -> Manifest:
    return _manifest([_entry(i, empty_ratio=empty_ratio) for i in range(n)])


def _read_lines(path: Path) -> List[str]:
    text = path.read_text().strip()
    return text.split("\n") if text else []


# ---------------------------------------------------------------------------
# SplitterConfig validation
# ---------------------------------------------------------------------------


def test_config_defaults() -> None:
    cfg = SplitterConfig()
    assert cfg.test_ratio == 0.20
    assert cfg.val_ratio == 0.10
    assert cfg.labeled_ratios == [0.10, 0.20, 0.30]
    assert cfg.seed == 42
    assert cfg.strategy == "stratified"
    assert cfg.sample_limit is None


def test_config_rejects_bad_ratio() -> None:
    with pytest.raises(Exception):
        SplitterConfig(test_ratio=1.5)
    with pytest.raises(Exception):
        SplitterConfig(val_ratio=-0.1)


def test_config_rejects_zero_sample_limit() -> None:
    with pytest.raises(Exception):
        SplitterConfig(sample_limit=0)


# ---------------------------------------------------------------------------
# _classify_entry
# ---------------------------------------------------------------------------


def test_classify_no_mask_empty() -> None:
    assert _classify_entry(_entry(0, empty_ratio=1.0)) == 0


def test_classify_no_mask_full() -> None:
    assert _classify_entry(_entry(0, empty_ratio=0.0)) == 1


def test_classify_no_mask_mixed() -> None:
    assert _classify_entry(_entry(0, empty_ratio=0.5)) == 2


def test_classify_mask_all_background() -> None:
    e = _entry(0, counts={"0": 64}, mask=True)
    assert _classify_entry(e) == 0


def test_classify_mask_all_labeled() -> None:
    e = _entry(0, counts={"1": 32, "2": 32}, mask=True)
    assert _classify_entry(e) == 1


def test_classify_mask_mixed() -> None:
    e = _entry(0, counts={"0": 32, "1": 32}, mask=True)
    assert _classify_entry(e) == 2


# ---------------------------------------------------------------------------
# split_dataset: empty manifest
# ---------------------------------------------------------------------------


def test_empty_manifest_writes_nothing(tmp_path: Path) -> None:
    split_dataset(_manifest([]), SplitterConfig(), tmp_path / "splits")
    assert not (tmp_path / "splits").exists()


# ---------------------------------------------------------------------------
# split_dataset: output files created
# ---------------------------------------------------------------------------


def test_output_files_created(tmp_path: Path) -> None:
    m = _make_manifest(100)
    split_dataset(m, SplitterConfig(), tmp_path)
    assert (tmp_path / "test.txt").exists()
    assert (tmp_path / "val.txt").exists()
    assert (tmp_path / "train.txt").exists()


def test_labeled_ratio_dirs_created(tmp_path: Path) -> None:
    m = _make_manifest(100)
    cfg = SplitterConfig(labeled_ratios=[0.10, 0.20])
    split_dataset(m, cfg, tmp_path)
    assert (tmp_path / "10" / "labeled.txt").exists()
    assert (tmp_path / "10" / "unlabeled.txt").exists()
    assert (tmp_path / "20" / "labeled.txt").exists()


# ---------------------------------------------------------------------------
# split_dataset: partition correctness
# ---------------------------------------------------------------------------


def test_all_filenames_partition_pool(tmp_path: Path) -> None:
    """test + val + train = the sampled pool (no losses, no duplicates)."""
    m = _make_manifest(50)
    split_dataset(m, SplitterConfig(labeled_ratios=[]), tmp_path)
    test = set(_read_lines(tmp_path / "test.txt"))
    val = set(_read_lines(tmp_path / "val.txt"))
    train = set(_read_lines(tmp_path / "train.txt"))
    assert test.isdisjoint(val)
    assert test.isdisjoint(train)
    assert val.isdisjoint(train)
    total = len(test) + len(val) + len(train)
    assert total == 50


def test_labeled_unlabeled_partition_train(tmp_path: Path) -> None:
    m = _make_manifest(100)
    split_dataset(m, SplitterConfig(labeled_ratios=[0.10]), tmp_path)
    train = set(_read_lines(tmp_path / "train.txt"))
    labeled = set(_read_lines(tmp_path / "10" / "labeled.txt"))
    unlabeled = set(_read_lines(tmp_path / "10" / "unlabeled.txt"))
    assert labeled.isdisjoint(unlabeled)
    assert labeled | unlabeled == train


def test_all_filenames_come_from_manifest(tmp_path: Path) -> None:
    m = _make_manifest(40)
    all_fnames = {e["filename"] for e in m.patches}
    split_dataset(m, SplitterConfig(labeled_ratios=[]), tmp_path)
    used: set[str] = set()
    for fname in ["test.txt", "val.txt", "train.txt"]:
        used |= set(_read_lines(tmp_path / fname))
    assert used <= all_fnames


# ---------------------------------------------------------------------------
# split_dataset: ratio sanity checks
# ---------------------------------------------------------------------------


def test_test_ratio_approx(tmp_path: Path) -> None:
    m = _make_manifest(100)
    split_dataset(m, SplitterConfig(test_ratio=0.20, labeled_ratios=[]), tmp_path)
    test = _read_lines(tmp_path / "test.txt")
    # ceil(100 * 0.20) = 20
    assert len(test) == 20


def test_val_ratio_approx(tmp_path: Path) -> None:
    m = _make_manifest(100)
    split_dataset(m, SplitterConfig(test_ratio=0.20, val_ratio=0.10, labeled_ratios=[]), tmp_path)
    val = _read_lines(tmp_path / "val.txt")
    # ceil(80 * 0.10) = 8
    assert len(val) == 8


def test_labeled_ratio_approx(tmp_path: Path) -> None:
    m = _make_manifest(100)
    split_dataset(m, SplitterConfig(test_ratio=0.20, val_ratio=0.10, labeled_ratios=[0.10]), tmp_path)
    train = _read_lines(tmp_path / "train.txt")
    labeled = _read_lines(tmp_path / "10" / "labeled.txt")
    # ceil(len(train) * 0.10)
    import math
    assert len(labeled) == math.ceil(len(train) * 0.10)


# ---------------------------------------------------------------------------
# split_dataset: sample_limit
# ---------------------------------------------------------------------------


def test_sample_limit_respected(tmp_path: Path) -> None:
    m = _make_manifest(200)
    split_dataset(m, SplitterConfig(sample_limit=50, labeled_ratios=[]), tmp_path)
    test = _read_lines(tmp_path / "test.txt")
    val = _read_lines(tmp_path / "val.txt")
    train = _read_lines(tmp_path / "train.txt")
    assert len(test) + len(val) + len(train) == 50


def test_sample_limit_capped_at_total(tmp_path: Path) -> None:
    m = _make_manifest(30)
    split_dataset(m, SplitterConfig(sample_limit=9999, labeled_ratios=[]), tmp_path)
    test = _read_lines(tmp_path / "test.txt")
    val = _read_lines(tmp_path / "val.txt")
    train = _read_lines(tmp_path / "train.txt")
    assert len(test) + len(val) + len(train) == 30


# ---------------------------------------------------------------------------
# split_dataset: determinism and seed sensitivity
# ---------------------------------------------------------------------------


def test_same_seed_same_split(tmp_path: Path) -> None:
    m = _make_manifest(100)
    out1 = tmp_path / "s1"
    out2 = tmp_path / "s2"
    cfg = SplitterConfig(seed=42, labeled_ratios=[])
    split_dataset(m, cfg, out1)
    split_dataset(m, cfg, out2)
    assert _read_lines(out1 / "test.txt") == _read_lines(out2 / "test.txt")
    assert _read_lines(out1 / "train.txt") == _read_lines(out2 / "train.txt")


def test_different_seed_different_split(tmp_path: Path) -> None:
    m = _make_manifest(100)
    out1 = tmp_path / "s1"
    out2 = tmp_path / "s2"
    split_dataset(m, SplitterConfig(seed=1, labeled_ratios=[]), out1)
    split_dataset(m, SplitterConfig(seed=2, labeled_ratios=[]), out2)
    assert _read_lines(out1 / "train.txt") != _read_lines(out2 / "train.txt")


# ---------------------------------------------------------------------------
# split_dataset: strategies
# ---------------------------------------------------------------------------


def test_random_strategy_still_partitions(tmp_path: Path) -> None:
    m = _make_manifest(60)
    cfg = SplitterConfig(strategy="random", labeled_ratios=[])
    split_dataset(m, cfg, tmp_path)
    test = set(_read_lines(tmp_path / "test.txt"))
    val = set(_read_lines(tmp_path / "val.txt"))
    train = set(_read_lines(tmp_path / "train.txt"))
    assert test.isdisjoint(val) and test.isdisjoint(train) and val.isdisjoint(train)
    assert len(test) + len(val) + len(train) == 60


def test_stratified_uses_all_classes(tmp_path: Path) -> None:
    entries = (
        [_entry(i, empty_ratio=1.0) for i in range(30)]   # class 0 - empty
        + [_entry(i + 30, empty_ratio=0.0) for i in range(30)]  # class 1 - labeled
        + [_entry(i + 60, empty_ratio=0.5) for i in range(30)]  # class 2 - mixed
    )
    m = _manifest(entries)
    cfg = SplitterConfig(strategy="stratified", labeled_ratios=[])
    split_dataset(m, cfg, tmp_path)
    used: set[str] = set()
    for fname in ["test.txt", "val.txt", "train.txt"]:
        used |= set(_read_lines(tmp_path / fname))
    class0_fnames = {e["filename"] for e in entries[:30]}
    class1_fnames = {e["filename"] for e in entries[30:60]}
    class2_fnames = {e["filename"] for e in entries[60:]}
    assert used & class0_fnames
    assert used & class1_fnames
    assert used & class2_fnames


# ---------------------------------------------------------------------------
# split_dataset: output_dir created if absent
# ---------------------------------------------------------------------------


def test_output_dir_created(tmp_path: Path) -> None:
    m = _make_manifest(20)
    nested = tmp_path / "a" / "b" / "splits"
    split_dataset(m, SplitterConfig(labeled_ratios=[]), nested)
    assert nested.is_dir()
