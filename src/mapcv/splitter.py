"""Dataset splitter: partitions a manifest into train/val/test and labeled/unlabeled lists."""

from __future__ import annotations

import math
import random
import warnings
from collections import defaultdict
from pathlib import Path
from typing import DefaultDict, Dict, Hashable, List, Literal, Optional, Sequence, Set, Tuple

from pydantic import BaseModel, Field, field_validator

from mapcv.writer import Manifest, ManifestEntry


class SplitterConfig(BaseModel):
    """Configuration for splitting a manifest into train/val/test subsets.

    ``spatial`` (the default) assigns whole square blocks of the raster to one
    split and drops train/val patches that overlap a held-out patch, so
    overlapping or neighbouring patches cannot leak between splits.
    ``stratified`` and ``random`` split individual patches and are only
    leakage-free when patches neither overlap nor repeat.
    """

    test_ratio: float = Field(default=0.20, ge=0.0, le=1.0)
    val_ratio: float = Field(default=0.10, ge=0.0, le=1.0)
    labeled_ratios: List[float] = Field(default_factory=lambda: [0.10, 0.20, 0.30])
    seed: int = 42
    strategy: Literal["spatial", "stratified", "random"] = "spatial"
    block_size: Optional[int] = Field(default=None, ge=1)
    sample_limit: Optional[int] = Field(default=None, ge=1)

    @field_validator("labeled_ratios")
    @classmethod
    def _validate_labeled_ratios(cls, ratios: List[float]) -> List[float]:
        for ratio in ratios:
            if not 0.0 < ratio <= 1.0:
                raise ValueError(f"labeled_ratios must be in (0, 1], got {ratio}")
        names = [ratio_dirname(ratio) for ratio in ratios]
        if len(names) != len(set(names)):
            raise ValueError("labeled_ratios must not contain duplicates")
        return ratios


def ratio_dirname(ratio: float) -> str:
    """Directory name for a labeled ratio as a percentage, e.g. ``0.29 -> "29"``."""
    return format(round(ratio * 100, 4), "g")


def _classify_entry(entry: ManifestEntry) -> int:
    """Classify a manifest entry as 0 (empty), 1 (fully labeled), or 2 (mixed).

    When a mask is present the classification uses per_class_pixel_counts; when
    absent it falls back to empty_ratio from the image.
    """
    counts = entry["per_class_pixel_counts"]
    if counts:
        has_bg = "0" in counts
        has_labeled = any(k != "0" for k in counts)
        if has_bg and not has_labeled:
            return 0
        if has_labeled and not has_bg:
            return 1
        return 2
    if entry["empty_ratio"] >= 1.0:
        return 0
    if entry["empty_ratio"] <= 0.0:
        return 1
    return 2


def _stratum(entry: ManifestEntry) -> Tuple[int, str]:
    """Stratify on labeled fraction and, when masked, the dominant foreground class."""
    counts = {k: v for k, v in entry["per_class_pixel_counts"].items() if k != "0"}
    dominant = max(sorted(counts), key=lambda k: counts[k]) if counts else ""
    return _classify_entry(entry), dominant


def _manifest_patch_geometry(manifest: Manifest) -> Tuple[Optional[int], Optional[int]]:
    """Return ``(patch_size, stride)`` recorded by the generating sampler, if known."""
    sampler = manifest.sampler or {}
    patch_size = sampler.get("patch_size")
    stride = sampler.get("stride")
    return (
        int(patch_size) if patch_size is not None else None,
        int(stride) if stride is not None else None,
    )


def _split_counts(n: int, config: SplitterConfig) -> Tuple[int, int]:
    test = math.ceil(n * config.test_ratio)
    val = math.ceil((n - test) * config.val_ratio)
    return test, val


def _assign_groups(
    groups: Sequence[List[ManifestEntry]],
    config: SplitterConfig,
) -> Tuple[List[ManifestEntry], List[ManifestEntry], List[ManifestEntry]]:
    """Fill test, then val, then train with whole groups in the given order."""
    total = sum(len(group) for group in groups)
    test_target, val_target = _split_counts(total, config)
    test: List[ManifestEntry] = []
    val: List[ManifestEntry] = []
    train: List[ManifestEntry] = []
    for group in groups:
        # Take a group only if it moves the split closer to its target, so large
        # spatial blocks do not overshoot small held-out sets.
        if abs(len(test) + len(group) - test_target) < abs(len(test) - test_target):
            test.extend(group)
        elif abs(len(val) + len(group) - val_target) < abs(len(val) - val_target):
            val.extend(group)
        else:
            train.extend(group)
    return test, val, train


def _overlapping(
    candidates: List[ManifestEntry], held_out: List[ManifestEntry], patch_size: int
) -> Set[int]:
    """Indices of ``candidates`` whose footprint overlaps any ``held_out`` footprint."""
    buckets: DefaultDict[Tuple[int, int], List[ManifestEntry]] = defaultdict(list)
    for entry in held_out:
        buckets[(entry["row"] // patch_size, entry["col"] // patch_size)].append(entry)

    overlapping: Set[int] = set()
    for index, entry in enumerate(candidates):
        row, col = entry["row"], entry["col"]
        cell_row, cell_col = row // patch_size, col // patch_size
        for d_row in (-1, 0, 1):
            for d_col in (-1, 0, 1):
                for other in buckets.get((cell_row + d_row, cell_col + d_col), ()):
                    if (
                        abs(other["row"] - row) < patch_size
                        and abs(other["col"] - col) < patch_size
                    ):
                        overlapping.add(index)
                        break
                if index in overlapping:
                    break
            if index in overlapping:
                break
    return overlapping


def _spatial_split(
    entries: List[ManifestEntry],
    config: SplitterConfig,
    rng: random.Random,
    patch_size: Optional[int],
) -> Tuple[List[ManifestEntry], List[ManifestEntry], List[ManifestEntry], int]:
    block = config.block_size or 4 * (patch_size or 1)
    blocks: DefaultDict[Tuple[int, int], List[ManifestEntry]] = defaultdict(list)
    for entry in entries:
        blocks[(entry["row"] // block, entry["col"] // block)].append(entry)
    ordered = [blocks[key] for key in sorted(blocks)]
    rng.shuffle(ordered)
    test, val, train = _assign_groups(ordered, config)
    if patch_size is None:
        return test, val, train, 0

    # Patches near a block edge can overlap a patch in a neighbouring block.
    dropped_val = _overlapping(val, test, patch_size)
    val = [entry for i, entry in enumerate(val) if i not in dropped_val]
    dropped_train = _overlapping(train, test + val, patch_size)
    train = [entry for i, entry in enumerate(train) if i not in dropped_train]
    return test, val, train, len(dropped_val) + len(dropped_train)


def _stratified_split(
    entries: List[ManifestEntry],
    config: SplitterConfig,
    rng: random.Random,
) -> Tuple[List[ManifestEntry], List[ManifestEntry], List[ManifestEntry]]:
    strata: Dict[Hashable, List[ManifestEntry]] = defaultdict(list)
    for entry in entries:
        strata[_stratum(entry)].append(entry)
    test: List[ManifestEntry] = []
    val: List[ManifestEntry] = []
    train: List[ManifestEntry] = []
    for key in sorted(strata, key=str):
        members = strata[key]
        rng.shuffle(members)
        n_test, n_val = _split_counts(len(members), config)
        test.extend(members[:n_test])
        val.extend(members[n_test : n_test + n_val])
        train.extend(members[n_test + n_val :])
    return test, val, train


def _apply_sample_limit(
    entries: List[ManifestEntry], config: SplitterConfig, rng: random.Random
) -> List[ManifestEntry]:
    if config.sample_limit is None or config.sample_limit >= len(entries):
        return list(entries)
    if config.strategy == "random":
        return rng.sample(entries, config.sample_limit)
    # Keep the stratum mix when subsampling.
    strata: Dict[Hashable, List[ManifestEntry]] = defaultdict(list)
    for entry in entries:
        strata[_stratum(entry)].append(entry)
    pool: List[ManifestEntry] = []
    for key in sorted(strata, key=str):
        members = strata[key]
        k = min(round(len(members) / len(entries) * config.sample_limit), len(members))
        pool.extend(rng.sample(members, k))
    return pool


def split_dataset(
    manifest: Manifest,
    config: SplitterConfig,
    output_dir: Path,
) -> Dict[str, int]:
    """Write train/val/test split lists derived from *manifest* to *output_dir*.

    Produces:
      ``test.txt``, ``val.txt``, ``train.txt`` - one filename per line.
      ``<percent>/labeled.txt`` and ``<percent>/unlabeled.txt`` for each ratio
      in ``config.labeled_ratios`` (e.g. ``10``, ``20``, ``30``, ``12.5``).

    The split is computed from the manifest only - no images are opened.

    Returns:
        Patch counts per split, plus ``dropped`` for train/val patches removed
        because they overlapped a held-out patch.
    """
    rng = random.Random(config.seed)
    entries = _apply_sample_limit(manifest.patches, config, rng)
    patch_size, stride = _manifest_patch_geometry(manifest)

    strategy = config.strategy
    if strategy == "spatial" and patch_size is None and config.block_size is None:
        warnings.warn(
            "The manifest does not record the patch size (created before mapcv 0.2), so a "
            "spatial split is impossible without split.block_size; falling back to "
            "'stratified', which can leak overlapping patches between splits.",
            UserWarning,
            stacklevel=2,
        )
        strategy = "stratified"
    elif strategy != "spatial" and stride is not None and patch_size is not None:
        if stride < patch_size or (manifest.sampler or {}).get("mode") == "random":
            warnings.warn(
                f"Patches overlap or repeat, so a '{strategy}' split leaks pixels between "
                "train and test; use strategy 'spatial'.",
                UserWarning,
                stacklevel=2,
            )

    dropped = 0
    if strategy == "spatial":
        test, val, train, dropped = _spatial_split(entries, config, rng, patch_size)
    elif strategy == "stratified":
        test, val, train = _stratified_split(entries, config, rng)
    else:
        shuffled = list(entries)
        rng.shuffle(shuffled)
        test, val, train = _assign_groups([[entry] for entry in shuffled], config)

    test_files = [entry["filename"] for entry in test]
    val_files = [entry["filename"] for entry in val]
    train_files = [entry["filename"] for entry in train]

    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "test.txt").write_text("\n".join(test_files))
    (output_dir / "val.txt").write_text("\n".join(val_files))
    (output_dir / "train.txt").write_text("\n".join(train_files))

    rng.shuffle(train_files)
    for ratio in config.labeled_ratios:
        ratio_dir = output_dir / ratio_dirname(ratio)
        ratio_dir.mkdir(parents=True, exist_ok=True)
        l_size = math.ceil(len(train_files) * ratio)
        (ratio_dir / "labeled.txt").write_text("\n".join(train_files[:l_size]))
        (ratio_dir / "unlabeled.txt").write_text("\n".join(train_files[l_size:]))

    return {
        "train": len(train_files),
        "val": len(val_files),
        "test": len(test_files),
        "dropped": dropped,
    }
