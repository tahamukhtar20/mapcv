"""Dataset splitter: partitions a manifest into train/val/test and labeled/unlabeled lists."""

from __future__ import annotations

import json
import math
import os
import random
import warnings
from collections import defaultdict
from collections.abc import Hashable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator

from mapcv.manifest import Manifest, ManifestEntry


class SplitterConfig(BaseModel):
    """Configuration for splitting a manifest into train/val/test subsets.

    ``spatial`` (the default) assigns whole square blocks of the raster to one
    split and drops train/val patches that overlap a held-out patch, so
    overlapping or neighbouring patches cannot leak between splits.
    ``stratified`` and ``random`` split individual patches and are only
    leakage-free when patches neither overlap nor repeat. ``region`` assigns whole
    regions of an area of interest (``region.path``) to one split each, the cleanest
    split when the regions are far apart.
    """

    # Unknown keys are errors, so typos and newer-version options are not silently ignored.
    model_config = ConfigDict(extra="forbid")

    test_ratio: float = Field(default=0.20, ge=0.0, le=1.0)
    val_ratio: float = Field(default=0.10, ge=0.0, le=1.0)
    labeled_ratios: list[float] = Field(default_factory=lambda: [0.10, 0.20, 0.30])
    seed: int = 42
    strategy: Literal["spatial", "stratified", "random", "region"] = "spatial"
    # Default: 4 x patch size, smaller for small rasters (at least 10 blocks).
    block_size: int | None = Field(default=None, ge=1)
    sample_limit: int | None = Field(default=None, ge=1)

    @field_validator("labeled_ratios")
    @classmethod
    def _validate_labeled_ratios(cls, ratios: list[float]) -> list[float]:
        for ratio in ratios:
            if not 0.0 < ratio <= 1.0:
                raise ValueError(f"labeled_ratios must be in (0, 1], got {ratio}")
        names = [ratio_dirname(ratio) for ratio in ratios]
        if len(names) != len(set(names)):
            raise ValueError("labeled_ratios must not contain duplicates")
        return ratios


@dataclass(frozen=True)
class SplitLists:
    """Patch names per split, in the order they were written to ``<split>.txt``.

    A patch's name is the file name of its image (:meth:`Manifest.patch_name`).
    """

    train: list[str]
    val: list[str]
    test: list[str]


def _list_text(names: Sequence[str]) -> str:
    """One filename per line, newline-terminated."""
    return "".join(f"{name}\n" for name in names)


def _stage(files: dict[Path, str]) -> dict[Path, Path]:
    """Write every file next to its place as ``.tmp``; return where each one went.

    Nothing is replaced yet, so a full disk or an interruption while writing leaves the
    previous files as they were (and no ``.tmp`` file).
    """
    temporary = {path: path.with_name(path.name + ".tmp") for path in files}
    try:
        for path, text in files.items():
            path.parent.mkdir(parents=True, exist_ok=True)
            temporary[path].write_text(text, encoding="utf-8", newline="\n")
    except BaseException:
        for tmp in temporary.values():
            tmp.unlink(missing_ok=True)
        raise
    return temporary


def _commit(temporary: dict[Path, Path]) -> None:
    """Move staged files into place."""
    for path, tmp in temporary.items():
        os.replace(tmp, path)


def ratio_dirname(ratio: float) -> str:
    """Directory name for a labeled ratio as a percentage, e.g. ``0.29 -> "29"``."""
    return format(round(ratio * 100, 4), "g")


def _class_counts(entry: ManifestEntry, ignore_key: str | None) -> dict[str, int]:
    """Per-class pixel counts without the ignore value (pixels with no imagery)."""
    counts = entry["summary"].get("class_pixels") or {}
    return {k: v for k, v in counts.items() if k != ignore_key}


def _classify_entry(entry: ManifestEntry, ignore_key: str | None = None) -> int:
    """Classify a manifest entry as 0 (empty), 1 (fully labeled), or 2 (mixed).

    When a mask is present the classification uses the summary's class pixel
    counts; when absent it falls back to the image's empty ratio.
    """
    counts = _class_counts(entry, ignore_key)
    if counts:
        has_bg = "0" in counts
        has_labeled = any(k != "0" for k in counts)
        if has_bg and not has_labeled:
            return 0
        if has_labeled and not has_bg:
            return 1
        return 2
    empty_ratio = entry["summary"].get("empty_ratio", 0.0)
    if empty_ratio >= 1.0:
        return 0
    if empty_ratio <= 0.0:
        return 1
    return 2


def _stratum(entry: ManifestEntry, ignore_key: str | None = None) -> tuple[int, str]:
    """Stratify on labeled fraction and, when masked, the dominant foreground class.

    Detection patches (``class_objects`` in the summary) are stratified on whether
    they hold objects and on their most frequent object class. Classification patches
    (``labels``) are stratified on whether they have a class label (not just
    ``background``) and on the assigned class with the largest coverage (ties: the
    lowest class ID).
    """
    labels = entry["summary"].get("labels")
    if labels is not None:
        coverage = entry["summary"].get("class_coverage") or {}
        classes = sorted((str(cid) for cid in labels if cid != 0), key=int)
        leading = max(classes, key=lambda k: coverage.get(k, 0.0)) if classes else ""
        return (1 if classes else 0), leading
    objects = entry["summary"].get("class_objects")
    if objects is not None:
        frequent = max(sorted(objects), key=lambda k: objects[k]) if objects else ""
        return (1 if objects else 0), frequent
    counts = {k: v for k, v in _class_counts(entry, ignore_key).items() if k != "0"}
    dominant = max(sorted(counts), key=lambda k: counts[k]) if counts else ""
    return _classify_entry(entry, ignore_key), dominant


def _manifest_patch_geometry(manifest: Manifest) -> tuple[int | None, int | None]:
    """Return ``(patch_size, stride)`` recorded by the generating sampler, if known."""
    sampler = manifest.sampler or {}
    patch_size = sampler.get("patch_size")
    stride = sampler.get("stride")
    return (
        int(patch_size) if patch_size is not None else None,
        int(stride) if stride is not None else None,
    )


def _split_counts(n: int, config: SplitterConfig) -> tuple[int, int]:
    test = math.ceil(n * config.test_ratio)
    val = math.ceil((n - test) * config.val_ratio)
    return test, val


def _assign_groups(
    groups: Sequence[list[ManifestEntry]],
    config: SplitterConfig,
) -> tuple[list[ManifestEntry], list[ManifestEntry], list[ManifestEntry]]:
    """Fill test, then val, then train with whole groups in the given order."""
    total = sum(len(group) for group in groups)
    test_target, val_target = _split_counts(total, config)
    test: list[ManifestEntry] = []
    val: list[ManifestEntry] = []
    train: list[ManifestEntry] = []
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
    candidates: list[ManifestEntry], held_out: list[ManifestEntry], patch_size: int
) -> set[int]:
    """Indices of ``candidates`` whose footprint overlaps any ``held_out`` footprint."""
    buckets: defaultdict[tuple[int, int], list[ManifestEntry]] = defaultdict(list)
    for entry in held_out:
        buckets[(entry["row"] // patch_size, entry["col"] // patch_size)].append(entry)

    overlapping: set[int] = set()
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


# Enough blocks that each split can get several of them.
_MIN_BLOCKS = 10


def _default_block_size(entries: list[ManifestEntry], patch_size: int) -> int:
    """4 x patch size, shrunk (to no less than one patch) so small rasters still
    divide into at least ``_MIN_BLOCKS`` blocks."""
    if not entries:
        return 4 * patch_size
    rows = max(entry["row"] for entry in entries) + patch_size
    cols = max(entry["col"] for entry in entries) + patch_size
    fitting = int(math.sqrt(rows * cols / _MIN_BLOCKS)) // patch_size * patch_size
    return max(patch_size, min(4 * patch_size, fitting))


def _spatial_split(
    entries: list[ManifestEntry],
    config: SplitterConfig,
    rng: random.Random,
    patch_size: int | None,
) -> tuple[list[ManifestEntry], list[ManifestEntry], list[ManifestEntry], int]:
    block = config.block_size or _default_block_size(entries, patch_size or 1)
    blocks: defaultdict[tuple[int, int], list[ManifestEntry]] = defaultdict(list)
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


def _region_split(
    entries: list[ManifestEntry],
    config: SplitterConfig,
    rng: random.Random,
    patch_size: int | None,
) -> tuple[list[ManifestEntry], list[ManifestEntry], list[ManifestEntry], int]:
    """Whole regions (``summary.region``) to one split each, in a seeded random order;
    patches that still overlap a held-out patch (adjacent regions) are dropped."""
    regions: defaultdict[str, list[ManifestEntry]] = defaultdict(list)
    for entry in entries:
        region = entry["summary"].get("region")
        if region is None:
            raise ValueError(
                "split.strategy 'region' needs patches made with region.path (an area of "
                "interest); this dataset records no regions"
            )
        regions[str(region)].append(entry)
    ordered = [regions[name] for name in sorted(regions)]
    rng.shuffle(ordered)
    test, val, train = _assign_groups(ordered, config)
    if patch_size is None:
        return test, val, train, 0
    dropped_val = _overlapping(val, test, patch_size)
    val = [entry for i, entry in enumerate(val) if i not in dropped_val]
    dropped_train = _overlapping(train, test + val, patch_size)
    train = [entry for i, entry in enumerate(train) if i not in dropped_train]
    return test, val, train, len(dropped_val) + len(dropped_train)


def _stratified_split(
    entries: list[ManifestEntry],
    config: SplitterConfig,
    rng: random.Random,
    ignore_key: str | None = None,
) -> tuple[list[ManifestEntry], list[ManifestEntry], list[ManifestEntry]]:
    strata: dict[Hashable, list[ManifestEntry]] = defaultdict(list)
    for entry in entries:
        strata[_stratum(entry, ignore_key)].append(entry)
    test: list[ManifestEntry] = []
    val: list[ManifestEntry] = []
    train: list[ManifestEntry] = []
    for key in sorted(strata, key=str):
        members = strata[key]
        rng.shuffle(members)
        n_test, n_val = _split_counts(len(members), config)
        test.extend(members[:n_test])
        val.extend(members[n_test : n_test + n_val])
        train.extend(members[n_test + n_val :])
    return test, val, train


def _apply_sample_limit(
    entries: list[ManifestEntry],
    config: SplitterConfig,
    rng: random.Random,
    ignore_key: str | None = None,
) -> list[ManifestEntry]:
    if config.sample_limit is None or config.sample_limit >= len(entries):
        return list(entries)
    if config.strategy == "random":
        return rng.sample(entries, config.sample_limit)
    # Keep the stratum mix when subsampling.
    strata: dict[Hashable, list[ManifestEntry]] = defaultdict(list)
    for entry in entries:
        strata[_stratum(entry, ignore_key)].append(entry)
    pool: list[ManifestEntry] = []
    for key in sorted(strata, key=str):
        members = strata[key]
        k = min(round(len(members) / len(entries) * config.sample_limit), len(members))
        pool.extend(rng.sample(members, k))
    return pool


def split_dataset(
    manifest: Manifest,
    config: SplitterConfig,
    output_dir: Path,
) -> dict[str, int]:
    """Write train/val/test split lists derived from *manifest* to *output_dir*.

    Returns:
        Patch counts per split, plus ``dropped`` for train/val patches removed
        because they overlapped a held-out patch. See :func:`split_manifest` for
        what is written.
    """
    counts, _ = _split(manifest, config, output_dir, stacklevel=3)
    return counts


def split_manifest(
    manifest: Manifest,
    config: SplitterConfig,
    output_dir: Path,
) -> tuple[dict[str, int], SplitLists]:
    """Like :func:`split_dataset`, but also return the train/val/test filename lists.

    Writes to *output_dir*:

      ``test.txt``, ``val.txt``, ``train.txt`` - one filename per line.
      ``<percent>/labeled.txt`` and ``<percent>/unlabeled.txt`` for each ratio
      in ``config.labeled_ratios`` (e.g. ``10``, ``20``, ``30``, ``12.5``).

    The split is computed from the manifest only - no images are opened.

    Returns:
        ``(counts, lists)``: patch counts per split plus ``dropped`` (train/val
        patches removed because they overlapped a held-out patch), and the lists.
    """
    return _split(manifest, config, output_dir, stacklevel=3)


def _warn_about_shares(
    counts: dict[str, int], config: SplitterConfig, strategy: str, stacklevel: int
) -> None:
    """Warn when train is empty, or when whole blocks or regions moved the test share far
    from the one asked for."""
    total = counts["train"] + counts["val"] + counts["test"]
    if total == 0:
        return
    if counts["train"] == 0:
        warnings.warn(
            f"No patch is left for train ({counts['test']} test, {counts['val']} val): lower "
            "split.test_ratio and split.val_ratio, or make more patches.",
            UserWarning,
            stacklevel=stacklevel,
        )
        return
    share = counts["test"] / total
    if strategy in ("spatial", "region") and total >= 10 and abs(share - config.test_ratio) > 0.1:
        unit = "regions" if strategy == "region" else "blocks"
        smaller = "more regions" if strategy == "region" else "a smaller split.block_size"
        warnings.warn(
            f"test holds {share:.0%} of the patches, not the {config.test_ratio:.0%} asked for, "
            f"because whole {unit} go to one split; use {smaller} for shares closer to the "
            "ratios.",
            UserWarning,
            stacklevel=stacklevel,
        )


def _split(
    manifest: Manifest,
    config: SplitterConfig,
    output_dir: Path,
    stacklevel: int,
) -> tuple[dict[str, int], SplitLists]:
    rng = random.Random(config.seed)
    ignore = manifest.ignore_index
    ignore_key = str(ignore) if ignore is not None else None
    entries = _apply_sample_limit(manifest.patches, config, rng, ignore_key)
    patch_size, stride = _manifest_patch_geometry(manifest)

    strategy = config.strategy
    if strategy == "spatial" and patch_size is None and config.block_size is None:
        warnings.warn(
            "The manifest does not record the patch size (created before mapcv 0.2), so a "
            "spatial split is impossible without split.block_size; falling back to "
            "'stratified', which can leak overlapping patches between splits.",
            UserWarning,
            stacklevel=stacklevel,
        )
        strategy = "stratified"
    elif strategy not in ("spatial", "region") and stride is not None and patch_size is not None:
        if stride < patch_size or (manifest.sampler or {}).get("mode") == "random":
            warnings.warn(
                f"Patches overlap or repeat, so a '{strategy}' split leaks pixels between "
                "train and test; use strategy 'spatial'.",
                UserWarning,
                stacklevel=stacklevel,
            )

    dropped = 0
    if strategy == "spatial":
        test, val, train, dropped = _spatial_split(entries, config, rng, patch_size)
    elif strategy == "region":
        test, val, train, dropped = _region_split(entries, config, rng, patch_size)
        regions = len({entry["summary"].get("region") for entry in entries})
        if regions < 3:
            warnings.warn(
                f"Only {regions} {'region' if regions == 1 else 'regions'}: a region split "
                "leaves some of train, val and "
                "test empty; split the area of interest into more regions.",
                UserWarning,
                stacklevel=stacklevel,
            )
    elif strategy == "stratified":
        test, val, train = _stratified_split(entries, config, rng, ignore_key)
    else:
        shuffled = list(entries)
        rng.shuffle(shuffled)
        test, val, train = _assign_groups([[entry] for entry in shuffled], config)

    test_files = [manifest.patch_name(entry) for entry in test]
    val_files = [manifest.patch_name(entry) for entry in val]
    train_files = [manifest.patch_name(entry) for entry in train]

    files = {
        output_dir / "test.txt": _list_text(test_files),
        output_dir / "val.txt": _list_text(val_files),
        output_dir / "train.txt": _list_text(train_files),
    }

    lists = SplitLists(train=list(train_files), val=val_files, test=test_files)

    rng.shuffle(train_files)
    for ratio in config.labeled_ratios:
        ratio_dir = output_dir / ratio_dirname(ratio)
        l_size = math.ceil(len(train_files) * ratio)
        files[ratio_dir / "labeled.txt"] = _list_text(train_files[:l_size])
        files[ratio_dir / "unlabeled.txt"] = _list_text(train_files[l_size:])

    counts = {
        "train": len(train_files),
        "val": len(val_files),
        "test": len(test_files),
        "dropped": dropped,
    }
    _warn_about_shares(counts, config, strategy, stacklevel + 1)
    # Record how the lists were made, so a split can be reproduced or audited later.
    record = {"settings": config.model_dump(mode="json"), "strategy_used": strategy, **counts}
    # Every list is written before any is replaced, and split.json is removed before the
    # lists are moved into place and written after: lists left by a run stopped while
    # they were moved have no split.json, so they are known to be unfinished.
    output_dir.mkdir(parents=True, exist_ok=True)
    record_path = output_dir / "split.json"
    staged = _stage(files)
    record_path.unlink(missing_ok=True)
    _commit(staged)
    _commit(_stage({record_path: json.dumps(record, indent=2) + "\n"}))
    return counts, lists
