"""Dataset splitter: partitions a manifest into train/val/test and labeled/unlabeled lists."""

from __future__ import annotations

import math
import random
from pathlib import Path
from typing import Dict, List, Literal, Optional

from pydantic import BaseModel, Field

from mapcv.writer import Manifest, ManifestEntry


class SplitterConfig(BaseModel):
    """Configuration for splitting a manifest into train/val/test subsets."""

    test_ratio: float = Field(default=0.20, ge=0.0, le=1.0)
    val_ratio: float = Field(default=0.10, ge=0.0, le=1.0)
    labeled_ratios: List[float] = Field(default_factory=lambda: [0.10, 0.20, 0.30])
    seed: int = 42
    strategy: Literal["random", "stratified"] = "stratified"
    sample_limit: Optional[int] = Field(default=None, ge=1)


def _classify_entry(entry: ManifestEntry) -> int:
    """Classify a manifest entry as 0 (empty), 1 (fully labeled), or 2 (mixed).

    When a mask is present the classification uses per_class_pixel_counts; when
    absent it falls back to empty_ratio from the image.
    """
    counts = entry["per_class_pixel_counts"]
    if counts:
        if "0" in counts:
            return 0 if len(counts) == 1 else 2
        return 1
    if entry["empty_ratio"] >= 1.0:
        return 0
    if entry["empty_ratio"] <= 0.0:
        return 1
    return 2


def split_dataset(
    manifest: Manifest,
    config: SplitterConfig,
    output_dir: Path,
) -> None:
    """Write train/val/test split lists derived from *manifest* to *output_dir*.

    Produces:
      ``test.txt``, ``val.txt``, ``train.txt`` - one filename per line.
      ``<ratio>/labeled.txt`` and ``<ratio>/unlabeled.txt`` for each ratio in
      ``config.labeled_ratios``, where ``<ratio>`` is the integer percentage
      (e.g. ``10``, ``20``, ``30``).

    The split is computed from filenames only - no images are opened.
    """
    entries = manifest.patches
    if not entries:
        return

    rng = random.Random(config.seed)
    total = len(entries)
    limit = min(config.sample_limit, total) if config.sample_limit is not None else total

    pool: List[str]
    if config.strategy == "stratified":
        groups: Dict[int, List[str]] = {0: [], 1: [], 2: []}
        for e in entries:
            groups[_classify_entry(e)].append(e["filename"])
        pool = []
        for cls_files in groups.values():
            k = min(int(round((len(cls_files) / total) * limit)), len(cls_files))
            pool.extend(rng.sample(cls_files, k))
    else:
        all_files = [e["filename"] for e in entries]
        rng.shuffle(all_files)
        pool = all_files[:limit]

    rng.shuffle(pool)

    test_size = math.ceil(len(pool) * config.test_ratio)
    test_files = pool[:test_size]
    train_val_pool = pool[test_size:]

    val_size = math.ceil(len(train_val_pool) * config.val_ratio)
    val_files = train_val_pool[:val_size]
    train_files = train_val_pool[val_size:]

    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "test.txt").write_text("\n".join(test_files))
    (output_dir / "val.txt").write_text("\n".join(val_files))
    (output_dir / "train.txt").write_text("\n".join(train_files))

    rng.shuffle(train_files)
    for ratio in config.labeled_ratios:
        ratio_dir = output_dir / str(int(ratio * 100))
        ratio_dir.mkdir(parents=True, exist_ok=True)
        l_size = math.ceil(len(train_files) * ratio)
        labeled = train_files[:l_size]
        unlabeled = train_files[l_size:]
        (ratio_dir / "labeled.txt").write_text("\n".join(labeled))
        (ratio_dir / "unlabeled.txt").write_text("\n".join(unlabeled))
