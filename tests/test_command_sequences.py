"""Commands run one after another, as users do: a command that has nothing to do must
leave the dataset exactly as it was."""

from __future__ import annotations

import warnings
from pathlib import Path

import pytest

pytest.importorskip("rasterio", reason="the GeoTIFF is written with rasterio")
from test_geotiff_imagery import config_for, make_raster, write_labels

from mapcv.config import MapcvConfig
from mapcv.pipeline import run_generate, run_split
from mapcv.splitter import SplitterConfig
from mapcv.verify import verify_dataset, write_checksums


def _config(tmp_path: Path) -> MapcvConfig:
    raster = make_raster(tmp_path, width=640, height=512, count=3)
    region = raster.region()
    labels = write_labels(tmp_path, region)
    cfg = config_for(tmp_path, {"path": str(raster.path)}, region, labels=labels)
    data = cfg.model_dump(mode="json", exclude_unset=True)
    data["split"] = {"strategy": "random", "test_ratio": 0.2, "val_ratio": 0.1}
    return MapcvConfig.model_validate(data)


def _tree(root: Path) -> dict[str, bytes]:
    return {p.relative_to(root).as_posix(): p.read_bytes() for p in root.rglob("*") if p.is_file()}


def test_a_finished_run_keeps_a_re_split_and_its_checksums(tmp_path: Path) -> None:
    config = _config(tmp_path)
    staging = Path(config.writer.staging_dir)
    run_generate(config)
    run_split(staging, SplitterConfig(strategy="random", test_ratio=0.4, val_ratio=0.2, seed=7))
    write_checksums(staging)
    before = _tree(staging)

    with pytest.warns(UserWarning, match="other split settings"):
        result = run_generate(config)

    assert result.new_patches == 0
    assert _tree(staging) == before  # nothing changed, not even the split lists
    assert verify_dataset(staging).ok
    assert result.split_counts is not None
    assert sum(result.split_counts[name] for name in ("train", "val", "test")) == len(
        result.manifest.patches
    )


def test_a_finished_run_with_the_same_split_settings_says_nothing(tmp_path: Path) -> None:
    config = _config(tmp_path)
    staging = Path(config.writer.staging_dir)
    run_generate(config)
    write_checksums(staging)
    before = _tree(staging)
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        run_generate(config)
    assert _tree(staging) == before


def test_a_run_stopped_before_its_split_still_splits(tmp_path: Path) -> None:
    config = _config(tmp_path)
    staging = Path(config.writer.staging_dir)
    run_generate(config)
    for path in (staging / "splits").rglob("*"):
        if path.is_file():
            path.unlink()
    result = run_generate(config)
    assert result.split_counts is not None and (staging / "splits" / "train.txt").exists()
