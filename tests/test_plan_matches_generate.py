"""``mapcv plan`` lists what ``generate`` refuses (tile cap, chunk guard, a foreign
``staging_dir``) the way it lists the patch limit, with generate's own wording, and exits 1."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

pytest.importorskip("rasterio", reason="these tests write their rasters with rasterio")

from test_geotiff_imagery import config_for, make_raster, write_labels
from typer.testing import CliRunner

from mapcv import pipeline
from mapcv.cli import app
from mapcv.config import MapcvConfig
from mapcv.locking import StagingDirError
from mapcv.pipeline import run_generate
from mapcv.planning import plan
from mapcv.sampler import TooManyPatchesError

_COLUMNS = {"COLUMNS": "200"}

_WORLD_RANDOM = """
region: {west: -179.9, south: -85, east: 179.9, north: 85}
imagery: {type: xyz, zoom: 22, url_template: "http://127.0.0.1:9/{z}/{x}/{y}.png"}
sampler: {patch_size: 64, stride: 1, mode: random}
writer: {staging_dir: STAGING}
"""


def _plan_command(config_path: Path) -> tuple[int, str]:
    result = CliRunner().invoke(app, ["plan", str(config_path)], env=_COLUMNS)
    return result.exit_code, " ".join(result.output.split())


def _wide_config(tmp_path: Path, staging: str = "dataset", **sampler: Any) -> MapcvConfig:
    raster = make_raster(tmp_path, width=1280, height=256, count=3)
    region = raster.region(margin=0.01)
    labels = write_labels(tmp_path, region)
    return config_for(
        tmp_path,
        {"path": str(raster.path), "chunk_rows": 128},
        region,
        labels=labels,
        staging=staging,
        **sampler,
    )


def test_plan_reports_the_tile_cap_and_exits_1(tmp_path: Path) -> None:
    path = tmp_path / "mapcv.yaml"
    path.write_text(_WORLD_RANDOM.replace("STAGING", str(tmp_path / "out")))
    config = MapcvConfig.from_yaml(path)
    estimate = plan(config)
    assert len(estimate.blocking) == 1
    assert "tiles at zoom [22], more than the limit of 16777216" in estimate.blocking[0]
    # The same text generate fails with.
    with pytest.raises(ValueError, match="more than the limit of 16777216") as raised:
        run_generate(config)
    assert str(raised.value) == estimate.blocking[0]

    code, output = _plan_command(path)
    assert code == 1
    assert "This config cannot be built" in output
    assert "more than the limit of 16777216" in output
    assert "Looks right?" not in output


def test_plan_reports_the_chunk_guard_and_exits_1(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config = _wide_config(tmp_path, stride=48)  # 3 rows x 25 columns = 75 anchors per chunk
    assert plan(config).blocking == []
    monkeypatch.setattr(pipeline, "MAX_CHUNK_ANCHORS", 74)
    estimate = plan(config)
    assert len(estimate.blocking) == 1
    assert "chunk 1 of 2 would read 75 patches at once" in estimate.blocking[0]
    assert "limit of 74 for one chunk" in estimate.blocking[0]
    with pytest.raises(TooManyPatchesError) as raised:
        run_generate(config)
    assert str(raised.value) == estimate.blocking[0]

    config_path = tmp_path / "mapcv.yaml"
    config_path.write_text(config.model_dump_json())  # JSON is YAML
    code, output = _plan_command(config_path)
    assert code == 1
    assert "This config cannot be built" in output
    assert "limit of 74 for one chunk" in output
    assert "Looks right?" not in output


def test_a_refusal_before_writing_leaves_no_folder_behind(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config = _wide_config(tmp_path, staging="made/by/the/run", stride=48)
    monkeypatch.setattr(pipeline, "MAX_CHUNK_ANCHORS", 74)
    with pytest.raises(TooManyPatchesError):
        run_generate(config)
    assert not (tmp_path / "made").exists()

    empty = tmp_path / "empty"  # a folder the user made is theirs to keep
    empty.mkdir()
    config = _wide_config(tmp_path, staging=str(empty), stride=48)
    with pytest.raises(TooManyPatchesError):
        run_generate(config)
    assert empty.is_dir() and list(empty.iterdir()) == []


def test_plan_reports_a_staging_dir_with_other_files_and_exits_1(tmp_path: Path) -> None:
    staging = tmp_path / "dataset"
    staging.mkdir()
    (staging / "other.txt").write_text("mine")
    config = _wide_config(tmp_path, staging=str(staging))
    estimate = plan(config)
    assert len(estimate.blocking) == 1
    assert "already holds files that mapcv did not write" in estimate.blocking[0]
    with pytest.raises(StagingDirError) as raised:
        run_generate(config)
    assert str(raised.value) == estimate.blocking[0]

    config_path = tmp_path / "mapcv.yaml"
    config_path.write_text(config.model_dump_json())
    code, output = _plan_command(config_path)
    assert code == 1
    assert "This config cannot be built" in output
    assert "other.txt" in output

    (staging / "manifest.json").write_text("{}")  # a dataset folder: resumed, not refused
    assert plan(config).blocking == []
