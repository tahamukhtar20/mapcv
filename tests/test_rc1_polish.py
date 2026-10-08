"""Polish found by the re-test of the release candidate: the advice that loops, a raw
warning, an empty dataset, booleans as numbers, a folder with only version-control files,
markup in file names, the label '0', and a few leftovers."""

from __future__ import annotations

import json
import os
import re
import shlex
import subprocess
import sys
import threading
import warnings
from pathlib import Path
from typing import Any

import numpy as np
import pytest
import yaml
from pydantic import ValidationError
from typer.testing import CliRunner

rasterio = pytest.importorskip("rasterio", reason="the GeoTIFFs are written with rasterio")
from PIL import Image
from test_dataset_integrity import _interrupt_after, _labeled
from test_geotiff_imagery import config_for, make_raster, write_labels

from mapcv import agent_tools
from mapcv.agent_tools import Sandbox, ToolFailure, ToolState
from mapcv.cli import app
from mapcv.config import MapcvConfig
from mapcv.manifest import (
    Manifest,
    ManifestMismatchError,
    SourceRecord,
    _restore_times_hint,
    format_file_time,
)
from mapcv.pipeline import run_generate, run_split
from mapcv.planning import LARGE_JOB_PATCHES, Plan
from mapcv.splitter import SplitterConfig
from mapcv.verify import CHECKSUMS_FILENAME, verify_dataset, write_checksums

runner = CliRunner()
WIDE = {"COLUMNS": "200"}


def _flat(text: str) -> str:
    return " ".join(text.split())


def _yaml(tmp_path: Path, extra: str = "", *, name: str = "mapcv.yaml", raster: str = "") -> Path:
    scene = make_raster(tmp_path, width=320, height=256, count=3, name=raster or "scene.tif")
    config = tmp_path / name
    config.write_text(
        f"region: {json.dumps(scene.region())}\n"
        f"imagery: {{type: geotiff, path: '{scene.path.as_posix()}'}}\n"
        "sampler: {patch_size: 64, edge_strategy: drop}\n"
        "writer: {staging_dir: out}\n" + extra,
        encoding="utf-8",
    )
    return config


# ── 1: verify --write-checksums after mapcv split ────────────────────────────


def test_write_checksums_records_the_files_split_rewrote(tmp_path: Path) -> None:
    config = _labeled(tmp_path)
    config.split = SplitterConfig()
    staging = config.writer.staging_dir
    run_generate(config)
    write_checksums(staging)
    run_split(staging, SplitterConfig(strategy="random", seed=3))
    assert not verify_dataset(staging).ok

    result = runner.invoke(app, ["verify", str(staging), "--write-checksums"], env=WIDE)
    assert result.exit_code == 0, result.output
    assert "recording the new hashes" in result.output
    assert CHECKSUMS_FILENAME in result.output
    again = runner.invoke(app, ["verify", str(staging)], env=WIDE)
    assert again.exit_code == 0, again.output
    assert verify_dataset(staging).ok


def test_write_checksums_still_refuses_a_changed_patch(tmp_path: Path) -> None:
    config = _labeled(tmp_path)
    config.split = SplitterConfig()
    staging = config.writer.staging_dir
    run_generate(config)
    write_checksums(staging)
    run_split(staging, SplitterConfig(strategy="random", seed=3))
    image = staging / Manifest.load(staging / "manifest.json").patches[0]["files"]["image"]
    pixels = np.asarray(Image.open(image)).copy()
    pixels[0, 0] ^= 1
    Image.fromarray(pixels).save(image)
    sums = (staging / CHECKSUMS_FILENAME).read_bytes()

    result = runner.invoke(app, ["verify", str(staging), "--write-checksums"], env=WIDE)
    assert result.exit_code == 1
    assert f"does not match its {CHECKSUMS_FILENAME} hash" in result.output
    assert (staging / CHECKSUMS_FILENAME).read_bytes() == sums


def test_the_mcp_verify_records_the_rewritten_files_too(tmp_path: Path) -> None:
    config = _labeled(tmp_path)
    config.split = SplitterConfig()
    staging = config.writer.staging_dir
    run_generate(config)
    write_checksums(staging)
    run_split(staging, SplitterConfig(strategy="random", seed=3))
    state = ToolState(Sandbox(tmp_path, True))
    result = agent_tools.verify(state, "dataset", write_sums=True)
    assert result.data["ok"] is True and result.data["checksums_written"] == "dataset/SHA256SUMS"
    assert verify_dataset(staging).ok


# ── 2: a warning raised while planning is shown once, in mapcv's style ───────


def _nodata_class_config(tmp_path: Path) -> Path:
    config = _yaml(tmp_path)
    scene = rasterio.open(tmp_path / "scene.tif")
    labels = np.zeros((scene.height, scene.width), np.uint8)
    labels[:, scene.width // 2 :] = 2
    labels[: scene.height // 2, : scene.width // 4] = 1
    with rasterio.open(
        tmp_path / "labels.tif",
        "w",
        driver="GTiff",
        height=scene.height,
        width=scene.width,
        count=1,
        dtype="uint8",
        crs=scene.crs,
        transform=scene.transform,
        nodata=2,
    ) as dst:
        dst.write(labels, 1)
    scene.close()
    with config.open("a", encoding="utf-8") as handle:
        handle.write(
            "labels: {type: raster, path: labels.tif, classes: "
            "{1: {id: 1, name: tree}, 2: {id: 2, name: water}}}\n"
        )
    return config


@pytest.mark.parametrize("command", ["plan", "generate"])
def test_a_planning_warning_is_one_mapcv_line(tmp_path: Path, command: str) -> None:
    config = _nodata_class_config(tmp_path)
    args = [command, str(config)] + (["--yes"] if command == "generate" else [])
    with warnings.catch_warnings(record=True) as escaped:
        warnings.simplefilter("always")
        result = runner.invoke(app, args, env=WIDE)
    assert result.exit_code == 0, result.output
    # A warning the CLI did not catch would reach the terminal as a raw Python warning.
    assert [str(each.message) for each in escaped if issubclass(each.category, UserWarning)] == []
    assert _flat(result.output).count("tags 2 as NoData") == 1
    assert "⚠" in result.output


# ── 3: no patches ────────────────────────────────────────────────────────────


@pytest.mark.parametrize("command", ["plan", "generate"])
def test_a_plan_without_patches_fails_instead_of_asking_if_it_looks_right(
    tmp_path: Path, command: str
) -> None:
    config = _yaml(tmp_path)
    config.write_text(config.read_text().replace("patch_size: 64", "patch_size: 4096"))
    args = [command, str(config)] + (["--yes"] if command == "generate" else [])
    result = runner.invoke(app, args, env=WIDE)
    assert result.exit_code == 1
    assert "no patches" in _flat(result.output)
    assert "Looks right" not in result.output and "Dataset ready" not in result.output
    assert not (tmp_path / "out").exists()


def test_a_run_that_writes_nothing_leaves_no_dataset(tmp_path: Path) -> None:
    config = _yaml(tmp_path, "labels: {path: labels.geojson, label_field: kind}\n")
    write_labels(tmp_path, json.loads(json.dumps(_region(config))))
    config.write_text(
        config.read_text().replace(
            "sampler: {patch_size: 64, edge_strategy: drop}",
            "sampler: {patch_size: 64, edge_strategy: drop, min_label_ratio: 1.0}",
        )
    )
    out = tmp_path / "out"
    result = runner.invoke(app, ["generate", str(config), "--yes"], env=WIDE)
    assert result.exit_code == 1
    assert "No patches were written" in result.output
    assert not (out / "manifest.json").exists() and not (out / "patches.geojson").exists()
    for command in ("verify", "info", "stats", "split"):
        again = runner.invoke(app, [command, str(out)], env=WIDE)
        assert again.exit_code != 0 and "No manifest found" in again.output, command
    assert runner.invoke(app, ["export", str(out), "-f", "terratorch"]).exit_code != 0


def test_the_mcp_generate_that_writes_nothing_is_a_failure_without_a_dataset(
    tmp_path: Path,
) -> None:
    config = _yaml(tmp_path, "labels: {path: labels.geojson, label_field: kind}\n")
    write_labels(tmp_path, _region(config))
    config.write_text(
        config.read_text().replace(
            "edge_strategy: drop}", "edge_strategy: drop, min_label_ratio: 1.0}"
        )
    )
    state = ToolState(Sandbox(tmp_path, True))
    job = agent_tools.prepare_generate(state, config.name)
    with pytest.raises(ToolFailure, match="No patches were written"):
        agent_tools.execute_generate(state, job, lambda done, total: None, threading.Event())
    assert not (tmp_path / "out" / "manifest.json").exists()


def _region(config: Path) -> dict[str, float]:
    return dict(yaml.safe_load(config.read_text())["region"])


# ── 4: booleans are not numbers; a patch count is a large job ────────────────


@pytest.mark.parametrize(
    "patch",
    [
        {"sampler": {"patch_size": True}},
        {"sampler": {"patch_size": 64, "stride": True}},
        {"sampler": {"patch_size": 64, "max_empty_ratio": True}},
        {"split": {"test_ratio": True}},
        {"imagery": {"type": "xyz", "zoom": True, "url_template": "http://x/{z}/{x}/{y}.png"}},
        {"region": {"west": True, "south": 0, "east": 1, "north": 1}},
        {"writer": {"staging_dir": "out", "jpg_quality": True}},
    ],
)
def test_booleans_are_not_numbers(patch: dict[str, Any]) -> None:
    base: dict[str, Any] = {
        "region": {"west": 0.0, "south": 0.0, "east": 0.01, "north": 0.01},
        "imagery": {"type": "xyz", "zoom": 17, "url_template": "http://x/{z}/{x}/{y}.png"},
        "sampler": {"patch_size": 64},
        "writer": {"staging_dir": "out"},
    }
    with pytest.raises(ValidationError, match="YAML boolean"):
        MapcvConfig.model_validate({**base, **patch})


def test_yaml_yes_is_refused_with_the_field_name(tmp_path: Path) -> None:
    config = _yaml(tmp_path)
    config.write_text(config.read_text().replace("patch_size: 64", "patch_size: yes"))
    result = runner.invoke(app, ["validate", str(config)], env=WIDE)
    assert result.exit_code == 1
    assert "patch_size" in result.output and "YAML boolean" in result.output


def test_whole_numbers_stay_valid_for_float_fields() -> None:
    config = MapcvConfig.model_validate(
        {
            "region": {"west": 0, "south": 0, "east": 1, "north": 1},
            "imagery": {"type": "xyz", "zoom": 17, "url_template": "http://x/{z}/{x}/{y}.png"},
            "sampler": {"patch_size": 64, "max_empty_ratio": 1},
            "split": {"test_ratio": 0, "val_ratio": 0},
            "writer": {"staging_dir": "out"},
        }
    )
    assert config.sampler.max_empty_ratio == 1.0 and config.region.east == 1.0


def _plan(patches: int, output_bytes: int = 10) -> Plan:
    return Plan(
        task="segmentation",
        region_km=(1.0, 1.0),
        imagery="x",
        resolution_m=1.0,
        raster_px=(1, 1),
        patches=patches,
        patch_size=1,
        tiles=None,
        download_bytes=None,
        output_bytes=output_bytes,
        chunk_memory_bytes=1,
        labels=None,
    )


def test_a_huge_patch_count_is_a_large_job_even_when_the_bytes_are_small() -> None:
    assert not _plan(LARGE_JOB_PATCHES).is_large
    big = _plan(LARGE_JOB_PATCHES + 1)
    assert big.is_large
    assert f"{LARGE_JOB_PATCHES + 1:,} patches" in (agent_tools.large_reason(big) or "")


# ── 5: a folder with only version-control files ──────────────────────────────


def test_generate_accepts_a_folder_with_only_hidden_entries_and_a_readme(tmp_path: Path) -> None:
    config = _labeled(tmp_path)
    staging = config.writer.staging_dir
    (staging / ".git").mkdir(parents=True)
    (staging / ".git" / "HEAD").write_text("ref: refs/heads/main\n")
    for name in (".gitkeep", ".gitattributes", "README.md"):
        (staging / name).write_text("x\n")
    run_generate(config)
    assert (staging / "manifest.json").is_file()
    assert (staging / "README.md").read_text() == "x\n"


def test_generate_still_refuses_a_folder_with_other_files(tmp_path: Path) -> None:
    config = _labeled(tmp_path)
    staging = config.writer.staging_dir
    staging.mkdir()
    (staging / ".gitkeep").write_text("")
    (staging / "notes.txt").write_text("mine")
    with pytest.raises(ValueError, match="notes.txt"):
        run_generate(config)
    assert (staging / "notes.txt").read_text() == "mine"


# ── 6: square brackets in names ──────────────────────────────────────────────


def test_brackets_in_the_config_and_imagery_names_are_shown(tmp_path: Path) -> None:
    config = _yaml(tmp_path, name="cfg [x].yaml", raster="img [a].tif")
    plan = runner.invoke(app, ["plan", str(config)], env=WIDE)
    assert plan.exit_code == 0, plan.output
    assert "Plan for cfg [x].yaml" in plan.output and "GeoTIFF" in plan.output
    assert "img [a].tif" in plan.output
    validate = runner.invoke(app, ["validate", str(config)], env=WIDE)
    assert "img [a].tif" in validate.output
    assert runner.invoke(app, ["generate", str(config), "--yes"], env=WIDE).exit_code == 0
    info = runner.invoke(app, ["info", str(tmp_path / "out")], env=WIDE)
    assert "geotiff · img [a].tif" in info.output


# ── 7: the label '0' ─────────────────────────────────────────────────────────


@pytest.mark.parametrize("command", ["plan", "generate"])
def test_a_label_named_zero_is_explained_once(tmp_path: Path, command: str) -> None:
    config = _yaml(tmp_path, "labels: {path: labels.geojson, label_field: kind}\n")
    region = _region(config)
    path = write_labels(tmp_path, region)
    data = json.loads(path.read_text())
    for feature, value in zip(data["features"], (0, 2)):
        feature["properties"]["kind"] = value
    data["features"].append(json.loads(json.dumps(data["features"][0])))
    data["features"][-1]["properties"]["kind"] = 10
    path.write_text(json.dumps(data))
    args = [command, str(config)] + (["--yes"] if command == "generate" else [])
    result = runner.invoke(app, args, env=WIDE)
    assert result.exit_code == 0, result.output
    assert _flat(result.output).count("label '0' is class 1, not background") == 1
    assert "set labels.classes" in _flat(result.output)


# ── 8: leftovers ─────────────────────────────────────────────────────────────


def test_a_debug_log_that_is_a_folder_is_a_usage_error(tmp_path: Path) -> None:
    result = runner.invoke(app, ["--debug-log", str(tmp_path), "doctor"], env=WIDE)
    assert result.exit_code == 2 and "Traceback" not in result.output
    assert "--debug-log" in result.output and "cannot write" in _flat(result.output)


def test_info_on_the_manifest_file_names_the_folder(tmp_path: Path) -> None:
    config = _labeled(tmp_path)
    run_generate(config)
    manifest = config.writer.staging_dir / "manifest.json"
    result = runner.invoke(app, ["info", str(manifest)], env=WIDE)
    assert result.exit_code == 1 and "manifest.json/manifest.json" not in result.output
    assert "is a file, not a dataset folder" in _flat(result.output)
    assert str(manifest.parent) in _flat(result.output)


@pytest.mark.parametrize("text", ['{"foo": 1}', '{"version": 3}', '{"version": 2, "x": []}'])
def test_a_json_file_that_is_no_manifest_is_rejected(tmp_path: Path, text: str) -> None:
    (tmp_path / "manifest.json").write_text(text)
    with pytest.raises(ManifestMismatchError, match="not a mapcv manifest"):
        Manifest.load(tmp_path / "manifest.json")
    assert not verify_dataset(tmp_path).ok


def test_a_manifest_from_a_newer_mapcv_still_says_so(tmp_path: Path) -> None:
    (tmp_path / "manifest.json").write_text('{"version": 99}')
    with pytest.raises(ManifestMismatchError, match="newer mapcv"):
        Manifest.load(tmp_path / "manifest.json")


def test_a_second_generate_leaves_the_footprints_untouched(tmp_path: Path) -> None:
    config = _labeled(tmp_path)
    run_generate(config)
    footprints = config.writer.staging_dir / "patches.geojson"
    os.utime(footprints, ns=(10**18, 10**18))
    before = footprints.read_bytes()
    run_generate(config)
    assert footprints.read_bytes() == before and footprints.stat().st_mtime_ns == 10**18
    assert not list(config.writer.staging_dir.glob("*.tmp"))


def test_the_mcp_generate_names_every_file_and_each_warning_once(tmp_path: Path) -> None:
    config = _nodata_class_config(tmp_path)
    state = ToolState(Sandbox(tmp_path, True))
    planned = agent_tools.plan(state, config.name)
    assert len(planned.data["warnings"]) == len(set(planned.data["warnings"]))
    job = agent_tools.prepare_generate(state, config.name)
    result = agent_tools.execute_generate(state, job, lambda done, total: None, threading.Event())
    assert "patches.geojson" in result.data["files"]
    assert len(result.data["warnings"]) == len(set(result.data["warnings"]))
    assert any("tags 2 as NoData" in text for text in result.data["warnings"])


def test_the_mcp_generate_refuses_a_run_without_patches(tmp_path: Path) -> None:
    config = _yaml(tmp_path)
    config.write_text(config.read_text().replace("patch_size: 64", "patch_size: 4096"))
    state = ToolState(Sandbox(tmp_path, True))
    with pytest.raises(ToolFailure, match="no patches"):
        agent_tools.prepare_generate(state, config.name)


# ── 10: the command that restores a file's modification time ─────────────────


@pytest.mark.skipif(sys.platform == "win32", reason="touch -d")
def test_the_printed_touch_command_restores_the_time_and_the_resume_works(tmp_path: Path) -> None:
    raster = make_raster(tmp_path, name="ortho [1].tif")
    config = config_for(tmp_path, {"path": str(raster.path), "chunk_rows": 64}, raster.region())
    recorded = raster.path.stat().st_mtime_ns + 123  # not a whole second
    os.utime(raster.path, ns=(recorded, recorded))
    _interrupt_after(config, 2)
    later = recorded + 7 * 10**9 + 456
    os.utime(raster.path, ns=(later, later))

    with pytest.raises(ManifestMismatchError) as caught:
        run_generate(config)
    message = str(caught.value)
    match = re.search(
        r"restore the recorded time with: (touch -d .+?) and run generate again", message
    )
    assert match, message
    command = match.group(1)
    assert shlex.quote(str(raster.path)) in command
    assert subprocess.run(command, shell=True, check=False).returncode == 0
    assert raster.path.stat().st_mtime_ns == recorded

    result = run_generate(config)
    assert result.manifest.complete is True and result.new_patches > 0


def test_the_time_is_written_with_every_digit() -> None:
    ns = 1_791_450_843_123_456_789
    assert (
        format_file_time(ns, "/data/a b.tif", windows=False)
        == "touch -d '2026-10-08T09:14:03.123456789Z' '/data/a b.tif'"
    )
    assert format_file_time(ns, "C:\\data\\it's.tif", windows=True) == (
        "(Get-Item -LiteralPath 'C:\\data\\it''s.tif').LastWriteTimeUtc = "
        "[DateTime]::Parse('2026-10-08T09:14:03.1234567Z').ToUniversalTime()"
    )


def test_a_mosaic_names_a_few_files_then_a_count() -> None:
    def source(times: dict[str, int]) -> SourceRecord:
        files = [{"name": name, "mtime_ns": ns} for name, ns in times.items()]
        return SourceRecord(name="image", fingerprint={"kind": "mosaic", "files": files})

    recorded = {f"f{i}.tif": 10**18 + i for i in range(5)}
    current = {**recorded, "f0.tif": 1, "f1.tif": 2, "f3.tif": 3, "f4.tif": 4}
    locations = {"image": {name: f"/d/{name}" for name in recorded}}
    old, new = Manifest(sources=[source(recorded)]), Manifest(sources=[source(current)])
    hint = _restore_times_hint(old, new, locations)
    assert hint.count("touch -d") == 3 and "/d/f0.tif" in hint and "/d/f3.tif" in hint
    assert "/d/f2.tif" not in hint and "(and 1 more file)" in hint
