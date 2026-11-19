"""``mapcv plan`` lists every limit ``generate`` refuses a config for, in the same words.

The tile limit of an XYZ source, the patch limit of one chunk and an output folder that
holds other files each stop ``generate`` before it writes anything; the plan must not
end with "Looks right?" for any of them.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

import pytest
import yaml
from test_crafted_inputs import _bounded
from typer.testing import CliRunner

from mapcv import _confine, pipeline, planning
from mapcv._mapcv_rs import tile_count, tiles
from mapcv.cli import app
from mapcv.config import MapcvConfig
from mapcv.locking import StagingDirError
from mapcv.pipeline import run_generate
from mapcv.planning import plan
from mapcv.sampler import MAX_CHUNK_ANCHORS, TooManyPatchesError

#: Nothing listens here: a run that reached the network would fail on it instead.
_NOWHERE = "http://127.0.0.1:9/{z}/{x}/{y}.png"


def _xyz_config(tmp_path: Path, **overrides: Any) -> MapcvConfig:
    data: dict[str, Any] = {
        "region": {"west": 10.0, "south": 50.0, "east": 10.01, "north": 50.01},
        "imagery": {"type": "xyz", "zoom": 15, "url_template": _NOWHERE},
        "sampler": {"patch_size": 64, "mode": "grid"},
        "writer": {"staging_dir": str(tmp_path / "dataset")},
    }
    data.update(overrides)
    return MapcvConfig.model_validate(data)


def _cli_plan(tmp_path: Path, config: MapcvConfig) -> tuple[int, str]:
    path = tmp_path / "mapcv.yaml"
    path.write_text(yaml.safe_dump(config.model_dump(mode="json", exclude_unset=True)))
    result = CliRunner().invoke(app, ["plan", str(path)], env={"COLUMNS": "400"})
    return result.exit_code, " ".join(result.output.split())


# ── 1. The tile limit of an XYZ source ───────────────────────────────────────


def _world(tmp_path: Path) -> MapcvConfig:
    return _xyz_config(
        tmp_path,
        region={"west": -179.9, "south": -85, "east": 179.9, "north": 85},
        imagery={"type": "xyz", "zoom": 22, "url_template": _NOWHERE},
        sampler={"patch_size": 64, "stride": 1, "mode": "random"},
    )


def test_the_plan_lists_the_tile_limit_in_the_words_of_generate(tmp_path: Path) -> None:
    config = _world(tmp_path)
    estimate = plan(config)
    assert estimate.tiles is not None and estimate.tiles > 16_777_216
    assert len(estimate.blocking) == 1
    assert "more than the limit of 16777216" in estimate.blocking[0]
    assert "use a lower zoom or a smaller region" in estimate.blocking[0]
    assert estimate.blocking[0] in estimate.warnings

    with pytest.raises(ValueError) as raised:
        run_generate(config)
    assert str(raised.value) == estimate.blocking[0]
    # The run made the folder and wrote nothing; it leaves none behind.
    assert not config.writer.staging_dir.exists()


def test_the_plan_command_stops_on_the_tile_limit(tmp_path: Path) -> None:
    code, output = _cli_plan(tmp_path, _world(tmp_path))
    assert code == 1
    assert "This config cannot be built" in output
    assert "more than the limit of 16777216" in output
    assert "Looks right?" not in output


def test_a_cover_at_the_tile_limit_is_not_refused(tmp_path: Path) -> None:
    # The whole world at zoom 12 is exactly 2^24 tiles.
    config = _xyz_config(
        tmp_path,
        region={"west": -180, "south": -85.05, "east": 180, "north": 85.05},
        imagery={"type": "xyz", "zoom": 12, "url_template": _NOWHERE},
        sampler={"patch_size": 256, "mode": "random", "random_count": 10},
    )
    estimate = plan(config)
    assert estimate.tiles == 16_777_216
    assert estimate.blocking == []


@pytest.mark.skipif(
    not sys.platform.startswith("linux"), reason="the subprocess is watched through /proc"
)
def test_the_plan_for_a_cover_just_over_the_limit_lists_no_tiles(tmp_path: Path) -> None:
    # 17,014,784 tiles at zoom 13: 1.4 % over the limit. The message comes from a count; a
    # list of that many tiles would take over a gigabyte, so the plan runs in bounded memory.
    (tmp_path / "mapcv.yaml").write_text(
        "region: {west: -90, south: -67, east: 90, north: 67}\n"
        f'imagery: {{type: xyz, zoom: 13, url_template: "{_NOWHERE}"}}\n'
        "sampler: {patch_size: 256, mode: random, random_count: 10}\n"
        f"writer: {{staging_dir: {tmp_path / 'dataset'}}}\n"
    )
    result = _bounded(
        f"""
        import sys

        import mapcv._mapcv_rs as rs
        from mapcv import planning
        from mapcv.config import MapcvConfig

        original = rs.tiles

        def no_list(*args):
            raise AssertionError("the plan listed the tile cover")

        # Every name mapcv bound to the call that lists tiles.
        for module in list(sys.modules.values()):
            for name, value in list(vars(module).items()) if hasattr(module, "__dict__") else []:
                if value is original and module.__name__.startswith("mapcv"):
                    setattr(module, name, no_list)
        estimate = planning.plan(MapcvConfig.from_yaml({str(tmp_path / "mapcv.yaml")!r}))
        print(estimate.tiles, "|", estimate.blocking[0])
        """,
        600 << 20,
    )
    assert result.returncode == 0, result.stderr[-1500:]
    count, message = result.stdout.strip().split(" | ")
    assert count == "17014784"
    assert "covers 17014784 tiles at zoom [13], more than the limit of 16777216" in message


def test_the_plan_and_the_tile_cover_refuse_with_one_message(tmp_path: Path) -> None:
    region = _world(tmp_path).region
    box = (region.west, region.south, region.east, region.north)
    with pytest.raises(ValueError) as counted:
        tile_count(*box, [22])
    with pytest.raises(ValueError) as listed:
        tiles(*box, [22])
    assert str(counted.value) == str(listed.value)


# ── 2. The patch limit of one chunk ──────────────────────────────────────────


def _dense(tmp_path: Path, **sampler: Any) -> MapcvConfig:
    # About 3,600 x 3,600 px at zoom 18, one 1,024-row chunk (strip_rows 4) at a time.
    return _xyz_config(
        tmp_path,
        region={"west": 10.0, "south": 50.0, "east": 10.018, "north": 50.012},
        imagery={"type": "xyz", "zoom": 18, "url_template": _NOWHERE, "strip_rows": 4},
        sampler={"patch_size": 8, "mode": "grid", "edge_strategy": "drop", **sampler},
    )


def test_the_plan_lists_a_chunk_over_the_limit_in_the_words_of_generate(tmp_path: Path) -> None:
    config = _dense(tmp_path, stride=1)
    estimate = plan(config)
    width, height = estimate.raster_px
    # Under the limit of a run, over the limit of a chunk: 1,024 rows of the grid.
    assert MAX_CHUNK_ANCHORS < estimate.patches < 50_000_000
    assert 1_024 * (width - 7) > MAX_CHUNK_ANCHORS
    assert len(estimate.blocking) == 1
    message = estimate.blocking[0]
    assert f"would read {1_024 * (width - 7):,} patches at once" in message
    assert "more than the limit of 3,000,000 for one chunk (a chunk is 1,024 rows" in message
    assert height > 1_024

    with pytest.raises(TooManyPatchesError) as raised:
        run_generate(config)
    assert str(raised.value) == message
    # Refused before anything was written, and no empty folder is left behind.
    assert not config.writer.staging_dir.exists()


def test_a_chunk_within_the_limit_is_not_refused(tmp_path: Path) -> None:
    # Stride 2 puts 512 x ~1,800 anchors into a chunk, under the limit.
    assert plan(_dense(tmp_path, stride=2)).blocking == []


def test_the_plan_leaves_the_chunks_of_a_random_sampler_to_generate(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # Drawing the positions would cost the plan as much as the run; generate checks them.
    def no_draw(*args: Any) -> None:
        raise AssertionError("the plan drew random positions")

    monkeypatch.setattr(pipeline, "random_positions_for", no_draw)
    monkeypatch.setattr(planning, "MAX_CHUNK_ANCHORS", 1_000)
    assert plan(_dense(tmp_path, mode="random", random_count=20_000)).blocking == []


def test_an_existing_output_folder_is_kept_when_a_chunk_is_refused(tmp_path: Path) -> None:
    config = _dense(tmp_path, stride=1)
    staging = config.writer.staging_dir
    staging.mkdir()
    with pytest.raises(TooManyPatchesError):
        run_generate(config)
    assert staging.is_dir()


# ── 3. An output folder that holds other files ───────────────────────────────


def test_the_plan_lists_a_staging_dir_with_other_files(tmp_path: Path) -> None:
    config = _xyz_config(tmp_path)
    staging = config.writer.staging_dir
    staging.mkdir()
    (staging / "other.txt").write_text("not mapcv's\n")
    estimate = plan(config)
    assert len(estimate.blocking) == 1
    assert (
        "already holds files that mapcv did not write (first: other.txt)" in (estimate.blocking[0])
    )

    with pytest.raises(StagingDirError) as raised:
        run_generate(config)
    assert str(raised.value) == estimate.blocking[0]
    assert sorted(path.name for path in staging.iterdir()) == ["other.txt"]

    code, output = _cli_plan(tmp_path, config)
    assert code == 1
    assert "This config cannot be built" in output and "other.txt" in output


def test_a_staging_dir_with_a_manifest_or_nothing_else_is_fine(tmp_path: Path) -> None:
    config = _xyz_config(tmp_path)
    assert plan(config).blocking == []  # not there yet
    config.writer.staging_dir.mkdir()
    (config.writer.staging_dir / ".gitkeep").write_text("")
    assert plan(config).blocking == []


# ── 4. A link out of the output folder ───────────────────────────────────────


def test_the_plan_lists_a_link_out_of_the_staging_dir_in_the_words_of_generate(
    tmp_path: Path,
) -> None:
    config = _xyz_config(tmp_path)
    staging = config.writer.staging_dir
    staging.mkdir()
    outside = tmp_path / "outside.txt"
    outside.write_text("not in the dataset\n")
    (staging / "train.txt").symlink_to(outside)
    estimate = plan(config)
    assert len(estimate.blocking) == 1
    assert estimate.blocking[0].startswith("Unsafe link: ")

    with pytest.raises(_confine.LinkEscapeError) as raised:
        run_generate(config)
    assert estimate.blocking[0] == f"Unsafe link: {raised.value}"
    assert outside.read_text() == "not in the dataset\n"

    code, output = _cli_plan(tmp_path, config)
    assert code == 1
    assert "Unsafe link" in output and "Looks right?" not in output


# ── 5. What a refused run leaves in the output folder ────────────────────────


def _fail_after_making(monkeypatch: pytest.MonkeyPatch, *names: str) -> None:
    """Make ``generate`` stop, after it made these files (folders end in a slash)."""

    def run(config: MapcvConfig, *args: Any) -> None:
        for name in names:
            path = config.writer.staging_dir / name
            if name.endswith("/"):
                path.mkdir(parents=True, exist_ok=True)
            else:
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text("data\n")
        raise RuntimeError("stopped")

    monkeypatch.setattr(pipeline, "_generate_locked", run)


def test_a_refused_run_removes_the_empty_folder_it_made_and_its_lock(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config = _xyz_config(tmp_path)
    _fail_after_making(monkeypatch, "images/train/")
    with pytest.raises(RuntimeError):
        run_generate(config)
    # The lock file is removed before the cleanup; otherwise the folder could not be.
    assert not config.writer.staging_dir.exists()


def test_a_refused_run_never_removes_a_folder_holding_a_file(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config = _xyz_config(tmp_path)
    _fail_after_making(monkeypatch, "images/a.png", "empty/")
    with pytest.raises(RuntimeError):
        run_generate(config)
    staging = config.writer.staging_dir
    assert (staging / "images" / "a.png").read_text() == "data\n"
    assert not (staging / "empty").exists()
    assert not (staging / ".mapcv.lock").exists()
