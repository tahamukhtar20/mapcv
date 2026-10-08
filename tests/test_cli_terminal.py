"""The CLI's terminal behaviour: plain output when piped, exit codes, and messages.

Every command is run through Typer's test runner, whose output is not a terminal, the
same as ``mapcv generate x.yaml | cat`` or ``> log.txt``.
"""

from __future__ import annotations

import errno
import json
from pathlib import Path
from typing import Any

import pytest
from typer.testing import CliRunner

from mapcv import cli
from mapcv.cli import app
from mapcv.manifest import Manifest, ManifestEntry, PatchSummary, SourceRecord, TargetRecord
from mapcv.pipeline import GenerateResult

runner = CliRunner()

_CONFIG = """\
region: {{west: 74.20, south: 31.40, east: 74.40, north: 31.60}}
imagery: {{type: xyz, zoom: 16, source: esri_satellite}}
sampler: {{patch_size: 256}}
writer: {{staging_dir: '{staging_dir}'}}
"""

# Spinner frames, the bar's characters, a carriage return and an escape sequence:
# what a live terminal gets and a log file must not.
_TTY_ONLY = ("\x1b", "\r", "⠋", "⠙", "⠹", "━", "eta ")


def _config(tmp_path: Path, staging: Path | None = None) -> Path:
    path = tmp_path / "config.yaml"
    path.write_text(_CONFIG.format(staging_dir=staging or tmp_path / "out"), encoding="utf-8")
    return path


def _dataset(staging: Path, patches: int = 3) -> Manifest:
    staging.mkdir(parents=True, exist_ok=True)
    manifest = Manifest(
        target=TargetRecord(type="segmentation", class_map={"obj": 1}, ignore_index=255),
        sources=[SourceRecord(patch_shape=[1, 1, 3], dtype="uint8", crs="EPSG:3857")],
    )
    for index in range(patches):
        manifest.patches.append(
            ManifestEntry(
                row=index,
                col=0,
                padded=False,
                chunk=0,
                files={"image": f"Images/p{index}.png", "mask": f"Masks/p{index}.png"},
                summary=PatchSummary(class_pixels={"0": 3, "1": 1}, empty_ratio=0.0),
            )
        )
    manifest.save(staging / "manifest.json")
    return manifest


def _fake_generate(staging: Path, chunks: int = 20) -> Any:
    def run(config: Any, on_chunk: Any = None) -> GenerateResult:
        manifest = _dataset(staging)
        for name in ("Images", "Masks", "splits"):
            (staging / name).mkdir(exist_ok=True)
        (staging / "patches.geojson").write_text("{}", encoding="utf-8")
        for done in range(chunks + 1):
            on_chunk(done, chunks)
        return GenerateResult(
            staging_dir=staging,
            manifest=manifest,
            new_patches=3,
            split_counts={"train": 2, "val": 0, "test": 1},
            tiles_requested=12,
            tiles_failed=0,
            seconds=3725,
        )

    return run


def test_piped_generate_prints_plain_progress_lines(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    staging = tmp_path / "out"
    monkeypatch.setattr("mapcv.cli.run_generate", _fake_generate(staging))
    result = runner.invoke(app, ["generate", str(_config(tmp_path))])
    assert result.exit_code == 0, result.output
    for marker in _TTY_ONLY:
        assert marker not in result.output, marker
    progress = [line for line in result.output.splitlines() if line.startswith("Reading imagery")]
    # One line per tenth of the chunks (0 % to 100 %), not one per chunk.
    assert len(progress) == 11
    assert progress[-1].startswith("Reading imagery and writing patches: 20/20 chunks (100%)")
    assert "Opening imagery…" in result.output
    assert "1h 02m" in result.output  # 3,725 seconds


def test_quiet_hides_the_plain_progress_lines(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr("mapcv.cli.run_generate", _fake_generate(tmp_path / "out"))
    result = runner.invoke(app, ["-q", "generate", str(_config(tmp_path))])
    assert result.exit_code == 0, result.output
    assert "Reading imagery" not in result.output and "Opening imagery" not in result.output
    assert "Dataset ready" in result.output


def test_the_summary_lists_the_files_that_were_written(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    staging = tmp_path / "out"
    monkeypatch.setattr("mapcv.cli.run_generate", _fake_generate(staging))
    result = runner.invoke(app, ["generate", str(_config(tmp_path))], env={"COLUMNS": "200"})
    assert result.exit_code == 0, result.output
    assert "(Images/, Masks/, manifest.json, splits/, patches.geojson)" in result.output


def test_dataset_files_lists_what_is_there_in_a_fixed_order(tmp_path: Path) -> None:
    manifest = _dataset(tmp_path)
    for name in ("Images", "Masks", "annotations", "labels", "splits"):
        (tmp_path / name).mkdir()
    for name in ("train.txt", "val.txt", "test.txt", "dataset.yaml", "classes.txt"):
        (tmp_path / name).write_text("", encoding="utf-8")
    assert cli._dataset_files(tmp_path, manifest) == [
        "Images/",
        "Masks/",
        "manifest.json",
        "splits/",
        "annotations/",
        "labels/",
        "train.txt",
        "val.txt",
        "test.txt",
        "dataset.yaml",
        "classes.txt",
    ]


def test_next_steps_quote_a_path_with_spaces(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    staging = tmp_path / "my data"
    monkeypatch.setattr("mapcv.cli.run_generate", _fake_generate(staging))
    result = runner.invoke(app, ["generate", str(_config(tmp_path, staging))])
    assert result.exit_code == 0, result.output
    assert f"mapcv info {cli._shell_path(staging)}" in result.output
    assert cli._shell_path(staging) != str(staging)


# --- exit codes ------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("args", "message"),
    [
        (["stats", "{data}", "--split", "nope"], "Invalid value for '--split'"),
        (["export", "{data}", "--format", "coco"], "Invalid value for '--format'"),
        (["export", "{data}", "--format", "zarr"], "required with --format zarr"),
        (["cache", "--expired"], "only applies with --clear"),
        (["split", "{data}", "--test-ratio", "2"], "Invalid value for '--test-ratio'"),
        (["split", "{data}", "--strategy", "nope"], "Invalid value for '--strategy'"),
        (["mcp", "--root", "{data}/nope"], "not a folder: "),
        (["plan"], "Missing argument"),
        (["nope"], "No such command"),
    ],
)
def test_bad_command_line_usage_exits_with_2(tmp_path: Path, args: list[str], message: str) -> None:
    _dataset(tmp_path / "data")
    args = [arg.replace("{data}", str(tmp_path / "data")) for arg in args]
    result = runner.invoke(app, args, env={"COLUMNS": "200"})
    assert result.exit_code == 2, result.output
    assert message in result.output and "Traceback" not in result.output


@pytest.mark.parametrize("command", ["info", "split", "stats", "card", "verify", "export"])
def test_every_dataset_command_reports_a_missing_dataset_alike(
    tmp_path: Path, command: str
) -> None:
    extra = ["--format", "terratorch"] if command == "export" else []
    result = runner.invoke(app, [command, str(tmp_path / "nowhere"), *extra])
    assert result.exit_code == 1
    assert "No manifest found at" in result.output
    assert "Pass the dataset folder that mapcv generate wrote" in result.output


def test_errors_go_to_stderr(tmp_path: Path) -> None:
    result = runner.invoke(app, ["plan", str(tmp_path / "missing.yaml")])
    assert result.exit_code == 1
    assert result.stdout == ""
    assert "Config file not found" in result.stderr and "mapcv init" in result.stderr


def test_ctrl_c_in_any_command_exits_130_with_one_line(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    def interrupt(config: Any) -> None:
        raise KeyboardInterrupt

    monkeypatch.setattr("mapcv.cli.make_plan", interrupt)
    result = runner.invoke(app, ["plan", str(_config(tmp_path))])
    assert result.exit_code == 130
    assert result.stderr.strip() == "Interrupted."


def test_ctrl_c_in_the_wizard_writes_nothing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    def interrupt() -> str:
        raise KeyboardInterrupt

    monkeypatch.setattr("mapcv.cli._wizard", interrupt)
    target = tmp_path / "mapcv.yaml"
    result = runner.invoke(app, ["init", "--interactive", str(target)])
    assert result.exit_code == 130
    assert "Interrupted. Nothing was written." in result.output
    assert not target.exists()


def test_the_wizard_says_what_to_do_when_the_input_ends(tmp_path: Path) -> None:
    target = tmp_path / "mapcv.yaml"
    result = runner.invoke(app, ["init", "--interactive", str(target)], input="esri\n")
    assert result.exit_code == 1
    assert "The input ended before the last question" in result.output
    assert "mapcv init --template xyz" in result.output and "Traceback" not in result.output
    assert not target.exists()


def test_the_wizard_asks_about_overwriting_before_the_questions(tmp_path: Path) -> None:
    target = tmp_path / "mapcv.yaml"
    target.write_text("mine", encoding="utf-8")
    result = runner.invoke(app, ["init", "--interactive", str(target)], input="n\n")
    assert result.exit_code == 1
    assert "1/4 Imagery" not in result.output and "--force" in result.output
    assert target.read_text(encoding="utf-8") == "mine"


def test_init_into_a_missing_folder_is_a_message(tmp_path: Path) -> None:
    result = runner.invoke(app, ["init", str(tmp_path / "no" / "such.yaml"), "-t", "xyz"])
    assert result.exit_code == 1
    assert "Cannot write" in result.output and "Traceback" not in result.output


# --- messages ----------------------------------------------------------------------------


def test_invalid_yaml_names_the_line(tmp_path: Path) -> None:
    config = tmp_path / "broken.yaml"
    config.write_text("region:\n  west: [1, 2\n", encoding="utf-8")
    result = runner.invoke(app, ["validate", str(config)], env={"COLUMNS": "200"})
    assert result.exit_code == 1
    assert "not valid YAML at line" in result.output and "Traceback" not in result.output


@pytest.mark.parametrize("text", ["", "just a string\n", "[1, 2]\n"])
def test_a_config_without_settings_says_so(tmp_path: Path, text: str) -> None:
    config = tmp_path / "empty.yaml"
    config.write_text(text, encoding="utf-8")
    result = runner.invoke(app, ["plan", str(config)], env={"COLUMNS": "200"})
    assert result.exit_code == 1
    assert "it holds no settings" in result.output


def test_a_folder_given_as_config_says_so(tmp_path: Path) -> None:
    result = runner.invoke(app, ["validate", str(tmp_path)], env={"COLUMNS": "200"})
    assert result.exit_code == 1
    assert "is a folder" in result.output and "Errno" not in result.output


@pytest.mark.parametrize(
    ("code", "expected"),
    [
        (errno.EACCES, "permission denied"),
        (errno.ENOSPC, "no space left on the disk"),
    ],
)
def test_generate_explains_permission_and_disk_full_errors(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, code: int, expected: str
) -> None:
    def fail(config: Any, on_chunk: Any = None) -> None:
        raise OSError(code, "the operating system's words", str(tmp_path / "out"))

    monkeypatch.setattr("mapcv.cli.run_generate", fail)
    result = runner.invoke(app, ["generate", str(_config(tmp_path))], env={"COLUMNS": "300"})
    assert result.exit_code == 1
    assert f"Generation failed: {expected}: {tmp_path / 'out'}" in result.output
    assert "writer.staging_dir" in result.output and "finished chunks are kept" in result.output
    assert "Errno" not in result.output


def test_validation_errors_wrap_under_their_message(tmp_path: Path) -> None:
    config = tmp_path / "bad.yaml"
    config.write_text(
        "region: {west: 5, south: 52, east: 4, north: 53}\n"
        "imagery: {type: xyz, zoom: 18, source: google_satellite}\n"
        "sampler: {patch_size: 256}\nwriter: {staging_dir: out}\n",
        encoding="utf-8",
    )
    result = runner.invoke(app, ["validate", str(config)], env={"COLUMNS": "60"})
    assert result.exit_code == 1
    lines = result.output.splitlines()
    first = next(index for index, line in enumerate(lines) if "imagery.source" in line)
    assert lines[first].startswith("  • ") and lines[first + 1].startswith("    ")
    items = [line for line in lines if line.startswith("  ")]
    assert all(line == line.rstrip() for line in items)  # not padded to the width


def test_info_keeps_the_end_of_a_long_path_in_its_title(tmp_path: Path) -> None:
    staging = tmp_path / ("a-very-long-folder-name-" * 4) / "dataset-at-the-end"
    _dataset(staging)
    result = runner.invoke(app, ["info", str(staging)], env={"COLUMNS": "80"})
    assert result.exit_code == 0, result.output
    title = result.output.splitlines()[0]
    # The start of the path gives way; its end, the dataset's folder, stays.
    assert "─ …" in title and "dataset-at-the-end ─" in title


@pytest.mark.parametrize(
    ("count", "text"), [(0, "0 patches"), (1, "1 patch"), (1234, "1,234 patches")]
)
def test_plural(count: int, text: str) -> None:
    assert cli._plural(count, "patch", "patches") == text


@pytest.mark.parametrize(
    ("seconds", "text"), [(0, "0s"), (59.9, "59s"), (65, "1m 05s"), (7260, "2h 01m")]
)
def test_duration(seconds: float, text: str) -> None:
    assert cli._duration(seconds) == text


@pytest.mark.parametrize(
    ("value", "text"), [(4.9829864501953125, "4.982986"), (52.0, "52"), (-0.5, "-0.5")]
)
def test_coordinates_have_six_decimals_at_most(value: float, text: str) -> None:
    assert cli._coordinate(value) == text


# Class names from a label file, as written there: Rich markup and terminal control
# sequences must be printed as text, never interpreted.
_MARKUP_NAMES = ("road [/]", "building [residential]", "x [link=https://evil.example]y[/link]")
_CONTROL_NAMES = ("\x1b[31mred", "\x1b]0;pwned\x07")


def test_plan_prints_class_names_and_paths_as_text(tmp_path: Path) -> None:
    features = [
        {
            "type": "Feature",
            "properties": {"cls": name},
            "geometry": {
                "type": "Polygon",
                "coordinates": [[[74.3, 31.5], [74.31, 31.5], [74.31, 31.51], [74.3, 31.5]]],
            },
        }
        for name in (*_MARKUP_NAMES, *_CONTROL_NAMES)
    ]
    labels = tmp_path / "labels [final].geojson"
    labels.write_text(json.dumps({"type": "FeatureCollection", "features": features}))
    config = _config(tmp_path, tmp_path / "out [v2]")
    config.write_text(
        config.read_text() + f"labels: {{path: '{labels.name}', label_field: cls}}\n",
        encoding="utf-8",
    )
    result = runner.invoke(app, ["plan", str(config)], env={"COLUMNS": "400"})
    assert result.exit_code == 0, result.output
    assert "\x1b" not in result.output and "\x07" not in result.output
    for name in _MARKUP_NAMES:
        assert name in result.output
    assert "\\x1b[31mred" in result.output and "\\x1b]0;pwned\\x07" in result.output
    assert "labels [final].geojson" in result.output and "out [v2]" in result.output


def test_info_prints_class_names_as_text(tmp_path: Path) -> None:
    staging = tmp_path / "dataset"
    names = (*_MARKUP_NAMES, *_CONTROL_NAMES)
    manifest = _dataset(staging)
    manifest.target = TargetRecord(
        type="segmentation",
        class_map={name: index for index, name in enumerate(names, start=1)},
        ignore_index=255,
    )
    for entry in manifest.patches:
        entry["summary"]["class_pixels"] = {str(cid): 1 for cid in range(len(names) + 1)}
    manifest.save(staging / "manifest.json")
    result = runner.invoke(app, ["info", str(staging)], env={"COLUMNS": "200"})
    assert result.exit_code == 0, result.output
    assert "\x1b" not in result.output and "\x07" not in result.output
    for name in _MARKUP_NAMES:
        assert name in result.output
    assert "\\x1b[31mred" in result.output


def test_escape_keeps_newlines_and_shows_other_control_characters() -> None:
    assert cli.escape("a [b]\n\x1b\u202e") == "a \\[b]\n\\x1b\\u202e"
