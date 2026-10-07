"""Tests for the mapcv CLI (typer commands)."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
from typer.testing import CliRunner

import mapcv
from mapcv.cli import app
from mapcv.manifest import (
    Manifest,
    ManifestEntry,
    ManifestMismatchError,
    PatchSummary,
    SourceRecord,
    TargetRecord,
)

runner = CliRunner()

_VALID_CONFIG = """\
region:
  west: 74.20
  south: 31.40
  east: 74.40
  north: 31.60
imagery:
  type: xyz
  zoom: 16
  source: esri_satellite
sampler:
  patch_size: 256
writer:
  staging_dir: {staging_dir}
"""


def _write_config(tmp_path: Path, staging: Path | None = None) -> Path:
    staging = staging or tmp_path / "output"
    p = tmp_path / "config.yaml"
    p.write_text(_VALID_CONFIG.format(staging_dir=staging))
    return p


def _write_manifest(staging_dir: Path, n: int = 20) -> None:
    staging_dir.mkdir(parents=True, exist_ok=True)
    m = Manifest(
        target=TargetRecord(type="segmentation", class_map={"obj": 1}, ignore_index=255),
        sources=[SourceRecord(patch_shape=[1, 1, 3], dtype="uint8", crs="EPSG:3857")],
    )
    for i in range(n):
        m.patches.append(
            ManifestEntry(
                row=i,
                col=0,
                padded=i == 0,
                chunk=0,
                files={"image": f"Images/patch_{i:07d}.png", "mask": f"Masks/patch_{i:07d}.png"},
                summary=PatchSummary(class_pixels={"0": 3, "1": 1}, empty_ratio=0.0),
            )
        )
    m.save(staging_dir / "manifest.json")


# ---------------------------------------------------------------------------
# validate command
# ---------------------------------------------------------------------------


def test_validate_valid_config(tmp_path: Path) -> None:
    cfg = _write_config(tmp_path)
    result = runner.invoke(app, ["validate", str(cfg)])
    assert result.exit_code == 0
    assert "valid" in result.output.lower()


def test_validate_missing_file(tmp_path: Path) -> None:
    result = runner.invoke(app, ["validate", str(tmp_path / "no_file.yaml")])
    assert result.exit_code != 0


def test_validate_bad_config(tmp_path: Path) -> None:
    bad = tmp_path / "bad.yaml"
    bad.write_text("region:\n  west: 0\n")
    result = runner.invoke(app, ["validate", str(bad)])
    assert result.exit_code != 0


def test_validate_shows_summary(tmp_path: Path) -> None:
    cfg = _write_config(tmp_path)
    result = runner.invoke(app, ["validate", str(cfg)])
    assert "16" in result.output  # zoom
    assert "256" in result.output  # patch_size


def test_validate_warns_missing_labels_path(tmp_path: Path) -> None:
    cfg = tmp_path / "config.yaml"
    staging = tmp_path / "output"
    cfg.write_text(
        _VALID_CONFIG.format(staging_dir=staging) + "labels:\n  path: /no/such/labels.kml\n"
    )
    result = runner.invoke(app, ["validate", str(cfg)])
    assert result.exit_code == 0
    assert "Warning" in result.output or "warning" in result.output.lower()


# ---------------------------------------------------------------------------
# init command
# ---------------------------------------------------------------------------


def test_init_prints_to_stdout(tmp_path: Path) -> None:
    result = runner.invoke(app, ["init", "--stdout"])
    assert result.exit_code == 0
    assert "patch_size" in result.output
    assert "region" in result.output


def test_init_writes_file(tmp_path: Path) -> None:
    out = tmp_path / "example.yaml"
    result = runner.invoke(app, ["init", str(out)])
    assert result.exit_code == 0
    assert out.exists()
    assert "patch_size" in out.read_text()


# ---------------------------------------------------------------------------
# split command
# ---------------------------------------------------------------------------


def test_split_creates_output_files(tmp_path: Path) -> None:
    staging = tmp_path / "staging"
    _write_manifest(staging, n=50)
    result = runner.invoke(app, ["split", str(staging)])
    assert result.exit_code == 0
    assert (staging / "splits" / "test.txt").exists()
    assert (staging / "splits" / "val.txt").exists()
    assert (staging / "splits" / "train.txt").exists()


def test_split_missing_directory(tmp_path: Path) -> None:
    result = runner.invoke(app, ["split", str(tmp_path / "no_dir")])
    assert result.exit_code != 0


def test_split_missing_manifest(tmp_path: Path) -> None:
    staging = tmp_path / "staging"
    staging.mkdir()
    result = runner.invoke(app, ["split", str(staging)])
    assert result.exit_code != 0


def test_split_custom_ratios(tmp_path: Path) -> None:
    staging = tmp_path / "staging"
    _write_manifest(staging, n=100)
    result = runner.invoke(
        app,
        [
            "split",
            str(staging),
            "--test-ratio",
            "0.10",
            "--val-ratio",
            "0.05",
            "--labeled-ratios",
            "0.20",
        ],
    )
    assert result.exit_code == 0
    test_lines = (staging / "splits" / "test.txt").read_text().strip().split("\n")
    import math

    assert len(test_lines) == math.ceil(100 * 0.10)


def test_split_strategy_flag(tmp_path: Path) -> None:
    staging = tmp_path / "staging"
    _write_manifest(staging, n=50)
    result = runner.invoke(app, ["split", str(staging), "--strategy", "random"])
    assert result.exit_code == 0


def test_split_invalid_test_ratio(tmp_path: Path) -> None:
    staging = tmp_path / "staging"
    _write_manifest(staging, n=50)
    result = runner.invoke(app, ["split", str(staging), "--test-ratio", "1.5"])
    assert result.exit_code != 0


def test_split_config_error(tmp_path: Path) -> None:
    # Trigger the except Exception block in split command
    staging = tmp_path / "staging"
    _write_manifest(staging, n=50)
    # providing an invalid strategy to trigger validation error
    result = runner.invoke(app, ["split", str(staging), "--strategy", "invalid_strategy"])
    assert result.exit_code != 0
    assert "Config error" in result.output


# ---------------------------------------------------------------------------
# generate command
# ---------------------------------------------------------------------------


def test_generate_missing_file(tmp_path: Path) -> None:
    result = runner.invoke(app, ["generate", str(tmp_path / "no_file.yaml")])
    assert result.exit_code != 0
    assert "not found" in result.output.lower()


def test_generate_bad_config(tmp_path: Path) -> None:
    bad = tmp_path / "bad.yaml"
    bad.write_text("region:\n  west: 0\n")
    result = runner.invoke(app, ["generate", str(bad)])
    assert result.exit_code != 0
    assert "Config error" in result.output


def test_generate_unreadable_config(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    cfg = _write_config(tmp_path)

    def mock_read_text(*args: Any, **kwargs: Any) -> str:
        raise OSError("Permission denied")

    monkeypatch.setattr(Path, "read_text", mock_read_text)

    result = runner.invoke(app, ["generate", str(cfg)])
    assert result.exit_code != 0
    assert "Config error" in result.output
    assert "Permission denied" in result.output


def test_generate_calls_run_generate(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    cfg = _write_config(tmp_path)
    called = False

    def mock_run_generate(config: Any, on_chunk: Any = None) -> None:
        nonlocal called
        called = True

    monkeypatch.setattr("mapcv.cli.run_generate", mock_run_generate)

    result = runner.invoke(app, ["generate", str(cfg)])
    assert result.exit_code == 0
    assert called


# ---------------------------------------------------------------------------
# Error handling in validate
# ---------------------------------------------------------------------------


def test_validate_unreadable_config(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    cfg = _write_config(tmp_path)

    def mock_read_text(*args: Any, **kwargs: Any) -> str:
        raise OSError("Permission denied")

    monkeypatch.setattr(Path, "read_text", mock_read_text)

    result = runner.invoke(app, ["validate", str(cfg)])
    assert result.exit_code != 0
    assert "Config error" in result.output
    assert "Permission denied" in result.output


def test_validate_shows_split_info(tmp_path: Path) -> None:
    cfg = tmp_path / "config.yaml"
    staging = tmp_path / "output"
    cfg.write_text(
        _VALID_CONFIG.format(staging_dir=staging)
        + "split:\n  test_ratio: 0.2\n  strategy: random\n"
    )
    result = runner.invoke(app, ["validate", str(cfg)])
    assert result.exit_code == 0
    assert "split" in result.output
    assert "0.2" in result.output


def test_generate_reports_runtime_errors_without_traceback(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    def fail(*args: Any, **kwargs: Any) -> None:
        raise ValueError("requested region does not intersect the EOPF product")

    monkeypatch.setattr("mapcv.cli.run_generate", fail)
    result = runner.invoke(app, ["generate", str(_write_config(tmp_path))])
    assert result.exit_code == 1
    assert "Generation failed" in result.output
    assert "does not intersect" in result.output


def test_generate_interrupt_says_how_to_resume(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    def interrupt(*args: Any, **kwargs: Any) -> None:
        raise KeyboardInterrupt

    monkeypatch.setattr("mapcv.cli.run_generate", interrupt)
    result = runner.invoke(app, ["generate", str(_write_config(tmp_path))])
    assert result.exit_code == 130
    assert "Interrupted" in result.output
    assert "run the same command again to resume" in result.output


def test_validate_redacts_url_template_secrets(tmp_path: Path) -> None:
    p = _write_config(tmp_path)
    p.write_text(
        p.read_text().replace(
            "source: esri_satellite",
            'url_template: "https://tiles.example.com/SECRET/{z}/{x}/{y}.png?key=SECRET"',
        )
    )
    result = runner.invoke(app, ["validate", str(p)])
    assert result.exit_code == 0
    assert "tiles.example.com" in result.output
    assert "SECRET" not in result.output


def _unwrapped(output: str) -> str:
    """Output with all whitespace removed, so assertions survive console line wrapping."""
    return "".join(output.split())


def test_validate_rejects_legacy_tiles_config_with_migration_hint(tmp_path: Path) -> None:
    p = _write_config(tmp_path)
    p.write_text(
        p.read_text()
        .replace("  north: 31.60\n", "  north: 31.60\n  zoom: 16\n")
        .replace("imagery:\n  type: xyz\n  zoom: 16\n", "tiles:\n")
    )
    result = runner.invoke(app, ["validate", str(p)])
    assert result.exit_code == 1
    output = _unwrapped(result.output)
    assert "Configerror" in output
    assert "`tiles:`wasreplacedby`imagery:`" in output
    assert "removedin0.3" in output
    assert "MIGRATION.md" in output


def test_validate_rejects_region_zoom_with_migration_hint(tmp_path: Path) -> None:
    p = _write_config(tmp_path)
    p.write_text(p.read_text().replace("  north: 31.60\n", "  north: 31.60\n  zoom: 16\n"))
    result = runner.invoke(app, ["validate", str(p)])
    assert result.exit_code == 1
    output = _unwrapped(result.output)
    assert "`region.zoom`wasmovedto`imagery.zoom`" in output
    assert "removedin0.3" in output
    assert "MIGRATION.md" in output


def test_init_links_provider_guidance(tmp_path: Path) -> None:
    out = tmp_path / "cfg.yaml"
    result = runner.invoke(app, ["init", str(out)])
    assert result.exit_code == 0
    assert "PROVIDERS.md" in out.read_text()


# ---------------------------------------------------------------------------
# plan / generate guard / info / init templates and wizard / version
# ---------------------------------------------------------------------------


def test_version_flag() -> None:
    result = runner.invoke(app, ["--version"])
    assert result.exit_code == 0
    assert result.output.startswith("mapcv ")
    assert result.output.strip() == f"mapcv {mapcv.__version__}"


@pytest.mark.parametrize(
    ("command", "argument"),
    [
        ("init", "[OUTPUT]"),
        ("plan", "CONFIG_PATH"),
        ("generate", "CONFIG_PATH"),
        ("validate", "CONFIG_PATH"),
        ("info", "STAGING_DIR"),
        ("split", "STAGING_DIR"),
    ],
)
def test_help_usage_names_arguments(command: str, argument: str) -> None:
    # Explicit metavars keep argument names uppercase across Typer versions
    # (Typer 0.27 wraps required ones in braces: {CONFIG_PATH}).
    result = runner.invoke(app, [command, "--help"], env={"COLUMNS": "120", "NO_COLOR": "1"})
    assert result.exit_code == 0
    usage = next(line for line in result.output.splitlines() if "Usage:" in line)
    assert f"mapcv {command} [OPTIONS]" in usage
    assert argument in usage


def test_plan_shows_estimates(tmp_path: Path) -> None:
    result = runner.invoke(app, ["plan", str(_write_config(tmp_path))])
    assert result.exit_code == 0
    assert "Plan for config.yaml" in result.output
    assert "tiles" in result.output
    assert "mapcv generate" in result.output


def test_generate_dry_run_does_not_generate(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    called: list[bool] = []
    monkeypatch.setattr("mapcv.cli.run_generate", lambda *a, **k: called.append(True))
    result = runner.invoke(app, ["generate", str(_write_config(tmp_path)), "--dry-run"])
    assert result.exit_code == 0
    assert called == []


def test_generate_large_job_needs_yes_without_a_terminal(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    called: list[bool] = []
    monkeypatch.setattr("mapcv.cli.run_generate", lambda *a, **k: called.append(True))
    config = _write_config(tmp_path)
    config.write_text(config.read_text().replace("zoom: 16", "zoom: 19"))
    result = runner.invoke(app, ["generate", str(config)])
    assert result.exit_code == 2
    assert "--yes" in result.output
    assert called == []
    result = runner.invoke(app, ["generate", str(config), "--yes"])
    assert result.exit_code == 0
    assert called == [True]


def test_config_errors_name_the_field(tmp_path: Path) -> None:
    config = _write_config(tmp_path)
    config.write_text(config.read_text().replace("zoom: 16", "zoom: 40"))
    result = runner.invoke(app, ["validate", str(config)])
    assert result.exit_code == 1
    assert "imagery.zoom" in result.output
    assert "mapcv init" in result.output


def test_info_summarizes_a_dataset(tmp_path: Path) -> None:
    staging = tmp_path / "out"
    _write_manifest(staging)
    result = runner.invoke(app, ["info", str(staging)])
    assert result.exit_code == 0
    assert "20" in result.output
    assert "segmentation" in result.output
    assert "version 3" in result.output
    assert "mask value 255" in result.output
    assert "obj" in result.output and "25.0%" in result.output  # class table
    assert runner.invoke(app, ["info", str(tmp_path / "nope")]).exit_code == 1


def test_info_reads_a_mapcv_0_2_dataset() -> None:
    dataset = Path(__file__).parent / "fixtures" / "mapcv-0.2.0" / "dataset"
    result = runner.invoke(app, ["info", str(dataset)])
    assert result.exit_code == 0, result.output
    assert "version 2 (mapcv 0.2; read as version 3)" in result.output
    assert "building" in result.output and "water" in result.output
    assert "train 7" in result.output


def test_info_refuses_a_manifest_from_a_newer_mapcv(tmp_path: Path) -> None:
    tmp_path.joinpath("manifest.json").write_text('{"version": 4, "patches": []}')
    result = runner.invoke(app, ["info", str(tmp_path)])
    assert result.exit_code == 1
    assert "newer mapcv" in result.output
    assert not isinstance(result.exception, ManifestMismatchError)


@pytest.mark.parametrize("template", ["xyz", "sentinel2", "detection", "earth-engine"])
def test_init_templates_write_parseable_configs(tmp_path: Path, template: str) -> None:
    out = tmp_path / f"{template}.yaml"
    result = runner.invoke(app, ["init", str(out), "--template", template])
    assert result.exit_code == 0
    from mapcv.config import MapcvConfig

    MapcvConfig.from_yaml(out)


def test_init_refuses_to_overwrite_without_force(tmp_path: Path) -> None:
    out = tmp_path / "mapcv.yaml"
    out.write_text("keep me")
    assert runner.invoke(app, ["init", str(out)]).exit_code == 1
    assert out.read_text() == "keep me"
    assert runner.invoke(app, ["init", str(out), "--force"]).exit_code == 0


def test_init_wizard_builds_a_config_from_a_label_file(tmp_path: Path) -> None:
    labels = tmp_path / "aoi.geojson"
    labels.write_text(
        '{"type":"FeatureCollection","features":[{"type":"Feature","properties":{"kind":"roof"},'
        '"geometry":{"type":"Polygon","coordinates":[[[74.3,31.5],[74.31,31.5],'
        "[74.31,31.51],[74.3,31.5]]]}}]}"
    )
    out = tmp_path / "mapcv.yaml"
    answers = "\n".join(["esri", str(labels), "17", "y", "kind", "", "256", "./ds", "y"]) + "\n"
    result = runner.invoke(app, ["init", str(out), "--interactive"], input=answers)
    assert result.exit_code == 0, result.output
    from mapcv.config import LabelsConfig, MapcvConfig

    config = MapcvConfig.from_yaml(out)
    assert isinstance(config.labels, LabelsConfig) and config.labels.label_field == "kind"
    assert config.split is not None and config.split.strategy == "spatial"
    assert config.region.west == pytest.approx(74.3)


@pytest.mark.parametrize(
    "value", [r"C:\Users\runner\aoi.geojson", "it's here", "https://t.example.com/{z}/{x}/{y}.png"]
)
def test_wizard_yaml_strings_round_trip(value: str) -> None:
    import yaml

    from mapcv.cli import _yaml_str

    assert yaml.safe_load(f"key: {_yaml_str(value)}")["key"] == value


def test_init_writes_mapcv_yaml_by_default(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.chdir(tmp_path)
    result = runner.invoke(app, ["init", "--template", "sentinel2"])
    assert result.exit_code == 0
    assert "eopf_zarr" in (tmp_path / "mapcv.yaml").read_text(encoding="utf-8")


def test_generate_reports_resume_mismatch_without_resume_advice(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from mapcv.writer import ManifestMismatchError

    def refuse(*args: Any, **kwargs: Any) -> None:
        raise ManifestMismatchError("dataset was generated with a different configuration")

    monkeypatch.setattr("mapcv.cli.run_generate", refuse)
    result = runner.invoke(app, ["generate", str(_write_config(tmp_path))])
    assert result.exit_code == 1
    assert "Cannot resume" in result.output
    assert "resumes" not in result.output


def test_generate_reports_unexpected_errors_and_warnings_once(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import warnings

    def fail(*args: Any, **kwargs: Any) -> None:
        warnings.warn("something odd", UserWarning)
        warnings.warn("something odd", UserWarning)
        raise TimeoutError("408 Request Timeout")

    monkeypatch.setattr("mapcv.cli.run_generate", fail)
    result = runner.invoke(app, ["generate", str(_write_config(tmp_path))])
    assert result.exit_code == 1
    assert "408 Request Timeout" in result.output
    assert result.output.count("something odd") == 1
    assert "Traceback" not in result.output


def test_split_writes_settings_and_prints_summary(tmp_path: Path) -> None:
    staging = tmp_path / "out"
    _write_manifest(staging)
    result = runner.invoke(app, ["--quiet", "split", str(staging), "--strategy", "random"])
    assert result.exit_code == 0
    assert "train" in result.output and "test" in result.output
    assert (staging / "splits" / "split.json").exists()
    assert (staging / "splits" / "train.txt").read_text().endswith("\n")


def test_wizard_hides_id_like_fields(tmp_path: Path) -> None:
    import json as _json

    from mapcv.cli import label_fields

    square = [[[0, 0], [1, 0], [1, 1], [0, 0]]]
    features = [
        {
            "type": "Feature",
            "properties": {"osm_id": i, "kind": "roof" if i % 2 else "road"},
            "geometry": {"type": "Polygon", "coordinates": square},
        }
        for i in range(300)
    ]
    path = tmp_path / "labels.geojson"
    path.write_text(_json.dumps({"type": "FeatureCollection", "features": features}))
    assert set(label_fields(path)) == {"kind"}


def test_redirected_output_on_a_legacy_code_page_does_not_crash(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import io

    from mapcv.cli import _make_output_encodable

    # What Windows gives `mapcv plan x.yaml > plan.txt`: a cp1252 file stream.
    redirected = io.TextIOWrapper(io.BytesIO(), encoding="cp1252")
    monkeypatch.setattr("sys.stdout", redirected)
    monkeypatch.setattr("sys.stderr", redirected)
    _make_output_encodable()
    print("Region 4.9 → 5.0 ✓ ⚠ ╭─╮", file=redirected)
    redirected.flush()
    assert "→ 5.0 ✓".encode() in redirected.buffer.getvalue()


def test_no_color_turns_colours_off_for_one_run(tmp_path: Path) -> None:
    from mapcv import cli

    before = cli._console.no_color
    result = runner.invoke(app, ["--no-color", "validate", str(tmp_path / "missing.yaml")])
    assert result.exit_code != 0
    assert cli._console.no_color
    runner.invoke(app, ["validate", str(tmp_path / "missing.yaml")])
    assert cli._console.no_color == before
