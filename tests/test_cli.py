"""Tests for the mapcv CLI (typer commands)."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Optional

import pytest
from typer.testing import CliRunner

from mapcv.cli import app
from mapcv.writer import Manifest

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


def _write_config(tmp_path: Path, staging: Optional[Path] = None) -> Path:
    staging = staging or tmp_path / "output"
    p = tmp_path / "config.yaml"
    p.write_text(_VALID_CONFIG.format(staging_dir=staging))
    return p


def _write_manifest(staging_dir: Path, n: int = 20) -> None:
    staging_dir.mkdir(parents=True, exist_ok=True)
    m = Manifest(class_map={"bg": 0, "obj": 1})
    for i in range(n):
        from mapcv.writer import ManifestEntry

        m.patches.append(
            ManifestEntry(
                filename=f"patch_{i:07d}.png",
                mask_filename=None,
                row=i,
                col=0,
                padded=False,
                strip_index=0,
                per_class_pixel_counts={},
                empty_ratio=0.0,
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
    result = runner.invoke(app, ["init"])
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

    def mock_run_generate(config: Any) -> None:
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


def test_validate_reports_deprecated_legacy_config(tmp_path: Path) -> None:
    p = _write_config(tmp_path)
    p.write_text(
        p.read_text()
        .replace("  north: 31.60\n", "  north: 31.60\n  zoom: 16\n")
        .replace("imagery:\n  type: xyz\n  zoom: 16\n", "tiles:\n")
    )
    result = runner.invoke(app, ["validate", str(p)])
    assert result.exit_code == 0
    assert "Deprecated" in result.output


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
    assert runner.invoke(app, ["info", str(tmp_path / "nope")]).exit_code == 1


@pytest.mark.parametrize("template", ["xyz", "sentinel2"])
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
    answers = "\n".join(["esri", str(labels), "17", "y", "kind", "256", "./ds", "y"]) + "\n"
    result = runner.invoke(app, ["init", str(out), "--interactive"], input=answers)
    assert result.exit_code == 0, result.output
    from mapcv.config import MapcvConfig

    config = MapcvConfig.from_yaml(out)
    assert config.labels is not None and config.labels.label_field == "kind"
    assert config.split is not None and config.split.strategy == "spatial"
    assert config.region.west == pytest.approx(74.3)


@pytest.mark.parametrize(
    "value", [r"C:\Users\runner\aoi.geojson", "it's here", "https://t.example.com/{z}/{x}/{y}.png"]
)
def test_wizard_yaml_strings_round_trip(value: str) -> None:
    import yaml

    from mapcv.cli import _yaml_str

    assert yaml.safe_load(f"key: {_yaml_str(value)}")["key"] == value
