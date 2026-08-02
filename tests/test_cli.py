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
  zoom: 16
tiles:
  source: google_satellite
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


def test_generate_unreadable_config(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    cfg = _write_config(tmp_path)

    def mock_read_text(*args: Any, **kwargs: Any) -> str:
        raise OSError("Permission denied")

    monkeypatch.setattr(Path, "read_text", mock_read_text)

    result = runner.invoke(app, ["generate", str(cfg)])
    assert result.exit_code != 0
    assert "Config error" in result.output
    assert "Permission denied" in result.output


def test_generate_calls_run_generate(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
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


def test_validate_unreadable_config(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
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
        _VALID_CONFIG.format(staging_dir=staging) + "split:\n  test_ratio: 0.2\n  strategy: random\n"
    )
    result = runner.invoke(app, ["validate", str(cfg)])
    assert result.exit_code == 0
    assert "split" in result.output
    assert "0.2" in result.output
