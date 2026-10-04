"""The configs in examples/ stay valid as mapcv evolves (no network needed)."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from typer.testing import CliRunner

from mapcv.cli import app
from mapcv.config import MapcvConfig
from mapcv.planning import plan

_EXAMPLES = Path(__file__).resolve().parents[1] / "examples"
_CONFIGS = sorted(_EXAMPLES.glob("*/mapcv.yaml"))
# Label files are committed to the repository; keep them small.
_MAX_LABEL_BYTES = 300_000

runner = CliRunner()


def test_documented_examples_exist() -> None:
    names = {path.parent.name for path in _CONFIGS}
    assert {"quickstart", "sentinel2-landcover"} <= names


@pytest.mark.parametrize("config_path", _CONFIGS, ids=lambda path: path.parent.name)
def test_example_config_is_valid(config_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    # Examples are run from their own folder, so relative paths resolve there.
    monkeypatch.chdir(config_path.parent)
    config = MapcvConfig.from_yaml(Path("mapcv.yaml"))

    assert config.labels is not None
    assert config.labels.path.exists()
    assert config.labels.path.stat().st_size < _MAX_LABEL_BYTES
    label_file = json.loads(config.labels.path.read_text())
    assert "ODbL" in label_file["osm"]["license"]
    assert not config.writer.staging_dir.is_absolute()

    estimate = plan(config)
    assert estimate.patches > 0
    assert estimate.labels is not None and estimate.labels.polygons > 0
    assert not estimate.warnings


def test_quickstart_plan_cli(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.chdir(_EXAMPLES / "quickstart")
    result = runner.invoke(app, ["plan", "mapcv.yaml"])
    assert result.exit_code == 0, result.output
    assert "77 tiles" in result.output
    assert "building → 1" in result.output
