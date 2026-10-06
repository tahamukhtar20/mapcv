"""The configs in examples/ stay valid as mapcv evolves (no network needed)."""

from __future__ import annotations

import importlib.util
import json
import shutil
from pathlib import Path

import numpy as np
import pytest
from PIL import Image
from typer.testing import CliRunner

from mapcv import Manifest
from mapcv.cli import app
from mapcv.config import LabelsConfig, MapcvConfig
from mapcv.planning import plan

_EXAMPLES = Path(__file__).resolve().parents[1] / "examples"
_CONFIGS = sorted(_EXAMPLES.glob("*/mapcv.yaml"))
# Label files are committed to the repository; keep them small.
_MAX_LABEL_BYTES = 300_000

runner = CliRunner()

# The sdist leaves examples/ out to stay small; these tests run from a git checkout.
pytestmark = pytest.mark.skipif(
    not _EXAMPLES.is_dir(), reason="examples/ is not shipped in the source distribution"
)


def test_documented_examples_exist() -> None:
    names = {path.parent.name for path in _CONFIGS}
    assert {"quickstart", "sentinel2-landcover"} <= names


@pytest.mark.parametrize("config_path", _CONFIGS, ids=lambda path: path.parent.name)
def test_example_config_is_valid(config_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    # Examples are run from their own folder, so relative paths resolve there.
    monkeypatch.chdir(config_path.parent)
    config = MapcvConfig.from_yaml(Path("mapcv.yaml"))

    assert isinstance(config.labels, LabelsConfig)
    assert config.labels.first_path.exists()
    assert config.labels.first_path.stat().st_size < _MAX_LABEL_BYTES
    label_file = json.loads(config.labels.first_path.read_text())
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


def test_torch_dataset_example_reads_an_upgraded_0_2_dataset(tmp_path: Path) -> None:
    dataset = tmp_path / "dataset"
    fixture = Path(__file__).parent / "fixtures" / "mapcv-0.2.0" / "dataset"
    shutil.copytree(fixture, dataset)
    spec = importlib.util.spec_from_file_location(
        "torch_dataset", _EXAMPLES / "scripts" / "torch_dataset.py"
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    with pytest.raises(ValueError, match="version 2"):
        module.MapcvDataset(dataset, "train")
    # The upgrade recipe from the migration guide.
    path = dataset / "manifest.json"
    Manifest.load(path).save(path)

    train = module.MapcvDataset(dataset, "train")
    names = (dataset / "splits" / "train.txt").read_text().split()
    assert len(train) == len(names)
    sample = train[0]
    assert sample["filename"] == names[0]
    assert sample["image"].shape == (3, 192, 192)
    mask = np.asarray(Image.open(fixture / "Masks" / names[0]), dtype=np.int64)
    np.testing.assert_array_equal(sample["mask"], mask)
    assert train.class_names == {0: "background", 1: "building", 2: "water"}
    assert train.ignore_index is None
    assert sum(train.class_pixel_counts().values()) == len(train) * 192 * 192
