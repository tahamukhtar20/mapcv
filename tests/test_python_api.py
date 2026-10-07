"""The Python API: ``mapcv.generate``, ``mapcv.split`` and ``mapcv.iter_patches``.

``iter_patches`` must yield exactly the patches ``generate`` writes (same order, pixels,
targets and places), the library must print nothing (messages go to the ``mapcv``
logger), and progress callbacks must see every chunk.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import numpy as np
import pytest
import yaml
from typer.testing import CliRunner

pytest.importorskip("rasterio", reason="these tests write their rasters with rasterio")

from test_multi_source import (
    PATCH,
    reference_transform,
    region_inside,
    write_labels,
    write_raster,
)

import mapcv
from mapcv.cli import app
from mapcv.config import MapcvConfig
from mapcv.manifest import Manifest
from mapcv.splitter import SplitterConfig

WIDTH, HEIGHT = 470, 400
runner = CliRunner()


@pytest.fixture
def config_file(tmp_path: Path) -> Path:
    ref = reference_transform()
    write_raster(tmp_path / "a.tif", ref, WIDTH, HEIGHT, count=4, dtype="uint16", seed=1)
    write_raster(tmp_path / "b.tif", ref, WIDTH, HEIGHT, count=2, dtype="uint16", seed=2)
    region = region_inside(ref, WIDTH, HEIGHT, margin=0.0)
    labels = write_labels(tmp_path, region_inside(ref, WIDTH, HEIGHT))
    data: dict[str, Any] = {
        "region": region,
        # Chunks of 128 rows: a few chunks, so progress and resuming are exercised.
        "imagery": {"type": "geotiff", "path": str(tmp_path / "a.tif"), "chunk_rows": 128},
        "labels": {"path": str(labels), "label_field": "kind", "classes": {"a": 1}},
        "sampler": {"patch_size": PATCH, "stride": 48, "edge_strategy": "pad"},
        "writer": {
            "staging_dir": str(tmp_path / "dataset"),
            "image_format": "npy",
            "mask_format": "npy",
        },
        "split": {"strategy": "spatial", "seed": 3},
    }
    path = tmp_path / "mapcv.yaml"
    path.write_text(yaml.safe_dump(data))
    return path


def test_generate_reports_progress_and_prints_nothing(
    config_file: Path, capsys: pytest.CaptureFixture[str], caplog: pytest.LogCaptureFixture
) -> None:
    calls: list[tuple[int, int]] = []
    result = mapcv.generate(config_file, progress=lambda done, total: calls.append((done, total)))
    total = calls[0][1]
    assert total > 1 and calls == [(done, total) for done in range(total + 1)]
    assert result.new_patches == len(result.manifest.patches) > 0
    assert result.split_counts is not None

    with caplog.at_level(logging.INFO, logger="mapcv"):
        again = mapcv.generate(MapcvConfig.from_yaml(config_file))
    assert again.new_patches == 0
    assert any("Nothing left to do" in record.getMessage() for record in caplog.records)
    out = capsys.readouterr()
    assert out.out == "" and out.err == ""


def test_a_stopped_generation_resumes_to_the_same_dataset(
    config_file: Path, tmp_path: Path
) -> None:
    whole = MapcvConfig.from_yaml(config_file)
    mapcv.generate(whole)

    class Stop(Exception):
        pass

    def stop_after_two(done: int, total: int) -> None:
        if done == 2:
            raise Stop

    other = whole.model_copy(
        update={"writer": whole.writer.model_copy(update={"staging_dir": tmp_path / "again"})}
    )
    with pytest.raises(Stop):
        mapcv.generate(other, progress=stop_after_two)
    assert 0 < len(Manifest.load(tmp_path / "again" / "manifest.json").patches)
    mapcv.generate(other)
    assert _files(tmp_path / "again") == _files(whole.writer.staging_dir)


def _files(root: Path) -> dict[str, bytes]:
    return {
        p.relative_to(root).as_posix(): p.read_bytes()
        for p in sorted(root.rglob("*"))
        if p.is_file()
    }


def test_split_takes_a_config_or_settings(config_file: Path, tmp_path: Path) -> None:
    dataset = MapcvConfig.from_yaml(config_file).writer.staging_dir
    mapcv.generate(config_file)
    by_settings = mapcv.split(dataset, strategy="random", test_ratio=0.25, seed=9)
    lists = (dataset / "splits" / "test.txt").read_text()
    by_config = mapcv.split(
        str(dataset), SplitterConfig(strategy="random", test_ratio=0.25, seed=9)
    )
    assert by_settings == by_config and (dataset / "splits" / "test.txt").read_text() == lists
    assert by_settings["test"] == -(-len(Manifest.load(dataset / "manifest.json").patches) // 4)
    with pytest.raises(TypeError, match="not both"):
        mapcv.split(dataset, SplitterConfig(), seed=1)


@pytest.mark.parametrize("sources", [1, 2])
def test_iter_patches_yields_what_generate_writes(config_file: Path, sources: int) -> None:
    config = MapcvConfig.from_yaml(config_file)
    if sources == 2:
        base = config.model_dump(mode="json")
        tif = base["imagery"]["path"]
        base["imagery"] = [
            {"type": "geotiff", "name": "a", "path": tif, "chunk_rows": 128},
            {"type": "geotiff", "name": "b", "path": tif.replace("a.tif", "b.tif")},
        ]
        config = MapcvConfig.model_validate(base)
    patches = list(mapcv.iter_patches(config))
    assert not config.writer.staging_dir.exists()  # nothing was written
    mapcv.generate(config)
    staging = config.writer.staging_dir
    manifest = Manifest.load(staging / "manifest.json")
    assert len(patches) == len(manifest.patches) > 0
    first = "a" if sources == 2 else "image"
    for patch, entry in zip(patches, manifest.patches):
        assert (patch.row, patch.col, patch.padded) == (entry["row"], entry["col"], entry["padded"])
        assert patch.transform == pytest.approx(manifest.patch_transform(entry))
        assert patch.crs == manifest.source.crs
        image = np.load(staging / entry["files"][first])
        np.testing.assert_array_equal(np.moveaxis(patch.image, -1, 0), image)
        np.testing.assert_array_equal(patch.target, np.load(staging / entry["files"]["mask"]))
        if sources == 2:
            assert set(patch.others) == {"b"}
            np.testing.assert_array_equal(
                np.moveaxis(patch.others["b"], -1, 0), np.load(staging / entry["files"]["b"])
            )
        else:
            assert patch.others == {}


def test_the_cli_shows_library_messages_unless_quiet(config_file: Path) -> None:
    env = {"COLUMNS": "200"}
    assert runner.invoke(app, ["generate", str(config_file), "--yes"], env=env).exit_code == 0
    shown = runner.invoke(app, ["generate", str(config_file), "--yes"], env=env)
    assert shown.exit_code == 0 and "Nothing left to do" in shown.output
    quiet = runner.invoke(app, ["-q", "generate", str(config_file), "--yes"], env=env)
    assert quiet.exit_code == 0 and "Nothing left to do" not in quiet.output
    assert "Dataset ready" in quiet.output
    # The handler is gone afterwards: no mapcv messages leak into later output.
    assert not [
        h for h in logging.getLogger("mapcv").handlers if type(h).__name__ == "_GenerateFeedback"
    ]
