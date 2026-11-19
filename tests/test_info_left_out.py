"""``mapcv info`` and the MCP ``info`` tool show the patches a spatial split left out."""

from __future__ import annotations

import json
from pathlib import Path

from typer.testing import CliRunner

from mapcv.agent_tools import Sandbox, ToolState, info
from mapcv.cli import app
from mapcv.manifest import Manifest, ManifestEntry, PatchSummary, SourceRecord, TargetRecord
from mapcv.splitter import SplitterConfig, split_dataset

runner = CliRunner()


def _dataset(folder: Path, patch: int, stride: int, size: int = 4096) -> dict[str, int]:
    """A grid dataset split spatially; returns the counts the splitter reports."""
    manifest = Manifest(
        target=TargetRecord(type="segmentation", class_map={"obj": 1}, ignore_index=255),
        sources=[SourceRecord(patch_shape=[1, patch, patch], dtype="uint8", crs="EPSG:3857")],
        sampler={"patch_size": patch, "stride": stride, "mode": "grid"},
    )
    for row in range(0, size - patch + 1, stride):
        for col in range(0, size - patch + 1, stride):
            name = f"patch_{len(manifest.patches):07d}.png"
            manifest.patches.append(
                ManifestEntry(
                    row=row,
                    col=col,
                    padded=False,
                    chunk=0,
                    files={"image": f"Images/{name}", "mask": f"Masks/{name}"},
                    summary=PatchSummary(class_pixels={"0": 3, "1": 1}, empty_ratio=0.0),
                )
            )
    folder.mkdir(parents=True)
    manifest.save(folder / "manifest.json")
    return split_dataset(manifest, SplitterConfig(labeled_ratios=[]), folder / "splits")


def _tool_info(root: Path) -> tuple[dict[str, object], str]:
    result = info(ToolState(Sandbox(root, False)), "ds")
    return result.data, result.summary


def test_info_shows_the_overlapping_patches_a_spatial_split_left_out(tmp_path: Path) -> None:
    counts = _dataset(tmp_path / "ds", patch=256, stride=128)
    assert counts["dropped"] > 0
    result = runner.invoke(app, ["info", str(tmp_path / "ds")])
    assert result.exit_code == 0, result.output
    assert f"{counts['dropped']} overlapping patches left out" in " ".join(
        result.output.replace("│", " ").split()
    )
    data, text = _tool_info(tmp_path)
    assert data["dropped_overlapping"] == counts["dropped"]
    assert "overlapping patches left out" in text


def test_info_says_nothing_about_left_out_without_overlaps(tmp_path: Path) -> None:
    counts = _dataset(tmp_path / "ds", patch=256, stride=256)
    assert counts["dropped"] == 0
    result = runner.invoke(app, ["info", str(tmp_path / "ds")])
    assert result.exit_code == 0, result.output
    assert "left out" not in result.output
    data, text = _tool_info(tmp_path)
    assert data["dropped_overlapping"] == 0
    assert "left out" not in text


def test_info_does_not_call_a_sample_limit_left_out(tmp_path: Path) -> None:
    folder = tmp_path / "ds"
    _dataset(folder, patch=256, stride=256)
    manifest = Manifest.load(folder / "manifest.json")
    split_dataset(manifest, SplitterConfig(labeled_ratios=[], sample_limit=20), folder / "splits")
    assert json.loads((folder / "splits" / "split.json").read_text())["dropped"] == 0
    result = runner.invoke(app, ["info", str(folder)])
    assert result.exit_code == 0, result.output
    assert "left out" not in result.output
