"""User errors found by the 0.3.0 docs audit: each one gives a message, never a traceback,
and nothing is silently accepted that cannot work."""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from typing import Any

import pytest
from pydantic import ValidationError
from typer.testing import CliRunner

pytest.importorskip("rasterio", reason="the GeoTIFFs are written with rasterio")
from test_geotiff_imagery import config_for, make_raster, write_labels

from mapcv.agent_tools import Sandbox, ToolFailure, ToolState, describe_config_schema, plan
from mapcv.cli import app
from mapcv.config import GeoTiffImageryConfig, LabelsConfig, MapcvConfig
from mapcv.manifest import Manifest, ManifestMismatchError
from mapcv.pipeline import run_generate
from mapcv.verify import verify_dataset, write_checksums

runner = CliRunner()


def _dataset(tmp_path: Path, **config: Any) -> Path:
    raster = make_raster(tmp_path, width=320, height=256, count=3)
    region = raster.region()
    labels = write_labels(tmp_path, region)
    cfg = config_for(tmp_path, {"path": str(raster.path)}, region, labels=labels)
    if config:
        cfg = MapcvConfig.model_validate(
            {**cfg.model_dump(mode="json", exclude_unset=True), **config}
        )
    run_generate(cfg)
    return Path(cfg.writer.staging_dir)


def _no_traceback(output: str) -> None:
    assert "Traceback" not in output and 'File "' not in output, output


# ── A corrupt manifest ───────────────────────────────────────────────────────


@pytest.mark.parametrize(
    "text", ['{"version": 3, ', '{"version": 3, "patches": "no"}', "\xff\xfe", "[]"]
)
def test_a_corrupt_manifest_is_a_manifest_error(tmp_path: Path, text: str) -> None:
    path = tmp_path / "manifest.json"
    path.write_bytes(text.encode("latin-1"))
    with pytest.raises(ManifestMismatchError, match="manifest"):
        Manifest.load(path)


@pytest.mark.parametrize("command", ["info", "split", "stats", "card"])
def test_commands_report_a_corrupt_manifest_without_a_traceback(
    tmp_path: Path, command: str
) -> None:
    (tmp_path / "manifest.json").write_text('{"version": 3, ', encoding="utf-8")
    result = runner.invoke(app, [command, str(tmp_path)])
    assert result.exit_code == 1
    _no_traceback(result.output)
    assert "cannot be read" in " ".join(result.output.split())


# ── split, stats ─────────────────────────────────────────────────────────────


def test_split_by_region_without_regions_is_a_message(tmp_path: Path) -> None:
    staging = _dataset(tmp_path)
    result = runner.invoke(app, ["split", str(staging), "--strategy", "region"])
    assert result.exit_code == 1
    _no_traceback(result.output)
    assert "region.path" in result.output


def test_stats_names_an_unreadable_patch(tmp_path: Path) -> None:
    staging = _dataset(tmp_path)
    manifest = Manifest.load(staging / "manifest.json")
    broken = manifest.patches[0]["files"]["image"]
    (staging / broken).write_bytes(b"not a png")
    result = runner.invoke(app, ["stats", str(staging), "--split", "all"])
    assert result.exit_code == 1
    _no_traceback(result.output)
    flat = " ".join(result.output.split())
    assert broken in flat and "mapcv verify --deep" in flat


# ── Config validation ────────────────────────────────────────────────────────

BASE: dict[str, Any] = {
    "region": {"west": 4.9, "south": 52.3, "east": 4.91, "north": 52.31},
    "imagery": {"type": "geotiff", "path": "ortho.tif"},
    "sampler": {"patch_size": 64},
    "writer": {"staging_dir": "out"},
}


def test_label_errors_name_the_config_key_not_the_union_tag(tmp_path: Path) -> None:
    config = tmp_path / "mapcv.yaml"
    config.write_text(
        "region: {west: 4.9, south: 52.3, east: 4.91, north: 52.31}\n"
        "imagery: {type: xyz, zoom: 15, source: esri_satellite}\n"
        "labels: {path: x.geojson, lable_field: a}\n"
        "sampler: {patch_size: 256}\nwriter: {staging_dir: out}\n",
        encoding="utf-8",
    )
    result = runner.invoke(app, ["validate", str(config)])
    assert result.exit_code == 1
    assert "labels.lable_field" in result.output
    assert "labels.vector" not in result.output


@pytest.mark.parametrize(
    ("changes", "message"),
    [
        (
            {"writer": {"staging_dir": "out", "image_format": "tif", "world_files": True}},
            "image-only",
        ),
        (
            {
                "task": "regression",
                "labels": {"type": "continuous", "path": "chm.tif"},
                "writer": {"staging_dir": "out", "image_format": "tif", "world_files": True},
            },
            "regression",
        ),
        ({"split": {"strategy": "region"}}, "region.path"),
    ],
)
def test_configs_that_cannot_work_are_refused(changes: dict[str, Any], message: str) -> None:
    with pytest.raises(ValidationError, match=message):
        MapcvConfig.model_validate({**BASE, **changes})


def test_a_loopback_ipv6_url_is_not_a_glob_pattern() -> None:
    config = GeoTiffImageryConfig(path="http://[::1]:8000/ortho.tif")
    assert not config.is_pattern
    assert config.files() == ["http://[::1]:8000/ortho.tif"]
    with pytest.raises(ValidationError, match="local files only"):
        GeoTiffImageryConfig(path="https://example.com/tiles/*.tif")


def test_a_folder_as_geotiff_path_suggests_a_pattern(tmp_path: Path) -> None:
    (tmp_path / "tiles").mkdir()
    config = GeoTiffImageryConfig(path=str(tmp_path / "tiles"))
    with pytest.raises(ValueError, match=r"is a folder.*tiles/\*\.tif"):
        config.files()


# ── MCP server ───────────────────────────────────────────────────────────────


@pytest.mark.skipif(sys.platform == "win32", reason="symlinks need privileges on Windows")
def test_a_mosaic_pattern_cannot_reach_outside_the_root(tmp_path: Path) -> None:
    outside = tmp_path / "outside"
    outside.mkdir()
    raster = make_raster(outside, width=128, height=128, count=3)
    root = tmp_path / "root"
    (root / "tiles").mkdir(parents=True)
    os.symlink(raster.path, root / "tiles" / "a.tif")
    region = raster.region()
    yaml_text = (
        f"region: {json.dumps(region)}\n"
        "imagery: {type: geotiff, path: 'tiles/*.tif'}\n"
        "sampler: {patch_size: 64}\nwriter: {staging_dir: out}\n"
    )
    state = ToolState(Sandbox(root))
    with pytest.raises(ToolFailure, match="outside the folder"):
        plan(state, yaml_text=yaml_text)


def test_the_schema_tool_knows_every_imagery_type() -> None:
    from typing import get_args

    from mapcv.config import ImageryConfig

    types = [model.model_fields["type"].default for model in get_args(get_args(ImageryConfig)[0])]
    rules = describe_config_schema(ToolState(Sandbox(Path.cwd()))).data["rules"]
    assert sorted(rules["imagery_types"]) == sorted(types)
    assert rules["image_formats_per_imagery"]["stac_cog"] == ["npy", "tif"]


# ── Checksums ────────────────────────────────────────────────────────────────


def test_checksums_cover_every_file_of_a_detection_dataset(tmp_path: Path) -> None:
    staging = _dataset(tmp_path, task="detection", detection={"formats": ["coco", "yolo"]})
    listed = {
        line.split("  ", 1)[1]
        for line in write_checksums(staging).read_text(encoding="utf-8").splitlines()
    }
    on_disk = {
        path.relative_to(staging).as_posix()
        for path in staging.rglob("*")
        if path.is_file() and path.name != "SHA256SUMS"
    }
    assert listed == on_disk
    assert any(name.startswith("annotations/") for name in listed)
    coco = next(staging.glob("annotations/instances_*.json"))
    coco.write_text(coco.read_text(encoding="utf-8") + " ", encoding="utf-8")
    report = verify_dataset(staging)
    assert not report.ok
    assert any(coco.name in problem for problem in report.problems)


# ── The wizard ───────────────────────────────────────────────────────────────


def test_the_wizard_asks_for_labels_when_the_area_file_is_not_them(tmp_path: Path) -> None:
    feature = (
        '{"type":"FeatureCollection","features":[{"type":"Feature","properties":{"kind":"roof"},'
        '"geometry":{"type":"Polygon","coordinates":[[[74.3,31.5],[74.31,31.5],'
        "[74.31,31.51],[74.3,31.5]]]}}]}"
    )
    area = tmp_path / "aoi.geojson"
    area.write_text(feature, encoding="utf-8")
    labels = tmp_path / "roofs.geojson"
    labels.write_text(feature, encoding="utf-8")
    out = tmp_path / "mapcv.yaml"
    answers = [
        "esri", str(area), "17", "n", str(labels), "kind", "", "256", "./ds", "y",
    ]  # fmt: skip
    result = runner.invoke(
        app, ["init", str(out), "--interactive"], input="\n".join(answers) + "\n"
    )
    assert result.exit_code == 0, result.output
    config = MapcvConfig.from_yaml(out)
    assert isinstance(config.labels, LabelsConfig) and config.labels.path is not None
    assert config.labels.path.name == "roofs.geojson"


# ── plan and generate stop early; messages say one thing once ────────────────


def _yaml(tmp_path: Path, extra: str = "") -> Path:
    raster = make_raster(tmp_path, width=320, height=256, count=3)
    region = raster.region()
    config = tmp_path / "mapcv.yaml"
    config.write_text(
        f"region: {json.dumps(region)}\n"
        f"imagery: {{type: geotiff, path: '{raster.path.as_posix()}'}}\n"
        "sampler: {patch_size: 64, edge_strategy: drop}\n"
        "writer: {staging_dir: out}\n" + extra,
        encoding="utf-8",
    )
    return config


@pytest.mark.parametrize("command", ["plan", "generate"])
def test_a_missing_label_file_stops_plan_and_generate(tmp_path: Path, command: str) -> None:
    config = _yaml(tmp_path, "labels: {path: nowhere.geojson}\n")
    result = runner.invoke(
        app, [command, str(config), "--yes"][: 3 if command == "generate" else 2]
    )
    assert result.exit_code == 1
    flat = " ".join(result.output.split())
    assert "labels.path not found" in flat and "Looks right" not in flat
    assert not (tmp_path / "out").exists()


def test_a_run_without_patches_fails(tmp_path: Path) -> None:
    config = _yaml(tmp_path)
    config.write_text(
        config.read_text().replace("patch_size: 64", "patch_size: 100000"), encoding="utf-8"
    )
    result = runner.invoke(app, ["generate", str(config), "--yes"])
    assert result.exit_code == 1
    assert "No patches were written" in result.output
    assert "Dataset ready" not in result.output


def test_a_typo_suggests_the_key(tmp_path: Path) -> None:
    config = _yaml(tmp_path)
    config.write_text(
        config.read_text()
        .replace("patch_size:", "patch_sise:")
        .replace("staging_dir", "stagin_dir"),
        encoding="utf-8",
    )
    result = runner.invoke(app, ["validate", str(config)])
    assert result.exit_code == 1
    flat = " ".join(result.output.split())
    assert "sampler.patch_sise: unknown key; did you mean patch_size?" in flat
    assert "writer.stagin_dir: unknown key; did you mean staging_dir?" in flat
    assert "Field required" not in flat


def test_library_deprecation_warnings_are_not_shown() -> None:
    import warnings as w

    from mapcv.cli import _show_warnings

    with w.catch_warnings(record=True) as caught:
        w.simplefilter("always")
        w.warn("shapely.ops.transform() is deprecated", DeprecationWarning, stacklevel=1)
        w.warn("a mapcv warning", UserWarning, stacklevel=1)
    from mapcv import cli

    printed: list[str] = []
    original = cli._console.print
    cli._console.print = lambda *args, **kwargs: printed.append(str(args[0]))  # type: ignore[method-assign]
    try:
        _show_warnings(caught, set())
    finally:
        cli._console.print = original  # type: ignore[method-assign]
    assert len(printed) == 1 and "a mapcv warning" in printed[0]
