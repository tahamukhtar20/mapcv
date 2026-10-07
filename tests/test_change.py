"""Change detection datasets (``task: change``): before/after patches and change masks.

Patches are compared with what rasterio reads from each file, and change masks with
an independent computation: ``rasterio.features.rasterize`` of the change polygons,
or of the before and after label sets, on each patch's own transform.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import numpy.typing as npt
import pytest
from PIL import Image
from pydantic import ValidationError
from typer.testing import CliRunner

pytest.importorskip("rasterio", reason="change tests write their rasters with rasterio")
import rasterio
import rasterio.features
import rasterio.warp
from rasterio.transform import Affine
from test_multi_source import (
    EPSG,
    PATCH,
    expected_on_patch_grid,
    read_png,
    reference_transform,
    region_inside,
    write_raster,
)

from mapcv.cli import app
from mapcv.config import MapcvConfig
from mapcv.manifest import Manifest, ManifestEntry
from mapcv.pipeline import run_generate, run_split
from mapcv.planning import plan

runner = CliRunner()
WIDTH, HEIGHT = 448, 384


# ── Scene: two images and labels ─────────────────────────────────────────────


def _polygon(region: dict[str, float], *corners: tuple[float, float]) -> list[list[float]]:
    west, south = region["west"], region["south"]
    dx, dy = region["east"] - west, region["north"] - south
    ring = [[west + fx * dx, south + fy * dy] for fx, fy in corners]
    return ring + [ring[0]]


def write_features(path: Path, rings: list[tuple[list[list[float]], str | None]]) -> Path:
    features = [
        {
            "type": "Feature",
            "properties": {"kind": kind} if kind is not None else {},
            "geometry": {"type": "Polygon", "coordinates": [ring]},
        }
        for ring, kind in rings
    ]
    path.write_text(json.dumps({"type": "FeatureCollection", "features": features}))
    return path


@pytest.fixture
def scene(tmp_path: Path) -> dict[str, Any]:
    ref = reference_transform()
    write_raster(tmp_path / "before.tif", ref, WIDTH, HEIGHT, seed=1)
    write_raster(tmp_path / "after.tif", ref, WIDTH, HEIGHT, seed=2)
    region = region_inside(ref, WIDTH, HEIGHT)
    kept = _polygon(region, (0.10, 0.10), (0.30, 0.12), (0.25, 0.35))
    gone = _polygon(region, (0.55, 0.15), (0.80, 0.20), (0.70, 0.45))
    new = _polygon(region, (0.20, 0.60), (0.45, 0.62), (0.40, 0.90))
    relabeled = _polygon(region, (0.60, 0.60), (0.90, 0.65), (0.75, 0.90))
    return {
        "region": region,
        "changes": write_features(tmp_path / "changes.geojson", [(gone, None), (new, None)]),
        "before_set": write_features(
            tmp_path / "b23.geojson",
            [(kept, "house"), (gone, "house"), (relabeled, "house")],
        ),
        "after_set": write_features(
            tmp_path / "b25.geojson",
            [(kept, "house"), (new, "house"), (relabeled, "shed")],
        ),
    }


def _sources(tmp_path: Path) -> list[dict[str, Any]]:
    return [
        {"type": "geotiff", "name": "before", "path": str(tmp_path / "before.tif")},
        {"type": "geotiff", "name": "after", "path": str(tmp_path / "after.tif")},
    ]


def change_config(
    tmp_path: Path,
    region: dict[str, float],
    *,
    labels: dict[str, Any] | None = None,
    change: dict[str, Any] | None = None,
    staging: str = "dataset",
    **writer: Any,
) -> MapcvConfig:
    data: dict[str, Any] = {
        "task": "change",
        "region": region,
        "imagery": _sources(tmp_path),
        "sampler": {"patch_size": PATCH, "mode": "grid", "edge_strategy": "drop"},
        "writer": {"staging_dir": str(tmp_path / staging), "image_format": "png", **writer},
        "split": {"strategy": "spatial", "val_ratio": 0.2, "test_ratio": 0.2},
    }
    if labels is not None:
        data["labels"] = labels
    if change is not None:
        data["change"] = change
    return MapcvConfig.model_validate(data)


def _burn(path: Path, transform: Affine, field: str | None = None) -> npt.NDArray[np.uint8]:
    """rasterio's mask of a label file on a patch grid: class IDs, or 1 per feature."""
    shapes = []
    ids = {"house": 1, "shed": 2}
    for feature in json.loads(path.read_text())["features"]:
        geometry = rasterio.warp.transform_geom(
            "EPSG:4326", f"EPSG:{EPSG}", feature["geometry"], precision=-1
        )
        value = ids[feature["properties"][field]] if field else 1
        shapes.append((geometry, value))
    burned: npt.NDArray[np.uint8] = rasterio.features.rasterize(
        shapes, out_shape=(PATCH, PATCH), transform=transform, fill=0, dtype="uint8"
    )
    return burned


def _mask(staging: Path, entry: ManifestEntry) -> npt.NDArray[np.uint8]:
    return np.asarray(Image.open(staging / entry["files"]["mask"]))


def _check_images(tmp_path: Path, config: MapcvConfig, manifest: Manifest) -> None:
    staging = config.writer.staging_dir
    for entry in manifest.patches:
        stem = Path(entry["files"]["before"]).name
        assert entry["files"] == {
            "before": f"A/{stem}",
            "after": f"B/{stem}",
            "mask": f"label/{stem}",
        }
        for name in ("before", "after"):
            want, _ = expected_on_patch_grid(tmp_path / f"{name}.tif", manifest, entry)
            np.testing.assert_array_equal(read_png(staging / entry["files"][name]), want)


# ── Config ───────────────────────────────────────────────────────────────────


XYZ = {"type": "xyz", "zoom": 18, "source": "esri_satellite"}
BASE: dict[str, Any] = {
    "task": "change",
    "region": {"west": 4.9, "south": 52.3, "east": 4.91, "north": 52.31},
    "imagery": [{**XYZ, "name": "before"}, {**XYZ, "name": "after"}],
    "sampler": {"patch_size": 256},
    "writer": {"staging_dir": "out"},
}


@pytest.mark.parametrize(
    ("changes", "message"),
    [
        ({"imagery": XYZ, "labels": {"path": "c.geojson"}}, "task: change needs two imagery"),
        (
            {"imagery": [{**XYZ, "name": n} for n in "abc"], "labels": {"path": "c.geojson"}},
            "task: change needs two imagery",
        ),
        ({}, "task: change needs labels that mark what changed"),
        ({"change": {"before": {"path": "a.geojson"}}}, "change.before and change.after come"),
        (
            {
                "labels": {"path": "c.geojson"},
                "change": {"before": {"path": "a.geojson"}, "after": {"path": "b.geojson"}},
            },
            "not both",
        ),
        (
            {
                "change": {
                    "before": {"path": "a.geojson", "label_field": "kind"},
                    "after": {"path": "b.geojson"},
                }
            },
            "the same classes mapping",
        ),
        (
            {
                "change": {
                    "before": {"path": "a.geojson", "label_field": "k", "classes": {"x": 1}},
                    "after": {"path": "b.geojson", "label_field": "k", "classes": {"x": 2}},
                }
            },
            "the same classes mapping",
        ),
        (
            {
                "change": {
                    "before": {"path": "a.geojson"},
                    "after": {"path": "b.geojson", "ignore_index": 99},
                }
            },
            "the same ignore_index",
        ),
        (
            {"labels": {"path": "c.geojson"}, "change": {"change_value": 255}},
            "change.change_value 255 is also the ignore value",
        ),
        (
            {
                "change": {
                    "before": {"type": "raster", "path": "a.tif"},
                    "after": {"path": "b.geojson"},
                }
            },
            "change.before must be vector labels",
        ),
    ],
)
def test_config_refuses_incomplete_change_settings(changes: dict[str, Any], message: str) -> None:
    with pytest.raises(ValidationError) as raised:
        MapcvConfig.model_validate({**BASE, **changes})
    assert message in str(raised.value)


def test_the_change_block_only_applies_to_task_change() -> None:
    with pytest.raises(ValidationError, match="the change block only applies to task: change"):
        MapcvConfig.model_validate(
            {**BASE, "task": "segmentation", "labels": {"path": "c.geojson"}, "change": {}}
        )


def test_label_set_paths_resolve_against_the_config_folder(tmp_path: Path) -> None:
    folder = tmp_path / "project"
    folder.mkdir()
    (folder / "mapcv.yaml").write_text(
        "task: change\n"
        "region: {west: 4.9, south: 52.3, east: 4.91, north: 52.31}\n"
        "imagery:\n"
        "  - {type: xyz, name: before, zoom: 18, source: esri_satellite}\n"
        "  - {type: xyz, name: after, zoom: 18, source: esri_satellite}\n"
        "change: {before: {path: sets/a.geojson}, after: {path: b.geojson}}\n"
        "sampler: {patch_size: 256}\n"
        "writer: {staging_dir: out}\n"
    )
    config = MapcvConfig.from_yaml(folder / "mapcv.yaml")
    options = config.change_options
    assert options.before is not None and options.after is not None
    assert options.before.path == folder / "sets" / "a.geojson"
    assert options.after.path == folder / "b.geojson"


# ── Datasets ─────────────────────────────────────────────────────────────────


def test_change_polygons_become_the_change_mask(tmp_path: Path, scene: dict[str, Any]) -> None:
    config = change_config(tmp_path, scene["region"], labels={"path": str(scene["changes"])})
    manifest = run_generate(config).manifest
    assert manifest.task == "change"
    assert manifest.writer is not None and manifest.writer["layout"] == "change"
    assert manifest.target is not None and manifest.target.class_map == {"change": 1}
    _check_images(tmp_path, config, manifest)
    changed = 0
    for entry in manifest.patches:
        want = _burn(scene["changes"], Affine(*manifest.patch_transform(entry)))
        mask = _mask(config.writer.staging_dir, entry)
        np.testing.assert_array_equal(mask, want, err_msg=f"{entry['row']},{entry['col']}")
        changed += int(want.any())
    assert changed > 0, "no patch shows a change: the comparison would prove nothing"
    names = (config.writer.staging_dir / "splits" / "train.txt").read_text().split()
    assert names and all((config.writer.staging_dir / "A" / name).exists() for name in names)


def test_before_and_after_label_sets_differ_where_objects_change(
    tmp_path: Path, scene: dict[str, Any]
) -> None:
    config = change_config(
        tmp_path,
        scene["region"],
        change={
            "before": {"path": str(scene["before_set"])},
            "after": {"path": str(scene["after_set"])},
        },
    )
    manifest = run_generate(config).manifest
    _check_images(tmp_path, config, manifest)
    target = manifest.target
    assert target is not None and target.labels is not None
    assert set(target.labels) == {"before", "after"}
    assert all("sha256" in target.labels[key] for key in ("before", "after"))
    changed = 0
    for entry in manifest.patches:
        transform = Affine(*manifest.patch_transform(entry))
        before = _burn(scene["before_set"], transform)
        after = _burn(scene["after_set"], transform)
        want = (before != after).astype(np.uint8)
        mask = _mask(config.writer.staging_dir, entry)
        np.testing.assert_array_equal(mask, want, err_msg=f"{entry['row']},{entry['col']}")
        changed += int(want.any())
    assert changed > 0


def test_classes_compare_object_kinds_too(tmp_path: Path, scene: dict[str, Any]) -> None:
    classes = {"house": 1, "shed": 2}
    label_set = {"label_field": "kind", "classes": classes, "ignore_index": None}
    config = change_config(
        tmp_path,
        scene["region"],
        change={
            "before": {"path": str(scene["before_set"]), **label_set},
            "after": {"path": str(scene["after_set"]), **label_set},
            "change_value": 255,
        },
        staging="classes",
    )
    manifest = run_generate(config).manifest
    relabeled = 0
    for entry in manifest.patches:
        transform = Affine(*manifest.patch_transform(entry))
        before = _burn(scene["before_set"], transform, "kind")
        after = _burn(scene["after_set"], transform, "kind")
        want = np.where(before != after, 255, 0).astype(np.uint8)
        np.testing.assert_array_equal(_mask(config.writer.staging_dir, entry), want)
        # The house that became a shed is change only because classes are compared.
        relabeled += int(((before == 1) & (after == 2)).any())
    assert relabeled > 0, "no patch shows the relabeled object"


def test_levir_style_masks_are_0_and_255(tmp_path: Path, scene: dict[str, Any]) -> None:
    config = change_config(
        tmp_path,
        scene["region"],
        labels={"path": str(scene["changes"]), "ignore_index": None},
        change={"change_value": 255},
    )
    manifest = run_generate(config).manifest
    values = set()
    for entry in manifest.patches:
        values |= set(np.unique(_mask(config.writer.staging_dir, entry)).tolist())
    assert values == {0, 255}
    assert manifest.target is not None and manifest.target.ignore_index is None


def test_a_change_raster_marks_change_and_keeps_its_ignore_pixels(
    tmp_path: Path, scene: dict[str, Any]
) -> None:
    # A change map on the image grid: 2 = change, 1 = no change, 0 = no data.
    ref = reference_transform()
    rng = np.random.default_rng(3)
    data = rng.choice(
        np.array([0, 1, 2], dtype=np.uint8), size=(1, HEIGHT, WIDTH), p=[0.1, 0.6, 0.3]
    )
    with rasterio.open(
        tmp_path / "change_map.tif",
        "w",
        driver="GTiff",
        height=HEIGHT,
        width=WIDTH,
        count=1,
        dtype="uint8",
        crs=f"EPSG:{EPSG}",
        transform=ref,
        nodata=0,
    ) as dst:
        dst.write(data)
    config = change_config(
        tmp_path,
        scene["region"],
        labels={"type": "raster", "path": str(tmp_path / "change_map.tif"), "classes": {2: 1}},
    )
    manifest = run_generate(config).manifest
    for entry in manifest.patches:
        _, inside = expected_on_patch_grid(tmp_path / "before.tif", manifest, entry)
        patch = Affine(*manifest.patch_transform(entry))
        col, row = (round(v) for v in ~ref * (patch.c, patch.f))
        source = data[0, row : row + PATCH, col : col + PATCH]
        want = np.where(source == 2, 1, 0).astype(np.uint8)
        want[source == 0] = 255  # the raster's no-data: no label, ignored
        np.testing.assert_array_equal(_mask(config.writer.staging_dir, entry), want)
        assert inside.all()


def test_pixels_missing_in_either_image_are_ignored(tmp_path: Path, scene: dict[str, Any]) -> None:
    # The after image covers only the top part of the before image.
    ref = reference_transform()
    write_raster(tmp_path / "after.tif", ref, WIDTH, HEIGHT // 2, seed=2)
    config = change_config(
        tmp_path, scene["region"], labels={"path": str(scene["changes"])}, staging="partial"
    )
    config.sampler.edge_strategy = "drop"
    with pytest.warns(UserWarning, match="region extends beyond the GeoTIFF 'after.tif'"):
        manifest = run_generate(config).manifest
    ignored = 0
    for entry in manifest.patches:
        _, inside = expected_on_patch_grid(tmp_path / "after.tif", manifest, entry)
        mask = _mask(config.writer.staging_dir, entry)
        assert (mask[~inside] == 255).all()
        ignored += int((~inside).any())
    assert ignored > 0


def test_resume_split_plan_and_info(tmp_path: Path, scene: dict[str, Any]) -> None:
    config = change_config(tmp_path, scene["region"], labels={"path": str(scene["changes"])})
    full = run_generate(config).manifest
    staging = config.writer.staging_dir
    files = {path: path.read_bytes() for path in staging.rglob("*.png")}
    cut = Manifest.load(staging / "manifest.json")
    cut.patches = cut.patches[: len(cut.patches) // 2]
    cut.save(staging / "manifest.json")
    assert run_generate(config).manifest.patches == full.patches
    assert {path: path.read_bytes() for path in staging.rglob("*.png")} == files

    counts = run_split(staging)
    assert sum(counts.values()) == len(full.patches)

    estimate = plan(config)
    assert estimate.task == "change"
    assert estimate.labels is not None and estimate.labels.polygons == 2
    result = runner.invoke(app, ["info", str(staging)], env={"COLUMNS": "200"})
    assert result.exit_code == 0, result.output
    assert "change" in result.output and "Source before" in result.output


def test_plan_reads_both_label_sets(tmp_path: Path, scene: dict[str, Any]) -> None:
    config = change_config(
        tmp_path,
        scene["region"],
        change={
            "before": {"path": str(scene["before_set"])},
            "after": {"path": str(scene["after_set"])},
        },
    )
    estimate = plan(config)
    assert estimate.labels is not None
    assert estimate.labels.polygons == 3
    assert "b23.geojson → " in estimate.labels.path


def test_change_template_is_a_valid_config() -> None:
    result = runner.invoke(app, ["init", "--template", "change", "--stdout"])
    assert result.exit_code == 0, result.output
    import yaml

    config = MapcvConfig.model_validate(yaml.safe_load(result.output))
    assert config.task == "change" and config.source_names == ["before", "after"]


# ── Smaller paths ────────────────────────────────────────────────────────────


def test_labels_without_features_give_an_unchanged_mask(
    tmp_path: Path, scene: dict[str, Any]
) -> None:
    empty = write_features(tmp_path / "none.geojson", [])
    config = change_config(
        tmp_path,
        scene["region"],
        change={"before": {"path": str(empty)}, "after": {"path": str(empty)}},
        staging="empty",
    )
    manifest = run_generate(config).manifest
    assert manifest.patches
    for entry in manifest.patches:
        assert not _mask(config.writer.staging_dir, entry).any()


def test_change_targets_and_writers_need_their_settings(tmp_path: Path) -> None:
    from mapcv.config import ChangeOptions
    from mapcv.targets.change import ChangeTarget
    from mapcv.writer import WriterConfig
    from mapcv.writers import ChangeWriter, create_writer

    options = ChangeOptions(change_value=3)
    target = ChangeTarget(options, _labels_config(tmp_path))
    assert target.options is options and target.class_map == {"change": 3}
    config = WriterConfig(staging_dir=tmp_path / "out")
    with pytest.raises(ValueError, match="needs two imagery sources"):
        create_writer(config, target)
    with pytest.raises(ValueError, match="two imagery sources"):
        ChangeWriter(config, ["only"])
    assert create_writer(config, target, ["a", "b"]).layout == "change"


def _labels_config(tmp_path: Path) -> Any:
    from mapcv.config import LabelsConfig

    return LabelsConfig(path=tmp_path / "c.geojson")


def test_cli_and_plan_describe_change_datasets(tmp_path: Path, scene: dict[str, Any]) -> None:
    import yaml

    data = {
        "task": "change",
        "region": scene["region"],
        "imagery": _sources(tmp_path),
        "change": {
            "before": {"path": str(scene["before_set"])},
            "after": {"path": str(tmp_path / "missing.geojson")},
        },
        "sampler": {"patch_size": PATCH},
        "writer": {"staging_dir": str(tmp_path / "cli")},
    }
    path = tmp_path / "change.yaml"
    path.write_text(yaml.safe_dump(data))
    result = runner.invoke(app, ["validate", str(path)], env={"COLUMNS": "200"})
    assert result.exit_code == 0, result.output
    assert "change · before → after · before/after label sets" in result.output
    assert f"before: {scene['before_set']}" in result.output
    assert "change.after.path not found" in result.output

    # Labels that miss the region: the plan says no patch shows a change.
    far = write_features(
        tmp_path / "far.geojson", [([[10.0, 50.0], [10.1, 50.0], [10.1, 50.1], [10.0, 50.0]], None)]
    )
    config = change_config(tmp_path, scene["region"], labels={"path": str(far)})
    estimate = plan(config)
    assert any("no patch would show a change" in message for message in estimate.warnings)


def test_mcp_sandbox_checks_both_label_sets(tmp_path: Path, scene: dict[str, Any]) -> None:
    from mapcv.agent_tools import config_paths

    config = change_config(
        tmp_path,
        scene["region"],
        change={
            "before": {"path": str(scene["before_set"])},
            "after": {"path": "/etc/passwd.geojson"},
        },
    )
    keys = dict(config_paths(config))
    assert keys["change.before.path"] == scene["before_set"]
    assert keys["change.after.path"] == Path("/etc/passwd.geojson")
