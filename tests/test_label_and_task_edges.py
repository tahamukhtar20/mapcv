"""Edge paths of the label forms and tasks added in 0.3: defensive checks, messages,
CLI and MCP summaries, and small helpers. The datasets themselves are tested against
rasterio in test_regression, test_label_files, test_label_area_buffer and test_aoi."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pytest
import yaml
from pydantic import ValidationError
from shapely.geometry import LineString, box
from typer.testing import CliRunner

pytest.importorskip("rasterio", reason="these tests write rasters with rasterio")

from mapcv.cli import app
from mapcv.config import (
    ContinuousLabelsConfig,
    LabelsConfig,
    MapcvConfig,
    RegionConfig,
)
from mapcv.labels import load_vector_labels
from mapcv.targets import create_target
from mapcv.targets.regression import RegressionTarget, ValueWindow

runner = CliRunner()
ENV = {"COLUMNS": "220"}
XYZ = {"type": "xyz", "zoom": 15, "source": "esri_satellite"}
REGION = {"west": 4.9, "south": 52.3, "east": 4.91, "north": 52.31}


def _geojson(path: Path, *geometries: Any, props: dict[str, Any] | None = None) -> Path:
    features = [
        {"type": "Feature", "properties": dict(props or {}), "geometry": g.__geo_interface__}
        for g in geometries
    ]
    path.write_text(json.dumps({"type": "FeatureCollection", "features": features}))
    return path


def _base(tmp_path: Path, **extra: Any) -> dict[str, Any]:
    return {
        "region": REGION,
        "imagery": XYZ,
        "sampler": {"patch_size": 256},
        "writer": {"staging_dir": str(tmp_path / "out")},
        **extra,
    }


# ── Targets refuse labels the config would refuse ────────────────────────────


@pytest.mark.parametrize(
    ("task", "labels", "message"),
    [
        ("detection", None, "needs labels"),
        ("detection", {"type": "raster", "path": "a.tif", "classes": {1: 1}}, "vector labels"),
        ("instance", None, "needs labels"),
        ("instance", {"type": "continuous", "path": "a.tif"}, "vector labels"),
        ("classification", None, "needs labels"),
        ("classification", {"type": "continuous", "path": "a.tif"}, "class labels"),
        ("change", {"type": "continuous", "path": "a.tif"}, "classified label raster"),
        ("regression", {"path": "a.geojson"}, "labels.type: continuous"),
        ("segmentation", {"type": "continuous", "path": "a.tif"}, "values need task: regression"),
    ],
)
def test_create_target_refuses_what_the_config_refuses(
    tmp_path: Path, task: str, labels: Any, message: str
) -> None:
    valid = MapcvConfig.model_validate(_base(tmp_path, labels={"path": "x.geojson"}))
    parsed = None
    if labels is not None:
        parsed = MapcvConfig.model_validate(
            _base(
                tmp_path,
                task="regression" if labels.get("type") == "continuous" else "segmentation",
                labels=labels,
            )
        ).labels
    config = valid.model_copy(update={"task": task, "labels": parsed})
    with pytest.raises(ValueError, match=message):
        create_target(config)


# ── Regression internals ─────────────────────────────────────────────────────


def test_regression_target_needs_prepare_and_has_no_classes() -> None:
    target = RegressionTarget(ContinuousLabelsConfig(type="continuous", path="a.tif"))
    assert target.class_map == {}
    with pytest.raises(RuntimeError, match="prepare"):
        _ = target.sampler
    with pytest.raises(RuntimeError, match="prepare"):
        target.record()


def test_value_windows_without_a_validity_mask_and_with_nothing_kept() -> None:
    window = ValueWindow(np.arange(16, dtype=np.float32).reshape(4, 4))
    patch = window.annotate(2, 2, 4, "zero", None)
    assert np.isnan(patch[2:, :]).all() and patch[0, 0] == 10.0
    assert window.accepts(patch, 0.0) and not window.accepts(patch, 0.5)
    assert window.collate([], 4).shape == (0, 4, 4)


def test_values_below_valid_min_and_nan_nodata_have_no_target(tmp_path: Path) -> None:
    import rasterio
    from rasterio.transform import from_origin

    from mapcv.targets.raster_labels import ValueRasterSampler

    data = np.array([[1.0, 5.0], [np.nan, 9.0]], dtype=np.float32)
    path = tmp_path / "v.tif"
    with rasterio.open(
        path,
        "w",
        driver="GTiff",
        height=2,
        width=2,
        count=1,
        dtype="float32",
        crs="EPSG:32631",
        transform=from_origin(500000, 5400000, 1, 1),
        nodata=np.nan,
    ) as dst:
        dst.write(data, 1)
    labels = ContinuousLabelsConfig(type="continuous", path=str(path), valid_min=2.0)
    sampler = ValueRasterSampler(labels, "EPSG:32631")
    values = sampler.sample((1.0, 0.0, 500000.0, 0.0, -1.0, 5400000.0), 2, 2)
    np.testing.assert_array_equal(np.isnan(values), [[True, False], [True, False]])


# ── Config messages ──────────────────────────────────────────────────────────


@pytest.mark.parametrize(
    ("labels", "message"),
    [
        ({"type": "continuous", "path": "a.tif", "offset": float("inf")}, "labels.offset"),
        ({"type": "continuous", "path": "a.tif", "valid_min": float("inf")}, "labels.valid_min"),
        ({"path": "a.geojson", "buffer": {"line": float("inf")}}, "finite number of metres"),
        ({"files": [{"path": "a.geojson", "class": "  "}]}, "non-empty name"),
    ],
)
def test_more_config_refusals(tmp_path: Path, labels: dict[str, Any], message: str) -> None:
    task = "regression" if labels.get("type") == "continuous" else "segmentation"
    with pytest.raises(ValidationError) as raised:
        MapcvConfig.model_validate(_base(tmp_path, task=task, labels=labels))
    assert message in str(raised.value)


def test_region_files_without_polygons_or_with_too_many_names(tmp_path: Path) -> None:
    lines = _geojson(tmp_path / "lines.geojson", LineString([(4.9, 52.3), (4.91, 52.31)]))
    with pytest.raises(ValidationError, match="holds no polygon"):
        RegionConfig.model_validate({"path": str(lines)})
    many = {
        "type": "FeatureCollection",
        "features": [
            {
                "type": "Feature",
                "properties": {"site": f"s{index}"},
                "geometry": box(
                    4.9 + index * 1e-4, 52.3, 4.90005 + index * 1e-4, 52.30005
                ).__geo_interface__,
            }
            for index in range(300)
        ],
    }
    path = tmp_path / "many.geojson"
    path.write_text(json.dumps(many))
    with pytest.raises(ValidationError, match="more than 255 distinct names"):
        RegionConfig.model_validate({"path": str(path), "name_field": "site"})
    labels = LabelsConfig.model_validate({"files": [{"path": "a.geojson", "class": "a"}]})
    assert [file.path for file in labels.label_files] == [Path("a.geojson")]


def test_kml_cannot_be_buffered(tmp_path: Path) -> None:
    kml = tmp_path / "a.kml"
    kml.write_text('<?xml version="1.0"?><kml xmlns="http://www.opengis.net/kml/2.2"/>')
    with pytest.raises(ValueError, match="does not read from KML"):
        load_vector_labels(kml, buffer=(3.0, None))


# ── CLI and MCP summaries ────────────────────────────────────────────────────


def test_validate_shows_values_files_and_missing_files(tmp_path: Path) -> None:
    values = _base(
        tmp_path,
        task="regression",
        labels={"type": "continuous", "path": str(tmp_path / "missing.tif"), "scale": 0.1},
    )
    values["writer"]["image_format"] = "tif"
    path = tmp_path / "values.yaml"
    path.write_text(yaml.safe_dump(values))
    result = runner.invoke(app, ["validate", str(path)], env=ENV)
    assert result.exit_code == 0, result.output
    assert "values of band 1 · × 0.1 + 0" in result.output
    assert "labels.path not found" in result.output

    a = _geojson(tmp_path / "a.geojson", box(4.9, 52.3, 4.905, 52.305), props={"kind": "x"})
    files = _base(
        tmp_path,
        labels={
            "files": [
                {"path": str(a), "label_field": "kind"},
                {"path": str(tmp_path / "gone.gpkg"), "layer": "roads", "class": "road"},
            ],
            "annotated_area": str(tmp_path / "area.geojson"),
        },
    )
    path.write_text(yaml.safe_dump(files))
    result = runner.invoke(app, ["validate", str(path)], env=ENV)
    assert result.exit_code == 0, result.output
    assert "field: kind" in result.output and "layer: roads · class: road" in result.output
    assert "labels.files[1].path not found" in result.output
    assert "labels.annotated_area not found" in result.output

    change = _base(
        tmp_path,
        task="change",
        imagery=[{**XYZ, "name": "before"}, {**XYZ, "name": "after"}],
        change={
            "before": {"files": [{"path": str(a), "class": "house"}]},
            "after": {"files": [{"path": str(tmp_path / "later.geojson"), "class": "house"}]},
        },
    )
    path.write_text(yaml.safe_dump(change))
    result = runner.invoke(app, ["validate", str(path)], env=ENV)
    assert result.exit_code == 0, result.output
    assert "change.after.files[0].path not found" in result.output


def test_mcp_summaries_and_warnings_for_new_label_forms(tmp_path: Path) -> None:
    from mapcv.agent_tools import Sandbox, ToolState, validate_config

    state = ToolState(Sandbox(tmp_path))
    a = _geojson(tmp_path / "a.geojson", box(4.9, 52.3, 4.905, 52.305))
    configs = {
        "values.yaml": {
            **_base(tmp_path, task="regression", labels={"type": "continuous", "path": "chm.tif"}),
        },
        "files.yaml": _base(
            tmp_path,
            labels={
                "files": [{"path": "a.geojson", "class": "a"}, {"path": "b.geojson", "class": "b"}],
                "annotated_area": "area.geojson",
            },
        ),
        "change.yaml": _base(
            tmp_path,
            task="change",
            imagery=[{**XYZ, "name": "before"}, {**XYZ, "name": "after"}],
            change={"before": {"path": "a.geojson"}, "after": {"path": "c.geojson"}},
        ),
    }
    configs["values.yaml"]["writer"]["image_format"] = "tif"
    for name, data in configs.items():
        (tmp_path / name).write_text(yaml.safe_dump(data))
    del a
    values = validate_config(state, "values.yaml").data
    assert values["summary"]["labels"]["type"] == "continuous"
    files = validate_config(state, "files.yaml").data
    assert [f["class"] for f in files["summary"]["labels"]["files"]] == ["a", "b"]
    assert any("labels.files[1].path not found" in text for text in files["warnings"])
    assert any("labels.annotated_area not found" in text for text in files["warnings"])
    change = validate_config(state, "change.yaml").data
    assert any("change.after.path not found" in text for text in change["warnings"])


def test_aoi_helpers(tmp_path: Path) -> None:
    from mapcv.aoi import AreaOfInterest, _to_pixels

    path = _geojson(tmp_path / "aoi.geojson", box(4.9, 52.3, 4.905, 52.305))
    region = RegionConfig.model_validate({"path": str(path), "layer": None})
    with pytest.raises(ValueError, match="not invertible"):
        _to_pixels(np.array([box(0, 0, 1, 1)], dtype=object), (0.0, 0.0, 0.0, 0.0, 0.0, 0.0))
    with pytest.raises(ValueError, match="needs region.path"):
        AreaOfInterest(RegionConfig.model_validate(REGION), "EPSG:3857", (1, 0, 0, 0, -1, 0))
    # Web Mercator imagery (XYZ) takes the vectorized projection.
    aoi = AreaOfInterest(region, "EPSG:3857", (1.0, 0.0, 545000.0, 0.0, -1.0, 6870000.0))
    assert aoi.keep([], 32) == []
    record = aoi.record()
    assert set(record) == {"aoi_sha256"}
