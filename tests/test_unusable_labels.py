"""A label file whose every feature is skipped is an error, not a dataset without labels.

The case that prompted it: a GeoJSON in UTM metres (it has no ``crs`` member, and RFC 7946
says it is lon/lat) used to give a dataset without masks while ``plan`` and ``generate``
exited 0. A file with no feature at all, and a file of which only some features are
skipped, still warn and go on.
"""

from __future__ import annotations

import json
import warnings
from pathlib import Path
from typing import Any

import pytest
import yaml

pytest.importorskip("rasterio", reason="the raster is written with rasterio")
from test_geotiff_imagery import Raster, make_raster
from typer.testing import CliRunner

from mapcv.cli import app
from mapcv.config import MapcvConfig
from mapcv.labels import NoUsableLabelsError, collect_unusable_files, parse_geojson
from mapcv.planning import plan
from mapcv.targets.segmentation import load_labels

runner = CliRunner()


def _square(x: float, y: float, size: float = 30.0) -> list[list[list[float]]]:
    return [[[x, y], [x + size, y], [x + size, y + size], [x, y + size], [x, y]]]


def _collection(features: list[dict[str, Any]]) -> str:
    return json.dumps({"type": "FeatureCollection", "features": features})


def _feature(rings: list[list[list[float]]], label: str | None = "building") -> dict[str, Any]:
    return {
        "type": "Feature",
        "properties": {"class": label},
        "geometry": {"type": "Polygon", "coordinates": rings},
    }


def _projected_labels(raster: Raster) -> str:
    """20 squares of 30 m in the raster's UTM coordinates (a GeoJSON without a CRS)."""
    x0, y0 = raster.transform.c, raster.transform.f
    return _collection([_feature(_square(x0 + 100 + 40 * i, y0 - 300 - 10 * i)) for i in range(20)])


def _config(tmp_path: Path, raster: Raster, labels: Path, **label_keys: Any) -> Path:
    west, south, east, north = raster.lonlat_bounds
    x, y = (west + east) / 2, (south + north) / 2
    data = {
        "region": {"west": x - 0.001, "south": y - 0.001, "east": x + 0.001, "north": y + 0.001},
        "imagery": {"type": "geotiff", "path": str(raster.path)},
        "labels": {"type": "vector", "path": str(labels), "label_field": "class", **label_keys},
        "sampler": {"patch_size": 256},
        "writer": {"staging_dir": str(tmp_path / "out")},
    }
    path = tmp_path / "mapcv.yaml"
    path.write_text(yaml.safe_dump(data))
    return path


@pytest.fixture
def raster(tmp_path: Path) -> Raster:
    return make_raster(tmp_path, width=1024, height=1024)


def _oneline(text: str) -> str:
    return " ".join(text.split())


def test_generate_fails_when_every_feature_has_projected_coordinates(
    tmp_path: Path, raster: Raster
) -> None:
    labels = tmp_path / "labels.geojson"
    labels.write_text(_projected_labels(raster))
    config = _config(tmp_path, raster, labels)

    result = runner.invoke(app, ["generate", str(config), "--yes"])

    output = _oneline(result.output)
    assert result.exit_code == 1, output
    assert "every feature in the label file was skipped" in output
    assert "20 with invalid coordinates (beyond" in output
    assert "look like projected coordinates" in output
    assert "WGS-84 longitude/latitude (RFC 7946)" in output
    assert "GeoPackage, Shapefile or GeoParquet" in output
    assert not (tmp_path / "out" / "Masks").exists()


def test_plan_fails_the_same_way(tmp_path: Path, raster: Raster) -> None:
    labels = tmp_path / "labels.geojson"
    labels.write_text(_projected_labels(raster))
    result = runner.invoke(app, ["plan", str(_config(tmp_path, raster, labels))])
    output = _oneline(result.output)
    assert result.exit_code == 1, output
    assert "look like projected coordinates" in output


def test_validate_still_checks_only_the_config(tmp_path: Path, raster: Raster) -> None:
    labels = tmp_path / "labels.geojson"
    labels.write_text(_projected_labels(raster))
    result = runner.invoke(app, ["validate", str(_config(tmp_path, raster, labels))])
    assert result.exit_code == 0, result.output


def test_planning_raises_for_all_skipped_but_warns_for_zero_features(
    tmp_path: Path, raster: Raster
) -> None:
    skipped = tmp_path / "skipped.geojson"
    skipped.write_text(_projected_labels(raster))
    config = MapcvConfig.model_validate(
        yaml.safe_load(_config(tmp_path, raster, skipped).read_text())
    )
    with pytest.raises(NoUsableLabelsError, match="projected coordinates"):
        plan(config)

    empty = tmp_path / "empty.geojson"
    empty.write_text(_collection([]))
    config = MapcvConfig.model_validate(
        yaml.safe_load(_config(tmp_path, raster, empty).read_text())
    )
    estimate = plan(config)
    assert any("no usable label polygon" in w for w in estimate.warnings)


def test_a_file_with_no_feature_still_generates_a_background_dataset(
    tmp_path: Path, raster: Raster
) -> None:
    labels = tmp_path / "empty.geojson"
    labels.write_text(_collection([]))
    result = runner.invoke(app, ["generate", str(_config(tmp_path, raster, labels)), "--yes"])
    output = _oneline(result.output)
    assert result.exit_code == 0, output
    assert output.count("no usable label polygon") == 1
    assert (tmp_path / "out" / "manifest.json").is_file()


def test_partial_skips_only_warn(tmp_path: Path, raster: Raster) -> None:
    good = _feature([[[lon, lat] for lon, lat in _lonlat_square(raster)]])
    labels = tmp_path / "mixed.geojson"
    labels.write_text(_collection([good, *json.loads(_projected_labels(raster))["features"]]))
    result = runner.invoke(app, ["plan", str(_config(tmp_path, raster, labels))])
    output = _oneline(result.output)
    assert result.exit_code == 0, output
    assert "skipped 20 with invalid coordinates (beyond" in output


def _lonlat_square(raster: Raster) -> list[tuple[float, float]]:
    west, south, east, north = raster.lonlat_bounds
    x, y = (west + east) / 2, (south + north) / 2
    d = 0.0002
    return [(x, y), (x + d, y), (x + d, y + d), (x, y + d), (x, y)]


def test_a_file_of_points_only_still_warns(tmp_path: Path, raster: Raster) -> None:
    """Skipped for their kind is not skipped for broken coordinates: it may be intended."""
    points = [
        {
            "type": "Feature",
            "properties": {"class": "tree"},
            "geometry": {"type": "Point", "coordinates": [3.0, 48.85]},
        }
    ]
    labels = tmp_path / "points.geojson"
    labels.write_text(_collection(points))
    config = MapcvConfig.model_validate(
        yaml.safe_load(_config(tmp_path, raster, labels).read_text())
    )
    with pytest.warns(UserWarning, match="without polygon geometry"):
        assert load_labels(config.labels)[0] == []  # type: ignore[arg-type]


def test_features_without_a_geometry_are_not_counted_as_skipped(tmp_path: Path) -> None:
    labels = tmp_path / "nulls.geojson"
    labels.write_text(
        _collection([{"type": "Feature", "properties": {"class": "a"}, "geometry": None}])
    )
    with collect_unusable_files() as unusable, warnings.catch_warnings():
        warnings.simplefilter("ignore")
        assert parse_geojson(labels.read_bytes(), "class") == ([], {})
    assert unusable == []


def test_parse_geojson_alone_still_returns_an_empty_list_with_a_warning() -> None:
    data = _collection([_feature(_square(448000.0, 5412000.0))]).encode()
    with pytest.warns(UserWarning, match="projected coordinates"):
        assert parse_geojson(data, "class") == ([], {})


def test_a_non_finite_coordinate_keeps_the_plain_wording() -> None:
    data = (
        b'{"type":"Feature","properties":{},"geometry":{"type":"Polygon",'
        b'"coordinates":[[[0,0],[1,0],[NaN,1],[0,0]]]}}'
    )
    with pytest.warns(UserWarning, match="1 with invalid coordinates") as caught:
        parse_geojson(data)
    assert "projected" not in str(caught[0].message)


def test_the_mcp_plan_and_generate_tools_fail_too(tmp_path: Path, raster: Raster) -> None:
    from mapcv.agent_tools import Sandbox, ToolFailure, ToolState, prepare_generate
    from mapcv.agent_tools import plan as plan_tool

    labels = tmp_path / "labels.geojson"
    labels.write_text(_projected_labels(raster))
    _config(tmp_path, raster, labels)
    state = ToolState(Sandbox(tmp_path, True))
    with pytest.raises(ToolFailure, match="projected coordinates"):
        plan_tool(state, "mapcv.yaml")
    with pytest.raises(ToolFailure, match="projected coordinates"):
        prepare_generate(state, "mapcv.yaml")
