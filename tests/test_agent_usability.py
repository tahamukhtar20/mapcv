"""What the MCP tools tell an agent before it runs a job: validation, schema size, warnings."""

from __future__ import annotations

import json
import shutil
import threading
import warnings
from pathlib import Path

import pytest

from mapcv.agent_tools import (
    Sandbox,
    ToolFailure,
    ToolState,
    capture_warnings,
    describe_config_schema,
    plan,
    prepare_generate,
    validate_config,
    write_config,
)
from mapcv.config import MapcvConfig

_VECTOR = Path(__file__).parent / "data" / "vector"

BASE = """\
region: {west: 16.3, south: 48.1, east: 16.4, north: 48.3}
imagery: {type: xyz, zoom: 16, source: esri_satellite}
sampler: {patch_size: 256}
writer: {staging_dir: out}
"""


def _state(root: Path, allow_write: bool = False) -> ToolState:
    return ToolState(Sandbox(root, allow_write))


def _errors(result: object) -> dict[str, str]:
    data = result.data  # type: ignore[attr-defined]
    return {error["field"]: error["message"] for error in data["errors"]}


# ── F3: labels.osm ───────────────────────────────────────────────────────────

OSM = BASE + "labels:\n  osm:\n    classes:\n      - {name: building, tags: {building: '*'}}\n"


def test_osm_labels_are_flagged_when_validating_and_writing(tmp_path: Path) -> None:
    state = _state(tmp_path, allow_write=True)
    checked = validate_config(state, None, OSM)
    assert checked.data["valid"] is True
    (warning,) = checked.data["warnings"]
    assert "labels.osm" in warning and "mapcv plan" in warning and "labels.path" in warning
    assert warning in checked.summary
    written = write_config(state, "osm.yaml", OSM)
    assert written.data["warnings"] == [warning]
    assert "Warnings" in written.summary
    # Its description says so too, before any config is written.
    notes = " ".join(describe_config_schema(state).data["notes"])
    assert "labels.osm" in notes and "refuse" in notes


# ── F8, F14: one pass, missing files, label_field ────────────────────────────


def test_a_typo_does_not_hide_the_other_errors(tmp_path: Path) -> None:
    state = _state(tmp_path)
    text = BASE.replace(
        "sampler: {patch_size: 256}", "sampler: {patch_size: 256, edge_stratgy: pad}"
    ).replace("staging_dir: out", "staging_dir: out, image_format: npy")
    errors = _errors(validate_config(state, None, text))
    assert "Did you mean 'edge_strategy'" in errors["sampler.edge_stratgy"]
    # Found behind the typo: npy is not an XYZ format.
    assert any("image_format" in message for message in errors.values())
    both = text.replace("source: esri_satellite", "sorce: esri_satellite")
    errors = _errors(validate_config(state, None, both))
    assert "Did you mean 'source'" in errors["imagery.sorce"]
    assert "Did you mean 'edge_strategy'" in errors["sampler.edge_stratgy"]


def test_nested_and_list_typos_name_the_closest_key(tmp_path: Path) -> None:
    state = _state(tmp_path)
    text = BASE + "labels:\n  files:\n    - {path: a.geojson, clas: road}\n"
    errors = _errors(validate_config(state, None, text))
    assert "Did you mean 'class'" in errors["labels.files.0.clas"]


def test_a_missing_label_file_is_an_error_and_write_config_says_so(tmp_path: Path) -> None:
    state = _state(tmp_path, allow_write=True)
    text = BASE + "labels: {path: missing.geojson}\n"
    checked = validate_config(state, None, text)
    assert checked.data["valid"] is False
    assert "file not found: missing.geojson" in _errors(checked)["labels.path"]
    written = write_config(state, "c.yaml", text)  # saved as a draft, with the problem named
    assert (tmp_path / "c.yaml").exists()
    assert written.data["valid"] is False and "missing.geojson" in written.summary
    shutil.copy(_VECTOR / "labels.geojson", tmp_path / "missing.geojson")
    assert validate_config(state, "c.yaml").data["valid"] is True


@pytest.mark.parametrize("name", ["labels.geojson", "labels.gpkg", "polygons.shp"])
def test_a_misspelt_label_field_gets_a_suggestion(tmp_path: Path, name: str) -> None:
    for file in _VECTOR.glob(Path(name).stem + ".*"):
        shutil.copy(file, tmp_path / file.name)
    state = _state(tmp_path)
    good = BASE + f"labels: {{path: {name}, label_field: class}}\n"
    assert validate_config(state, None, good).data["valid"] is True
    bad = _errors(
        validate_config(state, None, good.replace("label_field: class", "label_field: clas"))
    )
    message = bad["labels.label_field"]
    assert "'clas' is not an attribute" in message and "Did you mean 'class'" in message
    assert "class" in message.split("it has:")[1]
    in_files = BASE + f"labels:\n  files:\n    - {{path: {name}, label_field: Class}}\n"
    assert (
        "Did you mean 'class'"
        in _errors(validate_config(state, None, in_files))["labels.files[0].label_field"]
    )


def test_label_fields_of_a_change_label_set_are_checked(tmp_path: Path) -> None:
    shutil.copy(_VECTOR / "labels.geojson", tmp_path / "b.geojson")
    text = BASE.replace(
        "imagery: {type: xyz, zoom: 16, source: esri_satellite}",
        "task: change\nimagery:\n"
        "  - {type: xyz, name: a, zoom: 16, source: esri_satellite}\n"
        "  - {type: xyz, name: b, zoom: 16, source: esri_satellite}\n"
        "change:\n  before: {path: b.geojson, label_field: clas}\n  after: {path: b.geojson, label_field: class}",
    )
    errors = _errors(validate_config(_state(tmp_path), None, text))
    assert "Did you mean 'class'" in errors["change.before.label_field"]


# ── F21: stack_sources ───────────────────────────────────────────────────────

STACKED = """\
region: {west: 4.0, south: 52.0, east: 4.004, north: 52.003}
imagery:
  - {type: xyz, name: rgb, zoom: 18, source: esri_satellite}
  - {type: stac_cog, name: s2, search: {datetime: 2025-06-01/2025-06-30}, bands: [red, nir]}
sampler: {patch_size: 64}
writer: {staging_dir: out, image_format: tif, stack_sources: true}
"""


def test_stack_sources_mismatch_is_reported_before_generate(tmp_path: Path) -> None:
    state = _state(tmp_path, allow_write=True)
    errors = _errors(validate_config(state, None, STACKED))
    assert "'rgb' has 3 band(s), 's2' 2" in errors["writer.stack_sources"]
    ok = STACKED.replace("bands: [red, nir]", "bands: [red, green, blue]")
    # Same band count, different data types (uint8 tiles, uint16 COGs).
    assert "writer.stack_sources" in _errors(validate_config(state, None, ok))


def _geotiff(path: Path, count: int) -> None:
    import numpy as np

    rasterio = pytest.importorskip("rasterio")
    from rasterio.transform import from_origin

    data = np.full((count, 48, 48), 7, dtype="uint8")
    with rasterio.open(
        path,
        "w",
        driver="GTiff",
        width=48,
        height=48,
        count=count,
        dtype="uint8",
        crs="EPSG:4326",
        transform=from_origin(4.0, 52.001, 0.00002, 0.00002),
    ) as dataset:
        dataset.write(data)


def test_stack_sources_mismatch_of_geotiffs_is_reported_by_plan_and_generate(
    tmp_path: Path,
) -> None:
    _geotiff(tmp_path / "rgb.tif", 3)
    _geotiff(tmp_path / "one.tif", 1)
    text = """\
region: {west: 4.0, south: 52.0, east: 4.00096, north: 52.00096}
imagery:
  - {type: geotiff, name: rgb, path: rgb.tif}
  - {type: geotiff, name: one, path: one.tif}
sampler: {patch_size: 16, edge_strategy: drop}
writer: {staging_dir: out, image_format: tif, stack_sources: true}
"""
    state = _state(tmp_path, allow_write=True)
    # Only the files say how many bands there are: validation passes, plan names the problem.
    assert validate_config(state, None, text).data["valid"] is True
    planned = plan(state, None, text).data
    message = "writer.stack_sources needs every source to have the same bands and data type"
    assert message in planned["blocking"][0]
    assert any(message in warning for warning in planned["warnings"])
    assert "'rgb' has 3 band(s) of uint8, 'one' 1 of uint8" in planned["blocking"][0]
    (tmp_path / "c.yaml").write_text(text)
    with pytest.raises(ToolFailure, match="Nothing was started") as raised:
        prepare_generate(state, "c.yaml")
    assert message in raised.value.message
    assert not (tmp_path / "out").exists()
    matching = text.replace("path: one.tif", "path: rgb.tif")
    assert plan(state, None, matching).data["blocking"] == []


# ── F27: URLs as labels ──────────────────────────────────────────────────────


@pytest.mark.parametrize("url", ["file:///data/b.geojson", "https://example.com/b.geojson"])
def test_labels_must_be_local_files(tmp_path: Path, url: str) -> None:
    text = BASE + f"labels: {{path: '{url}'}}\n"
    errors = _errors(validate_config(_state(tmp_path), None, text))
    assert "not a URL" in errors["labels.path"] and "imagery.path" in errors["labels.path"]
    files = BASE + f"labels:\n  files:\n    - {{path: '{url}', class: road}}\n"
    assert (
        "not a URL"
        in _errors(validate_config(_state(tmp_path), None, files))["labels.files.0.path"]
    )


def test_local_label_paths_that_look_odd_are_still_accepted(tmp_path: Path) -> None:
    (tmp_path / "my dir").mkdir()
    shutil.copy(_VECTOR / "labels.geojson", tmp_path / "my dir" / "l 1.geojson")
    text = BASE + "labels: {path: 'my dir/l 1.geojson'}\n"
    assert validate_config(_state(tmp_path), None, text).data["valid"] is True


# ── F16: a schema description an agent can afford ────────────────────────────


def test_the_schema_description_is_short_and_narrows(tmp_path: Path) -> None:
    state = _state(tmp_path)
    default = describe_config_schema(state)
    size = len(json.dumps(default.data))
    assert "schema" not in default.data and size < 20_000
    full = describe_config_schema(state, full_schema=True)
    assert len(json.dumps(full.data)) > 2 * size and full.data["schema"]["properties"]["labels"]
    # Every top-level key of the config is described with its fields.
    fields = default.data["models"]["MapcvConfig"]["fields"]
    assert set(MapcvConfig.model_fields) <= set(fields)
    narrow = describe_config_schema(state, section="labels").data
    assert {"LabelsConfig", "LabelFile", "OsmLabelsSource"} <= set(narrow["models"])
    assert "WriterConfig" not in narrow["models"]
    assert "LabelFile" in describe_config_schema(state, section="LabelFile").data["models"]
    with pytest.raises(ToolFailure, match="Unknown section 'nope'"):
        describe_config_schema(state, section="nope")
    # relative_paths used to hold the docstring of `from_yaml`.
    paths = default.data["rules"]["relative_paths"]
    assert "Load and validate" not in paths and "relative to that file's folder" in paths
    assert any("class: <name>" in note for note in default.data["notes"])


# ── F7: captures of warnings that overlap ────────────────────────────────────


def test_overlapping_warning_captures_keep_their_own_warnings() -> None:
    inside = threading.Event()
    release = threading.Event()
    seen: dict[str, list[str]] = {}

    def slow() -> None:
        with capture_warnings(broad=True) as caught:
            warnings.warn("from the generation", UserWarning, stacklevel=1)
            inside.set()
            release.wait(10)
            warnings.warn("late, from the generation", UserWarning, stacklevel=1)
        seen["slow"] = [str(w.message) for w in caught]

    worker = threading.Thread(target=slow)
    worker.start()
    assert inside.wait(10)
    with capture_warnings() as caught:  # does not wait for the first capture
        warnings.warn("from the plan", UserWarning, stacklevel=1)
    assert [str(w.message) for w in caught] == ["from the plan"]
    release.set()
    worker.join(10)
    assert seen["slow"] == ["from the generation", "late, from the generation"]
    # The process-wide warning state is back to normal after the last capture.
    with warnings.catch_warnings(record=True) as after:
        warnings.simplefilter("always")
        warnings.warn("outside", UserWarning, stacklevel=1)
    assert [str(w.message) for w in after] == ["outside"]
