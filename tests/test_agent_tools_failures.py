"""Failure paths of the agent tools: what the agent is told when something goes wrong."""

from __future__ import annotations

import json
from collections.abc import Callable
from pathlib import Path
from threading import Event

import pytest

from mapcv.agent_tools import (
    Sandbox,
    ToolFailure,
    ToolResult,
    ToolState,
    execute_generate,
    info,
    inspect_labels,
    plan,
    prepare_generate,
    split,
    validate_config,
    write_config,
)
from mapcv.manifest import (
    Manifest,
    ManifestEntry,
    ManifestMismatchError,
    PatchSummary,
    SourceRecord,
    TargetRecord,
)
from mapcv.pipeline import GenerateResult

CONFIG = """\
region: {west: 4.0, south: 52.0, east: 4.02, north: 52.02}
imagery: {type: xyz, zoom: 16, source: esri_satellite}
sampler: {patch_size: 256}
writer: {staging_dir: out}
"""


def _state(root: Path, allow_write: bool = False) -> ToolState:
    return ToolState(Sandbox(root, allow_write))


def test_config_loading_failures(tmp_path: Path) -> None:
    state = _state(tmp_path)
    huge = "note: " + "a" * (1024 * 1024)
    with pytest.raises(ToolFailure, match="larger"):
        validate_config(state, None, huge)
    (tmp_path / "huge.yaml").write_text(huge)
    with pytest.raises(ToolFailure, match="larger"):
        validate_config(state, "huge.yaml")
    (tmp_path / "binary.yaml").write_bytes(b"\xff\xfe\x00 not utf-8")
    with pytest.raises(ToolFailure, match="Cannot read"):
        validate_config(state, "binary.yaml")
    listed = validate_config(state, None, "- a\n- b\n")
    assert listed.data["valid"] is False and "mapping" in listed.data["errors"][0]["message"]


def test_validate_reports_files_that_do_not_exist(tmp_path: Path) -> None:
    state = _state(tmp_path)
    vector = CONFIG + "labels: {path: nope.geojson}\n"
    raster = CONFIG + "labels: {type: raster, path: nope.tif, classes: {1: 1}}\n"
    tif = CONFIG.replace(
        "{type: xyz, zoom: 16, source: esri_satellite}", "{type: geotiff, path: nope.tif}"
    )
    expected = (
        (vector, "labels.path", "nope.geojson"),
        (raster, "labels.path", "nope.tif"),
        (tif, "imagery.path", "nope.tif"),
    )
    for text, key, name in expected:
        result = validate_config(state, None, text)
        # A missing input makes plan and generate fail, so it is an error, not a warning.
        assert result.data["valid"] is False and result.data["warnings"] == []
        (error,) = result.data["errors"]
        assert error["field"] == key and f"file not found: {name}" in error["message"]
    assert validate_config(state, None, raster).data["summary"]["labels"]["type"] == "raster"


def test_plan_reports_what_it_cannot_plan(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    def fail(config: object) -> None:
        raise ValueError("the imagery is unreachable")

    monkeypatch.setattr("mapcv.agent_tools.make_plan", fail)
    with pytest.raises(ToolFailure, match="Cannot plan this config: the imagery is unreachable"):
        plan(_state(tmp_path), None, CONFIG)


def test_write_config_failures(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    state = _state(tmp_path, allow_write=True)
    (tmp_path / "folder.yaml").mkdir()
    with pytest.raises(ToolFailure, match="is a folder"):
        write_config(state, "folder.yaml", CONFIG)

    def fail(*args: object) -> None:
        raise OSError("disk full")

    monkeypatch.setattr("mapcv.agent_tools.os.replace", fail)
    with pytest.raises(ToolFailure, match="Cannot write c.yaml: disk full"):
        write_config(state, "c.yaml", CONFIG)
    assert not list(tmp_path.glob(".mapcv-*")) and not (tmp_path / "c.yaml").exists()
    monkeypatch.undo()
    written = write_config(state, "c.yaml", CONFIG.replace("\n", "\r\n").rstrip())
    assert (tmp_path / "c.yaml").read_bytes() == CONFIG.encode()  # LF, one final newline
    assert written.data["written"] == "c.yaml"


def test_generate_refuses_an_invalid_config_and_reports_failures(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    state = _state(tmp_path, allow_write=True)
    (tmp_path / "bad.yaml").write_text(CONFIG.replace("zoom: 16", "zoom: 99"))
    with pytest.raises(ToolFailure, match="imagery.zoom"):
        prepare_generate(state, "bad.yaml")

    (tmp_path / "c.yaml").write_text(CONFIG)
    job = prepare_generate(state, "c.yaml")

    def run(error: Exception) -> ToolResult:
        def fake(config: object, hook: object) -> None:
            raise error

        monkeypatch.setattr("mapcv.agent_tools.run_generate", fake)
        return execute_generate(state, job, lambda done, total: None, Event())

    with pytest.raises(ToolFailure, match="Cannot resume: different configuration"):
        run(ManifestMismatchError("different configuration"))
    with pytest.raises(ToolFailure, match="Generation failed: the server said no. Fix"):
        run(RuntimeError("the server said no."))
    with pytest.raises(ToolFailure, match="Generation failed: KeyError"):
        run(KeyError())

    # A cancelled call stops at the next chunk.
    cancelled = Event()
    cancelled.set()

    def stops(config: object, hook: Callable[[int, int], None]) -> None:
        hook(0, 3)

    monkeypatch.setattr("mapcv.agent_tools.run_generate", stops)
    with pytest.raises(ToolFailure, match="cancelled"):
        execute_generate(state, job, lambda done, total: None, cancelled)

    # What a finished run with failed tiles reports (and the folder was released each time).
    done = GenerateResult(job.config.writer.staging_dir, Manifest(patches=[]), 0, None, 10, 2, 1.0)
    monkeypatch.setattr("mapcv.agent_tools.run_generate", lambda config, hook: done)
    result = execute_generate(state, job, lambda *a: None, Event())
    assert "2 of 10 tiles failed" in result.summary


def test_a_folder_is_not_generated_into_twice(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    state = _state(tmp_path, allow_write=True)
    (tmp_path / "c.yaml").write_text(CONFIG)
    job = prepare_generate(state, "c.yaml")
    inner: list[str] = []

    def nested(config: object, hook: object) -> None:
        try:
            execute_generate(state, job, lambda *a: None, Event())
        except ToolFailure as exc:
            inner.append(exc.message)
        raise RuntimeError("stop")

    monkeypatch.setattr("mapcv.agent_tools.run_generate", nested)
    with pytest.raises(ToolFailure, match="Generation failed"):
        execute_generate(state, job, lambda *a: None, Event())
    assert inner and "still running" in inner[0]


def _manifest(task: str) -> Manifest:
    entries = [
        ManifestEntry(
            row=i,
            col=0,
            padded=i == 0,
            chunk=0,
            files={"image": f"images/p{i}.png"},
            summary=PatchSummary(class_objects={"1": 2, "2": 1}),
        )
        for i in range(3)
    ]
    return Manifest(
        task=task,
        sources=[SourceRecord(patch_shape=[8, 8, 3], dtype="uint8", crs="EPSG:3857")],
        target=TargetRecord(type=task, class_map={"house": 1, "pond": 2}),
        patches=entries,
    )


def test_info_reports_objects_for_detection_datasets(tmp_path: Path) -> None:
    folder = tmp_path / "boxes"
    folder.mkdir()
    _manifest("detection").save(folder / "manifest.json")
    (folder / "splits").mkdir()
    (folder / "splits" / "train.txt").write_text("p0.png\np1.png\n")
    data = info(_state(tmp_path), "boxes").data
    assert data["task"] == "detection" and data["patches"] == 3 and data["padded_patches"] == 1
    assert data["class_balance"][0] == {
        "id": 1,
        "name": "house",
        "objects": 6,
        "share": 0.6667,
        "patches": 3,
    }
    assert data["splits"] == {"train": 2, "val": 0, "test": 0}


def test_info_reports_patches_per_label_for_classification_datasets(tmp_path: Path) -> None:
    folder = tmp_path / "scenes"
    folder.mkdir()
    manifest = _manifest("classification")
    for entry, labels in zip(manifest.patches, ([1], [1, 2], [0])):
        entry["summary"] = PatchSummary(labels=labels, empty_ratio=0.0)
    manifest.save(folder / "manifest.json")
    data = info(_state(tmp_path), "scenes").data
    assert data["task"] == "classification" and data["patches"] == 3
    assert data["class_balance"] == [
        {"id": 0, "name": "background", "patches": 1, "share": 0.3333},
        {"id": 1, "name": "house", "patches": 2, "share": 0.6667},
        {"id": 2, "name": "pond", "patches": 1, "share": 0.3333},
    ]


def test_info_and_split_failures(tmp_path: Path) -> None:
    state = _state(tmp_path, allow_write=True)
    with pytest.raises(ToolFailure, match="No manifest"):
        info(state, "nothing")
    (tmp_path / "newer").mkdir()
    (tmp_path / "newer" / "manifest.json").write_text('{"version": 99, "patches": []}')
    with pytest.raises(ToolFailure, match="newer mapcv"):
        info(state, "newer")
    with pytest.raises(ToolFailure, match="Not a folder"):
        split(state, "missing")
    with pytest.raises(ToolFailure, match="Invalid split settings: test_ratio"):
        split(state, "newer", test_ratio=2.0)
    (tmp_path / "empty").mkdir()
    with pytest.raises(ToolFailure, match="No manifest"):
        split(state, "empty")


def test_inspect_labels_tolerates_odd_geojson(tmp_path: Path) -> None:
    state = _state(tmp_path)
    polygon = {"type": "Polygon"}
    features: list[tuple[object, object]] = [
        ({"a": None, "b": {"x": 1}}, None),
        (None, {**polygon, "coordinates": "?"}),
        ({"a": ""}, {**polygon, "coordinates": []}),
    ]
    collection = {
        "type": "FeatureCollection",
        "features": [
            {"type": "Feature", "properties": props, "geometry": geometry}
            for props, geometry in features
        ],
    }
    (tmp_path / "odd.geojson").write_text(json.dumps(collection))
    odd = inspect_labels(state, "odd.geojson")
    assert odd.data["features"] == 3 and odd.data["fields"] == [] and odd.data["extent"] is None
    single = {"type": "Feature", "properties": {"k": "v"}, "geometry": None}
    (tmp_path / "one.geojson").write_text(json.dumps(single))
    assert inspect_labels(state, "one.geojson").data["features"] == 1
    (tmp_path / "bad.geojson").write_text('{"type": "Topology"}')
    with pytest.raises(ToolFailure, match="FeatureCollection"):
        inspect_labels(state, "bad.geojson")
