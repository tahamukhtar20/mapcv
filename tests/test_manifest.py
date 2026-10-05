"""Manifest version 3: schema, round-trip, and reading and resuming 0.1/0.2 datasets."""

from __future__ import annotations

import hashlib
import io
import json
import shutil
import threading
import warnings
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional, Tuple

import numpy as np
import numpy.typing as npt
import pytest
from PIL import Image

import mapcv.pipeline as pipeline
from mapcv.config import MapcvConfig
from mapcv.imagery import RasterMetadata
from mapcv.manifest import (
    MANIFEST_VERSION,
    Manifest,
    ManifestMismatchError,
    SourceRecord,
    TargetRecord,
    load_or_create_manifest,
    patch_folders,
)
from mapcv.pipeline import run_generate, run_split
from mapcv.splitter import SplitterConfig

FIXTURES = Path(__file__).parent / "fixtures"
V1_MANIFEST = FIXTURES / "mapcv-0.1.0" / "manifest.json"
V2_DATASET = FIXTURES / "mapcv-0.2.0"
V2_DEV_MANIFEST = FIXTURES / "manifest-v2-0.3.0-dev.json"
ENTRY_KEYS = {"row", "col", "padded", "chunk", "files", "summary"}


# ── helpers ──────────────────────────────────────────────────────────────────


class _Tiles(BaseHTTPRequestHandler):
    """The tile server the 0.2.0 fixture was generated against."""

    def log_message(self, *args: object) -> None:
        pass

    def do_GET(self) -> None:  # noqa: N802 - http.server API
        z, x, y = (int(part) for part in self.path.strip("/").split(".")[0].split("/"))
        buffer = io.BytesIO()
        Image.new("RGB", (256, 256), ((x * 37) % 256, (y * 53) % 256, z * 9)).save(buffer, "PNG")
        body = buffer.getvalue()
        self.send_response(200)
        self.send_header("Content-Type", "image/png")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)


@pytest.fixture
def tile_port() -> Iterator[int]:
    server = ThreadingHTTPServer(("127.0.0.1", 0), _Tiles)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield int(server.server_address[1])
    finally:
        server.shutdown()
        thread.join()


def _copy_v2_dataset(tmp_path: Path, port: int, ignore_index: Optional[str]) -> MapcvConfig:
    root = tmp_path / "v2"
    shutil.copytree(V2_DATASET, root)
    config = root / "mapcv.yaml"
    text = config.read_text(encoding="utf-8").replace("PORT", str(port))
    if ignore_index is not None:
        text = text.replace("  label_field: class\n", f"  label_field: class\n{ignore_index}\n")
    config.write_text(text, encoding="utf-8")
    return MapcvConfig.from_yaml(config)


def _tree(root: Path) -> Dict[str, str]:
    return {
        path.relative_to(root).as_posix(): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in sorted(root.rglob("*"))
        if path.is_file()
    }


def _mtimes(root: Path) -> Dict[str, int]:
    return {
        path.relative_to(root).as_posix(): path.stat().st_mtime_ns
        for path in root.rglob("*")
        if path.is_file()
    }


class _FakeSource:
    def __init__(self) -> None:
        self.image = np.arange(6 * 5 * 2, dtype=np.float32).reshape(6, 5, 2)
        self.metadata = RasterMetadata(
            source_type="eopf_zarr",
            product_id="S2_TEST.zarr",
            width=5,
            height=6,
            bands=["b08", "b04"],
            dtype="float32",
            crs="EPSG:32632",
            transform=(10.0, 0.0, 500000.0, 0.0, -10.0, 5000000.0),
            chunk_rows=2,
        )

    def read_window(
        self, row_start: int, row_stop: int, col_start: int, col_stop: int
    ) -> Tuple[npt.NDArray[np.float32], npt.NDArray[np.bool_]]:
        window = self.image[row_start:row_stop, col_start:col_stop]
        return window, np.ones(window.shape[:2], dtype=np.bool_)

    def close(self) -> None:
        pass


def _fake_config(tmp_path: Path, labels: Optional[Path] = None) -> MapcvConfig:
    data: Dict[str, Any] = {
        "region": {"west": 9.0, "south": 45.0, "east": 9.1, "north": 45.1},
        "imagery": {"type": "eopf_zarr", "path": str(tmp_path / "x.zarr"), "bands": ["b08", "b04"]},
        "sampler": {"patch_size": 3, "stride": 2, "edge_strategy": "pad"},
        "writer": {"staging_dir": str(tmp_path / "dataset"), "image_format": "npy"},
        "split": {"strategy": "random", "labeled_ratios": [0.5]},
    }
    if labels is not None:
        data["labels"] = {"path": str(labels), "label_field": "class"}
    return MapcvConfig.model_validate(data)


def _labels_over_fake_source(tmp_path: Path) -> Path:
    from pyproj import Transformer

    to_wgs84 = Transformer.from_crs("EPSG:32632", "EPSG:4326", always_xy=True)
    ring = [
        list(to_wgs84.transform(x, y))
        for x, y in (
            (500005, 4999945),
            (500035, 4999945),
            (500035, 4999985),
            (500005, 4999985),
            (500005, 4999945),
        )
    ]
    path = tmp_path / "labels.geojson"
    feature = {
        "type": "Feature",
        "properties": {"class": "field"},
        "geometry": {"type": "Polygon", "coordinates": [ring]},
    }
    path.write_text(json.dumps({"type": "FeatureCollection", "features": [feature]}))
    return path


def _generate_fake(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, labels: bool = True
) -> Tuple[MapcvConfig, Manifest]:
    monkeypatch.setattr(pipeline, "open_raster_source", lambda region, imagery: _FakeSource())
    config = _fake_config(tmp_path, _labels_over_fake_source(tmp_path) if labels else None)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        result = run_generate(config)
    return config, result.manifest


# ── the version-3 schema ─────────────────────────────────────────────────────


def test_generated_manifest_has_the_v3_schema(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config, _ = _generate_fake(tmp_path, monkeypatch)
    staging = config.writer.staging_dir
    data = json.loads((staging / "manifest.json").read_text(encoding="utf-8"))

    assert list(data) == [
        "version",
        "mapcv_version",
        "task",
        "sources",
        "target",
        "writer",
        "sampler",
        "patches",
    ]
    assert data["version"] == MANIFEST_VERSION == 3
    assert data["task"] == "segmentation"
    assert data["sources"] == [
        {
            "name": "image",
            "source_type": "eopf_zarr",
            "product_id": "S2_TEST.zarr",
            "bands": ["b08", "b04"],
            "dtype": "float32",
            "crs": "EPSG:32632",
            "transform": [10.0, 0.0, 500000.0, 0.0, -10.0, 5000000.0],
            "patch_shape": [2, 3, 3],
        }
    ]
    target = data["target"]
    assert target["type"] == "segmentation"
    assert target["class_map"] == {"field": 1}
    assert target["ignore_index"] == 255
    assert target["dtype"] == "uint8"
    assert target["options"] == {}
    assert set(target["labels"]) == {"label_field", "classes", "all_touched", "sha256"}
    assert target["labels"]["sha256"] == hashlib.sha256(config.labels.path.read_bytes()).hexdigest()  # type: ignore[union-attr]
    assert data["writer"] == {
        "layout": "files",
        "image_format": "npy",
        "jpg_quality": 95,
        "mask_format": "png",
    }
    assert data["sampler"]["patch_size"] == 3

    patches: List[Dict[str, Any]] = data["patches"]
    assert len(patches) == 9  # 3 x 3 anchors with stride 2 and padding
    for index, entry in enumerate(patches):
        assert set(entry) == ENTRY_KEYS
        assert entry["files"] == {
            "image": f"Images/patch_{index:07d}.npy",
            "mask": f"Masks/patch_{index:07d}.png",
        }
        for path in entry["files"].values():
            assert (staging / path).is_file()
        assert set(entry["summary"]) == {"class_pixels", "empty_ratio"}
        mask = np.asarray(Image.open(staging / entry["files"]["mask"]))
        values, counts = np.unique(mask, return_counts=True)
        assert entry["summary"]["class_pixels"] == {
            str(int(value)): int(count) for value, count in zip(values, counts)
        }
        assert list(entry["summary"]["class_pixels"]) == sorted(
            entry["summary"]["class_pixels"], key=int
        )
    assert {entry["chunk"] for entry in patches} == {0, 1, 2}
    assert any(entry["padded"] for entry in patches)
    assert any("1" in entry["summary"]["class_pixels"] for entry in patches)

    names = (staging / "splits" / "train.txt").read_text().split()
    assert names and all(name.startswith("patch_") and name.endswith(".npy") for name in names)


def test_image_only_manifest_has_no_target_and_only_image_files(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config, manifest = _generate_fake(tmp_path, monkeypatch, labels=False)
    data = json.loads((config.writer.staging_dir / "manifest.json").read_text())

    assert data["task"] == "segmentation"
    assert data["target"] is None
    for entry in data["patches"]:
        assert set(entry["files"]) == {"image"}
        assert set(entry["summary"]) == {"empty_ratio"}
    assert manifest.class_map == {}
    assert manifest.ignore_index is None
    assert patch_folders(manifest) == ["Images"]


def test_manifest_file_has_one_line_per_patch(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config, manifest = _generate_fake(tmp_path, monkeypatch)
    lines = (config.writer.staging_dir / "manifest.json").read_text().splitlines()
    patch_lines = [line for line in lines if line.startswith('    {"row":')]
    assert len(patch_lines) == len(manifest.patches)
    assert lines[-2:] == ["  ]", "}"]


def test_v3_round_trip_is_lossless(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    config, manifest = _generate_fake(tmp_path, monkeypatch)
    path = config.writer.staging_dir / "manifest.json"

    loaded = Manifest.load(path)
    assert loaded == manifest
    assert loaded.upgraded_from is None and loaded.loaded_version == 3

    copy = tmp_path / "copy.json"
    loaded.save(copy)
    assert copy.read_text() == path.read_text()


def test_unknown_keys_survive_a_round_trip(tmp_path: Path) -> None:
    data = {
        "version": 3,
        "task": "segmentation",
        "future": {"x": 1},
        "sources": [{"name": "image", "future_source": True}],
        "target": {"type": "segmentation", "future_target": [1]},
        "patches": [
            {
                "row": 0,
                "col": 0,
                "padded": False,
                "chunk": 0,
                "files": {"image": "Images/a.png", "extra": "Extra/a.json"},
                "summary": {"empty_ratio": 0.0, "future_summary": 3},
                "future_entry": "kept",
            }
        ],
    }
    path = tmp_path / "manifest.json"
    path.write_text(json.dumps(data))
    Manifest.load(path).save(path)
    saved = json.loads(path.read_text())

    assert saved["future"] == {"x": 1}
    assert saved["sources"][0]["future_source"] is True
    assert saved["target"]["future_target"] == [1]
    assert saved["patches"][0]["future_entry"] == "kept"
    assert saved["patches"][0]["summary"]["future_summary"] == 3
    assert saved["patches"][0]["files"]["extra"] == "Extra/a.json"


def test_empty_manifest_round_trips(tmp_path: Path) -> None:
    path = tmp_path / "manifest.json"
    Manifest(task="segmentation").save(path)
    data = json.loads(path.read_text())
    assert data["patches"] == [] and data["version"] == 3
    assert Manifest.load(path).patches == []
    assert not (tmp_path / "manifest.json.tmp").exists()


@pytest.mark.parametrize("version", [4, 99])
def test_manifest_from_a_newer_mapcv_is_refused(tmp_path: Path, version: int) -> None:
    path = tmp_path / "manifest.json"
    path.write_text(json.dumps({"version": version, "patches": []}))
    with pytest.raises(ManifestMismatchError, match="newer mapcv"):
        Manifest.load(path)


@pytest.mark.parametrize("version", ["3", 0, True, None])
def test_manifest_with_an_invalid_version_is_refused(tmp_path: Path, version: object) -> None:
    path = tmp_path / "manifest.json"
    path.write_text(json.dumps({"version": version, "patches": []}))
    with pytest.raises(ManifestMismatchError, match="unknown manifest version"):
        Manifest.load(path)


def test_patch_transform_and_bounds_are_derived_from_row_and_col() -> None:
    manifest = Manifest(
        sources=[SourceRecord(transform=(10.0, 0.0, 500000.0, 0.0, -10.0, 5000000.0))],
        sampler={"patch_size": 4},
    )
    entry: Any = {"row": 2, "col": 3}
    assert manifest.patch_transform(entry) == (10.0, 0.0, 500030.0, 0.0, -10.0, 4999980.0)
    assert manifest.patch_bounds(entry) == (500030.0, 4999940.0, 500070.0, 4999980.0)


def test_patch_geometry_needs_a_transform_and_patch_size() -> None:
    entry: Any = {"row": 0, "col": 0}
    with pytest.raises(ValueError, match="no transform"):
        Manifest(sources=[SourceRecord()]).patch_transform(entry)
    with pytest.raises(ValueError, match="no patch size"):
        Manifest(sources=[SourceRecord(transform=(1, 0, 0, 0, -1, 0))]).patch_bounds(entry)


# ── reading version 1 (mapcv 0.1) ────────────────────────────────────────────


def test_version_one_manifest_is_upgraded_on_load() -> None:
    raw = json.loads(V1_MANIFEST.read_text())
    manifest = Manifest.load(V1_MANIFEST)

    assert manifest.version == 3
    assert manifest.upgraded_from == manifest.loaded_version == 1
    assert manifest.task == "segmentation"
    assert manifest.source == SourceRecord(name="image")
    assert manifest.sampler is None and manifest.writer is None
    assert manifest.target == TargetRecord(
        type="segmentation", class_map={"building": 1, "water": 2}, dtype="uint8"
    )
    assert len(manifest.patches) == len(raw["patches"]) == 12
    for old, new in zip(raw["patches"], manifest.patches):
        assert new == {
            "row": old["row"],
            "col": old["col"],
            "padded": old["padded"],
            "chunk": old["strip_index"],
            "files": {
                "image": f"Images/{old['filename']}",
                "mask": f"Masks/{old['mask_filename']}",
            },
            "summary": {
                "class_pixels": old["per_class_pixel_counts"],
                "empty_ratio": old["empty_ratio"],
            },
        }


def test_version_one_dataset_splits_but_does_not_resume(tmp_path: Path) -> None:
    dataset = tmp_path / "dataset"
    dataset.mkdir()
    shutil.copy(V1_MANIFEST, dataset / "manifest.json")
    with pytest.warns(UserWarning, match="patch size"):
        counts = run_split(dataset, SplitterConfig())
    names = {
        name
        for split in ("train", "val", "test")
        for name in (dataset / "splits" / f"{split}.txt").read_text().split()
    }
    assert counts["train"] + counts["val"] + counts["test"] == len(names) == 12
    assert names <= {f"patch_{index:07d}.png" for index in range(12)}
    assert (dataset / "manifest.json").read_bytes() == V1_MANIFEST.read_bytes()

    with pytest.raises(ManifestMismatchError, match="version-1"):
        load_or_create_manifest(dataset / "manifest.json", Manifest())


# ── reading version 2 (mapcv 0.2 and the 0.3 development line) ───────────────


def test_mapcv_0_2_manifest_is_upgraded_on_load() -> None:
    path = V2_DATASET / "dataset" / "manifest.json"
    raw = json.loads(path.read_text())
    manifest = Manifest.load(path)

    assert manifest.version == 3 and manifest.upgraded_from == 2
    assert manifest.mapcv_version is None
    assert manifest.source.model_dump() == {
        "name": "image",
        "source_type": raw["source_type"],
        "product_id": raw["product_id"],
        "bands": raw["bands"],
        "dtype": raw["dtype"],
        "crs": raw["crs"],
        "transform": tuple(raw["transform"]),
        "patch_shape": raw["patch_shape"],
    }
    # mapcv 0.2 had no ignore index: pixels without imagery were background.
    assert manifest.target == TargetRecord(
        type="segmentation",
        class_map=raw["class_map"],
        ignore_index=None,
        dtype="uint8",
        labels=raw["labels"],
    )
    assert manifest.writer == {"layout": "files", **raw["writer"], "mask_format": "png"}
    assert manifest.sampler == raw["sampler"]
    for old, new in zip(raw["patches"], manifest.patches):
        assert new["files"] == {
            "image": f"Images/{old['filename']}",
            "mask": f"Masks/{old['mask_filename']}",
        }
        assert new["summary"]["class_pixels"] == old["per_class_pixel_counts"]
        assert new["chunk"] == old["strip_index"]
        assert manifest.patch_name(new) == old["filename"]
        for file in new["files"].values():
            assert (V2_DATASET / "dataset" / file).is_file()


def test_development_v2_manifest_keeps_its_ignore_index() -> None:
    raw = json.loads(V2_DEV_MANIFEST.read_text())
    manifest = Manifest.load(V2_DEV_MANIFEST)

    assert raw["labels"]["ignore_index"] == 255
    assert manifest.ignore_index == 255
    assert manifest.target is not None and manifest.target.labels is not None
    assert "ignore_index" not in manifest.target.labels
    padded = [entry for entry in manifest.patches if entry["padded"]]
    assert padded and all("255" in entry["summary"]["class_pixels"] for entry in padded)


def test_image_only_version_two_manifest_has_no_target(tmp_path: Path) -> None:
    raw = json.loads((V2_DATASET / "dataset" / "manifest.json").read_text())
    raw.update(class_map={}, labels=None)
    for entry in raw["patches"]:
        entry.update(mask_filename=None, per_class_pixel_counts={})
    path = tmp_path / "manifest.json"
    path.write_text(json.dumps(raw))

    manifest = Manifest.load(path)
    assert manifest.target is None
    assert all(set(entry["files"]) == {"image"} for entry in manifest.patches)
    assert all(set(entry["summary"]) == {"empty_ratio"} for entry in manifest.patches)


def test_loading_never_rewrites_an_old_manifest(tmp_path: Path) -> None:
    dataset = tmp_path / "dataset"
    shutil.copytree(V2_DATASET / "dataset", dataset)
    before = _tree(dataset)
    Manifest.load(dataset / "manifest.json")
    run_split(dataset, SplitterConfig(test_ratio=0.25, val_ratio=0.2, labeled_ratios=[0.5]))
    assert _tree(dataset) == before


def test_mapcv_0_2_dataset_splits_identically(tmp_path: Path) -> None:
    dataset = tmp_path / "dataset"
    shutil.copytree(V2_DATASET / "dataset", dataset)
    shutil.rmtree(dataset / "splits")
    run_split(dataset, SplitterConfig(test_ratio=0.25, val_ratio=0.2, labeled_ratios=[0.5]))
    assert _tree(dataset / "splits") == _tree(V2_DATASET / "dataset" / "splits")


# ── resuming a mapcv 0.2 dataset ─────────────────────────────────────────────


def test_finished_mapcv_0_2_dataset_resumes_without_writing(tmp_path: Path, tile_port: int) -> None:
    config = _copy_v2_dataset(tmp_path, tile_port, "  ignore_index: null")
    staging = config.writer.staging_dir
    before, mtimes = _tree(staging), _mtimes(staging)

    result = run_generate(config)

    assert result.new_patches == 0
    assert result.tiles_requested == 0
    assert _tree(staging) == before  # no new files, identical split lists
    changed = {name for name, mtime in _mtimes(staging).items() if mtimes[name] != mtime}
    assert not any(name.startswith(("Images/", "Masks/")) for name in changed)
    assert "manifest.json" not in changed  # still the 0.2 (version 2) file
    assert json.loads((staging / "manifest.json").read_text())["version"] == 2


def test_interrupted_mapcv_0_2_dataset_resumes_to_the_same_dataset(
    tmp_path: Path, tile_port: int
) -> None:
    config = _copy_v2_dataset(tmp_path, tile_port, "  ignore_index: null")
    staging = config.writer.staging_dir
    original = _tree(staging)
    raw = json.loads((staging / "manifest.json").read_text())
    # As if 0.2 was stopped after the first chunk row: keep the first 4 patches.
    kept, lost = raw["patches"][:4], raw["patches"][4:]
    (staging / "manifest.json").write_text(json.dumps(dict(raw, patches=kept)))
    for entry in lost:
        (staging / "Images" / entry["filename"]).unlink()
        (staging / "Masks" / entry["mask_filename"]).unlink()
    shutil.rmtree(staging / "splits")

    result = run_generate(config)

    assert result.new_patches == len(lost)
    resumed = _tree(staging)
    assert resumed.pop("manifest.json") != original.pop("manifest.json")
    assert resumed == original  # images, masks and split lists as made by 0.2
    manifest = Manifest.load(staging / "manifest.json")
    assert manifest.version == 3 and manifest.upgraded_from is None
    assert manifest.patches == Manifest.from_dict(raw).patches


def test_mapcv_0_2_dataset_needs_ignore_index_null_to_resume(
    tmp_path: Path, tile_port: int
) -> None:
    config = _copy_v2_dataset(tmp_path, tile_port, None)
    with pytest.raises(ManifestMismatchError, match="ignore_index") as caught:
        run_generate(config)
    assert "labels.ignore_index: null" in str(caught.value)


def test_mapcv_0_2_random_sample_is_not_resumed(tmp_path: Path) -> None:
    raw = json.loads((V2_DATASET / "dataset" / "manifest.json").read_text())
    raw["sampler"]["mode"] = "random"
    path = tmp_path / "manifest.json"
    path.write_text(json.dumps(raw))
    expected = Manifest.from_dict(dict(raw, patches=[]))
    with pytest.raises(ManifestMismatchError, match="random"):
        load_or_create_manifest(path, expected)


# ── resume checks on version 3 ───────────────────────────────────────────────


def _expected() -> Manifest:
    return Manifest(
        task="segmentation",
        sources=[
            SourceRecord(
                bands=["b04"],
                patch_shape=[1, 4, 4],
                product_id="S2.zarr",
                transform=(10.0, 0.0, 0.0, 0.0, -10.0, 0.0),
            )
        ],
        target=TargetRecord(
            type="segmentation",
            class_map={"y": 2},
            ignore_index=255,
            dtype="uint8",
            labels={"sha256": "abc"},
        ),
        writer={"layout": "files", "image_format": "png", "jpg_quality": 95, "mask_format": "png"},
        sampler={"patch_size": 4},
    )


def test_load_or_create_returns_the_expected_manifest_when_missing(tmp_path: Path) -> None:
    expected = _expected()
    assert load_or_create_manifest(tmp_path / "manifest.json", expected) is expected


def test_load_or_create_resumes_a_matching_manifest(tmp_path: Path) -> None:
    path = tmp_path / "manifest.json"
    stored = _expected()
    stored.mapcv_version = "0.3.0"  # an older patch release made it: still resumable
    stored.sources[0].transform = (10.0 + 1e-12, 0.0, 0.0, 0.0, -10.0, 0.0)
    stored.save(path)
    assert load_or_create_manifest(path, _expected()).mapcv_version == "0.3.0"


@pytest.mark.parametrize(
    ("change", "named"),
    [
        (lambda m: setattr(m, "task", "detection"), "task"),
        (lambda m: setattr(m.sources[0], "bands", ["b08"]), "bands"),
        (lambda m: setattr(m.sources[0], "patch_shape", [1, 8, 8]), "patch_shape"),
        (lambda m: setattr(m.sources[0], "product_id", "other.zarr"), "product_id"),
        (lambda m: setattr(m.sources[0], "transform", (20.0, 0, 0, 0, -20.0, 0)), "transform"),
        (lambda m: setattr(m.sources[0], "name", "before"), "sources"),
        (lambda m: setattr(m.target, "class_map", {"other": 1}), "class_map"),
        (lambda m: setattr(m.target, "ignore_index", None), "ignore_index"),
        (lambda m: setattr(m.target, "labels", {"sha256": "edited"}), "labels"),
        (lambda m: setattr(m.target, "options", {"rule": "any"}), "task options"),
        (lambda m: setattr(m, "target", None), "labels"),
        (lambda m: setattr(m, "sampler", {"patch_size": 8}), "sampler"),
        (lambda m: m.writer.update(image_format="jpg"), "writer"),
    ],
)
def test_load_or_create_names_what_differs(tmp_path: Path, change: Any, named: str) -> None:
    path = tmp_path / "manifest.json"
    _expected().save(path)
    expected = _expected()
    change(expected)
    with pytest.raises(ManifestMismatchError, match=named):
        load_or_create_manifest(path, expected)
