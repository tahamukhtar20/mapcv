"""``mapcv.data.MapcvDataset`` and ``mapcv export`` on generated datasets.

Items are compared with the files as rasterio, Pillow and numpy read them, labels with
``labels.csv``, boxes with the COCO files, and exports read back with pyarrow and YAML.
"""

from __future__ import annotations

import csv
import json
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import numpy.typing as npt
import pytest
import yaml
from PIL import Image
from typer.testing import CliRunner

pytest.importorskip("rasterio", reason="these tests write their rasters with rasterio")
import rasterio
from test_multi_source import (
    PATCH,
    reference_transform,
    region_inside,
    write_labels,
    write_raster,
)

from mapcv.cli import app
from mapcv.config import MapcvConfig
from mapcv.data import MapcvDataset, _to_tensors, read_image, read_mask, splits_of
from mapcv.export import export_hf_parquet, terratorch_config
from mapcv.manifest import Manifest
from mapcv.pipeline import run_generate
from mapcv.stats import write_stats

WIDTH, HEIGHT = 470, 400
runner = CliRunner()
ENV = {"COLUMNS": "200"}


@pytest.fixture
def scene(tmp_path: Path) -> dict[str, Any]:
    ref = reference_transform()
    write_raster(tmp_path / "multi.tif", ref, WIDTH, HEIGHT, count=4, dtype="uint16", seed=1)
    write_raster(tmp_path / "other.tif", ref, WIDTH, HEIGHT, count=4, dtype="uint16", seed=2)
    write_raster(tmp_path / "rgb.tif", ref, WIDTH, HEIGHT, seed=3)
    region = region_inside(ref, WIDTH, HEIGHT, margin=0.0)
    return {"region": region, "labels": write_labels(tmp_path, region_inside(ref, WIDTH, HEIGHT))}


def _generate(tmp_path: Path, scene: dict[str, Any], name: str, **changes: Any) -> Path:
    data: dict[str, Any] = {
        "region": scene["region"],
        "imagery": {"type": "geotiff", "path": str(tmp_path / "multi.tif")},
        "labels": {"path": str(scene["labels"]), "label_field": "kind", "classes": {"a": 1}},
        "sampler": {"patch_size": PATCH, "edge_strategy": "pad"},
        "writer": {
            "staging_dir": str(tmp_path / name),
            "image_format": "tif",
            "mask_format": "tif",
        },
        "split": {"strategy": "random", "seed": 4},
    }
    for key, value in changes.items():
        if isinstance(value, dict) and isinstance(data.get(key), dict):
            data[key] = {**data[key], **value}
        else:
            data[key] = value
    if data.get("task") in ("classification", "detection", "instance"):
        data["writer"].pop("mask_format")  # these tasks write no masks
    run_generate(MapcvConfig.model_validate(data))
    return tmp_path / name


def _rasterio(path: Path) -> npt.NDArray[Any]:
    with rasterio.open(path) as src:
        data: npt.NDArray[Any] = src.read()
    return data


def _split(root: Path, split: str) -> list[str]:
    return (root / "splits" / f"{split}.txt").read_text().split()


# ── Reading files ────────────────────────────────────────────────────────────


@pytest.mark.parametrize(
    ("image_format", "mask_format"), [("tif", "tif"), ("png", "png"), ("npy", "npy")]
)
def test_items_match_the_files(
    tmp_path: Path, scene: dict[str, Any], image_format: str, mask_format: str
) -> None:
    imagery = {"path": str(tmp_path / ("rgb.tif" if image_format == "png" else "multi.tif"))}
    root = _generate(
        tmp_path,
        scene,
        "d",
        imagery={"type": "geotiff", **imagery},
        writer={"image_format": image_format, "mask_format": mask_format},
    )
    dataset = MapcvDataset(root, "val", as_tensors=False)
    names = _split(root, "val")
    assert len(dataset) == len(names) > 0 and repr(dataset).endswith(f"{len(names)} patches)")
    manifest = Manifest.load(root / "manifest.json")
    assert dataset.ignore_index == 255 and dataset.classes == {1: "a"}
    for index, name in enumerate(names):
        item = dataset[index]
        entry = next(e for e in manifest.patches if manifest.patch_name(e) == name)
        assert item["name"] == name and (item["row"], item["col"]) == (entry["row"], entry["col"])
        assert (
            item["transform"] == manifest.patch_transform(entry)
            and item["crs"] == manifest.source.crs
        )
        image_path = root / entry["files"]["image"]
        mask_path = root / entry["files"]["mask"]
        if image_format == "tif":
            expected, mask = _rasterio(image_path), _rasterio(mask_path)[0]
        elif image_format == "png":
            expected = np.moveaxis(np.asarray(Image.open(image_path)), -1, 0)
            mask = np.asarray(Image.open(mask_path))
        else:
            expected, mask = np.load(image_path), np.load(mask_path)
        np.testing.assert_array_equal(item["image"], expected)
        np.testing.assert_array_equal(item["mask"], mask)
        assert item["image"].shape[1:] == item["mask"].shape == (PATCH, PATCH)
        assert "images" not in item and "labels" not in item


def test_normalisation_uses_the_train_statistics(tmp_path: Path, scene: dict[str, Any]) -> None:
    root = _generate(tmp_path, scene, "d")
    raw = MapcvDataset(root, "test", as_tensors=False)
    computed = MapcvDataset(root, "test", normalize=True, as_tensors=False)  # no stats.json yet
    _, stats = write_stats(root, "train")
    stored = MapcvDataset(root, "test", normalize=True, as_tensors=False)
    mean = np.array(stats["sources"]["image"]["mean"], dtype=np.float32)[:, None, None]
    std = np.array(stats["sources"]["image"]["std"], dtype=np.float32)[:, None, None]
    for index in range(len(raw)):
        expected = (raw[index]["image"].astype(np.float32) - mean) / std
        np.testing.assert_allclose(stored[index]["image"], expected, rtol=1e-6)
        np.testing.assert_allclose(computed[index]["image"], expected, rtol=1e-6)
        assert stored[index]["image"].dtype == np.float32


def test_splits_subsets_and_missing_lists(tmp_path: Path, scene: dict[str, Any]) -> None:
    root = _generate(tmp_path, scene, "d")
    total = len(Manifest.load(root / "manifest.json").patches)
    assert len(MapcvDataset(root, "all", as_tensors=False)) == total
    labeled = (root / "splits" / "10" / "labeled.txt").read_text().split()
    assert [i["name"] for i in MapcvDataset(root, "10/labeled", as_tensors=False)] == labeled
    assert splits_of(root) == ["train", "val", "test"]
    with pytest.raises(FileNotFoundError, match="available: .*10/labeled.*train.*all"):
        MapcvDataset(root, "holdout")
    with pytest.raises(FileNotFoundError, match="No manifest"):
        MapcvDataset(tmp_path / "nowhere")
    plain = _generate(tmp_path, scene, "plain", split=None)
    assert splits_of(plain) == ("all",)
    with pytest.raises(FileNotFoundError, match="available: all"):
        MapcvDataset(plain, "train")


def test_several_sources_and_stacks(tmp_path: Path, scene: dict[str, Any]) -> None:
    sources = [
        {"type": "geotiff", "name": "t1", "path": str(tmp_path / "multi.tif")},
        {"type": "geotiff", "name": "t2", "path": str(tmp_path / "other.tif")},
    ]
    separate = _generate(tmp_path, scene, "separate", imagery=sources)
    stacked = _generate(
        tmp_path,
        scene,
        "stacked",
        imagery=sources,
        writer={"image_format": "npy", "stack_sources": True},
    )
    a = MapcvDataset(separate, "all", as_tensors=False)
    b = MapcvDataset(stacked, "all", as_tensors=False)
    manifest = Manifest.load(separate / "manifest.json")
    for index in range(len(a)):
        item, stack = a[index], b[index]
        assert set(item["images"]) == {"t1", "t2"} and item["image"] is item["images"]["t1"]
        entry = manifest.patches[index]
        np.testing.assert_array_equal(
            item["images"]["t2"], _rasterio(separate / entry["files"]["t2"])
        )
        np.testing.assert_array_equal(
            stack["image"], np.stack([item["images"]["t1"], item["images"]["t2"]])
        )
    write_stats(stacked, "all")
    write_stats(separate, "all")
    norm_a = MapcvDataset(separate, "all", normalize=True, as_tensors=False)[0]
    norm_b = MapcvDataset(stacked, "all", normalize=True, as_tensors=False)[0]
    np.testing.assert_allclose(norm_b["image"][1], norm_a["images"]["t2"], rtol=1e-6)


def test_classification_labels_and_detection_boxes(tmp_path: Path, scene: dict[str, Any]) -> None:
    rgb = {"type": "geotiff", "path": str(tmp_path / "rgb.tif")}
    classes = _generate(
        tmp_path,
        scene,
        "classes",
        task="classification",
        imagery=rgb,
        classification={"mode": "multi", "min_fraction": 0.0},
        writer={"image_format": "png"},
    )
    with (classes / "labels.csv").open(newline="") as handle:
        rows = {row["image"].rsplit("/", 1)[-1]: row["labels"] for row in csv.DictReader(handle)}
    ids = {name: cid for name, cid in Manifest.load(classes / "manifest.json").class_map.items()}
    dataset = MapcvDataset(classes, "all", as_tensors=False)
    assert len(dataset) > 0
    for item in dataset:  # iterable through __getitem__/__len__
        names = [n for n in rows[item["name"]].split("|") if n]
        assert item["labels"].tolist() == sorted(ids[n] for n in names) and "mask" not in item

    from test_dataset_tools import _small_boxes

    small = {
        "path": str(_small_boxes(tmp_path, scene["region"])),
        "label_field": "kind",
        "classes": {"a": 1},
    }
    boxes = _generate(
        tmp_path,
        scene,
        "boxes",
        task="detection",
        imagery=rgb,
        labels=small,
        writer={"image_format": "png"},
    )
    coco: dict[str, list[dict[str, Any]]] = {}
    for path in (boxes / "annotations").glob("instances_*.json"):
        doc = json.loads(path.read_text())
        files = {image["id"]: image["file_name"] for image in doc["images"]}
        for annotation in doc["annotations"]:
            coco.setdefault(files[annotation["image_id"]], []).append(annotation)
    found = 0
    for item in MapcvDataset(boxes, "all", as_tensors=False):
        expected = coco.get(item["name"], [])
        assert item["boxes"].shape == (len(expected), 4)
        np.testing.assert_allclose(item["boxes"], [a["bbox"] for a in expected] or np.zeros((0, 4)))
        assert item["categories"].tolist() == [a["category_id"] for a in expected]
        found += len(expected)
    assert found > 0
    for path in (boxes / "annotations").glob("instances_*.json"):
        path.unlink()
    with pytest.raises(FileNotFoundError, match="No COCO files"):
        MapcvDataset(boxes, "all")


def test_tensors_and_transforms(
    tmp_path: Path, scene: dict[str, Any], monkeypatch: pytest.MonkeyPatch
) -> None:
    root = _generate(tmp_path, scene, "d")
    fake = SimpleNamespace(from_numpy=lambda array: ("tensor", array.dtype, array.shape))
    seen: list[bool] = []
    record = SimpleNamespace(from_numpy=lambda array: seen.append(array.flags.writeable))
    monkeypatch.setitem(sys.modules, "torch", record)
    read_only = np.zeros(3)
    read_only.flags.writeable = False
    _to_tensors({"image": read_only})
    assert seen == [True] and not read_only.flags.writeable  # torch gets a writable copy
    monkeypatch.setitem(sys.modules, "torch", fake)
    item = _to_tensors(
        {
            "image": np.zeros((2, 2), np.uint16),
            "images": {"x": np.ones(3)},
            "name": "a",
            "annotations": [{"id": 1}],
        }
    )
    assert item["image"] == ("tensor", np.dtype(np.int32), (2, 2))  # uint16 widened for torch
    assert (
        item["images"]["x"][0] == "tensor"
        and item["name"] == "a"
        and item["annotations"] == [{"id": 1}]
    )
    monkeypatch.setattr("mapcv.data.importlib.util.find_spec", lambda name: object())
    assert MapcvDataset(root, "val").as_tensors is True  # torch "installed": tensors by default
    assert MapcvDataset(root, "val")[0]["image"][0] == "tensor"
    monkeypatch.setattr("mapcv.data.importlib.util.find_spec", lambda name: None)
    assert MapcvDataset(root, "val").as_tensors is False
    with pytest.raises(ImportError, match="needs PyTorch"):
        MapcvDataset(root, "val", as_tensors=True)
    shapes = MapcvDataset(root, "val", transform=lambda item: item["image"].shape)
    assert shapes[0] == (4, PATCH, PATCH)


def test_read_helpers(tmp_path: Path) -> None:
    gray = tmp_path / "g.png"
    Image.fromarray(np.arange(16, dtype=np.uint8).reshape(4, 4)).save(gray)
    assert read_image(gray).shape == (1, 4, 4) and read_mask(gray).shape == (4, 4)
    assert read_image(gray).flags.writeable  # augmentations may write in place
    np.save(tmp_path / "m.npy", np.zeros((3, 3), np.uint8))
    assert read_image(tmp_path / "m.npy").shape == (1, 3, 3)


# ── Exports ──────────────────────────────────────────────────────────────────


def test_hf_parquet_export(tmp_path: Path, scene: dict[str, Any]) -> None:
    pq = pytest.importorskip("pyarrow.parquet")
    root = _generate(
        tmp_path,
        scene,
        "d",
        writer={"image_format": "png", "mask_format": "png"},
        imagery={"type": "geotiff", "path": str(tmp_path / "rgb.tif")},
    )
    written = export_hf_parquet(root, tmp_path / "hf")
    assert sorted(p.name for p in written) == [
        f"{s}-00000-of-00001.parquet" for s in ("test", "train", "val")
    ]
    manifest = Manifest.load(root / "manifest.json")
    by_name = {manifest.patch_name(e): e for e in manifest.patches}
    for split in ("train", "val", "test"):
        table = pq.read_table(tmp_path / "hf" / "data" / f"{split}-00000-of-00001.parquet")
        rows = table.to_pylist()
        assert [row["name"] for row in rows] == _split(root, split)
        for row in rows:
            entry = by_name[row["name"]]
            assert row["image"]["bytes"] == (root / entry["files"]["image"]).read_bytes()
            assert row["mask"]["bytes"] == (root / entry["files"]["mask"]).read_bytes()
            assert (row["row"], row["col"]) == (entry["row"], entry["col"])
            assert row["transform"] == pytest.approx(list(manifest.patch_transform(entry)))
        features = json.loads(table.schema.metadata[b"huggingface"])["info"]["features"]
        assert features["image"] == {"_type": "Image"} and features["mask"] == {"_type": "Image"}
    card = (tmp_path / "hf" / "README.md").read_text(encoding="utf-8")
    front = yaml.safe_load(card.split("---\n")[1])
    assert front["configs"][0]["data_files"][0] == {"split": "train", "path": "data/train-*"}
    with pytest.raises(ValueError, match="new folder"):
        export_hf_parquet(root, root)


def test_hf_parquet_classification_and_binary_columns(
    tmp_path: Path, scene: dict[str, Any]
) -> None:
    pq = pytest.importorskip("pyarrow.parquet")
    root = _generate(
        tmp_path,
        scene,
        "c",
        task="classification",
        split=None,
        classification={"mode": "multi", "min_fraction": 0.0},
        writer={"image_format": "npy"},
    )
    export_hf_parquet(root, tmp_path / "hf")
    table = pq.read_table(tmp_path / "hf" / "data" / "all-00000-of-00001.parquet")
    features = json.loads(table.schema.metadata[b"huggingface"])["info"]["features"]
    assert features["image"]["bytes"] == {"dtype": "binary", "_type": "Value"}
    manifest = Manifest.load(root / "manifest.json")
    for row, entry in zip(table.to_pylist(), manifest.patches):
        assert row["labels"] == [int(v) for v in entry["summary"].get("labels") or []]
        assert row["image"]["path"].endswith(".npy")


def test_hf_parquet_needs_pyarrow(
    tmp_path: Path, scene: dict[str, Any], monkeypatch: pytest.MonkeyPatch
) -> None:
    root = _generate(tmp_path, scene, "d")
    monkeypatch.setitem(sys.modules, "pyarrow", None)
    with pytest.raises(RuntimeError, match=r"mapcv\[export\]"):
        export_hf_parquet(root, tmp_path / "hf")


def test_terratorch_config(tmp_path: Path, scene: dict[str, Any]) -> None:
    root = _generate(tmp_path, scene, "d")
    _, stats = write_stats(root, "train")
    text = terratorch_config(root)
    data = yaml.safe_load(text)["data"]
    assert data["class_path"] == "terratorch.datamodules.GenericNonGeoSegmentationDataModule"
    args = data["init_args"]
    assert (
        args["num_classes"] == 2 and args["img_grep"] == "*.tif" and args["label_grep"] == "*.tif"
    )
    assert args["train_data_root"] == str((root / "Images").resolve())
    assert args["val_label_data_root"] == str((root / "Masks").resolve())
    assert args["test_split"] == str((root / "splits" / "test.txt").resolve())
    assert args["means"] == pytest.approx(stats["sources"]["image"]["mean"], rel=1e-6)
    assert args["stds"] == pytest.approx(stats["sources"]["image"]["std"], rel=1e-6)
    assert args["no_label_replace"] == -1 and "ignore_index to -1" in text

    png = _generate(
        tmp_path,
        scene,
        "png",
        imagery={"type": "geotiff", "path": str(tmp_path / "rgb.tif")},
        writer={"image_format": "png", "mask_format": "png"},
        split=None,
    )
    text = terratorch_config(png)
    assert "ignore_index to 255" in text and "no split lists" in text
    assert "train_split" not in yaml.safe_load(text)["data"]["init_args"]

    with pytest.raises(ValueError, match="NPY"):
        terratorch_config(_generate(tmp_path, scene, "npy", writer={"image_format": "npy"}))
    with pytest.raises(ValueError, match="single-source segmentation and regression"):
        terratorch_config(
            _generate(
                tmp_path,
                scene,
                "det",
                task="detection",
                imagery={"type": "geotiff", "path": str(tmp_path / "rgb.tif")},
                writer={"image_format": "png"},
            )
        )


def test_export_cli(tmp_path: Path, scene: dict[str, Any]) -> None:
    root = _generate(tmp_path, scene, "d")
    result = runner.invoke(app, ["export", str(root), "--format", "terratorch"], env=ENV)
    assert result.exit_code == 0, result.output
    assert (
        yaml.safe_load((root / "terratorch.yaml").read_text())["data"]["init_args"]["num_classes"]
        == 2
    )
    out = tmp_path / "custom.yaml"
    assert (
        runner.invoke(
            app, ["export", str(root), "-f", "terratorch", "-o", str(out)], env=ENV
        ).exit_code
        == 0
    )
    assert out.exists()
    result = runner.invoke(app, ["export", str(root), "--format", "coco"], env=ENV)
    assert result.exit_code == 2 and "hf-parquet, terratorch" in result.output
    result = runner.invoke(app, ["export", str(root), "--format", "hf-parquet"], env=ENV)
    assert result.exit_code == 2 and "required with --format hf-parquet" in result.output
    result = runner.invoke(app, ["export", str(tmp_path / "nowhere"), "-f", "terratorch"], env=ENV)
    assert result.exit_code == 1 and "No manifest" in result.output
    pytest.importorskip("pyarrow")
    result = runner.invoke(
        app, ["export", str(root), "-f", "hf-parquet", "-o", str(tmp_path / "hf")], env=ENV
    )
    assert result.exit_code == 0, result.output
    assert "3 Parquet files" in result.output and (tmp_path / "hf" / "README.md").exists()


# ── Unfinished datasets ──────────────────────────────────────────────────────


def _png_dataset(tmp_path: Path, scene: dict[str, Any], name: str = "d") -> Path:
    return _generate(
        tmp_path,
        scene,
        name,
        writer={"image_format": "png", "mask_format": "png"},
        imagery={"type": "geotiff", "path": str(tmp_path / "rgb.tif")},
    )


def _unfinish(root: Path) -> None:
    """Make the manifest say that generate stopped before it finished."""
    manifest = Manifest.load(root / "manifest.json")
    manifest.complete = False
    manifest.save(root / "manifest.json")


def test_stats_and_card_warn_about_a_dataset_that_generate_did_not_finish(
    tmp_path: Path, scene: dict[str, Any]
) -> None:
    root = _png_dataset(tmp_path, scene)
    fine = runner.invoke(app, ["stats", str(root)], env=ENV)
    assert fine.exit_code == 0 and "incomplete" not in fine.output
    _unfinish(root)
    for command in ("stats", "card"):
        result = runner.invoke(
            app, [command, str(root), *(["--force"] if command == "card" else [])], env=ENV
        )
        text = " ".join(result.output.split())
        assert result.exit_code == 0, result.output
        assert "⚠" in text and "the dataset is incomplete" in text, text
        assert "Run mapcv generate again" in text
    assert (root / "stats.json").exists() and (root / "README.md").exists()
    with pytest.warns(UserWarning, match="the dataset is incomplete"):
        write_stats(root)
