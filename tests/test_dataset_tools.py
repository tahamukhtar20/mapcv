"""``mapcv stats``, ``mapcv card`` and ``mapcv verify`` on generated datasets.

Statistics are checked against numpy over the written patch files: every valid pixel
of the counted split is gathered and ``np.mean``/``np.std`` (population) taken in one
go, and the class weights are re-derived from the masks (Eigen & Fergus median
frequency), so neither the streaming merge nor the manifest summaries are trusted.
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Any

import numpy as np
import numpy.typing as npt
import pytest
import yaml
from PIL import Image
from typer.testing import CliRunner

pytest.importorskip("rasterio", reason="these tests write their rasters with rasterio")

from test_multi_source import (
    PATCH,
    reference_transform,
    region_inside,
    write_labels,
    write_raster,
)

from mapcv.card import card_text, write_card
from mapcv.cli import app
from mapcv.config import MapcvConfig
from mapcv.manifest import Manifest, SourceRecord
from mapcv.pipeline import run_generate
from mapcv.stats import Moments, _valid_pixels, dataset_stats, write_stats
from mapcv.verify import CHECKSUMS_FILENAME, verify_dataset, write_checksums

runner = CliRunner()
ENV = {"COLUMNS": "200"}
# Not whole patches across: padded edge patches have pixels without imagery.
WIDTH, HEIGHT = 470, 400


def _config(tmp_path: Path, imagery: Any, labels: Path | None, **extra: Any) -> MapcvConfig:
    writer = {"staging_dir": str(tmp_path / "dataset"), **extra.pop("writer", {})}
    data: dict[str, Any] = {
        "region": extra.pop("region"),
        "imagery": imagery,
        "sampler": {"patch_size": PATCH, "edge_strategy": "pad"},
        "writer": writer,
        "split": {"strategy": "random", "test_ratio": 0.25, "val_ratio": 0.2, "seed": 3},
        **extra,
    }
    if labels is not None:
        data["labels"] = {"path": str(labels), "label_field": "kind", "classes": {"a": 1}}
    return MapcvConfig.model_validate(data)


@pytest.fixture
def scene(tmp_path: Path) -> dict[str, Any]:
    ref = reference_transform()
    write_raster(tmp_path / "image.tif", ref, WIDTH, HEIGHT, count=4, dtype="uint16", seed=5)
    write_raster(tmp_path / "other.tif", ref, WIDTH, HEIGHT, count=4, dtype="uint16", seed=6)
    write_raster(tmp_path / "rgb.tif", ref, WIDTH, HEIGHT, seed=7)
    # The whole raster: edge patches are padded with pixels without imagery, which the
    # mask marks as ignored and the statistics must leave out.
    region = region_inside(ref, WIDTH, HEIGHT, margin=0.0)
    inner = region_inside(ref, WIDTH, HEIGHT)
    return {"region": region, "labels": write_labels(tmp_path, inner)}


def _names(staging: Path, split: str) -> list[str]:
    return (staging / "splits" / f"{split}.txt").read_text(encoding="utf-8").split()


def _entries(staging: Path, split: str) -> list[Any]:
    manifest = Manifest.load(staging / "manifest.json")
    if split == "all":
        return list(manifest.patches)
    names = set(_names(staging, split))
    return [e for e in manifest.patches if manifest.patch_name(e) in names]


def _image(path: Path) -> npt.NDArray[Any]:
    """A patch file as ``(H, W, C)``."""
    if path.suffix == ".npy":
        array = np.load(path)
        return array[:, :, None] if array.ndim == 2 else np.moveaxis(array, 0, -1)
    import rasterio

    if path.suffix == ".tif":
        with rasterio.open(path) as src:
            return np.moveaxis(src.read(), 0, -1)
    array = np.asarray(Image.open(path))
    return array[:, :, None] if array.ndim == 2 else array


def _expected_moments(
    staging: Path, entries: list[Any], key: str, ignore: int = 255
) -> dict[str, Any]:
    pixels = []
    for entry in entries:
        image = _image(staging / entry["files"][key]).astype(np.float64)
        mask = _image(staging / entry["files"]["mask"])[:, :, 0]
        pixels.append(image[mask != ignore])
    values = np.concatenate(pixels)
    return {
        "mean": values.mean(axis=0),
        "std": values.std(axis=0),
        "min": values.min(axis=0),
        "max": values.max(axis=0),
        "pixels": len(values),
    }


def _assert_source(found: dict[str, Any], expected: dict[str, Any]) -> None:
    np.testing.assert_allclose(found["mean"], expected["mean"], rtol=1e-12)
    np.testing.assert_allclose(found["std"], expected["std"], rtol=1e-10)
    np.testing.assert_array_equal(found["min"], expected["min"])
    np.testing.assert_array_equal(found["max"], expected["max"])
    assert found["pixels"] == [expected["pixels"]] * len(found["mean"])


def _expected_weights(staging: Path, entries: list[Any]) -> dict[int, float]:
    pixels: dict[int, int] = {}
    present_in: dict[int, int] = {}
    for entry in entries:
        mask = _image(staging / entry["files"]["mask"])[:, :, 0]
        valid = int((mask != 255).sum())
        for cid in np.unique(mask[mask != 255]):
            count = int((mask == cid).sum())
            pixels[int(cid)] = pixels.get(int(cid), 0) + count
            present_in[int(cid)] = present_in.get(int(cid), 0) + valid
    freq = {c: pixels[c] / present_in[c] for c in pixels}
    median = float(np.median(list(freq.values())))
    return {c: median / f for c, f in freq.items()}


# ── Streaming moments ────────────────────────────────────────────────────────


def test_moments_merge_batches_like_one_pass_even_far_from_zero() -> None:
    rng = np.random.default_rng(0)
    state = Moments(2)
    batches = []
    for size in (1, 7, 1000, 3, 4096):
        # Values near 1e9 with unit spread: a naive sum of squares loses every digit.
        batch = 1e9 + rng.normal(size=(size, 1, 2))
        valid = rng.random((size, 1)) > 0.3
        state.add(batch, valid)
        batches.append(batch[valid])
    every = np.concatenate(batches)
    summary = state.summary(["x", "y"])
    np.testing.assert_allclose(summary["mean"], every.mean(axis=0), rtol=1e-14)
    np.testing.assert_allclose(summary["std"], every.std(axis=0), rtol=1e-6)
    assert summary["pixels"] == [len(every)] * 2


def test_moments_skip_nan_per_band_and_report_empty_bands_as_none() -> None:
    state = Moments(3)
    values = np.array([[[1.0, np.nan, np.nan], [3.0, 4.0, np.nan]]])
    state.add(values, np.ones((1, 2), dtype=bool))
    summary = state.summary(["a", "b", "c"])
    assert summary["mean"] == [2.0, 4.0, None] and summary["std"] == [1.0, 0.0, None]
    assert summary["pixels"] == [2, 1, 0]


def test_valid_pixels_rules() -> None:
    image = np.array([[[0, 0], [0, 5], [7, 7]]], dtype=np.uint8)
    assert _valid_pixels(image, None, False).tolist() == [[True, True, True]]
    assert _valid_pixels(image, None, True).tolist() == [[False, True, True]]
    assert _valid_pixels(image, 7.0, False).tolist() == [[True, True, False]]
    floats = np.array([[[1.0, np.nan], [2.0, 2.0]]], dtype=np.float32)
    assert _valid_pixels(floats, float("nan"), False).tolist() == [[False, True]]


# ── Statistics against numpy ─────────────────────────────────────────────────


@pytest.mark.parametrize("image_format", ["png", "npy", "tif"])
def test_band_stats_and_class_weights_over_the_train_split(
    tmp_path: Path, scene: dict[str, Any], image_format: str
) -> None:
    count = 3 if image_format == "png" else 4
    imagery = {"type": "geotiff", "path": str(tmp_path / "image.tif")}
    if image_format == "png":
        imagery["path"] = str(tmp_path / "rgb.tif")
    config = _config(
        tmp_path,
        imagery,
        scene["labels"],
        region=scene["region"],
        writer={
            "image_format": image_format,
            "mask_format": "npy" if image_format == "npy" else "png",
        },
    )
    run_generate(config)
    staging = tmp_path / "dataset"
    train = _entries(staging, "train")
    assert 0 < len(train) < len(_entries(staging, "all"))
    masks = [_image(staging / e["files"]["mask"]) for e in train]
    assert any((m == 255).any() for m in masks), "no patch has pixels without imagery"

    path, stats = write_stats(staging)
    assert path == staging / "stats.json" and json.loads(path.read_text(encoding="utf-8")) == stats
    assert stats["split"] == "train" and stats["patches"] == len(train)
    found = stats["sources"]["image"]
    assert len(found["bands"]) == count
    _assert_source(found, _expected_moments(staging, train, "image"))

    weights = _expected_weights(staging, train)
    classes = stats["classes"]
    assert classes["median_frequency_weights"] == pytest.approx(
        {"background": weights[0], "a": weights[1]}, rel=1e-12
    )
    pixels = {name: classes["pixels"][name] for name in ("background", "a")}
    total = sum(pixels.values())
    assert classes["frequency"] == pytest.approx({k: v / total for k, v in pixels.items()})

    every = dataset_stats(staging, "all")
    assert every["split"] == "all" and every["patches"] == len(_entries(staging, "all"))
    _assert_source(
        every["sources"]["image"], _expected_moments(staging, _entries(staging, "all"), "image")
    )
    assert every["sources"] != stats["sources"]


def test_stacked_sources_match_separate_files(tmp_path: Path, scene: dict[str, Any]) -> None:
    imagery = [
        {"type": "geotiff", "name": "t1", "path": str(tmp_path / "image.tif")},
        {
            "type": "geotiff",
            "name": "t2",
            "path": str(tmp_path / "other.tif"),
            "bands": [4, 2, 1, 3],
        },
    ]
    separate = dataset_stats(_generated(tmp_path, scene, imagery, "separate", stack=False), "all")
    stacked = dataset_stats(_generated(tmp_path, scene, imagery, "stacked", stack=True), "all")
    assert stacked["sources"] == separate["sources"]
    assert set(stacked["sources"]) == {"t1", "t2"}
    staging = tmp_path / "separate"
    entries = _entries(staging, "all")
    for name in ("t1", "t2"):
        _assert_source(separate["sources"][name], _expected_moments(staging, entries, name))


def _generated(
    tmp_path: Path, scene: dict[str, Any], imagery: Any, staging: str, *, stack: bool
) -> Path:
    config = _config(
        tmp_path,
        imagery,
        scene["labels"],
        region=scene["region"],
        writer={
            "staging_dir": str(tmp_path / staging),
            "image_format": "npy",
            "stack_sources": stack,
        },
    )
    run_generate(config)
    return tmp_path / staging


def test_image_only_datasets_leave_out_nodata_pixels(tmp_path: Path) -> None:
    import rasterio

    ref = reference_transform()
    data = write_raster(tmp_path / "f.tif", ref, WIDTH, HEIGHT, count=2, dtype="float32", seed=2)
    data[:, :40, :] = -9999.0  # NoData rows
    data[1, 100:120, :] = np.nan  # one band missing: the pixel has no imagery
    with rasterio.open(tmp_path / "f.tif", "r+") as dst:
        dst.write(data)
        dst.nodata = -9999.0
    region = region_inside(ref, WIDTH, HEIGHT, margin=0.0)
    config = MapcvConfig.model_validate(
        {
            "region": region,
            "imagery": {"type": "geotiff", "path": str(tmp_path / "f.tif")},
            "sampler": {"patch_size": PATCH, "edge_strategy": "drop", "max_empty_ratio": 1.0},
            "writer": {"staging_dir": str(tmp_path / "dataset"), "image_format": "npy"},
        }
    )
    run_generate(config)
    staging = tmp_path / "dataset"
    stats = dataset_stats(staging)
    assert stats["split"] == "all"  # no split lists: every patch
    values = np.concatenate(
        [_image(staging / e["files"]["image"]).reshape(-1, 2) for e in _entries(staging, "all")]
    ).astype(np.float64)
    keep = np.all(np.isfinite(values), axis=1) & ~np.all(values == -9999.0, axis=1)
    assert (~keep).sum() > 0
    found = stats["sources"]["image"]
    np.testing.assert_allclose(found["mean"], values[keep].mean(axis=0), rtol=1e-12)
    np.testing.assert_allclose(found["std"], values[keep].std(axis=0), rtol=1e-10)
    assert "classes" not in stats


def _small_boxes(tmp_path: Path, region: dict[str, float]) -> Path:
    """Small squares spread over the region, each a few metres across."""
    west, south = region["west"], region["south"]
    dx, dy = region["east"] - west, region["north"] - south
    features = []
    for i in range(1, 8):
        for j in range(1, 6):
            x, y = west + dx * i / 8, south + dy * j / 6
            ring = [[x, y], [x + 1e-4, y], [x + 1e-4, y + 1e-4], [x, y + 1e-4], [x, y]]
            features.append(
                {
                    "type": "Feature",
                    "properties": {"kind": "a"},
                    "geometry": {"type": "Polygon", "coordinates": [ring]},
                }
            )
    path = tmp_path / "boxes.geojson"
    path.write_text(json.dumps({"type": "FeatureCollection", "features": features}))
    return path


def test_regression_targets_and_detection_objects(tmp_path: Path, scene: dict[str, Any]) -> None:
    import rasterio

    ref = reference_transform()
    values = write_raster(tmp_path / "v.tif", ref, WIDTH, HEIGHT, count=1, dtype="float32", seed=9)
    values[0, 50:60, :] = np.nan
    with rasterio.open(tmp_path / "v.tif", "r+") as dst:
        dst.write(values)
        dst.nodata = float("nan")
    region = region_inside(ref, WIDTH, HEIGHT)
    config = MapcvConfig.model_validate(
        {
            "task": "regression",
            "region": region,
            "imagery": {"type": "geotiff", "path": str(tmp_path / "image.tif")},
            "labels": {"type": "continuous", "path": str(tmp_path / "v.tif")},
            "sampler": {"patch_size": PATCH, "edge_strategy": "drop"},
            "writer": {"staging_dir": str(tmp_path / "values"), "image_format": "tif"},
        }
    )
    run_generate(config)
    staging = tmp_path / "values"
    stats = dataset_stats(staging, "all")
    targets = np.concatenate(
        [_image(staging / e["files"]["mask"]).ravel() for e in _entries(staging, "all")]
    )
    targets = targets[np.isfinite(targets)].astype(np.float64)
    assert stats["targets"]["mean"] == pytest.approx(targets.mean(), rel=1e-12)
    assert stats["targets"]["std"] == pytest.approx(targets.std(), rel=1e-10)
    assert stats["targets"]["pixels"] == targets.size
    assert "classes" not in stats

    boxes = MapcvConfig.model_validate(
        {
            "task": "detection",
            "region": region,
            "imagery": {"type": "geotiff", "path": str(tmp_path / "rgb.tif")},
            "labels": {
                "path": str(_small_boxes(tmp_path, region)),
                "label_field": "kind",
                "classes": {"a": 1},
            },
            "sampler": {"patch_size": PATCH, "edge_strategy": "drop"},
            "writer": {"staging_dir": str(tmp_path / "boxes")},
        }
    )
    run_generate(boxes)
    coco = next((tmp_path / "boxes").rglob("*.json"), None)
    found = dataset_stats(tmp_path / "boxes", "all")["classes"]["objects"]
    annotations = [
        a
        for path in (tmp_path / "boxes").rglob("*.json")
        if path.name != "manifest.json"
        for a in json.loads(path.read_text(encoding="utf-8")).get("annotations", [])
    ]
    assert coco is not None and found == {"a": len(annotations)} and annotations


def test_classification_balance_empty_splits_and_weight_edges(
    tmp_path: Path, scene: dict[str, Any]
) -> None:
    import csv

    from mapcv.stats import _median_frequency_weights

    region = region_inside(reference_transform(), WIDTH, HEIGHT)
    config = MapcvConfig.model_validate(
        {
            "task": "classification",
            "region": region,
            "imagery": {"type": "geotiff", "path": str(tmp_path / "rgb.tif")},
            "labels": {"path": str(scene["labels"]), "label_field": "kind", "classes": {"a": 1}},
            "classification": {"mode": "multi", "min_fraction": 0.0},
            "sampler": {"patch_size": PATCH, "edge_strategy": "drop"},
            "writer": {"staging_dir": str(tmp_path / "classes")},
        }
    )
    run_generate(config)
    staging = tmp_path / "classes"
    with (staging / "labels.csv").open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    expected = sum(1 for row in rows if "a" in row["labels"].split("|"))
    found = dataset_stats(staging, "all")["classes"]["patches"]
    assert found.get("a") == expected and expected > 0

    # A split without patches: no statistics, but no failure either.
    (staging / "splits").mkdir(exist_ok=True)
    (staging / "splits" / "val.txt").write_text("", encoding="utf-8")
    empty = dataset_stats(staging, "val")
    assert (empty["split"], empty["patches"], empty["sources"]) == ("val", 0, {})
    assert _median_frequency_weights({"1": 0}, {}) == {}


# ── verify ───────────────────────────────────────────────────────────────────


@pytest.fixture
def dataset(tmp_path: Path, scene: dict[str, Any]) -> Path:
    config = _config(
        tmp_path,
        {"type": "geotiff", "path": str(tmp_path / "image.tif")},
        scene["labels"],
        region=scene["region"],
        writer={"image_format": "npy"},
    )
    run_generate(config)
    return tmp_path / "dataset"


def test_a_fresh_dataset_verifies_deeply_and_with_checksums(dataset: Path) -> None:
    report = verify_dataset(dataset, deep=True)
    assert report.ok and not report.notes and report.checked_hashes == 0
    manifest = Manifest.load(dataset / "manifest.json")
    assert report.patches == len(manifest.patches) and report.files == 2 * report.patches

    write_checksums(dataset)
    lines = (dataset / CHECKSUMS_FILENAME).read_text(encoding="utf-8").splitlines()
    import hashlib

    for line in lines:
        digest, rel = line.split("  ")
        assert hashlib.sha256((dataset / rel).read_bytes()).hexdigest() == digest
    listed = {line.split("  ")[1] for line in lines}
    assert {"manifest.json", "splits/train.txt", "splits/split.json"} <= listed
    report = verify_dataset(dataset)
    assert report.ok and report.checked_hashes == len(lines)


def test_verify_finds_missing_empty_changed_and_extra_files(dataset: Path) -> None:
    write_checksums(dataset)
    manifest = Manifest.load(dataset / "manifest.json")
    first, second, third = (e["files"] for e in manifest.patches[:3])
    (dataset / first["image"]).unlink()
    (dataset / second["mask"]).write_bytes(b"")
    image = np.load(dataset / third["image"])
    np.save(dataset / third["image"], image + 1)
    shutil.copy(dataset / third["mask"], dataset / "Masks" / "stray.png")

    report = verify_dataset(dataset)
    assert not report.ok
    text = "\n".join(report.problems)
    assert f"{first['image']} is missing" in text
    assert f"{second['mask']} is empty" in text
    assert f"{third['image']} does not match its SHA256SUMS hash" in text
    assert f"{first['image']} is listed in SHA256SUMS but missing" in text
    assert report.notes and "Masks/stray.png" in report.notes[0]


def test_deep_verify_decodes_and_checks_shapes(dataset: Path) -> None:
    manifest = Manifest.load(dataset / "manifest.json")
    files = manifest.patches[0]["files"]
    assert verify_dataset(dataset, deep=True).ok
    np.save(dataset / files["image"], np.zeros((4, PATCH, PATCH - 1), dtype=np.uint16))
    (dataset / manifest.patches[1]["files"]["image"]).write_bytes(b"not an array")
    assert verify_dataset(dataset).ok  # only a deep check reads the files
    problems = verify_dataset(dataset, deep=True).problems
    assert len(problems) == 2
    assert "has shape (64, 63, 4) (H, W, C), the manifest says (64, 64, 4)" in problems[0]
    assert "cannot be read" in problems[1]


def test_verify_split_lists_and_missing_or_broken_manifests(dataset: Path, tmp_path: Path) -> None:
    with (dataset / "splits" / "val.txt").open("a") as handle:
        handle.write("ghost.npy\n")
    problems = verify_dataset(dataset).problems
    assert len(problems) == 1 and "splits/val.txt names 1 patch(es)" in problems[0]
    assert "ghost.npy" in problems[0]

    assert "no manifest.json" in verify_dataset(tmp_path / "nowhere").problems[0]
    (tmp_path / "broken").mkdir()
    (tmp_path / "broken" / "manifest.json").write_text("{not json")
    assert "cannot be read" in verify_dataset(tmp_path / "broken").problems[0]


@pytest.mark.parametrize("image_format", ["png", "jpg", "tif"])
def test_deep_verify_reads_every_format(
    tmp_path: Path, scene: dict[str, Any], image_format: str
) -> None:
    config = _config(
        tmp_path,
        {"type": "geotiff", "path": str(tmp_path / "rgb.tif")},
        scene["labels"],
        region=scene["region"],
        writer={
            "image_format": image_format,
            "mask_format": "tif" if image_format == "tif" else "png",
        },
    )
    run_generate(config)
    staging = tmp_path / "dataset"
    assert verify_dataset(staging, deep=True).ok
    entry = Manifest.load(staging / "manifest.json").patches[0]
    path = staging / entry["files"]["image"]
    if image_format == "tif":
        import rasterio

        with rasterio.open(path) as src:
            profile, data = src.profile, src.read()
        profile.update(width=PATCH // 2)
        with rasterio.open(path, "w", **profile) as dst:
            dst.write(data[:, :, : PATCH // 2])
    else:
        Image.new("RGB", (PATCH // 2, PATCH)).save(path, "JPEG" if image_format == "jpg" else "PNG")
    problems = verify_dataset(staging, deep=True).problems
    assert len(problems) == 1
    assert "has shape (64, 32, 3) (H, W, C), the manifest says (64, 64, 3)" in problems[0]


def test_checksums_without_split_lists_and_with_blank_lines(tmp_path: Path) -> None:
    from mapcv.verify import _check_image

    ref = reference_transform()
    write_raster(tmp_path / "rgb.tif", ref, WIDTH, HEIGHT, seed=7)
    config = MapcvConfig.model_validate(
        {
            "region": region_inside(ref, WIDTH, HEIGHT),
            "imagery": {"type": "geotiff", "path": str(tmp_path / "rgb.tif")},
            "sampler": {"patch_size": PATCH, "edge_strategy": "drop"},
            "writer": {"staging_dir": str(tmp_path / "plain")},
        }
    )
    run_generate(config)
    staging = tmp_path / "plain"
    assert not (staging / "splits").exists()
    sums = write_checksums(staging)
    listed = sums.read_text(encoding="utf-8").splitlines()
    names = {line.split("  ", 1)[1] for line in listed}
    assert {"manifest.json", "patches.geojson"} <= names
    sums.write_text("\n".join(listed) + "\n\n\n", encoding="utf-8")
    report = verify_dataset(staging)
    assert report.ok and report.checked_hashes == len(listed)
    # Shapes other than three dimensions are not compared.
    image = staging / Manifest.load(staging / "manifest.json").patches[0]["files"]["image"]
    assert _check_image(image, []) is None


# ── card ─────────────────────────────────────────────────────────────────────


def test_card_front_matter_and_sections(dataset: Path) -> None:
    text = card_text(dataset)
    _, front, body = text.split("---\n", 2)
    meta = yaml.safe_load(front)
    assert meta["license"] == "other" and meta["pretty_name"] == "dataset"
    assert meta["task_categories"] == ["image-segmentation"]
    assert meta["size_categories"] == ["n<1K"] and "remote-sensing" in meta["tags"]
    manifest = Manifest.load(dataset / "manifest.json")
    assert f"segmentation dataset of {len(manifest.patches)} patches of 64 × 64" in body
    assert (
        "| `image` | geotiff | image.tif | EPSG:32631 | ≈ 1.00 m | b1, b2, b3, b4 (uint16) |"
        in body
    )
    assert "| 0 | background |\n| 1 | a |\n| 255 | ignore" in body
    train = len(_names(dataset, "train"))
    assert f"| train | {train} |" in body
    assert "## Normalisation" not in body and "## Attribution" not in body
    assert manifest.target is not None and manifest.target.labels is not None
    assert manifest.target.labels["sha256"] in body

    write_stats(dataset)
    stats = json.loads((dataset / "stats.json").read_text(encoding="utf-8"))
    with_stats = card_text(dataset)
    mean = stats["sources"]["image"]["mean"][0]
    assert "## Normalisation" in with_stats and f"| `image` | b1 | {mean:.6g} |" in with_stats


def test_card_of_tile_detection_datasets(dataset: Path) -> None:
    data = json.loads((dataset / "manifest.json").read_text(encoding="utf-8"))
    data["task"] = "detection"
    data["target"]["ignore_index"] = None
    source = data["sources"][0]
    source.update(source_type="xyz", product_id="esri_satellite", crs=None, transform=None)
    source.pop("fingerprint", None)
    data["sources"].append({**source, "name": "carto", "product_id": "cartodb_positron"})
    (dataset / "manifest.json").write_text(json.dumps(data), encoding="utf-8")
    (dataset / "splits" / "test.txt").unlink()
    body = card_text(dataset)
    assert "task_categories:\n- object-detection" in body
    assert "| `image` | xyz | Esri World Imagery | ? | ? |" in body
    assert "| `carto` | xyz | CARTO Positron basemap |" in body
    assert "| 0 | background |" not in body and "ignore" not in body.split("## Splits")[0]
    assert "| test |" not in body and "fingerprint" not in body
    assert (
        "## Attribution\n\n- Source: Esri, Maxar" in body
        and "- © OpenStreetMap contributors © CARTO\n" in body
    )


def test_card_attribution_and_task_names() -> None:
    from mapcv.card import _attribution, _gsd, _size_category

    assert _gsd(SourceRecord()) is None
    broken = SourceRecord(crs="EPSG:999999", transform=(1.0, 0.0, 0.0, 0.0, -1.0, 0.0))
    assert _gsd(broken) is None

    esri = SourceRecord(source_type="xyz", product_id="esri_satellite")
    assert "Esri" in (_attribution(esri) or "")
    s2 = SourceRecord(source_type="eopf_zarr", product_id="S2B_MSIL2A_x")
    assert "Copernicus Sentinel" in (_attribution(s2) or "")
    assert _attribution(SourceRecord(source_type="geotiff", product_id="a.tif")) is None
    assert [_size_category(n) for n in (0, 999, 1000, 99_999, 100_000, 10**6)] == [
        "n<1K",
        "n<1K",
        "1K<n<10K",
        "10K<n<100K",
        "100K<n<1M",
        "n>1M",
    ]


# ── CLI ──────────────────────────────────────────────────────────────────────


def test_cli_commands(dataset: Path, tmp_path: Path) -> None:
    result = runner.invoke(app, ["stats", str(dataset)], env=ENV)
    assert result.exit_code == 0, result.output
    assert "b4" in result.output and "Class weights (median frequency): background" in result.output
    assert "split → " in result.output and (dataset / "stats.json").exists()
    result = runner.invoke(app, ["stats", str(dataset), "--split", "nope"], env=ENV)
    assert result.exit_code == 1 and "--split must be" in result.output
    result = runner.invoke(app, ["stats", str(tmp_path / "nowhere")], env=ENV)
    assert result.exit_code == 1 and "No manifest found" in result.output

    result = runner.invoke(app, ["card", str(dataset)], env=ENV)
    assert result.exit_code == 0, result.output
    assert (dataset / "README.md").read_text(encoding="utf-8") == card_text(dataset)
    result = runner.invoke(app, ["card", str(dataset)], env=ENV)
    assert result.exit_code == 1 and "--force" in result.output
    (dataset / "README.md").write_text("mine")
    assert runner.invoke(app, ["card", str(dataset), "--force"], env=ENV).exit_code == 0
    assert (dataset / "README.md").read_text(encoding="utf-8") != "mine"
    with pytest.raises(FileExistsError):
        write_card(dataset)
    result = runner.invoke(app, ["card", str(tmp_path / "nowhere")], env=ENV)
    assert result.exit_code == 1 and "No manifest found" in result.output

    result = runner.invoke(app, ["verify", str(dataset), "--deep", "--write-checksums"], env=ENV)
    assert result.exit_code == 0, result.output
    assert "file(s) present" in result.output and (dataset / CHECKSUMS_FILENAME).exists()
    result = runner.invoke(app, ["verify", str(dataset)], env=ENV)
    assert result.exit_code == 0 and "hash(es) match" in result.output
    manifest = Manifest.load(dataset / "manifest.json")
    for entry in manifest.patches[:22]:
        (dataset / entry["files"]["mask"]).unlink()
    shutil.copy(dataset / "manifest.json", dataset / "Images" / "extra.json")
    result = runner.invoke(app, ["verify", str(dataset), "--write-checksums"], env=ENV)
    assert result.exit_code == 1
    assert "is missing" in result.output and "… and 24 more" in result.output
    assert "Note:" in result.output and "Images/extra.json" in result.output
