"""``mapcv export --format webdataset|zarr``: the same patches in few large files.

Tar members must hold the patch files' bytes and one JSON record per sample, grouped by
split as the split lists say, within the shard size, and byte-identical across runs.
A Zarr export must hold the same arrays as the files, and ``MapcvDataset`` must read it
exactly as it reads the dataset folder.
"""

from __future__ import annotations

import json
import sys
import tarfile
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import pytest
from typer.testing import CliRunner

pytest.importorskip("rasterio", reason="these tests write their rasters with rasterio")

from mapcv.cli import app  # noqa: E402
from mapcv.data import MapcvDataset  # noqa: E402
from mapcv.manifest import Manifest  # noqa: E402
from mapcv.shards import export_webdataset, export_zarr  # noqa: E402
from test_data_and_export import HEIGHT, WIDTH, _generate  # noqa: E402
from test_multi_source import reference_transform, region_inside, write_labels, write_raster  # noqa: E402

runner = CliRunner()


@pytest.fixture
def scene(tmp_path: Path) -> Dict[str, Any]:
    ref = reference_transform()
    write_raster(tmp_path / "multi.tif", ref, WIDTH, HEIGHT, count=4, dtype="uint16", seed=1)
    write_raster(tmp_path / "other.tif", ref, WIDTH, HEIGHT, count=4, dtype="uint16", seed=2)
    write_raster(tmp_path / "rgb.tif", ref, WIDTH, HEIGHT, seed=3)
    region = region_inside(ref, WIDTH, HEIGHT, margin=0.0)
    return {"region": region, "labels": write_labels(tmp_path, region_inside(ref, WIDTH, HEIGHT))}


def _tar_samples(path: Path) -> Dict[str, Dict[str, bytes]]:
    samples: Dict[str, Dict[str, bytes]] = {}
    with tarfile.open(path) as tar:
        names = tar.getnames()
        for member in tar.getmembers():
            stem, _, field = member.name.partition(".")
            data = tar.extractfile(member)
            assert data is not None and member.mtime == 0 and member.uid == 0
            samples.setdefault(stem, {})[field] = data.read()
    # Each sample's members are contiguous (what WebDataset readers group on).
    runs = [
        stem
        for index, stem in enumerate(n.split(".")[0] for n in names)
        if index == 0 or names[index - 1].split(".")[0] != stem
    ]
    assert len(runs) == len(set(runs)) == len(samples)
    return samples


def test_webdataset_shards(tmp_path: Path, scene: Dict[str, Any]) -> None:
    rgb = {"type": "geotiff", "path": str(tmp_path / "rgb.tif")}
    root = _generate(
        tmp_path, scene, "d", imagery=rgb, writer={"image_format": "png", "mask_format": "png"}
    )
    manifest = Manifest.load(root / "manifest.json")
    limit = 20_000
    shards = export_webdataset(root, tmp_path / "wds", shard_bytes=limit)
    index = json.loads((tmp_path / "wds" / "shards.json").read_text())
    assert list(index["splits"]) == ["train", "val", "test"] and index["task"] == "segmentation"
    assert len(index["splits"]["train"]) > 1  # the limit forces several shards
    by_name = {Path(manifest.patch_name(e)).stem: e for e in manifest.patches}
    for split, listed in index["splits"].items():
        stems: List[str] = []
        for shard in listed:
            path = tmp_path / "wds" / shard["file"]
            samples = _tar_samples(path)
            assert len(samples) == shard["samples"]
            if shard["samples"] > 1:
                # The members (headers and padded data) fit the limit; tar adds its end
                # marker and pads the file to whole 10 KiB records on top.
                with tarfile.open(path) as tar:
                    used = sum(512 + -(-m.size // 512) * 512 for m in tar.getmembers())
                assert used <= limit
            for stem, fields in samples.items():
                entry = by_name[stem]
                assert set(fields) == {"png", "mask.png", "json"}
                assert fields["png"] == (root / entry["files"]["image"]).read_bytes()
                assert fields["mask.png"] == (root / entry["files"]["mask"]).read_bytes()
                record = json.loads(fields["json"])
                assert record["split"] == split and (record["row"], record["col"]) == (
                    entry["row"],
                    entry["col"],
                )
                assert record["summary"] == entry["summary"]
                assert record["transform"] == pytest.approx(list(manifest.patch_transform(entry)))
                stems.append(stem)
        listed_names = (root / "splits" / f"{split}.txt").read_text().split()
        assert stems == [Path(n).stem for n in listed_names]
    again = export_webdataset(root, tmp_path / "wds2", shard_bytes=limit)
    assert [p.read_bytes() for p in shards] == [p.read_bytes() for p in again]  # deterministic
    assert len(export_webdataset(root, tmp_path / "one")) == 3  # one shard per split by default


def test_webdataset_sources_and_boxes(tmp_path: Path, scene: Dict[str, Any]) -> None:
    sources = [
        {"type": "geotiff", "name": "t1", "path": str(tmp_path / "multi.tif")},
        {"type": "geotiff", "name": "t2", "path": str(tmp_path / "other.tif")},
    ]
    root = _generate(tmp_path, scene, "two", imagery=sources, split=None)
    export_webdataset(root, tmp_path / "wds")
    samples = _tar_samples(tmp_path / "wds" / "all-000000.tar")
    first = next(iter(samples.values()))
    assert set(first) == {"tif", "t2.tif", "mask.tif", "json"}

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
        labels=small,
        imagery={"type": "geotiff", "path": str(tmp_path / "rgb.tif")},
        writer={"image_format": "png"},
    )
    export_webdataset(boxes, tmp_path / "boxes-wds")
    expected = {
        item["name"]: item["annotations"] for item in MapcvDataset(boxes, "all", as_tensors=False)
    }
    found = 0
    for shard in sorted((tmp_path / "boxes-wds").glob("*.tar")):
        for stem, fields in _tar_samples(shard).items():
            record = json.loads(fields["json"])
            assert record["annotations"] == expected[record["name"]]
            found += len(record["annotations"])
    assert found > 0


@pytest.mark.parametrize("image_format", ["tif", "npy"])
def test_zarr_store_reads_like_the_folder(
    tmp_path: Path, scene: Dict[str, Any], image_format: str
) -> None:
    zarr = pytest.importorskip("zarr")
    root = _generate(
        tmp_path, scene, "d", writer={"image_format": image_format, "mask_format": image_format}
    )
    out = export_zarr(root, tmp_path / "d.zarr")
    group = zarr.open_group(str(out), mode="r")
    manifest = Manifest.load(root / "manifest.json")
    assert group["images"]["image"].shape[0] == len(manifest.patches)
    assert group["images"]["image"].chunks[0] == 1
    for split in ("train", "val", "test"):
        folder = MapcvDataset(root, split, as_tensors=False)
        store = MapcvDataset(out, split, as_tensors=False)
        assert len(store) == len(folder) > 0
        for a, b in zip(folder, store):
            assert a["name"] == b["name"] and (a["row"], a["col"]) == (b["row"], b["col"])
            np.testing.assert_array_equal(a["image"], b["image"])
            np.testing.assert_array_equal(a["mask"], b["mask"])
    assert len(MapcvDataset(out, "all", as_tensors=False)) == len(manifest.patches)
    codes = np.asarray(group["split"][:])
    assert set(codes.tolist()) == {0, 1, 2}
    np.testing.assert_array_equal(group["row"][:], [e["row"] for e in manifest.patches])
    labeled = MapcvDataset(out, "10/labeled", as_tensors=False)
    assert [i["name"] for i in labeled] == (
        root / "splits" / "10" / "labeled.txt"
    ).read_text().split()


def test_zarr_boxes_and_refusals(
    tmp_path: Path, scene: Dict[str, Any], monkeypatch: pytest.MonkeyPatch
) -> None:
    pytest.importorskip("zarr")
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
        labels=small,
        split=None,
        imagery={"type": "geotiff", "path": str(tmp_path / "rgb.tif")},
        writer={"image_format": "png"},
    )
    out = export_zarr(boxes, tmp_path / "boxes.zarr")
    folder = MapcvDataset(boxes, "all", as_tensors=False)
    store = MapcvDataset(out, "all", as_tensors=False)
    for a, b in zip(folder, store):
        np.testing.assert_array_equal(a["boxes"], b["boxes"])
        np.testing.assert_array_equal(a["image"], b["image"])
    with pytest.raises(ValueError, match="outside the dataset folder"):
        export_zarr(boxes, boxes / "inside.zarr")
    with pytest.raises(ValueError, match="outside the dataset folder"):
        export_webdataset(boxes, boxes)
    with pytest.raises(ValueError, match="positive"):
        export_webdataset(boxes, tmp_path / "x", shard_bytes=0)
    monkeypatch.setitem(sys.modules, "zarr", None)
    with pytest.raises(RuntimeError, match=r"mapcv\[zarr\]"):
        export_zarr(boxes, tmp_path / "again.zarr")


def test_export_cli_for_shards(tmp_path: Path, scene: Dict[str, Any]) -> None:
    root = _generate(tmp_path, scene, "d")
    env = {"COLUMNS": "200"}
    result = runner.invoke(
        app,
        ["export", str(root), "-f", "webdataset", "-o", str(tmp_path / "w"), "--shard-mb", "1"],
        env=env,
    )
    assert result.exit_code == 0, result.output
    assert "tar shard(s) and shards.json" in result.output
    result = runner.invoke(app, ["export", str(root), "-f", "zarr"], env=env)
    assert result.exit_code == 1 and "--out is required for zarr" in result.output
    pytest.importorskip("zarr")
    result = runner.invoke(
        app, ["export", str(root), "-f", "zarr", "-o", str(tmp_path / "z.zarr")], env=env
    )
    assert result.exit_code == 0 and "Zarr store written" in result.output
