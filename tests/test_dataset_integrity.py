"""A dataset is one consistent whole: resume, interruptions, concurrent runs and checks.

Resuming refuses imagery or a region other than the dataset's; an interrupted
``generate`` or ``split`` leaves a dataset that says so; two runs never write one folder
at once; ``verify`` finds cut-short files and wrong masks; exports never mix with what
was in their folder before; patches without any imagery are left out.
"""

from __future__ import annotations

import io
import json
import os
import shutil
import subprocess
import sys
import threading
import warnings
from collections.abc import Iterator
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any

import numpy as np
import pytest
from PIL import Image
from typer.testing import CliRunner

pytest.importorskip("rasterio", reason="these tests write their rasters with rasterio")
import rasterio
from rasterio.transform import Affine
from test_geotiff_imagery import config_for, make_raster, write_labels

from mapcv import pipeline
from mapcv.agent_tools import Sandbox, ToolState
from mapcv.agent_tools import info as mcp_info
from mapcv.cli import app
from mapcv.config import MapcvConfig
from mapcv.imagery import geotiff_fingerprint
from mapcv.locking import LOCK_FILENAME, DatasetBusyError, StagingDirError, dataset_lock
from mapcv.manifest import Manifest, ManifestMismatchError
from mapcv.pipeline import run_generate, run_split
from mapcv.shards import export_webdataset, export_zarr
from mapcv.splitter import SplitterConfig
from mapcv.verify import CHECKSUMS_FILENAME, verify_dataset, write_checksums

runner = CliRunner()


def _tree(root: Path, *folders: str) -> dict[str, bytes]:
    """Every file under ``folders`` of ``root`` (all of it without folders)."""
    bases = [root / folder for folder in folders] if folders else [root]
    return {
        path.relative_to(root).as_posix(): path.read_bytes()
        for base in bases
        for path in sorted(base.rglob("*"))
        if path.is_file()
    }


def _labeled(tmp_path: Path, **kwargs: Any) -> MapcvConfig:
    raster = make_raster(tmp_path)
    region = raster.region()
    labels = write_labels(tmp_path, region)
    return config_for(
        tmp_path, {"path": str(raster.path), "chunk_rows": 64}, region, labels=labels, **kwargs
    )


class _Stop(Exception):
    pass


def _interrupt_after(config: MapcvConfig, chunks: int) -> None:
    """Run ``generate`` and stop it, as Ctrl-C would, after ``chunks`` chunks."""

    def stop(done: int, total: int) -> None:
        if done == chunks:
            raise _Stop

    with pytest.raises(_Stop):
        run_generate(config, stop)


# ── A-6: a URL template on the same host ─────────────────────────────────────


class _Tiles(BaseHTTPRequestHandler):
    def log_message(self, *args: object) -> None:
        pass

    def do_GET(self) -> None:
        parts = self.path.split("?")[0].strip("/").split("/")
        z, x, y = (int(part.split(".")[0]) for part in parts[-3:])
        rng = np.random.default_rng([len(parts[0]), z, x, y])
        buffer = io.BytesIO()
        Image.fromarray(rng.integers(1, 256, (256, 256, 3), dtype=np.uint8)).save(buffer, "PNG")
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


def _xyz(tmp_path: Path, template: str) -> MapcvConfig:
    return MapcvConfig.model_validate(
        {
            "region": {"west": 4.8900, "south": 52.3700, "east": 4.8960, "north": 52.3740},
            "imagery": {"type": "xyz", "zoom": 17, "url_template": template, "strip_rows": 1},
            "sampler": {"patch_size": 256},
            "writer": {"staging_dir": str(tmp_path / "dataset")},
        }
    )


def test_resume_refuses_another_url_template_on_the_same_host(
    tmp_path: Path, tile_port: int
) -> None:
    base = f"http://127.0.0.1:{tile_port}"
    first = _xyz(tmp_path, base + "/layer2019/{z}/{x}/{y}.png?key=SECRET-KEY-123")
    _interrupt_after(first, 1)
    text = (tmp_path / "dataset" / "manifest.json").read_text(encoding="utf-8")
    assert "SECRET-KEY-123" not in text and "layer2019" not in text

    other_layer = _xyz(tmp_path, base + "/layer2023/{z}/{x}/{y}.png?key=SECRET-KEY-123")
    with pytest.raises(ManifestMismatchError, match="URL template"):
        run_generate(other_layer)
    result = run_generate(first)  # the same template resumes
    assert result.new_patches > 0 and result.manifest.complete is True

    # A dataset made before the template was recorded still resumes.
    _interrupt_after(_xyz(tmp_path / "old", base + "/a/{z}/{x}/{y}.png"), 1)
    path = tmp_path / "old" / "dataset" / "manifest.json"
    data = json.loads(path.read_text(encoding="utf-8"))
    assert data["sources"][0].pop("fingerprint")["url_template_pbkdf2"]
    path.write_text(json.dumps(data), encoding="utf-8")
    assert run_generate(_xyz(tmp_path / "old", base + "/a/{z}/{x}/{y}.png")).new_patches > 0


# ── B-8: a GeoTIFF edited in place ───────────────────────────────────────────


def _striped_raster(path: Path, seed: int = 3) -> dict[str, float]:
    """An uncompressed, striped GeoTIFF: editing its pixels keeps its size and both ends."""
    width, height = 256, 512
    transform = Affine(1.0, 0.0, 448_000.0, 0.0, -1.0, 5_412_000.0)
    data = np.random.default_rng(seed).integers(1, 250, (3, height, width), dtype=np.uint8)
    profile: dict[str, Any] = {
        "driver": "GTiff",
        "width": width,
        "height": height,
        "count": 3,
        "dtype": "uint8",
        "crs": "EPSG:32631",
        "transform": transform,
    }
    with rasterio.open(path, "w", **profile) as dst:
        dst.write(data)
    from pyproj import Transformer

    to_lonlat = Transformer.from_crs("EPSG:32631", "EPSG:4326", always_xy=True)
    west, north = to_lonlat.transform(448_000.0 + 1, 5_412_000.0 - 1)
    east, south = to_lonlat.transform(448_000.0 + width - 1, 5_412_000.0 - height + 1)
    return {"west": west, "south": south, "east": east, "north": north}


def test_resume_refuses_a_geotiff_whose_pixels_changed_in_place(tmp_path: Path) -> None:
    path = tmp_path / "ortho.tif"
    region = _striped_raster(path)
    config = config_for(tmp_path, {"path": str(path), "chunk_rows": 64}, region)
    _interrupt_after(config, 2)
    before = geotiff_fingerprint(str(path))
    mtime = path.stat().st_mtime_ns

    with rasterio.open(path, "r+") as dst:  # a cloud painted over, in the middle
        dst.write(np.zeros((3, 100, 100), dtype=np.uint8), window=((200, 300), (50, 150)))
    os.utime(path, ns=(mtime + 10**9, mtime + 10**9))
    with pytest.raises(ManifestMismatchError, match="modification time"):
        run_generate(config)
    # Size and both ends of the file are unchanged: only the modification time tells.
    after = geotiff_fingerprint(str(path))
    for key in ("size", "sha256_head_tail"):
        assert after[key] == before[key]


# ── D-4: a smaller region ────────────────────────────────────────────────────


def test_resume_refuses_another_region_on_the_same_grid(tmp_path: Path) -> None:
    raster = make_raster(tmp_path)
    region = raster.region(margin=0.0)
    run_generate(config_for(tmp_path, {"path": str(raster.path)}, region))
    # Same west and north edges, so the same origin and pixel grid, but a narrower raster.
    smaller = {**region, "east": region["west"] + (region["east"] - region["west"]) / 2}
    with pytest.raises(ManifestMismatchError, match="region"):
        run_generate(config_for(tmp_path, {"path": str(raster.path)}, smaller))

    # A manifest that predates the raster size resumes as before.
    path = tmp_path / "dataset" / "manifest.json"
    data = json.loads(path.read_text(encoding="utf-8"))
    for key in ("width", "height"):
        del data["sources"][0][key]
    path.write_text(json.dumps(data), encoding="utf-8")
    assert run_generate(config_for(tmp_path, {"path": str(raster.path)}, region)).new_patches == 0


# ── D-2: one writer per folder ───────────────────────────────────────────────


def test_a_second_run_into_a_busy_folder_is_refused(tmp_path: Path) -> None:
    config = _labeled(tmp_path)
    staging = config.writer.staging_dir
    staging.mkdir()
    with (
        dataset_lock(staging),
        pytest.raises(DatasetBusyError, match="another mapcv command is writing"),
    ):
        run_generate(config)
    assert not (staging / LOCK_FILENAME).exists()
    # A lock file left by a crashed run holds no lock: the next run goes ahead.
    (staging / LOCK_FILENAME).write_text("12345\n")
    run_generate(config)
    assert not (staging / LOCK_FILENAME).exists()
    with dataset_lock(staging), pytest.raises(DatasetBusyError):
        run_split(staging)


_HOLD_LOCK = """
import sys
from pathlib import Path
from mapcv.locking import dataset_lock
with dataset_lock(Path(sys.argv[1])):
    print("locked", flush=True)
    sys.stdin.readline()
"""


def test_the_lock_holds_across_processes(tmp_path: Path) -> None:
    staging = tmp_path / "dataset"
    staging.mkdir()
    holder = subprocess.Popen(
        [sys.executable, "-c", _HOLD_LOCK, str(staging)],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        text=True,
    )
    assert holder.stdin is not None and holder.stdout is not None
    try:
        assert holder.stdout.readline().strip() == "locked"
        # Windows locks the file's first byte, so the holder's process ID cannot be read.
        named = "" if sys.platform == "win32" else f" \\(process {holder.pid}\\)"
        with pytest.raises(DatasetBusyError, match=f"writing to .*{named}"), dataset_lock(staging):
            pass
    finally:
        holder.stdin.close()
        holder.wait(30)
    assert holder.returncode == 0
    with dataset_lock(staging):
        pass


# ── E-8: a folder with files of its own ──────────────────────────────────────


def test_generate_refuses_a_folder_of_other_files(tmp_path: Path) -> None:
    config = _labeled(tmp_path, staging="mine")
    staging = config.writer.staging_dir
    staging.mkdir()
    (staging / "dataset.yaml").write_text("my own dataset\n")
    (staging / "train.txt").write_text("a.png\n")
    with pytest.raises(StagingDirError, match="first: dataset.yaml"):
        run_generate(config)
    assert (staging / "dataset.yaml").read_text() == "my own dataset\n"
    assert sorted(path.name for path in staging.iterdir()) == ["dataset.yaml", "train.txt"]

    # A first run stopped before its manifest left only patch folders: that goes on.
    shutil.rmtree(staging)
    (staging / "Images").mkdir(parents=True)
    (staging / "Images" / "patch_0000000.png").write_bytes(b"partial")
    assert run_generate(config).manifest.patches


# ── M-12, F11, F29: an interrupted generate says so ──────────────────────────


def test_an_interrupted_generate_is_reported_incomplete_until_finished(tmp_path: Path) -> None:
    config = _labeled(tmp_path)
    staging = config.writer.staging_dir
    _interrupt_after(config, 2)
    manifest = Manifest.load(staging / "manifest.json")
    assert manifest.complete is False and manifest.patches

    report = verify_dataset(staging)
    assert report.incomplete and "incomplete" in report.problems[0]
    result = runner.invoke(app, ["verify", str(staging)], env={"COLUMNS": "200"})
    assert result.exit_code == 1 and "mapcv generate again" in result.output
    result = runner.invoke(app, ["info", str(staging)], env={"COLUMNS": "200"})
    assert result.exit_code == 0 and "incomplete" in result.output
    tool = mcp_info(ToolState(Sandbox(tmp_path)), "dataset")
    assert tool.data["complete"] is False and "Incomplete" in tool.summary

    finished = run_generate(config)
    assert finished.manifest.complete is True
    assert Manifest.load(staging / "manifest.json").complete is True
    assert verify_dataset(staging).ok
    assert mcp_info(ToolState(Sandbox(tmp_path)), "dataset").data["complete"] is True
    (tmp_path / "ref").mkdir()
    reference = run_generate(_labeled(tmp_path / "ref"))
    assert _tree(staging, "Images", "Masks", "splits") == _tree(
        reference.staging_dir, "Images", "Masks", "splits"
    )


def test_a_failure_while_splitting_leaves_the_dataset_incomplete(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config = _labeled(tmp_path)
    config.split = SplitterConfig()

    def fail(*args: Any, **kwargs: Any) -> Any:
        raise OSError("disk full")

    monkeypatch.setattr(pipeline, "split_manifest", fail)
    with pytest.raises(OSError, match="disk full"):
        run_generate(config)
    manifest = Manifest.load(config.writer.staging_dir / "manifest.json")
    assert manifest.complete is False and manifest.patches
    monkeypatch.undo()
    result = run_generate(config)
    assert result.new_patches == 0 and result.split_counts is not None
    assert Manifest.load(config.writer.staging_dir / "manifest.json").complete is True


def test_a_finished_old_dataset_is_left_as_it_is(tmp_path: Path) -> None:
    config = _labeled(tmp_path)
    run_generate(config)
    path = config.writer.staging_dir / "manifest.json"
    data = json.loads(path.read_text(encoding="utf-8"))
    del data["complete"]  # as written before mapcv 0.3
    path.write_text(json.dumps(data), encoding="utf-8")
    before = path.read_bytes()
    assert run_generate(config).new_patches == 0
    assert path.read_bytes() == before
    assert verify_dataset(config.writer.staging_dir).ok


# ── D-14: files lost after the manifest listed them ──────────────────────────


def test_resume_writes_missing_and_empty_patch_files_again(tmp_path: Path) -> None:
    config = _labeled(tmp_path)
    staging = config.writer.staging_dir
    run_generate(config)
    clean = _tree(staging, "Images", "Masks")
    manifest = Manifest.load(staging / "manifest.json")
    late = manifest.patches[-1]["files"]
    (staging / manifest.patches[3]["files"]["image"]).unlink()
    (staging / late["mask"]).write_bytes(b"")

    with pytest.warns(UserWarning, match="missing or empty"):
        result = run_generate(config)
    assert result.new_patches > 0
    assert _tree(staging, "Images", "Masks") == clean
    assert Manifest.load(staging / "manifest.json").patches == manifest.patches
    assert verify_dataset(staging, deep=True).ok


# ── D-6: an interrupted split ────────────────────────────────────────────────


def test_an_interrupted_split_leaves_the_old_lists_or_says_they_are_unfinished(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config = _labeled(tmp_path)
    config.split = SplitterConfig()
    staging = config.writer.staging_dir
    run_generate(config)
    before = _tree(staging, "splits")

    # A full disk while the lists are written: nothing is replaced.
    original_write = Path.write_text

    def full_disk(self: Path, *args: Any, **kwargs: Any) -> int:
        if self.name == "train.txt.tmp":
            raise OSError(28, "No space left on device")
        return original_write(self, *args, **kwargs)

    monkeypatch.setattr(Path, "write_text", full_disk)
    with pytest.raises(OSError, match="No space"):
        run_split(staging, SplitterConfig(strategy="random", seed=9))
    monkeypatch.undo()
    assert _tree(staging, "splits") == before
    assert verify_dataset(staging).ok

    # Stopped while the lists are moved into place: verify says they are unfinished.
    original_replace = os.replace
    moved: list[str] = []

    def stop_after_one(src: Any, dst: Any) -> None:
        if moved:
            raise KeyboardInterrupt
        moved.append(str(dst))
        original_replace(src, dst)

    monkeypatch.setattr(os, "replace", stop_after_one)
    with pytest.raises(KeyboardInterrupt):
        run_split(staging, SplitterConfig(strategy="random", seed=9))
    monkeypatch.undo()
    assert not (staging / "splits" / "split.json").exists()
    problems = verify_dataset(staging).problems
    assert any("split.json is missing" in problem for problem in problems)

    run_split(staging, SplitterConfig(strategy="random", seed=9))
    assert verify_dataset(staging).ok
    assert not list((staging / "splits").rglob("*.tmp"))


def test_verify_finds_split_lists_that_share_patches(tmp_path: Path) -> None:
    config = _labeled(tmp_path)
    config.split = SplitterConfig(strategy="random")
    staging = config.writer.staging_dir
    run_generate(config)
    splits = staging / "splits"
    test = (splits / "test.txt").read_text().split()
    train = (splits / "train.txt").read_text().split()
    (splits / "train.txt").write_text("".join(f"{n}\n" for n in train[:-1] + test[:1]))
    problems = verify_dataset(staging).problems
    assert any("in both splits/train.txt and splits/test.txt" in p for p in problems)


# ── D-13: cut-short files and wrong masks ────────────────────────────────────


def test_verify_finds_cut_short_files_and_wrong_masks(tmp_path: Path) -> None:
    config = _labeled(tmp_path)
    staging = config.writer.staging_dir
    run_generate(config)
    manifest = Manifest.load(staging / "manifest.json")
    files = [entry["files"] for entry in manifest.patches]
    assert verify_dataset(staging, deep=True).ok

    image = staging / files[1]["image"]
    image.write_bytes(image.read_bytes()[: image.stat().st_size // 2])
    mask = staging / files[4]["mask"]
    mask.write_bytes(mask.read_bytes()[: mask.stat().st_size // 2])
    Image.fromarray(np.zeros((10, 10), dtype=np.uint8)).save(staging / files[3]["mask"])
    Image.fromarray(np.full((64, 64), 200, dtype=np.uint8)).save(staging / files[5]["mask"])
    # One pixel of another class: the pixel counts per class tell.
    changed = np.asarray(Image.open(staging / files[6]["mask"])).copy()
    changed[0, 0] = 1 if changed[0, 0] != 1 else 2
    Image.fromarray(changed).save(staging / files[6]["mask"])

    plain = verify_dataset(staging).problems
    assert plain == [
        f"{files[1]['image']} is cut short (no PNG end chunk)",
        f"{files[4]['mask']} is cut short (no PNG end chunk)",
    ]
    deep = "\n".join(verify_dataset(staging, deep=True).problems)
    assert f"{files[3]['mask']} is 10x10 pixels, the manifest says 64x64" in deep
    assert f"{files[5]['mask']} holds value(s) 200" in deep
    assert files[6]["mask"] in deep


# ── D-11: verify after mapcv split ───────────────────────────────────────────


def test_verify_names_the_files_mapcv_split_rewrote(tmp_path: Path) -> None:
    config = _labeled(tmp_path)
    config.split = SplitterConfig()
    staging = config.writer.staging_dir
    run_generate(config)
    write_checksums(staging)
    run_split(staging, SplitterConfig(strategy="random", seed=3))
    report = verify_dataset(staging)
    assert report.rewritten and len(report.rewritten) == len(report.problems)
    assert "splits/train.txt" in report.rewritten
    result = runner.invoke(app, ["verify", str(staging)], env={"COLUMNS": "200"})
    assert result.exit_code == 1
    assert "mapcv split, stats and card rewrite it" in result.output
    assert "--write-checksums" in result.output and "Copy the dataset again" not in result.output

    # A changed patch is not one of them.
    image = staging / Manifest.load(staging / "manifest.json").patches[0]["files"]["image"]
    pixels = np.asarray(Image.open(image)).copy()
    pixels[0, 0] ^= 1
    Image.fromarray(pixels).save(image)
    result = runner.invoke(app, ["verify", str(staging)], env={"COLUMNS": "200"})
    assert f"does not match its {CHECKSUMS_FILENAME} hash" in result.output
    assert "Copy the dataset again" in result.output


# ── D-12: a staging_dir inside the mosaic's folder ───────────────────────────


def test_a_mosaic_glob_never_reads_the_dataset_written_inside_it(tmp_path: Path) -> None:
    survey = tmp_path / "survey"
    survey.mkdir()
    raster = make_raster(survey, name="a.tif")
    region = raster.region()
    data = {
        "region": region,
        "imagery": {"type": "geotiff", "path": str(survey / "**" / "*.tif")},
        "sampler": {"patch_size": 64, "edge_strategy": "drop"},
        "writer": {"staging_dir": str(survey / "out"), "image_format": "tif"},
    }
    first = run_generate(MapcvConfig.model_validate(data))
    assert list((survey / "out" / "Images").glob("*.tif"))
    config = MapcvConfig.model_validate(data)
    assert config.primary_imagery.files() == [str(survey / "a.tif")]  # type: ignore[union-attr]
    again = run_generate(config)
    assert again.new_patches == 0 and len(again.manifest.patches) == len(first.manifest.patches)


# ── #238: patches without imagery ────────────────────────────────────────────


def test_patches_without_any_imagery_are_left_out(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    # NoData over a whole 64 px patch (rows 64-127, cols 64-127) and part of another.
    raster = make_raster(tmp_path, nodata=0, block=(64, 128, 64, 160))
    region = raster.region(margin=0.0)
    config = config_for(tmp_path, {"path": str(raster.path)}, region, max_empty_ratio=1.0)
    with caplog.at_level("INFO", logger="mapcv"):
        result = run_generate(config)
    anchors = {(entry["row"], entry["col"]) for entry in result.manifest.patches}
    assert (64, 64) not in anchors  # no imagery at all
    assert (64, 128) in anchors  # half of it has imagery: max_empty_ratio decides
    assert result.patches_without_imagery == 1
    assert "Left out 1 patch without any imagery" in caplog.text
    assert all(entry["summary"]["empty_ratio"] < 1.0 for entry in result.manifest.patches)


def test_the_failed_tile_warning_describes_the_masks(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from test_pipeline import FakeRasterSource
    from test_pipeline import _config as fake_config

    def open_source(*args: Any, **kwargs: Any) -> FakeRasterSource:
        source = FakeRasterSource()
        source.tiles_requested = 10  # type: ignore[attr-defined]
        source.tiles_failed = 1  # type: ignore[attr-defined]
        return source

    monkeypatch.setattr("mapcv.pipeline.open_raster_source", open_source)
    config = fake_config(tmp_path)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        run_generate(config)
    text = " ".join(str(w.message) for w in caught)
    assert "filled with black" in text and "labels over black pixels" not in text


# ── D-9, D-10: exports into a used folder ────────────────────────────────────


def test_webdataset_export_replaces_an_earlier_export_and_refuses_other_files(
    tmp_path: Path,
) -> None:
    config = _labeled(tmp_path)
    config.split = SplitterConfig()
    run_generate(config)
    staging = config.writer.staging_dir
    out = tmp_path / "wds"
    many = export_webdataset(staging, out, shard_bytes=20_000)
    assert len(many) > 3
    few = export_webdataset(staging, out)
    assert sorted(path.name for path in out.iterdir()) == sorted(
        [path.name for path in few] + ["shards.json"]
    )

    mine = tmp_path / "mine"
    mine.mkdir()
    (mine / "notes.txt").write_text("keep me")
    with pytest.raises(ValueError, match="not empty"):
        export_webdataset(staging, mine)
    assert sorted(path.name for path in mine.iterdir()) == ["notes.txt"]


def test_zarr_export_needs_zarr_2_and_cleans_up_after_a_failure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    zarr = pytest.importorskip("zarr")
    config = _labeled(tmp_path)
    run_generate(config)
    staging = config.writer.staging_dir
    monkeypatch.setattr(zarr, "__version__", "3.0.8")
    with pytest.raises(RuntimeError, match="needs zarr 2.x"):
        export_zarr(staging, tmp_path / "z3.zarr")
    assert not (tmp_path / "z3.zarr").exists()
    monkeypatch.undo()

    def broken(path: Path) -> Any:
        raise OSError("read error")

    monkeypatch.setattr("mapcv.shards.read_mask", broken)
    with pytest.raises(OSError, match="read error"):
        export_zarr(staging, tmp_path / "half.zarr")
    assert not (tmp_path / "half.zarr").exists()
    monkeypatch.undo()
    out = export_zarr(staging, tmp_path / "ok.zarr")
    assert export_zarr(staging, out) == out  # an earlier export is replaced
