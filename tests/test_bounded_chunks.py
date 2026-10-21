"""Chunk windows stay below a memory budget, and an interrupted run saves what it finished.

A wide raster is read in column windows once a chunk's window would pass
``pipeline._WINDOW_BYTES``; the patches must not change. The manifest is saved every
few seconds and when the run stops, keeping only the entries of finished chunks.
"""

from __future__ import annotations

from contextlib import nullcontext
from pathlib import Path
from typing import Any

import numpy as np
import numpy.typing as npt
import pytest

pytest.importorskip("rasterio", reason="these tests write their rasters with rasterio")

from test_geotiff_imagery import config_for, make_raster, write_labels

from mapcv import pipeline
from mapcv.manifest import Manifest
from mapcv.pipeline import _column_windows, run_generate
from mapcv.planning import plan
from mapcv.writers import create_writer

PATCH = 64


def _files(staging: Path) -> dict[str, bytes]:
    return {
        path.relative_to(staging).as_posix(): path.read_bytes()
        for folder in ("Images", "Masks")
        for path in sorted((staging / folder).iterdir())
    }


def _wide_config(tmp_path: Path, staging: str = "dataset", **sampler: Any) -> Any:
    raster = make_raster(tmp_path, width=1280, height=256, count=3)
    region = raster.region(margin=0.01)
    labels = write_labels(tmp_path, region)
    return config_for(
        tmp_path,
        {"path": str(raster.path), "chunk_rows": 128},
        region,
        labels=labels,
        staging=staging,
        **sampler,
    )


def _windows(monkeypatch: pytest.MonkeyPatch) -> list[tuple[int, int]]:
    """Record (rows, cols) of every chunk window the pipeline reads."""
    seen: list[tuple[int, int]] = []
    original = pipeline._process_anchor_chunk

    def record(source: Any, group: list[tuple[int, int]], *args: Any) -> Any:
        rows = max(r for r, _ in group) + PATCH - min(r for r, _ in group)
        cols = max(c for _, c in group) + PATCH - min(c for _, c in group)
        seen.append((rows, cols))
        return original(source, group, *args)

    monkeypatch.setattr(pipeline, "_process_anchor_chunk", record)
    return seen


def _cols(*values: int) -> npt.NDArray[np.uint64]:
    return np.array(values, dtype=np.uint64)


def test_column_windows_keep_a_small_group_whole() -> None:
    assert _column_windows(0, 64, _cols(*range(0, 1024, 64)), PATCH, pixel_bytes=3) is None


def test_column_windows_split_a_wide_group_in_order(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(pipeline, "_WINDOW_BYTES", 128 * 300 * 3)
    cols = _cols(*range(0, 1024, 64))
    keys = _column_windows(0, 64, cols, PATCH, pixel_bytes=3)
    assert keys is not None and len(set(keys.tolist())) > 1
    # Windows are numbered left to right, and each stays within the budget.
    assert keys.tolist() == sorted(keys.tolist())
    for key in set(keys.tolist()):
        part = cols[keys == key]
        assert 128 * (int(part.max()) + PATCH - int(part.min())) * 3 <= pipeline._WINDOW_BYTES


def test_column_windows_never_go_below_two_patches(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(pipeline, "_WINDOW_BYTES", 1)
    cols = _cols(*range(0, 512, 32))
    keys = _column_windows(0, 0, cols, PATCH, pixel_bytes=3)
    assert keys is not None
    for key in set(keys.tolist()):
        part = cols[keys == key]
        assert int(part.max()) + PATCH - int(part.min()) <= 2 * PATCH


# ── The compact chunks are the chunks of the anchor lists they replaced ─────────────


def _list_windows(group: list[tuple[int, int]], pixel_bytes: int) -> list[list[tuple[int, int]]]:
    """How mapcv 0.3.0 split a group of anchors into column windows."""
    rows = max(r for r, _ in group) + PATCH - min(r for r, _ in group)
    first = min(c for _, c in group)
    width = max(c for _, c in group) + PATCH - first
    max_width = pipeline._max_window_width(rows, width, pixel_bytes, PATCH)
    if max_width >= width:
        return [group]
    split: dict[int, list[tuple[int, int]]] = {}
    for anchor in group:
        split.setdefault((anchor[1] - first) // (max_width - PATCH), []).append(anchor)
    return [split[key] for key in sorted(split)]


def _list_clusters(group: list[tuple[int, int]]) -> list[list[tuple[int, int]]]:
    """How mapcv 0.3.0 split a group of anchors over an area of interest at column gaps."""
    columns = sorted({c for _, c in group})
    cluster_of: dict[int, int] = {}
    cluster = 0
    for previous, column in zip([None, *columns], columns):
        if previous is not None and column - previous > 2 * PATCH:
            cluster += 1
        cluster_of[column] = cluster
    clusters: list[list[tuple[int, int]]] = [[] for _ in range(cluster + 1)]
    for anchor in group:
        clusters[cluster_of[anchor[1]]].append(anchor)
    return clusters


def _list_chunks(
    anchors: list[tuple[int, int]], chunk_rows: int, pixel_bytes: int, aoi: bool
) -> list[list[tuple[int, int]]]:
    """The chunks mapcv 0.3.0 read, from the full list of anchors."""
    if aoi:
        anchors = [anchor for anchor in anchors if _Strips.inside(*anchor)]
    grouped: dict[int, list[tuple[int, int]]] = {}
    for anchor in anchors:
        grouped.setdefault(anchor[0] // chunk_rows, []).append(anchor)
    groups = [grouped[key] for key in sorted(grouped)]
    if aoi:
        groups = [cluster for group in groups for cluster in _list_clusters(group)]
    return [part for group in groups for part in _list_windows(group, pixel_bytes)]


class _Strips:
    """An area of interest of three column strips, with every third row left out."""

    @staticmethod
    def inside(row: int, col: int) -> bool:
        return row % 3 != 1 and (col < 90 or 400 <= col < 470 or col >= 900)

    def keep_mask(
        self, rows: npt.NDArray[Any], cols: npt.NDArray[Any], size: int
    ) -> npt.NDArray[np.bool_]:
        return np.array([self.inside(int(r), int(c)) for r, c in zip(rows, cols)], dtype=bool)


@pytest.mark.parametrize("aoi", [False, True], ids=["whole", "aoi"])
@pytest.mark.parametrize("budget", [256 * 2**20, 128 * 300 * 3 + 7], ids=["one", "windows"])
@pytest.mark.parametrize(
    "sampler",
    [
        {"stride": 48, "edge_strategy": "pad"},
        {"stride": 40, "edge_strategy": "shift"},
        {"stride": 17, "edge_strategy": "drop"},
        {"mode": "random", "random_count": 700, "random_seed": 5},
        {"mode": "random", "random_count": 10**9, "edge_strategy": "drop"},
    ],
    ids=["pad", "shift", "drop", "random", "random-all"],
)
def test_chunks_hold_the_same_anchors_in_the_same_order(
    monkeypatch: pytest.MonkeyPatch, sampler: dict[str, Any], budget: int, aoi: bool
) -> None:
    from types import SimpleNamespace

    from mapcv.config import MapcvConfig
    from mapcv.sampler import grid_anchors_for, random_anchors_for

    monkeypatch.setattr(pipeline, "_WINDOW_BYTES", budget)
    config = MapcvConfig.model_validate(
        {
            "region": {"west": 10.0, "south": 50.0, "east": 10.03, "north": 50.03},
            "imagery": {"type": "xyz", "zoom": 18, "url_template": "http://x/{z}/{x}/{y}.png"},
            "sampler": {"patch_size": PATCH, **sampler},
            "writer": {"staging_dir": "out"},
        }
    )
    height, width, chunk_rows = 300, 1_200, 128
    meta = SimpleNamespace(
        height=height, width=width, chunk_rows=chunk_rows, bands=[1, 2, 3], dtype="uint8"
    )
    source: Any = SimpleNamespace(metadata=meta)
    with pytest.warns() if sampler.get("random_count") == 10**9 else nullcontext():
        chunks = pipeline._anchor_groups(config, [source], _Strips() if aoi else None)  # type: ignore[arg-type]
        if config.sampler.mode == "random":
            anchors = random_anchors_for(height, width, config.sampler)
        else:
            anchors = grid_anchors_for(height, width, config.sampler)
    expected = _list_chunks(anchors, chunk_rows, 3, aoi)
    assert len(expected) >= (7 if budget < 2**20 else 2)
    assert [chunk.anchors() for chunk in chunks] == expected


def test_a_wide_raster_is_read_in_bounded_windows_with_the_same_patches(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    whole = _wide_config(tmp_path, staging="whole", stride=48)
    reference = run_generate(whole).manifest

    budget = 128 * 400 * 3
    monkeypatch.setattr(pipeline, "_WINDOW_BYTES", budget)
    windows = _windows(monkeypatch)
    split = _wide_config(tmp_path, staging="split", stride=48)
    manifest = run_generate(split).manifest

    assert windows and all(rows * cols * 3 <= budget for rows, cols in windows)
    assert len(windows) > 2  # 2 row chunks, each read as several column windows
    # Patches are numbered in the order they are written, which follows the windows,
    # so compare them by anchor.
    by_anchor = {(e["row"], e["col"]): e for e in reference.patches}
    whole_files = _files(whole.writer.staging_dir)
    split_files = _files(split.writer.staging_dir)
    assert len(manifest.patches) == len(reference.patches)
    for entry in manifest.patches:
        same = by_anchor[(entry["row"], entry["col"])]
        assert entry["summary"] == same["summary"]
        for kind, name in entry["files"].items():
            assert split_files[name] == whole_files[same["files"][kind]]


def test_the_manifest_is_saved_every_few_seconds_not_after_every_chunk(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config = _wide_config(tmp_path)
    manifest_path = config.writer.staging_dir / "manifest.json"
    saved: list[int] = []

    def on_chunk(done: int, total: int) -> None:
        if done:
            exists = manifest_path.exists()
            saved.append(len(Manifest.load(manifest_path).patches) if exists else 0)

    monkeypatch.setattr(pipeline, "_SAVE_EVERY_S", 3600.0)
    result = run_generate(config, on_chunk=on_chunk).manifest
    assert len(saved) == 2 and set(saved) == {0}
    assert len(Manifest.load(manifest_path).patches) == len(result.patches)

    monkeypatch.setattr(pipeline, "_SAVE_EVERY_S", 0.0)
    saved.clear()
    again = _wide_config(tmp_path, staging="again")
    run_generate(again, on_chunk=on_chunk)
    manifest_path = again.writer.staging_dir / "manifest.json"
    assert saved[0] > 0 and saved[-1] == len(result.patches)


def test_an_interrupted_run_saves_its_finished_chunks_and_resumes_to_the_same_bytes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(pipeline, "_SAVE_EVERY_S", 3600.0)
    reference_config = _wide_config(tmp_path, staging="reference")
    reference = run_generate(reference_config).manifest

    config = _wide_config(tmp_path)
    manifest_path = config.writer.staging_dir / "manifest.json"

    def stop_after_first(done: int, total: int) -> None:
        if done == 1:
            raise KeyboardInterrupt

    with pytest.raises(KeyboardInterrupt):
        run_generate(config, on_chunk=stop_after_first)
    partial = Manifest.load(manifest_path).patches
    first_chunk = [e for e in reference.patches if e["chunk"] == reference.patches[0]["chunk"]]
    assert partial == first_chunk

    resumed = run_generate(config).manifest
    assert resumed.patches == reference.patches
    assert _files(config.writer.staging_dir) == _files(reference_config.writer.staging_dir)


def test_a_chunk_stopped_part_way_is_dropped_from_the_saved_manifest(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config = _wide_config(tmp_path)
    manifest_path = config.writer.staging_dir / "manifest.json"
    writes = 0
    original = create_writer

    def failing_writer(*args: Any, **kwargs: Any) -> Any:
        writer = original(*args, **kwargs)
        write = writer.write

        def second_write_fails(*a: Any, **k: Any) -> None:
            nonlocal writes
            writes += 1
            write(*a, **k)
            if writes == 2:
                raise OSError("disk full")

        writer.write = second_write_fails  # type: ignore[method-assign]
        return writer

    monkeypatch.setattr(pipeline, "create_writer", failing_writer)
    with pytest.raises(OSError, match="disk full"):
        run_generate(config)
    saved = Manifest.load(manifest_path).patches
    assert saved and len({e["chunk"] for e in saved}) == 1

    monkeypatch.setattr(pipeline, "create_writer", original)
    (tmp_path / "reference").mkdir()
    reference = run_generate(_wide_config(tmp_path / "reference")).manifest
    assert run_generate(config).manifest.patches == reference.patches


def test_the_plan_estimates_memory_for_a_column_window(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config = _wide_config(tmp_path)
    whole = plan(config).chunk_memory_bytes
    monkeypatch.setattr(pipeline, "_WINDOW_BYTES", 192 * 256 * 3)
    capped = plan(config).chunk_memory_bytes
    # 192-row windows (128 chunk rows + one patch), 256 instead of ~1,250 columns.
    assert capped * 4 < whole
    assert capped == 192 * 256 * (3 * 2 + 2)
