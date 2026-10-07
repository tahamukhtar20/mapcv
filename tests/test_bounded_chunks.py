"""Chunk windows stay below a memory budget, and an interrupted run saves what it finished.

A wide raster is read in column windows once a chunk's window would pass
``pipeline._WINDOW_BYTES``; the patches must not change. The manifest is saved every
few seconds and when the run stops, keeping only the entries of finished chunks.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Tuple

import pytest

pytest.importorskip("rasterio", reason="these tests write their rasters with rasterio")

from test_geotiff_imagery import config_for, make_raster, write_labels  # noqa: E402

from mapcv import pipeline  # noqa: E402
from mapcv.manifest import Manifest  # noqa: E402
from mapcv.pipeline import _column_windows, run_generate  # noqa: E402
from mapcv.planning import plan  # noqa: E402
from mapcv.writers import create_writer  # noqa: E402

PATCH = 64


def _files(staging: Path) -> Dict[str, bytes]:
    return {
        str(path.relative_to(staging)): path.read_bytes()
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


def _windows(monkeypatch: pytest.MonkeyPatch) -> List[Tuple[int, int]]:
    """Record (rows, cols) of every chunk window the pipeline reads."""
    seen: List[Tuple[int, int]] = []
    original = pipeline._process_anchor_chunk

    def record(source: Any, group: List[Tuple[int, int]], *args: Any) -> Any:
        rows = max(r for r, _ in group) + PATCH - min(r for r, _ in group)
        cols = max(c for _, c in group) + PATCH - min(c for _, c in group)
        seen.append((rows, cols))
        return original(source, group, *args)

    monkeypatch.setattr(pipeline, "_process_anchor_chunk", record)
    return seen


def test_column_windows_keep_a_small_group_whole() -> None:
    group = [(row, col) for row in (0, 64) for col in range(0, 1024, 64)]
    assert _column_windows(group, PATCH, pixel_bytes=3) == [group]


def test_column_windows_split_a_wide_group_in_order(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(pipeline, "_WINDOW_BYTES", 128 * 300 * 3)
    group = [(row, col) for row in (0, 64) for col in range(0, 1024, 64)]
    parts = _column_windows(group, PATCH, pixel_bytes=3)
    assert len(parts) > 1
    assert sorted(a for part in parts for a in part) == sorted(group)
    for part in parts:
        width = max(c for _, c in part) + PATCH - min(c for _, c in part)
        assert 128 * width * 3 <= pipeline._WINDOW_BYTES
        # Anchors keep their order inside a part, and parts run left to right.
        assert part == [a for a in group if a in part]
    assert [min(c for _, c in p) for p in parts] == sorted(min(c for _, c in p) for p in parts)


def test_column_windows_never_go_below_two_patches(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(pipeline, "_WINDOW_BYTES", 1)
    group = [(0, col) for col in range(0, 512, 32)]
    parts = _column_windows(group, PATCH, pixel_bytes=3)
    assert sorted(a for part in parts for a in part) == sorted(group)
    for part in parts:
        assert max(c for _, c in part) + PATCH - min(c for _, c in part) <= 2 * PATCH


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
    saved: List[int] = []

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
    reference = run_generate(_wide_config(tmp_path, staging="reference")).manifest
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
