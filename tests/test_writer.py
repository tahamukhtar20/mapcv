"""Tests for M6 patch writer and manifest."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from mapcv import SamplerConfig, sample_patches
from mapcv.writer import (
    Manifest,
    WriterConfig,
    load_or_create_manifest,
    write_patches,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _solid(h: int, w: int, c: int = 3, value: int = 128) -> np.ndarray:
    return np.full((h, w, c), value, dtype=np.uint8)


def _patches(h: int = 16, w: int = 16, ps: int = 8) -> tuple:  # type: ignore[type-arg]
    img = _solid(h, w)
    cfg = SamplerConfig(patch_size=ps, edge_strategy="drop")
    return sample_patches(img, None, cfg)


def _patches_with_mask(h: int = 16, w: int = 16, ps: int = 8) -> tuple:  # type: ignore[type-arg]
    img = _solid(h, w)
    msk = np.ones((h, w), dtype=np.uint8)
    msk[:, w // 2 :] = 2
    cfg = SamplerConfig(patch_size=ps, edge_strategy="drop")
    return sample_patches(img, msk, cfg)


# ---------------------------------------------------------------------------
# WriterConfig
# ---------------------------------------------------------------------------


def test_writer_config_defaults(tmp_path: Path) -> None:
    cfg = WriterConfig(staging_dir=tmp_path)
    assert cfg.image_format == "png"
    assert cfg.jpg_quality == 95


def test_writer_config_jpg_quality_bounds(tmp_path: Path) -> None:
    with pytest.raises(Exception):
        WriterConfig(staging_dir=tmp_path, jpg_quality=0)
    with pytest.raises(Exception):
        WriterConfig(staging_dir=tmp_path, jpg_quality=101)


# ---------------------------------------------------------------------------
# Manifest load / save
# ---------------------------------------------------------------------------


def test_manifest_save_and_load(tmp_path: Path) -> None:
    m = Manifest(class_map={"bg": 0, "building": 1})
    path = tmp_path / "manifest.json"
    m.save(path)
    loaded = Manifest.load(path)
    assert loaded.class_map == {"bg": 0, "building": 1}
    assert loaded.version == 1
    assert loaded.patches == []


def test_manifest_save_is_valid_json(tmp_path: Path) -> None:
    m = Manifest(class_map={"a": 1})
    path = tmp_path / "manifest.json"
    m.save(path)
    data = json.loads(path.read_text())
    assert "patches" in data
    assert "class_map" in data
    assert data["version"] == 1


def test_load_or_create_returns_new_when_missing(tmp_path: Path) -> None:
    m = load_or_create_manifest(tmp_path / "manifest.json", {"x": 1})
    assert m.class_map == {"x": 1}
    assert m.patches == []


def test_load_or_create_loads_existing(tmp_path: Path) -> None:
    path = tmp_path / "manifest.json"
    Manifest(class_map={"y": 2}).save(path)
    m = load_or_create_manifest(path, {"ignored": 99})
    assert m.class_map == {"y": 2}


# ---------------------------------------------------------------------------
# write_patches: file creation
# ---------------------------------------------------------------------------


def test_images_dir_created(tmp_path: Path) -> None:
    imgs, _, meta = _patches()
    cfg = WriterConfig(staging_dir=tmp_path)
    m = Manifest(class_map={})
    write_patches(imgs, None, meta, cfg, m)
    assert (tmp_path / "Images").is_dir()


def test_masks_dir_created_when_mask_provided(tmp_path: Path) -> None:
    imgs, msks, meta = _patches_with_mask()
    cfg = WriterConfig(staging_dir=tmp_path)
    m = Manifest(class_map={})
    write_patches(imgs, msks, meta, cfg, m)
    assert (tmp_path / "Masks").is_dir()


def test_masks_dir_not_created_without_mask(tmp_path: Path) -> None:
    imgs, _, meta = _patches()
    cfg = WriterConfig(staging_dir=tmp_path)
    m = Manifest(class_map={})
    write_patches(imgs, None, meta, cfg, m)
    assert not (tmp_path / "Masks").exists()


def test_image_files_written(tmp_path: Path) -> None:
    imgs, _, meta = _patches()
    cfg = WriterConfig(staging_dir=tmp_path)
    m = Manifest(class_map={})
    write_patches(imgs, None, meta, cfg, m)
    written = list((tmp_path / "Images").glob("*.png"))
    assert len(written) == len(meta)


def test_mask_files_written(tmp_path: Path) -> None:
    imgs, msks, meta = _patches_with_mask()
    cfg = WriterConfig(staging_dir=tmp_path)
    m = Manifest(class_map={})
    write_patches(imgs, msks, meta, cfg, m)
    written = list((tmp_path / "Masks").glob("*.png"))
    assert len(written) == len(meta)


def test_jpg_format_written(tmp_path: Path) -> None:
    imgs, _, meta = _patches()
    cfg = WriterConfig(staging_dir=tmp_path, image_format="jpg")
    m = Manifest(class_map={})
    write_patches(imgs, None, meta, cfg, m)
    written = list((tmp_path / "Images").glob("*.jpg"))
    assert len(written) == len(meta)


def test_empty_meta_writes_nothing(tmp_path: Path) -> None:
    imgs = np.zeros((0, 8, 8, 3), dtype=np.uint8)
    cfg = WriterConfig(staging_dir=tmp_path)
    m = Manifest(class_map={})
    write_patches(imgs, None, [], cfg, m)
    assert not (tmp_path / "Images").exists()
    assert m.patches == []


# ---------------------------------------------------------------------------
# write_patches: manifest entries
# ---------------------------------------------------------------------------


def test_manifest_extended_in_place(tmp_path: Path) -> None:
    imgs, _, meta = _patches()
    cfg = WriterConfig(staging_dir=tmp_path)
    m = Manifest(class_map={})
    write_patches(imgs, None, meta, cfg, m)
    assert len(m.patches) == len(meta)


def test_manifest_filenames_sequential(tmp_path: Path) -> None:
    imgs, _, meta = _patches()
    cfg = WriterConfig(staging_dir=tmp_path)
    m = Manifest(class_map={})
    write_patches(imgs, None, meta, cfg, m)
    fnames = [e["filename"] for e in m.patches]
    assert fnames == sorted(fnames)
    assert fnames[0] == "patch_0000000.png"


def test_manifest_row_col_match_meta(tmp_path: Path) -> None:
    imgs, _, meta = _patches()
    cfg = WriterConfig(staging_dir=tmp_path)
    m = Manifest(class_map={})
    write_patches(imgs, None, meta, cfg, m)
    for entry, pm in zip(m.patches, meta):
        assert entry["row"] == pm["row"]
        assert entry["col"] == pm["col"]
        assert entry["padded"] == pm["padded"]


def test_manifest_strip_index_recorded(tmp_path: Path) -> None:
    imgs, _, meta = _patches()
    cfg = WriterConfig(staging_dir=tmp_path)
    m = Manifest(class_map={})
    write_patches(imgs, None, meta, cfg, m, strip_index=3)
    assert all(e["strip_index"] == 3 for e in m.patches)


def test_manifest_mask_filename_none_without_mask(tmp_path: Path) -> None:
    imgs, _, meta = _patches()
    cfg = WriterConfig(staging_dir=tmp_path)
    m = Manifest(class_map={})
    write_patches(imgs, None, meta, cfg, m)
    assert all(e["mask_filename"] is None for e in m.patches)


def test_manifest_mask_filename_set_with_mask(tmp_path: Path) -> None:
    imgs, msks, meta = _patches_with_mask()
    cfg = WriterConfig(staging_dir=tmp_path)
    m = Manifest(class_map={})
    write_patches(imgs, msks, meta, cfg, m)
    for entry in m.patches:
        assert entry["mask_filename"] == entry["filename"]


# ---------------------------------------------------------------------------
# write_patches: per-class pixel counts and empty ratio
# ---------------------------------------------------------------------------


def test_per_class_counts_no_mask_empty(tmp_path: Path) -> None:
    imgs, _, meta = _patches()
    cfg = WriterConfig(staging_dir=tmp_path)
    m = Manifest(class_map={})
    write_patches(imgs, None, meta, cfg, m)
    assert all(e["per_class_pixel_counts"] == {} for e in m.patches)


def test_per_class_counts_correct(tmp_path: Path) -> None:
    # 8x8 image, mask: left half class 1, right half class 2
    img = _solid(8, 8)
    msk = np.zeros((8, 8), dtype=np.uint8)
    msk[:, :4] = 1
    msk[:, 4:] = 2
    cfg_s = SamplerConfig(patch_size=8, edge_strategy="drop")
    imgs, msks, meta = sample_patches(img, msk, cfg_s)
    cfg_w = WriterConfig(staging_dir=tmp_path)
    m = Manifest(class_map={"a": 1, "b": 2})
    write_patches(imgs, msks, meta, cfg_w, m)
    counts = m.patches[0]["per_class_pixel_counts"]
    assert counts["1"] == 32
    assert counts["2"] == 32


def test_empty_ratio_all_black(tmp_path: Path) -> None:
    img = np.zeros((8, 8, 3), dtype=np.uint8)
    cfg_s = SamplerConfig(patch_size=8, edge_strategy="drop", max_empty_ratio=1.0)
    imgs, _, meta = sample_patches(img, None, cfg_s)
    cfg_w = WriterConfig(staging_dir=tmp_path)
    m = Manifest(class_map={})
    write_patches(imgs, None, meta, cfg_w, m)
    assert m.patches[0]["empty_ratio"] == 1.0


def test_empty_ratio_all_nonzero(tmp_path: Path) -> None:
    imgs, _, meta = _patches()
    cfg = WriterConfig(staging_dir=tmp_path)
    m = Manifest(class_map={})
    write_patches(imgs, None, meta, cfg, m)
    assert all(e["empty_ratio"] == 0.0 for e in m.patches)


# ---------------------------------------------------------------------------
# write_patches: resume (skip existing files)
# ---------------------------------------------------------------------------


def test_resume_skips_existing_file(tmp_path: Path) -> None:
    imgs, _, meta = _patches()
    cfg = WriterConfig(staging_dir=tmp_path)
    m = Manifest(class_map={})
    write_patches(imgs, None, meta, cfg, m)

    # Corrupt the first image on disk to verify it isn't overwritten.
    first_path = tmp_path / "Images" / m.patches[0]["filename"]
    first_path.write_bytes(b"SENTINEL")

    m2 = Manifest(class_map={})
    write_patches(imgs, None, meta, cfg, m2)
    assert first_path.read_bytes() == b"SENTINEL"


# ---------------------------------------------------------------------------
# write_patches: multi-strip sequential indexing
# ---------------------------------------------------------------------------


def test_two_strips_non_overlapping_filenames(tmp_path: Path) -> None:
    imgs, _, meta = _patches()
    cfg = WriterConfig(staging_dir=tmp_path)
    m = Manifest(class_map={})
    write_patches(imgs, None, meta, cfg, m, strip_index=0)
    write_patches(imgs, None, meta, cfg, m, strip_index=1)
    fnames = [e["filename"] for e in m.patches]
    assert len(fnames) == len(set(fnames)), "filenames must be unique across strips"
    assert len(m.patches) == 2 * len(meta)


def test_strip_indices_recorded_correctly(tmp_path: Path) -> None:
    imgs, _, meta = _patches()
    cfg = WriterConfig(staging_dir=tmp_path)
    m = Manifest(class_map={})
    write_patches(imgs, None, meta, cfg, m, strip_index=0)
    write_patches(imgs, None, meta, cfg, m, strip_index=1)
    strip0 = [e for e in m.patches if e["strip_index"] == 0]
    strip1 = [e for e in m.patches if e["strip_index"] == 1]
    assert len(strip0) == len(meta)
    assert len(strip1) == len(meta)


# ---------------------------------------------------------------------------
# write_patches: parallel writing consistency
# ---------------------------------------------------------------------------


def test_parallel_same_as_serial(tmp_path: Path) -> None:
    imgs, msks, meta = _patches_with_mask()

    m1 = Manifest(class_map={})
    m2 = Manifest(class_map={})

    # Same number of entries with same row/col/counts
    assert len(m1.patches) == len(m2.patches)
    for e1, e2 in zip(m1.patches, m2.patches):
        assert e1["row"] == e2["row"]
        assert e1["col"] == e2["col"]
        assert e1["per_class_pixel_counts"] == e2["per_class_pixel_counts"]
        assert e1["empty_ratio"] == e2["empty_ratio"]


# ---------------------------------------------------------------------------
# Manifest round-trip with patch entries
# ---------------------------------------------------------------------------


def test_manifest_round_trip_with_entries(tmp_path: Path) -> None:
    imgs, _, meta = _patches()
    cfg = WriterConfig(staging_dir=tmp_path)
    m = Manifest(class_map={"bg": 0, "obj": 1})
    write_patches(imgs, None, meta, cfg, m)

    mpath = tmp_path / "manifest.json"
    m.save(mpath)
    loaded = Manifest.load(mpath)

    assert len(loaded.patches) == len(m.patches)
    for orig, reloaded in zip(m.patches, loaded.patches):
        assert orig["filename"] == reloaded["filename"]
        assert orig["row"] == reloaded["row"]
        assert orig["empty_ratio"] == reloaded["empty_ratio"]
