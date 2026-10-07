"""Tests for the patch writer and the manifest entries it appends."""

from __future__ import annotations

import io
from pathlib import Path
from typing import Any

import numpy as np
import numpy.typing as npt
import pytest
from PIL import Image, JpegImagePlugin
from pydantic import ValidationError

from mapcv import SamplerConfig, sample_patches
from mapcv.manifest import Manifest, TargetRecord
from mapcv.sampler import PatchMeta
from mapcv.writer import WriterConfig, write_patches

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _solid(h: int, w: int, c: int = 3, value: int = 128) -> npt.NDArray[np.uint8]:
    return np.full((h, w, c), value, dtype=np.uint8)


def _patches(
    h: int = 16, w: int = 16, ps: int = 8
) -> tuple[npt.NDArray[np.uint8], npt.NDArray[np.uint8] | None, list[PatchMeta]]:
    img = _solid(h, w)
    cfg = SamplerConfig(patch_size=ps, edge_strategy="drop")
    return sample_patches(img, None, cfg)


def _patches_with_mask(
    h: int = 16, w: int = 16, ps: int = 8
) -> tuple[npt.NDArray[np.uint8], npt.NDArray[np.uint8] | None, list[PatchMeta]]:
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
    assert cfg.jpg_subsampling == "4:2:0"


def test_writer_config_jpg_subsampling_is_validated(tmp_path: Path) -> None:
    assert WriterConfig(staging_dir=tmp_path, jpg_subsampling="4:4:4").jpg_subsampling == "4:4:4"
    for bad in ("4:2:2", "420", "", "4:4:4 "):
        with pytest.raises(ValidationError, match="jpg_subsampling"):
            WriterConfig(staging_dir=tmp_path, jpg_subsampling=bad)  # type: ignore[arg-type]


def test_writer_config_jpg_quality_bounds(tmp_path: Path) -> None:
    with pytest.raises(ValidationError):
        WriterConfig(staging_dir=tmp_path, jpg_quality=0)
    with pytest.raises(ValidationError):
        WriterConfig(staging_dir=tmp_path, jpg_quality=101)


# ---------------------------------------------------------------------------
# write_patches: file creation
# ---------------------------------------------------------------------------


def test_images_dir_created(tmp_path: Path) -> None:
    imgs, _, meta = _patches()
    cfg = WriterConfig(staging_dir=tmp_path)
    m = Manifest()
    write_patches(imgs, None, meta, cfg, m)
    assert (tmp_path / "Images").is_dir()


def test_masks_dir_created_when_mask_provided(tmp_path: Path) -> None:
    imgs, msks, meta = _patches_with_mask()
    cfg = WriterConfig(staging_dir=tmp_path)
    m = Manifest()
    write_patches(imgs, msks, meta, cfg, m)
    assert (tmp_path / "Masks").is_dir()


def test_masks_dir_not_created_without_mask(tmp_path: Path) -> None:
    imgs, _, meta = _patches()
    cfg = WriterConfig(staging_dir=tmp_path)
    m = Manifest()
    write_patches(imgs, None, meta, cfg, m)
    assert not (tmp_path / "Masks").exists()


def test_image_files_written(tmp_path: Path) -> None:
    imgs, _, meta = _patches()
    cfg = WriterConfig(staging_dir=tmp_path)
    m = Manifest()
    write_patches(imgs, None, meta, cfg, m)
    written = list((tmp_path / "Images").glob("*.png"))
    assert len(written) == len(meta)


def test_mask_files_written(tmp_path: Path) -> None:
    imgs, msks, meta = _patches_with_mask()
    cfg = WriterConfig(staging_dir=tmp_path)
    m = Manifest()
    write_patches(imgs, msks, meta, cfg, m)
    written = list((tmp_path / "Masks").glob("*.png"))
    assert len(written) == len(meta)


def test_jpg_format_written(tmp_path: Path) -> None:
    imgs, _, meta = _patches()
    cfg = WriterConfig(staging_dir=tmp_path, image_format="jpg")
    m = Manifest()
    write_patches(imgs, None, meta, cfg, m)
    written = list((tmp_path / "Images").glob("*.jpg"))
    assert len(written) == len(meta)


def _corpus(n: int = 24, size: int = 128) -> npt.NDArray[np.uint8]:
    """Fixed, imagery-like patches: smooth colour gradients, hard edges and mild noise."""
    rng = np.random.default_rng(1234)
    yy, xx = np.mgrid[0:size, 0:size].astype(np.float64) / size
    out = np.empty((n, size, size, 3), dtype=np.uint8)
    for i in range(n):
        phase = rng.uniform(0, 6.28, size=3)
        base = np.stack(
            [
                90 + 80 * np.sin(3 * xx + phase[c]) * np.cos(2 * yy + phase[(c + 1) % 3])
                for c in range(3)
            ],
            axis=-1,
        )
        base[(xx + yy * 0.5 > rng.uniform(0.6, 1.2))] += rng.uniform(-40, 40, size=3)
        base += rng.normal(0, 4, size=base.shape)
        out[i] = np.clip(base, 0, 255).astype(np.uint8)
    return out


def _write_jpgs(images: npt.NDArray[np.uint8], tmp_path: Path, **config: Any) -> list[Path]:
    meta = [PatchMeta(row=i, col=0, padded=False, empty_ratio=0.0) for i in range(len(images))]
    cfg = WriterConfig(staging_dir=tmp_path, image_format="jpg", **config)
    write_patches(images, None, meta, cfg, Manifest())
    return sorted((tmp_path / "Images").glob("*.jpg"))


@pytest.mark.parametrize(("setting", "sampling"), [(None, 2), ("4:2:0", 2), ("4:4:4", 0)])
def test_jpg_sampling_factors_match_the_setting(
    tmp_path: Path, setting: str | None, sampling: int
) -> None:
    extra = {} if setting is None else {"jpg_subsampling": setting}
    files = _write_jpgs(_corpus(2), tmp_path, **extra)
    for path in files:
        with Image.open(path) as img:
            # Pillow reports 0 = 4:4:4, 1 = 4:2:2, 2 = 4:2:0.
            assert JpegImagePlugin.get_sampling(img) == sampling


def test_jpg_size_is_close_to_pillows_at_quality_95(tmp_path: Path) -> None:
    images = _corpus()
    ours = sum(p.stat().st_size for p in _write_jpgs(images, tmp_path, jpg_quality=95))
    theirs = 0
    for patch in images:
        buffer = io.BytesIO()
        Image.fromarray(patch).save(buffer, "JPEG", quality=95, subsampling=2)
        theirs += buffer.tell()
    assert 0.85 * theirs <= ours <= 1.15 * theirs, (ours, theirs)


def test_jpg_444_is_larger_and_not_worse_than_420(tmp_path: Path) -> None:
    images = _corpus(8)

    def run(sub: str) -> tuple[int, float]:
        files = _write_jpgs(images, tmp_path / sub.replace(":", ""), jpg_subsampling=sub)
        size = sum(p.stat().st_size for p in files)
        err = 0.0
        for patch, path in zip(images, files):
            with Image.open(path) as img:
                err += float(np.mean((np.asarray(img.convert("RGB"), np.float64) - patch) ** 2))
        return size, err

    size420, err420 = run("4:2:0")
    size444, err444 = run("4:4:4")
    assert size444 > 1.2 * size420
    assert err444 <= err420


def test_jpg_png_output_ignores_subsampling(tmp_path: Path) -> None:
    images = _corpus(2, 32)
    meta = [PatchMeta(row=i, col=0, padded=False, empty_ratio=0.0) for i in range(2)]
    for sub in ("4:2:0", "4:4:4"):
        out = tmp_path / sub.replace(":", "")
        cfg = WriterConfig(staging_dir=out, image_format="png", jpg_subsampling=sub)
        write_patches(images, None, meta, cfg, Manifest())
    for name in ("patch_0000000.png", "patch_0000001.png"):
        assert (tmp_path / "420" / "Images" / name).read_bytes() == (
            tmp_path / "444" / "Images" / name
        ).read_bytes()


def test_npy_format_preserves_float32_bands_first(tmp_path: Path) -> None:
    images = np.arange(2 * 4 * 4 * 5, dtype=np.float32).reshape(2, 4, 4, 5)
    meta = [
        PatchMeta(row=0, col=0, padded=False, empty_ratio=0.0),
        PatchMeta(row=4, col=0, padded=False, empty_ratio=0.0),
    ]
    config = WriterConfig(staging_dir=tmp_path, image_format="npy")
    manifest = Manifest()

    write_patches(images, None, meta, config, manifest)

    stored = np.load(tmp_path / "Images" / "patch_0000000.npy", allow_pickle=False)
    assert stored.shape == (5, 4, 4)
    assert stored.dtype == np.float32
    np.testing.assert_array_equal(stored[2], images[0, :, :, 2])


def test_npy_overwrites_orphaned_tensor(tmp_path: Path) -> None:
    images = np.ones((1, 2, 2, 2), dtype=np.float32)
    meta = [PatchMeta(row=0, col=0, padded=False, empty_ratio=0.0)]
    config = WriterConfig(staging_dir=tmp_path, image_format="npy")
    first_manifest = Manifest()
    write_patches(images, None, meta, config, first_manifest)
    tensor_path = tmp_path / "Images" / "patch_0000000.npy"
    np.save(tensor_path, np.zeros((2, 2, 2), dtype=np.float32), allow_pickle=False)

    write_patches(images, None, meta, config, Manifest())

    assert np.all(np.load(tensor_path, allow_pickle=False) == 1)


def test_empty_meta_writes_nothing(tmp_path: Path) -> None:
    imgs = np.zeros((0, 8, 8, 3), dtype=np.uint8)
    cfg = WriterConfig(staging_dir=tmp_path)
    m = Manifest()
    write_patches(imgs, None, [], cfg, m)
    assert not (tmp_path / "Images").exists()
    assert m.patches == []


# ---------------------------------------------------------------------------
# write_patches: manifest entries
# ---------------------------------------------------------------------------


def test_manifest_extended_in_place(tmp_path: Path) -> None:
    imgs, _, meta = _patches()
    cfg = WriterConfig(staging_dir=tmp_path)
    m = Manifest()
    write_patches(imgs, None, meta, cfg, m)
    assert len(m.patches) == len(meta)


def test_manifest_filenames_sequential(tmp_path: Path) -> None:
    imgs, _, meta = _patches()
    cfg = WriterConfig(staging_dir=tmp_path)
    m = Manifest()
    write_patches(imgs, None, meta, cfg, m)
    fnames = [e["files"]["image"] for e in m.patches]
    assert fnames == sorted(fnames)
    assert fnames[0] == "Images/patch_0000000.png"


def test_manifest_row_col_match_meta(tmp_path: Path) -> None:
    imgs, _, meta = _patches()
    cfg = WriterConfig(staging_dir=tmp_path)
    m = Manifest()
    write_patches(imgs, None, meta, cfg, m)
    for entry, pm in zip(m.patches, meta):
        assert entry["row"] == pm["row"]
        assert entry["col"] == pm["col"]
        assert entry["padded"] == pm["padded"]


def test_manifest_chunk_index_recorded(tmp_path: Path) -> None:
    imgs, _, meta = _patches()
    cfg = WriterConfig(staging_dir=tmp_path)
    m = Manifest()
    write_patches(imgs, None, meta, cfg, m, chunk_index=3)
    assert all(e["chunk"] == 3 for e in m.patches)


def test_manifest_lists_only_the_image_without_mask(tmp_path: Path) -> None:
    imgs, _, meta = _patches()
    cfg = WriterConfig(staging_dir=tmp_path)
    m = Manifest()
    write_patches(imgs, None, meta, cfg, m)
    assert all(set(e["files"]) == {"image"} for e in m.patches)


@pytest.mark.parametrize("image_format", ["png", "jpg", "npy"])
def test_manifest_lists_image_and_mask_paths(tmp_path: Path, image_format: Any) -> None:
    imgs, msks, meta = _patches_with_mask()
    cfg = WriterConfig(staging_dir=tmp_path, image_format=image_format)
    m = Manifest()
    write_patches(imgs, msks, meta, cfg, m)
    for index, entry in enumerate(m.patches):
        assert entry["files"] == {
            "image": f"Images/patch_{index:07d}.{image_format}",
            "mask": f"Masks/patch_{index:07d}.png",
        }
        assert all((tmp_path / path).is_file() for path in entry["files"].values())
        assert set(entry) == {"row", "col", "padded", "chunk", "files", "summary"}


# ---------------------------------------------------------------------------
# write_patches: per-class pixel counts and empty ratio
# ---------------------------------------------------------------------------


def test_per_class_counts_no_mask_empty(tmp_path: Path) -> None:
    imgs, _, meta = _patches()
    cfg = WriterConfig(staging_dir=tmp_path)
    m = Manifest()
    write_patches(imgs, None, meta, cfg, m)
    assert all(e["summary"] == {"empty_ratio": 0.0} for e in m.patches)


def test_per_class_counts_correct(tmp_path: Path) -> None:
    # 8x8 image, mask: left half class 1, right half class 2
    img = _solid(8, 8)
    msk = np.zeros((8, 8), dtype=np.uint8)
    msk[:, :4] = 1
    msk[:, 4:] = 2
    cfg_s = SamplerConfig(patch_size=8, edge_strategy="drop")
    imgs, msks, meta = sample_patches(img, msk, cfg_s)
    cfg_w = WriterConfig(staging_dir=tmp_path)
    m = Manifest(target=TargetRecord(type="segmentation", class_map={"a": 1, "b": 2}))
    write_patches(imgs, msks, meta, cfg_w, m)
    counts = m.patches[0]["summary"]["class_pixels"]
    assert counts["1"] == 32
    assert counts["2"] == 32


def test_empty_ratio_all_black(tmp_path: Path) -> None:
    img = np.zeros((8, 8, 3), dtype=np.uint8)
    cfg_s = SamplerConfig(patch_size=8, edge_strategy="drop", max_empty_ratio=1.0)
    imgs, _, meta = sample_patches(img, None, cfg_s)
    cfg_w = WriterConfig(staging_dir=tmp_path)
    m = Manifest()
    write_patches(imgs, None, meta, cfg_w, m)
    assert m.patches[0]["summary"]["empty_ratio"] == 1.0


def test_empty_ratio_all_nonzero(tmp_path: Path) -> None:
    imgs, _, meta = _patches()
    cfg = WriterConfig(staging_dir=tmp_path)
    m = Manifest()
    write_patches(imgs, None, meta, cfg, m)
    assert all(e["summary"]["empty_ratio"] == 0.0 for e in m.patches)


# ---------------------------------------------------------------------------
# write_patches: orphaned files from interrupted runs
# ---------------------------------------------------------------------------


def test_orphaned_file_is_overwritten(tmp_path: Path) -> None:
    imgs, _, meta = _patches()
    cfg = WriterConfig(staging_dir=tmp_path)
    m = Manifest()
    write_patches(imgs, None, meta, cfg, m)

    # A file left by an interrupted run is not in the manifest and must be replaced.
    first_path = tmp_path / m.patches[0]["files"]["image"]
    first_path.write_bytes(b"STALE")

    m2 = Manifest()
    write_patches(imgs, None, meta, cfg, m2)
    assert first_path.read_bytes() != b"STALE"


# ---------------------------------------------------------------------------
# write_patches: multi-strip sequential indexing
# ---------------------------------------------------------------------------


def test_two_strips_non_overlapping_filenames(tmp_path: Path) -> None:
    imgs, _, meta = _patches()
    cfg = WriterConfig(staging_dir=tmp_path)
    m = Manifest()
    write_patches(imgs, None, meta, cfg, m, chunk_index=0)
    write_patches(imgs, None, meta, cfg, m, chunk_index=1)
    fnames = [e["files"]["image"] for e in m.patches]
    assert len(fnames) == len(set(fnames)), "filenames must be unique across chunks"
    assert len(m.patches) == 2 * len(meta)


def test_strip_indices_recorded_correctly(tmp_path: Path) -> None:
    imgs, _, meta = _patches()
    cfg = WriterConfig(staging_dir=tmp_path)
    m = Manifest()
    write_patches(imgs, None, meta, cfg, m, chunk_index=0)
    write_patches(imgs, None, meta, cfg, m, chunk_index=1)
    strip0 = [e for e in m.patches if e["chunk"] == 0]
    strip1 = [e for e in m.patches if e["chunk"] == 1]
    assert len(strip0) == len(meta)
    assert len(strip1) == len(meta)


# ---------------------------------------------------------------------------
# write_patches: parallel writing consistency
# ---------------------------------------------------------------------------


def test_parallel_same_as_serial(tmp_path: Path) -> None:
    imgs, msks, meta = _patches_with_mask()
    cfg1 = WriterConfig(staging_dir=tmp_path / "run1")
    cfg2 = WriterConfig(staging_dir=tmp_path / "run2")

    m1 = Manifest()
    m2 = Manifest()
    write_patches(imgs, msks, meta, cfg1, m1)
    write_patches(imgs, msks, meta, cfg2, m2)

    assert len(m1.patches) == len(m2.patches)
    for e1, e2 in zip(m1.patches, m2.patches):
        assert e1["row"] == e2["row"]
        assert e1["col"] == e2["col"]
        assert e1["summary"] == e2["summary"]


# ---------------------------------------------------------------------------
# write_patches: deterministic manifests
# ---------------------------------------------------------------------------


def _multi_class_patches() -> tuple[
    npt.NDArray[np.uint8], npt.NDArray[np.uint8] | None, list[PatchMeta]
]:
    rng = np.random.default_rng(0)
    img = rng.integers(1, 255, size=(32, 32, 3), dtype=np.uint8)
    msk = rng.choice(np.array([0, 2, 10, 37, 100, 255], dtype=np.uint8), size=(32, 32))
    cfg = SamplerConfig(patch_size=8, edge_strategy="drop")
    return sample_patches(img, msk, cfg)


@pytest.mark.parametrize("image_format", ["png", "npy"])
def test_manifest_json_byte_identical_across_runs(tmp_path: Path, image_format: Any) -> None:
    imgs, msks, meta = _multi_class_patches()
    manifests: list[bytes] = []
    for run in ("run1", "run2"):
        cfg = WriterConfig(staging_dir=tmp_path / run, image_format=image_format)
        m = Manifest(target=TargetRecord(type="segmentation", class_map={"a": 2, "b": 10}))
        write_patches(imgs, msks, meta, cfg, m)
        path = tmp_path / run / "manifest.json"
        m.save(path)
        manifests.append(path.read_bytes())
    assert manifests[0] == manifests[1]


@pytest.mark.parametrize("image_format", ["png", "npy"])
def test_per_class_counts_keys_numerically_ordered(tmp_path: Path, image_format: Any) -> None:
    img = _solid(8, 8)
    msk = np.zeros((8, 8), dtype=np.uint8)
    msk[0, :] = 10
    msk[1, :] = 2
    msk[2, :] = 255
    msk[3, :] = 100
    cfg_s = SamplerConfig(patch_size=8, edge_strategy="drop")
    imgs, msks, meta = sample_patches(img, msk, cfg_s)
    cfg_w = WriterConfig(staging_dir=tmp_path, image_format=image_format)
    m = Manifest()
    write_patches(imgs, msks, meta, cfg_w, m)
    counts = m.patches[0]["summary"]["class_pixels"]
    assert list(counts) == ["0", "2", "10", "100", "255"]
    assert counts == {"0": 32, "2": 8, "10": 8, "100": 8, "255": 8}


# ---------------------------------------------------------------------------
# Manifest round-trip with patch entries
# ---------------------------------------------------------------------------


def test_manifest_round_trip_with_entries(tmp_path: Path) -> None:
    imgs, _, meta = _patches()
    cfg = WriterConfig(staging_dir=tmp_path)
    m = Manifest(target=TargetRecord(type="segmentation", class_map={"bg": 0, "obj": 1}))
    write_patches(imgs, None, meta, cfg, m)

    mpath = tmp_path / "manifest.json"
    m.save(mpath)
    loaded = Manifest.load(mpath)

    assert len(loaded.patches) == len(m.patches)
    for orig, reloaded in zip(m.patches, loaded.patches):
        assert orig == reloaded
