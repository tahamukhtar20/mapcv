"""CRS helpers, world files, footprints and the writer's argument checks (no rasterio needed)."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any, Tuple

import numpy as np
import pytest

from mapcv._georef import epsg_code, is_geographic, to_lonlat, world_file_text
from mapcv.footprints import write_footprints
from mapcv.manifest import Manifest, SourceRecord, TargetRecord
from mapcv.sampler import PatchMeta
from mapcv.writer import WriterConfig, write_patches

Transform = Tuple[float, float, float, float, float, float]
MERCATOR: Transform = (2.0, 0.0, 100.0, 0.0, -2.0, 900.0)


def _manifest(crs: Any = "EPSG:3857", transform: Any = MERCATOR, **extra: Any) -> Manifest:
    return Manifest(
        sources=[SourceRecord(crs=crs, transform=transform, bands=["r", "g", "b"], **extra)],
        sampler={"patch_size": 8},
    )


def _meta(row: int = 0, col: int = 0) -> PatchMeta:
    return PatchMeta(row=row, col=col, padded=False, empty_ratio=0.0)


def test_epsg_codes_are_parsed_and_others_rejected() -> None:
    assert epsg_code("EPSG:32633") == 32633
    assert epsg_code(" epsg:4326 ") == 4326
    for bad in ("PROJ:x", "EPSG:", "EPSG:0", "EPSG:40000", "32633"):
        with pytest.raises(ValueError, match="EPSG"):
            epsg_code(bad)


def test_geographic_codes_without_pyproj_use_the_epsg_range(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setitem(sys.modules, "pyproj", None)
    assert is_geographic(4326) and is_geographic(4269)
    assert not is_geographic(3857) and not is_geographic(32633)


def test_world_file_holds_the_pixel_centre() -> None:
    text = world_file_text((2.0, 0.5, 100.0, 0.25, -2.0, 900.0))
    assert text.split() == ["2.0", "0.25", "0.5", "-2.0", "101.25", "899.125"]
    assert text.endswith("\n")


def test_web_mercator_inverse_and_identity() -> None:
    x, y = np.array([0.0, 1113194.9079327357]), np.array([0.0, 4865942.279503176])
    lon, lat = to_lonlat("EPSG:3857")(x, y)
    np.testing.assert_allclose(lon, [0.0, 10.0], atol=1e-9)
    np.testing.assert_allclose(lat, [0.0, 40.0], atol=1e-9)
    same = to_lonlat("epsg:4326")(x, y)
    assert same[0] is x and same[1] is y


def test_other_crs_need_pyproj(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setitem(sys.modules, "pyproj", None)
    with pytest.raises(RuntimeError, match="needs pyproj"):
        to_lonlat("EPSG:32633")


def test_south_up_footprints_are_still_counter_clockwise(tmp_path: Path) -> None:
    manifest = _manifest(transform=(2.0, 0.0, 100.0, 0.0, 2.0, 900.0))
    manifest.patches = [
        {
            "row": 0,
            "col": 0,
            "padded": False,
            "chunk": 0,
            "files": {"image": "Images/patch_0000000.png"},
            "summary": {"empty_ratio": 0.0},
        }
    ]
    write_footprints(manifest, None, tmp_path / "p.geojson")
    feature = json.loads((tmp_path / "p.geojson").read_text())["features"][0]
    ring = np.array(feature["geometry"]["coordinates"][0])
    assert 0.5 * np.sum(ring[:-1, 0] * ring[1:, 1] - ring[1:, 0] * ring[:-1, 1]) > 0
    assert feature["properties"]["split"] is None
    assert "class_pixels" not in feature["properties"]


def test_footprints_need_a_patch_size_and_a_transform(tmp_path: Path) -> None:
    manifest = _manifest()
    manifest.sampler = None
    with pytest.raises(ValueError, match="patch size"):
        write_footprints(manifest, None, tmp_path / "p.geojson")
    manifest = Manifest(sampler={"patch_size": 8})
    manifest.patches = [
        {"row": 0, "col": 0, "padded": False, "chunk": 0, "files": {}, "summary": {}}
    ]
    with pytest.raises(ValueError, match="transform"):
        write_footprints(manifest, None, tmp_path / "p.geojson")


def test_gap_in_writer_arguments_is_reported(tmp_path: Path) -> None:
    images = np.zeros((1, 8, 8, 3), np.uint8)
    config = WriterConfig(staging_dir=tmp_path, mask_format="npy")
    with pytest.raises(ValueError, match=r"\(1, H, W\) array"):
        write_patches(images, np.zeros((2, 8, 8), np.uint8), [_meta()], config, _manifest())
    with pytest.raises(ValueError, match="numeric"):
        write_patches(images, np.zeros((1, 8, 8), dtype=object), [_meta()], config, _manifest())
    tif = WriterConfig(staging_dir=tmp_path, mask_format="tif")
    with pytest.raises(ValueError, match="cannot hold bool"):
        write_patches(images, np.zeros((1, 8, 8), np.bool_), [_meta()], tif, _manifest())
    no_crs = Manifest(sources=[SourceRecord(transform=MERCATOR)], sampler={"patch_size": 8})
    with pytest.raises(ValueError, match="no CRS"):
        write_patches(images, np.zeros((1, 8, 8), np.uint8), [_meta()], tif, no_crs)
    with pytest.raises(ValueError, match="no transform"):
        write_patches(
            images, None, [_meta()], WriterConfig(staging_dir=tmp_path, image_format="tif"),
            _manifest(transform=None),
        )  # fmt: skip


def test_png_mask_counts_for_wide_and_signed_masks(tmp_path: Path) -> None:
    manifest = _manifest()
    manifest.target = TargetRecord(type="segmentation", ignore_index=None, dtype="uint16")
    config = WriterConfig(staging_dir=tmp_path, mask_format="npy")
    masks = np.zeros((1, 8, 8), np.int32)
    masks[0, :2] = 70000
    masks[0, 2:4] = -3
    write_patches(np.zeros((1, 8, 8, 3), np.uint8), masks, [_meta()], config, manifest)
    assert manifest.patches[0]["summary"]["class_pixels"] == {"-3": 16, "0": 32, "70000": 16}


def test_a_source_nodata_is_declared_only_when_the_dtype_can_hold_it(tmp_path: Path) -> None:
    from mapcv.writer import _image_nodata

    def nodata(dtype: str, declared: Any) -> Any:
        manifest = _manifest(fingerprint={"nodata": declared})
        return _image_nodata(np.dtype(dtype), manifest)

    assert nodata("uint16", 0) == 0
    assert nodata("uint8", -9999) is None  # out of range
    assert nodata("uint8", 2.5) is None  # not an integer
    assert nodata("uint8", "nan") is None
    assert np.isnan(nodata("float32", "nan"))
    assert nodata("float32", -9999) == -9999
    assert nodata("float32", None) is not None and np.isnan(nodata("float32", None))
    assert nodata("uint8", None) is None
