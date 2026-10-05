"""Tests for dry-run planning estimates."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict

import pytest

from mapcv._mapcv_rs import snap_bbox, tiles
from mapcv.config import MapcvConfig
from mapcv.planning import (
    LARGE_JOB_TILES,
    ground_resolution_m,
    human_bytes,
    plan,
    region_size_km,
)


def _config(**overrides: Any) -> MapcvConfig:
    data: Dict[str, Any] = {
        "region": {"west": 74.2, "south": 31.4, "east": 74.4, "north": 31.6},
        "imagery": {"type": "xyz", "zoom": 15, "source": "esri_satellite"},
        "sampler": {"patch_size": 256},
        "writer": {"staging_dir": "out"},
    }
    data.update(overrides)
    return MapcvConfig.model_validate(data)


@pytest.mark.parametrize("zoom", [12, 15, 17])
def test_tile_count_matches_enumeration(zoom: int) -> None:
    config = _config(imagery={"type": "xyz", "zoom": zoom, "source": "esri_satellite"})
    snapped = snap_bbox(74.2, 31.4, 74.4, 31.6, zoom)
    expected = len(tiles(snapped.west, snapped.south, snapped.east, snapped.north, [zoom]))
    estimate = plan(config)
    assert estimate.tiles == expected
    width, height = estimate.raster_px
    assert width * height == expected * 256 * 256


def test_grid_patch_count_follows_the_raster() -> None:
    estimate = plan(_config())
    width, height = estimate.raster_px
    assert estimate.patches == (width // 256) * (height // 256)
    assert estimate.download_bytes and estimate.output_bytes > 0


def test_random_mode_uses_random_count() -> None:
    config = _config(sampler={"patch_size": 256, "mode": "random", "random_count": 37})
    assert plan(config).patches == 37


def test_random_mode_estimate_is_capped_at_the_distinct_positions() -> None:
    config = _config(sampler={"patch_size": 256, "mode": "random", "random_count": 10**9})
    estimate = plan(config)
    width, height = estimate.raster_px
    assert estimate.patches == (width - 255) * (height - 255)
    assert any("distinct patch positions" in warning for warning in estimate.warnings)


def test_eopf_estimate_uses_resolution() -> None:
    config = _config(
        region={"west": 10.0, "south": 45.0, "east": 10.1, "north": 45.1},
        imagery={"type": "eopf_zarr", "path": "/data/S2.zarr", "resolution": 10},
        writer={"staging_dir": "out", "image_format": "npy"},
    )
    estimate = plan(config)
    width, height = estimate.raster_px
    width_km, height_km = region_size_km(10.0, 45.0, 10.1, 45.1)
    # UTM grid (with pyproj) and the km approximation agree within a few pixels.
    assert width == pytest.approx(width_km * 100, rel=0.03)
    assert height == pytest.approx(height_km * 100, rel=0.03)
    assert estimate.tiles is None


def test_large_jobs_are_flagged() -> None:
    small = plan(_config())
    assert not small.is_large
    big = plan(_config(imagery={"type": "xyz", "zoom": 19, "source": "esri_satellite"}))
    assert big.tiles is not None and big.tiles > LARGE_JOB_TILES
    assert big.is_large


def test_labels_are_summarized(tmp_path: Path) -> None:
    labels = tmp_path / "labels.geojson"
    labels.write_text(
        '{"type":"FeatureCollection","features":[{"type":"Feature","properties":{"k":"roof"},'
        '"geometry":{"type":"Polygon","coordinates":[[[74.3,31.5],[74.31,31.5],'
        "[74.31,31.51],[74.3,31.5]]]}}]}"
    )
    estimate = plan(_config(labels={"path": str(labels), "label_field": "k"}))
    assert estimate.labels is not None
    assert estimate.labels.polygons == 1
    assert estimate.labels.classes == {"roof": 1}


def test_missing_label_file_is_a_warning_not_a_crash(tmp_path: Path) -> None:
    estimate = plan(_config(labels={"path": str(tmp_path / "missing.geojson")}))
    assert any("not found" in warning for warning in estimate.warnings)


def test_helpers() -> None:
    assert ground_resolution_m(0, 0.0) == pytest.approx(156_543.03, rel=1e-4)
    assert ground_resolution_m(17, 0.0) == pytest.approx(1.194, rel=1e-3)
    assert human_bytes(999) == "999 B"
    assert human_bytes(1_500_000) == "1.5 MB"


def test_labels_outside_the_region_are_flagged(tmp_path: Path) -> None:
    labels = tmp_path / "far.geojson"
    labels.write_text(
        '{"type":"FeatureCollection","features":[{"type":"Feature","properties":{},'
        '"geometry":{"type":"Polygon","coordinates":[[[10,45],[10.1,45],[10.1,45.1],[10,45]]]}}]}'
    )
    estimate = plan(_config(labels={"path": str(labels)}))
    assert any("no label polygon intersects the region" in w for w in estimate.warnings)


def test_output_estimate_follows_the_writer_formats() -> None:
    region = {"west": 10.0, "south": 45.0, "east": 10.1, "north": 45.1}
    imagery = {"type": "eopf_zarr", "path": "/data/S2.zarr", "resolution": 10}

    def output(**writer: str) -> int:
        config = _config(region=region, imagery=imagery, writer={"staging_dir": "out", **writer})
        return plan(config).output_bytes

    npy, tif = output(image_format="npy"), output(image_format="tif")
    assert 0.5 * npy < tif < npy  # deflate takes a fifth off float32 bands
    assert output(image_format="npy", mask_format="npy") == output(image_format="npy")  # no labels
