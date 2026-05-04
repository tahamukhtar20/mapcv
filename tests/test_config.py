"""Tests for MapcvConfig YAML loading and validation."""

from __future__ import annotations

from pathlib import Path

import pytest

from mapcv.config import MapcvConfig


_MINIMAL = """\
region:
  west: 74.20
  south: 31.40
  east: 74.40
  north: 31.60
  zoom: 16
tiles:
  source: google_satellite
sampler:
  patch_size: 256
writer:
  staging_dir: ./output
"""

_WITH_LABELS = """\
region:
  west: 74.20
  south: 31.40
  east: 74.40
  north: 31.60
  zoom: 16
tiles:
  source: google_satellite
labels:
  path: labels.kml
  label_field: class
  all_touched: true
sampler:
  patch_size: 256
writer:
  staging_dir: ./output
"""

_WITH_SPLIT = """\
region:
  west: 74.20
  south: 31.40
  east: 74.40
  north: 31.60
  zoom: 16
tiles:
  source: google_satellite
sampler:
  patch_size: 256
writer:
  staging_dir: ./output
split:
  test_ratio: 0.20
  val_ratio: 0.10
  labeled_ratios: [0.10, 0.20, 0.30]
  seed: 42
  strategy: stratified
"""


def _write(tmp_path: Path, content: str) -> Path:
    p = tmp_path / "config.yaml"
    p.write_text(content)
    return p


# ---------------------------------------------------------------------------
# Valid configs
# ---------------------------------------------------------------------------


def test_minimal_config_loads(tmp_path: Path) -> None:
    cfg = MapcvConfig.from_yaml(_write(tmp_path, _MINIMAL))
    assert cfg.region.zoom == 16
    assert cfg.tiles.source == "google_satellite"
    assert cfg.sampler.patch_size == 256
    assert cfg.labels is None
    assert cfg.split is None


def test_labels_section_parsed(tmp_path: Path) -> None:
    cfg = MapcvConfig.from_yaml(_write(tmp_path, _WITH_LABELS))
    assert cfg.labels is not None
    assert cfg.labels.path == Path("labels.kml")
    assert cfg.labels.label_field == "class"
    assert cfg.labels.all_touched is True


def test_split_section_parsed(tmp_path: Path) -> None:
    cfg = MapcvConfig.from_yaml(_write(tmp_path, _WITH_SPLIT))
    assert cfg.split is not None
    assert cfg.split.test_ratio == 0.20
    assert cfg.split.labeled_ratios == [0.10, 0.20, 0.30]


def test_tiles_url_template(tmp_path: Path) -> None:
    content = _MINIMAL.replace(
        "source: google_satellite", "url_template: https://example.com/{z}/{x}/{y}.png"
    )
    cfg = MapcvConfig.from_yaml(_write(tmp_path, content))
    assert cfg.tiles.url_template == "https://example.com/{z}/{x}/{y}.png"
    assert cfg.tiles.source is None


def test_sampler_defaults_applied(tmp_path: Path) -> None:
    cfg = MapcvConfig.from_yaml(_write(tmp_path, _MINIMAL))
    assert cfg.sampler.mode == "grid"
    assert cfg.sampler.edge_strategy == "pad"
    assert cfg.sampler.max_empty_ratio == 1.0


def test_writer_defaults_applied(tmp_path: Path) -> None:
    cfg = MapcvConfig.from_yaml(_write(tmp_path, _MINIMAL))
    assert cfg.writer.image_format == "png"
    assert cfg.writer.jpg_quality == 95


# ---------------------------------------------------------------------------
# Validation errors
# ---------------------------------------------------------------------------


def test_missing_region_raises(tmp_path: Path) -> None:
    content = _MINIMAL.replace(
        "region:\n  west: 74.20\n  south: 31.40\n  east: 74.40\n  north: 31.60\n  zoom: 16\n", ""
    )
    with pytest.raises(Exception):
        MapcvConfig.from_yaml(_write(tmp_path, content))


def test_missing_tiles_source_raises(tmp_path: Path) -> None:
    content = _MINIMAL.replace("source: google_satellite", "max_connections: 8")
    with pytest.raises(Exception):
        MapcvConfig.from_yaml(_write(tmp_path, content))


def test_zoom_out_of_range_raises(tmp_path: Path) -> None:
    content = _MINIMAL.replace("zoom: 16", "zoom: 0")
    with pytest.raises(Exception):
        MapcvConfig.from_yaml(_write(tmp_path, content))


def test_missing_patch_size_raises(tmp_path: Path) -> None:
    content = _MINIMAL.replace("  patch_size: 256\n", "")
    with pytest.raises(Exception):
        MapcvConfig.from_yaml(_write(tmp_path, content))


def test_missing_staging_dir_raises(tmp_path: Path) -> None:
    content = _MINIMAL.replace("  staging_dir: ./output\n", "")
    with pytest.raises(Exception):
        MapcvConfig.from_yaml(_write(tmp_path, content))


def test_nonexistent_file_raises(tmp_path: Path) -> None:
    with pytest.raises(Exception):
        MapcvConfig.from_yaml(tmp_path / "no_such_file.yaml")
