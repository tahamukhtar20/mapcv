"""Tests for mapcv configuration loading and migration."""

from __future__ import annotations

from pathlib import Path

import pytest

from mapcv.config import (
    DEFAULT_SENTINEL2_L2A_BANDS,
    EOPFZarrImageryConfig,
    MapcvConfig,
    XYZImageryConfig,
)


_MINIMAL = """\
region:
  west: 74.20
  south: 31.40
  east: 74.40
  north: 31.60
imagery:
  type: xyz
  zoom: 16
  source: osm
sampler:
  patch_size: 256
writer:
  staging_dir: ./output
"""

_LEGACY = """\
region:
  west: 74.20
  south: 31.40
  east: 74.40
  north: 31.60
  zoom: 16
tiles:
  source: osm
sampler:
  patch_size: 256
writer:
  staging_dir: ./output
"""

_ZARR = """\
region:
  west: 10.0
  south: 45.0
  east: 10.2
  north: 45.2
imagery:
  type: eopf_zarr
  path: /data/S2_L2A_PRODUCT.zarr
  resolution: 10
sampler:
  patch_size: 64
writer:
  staging_dir: ./output
  image_format: npy
"""


def _write(tmp_path: Path, content: str) -> Path:
    path = tmp_path / "config.yaml"
    path.write_text(content)
    return path


def test_xyz_config_loads(tmp_path: Path) -> None:
    config = MapcvConfig.from_yaml(_write(tmp_path, _MINIMAL))
    assert isinstance(config.imagery, XYZImageryConfig)
    assert config.imagery.zoom == 16
    assert config.imagery.source == "osm"
    assert config.tiles is None
    assert config.sampler.patch_size == 256


def test_legacy_tiles_config_is_normalized(tmp_path: Path) -> None:
    with pytest.warns(DeprecationWarning, match="0.3.0"):
        config = MapcvConfig.from_yaml(_write(tmp_path, _LEGACY))
    assert isinstance(config.imagery, XYZImageryConfig)
    assert config.imagery.zoom == 16
    assert config.tiles is not None
    assert config.tiles.source == "osm"


def test_xyz_url_template(tmp_path: Path) -> None:
    content = _MINIMAL.replace("source: osm", "url_template: https://example.com/{z}/{x}/{y}.png")
    config = MapcvConfig.from_yaml(_write(tmp_path, content))
    assert isinstance(config.imagery, XYZImageryConfig)
    assert config.imagery.url_template == "https://example.com/{z}/{x}/{y}.png"


def test_eopf_zarr_defaults(tmp_path: Path) -> None:
    config = MapcvConfig.from_yaml(_write(tmp_path, _ZARR))
    assert isinstance(config.imagery, EOPFZarrImageryConfig)
    assert config.imagery.bands == DEFAULT_SENTINEL2_L2A_BANDS
    assert config.imagery.resolution == 10
    assert config.imagery.chunk_rows == 1024


def test_eopf_band_order_is_preserved_and_normalized(tmp_path: Path) -> None:
    content = _ZARR.replace("  resolution: 10\n", "  resolution: 20\n  bands: [B08, B04, B03]\n")
    config = MapcvConfig.from_yaml(_write(tmp_path, content))
    assert isinstance(config.imagery, EOPFZarrImageryConfig)
    assert config.imagery.bands == ["b08", "b04", "b03"]


def test_labels_and_split_sections_parse(tmp_path: Path) -> None:
    content = (
        _MINIMAL
        + "labels:\n  path: labels.kml\n  label_field: class\n  all_touched: true\n"
        + "split:\n  test_ratio: 0.2\n  val_ratio: 0.1\n"
    )
    config = MapcvConfig.from_yaml(_write(tmp_path, content))
    assert config.labels is not None and config.labels.path == Path("labels.kml")
    assert config.split is not None and config.split.test_ratio == 0.2


def test_eopf_requires_npy_writer(tmp_path: Path) -> None:
    with pytest.raises(Exception, match="requires writer.image_format='npy'"):
        MapcvConfig.from_yaml(_write(tmp_path, _ZARR.replace("  image_format: npy\n", "")))


def test_xyz_rejects_npy_writer(tmp_path: Path) -> None:
    content = _MINIMAL.replace(
        "  staging_dir: ./output\n", "  staging_dir: ./output\n  image_format: npy\n"
    )
    with pytest.raises(Exception, match="XYZ imagery supports"):
        MapcvConfig.from_yaml(_write(tmp_path, content))


def test_duplicate_eopf_bands_raise(tmp_path: Path) -> None:
    content = _ZARR.replace("  resolution: 10\n", "  resolution: 10\n  bands: [b04, B04]\n")
    with pytest.raises(Exception, match="duplicates"):
        MapcvConfig.from_yaml(_write(tmp_path, content))


def test_invalid_region_bounds_raise(tmp_path: Path) -> None:
    content = _MINIMAL.replace("  east: 74.40", "  east: 74.10")
    with pytest.raises(Exception, match="west"):
        MapcvConfig.from_yaml(_write(tmp_path, content))


def test_missing_imagery_raises(tmp_path: Path) -> None:
    content = _MINIMAL.replace("imagery:\n  type: xyz\n  zoom: 16\n  source: osm\n", "")
    with pytest.raises(Exception):
        MapcvConfig.from_yaml(_write(tmp_path, content))


def test_nonexistent_file_raises(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        MapcvConfig.from_yaml(tmp_path / "no_such_file.yaml")
