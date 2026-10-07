"""Tests for mapcv configuration loading and migration."""

from __future__ import annotations

import warnings
from pathlib import Path

import pytest
from pydantic import ValidationError

from mapcv.config import (
    DEFAULT_SENTINEL2_L2A_BANDS,
    EOPFZarrImageryConfig,
    LabelsConfig,
    MapcvConfig,
    RegionConfig,
    XYZImageryConfig,
    eopf_local_path,
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
  source: esri_satellite
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
  source: esri_satellite
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
    assert config.imagery.source == "esri_satellite"
    assert config.sampler.patch_size == 256


def test_legacy_tiles_block_is_rejected_with_migration_hint(tmp_path: Path) -> None:
    with pytest.raises(ValidationError) as excinfo:
        MapcvConfig.from_yaml(_write(tmp_path, _LEGACY))
    message = str(excinfo.value)
    assert "`tiles:` was replaced by `imagery:` in mapcv 0.2 and removed in 0.3" in message
    assert "imagery: {type: xyz, zoom: ..., source: ...}" in message
    assert "MIGRATION.md" in message


@pytest.mark.parametrize("content", [_MINIMAL, _ZARR])
def test_region_zoom_is_rejected_with_migration_hint(tmp_path: Path, content: str) -> None:
    content = content.replace("  north: ", "  zoom: 12\n  north: ", 1)
    with pytest.raises(ValidationError) as excinfo:
        MapcvConfig.from_yaml(_write(tmp_path, content))
    message = str(excinfo.value)
    assert "`region.zoom` was moved to `imagery.zoom` in mapcv 0.2 and removed in 0.3" in message
    assert "MIGRATION.md" in message
    assert "region" in excinfo.value.errors()[0]["loc"]


def test_region_zoom_error_applies_to_region_config_directly() -> None:
    with pytest.raises(ValidationError, match="removed in 0.3"):
        RegionConfig.model_validate(
            {"west": 74.2, "south": 31.4, "east": 74.4, "north": 31.6, "zoom": 16}
        )


def test_unknown_region_key_is_still_an_extra_forbidden_error(tmp_path: Path) -> None:
    content = _MINIMAL.replace("  north: ", "  zoom_level: 12\n  north: ", 1)
    with pytest.raises(ValidationError, match="Extra inputs are not permitted"):
        MapcvConfig.from_yaml(_write(tmp_path, content))


def test_config_round_trips_without_warnings(tmp_path: Path) -> None:
    config = MapcvConfig.from_yaml(_write(tmp_path, _MINIMAL))
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        again = MapcvConfig.model_validate(config.model_dump())
    assert again == config


def test_missing_imagery_type_has_clear_error(tmp_path: Path) -> None:
    content = _MINIMAL.replace("  type: xyz\n", "")
    with pytest.raises(ValueError, match="imagery.type is required"):
        MapcvConfig.from_yaml(_write(tmp_path, content))


@pytest.mark.parametrize(
    ("source", "reason"), [("google_satellite", "Google"), ("osm", "forbid bulk downloading")]
)
def test_removed_presets_are_rejected(tmp_path: Path, source: str, reason: str) -> None:
    content = _MINIMAL.replace("source: esri_satellite", f"source: {source}")
    with pytest.raises(ValueError, match=f"removed in mapcv 0.2.0: .*{reason}"):
        MapcvConfig.from_yaml(_write(tmp_path, content))


def test_unknown_tile_source_is_rejected(tmp_path: Path) -> None:
    content = _MINIMAL.replace("source: esri_satellite", "source: nope")
    with pytest.raises(ValueError, match="unknown tile source 'nope'"):
        MapcvConfig.from_yaml(_write(tmp_path, content))


@pytest.mark.parametrize(
    "path",
    [
        "https://user:pw@example.com/S2.zarr",
        "https://example.com/S2.zarr?token=secret",
        "https://example.com/S2.zarr#frag",
        "http://example.com/S2.zarr",
        "gs://bucket/S2.zarr",
    ],
)
def test_eopf_path_rejects_unsafe_urls(path: str) -> None:
    with pytest.raises(ValueError):
        EOPFZarrImageryConfig(path=path)


@pytest.mark.parametrize(
    ("path", "expected"),
    [
        ("/data/S2.zarr", "/data/S2.zarr"),
        (r"C:\data\S2.zarr", r"C:\data\S2.zarr"),
        ("file:///C:/data/S2.zarr", "C:/data/S2.zarr"),
        ("file:///data/S2.zarr", "/data/S2.zarr"),
        ("s3://bucket/S2.zarr", None),
        ("https://example.com/S2.zarr", None),
    ],
)
def test_eopf_local_path(path: str, expected: str | None) -> None:
    EOPFZarrImageryConfig(path=path)
    local = eopf_local_path(path)
    assert (str(local).replace("\\", "/") if local else None) == (
        expected.replace("\\", "/") if expected else None
    )


def test_xyz_url_template(tmp_path: Path) -> None:
    content = _MINIMAL.replace(
        "source: esri_satellite", "url_template: https://example.com/{z}/{x}/{y}.png"
    )
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
    # Relative paths resolve against the config file's folder.
    assert config.labels is not None and config.labels.path == tmp_path / "labels.kml"
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


@pytest.mark.parametrize(
    ("old", "new", "message"),
    [
        ("  west: 74.20", "  west: -200", "longitude in -180..180"),
        ("  east: 74.40", "  east: 181", "longitude in -180..180"),
        ("  south: 31.40", "  south: -95", "latitude in -90..90"),
        # Swapped lon/lat: 74.3 is a valid longitude but not a latitude.
        ("  north: 31.60", "  north: 91", "not swapped"),
        ("  north: 31.60", "  north: 86", "Web Mercator"),
    ],
)
def test_region_must_be_on_the_globe(tmp_path: Path, old: str, new: str, message: str) -> None:
    with pytest.raises(Exception, match=message):
        MapcvConfig.from_yaml(_write(tmp_path, _MINIMAL.replace(old, new)))


def test_web_mercator_limit_applies_only_to_xyz() -> None:
    region = RegionConfig(west=0, south=80, east=1, north=89)
    assert region.north == 89  # valid WGS-84; EOPF products are not limited to Web Mercator


def test_missing_imagery_raises(tmp_path: Path) -> None:
    content = _MINIMAL.replace("imagery:\n  type: xyz\n  zoom: 16\n  source: esri_satellite\n", "")
    with pytest.raises(ValidationError):
        MapcvConfig.from_yaml(_write(tmp_path, content))


def test_nonexistent_file_raises(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        MapcvConfig.from_yaml(tmp_path / "no_such_file.yaml")


@pytest.mark.parametrize(
    ("template", "message"),
    [
        ("https://{s}.tile.example.com/{z}/{x}/{y}.png", "replace {s} with one subdomain"),
        ("https://tiles.example.com/{z}/{x}.png", "missing {y}"),
        ("ftp://tiles.example.com/{z}/{x}/{y}.png", "http:// or https://"),
    ],
)
def test_url_template_is_validated(tmp_path: Path, template: str, message: str) -> None:
    content = _MINIMAL.replace("source: esri_satellite", f'url_template: "{template}"')
    with pytest.raises(ValueError, match=message):
        MapcvConfig.from_yaml(_write(tmp_path, content))


def test_relative_paths_resolve_from_the_config_folder(tmp_path: Path) -> None:
    sub = tmp_path / "configs"
    sub.mkdir()
    content = _MINIMAL.replace("staging_dir: ./output", "staging_dir: ../data/out") + (
        "labels:\n  path: labels.geojson\n"
    )
    config = MapcvConfig.from_yaml(_write(sub, content))
    assert config.writer.staging_dir == tmp_path / "data" / "out"
    assert config.labels is not None and config.labels.path == sub / "labels.geojson"
    absolute = _MINIMAL.replace("staging_dir: ./output", f"staging_dir: {tmp_path / 'abs'}")
    assert MapcvConfig.from_yaml(_write(sub, absolute)).writer.staging_dir == tmp_path / "abs"


def test_from_yaml_accepts_a_string_path(tmp_path: Path) -> None:
    path = _write(tmp_path, _MINIMAL)
    assert MapcvConfig.from_yaml(str(path)) == MapcvConfig.from_yaml(path)


@pytest.mark.parametrize(
    ("original", "typo"),
    [("patch_size: 256", "patch_size: 256\n  stide: 128"), ("zoom: 16", "zoom: 16\n  sorce: x")],
)
def test_unknown_keys_are_rejected(tmp_path: Path, original: str, typo: str) -> None:
    with pytest.raises(ValueError, match="Extra inputs are not permitted"):
        MapcvConfig.from_yaml(_write(tmp_path, _MINIMAL.replace(original, typo)))


def test_source_and_url_template_are_exclusive(tmp_path: Path) -> None:
    content = _MINIMAL.replace(
        "source: esri_satellite",
        'source: esri_satellite\n  url_template: "https://t.example.com/{z}/{x}/{y}.png"',
    )
    with pytest.raises(ValueError, match="not both"):
        MapcvConfig.from_yaml(_write(tmp_path, content))


def test_ignore_index_defaults_to_255_and_can_be_disabled(tmp_path: Path) -> None:
    labels = "labels:\n  path: labels.geojson\n"
    assert MapcvConfig.from_yaml(_write(tmp_path, _MINIMAL + labels)).labels.ignore_index == 255  # type: ignore[union-attr]
    off = MapcvConfig.from_yaml(_write(tmp_path, _MINIMAL + labels + "  ignore_index: null\n"))
    assert isinstance(off.labels, LabelsConfig) and off.labels.ignore_index is None


def test_classes_cannot_use_the_ignore_index(tmp_path: Path) -> None:
    labels = "labels:\n  path: l.geojson\n  label_field: kind\n  classes: {road: 255}\n"
    with pytest.raises(Exception, match="ignore_index"):
        MapcvConfig.from_yaml(_write(tmp_path, _MINIMAL + labels))


def test_task_defaults_to_segmentation(tmp_path: Path) -> None:
    assert MapcvConfig.from_yaml(_write(tmp_path, _MINIMAL)).task == "segmentation"
    explicit = MapcvConfig.from_yaml(_write(tmp_path, "task: segmentation\n" + _MINIMAL))
    assert explicit.task == "segmentation"


def test_every_planned_task_is_supported(tmp_path: Path) -> None:
    from mapcv.config import PLANNED_TASKS, SUPPORTED_TASKS

    assert PLANNED_TASKS == ()
    assert "change" in SUPPORTED_TASKS and "regression" in SUPPORTED_TASKS


@pytest.mark.parametrize("task", ["segmentaton", "Segmentation", 3])
def test_unknown_tasks_are_rejected(tmp_path: Path, task: object) -> None:
    with pytest.raises(ValidationError, match="task"):
        MapcvConfig.from_yaml(_write(tmp_path, f"task: {task}\n" + _MINIMAL))
