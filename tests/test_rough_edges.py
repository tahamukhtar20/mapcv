"""Rough edges found by the 0.3.0 release-candidate journey."""

from __future__ import annotations

import json
import math
import warnings
from pathlib import Path

import pytest
from pyproj import Transformer
from shapely.geometry import Polygon

from mapcv.config import RegionConfig
from mapcv.imagery import transform_geometry_to_crs
from mapcv.labels import transform_to_mercator
from mapcv.manifest import Manifest
from mapcv.planning import _pixel_size_m, ground_resolution_m
from mapcv.splitter import SplitterConfig, split_manifest

FIXTURE = Path(__file__).parent / "fixtures" / "mapcv-0.2.0" / "dataset" / "manifest.json"
SQUARE = Polygon([(4.9, 52.37), (4.91, 52.37), (4.91, 52.38), (4.9, 52.37)])


def test_reprojection_raises_no_deprecation_warning() -> None:
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        utm = transform_geometry_to_crs(SQUARE, "EPSG:32631")
        mercator = transform_to_mercator(SQUARE)
    to_utm = Transformer.from_crs("EPSG:4326", "EPSG:32631", always_xy=True)
    assert utm.exterior.coords[0] == pytest.approx(to_utm.transform(4.9, 52.37))
    x, y = mercator.exterior.coords[1]
    assert x == pytest.approx(math.radians(4.91) * 6378137.0)
    assert y == pytest.approx(math.log(math.tan(math.pi / 4 + math.radians(52.37) / 2)) * 6378137.0)


def test_web_mercator_pixels_are_measured_on_the_ground() -> None:
    x, y = Transformer.from_crs(4326, 3857, always_xy=True).transform(4.94, 52.375)
    size = 2 * math.pi * 6378137.0 / 256 / 2**18
    measured = _pixel_size_m("EPSG:3857", (size, 0.0, x, 0.0, -size, y))
    assert measured == pytest.approx(ground_resolution_m(18, 52.375), rel=1e-6)
    assert size == pytest.approx(0.597, abs=1e-3)  # what the card used to call metres


def test_manifest_paths_may_be_strings(tmp_path: Path) -> None:
    manifest = Manifest.load(str(FIXTURE))
    manifest.save(str(tmp_path / "manifest.json"))
    assert Manifest.load(tmp_path / "manifest.json").patches == manifest.patches


def test_a_region_with_swapped_axes_is_named(tmp_path: Path) -> None:
    pytest.importorskip("rasterio")
    from test_geotiff_imagery import make_raster

    from mapcv.config import GeoTiffImageryConfig
    from mapcv.imagery import open_geotiff_source

    raster = make_raster(tmp_path)
    box = raster.region()
    swapped = RegionConfig(
        west=box["south"], south=box["west"], east=box["north"], north=box["east"]
    )
    with pytest.raises(ValueError, match="latitude and longitude swapped") as caught:
        open_geotiff_source(swapped, GeoTiffImageryConfig(path=str(raster.path)))
    assert "the file covers lon" in str(caught.value)


def _manifest(tmp_path: Path, patches: int) -> Manifest:
    raw = json.loads(FIXTURE.read_text())
    raw["patches"] = (raw["patches"] * (patches // len(raw["patches"]) + 1))[:patches]
    return Manifest.from_dict(raw)


def test_a_split_that_leaves_train_empty_warns(tmp_path: Path) -> None:
    manifest = _manifest(tmp_path, 6)
    config = SplitterConfig(strategy="random", test_ratio=0.9, val_ratio=0.9)
    with pytest.warns(UserWarning, match="No patch is left for train"):
        counts, _ = split_manifest(manifest, config, tmp_path / "splits")
    assert counts["train"] == 0


def test_a_balanced_random_split_does_not_warn(tmp_path: Path) -> None:
    manifest = _manifest(tmp_path, 40)
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        split_manifest(manifest, SplitterConfig(strategy="random"), tmp_path / "splits")
