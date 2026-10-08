"""GeoTIFF imagery inputs that used to be accepted and misread: a file named like a glob
pattern, a CRS that is not a map CRS, an ``imagery.nodata`` the data type cannot hold, and
mosaic overviews that do not share a grid."""

from __future__ import annotations

import shutil
import warnings
from pathlib import Path
from typing import Any

import numpy as np
import pytest

pytest.importorskip("rasterio", reason="the rasters are written with rasterio")
import rasterio
from rasterio.enums import Resampling
from test_geotiff_imagery import make_raster
from test_geotiff_mosaic import _cut, _region, _retag

from mapcv._mapcv_rs import write_geotiffs
from mapcv.config import GeoTiffImageryConfig, RegionConfig
from mapcv.imagery import open_geotiff_source


def _read_all(source: Any) -> np.ndarray:
    meta = source.metadata
    data, _ = source.read_window(0, meta.height, 0, meta.width)
    return np.asarray(data)


def test_a_file_named_like_a_pattern_is_read_as_that_file(tmp_path: Path) -> None:
    wanted = make_raster(tmp_path, name="img[12].tif", seed=1)
    make_raster(tmp_path, name="img1.tif", seed=2)  # what the name would match as a pattern
    source = open_geotiff_source(
        _region(wanted), GeoTiffImageryConfig(path=str(tmp_path / "img[12].tif"))
    )
    with rasterio.open(wanted.path) as reference:
        expected = reference.read().transpose(1, 2, 0)
    assert np.array_equal(_read_all(source), expected)


def test_a_folder_named_like_a_pattern_holds_a_mosaic(tmp_path: Path) -> None:
    raster = make_raster(tmp_path, name="whole.tif", width=320, height=256)
    _cut(raster.path, tmp_path / "survey [2024]", 1, 2)
    decoy = tmp_path / "survey 2"
    decoy.mkdir()
    shutil.copy(raster.path, decoy / "tile_0_0.tif")
    config = GeoTiffImageryConfig(path=str(tmp_path / "survey [2024]" / "*.tif"))
    assert len(config.files()) == 2
    source = open_geotiff_source(_region(raster), config)
    with rasterio.open(raster.path) as reference:
        assert np.array_equal(_read_all(source), reference.read().transpose(1, 2, 0))


def _write_tagged(path: Path, epsg: int) -> Path:
    """A small GeoTIFF whose CRS key is ``epsg``, whatever kind of CRS that is."""
    path.parent.mkdir(parents=True, exist_ok=True)
    data = np.full(3 * 16 * 16, 100, np.uint8)
    transform = [10.0, 0.0, 500000.0, 0.0, -10.0, 1000.0]
    write_geotiffs(
        data, "uint8", (1, 3, 16, 16), [transform], [path.name], str(path.parent), epsg, False
    )
    return path


@pytest.mark.parametrize("epsg", [4978, 5703])
def test_a_file_with_a_non_horizontal_crs_is_rejected_by_name(tmp_path: Path, epsg: int) -> None:
    path = _write_tagged(tmp_path / "odd.tif", epsg)
    region = RegionConfig(west=0.0, south=0.0, east=1.0, north=1.0)
    with pytest.raises(ValueError, match=rf"odd\.tif.*EPSG:{epsg}.*not a map CRS"):
        open_geotiff_source(region, GeoTiffImageryConfig(path=str(path)))


def test_a_non_horizontal_crs_is_rejected_in_a_mosaic(tmp_path: Path) -> None:
    _write_tagged(tmp_path / "tiles" / "a.tif", 5703)
    _write_tagged(tmp_path / "tiles" / "b.tif", 5703)
    region = RegionConfig(west=0.0, south=0.0, east=1.0, north=1.0)
    with pytest.raises(ValueError, match="not a map CRS"):
        open_geotiff_source(region, GeoTiffImageryConfig(path=str(tmp_path / "tiles" / "*.tif")))


def test_a_map_crs_is_still_accepted(tmp_path: Path) -> None:
    path = _write_tagged(tmp_path / "ok.tif", 32631)
    region = RegionConfig(west=3.0001, south=0.0078, east=3.001, north=0.0088)
    source = open_geotiff_source(region, GeoTiffImageryConfig(path=str(path)))
    assert source.metadata.crs == "EPSG:32631"


def test_nodata_the_data_type_cannot_hold_is_reported(tmp_path: Path) -> None:
    raster = make_raster(tmp_path, dtype="uint16")
    for nodata in (-1.0, 0.5, 70000.0):
        config = GeoTiffImageryConfig(path=str(raster.path), nodata=nodata)
        with pytest.warns(UserWarning, match="cannot occur.*uint16"):
            open_geotiff_source(_region(raster), config)


@pytest.mark.parametrize("nodata", [0.0, 65535.0, float("nan")])
def test_nodata_the_data_type_can_hold_is_not_reported(tmp_path: Path, nodata: float) -> None:
    raster = make_raster(tmp_path, dtype="uint16")
    with warnings.catch_warnings():
        warnings.filterwarnings("error", message=".*cannot occur.*")
        open_geotiff_source(
            _region(raster), GeoTiffImageryConfig(path=str(raster.path), nodata=nodata)
        )


def test_float_nodata_beyond_the_float_range_is_reported(tmp_path: Path) -> None:
    raster = make_raster(tmp_path, dtype="float32")
    with pytest.warns(UserWarning, match="cannot occur.*float32"):
        open_geotiff_source(
            _region(raster), GeoTiffImageryConfig(path=str(raster.path), nodata=1e300)
        )


def _mosaic_with_overviews(tmp_path: Path, width: int) -> tuple[Any, GeoTiffImageryConfig]:
    raster = make_raster(tmp_path, name="whole.tif", width=width, height=256)
    for tile in _cut(raster.path, tmp_path / "tiles", 1, 2):
        with rasterio.open(tile, "r+") as dataset:
            dataset.build_overviews([2], Resampling.average)
    return raster, GeoTiffImageryConfig(path=str(tmp_path / "tiles" / "*.tif"), overview=1)


def test_overviews_that_do_not_share_a_grid_are_not_blamed_on_the_pixel_size(
    tmp_path: Path,
) -> None:
    # Tiles 512 and 513 px wide: their half-resolution overviews are 256 and 257 px wide.
    raster, config = _mosaic_with_overviews(tmp_path, 1025)
    with pytest.raises(ValueError, match=r"overviews do not share one grid.*imagery.overview: 0"):
        open_geotiff_source(_region(raster), config)
    full = config.model_copy(update={"overview": 0})
    assert open_geotiff_source(_region(raster), full).metadata.width > 0


def test_tiles_with_a_different_pixel_size_still_get_the_resample_message(
    tmp_path: Path,
) -> None:
    raster = make_raster(tmp_path, name="whole.tif", width=320, height=256)
    tiles = _cut(raster.path, tmp_path / "tiles", 1, 2)
    with rasterio.open(tiles[1]) as source:
        t = source.transform
    _retag(tiles[1], transform=rasterio.Affine(t.a * 2, 0, t.c, 0, t.e * 2, t.f))
    config = GeoTiffImageryConfig(path=str(tmp_path / "tiles" / "*.tif"), overview=0)
    with pytest.raises(ValueError, match="resample the files to one pixel size"):
        open_geotiff_source(_region(raster), config)
