"""Several GeoTIFFs as one raster (``imagery.path`` with a glob pattern).

The reference is the raster the tiles were cut from, and ``rasterio.merge`` where tiles
overlap or leave gaps: a mosaic must give the same pixels, and a dataset generated from the
tiles the same bytes as one generated from the whole raster.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pytest
from pydantic import ValidationError

pytest.importorskip("rasterio", reason="the tiles are written and merged with rasterio")
import rasterio
import rasterio.merge
from rasterio.transform import Affine
from rasterio.windows import Window
from test_geotiff_imagery import config_for, make_raster, write_labels

from mapcv.cli import app
from mapcv.config import GeoTiffImageryConfig, RegionConfig
from mapcv.imagery import (
    GeoTiffMosaicSource,
    GeoTiffRasterSource,
    open_geotiff_source,
)
from mapcv.pipeline import run_generate


def _cut(source: Path, folder: Path, rows: int, cols: int, overlap: int = 0) -> list[Path]:
    """Cut ``source`` into ``rows`` x ``cols`` tiles (each grown by ``overlap`` pixels)."""
    folder.mkdir(parents=True, exist_ok=True)
    paths = []
    with rasterio.open(source) as src:
        height, width = src.height, src.width
        for r in range(rows):
            for c in range(cols):
                row0 = max(0, r * height // rows - overlap)
                row1 = min(height, (r + 1) * height // rows + overlap)
                col0 = max(0, c * width // cols - overlap)
                col1 = min(width, (c + 1) * width // cols + overlap)
                window = Window(col0, row0, col1 - col0, row1 - row0)
                profile = src.profile | {
                    "height": row1 - row0,
                    "width": col1 - col0,
                    "transform": src.window_transform(window),
                    "tiled": False,
                }
                profile.pop("blockxsize", None)
                profile.pop("blockysize", None)
                path = folder / f"tile_{r}_{c}.tif"
                with rasterio.open(path, "w", **profile) as dst:
                    dst.write(src.read(window=window))
                paths.append(path)
    return paths


def _region(raster: Any) -> RegionConfig:
    return RegionConfig.model_validate(raster.region(margin=0.0))


def _read_all(source: Any) -> tuple[np.ndarray, np.ndarray]:
    meta = source.metadata
    data, valid = source.read_window(0, meta.height, 0, meta.width)
    return np.asarray(data), np.asarray(valid)


def test_tiles_cut_from_a_raster_read_back_as_the_raster(tmp_path: Path) -> None:
    raster = make_raster(tmp_path, width=640, height=512, count=3)
    _cut(raster.path, tmp_path / "tiles", 3, 4)
    region = _region(raster)
    single = open_geotiff_source(region, GeoTiffImageryConfig(path=str(raster.path)))
    mosaic = open_geotiff_source(
        region, GeoTiffImageryConfig(path=str(tmp_path / "tiles" / "*.tif"))
    )
    assert isinstance(single, GeoTiffRasterSource) and isinstance(mosaic, GeoTiffMosaicSource)
    assert mosaic.metadata.transform == single.metadata.transform
    assert (mosaic.metadata.height, mosaic.metadata.width) == (
        single.metadata.height,
        single.metadata.width,
    )
    a, valid_a = _read_all(single)
    b, valid_b = _read_all(mosaic)
    assert np.array_equal(a, b) and np.array_equal(valid_a, valid_b)
    assert mosaic.metadata.product_id == "*.tif (12 files)"


def test_overlaps_and_gaps_match_rasterio_merge(tmp_path: Path) -> None:
    raster = make_raster(tmp_path, width=600, height=480, count=2, dtype="uint16", nodata=0)
    tiles = _cut(raster.path, tmp_path / "tiles", 2, 3, overlap=7)
    # Make overlaps visible: each tile gets its own values, so the "first wins" rule shows.
    for index, path in enumerate(tiles):
        with rasterio.open(path, "r+") as dst:
            dst.write((dst.read() // 4 + 1 + index * 1000).astype("uint16"))  # never 0
    # A NoData hole in the first tile, inside its overlap with the second: the second
    # tile's pixels show there.
    with rasterio.open(tiles[0], "r+") as dst:
        data = dst.read()
        data[:, 100:180, dst.width - 12 : dst.width] = 0
        dst.write(data)
    tiles[4].unlink()  # a gap in the middle of the mosaic
    files = sorted(tmp_path.joinpath("tiles").glob("*.tif"))
    merged, merged_transform = rasterio.merge.merge([str(p) for p in files], nodata=0)
    source = open_geotiff_source(
        _region(raster), GeoTiffImageryConfig(path=str(tmp_path / "tiles" / "*.tif"))
    )
    data, valid = _read_all(source)
    a, _, c, _, e, f = source.metadata.transform
    col0 = round((c - merged_transform.c) / a)
    row0 = round((f - merged_transform.f) / e)
    expected = np.moveaxis(merged, 0, -1)[row0 : row0 + data.shape[0], col0 : col0 + data.shape[1]]
    assert np.array_equal(data, expected)
    assert np.array_equal(valid, np.any(expected != 0, axis=-1))
    assert not valid.all()  # the gap is no imagery
    # The hole (rows 100-180, the last 12 columns of tile 0 at 207 px wide) shows tile 1.
    hole = merged[:, 100:180, 195:207]
    assert (hole >= 1000).all() and (hole < 2000).all()


def test_a_dataset_from_tiles_equals_one_from_the_whole_raster(tmp_path: Path) -> None:
    raster = make_raster(tmp_path, width=640, height=512, count=3)
    _cut(raster.path, tmp_path / "tiles", 2, 3)
    region = raster.region()
    labels = write_labels(tmp_path, region)
    whole = config_for(tmp_path, {"path": str(raster.path)}, region, labels=labels, staging="a")
    tiles = config_for(
        tmp_path,
        {"path": str(tmp_path / "tiles" / "*.tif")},
        region,
        labels=labels,
        staging="b",
    )
    run_generate(whole)
    run_generate(tiles)
    for folder in ("Images", "Masks"):
        names = sorted(p.name for p in (tmp_path / "a" / folder).iterdir())
        assert names == sorted(p.name for p in (tmp_path / "b" / folder).iterdir())
        for name in names:
            assert (tmp_path / "a" / folder / name).read_bytes() == (
                tmp_path / "b" / folder / name
            ).read_bytes()


def _retag(path: Path, **changes: Any) -> None:
    """Rewrite a tile with another transform, CRS or dtype."""
    with rasterio.open(path) as src:
        data, profile = src.read(), src.profile
    if "dtype" in changes:
        data = data.astype(changes["dtype"])
    profile.update(changes)
    with rasterio.open(path, "w", **profile) as dst:
        dst.write(data)


@pytest.mark.parametrize(
    ("change", "message"),
    [
        ("shift", "not on the same pixel grid"),
        ("scale", "pixel size"),
        ("crs", "CRS"),
        ("dtype", "data type"),
        ("nodata", "NoData"),
    ],
)
def test_files_that_do_not_fit_together_are_refused(
    tmp_path: Path, change: str, message: str
) -> None:
    raster = make_raster(tmp_path, width=320, height=256, count=3)
    tiles = _cut(raster.path, tmp_path / "tiles", 1, 2)
    with rasterio.open(tiles[1]) as src:
        t = src.transform
    changes: dict[str, Any] = {
        "shift": {"transform": Affine(t.a, 0, t.c + t.a / 2, 0, t.e, t.f)},
        "scale": {"transform": Affine(t.a * 2, 0, t.c, 0, t.e * 2, t.f)},
        "crs": {"crs": "EPSG:32632"},
        "dtype": {"dtype": "uint16"},
        "nodata": {"nodata": 7},
    }[change]
    _retag(tiles[1], **changes)
    config = GeoTiffImageryConfig(path=str(tmp_path / "tiles" / "*.tif"))
    with pytest.raises(ValueError, match=message):
        open_geotiff_source(_region(raster), config)
    if change == "nodata":  # one NoData value for all files makes it acceptable
        fixed = GeoTiffImageryConfig(path=str(tmp_path / "tiles" / "*.tif"), nodata=7)
        assert isinstance(open_geotiff_source(_region(raster), fixed), GeoTiffMosaicSource)


def test_patterns_need_local_files_that_exist(tmp_path: Path) -> None:
    with pytest.raises(ValidationError, match="local files only"):
        GeoTiffImageryConfig(path="https://example.com/tiles/*.tif")
    with pytest.raises(FileNotFoundError, match="no files match"):
        GeoTiffImageryConfig(path=str(tmp_path / "none" / "*.tif")).files()
    raster = make_raster(tmp_path, name="one.tif")
    nested = tmp_path / "survey" / "day1"
    _cut(raster.path, nested, 1, 2)
    found = GeoTiffImageryConfig(path=str(tmp_path / "survey" / "**" / "*.tif")).files()
    assert [Path(p).name for p in found] == ["tile_0_0.tif", "tile_0_1.tif"]


def test_validate_accepts_a_pattern_without_a_missing_file_warning(tmp_path: Path) -> None:
    from typer.testing import CliRunner

    raster = make_raster(tmp_path, width=320, height=256, count=3)
    _cut(raster.path, tmp_path / "tiles", 1, 2)
    region = raster.region()
    config = tmp_path / "mapcv.yaml"
    config.write_text(
        f"region: {{west: {region['west']}, south: {region['south']}, east: {region['east']}, "
        f"north: {region['north']}}}\n"
        "imagery: {type: geotiff, path: 'tiles/*.tif'}\n"
        "sampler: {patch_size: 64}\nwriter: {staging_dir: out}\n",
        encoding="utf-8",
    )
    runner = CliRunner()
    result = runner.invoke(app, ["validate", str(config)])
    assert result.exit_code == 0, result.output
    assert "not found" not in result.output
    planned = runner.invoke(app, ["plan", str(config)])
    assert planned.exit_code == 0, planned.output
    assert "2 files" in " ".join(planned.output.split())
