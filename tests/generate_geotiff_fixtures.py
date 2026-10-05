"""Write the small GeoTIFF fixtures in tests/data/geotiff with rasterio (GDAL).

Run from the repository root with an environment that has rasterio:

    .venv/bin/python tests/generate_geotiff_fixtures.py

The pixel values follow simple formulas that tests/test_geotiff.py recomputes,
so the reader is checked against them without rasterio.
"""

from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np
import rasterio
import rasterio.shutil
from rasterio.crs import CRS
from rasterio.transform import Affine

OUT = Path(__file__).parent / "data" / "geotiff"


def rgb_cog() -> None:
    """48x64 RGB uint8 COG: 16x16 Deflate tiles, predictor 2, one overview."""
    r, c, b = np.meshgrid(np.arange(48), np.arange(64), np.arange(3), indexing="ij")
    data = ((b * 37 + r * 3 + c) % 251).astype(np.uint8).transpose(2, 0, 1)
    with tempfile.TemporaryDirectory() as tmp:
        src_path = Path(tmp) / "src.tif"
        with rasterio.open(
            src_path,
            "w",
            driver="GTiff",
            height=48,
            width=64,
            count=3,
            dtype="uint8",
            crs=CRS.from_epsg(3857),
            transform=Affine(2.5, 0, 1000, 0, -2.5, 2000),
        ) as dst:
            dst.write(data)
        rasterio.shutil.copy(
            src_path,
            OUT / "rgb_u8_deflate_cog_3857.tif",
            driver="COG",
            BLOCKSIZE=16,
            COMPRESS="DEFLATE",
            PREDICTOR="YES",
            OVERVIEWS="IGNORE_EXISTING",
            OVERVIEW_COUNT=1,
            RESAMPLING="NEAREST",
        )


def float_point() -> None:
    """30x20 float32, big-endian, 7-row LZW strips, predictor 3, PixelIsPoint, NaN nodata."""
    r, c = np.meshgrid(np.arange(30), np.arange(20), indexing="ij")
    data = (r + c / 100).astype(np.float32)
    data[::5, ::3] = np.nan
    path = OUT / "f32_point_nan_4326_be.tif"
    with rasterio.open(
        path,
        "w",
        driver="GTiff",
        height=30,
        width=20,
        count=1,
        dtype="float32",
        crs=CRS.from_epsg(4326),
        transform=Affine(0.5, 0, 10, 0, -0.5, 50),
        nodata=float("nan"),
        blockysize=7,
        compress="lzw",
        predictor=3,
        ENDIANNESS="BIG",
    ) as dst:
        dst.update_tags(AREA_OR_POINT="Point")
        dst.write(data, 1)
    with rasterio.open(path) as src:
        assert src.tags()["AREA_OR_POINT"] == "Point"
        assert tuple(src.transform)[:6] == (0.5, 0, 10, 0, -0.5, 50)


def int16_planar_rotated() -> None:
    """24x24x4 int16, band-interleaved 16x16 ZSTD tiles, rotated transform, nodata -9999."""
    r, c, b = np.meshgrid(np.arange(24), np.arange(24), np.arange(4), indexing="ij")
    data = (b * 1000 - r * 50 + c).astype(np.int16).transpose(2, 0, 1)
    with rasterio.open(
        OUT / "i16_planar_rotated_32633.tif",
        "w",
        driver="GTiff",
        height=24,
        width=24,
        count=4,
        dtype="int16",
        crs=CRS.from_epsg(32633),
        transform=Affine(10, 2, 500000, 1.5, -10, 4500000),
        nodata=-9999,
        tiled=True,
        blockxsize=16,
        blockysize=16,
        interleave="band",
        compress="zstd",
    ) as dst:
        dst.write(data)


if __name__ == "__main__":
    OUT.mkdir(parents=True, exist_ok=True)
    rgb_cog()
    float_point()
    int16_planar_rotated()
    for path in sorted(OUT.glob("*.tif")):
        print(f"{path.name}: {path.stat().st_size} bytes")
