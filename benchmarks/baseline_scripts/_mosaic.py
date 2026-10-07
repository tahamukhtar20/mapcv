"""Fetch a region's XYZ tiles into one Web Mercator GeoTIFF (for baselines whose tool
reads rasters but does not fetch tiles itself).

The same fetch as ``rasterio_script.py``: every tile the region touches, concurrently,
with the given number of connections, stitched on the zoom level's pixel grid.
"""

from __future__ import annotations

import io
import math
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import mercantile
import numpy as np
import rasterio
from PIL import Image
from rasterio.transform import Affine

TILE = 256
EARTH = 2 * math.pi * 6378137.0


def fetch_geotiff(
    url: str,
    zoom: int,
    bounds: tuple[float, float, float, float],
    connections: int,
    path: Path,
) -> None:
    west, south, east, north = bounds
    tiles = list(mercantile.tiles(west, south, east, north, [zoom]))
    xs = sorted({t.x for t in tiles})
    ys = sorted({t.y for t in tiles})
    width, height = len(xs) * TILE, len(ys) * TILE

    def fetch(tile: mercantile.Tile) -> tuple[mercantile.Tile, bytes]:
        address = (
            url.replace("{z}", str(tile.z)).replace("{x}", str(tile.x)).replace("{y}", str(tile.y))
        )
        with urllib.request.urlopen(address, timeout=30) as response:
            return tile, response.read()

    image = np.zeros((3, height, width), dtype=np.uint8)
    with ThreadPoolExecutor(max_workers=connections) as pool:
        for tile, data in pool.map(fetch, tiles):
            pixels = np.asarray(Image.open(io.BytesIO(data)).convert("RGB"))
            row, col = (tile.y - ys[0]) * TILE, (tile.x - xs[0]) * TILE
            image[:, row : row + TILE, col : col + TILE] = np.moveaxis(pixels, -1, 0)

    left, top = mercantile.xy(*mercantile.ul(xs[0], ys[0], zoom))
    size = EARTH / (TILE * 2**zoom)
    with rasterio.open(
        path,
        "w",
        driver="GTiff",
        width=width,
        height=height,
        count=3,
        dtype="uint8",
        crs="EPSG:3857",
        transform=Affine(size, 0.0, left, 0.0, -size, top),
        tiled=True,
    ) as dst:
        dst.write(image)
