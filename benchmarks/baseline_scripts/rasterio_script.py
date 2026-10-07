"""The hand-rolled way to build mapcv's dataset: a well-written rasterio + Pillow script.

What a careful user writes without mapcv: fetch the region's XYZ tiles concurrently
(the same number of connections as mapcv), stitch them, reproject the labels to Web
Mercator with pyproj, rasterize them with rasterio, cut ``patch × patch`` windows with
the given stride (whole windows only) and write PNG image/mask pairs with Pillow.
It writes ``images/r<row>_c<col>.png`` and ``masks/r<row>_c<col>.png``, where row and
col are the window's pixel offset on the stitched raster (mapcv's manifest ``row``/
``col``), so the benchmark can compare the two outputs patch by patch.

Usage: rasterio_script.py TILE_URL ZOOM WEST SOUTH EAST NORTH LABELS PATCH STRIDE
       CONNECTIONS OUTPUT_DIR
"""

from __future__ import annotations

import io
import json
import math
import sys
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import mercantile
import numpy as np
from PIL import Image
from pyproj import Transformer
from rasterio.features import rasterize
from rasterio.transform import Affine
from shapely.geometry import shape
from shapely.ops import transform as reproject

TILE = 256
EARTH = 2 * math.pi * 6378137.0


def main() -> None:
    url, zoom, west, south, east, north, labels, patch, stride, connections, out = sys.argv[1:12]
    zoom_i, patch_i, stride_i = int(zoom), int(patch), int(stride)
    tiles = list(mercantile.tiles(float(west), float(south), float(east), float(north), [zoom_i]))
    xs = sorted({t.x for t in tiles})
    ys = sorted({t.y for t in tiles})
    width, height = len(xs) * TILE, len(ys) * TILE

    def fetch(tile: mercantile.Tile) -> tuple[mercantile.Tile, bytes]:
        address = (
            url.replace("{z}", str(tile.z)).replace("{x}", str(tile.x)).replace("{y}", str(tile.y))
        )
        with urllib.request.urlopen(address, timeout=30) as response:
            return tile, response.read()

    image = np.zeros((height, width, 3), dtype=np.uint8)
    with ThreadPoolExecutor(max_workers=int(connections)) as pool:
        for tile, data in pool.map(fetch, tiles):
            pixels = np.asarray(Image.open(io.BytesIO(data)).convert("RGB"))
            row, col = (tile.y - ys[0]) * TILE, (tile.x - xs[0]) * TILE
            image[row : row + TILE, col : col + TILE] = pixels

    left, top = mercantile.xy(*mercantile.ul(xs[0], ys[0], zoom_i))
    size = EARTH / (TILE * 2**zoom_i)
    transform = Affine(size, 0.0, left, 0.0, -size, top)

    features = json.loads(Path(labels).read_text(encoding="utf-8"))["features"]
    names = sorted({f["properties"]["class"] for f in features})
    ids = {name: index for index, name in enumerate(names, start=1)}
    to_mercator = Transformer.from_crs("EPSG:4326", "EPSG:3857", always_xy=True).transform
    shapes = [
        (reproject(to_mercator, shape(f["geometry"])), ids[f["properties"]["class"]])
        for f in features
    ]
    mask = rasterize(shapes, out_shape=(height, width), transform=transform, fill=0, dtype="uint8")

    images, masks = Path(out) / "images", Path(out) / "masks"
    images.mkdir(parents=True, exist_ok=True)
    masks.mkdir(parents=True, exist_ok=True)
    for row in range(0, height - patch_i + 1, stride_i):
        for col in range(0, width - patch_i + 1, stride_i):
            name = f"r{row}_c{col}.png"
            Image.fromarray(image[row : row + patch_i, col : col + patch_i]).save(images / name)
            Image.fromarray(mask[row : row + patch_i, col : col + patch_i]).save(masks / name)


if __name__ == "__main__":
    main()
