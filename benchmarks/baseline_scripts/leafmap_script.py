"""leafmap's way to build mapcv's dataset (the samgeo / geoai notebooks' recipe).

``leafmap.map_tiles_to_geotiff`` downloads the region's XYZ tiles and writes a Web
Mercator GeoTIFF (its own concurrent downloader, through GDAL). leafmap has no
rasterize-and-chip step of its own, so the labels are burned with rasterio on the
GeoTIFF's grid and ``patch x patch`` windows (the given stride, whole windows only) are
cut from both and written as PNG, as those notebooks do with rasterio.

Writes ``images/r<row>_c<col>.png`` and ``masks/r<row>_c<col>.png`` like the other
baselines. Needs leafmap and GDAL's Python bindings (``osgeo``).

Usage: leafmap_script.py TILE_URL ZOOM WEST SOUTH EAST NORTH LABELS PATCH STRIDE
       CONNECTIONS OUTPUT_DIR
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

# GDAL's and PROJ's data in this interpreter's environment, as activating it would set.
for _variable, _data in (("PROJ_DATA", "share/proj"), ("GDAL_DATA", "share/gdal")):
    if _variable not in os.environ and (Path(sys.prefix) / _data).is_dir():
        os.environ[_variable] = str(Path(sys.prefix) / _data)

import leafmap
import mercantile
import numpy as np
import rasterio
from PIL import Image
from pyproj import Transformer
from rasterio.features import rasterize
from shapely.geometry import shape
from shapely.ops import transform as reproject


def main() -> None:
    url, zoom, west, south, east, north, labels, patch, stride, _connections, out = sys.argv[1:12]
    out_dir = Path(out)
    work = out_dir / "work"
    work.mkdir(parents=True, exist_ok=True)
    image_path = work / "image.tif"
    # leafmap picks its own download concurrency; it has no setting for it.
    leafmap.map_tiles_to_geotiff(
        str(image_path),
        [float(west), float(south), float(east), float(north)],
        zoom=int(zoom),
        source=url,
        crs="EPSG:3857",
        quiet=True,
    )
    with rasterio.open(image_path) as src:
        image = np.moveaxis(src.read([1, 2, 3]), 0, -1)
        transform, height, width = src.transform, src.height, src.width

    features = json.loads(Path(labels).read_text(encoding="utf-8"))["features"]
    names = sorted({f["properties"]["class"] for f in features})
    ids = {name: index for index, name in enumerate(names, start=1)}
    to_mercator = Transformer.from_crs("EPSG:4326", "EPSG:3857", always_xy=True).transform
    shapes = [
        (reproject(to_mercator, shape(f["geometry"])), ids[f["properties"]["class"]])
        for f in features
    ]
    mask = rasterize(shapes, out_shape=(height, width), transform=transform, fill=0, dtype="uint8")

    # leafmap may crop to the bbox rather than to whole tiles: cut the windows on the grid
    # that starts at the region's first tile (mapcv's), so patches can be compared.
    tile = mercantile.tile(float(west), float(north), int(zoom))
    left, top = mercantile.xy(*mercantile.ul(tile.x, tile.y, tile.z))
    dy = round((top - transform.f) / transform.a)
    dx = round((transform.c - left) / transform.a)
    size, step = int(patch), int(stride)
    images, masks = out_dir / "images", out_dir / "masks"
    images.mkdir(exist_ok=True)
    masks.mkdir(exist_ok=True)
    first_row, first_col = -(-dy // step) * step, -(-dx // step) * step
    for row in range(first_row, dy + height - size + 1, step):
        for col in range(first_col, dx + width - size + 1, step):
            name = f"r{row}_c{col}.png"
            r, c = row - dy, col - dx
            Image.fromarray(image[r : r + size, c : c + size]).save(images / name)
            Image.fromarray(mask[r : r + size, c : c + size]).save(masks / name)
    image_path.unlink()
    work.rmdir()


if __name__ == "__main__":
    main()
