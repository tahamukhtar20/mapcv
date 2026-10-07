"""TorchGeo's way to build mapcv's dataset: its geo-datasets, grid sampler and rasterizer.

TorchGeo reads rasters but does not fetch XYZ tiles, so the tiles are first stitched
into a GeoTIFF (``_mosaic.py``, the same fetch as the script baseline). Then a
``RasterDataset`` (the imagery) intersected with a ``VectorDataset`` (the labels,
rasterized per sample by TorchGeo with rasterio) is sampled by a ``GridGeoSampler``
with the patch size and stride in pixels, through a ``DataLoader`` with a worker per
core, and every sample is written as PNG.

Writes ``images/r<row>_c<col>.png`` and ``masks/r<row>_c<col>.png``, where row and col
come from each sample's transform on the stitched raster.

Usage: torchgeo_script.py TILE_URL ZOOM WEST SOUTH EAST NORTH LABELS PATCH STRIDE
       CONNECTIONS OUTPUT_DIR
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import numpy as np
import rasterio
from PIL import Image
from torch.utils.data import DataLoader
from torchgeo.datasets import RasterDataset, VectorDataset, stack_samples
from torchgeo.samplers import GridGeoSampler

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _mosaic import fetch_geotiff


def main() -> None:
    url, zoom, west, south, east, north, labels, patch, stride, connections, out = sys.argv[1:12]
    out_dir = Path(out)
    work = out_dir / "work"
    work.mkdir(parents=True, exist_ok=True)
    image_path = work / "image.tif"
    bounds = (float(west), float(south), float(east), float(north))
    fetch_geotiff(url, int(zoom), bounds, int(connections), image_path)

    # VectorDataset burns a numeric property: number the classes as mapcv does.
    collection = json.loads(Path(labels).read_text(encoding="utf-8"))
    names = sorted({f["properties"]["class"] for f in collection["features"]})
    ids = {name: index for index, name in enumerate(names, start=1)}
    for feature in collection["features"]:
        feature["properties"]["cid"] = ids[feature["properties"]["class"]]
    numbered = work / "labels.geojson"
    numbered.write_text(json.dumps(collection), encoding="utf-8")

    imagery = RasterDataset(paths=str(image_path))
    masks = VectorDataset(paths=str(numbered), crs=imagery.crs, res=imagery.res, label_name="cid")
    dataset = imagery & masks
    sampler = GridGeoSampler(dataset, size=int(patch), stride=int(stride))
    loader = DataLoader(
        dataset,
        sampler=sampler,
        batch_size=32,
        num_workers=os.cpu_count() or 1,
        collate_fn=stack_samples,
    )

    with rasterio.open(image_path) as src:
        left, top, res = src.transform.c, src.transform.f, src.transform.a
    images, labels_dir = out_dir / "images", out_dir / "masks"
    images.mkdir(exist_ok=True)
    labels_dir.mkdir(exist_ok=True)
    for batch in loader:
        for image, mask, transform in zip(batch["image"], batch["mask"], batch["transform"]):
            values = transform.flatten().tolist()  # rasterio Affine order a, b, c, d, e, f
            col = round((values[2] - left) / res)
            row = round((top - values[5]) / res)
            name = f"r{row}_c{col}.png"
            pixels = np.moveaxis(image.numpy(), 0, -1).round().astype(np.uint8)
            Image.fromarray(pixels).save(images / name)
            Image.fromarray(mask.numpy().astype(np.uint8)).save(labels_dir / name)
    for path in work.iterdir():
        path.unlink()
    work.rmdir()


if __name__ == "__main__":
    main()
