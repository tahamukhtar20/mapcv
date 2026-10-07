"""The GDAL command-line way to build mapcv's dataset (the classic reference pipeline).

1. ``gdal_translate`` reads the region from the XYZ server through GDAL's TMS driver
   (``<GDAL_WMS>`` description, the same number of connections as mapcv) into a
   GeoTIFF on the zoom level's Web Mercator pixel grid, whole tiles only.
2. ``ogr2ogr`` reprojects the labels to EPSG:3857 and numbers the classes
   (sorted names from 1, as mapcv does without ``labels.classes``).
3. ``gdal_rasterize`` burns them into a mask on the same grid.
4. ``gdal_translate -srcwin`` cuts every ``patch x patch`` window (the given stride,
   whole windows only) of image and mask to PNG, one process per file, run in parallel
   on every core as a shell ``xargs -P`` loop would.

Writes ``images/r<row>_c<col>.png`` and ``masks/r<row>_c<col>.png`` like the other
baselines. GDAL's programs are looked up in ``GDAL_BIN`` (a folder), else on ``PATH``.

Usage: gdal_cli.py TILE_URL ZOOM WEST SOUTH EAST NORTH LABELS PATCH STRIDE CONNECTIONS
       OUTPUT_DIR
"""

from __future__ import annotations

import json
import math
import os
import shutil
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import List

TILE = 256
HALF = math.pi * 6378137.0  # half the Web Mercator world width, in metres


def tool(name: str) -> str:
    folder = os.environ.get("GDAL_BIN")
    found = shutil.which(name, path=folder) if folder else shutil.which(name)
    if found is None:
        raise SystemExit(f"{name} not found (set GDAL_BIN to GDAL's bin folder)")
    return found


def gdal_environment() -> None:
    """PROJ's and GDAL's data folders next to the programs, as activating a conda
    environment would set them (GDAL fails to read any CRS without them)."""
    folder = os.environ.get("GDAL_BIN")
    prefix = Path(folder).resolve().parent if folder else None
    if prefix is None:
        return
    for variable, data in (("PROJ_DATA", "share/proj"), ("GDAL_DATA", "share/gdal")):
        if variable not in os.environ and (prefix / data).is_dir():
            os.environ[variable] = str(prefix / data)


def run(*args: str) -> None:
    subprocess.run(args, check=True, stdout=subprocess.DEVNULL)


def tile_range(lon: float, lat: float, zoom: int) -> tuple[int, int]:
    n = 2**zoom
    x = int((lon + 180.0) / 360.0 * n)
    y = int((1.0 - math.asinh(math.tan(math.radians(lat))) / math.pi) / 2.0 * n)
    return min(x, n - 1), min(y, n - 1)


def main() -> None:
    url, zoom, west, south, east, north, labels, patch, stride, connections, out = sys.argv[1:12]
    gdal_environment()
    z, size, step = int(zoom), int(patch), int(stride)
    out_dir = Path(out)
    work = out_dir / "work"
    work.mkdir(parents=True, exist_ok=True)

    # The region's tiles, as mercantile / mapcv count them.
    x0, y0 = tile_range(float(west), float(north), z)
    x1, y1 = tile_range(float(east), float(south), z)
    server = (
        url.replace("{z}", "${z}").replace("{x}", "${x}").replace("{y}", "${y}")
    )  # GDAL's TMS placeholders
    (work / "tms.xml").write_text(
        f"""<GDAL_WMS>
  <Service name="TMS"><ServerUrl>{server}</ServerUrl></Service>
  <DataWindow>
    <UpperLeftX>{-HALF!r}</UpperLeftX><UpperLeftY>{HALF!r}</UpperLeftY>
    <LowerRightX>{HALF!r}</LowerRightX><LowerRightY>{-HALF!r}</LowerRightY>
    <TileLevel>{z}</TileLevel><TileCountX>1</TileCountX><TileCountY>1</TileCountY>
    <YOrigin>top</YOrigin>
  </DataWindow>
  <Projection>EPSG:3857</Projection>
  <BlockSizeX>{TILE}</BlockSizeX><BlockSizeY>{TILE}</BlockSizeY>
  <BandsCount>3</BandsCount>
  <MaxConnections>{int(connections)}</MaxConnections>
  <ZeroBlockHttpCodes>204,404</ZeroBlockHttpCodes>
</GDAL_WMS>
""",
        encoding="utf-8",
    )
    width, height = (x1 - x0 + 1) * TILE, (y1 - y0 + 1) * TILE
    image = work / "image.tif"
    run(
        tool("gdal_translate"),
        "-q",
        "-srcwin",
        str(x0 * TILE),
        str(y0 * TILE),
        str(width),
        str(height),
        "-co",
        "TILED=YES",
        str(work / "tms.xml"),
        str(image),
    )

    names = sorted(
        {
            feature["properties"]["class"]
            for feature in json.loads(Path(labels).read_text(encoding="utf-8"))["features"]
        }
    )
    case = " ".join(f"WHEN '{name}' THEN {index}" for index, name in enumerate(names, start=1))
    layer = Path(labels).stem
    reprojected = work / "labels.gpkg"
    run(
        tool("ogr2ogr"),
        "-t_srs",
        "EPSG:3857",
        "-dialect",
        "SQLite",
        "-sql",
        f'SELECT *, CASE "class" {case} END AS cid FROM "{layer}"',
        str(reprojected),
        str(labels),
    )
    pixel = 2 * HALF / (TILE * 2**z)
    left, top = -HALF + x0 * TILE * pixel, HALF - y0 * TILE * pixel
    mask = work / "mask.tif"
    run(
        tool("gdal_rasterize"),
        "-q",
        "-a",
        "cid",
        "-init",
        "0",
        "-ot",
        "Byte",
        "-a_srs",
        "EPSG:3857",
        "-te",
        repr(left),
        repr(top - height * pixel),
        repr(left + width * pixel),
        repr(top),
        "-ts",
        str(width),
        str(height),
        str(reprojected),
        str(mask),
    )

    images, masks = out_dir / "images", out_dir / "masks"
    images.mkdir(exist_ok=True)
    masks.mkdir(exist_ok=True)
    translate = tool("gdal_translate")
    jobs: List[List[str]] = []
    for row in range(0, height - size + 1, step):
        for col in range(0, width - size + 1, step):
            name = f"r{row}_c{col}.png"
            for source, folder in ((image, images), (mask, masks)):
                jobs.append(
                    [
                        translate,
                        "-q",
                        "-of",
                        "PNG",
                        "-srcwin",
                        str(col),
                        str(row),
                        str(size),
                        str(size),
                        str(source),
                        str(folder / name),
                    ]
                )
    with ThreadPoolExecutor(max_workers=os.cpu_count() or 1) as pool:
        for _ in pool.map(lambda args: run(*args), jobs):
            pass
    # GDAL leaves .aux.xml files next to PNGs it writes; a dataset has none.
    for extra in out_dir.rglob("*.aux.xml"):
        extra.unlink()
    shutil.rmtree(work)


if __name__ == "__main__":
    main()
