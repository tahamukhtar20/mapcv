"""Golden fixture generator for the MV validation suite.

Run with the GDAL conda env:
  /home/tahamukhtar20/miniconda3/envs/GDAL/bin/python tests/generate_golden.py

Produces tests/golden/{tile_math,transform,rasterize_golden,label_parse}_golden.{json,npz}.
"""

from __future__ import annotations

import io
import json
import random
from pathlib import Path
from typing import Any, Dict, List, Tuple

import geopandas as gpd
import mercantile
import numpy as np
import pyproj
from rasterio.features import rasterize as rio_rasterize
from rasterio.transform import Affine
from shapely.geometry import MultiPolygon, Polygon, mapping
from shapely.wkt import dumps as wkt_dumps

GOLDEN_DIR = Path(__file__).parent / "golden"
GOLDEN_DIR.mkdir(exist_ok=True)

SEED = 42
_rng = random.Random(SEED)

_KML_HEADER = b'<?xml version="1.0" encoding="UTF-8"?><kml xmlns="http://www.opengis.net/kml/2.2">'
_KML_FOOTER = b"</kml>"


def _kml(body: bytes) -> bytes:
    return _KML_HEADER + body + _KML_FOOTER


# ---- KML fixtures (same strings as test_labels.py) ----

BINARY_KML = _kml(
    b"""
    <Document>
      <Placemark>
        <name>Field A</name>
        <Polygon>
          <outerBoundaryIs><LinearRing>
            <coordinates>10,20 11,20 11,21 10,21 10,20</coordinates>
          </LinearRing></outerBoundaryIs>
        </Polygon>
      </Placemark>
    </Document>
"""
)

MULTICLASS_KML = _kml(
    b"""
    <Document>
      <Placemark>
        <name>P1</name>
        <ExtendedData>
          <Data name="land_use"><value>residential</value></Data>
        </ExtendedData>
        <Polygon>
          <outerBoundaryIs><LinearRing>
            <coordinates>0,0 1,0 1,1 0,1 0,0</coordinates>
          </LinearRing></outerBoundaryIs>
        </Polygon>
      </Placemark>
      <Placemark>
        <name>P2</name>
        <ExtendedData>
          <Data name="land_use"><value>industrial</value></Data>
        </ExtendedData>
        <Polygon>
          <outerBoundaryIs><LinearRing>
            <coordinates>2,0 3,0 3,1 2,1 2,0</coordinates>
          </LinearRing></outerBoundaryIs>
        </Polygon>
      </Placemark>
      <Placemark>
        <name>P3</name>
        <ExtendedData>
          <Data name="land_use"><value>residential</value></Data>
        </ExtendedData>
        <Polygon>
          <outerBoundaryIs><LinearRing>
            <coordinates>4,0 5,0 5,1 4,1 4,0</coordinates>
          </LinearRing></outerBoundaryIs>
        </Polygon>
      </Placemark>
    </Document>
"""
)

NESTED_FOLDER_KML = _kml(
    b"""
    <Document>
      <Folder>
        <name>Zone A</name>
        <Folder>
          <name>Sub-zone A1</name>
          <Placemark>
            <name>Nested</name>
            <Polygon>
              <outerBoundaryIs><LinearRing>
                <coordinates>5,5 6,5 6,6 5,6 5,5</coordinates>
              </LinearRing></outerBoundaryIs>
            </Polygon>
          </Placemark>
        </Folder>
      </Folder>
    </Document>
"""
)

HOLE_KML = _kml(
    b"""
    <Document>
      <Placemark>
        <Polygon>
          <outerBoundaryIs><LinearRing>
            <coordinates>0,0 10,0 10,10 0,10 0,0</coordinates>
          </LinearRing></outerBoundaryIs>
          <innerBoundaryIs><LinearRing>
            <coordinates>2,2 8,2 8,8 2,8 2,2</coordinates>
          </LinearRing></innerBoundaryIs>
        </Polygon>
      </Placemark>
    </Document>
"""
)

MULTIGEOMETRY_KML = _kml(
    b"""
    <Document>
      <Placemark>
        <MultiGeometry>
          <Polygon>
            <outerBoundaryIs><LinearRing>
              <coordinates>0,0 1,0 1,1 0,1 0,0</coordinates>
            </LinearRing></outerBoundaryIs>
          </Polygon>
          <Polygon>
            <outerBoundaryIs><LinearRing>
              <coordinates>2,0 3,0 3,1 2,1 2,0</coordinates>
            </LinearRing></outerBoundaryIs>
          </Polygon>
        </MultiGeometry>
      </Placemark>
    </Document>
"""
)

# nested_folder is excluded: geopandas treats each KML Folder as a separate layer
# and cannot serve as a clean oracle. Nested folder parsing is covered in test_labels.py.
_KML_CASES = [
    ("binary", BINARY_KML, None),
    ("multiclass", MULTICLASS_KML, "land_use"),
    ("hole", HOLE_KML, None),
    ("multigeometry", MULTIGEOMETRY_KML, None),
]

# ---- hand-picked coordinate pairs ----

_HAND_PICKED: List[Tuple[float, float]] = [
    (-122.4194, 37.7749),  # San Francisco
    (139.6917, 35.6895),  # Tokyo
    (-43.1729, -22.9068),  # Rio de Janeiro
    (2.3522, 48.8566),  # Paris
    (77.1025, 28.7041),  # Delhi
    (0.0, 0.0),  # null island
    (179.9, 85.0),  # near antimeridian + north pole
    (-179.9, -85.0),  # near antimeridian + south pole
    (90.0, 0.0),  # equator mid
    (-90.0, 45.0),  # North America mid
]

# ---- rasterize transform (1 pixel = 1 unit, north-up, origin at top-left) ----
# x = 1*col + 0*row + 0  ->  col = x
# y = 0*col + (-1)*row + 128  ->  row = 128 - y
_RASTER_SIZE = 128
_RASTER_TRANSFORM = (1.0, 0.0, 0.0, 0.0, -1.0, float(_RASTER_SIZE))


def _random_lng_lat(rng: random.Random) -> Tuple[float, float]:
    lng = rng.uniform(-180.0, 180.0)
    lat = rng.uniform(-85.0, 85.0)
    return lng, lat


def _random_bbox(rng: random.Random) -> Tuple[float, float, float, float]:
    """Return (west, south, east, north) with west < east, max 2 degrees wide/tall."""
    lng1 = rng.uniform(-170.0, 160.0)
    lat1 = rng.uniform(-80.0, 75.0)
    lng2 = lng1 + rng.uniform(0.01, 2.0)
    lat2 = lat1 + rng.uniform(0.01, 2.0)
    return lng1, lat1, min(lng2, 179.9), min(lat2, 84.9)


def _rect_polygon(x1: float, y1: float, x2: float, y2: float) -> Polygon:
    return Polygon([(x1, y1), (x2, y1), (x2, y2), (x1, y2)])


def generate_tile_math_golden() -> None:
    """Write tests/golden/tile_math_golden.json."""
    rng2 = random.Random(SEED)

    # ---- xy section ----
    xy_entries: List[Dict[str, Any]] = []
    coords_500: List[Tuple[float, float]] = list(_HAND_PICKED)
    while len(coords_500) < 500:
        coords_500.append(_random_lng_lat(rng2))

    for lng, lat in coords_500:
        mx, my = mercantile.xy(lng, lat, truncate=False)
        xy_entries.append({"lng": lng, "lat": lat, "mx": mx, "my": my})

    # ---- tile section ----
    tile_entries: List[Dict[str, Any]] = []
    zooms_cycle = list(range(1, 23))
    for i, (lng, lat) in enumerate(coords_500):
        zoom = zooms_cycle[i % len(zooms_cycle)]
        t = mercantile.tile(lng, lat, zoom)
        tile_entries.append({"lng": lng, "lat": lat, "zoom": zoom, "x": t.x, "y": t.y, "z": t.z})

    # ---- xy_bounds + bounds from first 200 tiles ----
    xy_bounds_entries: List[Dict[str, Any]] = []
    bounds_entries: List[Dict[str, Any]] = []
    for entry in tile_entries[:200]:
        t = mercantile.Tile(x=entry["x"], y=entry["y"], z=entry["z"])
        xb = mercantile.xy_bounds(t)
        xy_bounds_entries.append(
            {
                "x": t.x,
                "y": t.y,
                "z": t.z,
                "west": xb.left,
                "south": xb.bottom,
                "east": xb.right,
                "north": xb.top,
            }
        )
        b = mercantile.bounds(t)
        bounds_entries.append(
            {
                "x": t.x,
                "y": t.y,
                "z": t.z,
                "west": b.west,
                "south": b.south,
                "east": b.east,
                "north": b.north,
            }
        )

    # ---- tiles section (50 standard bboxes, west < east only) ----
    # Use zoom <= 14 and stream-with-limit to avoid OOM on high-zoom large bboxes.
    tiles_entries: List[Dict[str, Any]] = []
    rng3 = random.Random(SEED)
    zoom_list = [8, 10, 12]
    _TILE_LIMIT = 5_000
    while len(tiles_entries) < 50:
        w, s, e, n = _random_bbox(rng3)
        zoom = rng3.choice(zoom_list)
        tile_set: List[List[int]] = []
        overflow = False
        for t in mercantile.tiles(w, s, e, n, zooms=[zoom]):
            tile_set.append([t.x, t.y, t.z])
            if len(tile_set) > _TILE_LIMIT:
                overflow = True
                break
        if overflow:
            continue
        tiles_entries.append(
            {"west": w, "south": s, "east": e, "north": n, "zoom": zoom, "tiles": tile_set}
        )

    golden: Dict[str, Any] = {
        "xy": xy_entries,
        "tile": tile_entries,
        "xy_bounds": xy_bounds_entries,
        "bounds": bounds_entries,
        "tiles": tiles_entries,
    }
    out = GOLDEN_DIR / "tile_math_golden.json"
    out.write_text(json.dumps(golden, indent=2))
    print(
        f"  xy: {len(xy_entries)}, tile: {len(tile_entries)}, xy_bounds: {len(xy_bounds_entries)}, "
        f"bounds: {len(bounds_entries)}, tiles: {len(tiles_entries)} bboxes"
    )


def generate_transform_golden() -> None:
    """Write tests/golden/transform_golden.json."""
    transformer = pyproj.Transformer.from_crs("EPSG:4326", "EPSG:3857", always_xy=True)
    rng2 = random.Random(SEED)
    entries: List[Dict[str, Any]] = []
    for _ in range(1000):
        lng = rng2.uniform(-180.0, 180.0)
        lat = rng2.uniform(-85.0, 85.0)
        mx, my = transformer.transform(lng, lat)
        entries.append({"lng": lng, "lat": lat, "mx": mx, "my": my})
    out = GOLDEN_DIR / "transform_golden.json"
    out.write_text(json.dumps(entries, indent=2))
    print(f"  {len(entries)} transform points")


def _random_convex_polygon(rng2: random.Random) -> Polygon:
    """Generate a random-ish rectangle within [10, 118]."""
    x1 = rng2.uniform(10.0, 80.0)
    y1 = rng2.uniform(10.0, 80.0)
    w = rng2.uniform(5.0, 30.0)
    h = rng2.uniform(5.0, 30.0)
    x2 = min(x1 + w, 118.0)
    y2 = min(y1 + h, 118.0)
    return _rect_polygon(x1, y1, x2, y2)


def generate_rasterize_golden() -> None:
    """Write tests/golden/rasterize_golden.npz and rasterize_golden_meta.json."""
    a, b, c, d, e, f = _RASTER_TRANSFORM
    transform = Affine(a, b, c, d, e, f)
    size = _RASTER_SIZE

    geom_cases: List[Tuple[Any, int]] = [
        # case 0: unit square -> 1 pixel at (row=127, col=0)
        (_rect_polygon(0, 0, 1, 1), 1),
        # case 1: full image square
        (_rect_polygon(0, 0, size, size), 1),
        # case 2: polygon with hole, class 2
        (
            Polygon(
                [(10, 10), (110, 10), (110, 110), (10, 110)],
                [[(20, 20), (100, 20), (100, 100), (20, 100)]],
            ),
            2,
        ),
        # case 3: small 5x5 square, class 3
        (_rect_polygon(60, 60, 65, 65), 3),
        # case 4: MultiPolygon two squares far apart, class 1
        (
            MultiPolygon(
                [
                    _rect_polygon(5, 5, 25, 25),
                    _rect_polygon(95, 95, 115, 115),
                ]
            ),
            1,
        ),
        # case 5: thin diagonal polygon (3px wide), class 1
        (Polygon([(0, 0), (128, 125), (128, 128), (0, 3)]), 1),
        # case 6: polygon outside image bounds -> all zeros, class 1
        (_rect_polygon(200, 200, 210, 210), 1),
    ]

    rng2 = random.Random(SEED)
    for _ in range(13):
        geom_cases.append((_random_convex_polygon(rng2), rng2.randint(1, 5)))

    masks: Dict[str, Any] = {}
    meta: List[Dict[str, Any]] = []

    for i, (geom, class_id) in enumerate(geom_cases):
        mask = rio_rasterize(
            [(mapping(geom), class_id)],
            out_shape=(size, size),
            transform=transform,
            dtype="uint8",
            fill=0,
            all_touched=False,
        )
        masks[f"case_{i}"] = mask
        meta.append(
            {
                "case": i,
                "wkt": wkt_dumps(geom),
                "class_id": class_id,
                "transform": list(_RASTER_TRANSFORM),
                "out_shape": [size, size],
                "nonzero": int(np.count_nonzero(mask)),
            }
        )

    np.savez_compressed(GOLDEN_DIR / "rasterize_golden.npz", **masks)
    (GOLDEN_DIR / "rasterize_golden_meta.json").write_text(json.dumps(meta, indent=2))
    total_nonzero = sum(m["nonzero"] for m in meta)
    print(f"  {len(geom_cases)} rasterize cases, {total_nonzero} total nonzero pixels")


def generate_label_parse_golden() -> None:
    """Write tests/golden/label_parse_golden.json."""
    cases: List[Dict[str, Any]] = []

    for kml_id, kml_bytes, _label_field in _KML_CASES:
        gdf = gpd.read_file(io.BytesIO(kml_bytes), driver="KML")
        features: List[Dict[str, Any]] = []
        for _, row in gdf.iterrows():
            geom = row.geometry
            if geom is None or geom.is_empty:
                continue
            gtype = geom.geom_type
            if gtype == "Polygon":
                vertex_count = len(list(geom.exterior.coords))
                coords = list(geom.exterior.coords)
            elif gtype == "MultiPolygon":
                vertex_count = sum(len(list(p.exterior.coords)) for p in geom.geoms)
                coords = list(geom.geoms[0].exterior.coords)
            else:
                continue
            features.append(
                {
                    "wkt": wkt_dumps(geom, rounding_precision=-1),
                    "type": gtype,
                    "vertex_count": vertex_count,
                    "exterior_coords": [[c[0], c[1]] for c in coords],
                }
            )
        cases.append({"kml_id": kml_id, "features": features})

    (GOLDEN_DIR / "label_parse_golden.json").write_text(json.dumps(cases, indent=2))
    total = sum(len(c["features"]) for c in cases)
    print(f"  {len(cases)} KML cases, {total} total features")


if __name__ == "__main__":
    print("Generating tile_math_golden.json ...")
    generate_tile_math_golden()
    print("Generating transform_golden.json ...")
    generate_transform_golden()
    print("Generating rasterize_golden.npz ...")
    generate_rasterize_golden()
    print("Generating label_parse_golden.json ...")
    generate_label_parse_golden()
    print("Done. Golden files written to", GOLDEN_DIR)
