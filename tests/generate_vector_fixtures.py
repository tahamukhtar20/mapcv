"""Write the small vector label fixtures in tests/data/vector with GDAL (pyogrio) and geopandas.

mapcv reads these formats without GDAL, so the fixtures are written by an independent
implementation. Run it from the repository root in a throwaway environment, not the
project's one (geopandas and pyogrio are not project dependencies):

    uv venv .venv-fixtures
    uv pip install --python .venv-fixtures/bin/python geopandas pyogrio pyarrow shapely pyproj
    .venv-fixtures/bin/python tests/generate_vector_fixtures.py

``labels.geojson`` is written with plain ``json`` (full coordinate precision) and is the
reference: tests/test_vector_labels.py checks that every other file yields the same
geometries and class map. The data is a handful of features near Vienna (inside UTM zone
33N, so the projected copies are meaningful):

* a polygon with a hole, a multipolygon whose first part has a hole, a polygon that
  is an island inside that hole's multipolygon, a plain polygon;
* a polygon with a non-ASCII label, a polygon without a label;
* a point and a line (Shapefiles hold one geometry type each, so those formats get
  separate files).
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Any, Dict, List

import geopandas as gpd
import pandas as pd
from shapely.geometry import LineString, MultiPolygon, Point, Polygon, box, mapping

OUT = Path(__file__).parent / "data" / "vector"
LON, LAT = 16.36, 48.19  # south-west corner of the data, near Vienna


def _square(x: float, y: float, size: float) -> Polygon:
    return box(LON + x, LAT + y, LON + x + size, LAT + y + size)


def features() -> List[Dict[str, Any]]:
    """The reference features: ``geometry`` plus the ``class``, ``rank`` and ``score`` fields."""
    holed = Polygon(
        _square(0.000, 0.000, 0.004).exterior.coords,
        [_square(0.001, 0.001, 0.002).exterior.coords],
    )
    # Two parts; the first has a hole that holds an island of the second part.
    outer = Polygon(
        _square(0.010, 0.000, 0.006).exterior.coords,
        [_square(0.011, 0.001, 0.004).exterior.coords],
    )
    island = _square(0.012, 0.002, 0.002)
    nested = MultiPolygon([outer, island])
    holed_part = Polygon(
        _square(0.020, 0.000, 0.004).exterior.coords,
        [_square(0.021, 0.001, 0.002).exterior.coords],
    )
    two_parts = MultiPolygon([holed_part, _square(0.026, 0.000, 0.003)])
    return [
        {"geometry": holed, "class": "building", "rank": 1, "score": 1.0},
        {"geometry": nested, "class": "farmland", "rank": 2, "score": 2.0},
        {"geometry": two_parts, "class": "building", "rank": 1, "score": 1.0},
        {"geometry": _square(0.000, 0.010, 0.003), "class": "forêt", "rank": 3, "score": 3.0},
        {"geometry": _square(0.010, 0.010, 0.003), "class": None, "rank": None, "score": 4.0},
        {
            "geometry": Point(LON + 0.0015, LAT + 0.0125),
            "class": "building",
            "rank": 1,
            "score": 1.0,
        },
        {
            "geometry": LineString([(LON, LAT + 0.02), (LON + 0.01, LAT + 0.021)]),
            "class": "road",
            "rank": 4,
            "score": 5.0,
        },
    ]


def frame(rows: List[Dict[str, Any]], crs: str = "EPSG:4326") -> gpd.GeoDataFrame:
    data = pd.DataFrame(
        {
            "class": pd.Series([row["class"] for row in rows], dtype="object"),
            "rank": pd.Series([row["rank"] for row in rows], dtype="Int64"),
            "score": pd.Series([row["score"] for row in rows], dtype="float64"),
        }
    )
    return gpd.GeoDataFrame(data, geometry=[row["geometry"] for row in rows], crs=crs)


def write_geojson(rows: List[Dict[str, Any]], path: Path) -> None:
    collection = {
        "type": "FeatureCollection",
        "features": [
            {
                "type": "Feature",
                "properties": {key: row[key] for key in ("class", "rank", "score")},
                "geometry": mapping(row["geometry"]),
            }
            for row in rows
        ],
    }
    path.write_text(json.dumps(collection, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")


def is_polygonal(row: Dict[str, Any]) -> bool:
    return row["geometry"].geom_type in ("Polygon", "MultiPolygon")


def main() -> None:
    if OUT.exists():
        shutil.rmtree(OUT)
    OUT.mkdir(parents=True)
    rows = features()
    polygons = [row for row in rows if is_polygonal(row)]
    points = [row for row in rows if row["geometry"].geom_type == "Point"]
    lines = [row for row in rows if row["geometry"].geom_type == "LineString"]

    write_geojson(rows, OUT / "labels.geojson")

    everything = frame(rows)
    projected = everything.to_crs(32633)
    everything.to_file(OUT / "labels.gpkg", layer="labels", driver="GPKG")
    projected.to_file(OUT / "labels_32633.gpkg", layer="labels", driver="GPKG")
    everything.to_crs(4269).to_file(OUT / "labels_4269.gpkg", layer="labels", driver="GPKG")
    everything.to_parquet(OUT / "labels.parquet")
    projected.to_parquet(OUT / "labels_32633.parquet")
    frame(polygons).to_parquet(OUT / "labels_geoarrow.parquet", geometry_encoding="geoarrow")

    # Two layers with different features: tests pick one by name.
    buildings = [row for row in polygons if row["class"] == "building"]
    landuse = [row for row in polygons if row["class"] != "building"]
    frame(buildings).to_file(OUT / "labels_2layers.gpkg", layer="buildings", driver="GPKG")
    frame(landuse).to_file(OUT / "labels_2layers.gpkg", layer="landuse", driver="GPKG", mode="a")
    write_geojson(buildings, OUT / "layer_buildings.geojson")
    write_geojson(landuse, OUT / "layer_landuse.geojson")

    # Shapefiles hold one geometry type, so polygons, points and lines are separate.
    frame(polygons).to_file(OUT / "polygons.shp")
    frame(points).to_file(OUT / "points.shp")
    frame(lines).to_file(OUT / "lines.shp")
    frame(polygons).to_crs(32633).to_file(OUT / "polygons_32633.shp")
    frame(polygons).to_file(OUT / "polygons_latin1.shp", encoding="ISO-8859-1")
    write_geojson(polygons, OUT / "polygons.geojson")
    write_geojson(points, OUT / "points.geojson")
    write_geojson(lines, OUT / "lines.geojson")

    sizes = sorted((path.name, path.stat().st_size) for path in OUT.iterdir())
    for name, size in sizes:
        print(f"{size:7d}  {name}")


if __name__ == "__main__":
    main()
