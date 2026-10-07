"""Rebuild the example label files from OpenStreetMap via the Overpass API.

The GeoJSON files shipped with the examples were produced by this script, so
anyone can check where they came from or refresh them::

    python examples/scripts/fetch_osm_labels.py quickstart
    python examples/scripts/fetch_osm_labels.py sentinel2-landcover

Each run sends ONE small Overpass query (please keep it that way: Overpass is a
shared, donated service) and writes a compact WGS-84 GeoJSON next to the
example's ``mapcv.yaml``. The output records the query, the OSM data timestamp
and the licence in a top-level ``"osm"`` member.

Data © OpenStreetMap contributors, available under the Open Database Licence
(ODbL 1.0): https://www.openstreetmap.org/copyright
"""

from __future__ import annotations

import argparse
import json
import urllib.parse
import urllib.request
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from shapely.geometry import LineString, MultiPolygon, Polygon, box, mapping
from shapely.geometry.base import BaseGeometry
from shapely.ops import polygonize, unary_union

OVERPASS_URL = "https://overpass-api.de/api/interpreter"
USER_AGENT = "mapcv-examples (https://github.com/tahamukhtar20/mapcv)"
EXAMPLES_DIR = Path(__file__).resolve().parent.parent
ATTRIBUTION = "© OpenStreetMap contributors"
LICENSE = "ODbL-1.0"

Bbox = tuple[float, float, float, float]  # west, south, east, north
Classifier = Callable[[dict[str, str]], str | None]


@dataclass(frozen=True)
class LabelSpec:
    """One example's label file: where it goes, what to query and how to classify."""

    output: Path
    bbox: Bbox
    query: str
    classify: Classifier
    keep_tags: tuple[str, ...]
    class_order: tuple[str, ...]
    simplify_deg: float
    precision: int
    min_area_deg2: float
    clip: bool
    dissolve: bool


def _overpass_bbox(bbox: Bbox) -> str:
    west, south, east, north = bbox
    return f"{south},{west},{north},{east}"


# ── quickstart: building footprints in Amsterdam's Eastern Docklands ─────────

QUICKSTART_BBOX: Bbox = (4.9360, 52.3715, 4.9530, 52.3790)
QUICKSTART_QUERY = f"""[out:json][timeout:60];
(
  way["building"]({_overpass_bbox(QUICKSTART_BBOX)});
  relation["building"]["type"="multipolygon"]({_overpass_bbox(QUICKSTART_BBOX)});
);
out geom;"""


def _classify_building(tags: dict[str, str]) -> str | None:
    return "building" if tags.get("building", "no") != "no" else None


# ── sentinel2-landcover: land use around the Loosdrecht lakes (Utrecht) ─────

LANDCOVER_BBOX: Bbox = (5.397, 51.972, 5.503, 52.038)
_LANDUSE = "farmland|meadow|orchard|forest|residential|commercial|industrial|retail|reservoir|basin"
_NATURAL = "water|wood"
LANDCOVER_QUERY = f"""[out:json][timeout:90];
(
  way["landuse"~"^({_LANDUSE})$"]({_overpass_bbox(LANDCOVER_BBOX)});
  relation["landuse"~"^({_LANDUSE})$"]["type"="multipolygon"]({_overpass_bbox(LANDCOVER_BBOX)});
  way["natural"~"^({_NATURAL})$"]({_overpass_bbox(LANDCOVER_BBOX)});
  relation["natural"~"^({_NATURAL})$"]["type"="multipolygon"]({_overpass_bbox(LANDCOVER_BBOX)});
);
out geom;"""

_LANDCOVER_CLASSES = {
    ("landuse", "residential"): "built_up",
    ("landuse", "commercial"): "built_up",
    ("landuse", "industrial"): "built_up",
    ("landuse", "retail"): "built_up",
    ("landuse", "farmland"): "farmland",
    ("landuse", "meadow"): "farmland",
    ("landuse", "orchard"): "farmland",
    ("landuse", "forest"): "forest",
    ("natural", "wood"): "forest",
    ("natural", "water"): "water",
    ("landuse", "reservoir"): "water",
    ("landuse", "basin"): "water",
}


def _classify_landcover(tags: dict[str, str]) -> str | None:
    # natural=* wins over landuse=* (e.g. a lake inside a residential area).
    for key in ("natural", "landuse"):
        name = _LANDCOVER_CLASSES.get((key, tags.get(key, "")))
        if name is not None:
            return name
    return None


SPECS: dict[str, LabelSpec] = {
    "quickstart": LabelSpec(
        output=EXAMPLES_DIR / "quickstart" / "buildings.geojson",
        bbox=QUICKSTART_BBOX,
        query=QUICKSTART_QUERY,
        classify=_classify_building,
        keep_tags=("building",),
        class_order=("building",),
        simplify_deg=0.000002,  # ~0.15 m: drops near-collinear vertices only
        precision=6,  # ~0.1 m, below the 0.36 m pixel size at zoom 18
        min_area_deg2=0.0,
        clip=False,
        dissolve=False,
    ),
    "sentinel2-landcover": LabelSpec(
        output=EXAMPLES_DIR / "sentinel2-landcover" / "landuse.geojson",
        bbox=LANDCOVER_BBOX,
        query=LANDCOVER_QUERY,
        classify=_classify_landcover,
        keep_tags=(),
        # Later classes are drawn on top of earlier ones when rasterized.
        class_order=("built_up", "farmland", "forest", "water"),
        simplify_deg=0.0001,  # ~7-11 m, about one 10 m Sentinel-2 pixel
        precision=5,  # ~1 m
        min_area_deg2=3e-7,  # ~0.25 ha, 25 Sentinel-2 pixels
        clip=True,
        dissolve=True,
    ),
}


def fetch(query: str) -> dict[str, Any]:
    """Run one Overpass query and return the decoded JSON."""
    body = urllib.parse.urlencode({"data": query}).encode()
    request = urllib.request.Request(OVERPASS_URL, data=body, headers={"User-Agent": USER_AGENT})
    with urllib.request.urlopen(request, timeout=180) as response:
        result: dict[str, Any] = json.load(response)
    return result


def _line(points: list[dict[str, float]]) -> list[tuple[float, float]]:
    return [(point["lon"], point["lat"]) for point in points]


def _way_polygon(element: dict[str, Any]) -> BaseGeometry | None:
    coords = _line(element.get("geometry", []))
    if len(coords) < 4 or coords[0] != coords[-1]:
        return None  # unclosed way: not an area
    polygon = Polygon(coords)
    return polygon if polygon.is_valid else polygon.buffer(0)


def _relation_polygon(element: dict[str, Any]) -> BaseGeometry | None:
    rings: dict[str, list[LineString]] = {"outer": [], "inner": []}
    for member in element.get("members", []):
        role = member.get("role") or "outer"
        if member.get("type") == "way" and role in rings and member.get("geometry"):
            rings[role].append(LineString(_line(member["geometry"])))
    outer = unary_union(list(polygonize(rings["outer"])))
    if outer.is_empty:
        return None
    inner = unary_union(list(polygonize(rings["inner"])))
    return outer.difference(inner) if not inner.is_empty else outer


def _polygon_parts(geometry: BaseGeometry) -> list[Polygon]:
    if isinstance(geometry, Polygon):
        return [geometry]
    if isinstance(geometry, MultiPolygon):
        return list(geometry.geoms)
    return [part for g in getattr(geometry, "geoms", []) for part in _polygon_parts(g)]


def _round(value: Any, precision: int) -> Any:
    if isinstance(value, float):
        return round(value, precision)
    if isinstance(value, (list, tuple)):
        return [_round(item, precision) for item in value]
    return value


def _clean(spec: LabelSpec, geometry: BaseGeometry) -> BaseGeometry | None:
    if spec.clip:
        geometry = geometry.intersection(box(*spec.bbox))
    if spec.simplify_deg:
        geometry = geometry.simplify(spec.simplify_deg, preserve_topology=True)
    parts = [part for part in _polygon_parts(geometry) if part.area > spec.min_area_deg2]
    if not parts:
        return None
    return parts[0] if len(parts) == 1 else MultiPolygon(parts)


def _feature(geometry: BaseGeometry, properties: dict[str, str], precision: int) -> dict[str, Any]:
    geojson = mapping(geometry)
    return {
        "type": "Feature",
        "properties": properties,
        "geometry": {
            "type": geojson["type"],
            "coordinates": _round(geojson["coordinates"], precision),
        },
    }


def build(spec: LabelSpec, data: dict[str, Any]) -> dict[str, Any]:
    """Convert an Overpass ``out geom`` response into a GeoJSON FeatureCollection.

    Features are ordered by ``spec.class_order`` because mapcv draws later
    polygons on top of earlier ones. With ``spec.dissolve`` the polygons of each
    class are merged first, which keeps the file small when OSM maps many
    adjacent parcels (fields, ditches) that a 10 m pixel cannot tell apart.
    """
    by_class: dict[str, list[tuple[str, dict[str, str], BaseGeometry]]] = {}
    skipped = 0
    for element in data.get("elements", []):
        tags: dict[str, str] = element.get("tags", {})
        name = spec.classify(tags)
        if name is None:
            continue
        if element["type"] == "way":
            geometry = _way_polygon(element)
        else:
            geometry = _relation_polygon(element)
        if geometry is None or geometry.is_empty:
            skipped += 1
            continue
        osm_id = f"{element['type']}/{element['id']}"
        by_class.setdefault(name, []).append((osm_id, tags, geometry))

    features: list[dict[str, Any]] = []
    for name in spec.class_order:
        members = sorted(by_class.get(name, []), key=lambda member: member[0])
        if spec.dissolve:
            merged = _clean(spec, unary_union([geometry for _, _, geometry in members]))
            parts = _polygon_parts(merged) if merged is not None else []
            parts.sort(key=lambda part: -part.area)
            features.extend(_feature(part, {"class": name}, spec.precision) for part in parts)
            continue
        for osm_id, tags, geometry in members:
            cleaned = _clean(spec, geometry)
            if cleaned is None:
                continue
            properties = {"class": name, "osm_id": osm_id}
            properties.update({key: tags[key] for key in spec.keep_tags if key in tags})
            features.append(_feature(cleaned, properties, spec.precision))
    if skipped:
        print(f"skipped {skipped} element(s) that do not form a closed area")
    return {
        "type": "FeatureCollection",
        "osm": {
            "attribution": ATTRIBUTION,
            "license": LICENSE,
            "license_url": "https://opendatacommons.org/licenses/odbl/1-0/",
            "source": OVERPASS_URL,
            "query": spec.query,
            "timestamp_osm_base": data.get("osm3s", {}).get("timestamp_osm_base"),
        },
        "features": features,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("example", choices=sorted(SPECS))
    parser.add_argument("--output", type=Path, help="Write here instead of the example folder.")
    parser.add_argument(
        "--raw",
        type=Path,
        help="Cache for the raw Overpass response: read it if it exists, else save it there.",
    )
    args = parser.parse_args()

    spec = SPECS[args.example]
    raw: Path | None = args.raw
    if raw is not None and raw.exists():
        data = json.loads(raw.read_text())
    else:
        data = fetch(spec.query)
        if raw is not None:
            raw.write_text(json.dumps(data))
    collection = build(spec, data)
    output: Path = args.output or spec.output
    output.write_text(json.dumps(collection, separators=(",", ":"), ensure_ascii=False) + "\n")
    counts: dict[str, int] = {}
    for feature in collection["features"]:
        counts[feature["properties"]["class"]] = counts.get(feature["properties"]["class"], 0) + 1
    size_kb = output.stat().st_size / 1000
    print(f"wrote {output} ({size_kb:.0f} KB): {counts}")
    print(f"OSM data as of {collection['osm']['timestamp_osm_base']} — {ATTRIBUTION}, {LICENSE}")


if __name__ == "__main__":
    main()
