"""Labels straight from OpenStreetMap (``labels.osm``), through the Overpass API.

One Overpass query fetches every feature in the box whose tags match a class
(``out geom``: nodes, ways and relations with their coordinates). Each feature goes to
the first class it matches, in config order, and becomes:

* a polygon: a closed way that is an area (not a highway, barrier or ``area=no`` loop),
  or a ``multipolygon`` relation, assembled from its outer and inner member ways;
* a line: any other way; a point: a node. Lines and points only become labels with
  ``labels.buffer``, as for label files.

The result is written to a GeoJSON file in the cache folder (``<cache dir>/osm/``,
named by a hash of the query) with each feature's ``class`` and ``osm_id``, sorted, and
the data's timestamp (``osm_base``) and licence. That file is then read like any label
file, so its SHA-256 identifies the labels in the manifest and a later run with fresh
OpenStreetMap data does not resume into a dataset made from older data. A cached file
is reused until deleted: data from the same query stays the same.

OpenStreetMap data is © OpenStreetMap contributors, under the Open Database License
(ODbL): credit it, and share a database adapted from it (such as these masks) under the
ODbL too. See https://www.openstreetmap.org/copyright.
"""

from __future__ import annotations

import hashlib
import json
import os
import urllib.parse
import urllib.request
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from shapely.geometry import LineString, Point, Polygon, mapping
from shapely.geometry.base import BaseGeometry
from shapely.ops import polygonize, unary_union

from mapcv.config import OsmClass, OsmLabelsSource
from mapcv.tile_cache import cache_dir

ATTRIBUTION = "© OpenStreetMap contributors (ODbL)"
# Closed ways with these keys are lines (a ring road, a fence), unless area=yes.
_LINEAR_KEYS = ("highway", "barrier", "railway", "waterway")


def _selector(tags: dict[str, Any]) -> str:
    parts = []
    for key, value in tags.items():
        if value == "*":
            parts.append(f'["{key}"]')
        elif isinstance(value, list):
            choices = "|".join(_regex_escape(v) for v in value)
            parts.append(f'["{key}"~"^({choices})$"]')
        else:
            parts.append(f'["{key}"="{value}"]')
    return "".join(parts)


def _regex_escape(value: str) -> str:
    return "".join("\\\\" + ch if ch in ".^$*+?()[]{}|" else ch for ch in value)


def overpass_query(source: OsmLabelsSource) -> str:
    """The Overpass QL query for every class in the box."""
    assert source.bbox is not None  # MapcvConfig fills it from the region
    west, south, east, north = source.bbox
    box = f"({south!r},{west!r},{north!r},{east!r})"
    statements = "".join(f"  nwr{_selector(entry.tags)}{box};\n" for entry in source.classes)
    return f"[out:json][timeout:{source.timeout}];\n(\n{statements});\nout geom;\n"


def _matches(tags: dict[str, str], wanted: dict[str, Any]) -> bool:
    for key, value in wanted.items():
        if key not in tags:
            return False
        if value == "*":
            continue
        values = value if isinstance(value, list) else [value]
        if tags[key] not in values:
            return False
    return True


def _class_of(tags: dict[str, str], classes: Sequence[OsmClass]) -> str | None:
    for entry in classes:
        if _matches(tags, entry.tags):
            return entry.name
    return None


def _coords(points: Sequence[dict[str, float]]) -> list[tuple[float, float]]:
    return [(float(p["lon"]), float(p["lat"])) for p in points if p is not None]


def _way_geometry(element: dict[str, Any]) -> BaseGeometry | None:
    coords = _coords(element.get("geometry") or [])
    if len(coords) < 2:
        return None
    tags = element.get("tags", {})
    closed = len(coords) >= 4 and coords[0] == coords[-1]
    linear = tags.get("area") == "no" or (
        tags.get("area") != "yes" and any(key in tags for key in _LINEAR_KEYS)
    )
    if closed and not linear:
        polygon = Polygon(coords)
        return polygon if polygon.is_valid else polygon.buffer(0)
    return LineString(coords)


def _relation_geometry(element: dict[str, Any]) -> BaseGeometry | None:
    if element.get("tags", {}).get("type") not in ("multipolygon", "boundary"):
        return None
    rings: dict[str, list[LineString]] = {"outer": [], "inner": []}
    for member in element.get("members", []):
        coords = _coords(member.get("geometry") or [])
        if member.get("type") == "way" and len(coords) >= 2:
            rings["inner" if member.get("role") == "inner" else "outer"].append(LineString(coords))
    outer = unary_union(list(polygonize(rings["outer"])))
    if outer.is_empty:
        return None
    inner = unary_union(list(polygonize(rings["inner"])))
    result: BaseGeometry = outer.difference(inner) if not inner.is_empty else outer
    return result


def features_from_overpass(
    answer: dict[str, Any], classes: Sequence[OsmClass]
) -> list[dict[str, Any]]:
    """GeoJSON features (``class``, ``osm_id``) from an Overpass ``out geom`` answer, in
    class order, then by OSM type and ID."""
    order = {entry.name: index for index, entry in enumerate(classes)}
    found: list[tuple[int, str, int, dict[str, Any]]] = []
    for element in answer.get("elements", []):
        name = _class_of(element.get("tags", {}), classes)
        if name is None:
            continue
        kind = element.get("type")
        geometry: BaseGeometry | None
        if kind == "node":
            geometry = Point(float(element["lon"]), float(element["lat"]))
        elif kind == "way":
            geometry = _way_geometry(element)
        elif kind == "relation":
            geometry = _relation_geometry(element)
        else:
            geometry = None
        if geometry is None or geometry.is_empty:
            continue
        feature = {
            "type": "Feature",
            "properties": {"class": name, "osm_id": f"{kind}/{element['id']}"},
            "geometry": mapping(geometry),
        }
        found.append((order[name], str(kind), int(element["id"]), feature))
    found.sort(key=lambda item: item[:3])
    return [feature for *_, feature in found]


def _fetch(source: OsmLabelsSource, query: str) -> dict[str, Any]:
    data = urllib.parse.urlencode({"data": query}).encode("utf-8")
    request = urllib.request.Request(
        source.overpass_url,
        data=data,
        headers={"User-Agent": "mapcv (+https://github.com/tahamukhtar20/mapcv)"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=source.timeout + 30) as response:
            answer: dict[str, Any] = json.load(response)
    except OSError as exc:
        raise RuntimeError(
            f"Overpass request to {source.overpass_url} failed: {exc}. The public server "
            "limits load; try again later or set labels.osm.overpass_url to another instance"
        ) from exc
    return answer


def osm_labels_file(source: OsmLabelsSource, folder: Path | None = None) -> Path:
    """The cached GeoJSON of ``source``'s labels, fetched from Overpass the first time."""
    query = overpass_query(source)
    key = hashlib.sha256(f"{source.overpass_url}\n{query}".encode()).hexdigest()[:24]
    folder = folder if folder is not None else cache_dir() / "osm"
    path = folder / f"{key}.geojson"
    if path.is_file():
        return path
    answer = _fetch(source, query)
    # Overpass answers a timed-out or out-of-memory query with 200 and a remark.
    if answer.get("remark") and "error" in str(answer["remark"]).lower():
        raise RuntimeError(f"Overpass reported an error: {answer['remark']}")
    collection = {
        "type": "FeatureCollection",
        "osm_base": (answer.get("osm3s") or {}).get("timestamp_osm_base"),
        "license": ATTRIBUTION,
        "query": query,
        "features": features_from_overpass(answer, source.classes),
    }
    folder.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temp.write_text(json.dumps(collection, separators=(",", ":")), encoding="utf-8")
    os.replace(temp, path)
    return path


def default_class_ids(source: OsmLabelsSource) -> dict[str, int]:
    """Class IDs in config order (1, 2, 3, ...) when ``labels.classes`` is not given."""
    return {entry.name: index for index, entry in enumerate(source.classes, start=1)}
