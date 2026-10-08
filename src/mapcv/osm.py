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
import math
import os
import time
import urllib.parse
import urllib.request
import warnings
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np
from shapely import make_valid
from shapely.geometry import LineString, MultiPolygon, Point, Polygon, mapping
from shapely.geometry.base import BaseGeometry
from shapely.ops import polygonize

from mapcv.config import OsmClass, OsmLabelsSource
from mapcv.tile_cache import cache_dir
from mapcv.vector_files import organize_rings

ATTRIBUTION = "© OpenStreetMap contributors (ODbL)"
#: Seconds an Overpass answer may take beyond ``labels.osm.timeout`` (the query's own
#: limit on the server) to arrive in full.
ANSWER_MARGIN_SECONDS = 60
#: The largest Overpass answer mapcv reads (Overpass's own default ``maxsize``).
MAX_ANSWER_BYTES = 512 << 20
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
    if source.bbox is None:
        # MapcvConfig fills it from the region; a source made on its own needs one.
        raise ValueError(
            "labels.osm.bbox is not set: give (west, south, east, north), or load the labels "
            "through a MapcvConfig, which takes the region's box"
        )
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


class _Malformed(Exception):
    """An element of the Overpass answer that is not what Overpass sends; it is skipped."""


def _coords(points: Any) -> list[tuple[float, float]]:
    if not isinstance(points, list):
        raise _Malformed
    coords = []
    for point in points:
        if point is None:  # a node Overpass left out
            continue
        try:
            lon, lat = float(point["lon"]), float(point["lat"])
        except (TypeError, KeyError, ValueError):
            raise _Malformed from None
        if not (math.isfinite(lon) and math.isfinite(lat)):
            raise _Malformed
        coords.append((lon, lat))
    return coords


def _tags(element: dict[str, Any]) -> dict[str, Any]:
    tags = element.get("tags")
    return tags if isinstance(tags, dict) else {}


def _collect_polygons(geometry: BaseGeometry, parts: list[Polygon]) -> None:
    if isinstance(geometry, Polygon):
        if not geometry.is_empty:
            parts.append(geometry)
    elif geometry.geom_type in ("MultiPolygon", "GeometryCollection"):
        for part in getattr(geometry, "geoms", []):
            _collect_polygons(part, parts)


def _polygonal(geometry: BaseGeometry) -> BaseGeometry | None:
    """A polygon as it is when valid, else repaired with ``make_valid``, which keeps every
    part of a self-intersecting ring (both lobes of a figure eight); polygons only."""
    if geometry.is_valid:
        return geometry
    parts: list[Polygon] = []
    _collect_polygons(make_valid(geometry), parts)
    if not parts:
        return None
    return parts[0] if len(parts) == 1 else MultiPolygon(parts)


def _way_geometry(element: dict[str, Any]) -> BaseGeometry | None:
    coords = _coords(element.get("geometry") or [])
    if len(coords) < 2:
        return None
    tags = _tags(element)
    closed = len(coords) >= 4 and coords[0] == coords[-1]
    linear = tags.get("area") == "no" or (
        tags.get("area") != "yes" and any(key in tags for key in _LINEAR_KEYS)
    )
    if closed and not linear:
        return _polygonal(Polygon(coords))
    return LineString(coords)


def _relation_geometry(element: dict[str, Any]) -> BaseGeometry | None:
    """A multipolygon relation assembled from its member ways.

    The ways are joined into closed rings and, as osmium and GDAL do, the nesting of the
    rings decides what is filled rather than the members' outer/inner roles: a ring inside
    an even number of others bounds an area, one inside an odd number is a hole. So an
    island in a lake (an outer ring inside an inner one) is kept.
    """
    if _tags(element).get("type") not in ("multipolygon", "boundary"):
        return None
    members = element.get("members") or []
    if not isinstance(members, list):
        raise _Malformed
    lines: list[LineString] = []
    for member in members:
        if not isinstance(member, dict):
            raise _Malformed
        coords = _coords(member.get("geometry") or [])
        if member.get("type") == "way" and len(coords) >= 2:
            lines.append(LineString(coords))
    rings = [np.asarray(face.exterior.coords) for face in polygonize(lines)]
    if not rings:
        return None
    assembled = organize_rings(rings)
    return None if assembled is None else _polygonal(assembled)


def features_from_overpass(
    answer: dict[str, Any], classes: Sequence[OsmClass]
) -> list[dict[str, Any]]:
    """GeoJSON features (``class``, ``osm_id``) from an Overpass ``out geom`` answer, in
    class order, then by OSM type and ID."""
    order = {entry.name: index for index, entry in enumerate(classes)}
    found: list[tuple[int, str, int, dict[str, Any]]] = []
    elements = answer.get("elements") or []
    if not isinstance(elements, list):
        raise ValueError("its 'elements' is not a list")
    malformed = 0
    for element in elements:
        if not isinstance(element, dict):
            malformed += 1
            continue
        name = _class_of(_tags(element), classes)
        if name is None:
            continue
        kind = element.get("type")
        identifier = element.get("id")
        geometry: BaseGeometry | None
        try:
            if not isinstance(identifier, int) or isinstance(identifier, bool):
                raise _Malformed
            if kind == "node":
                (point,) = _coords([element])
                geometry = Point(point)
            elif kind == "way":
                geometry = _way_geometry(element)
            elif kind == "relation":
                geometry = _relation_geometry(element)
            else:
                geometry = None
        except (_Malformed, ValueError):
            malformed += 1
            continue
        if geometry is None or geometry.is_empty:
            continue
        feature = {
            "type": "Feature",
            "properties": {"class": name, "osm_id": f"{kind}/{identifier}"},
            "geometry": mapping(geometry),
        }
        found.append((order[name], str(kind), identifier, feature))
    if malformed:
        warnings.warn(
            f"OpenStreetMap: skipped {malformed} element(s) of the Overpass answer that are "
            "not valid Overpass JSON (an id, coordinates or members missing or malformed).",
            UserWarning,
            stacklevel=2,
        )
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
    # Overpass runs the query for up to `timeout` seconds; the answer then has a margin to
    # arrive. The deadline is for the whole exchange, however slowly the bytes come.
    seconds = source.timeout + ANSWER_MARGIN_SECONDS
    deadline = time.monotonic() + seconds
    what = f"Overpass at {_server(source.overpass_url)}"
    try:
        with urllib.request.urlopen(request, timeout=seconds) as response:
            body = _read_answer(response, deadline, what)
    except TimeoutError:
        raise RuntimeError(
            f"{what} did not answer within {seconds} s (labels.osm.timeout {source.timeout} s "
            f"plus {ANSWER_MARGIN_SECONDS} s for the answer to arrive). The server may be "
            "busy: try again later, use a smaller region or set labels.osm.overpass_url to "
            "another instance"
        ) from None
    except OSError as exc:
        raise RuntimeError(
            f"Overpass request to {source.overpass_url} failed: {exc}. The public server "
            "limits load; try again later or set labels.osm.overpass_url to another instance"
        ) from exc
    return _decode_answer(body, what)


def _server(url: str) -> str:
    """The Overpass URL without user info or query, either of which may hold a key."""
    parts = urllib.parse.urlsplit(url)
    host = parts.hostname or ""
    if parts.port is not None:
        host += f":{parts.port}"
    return urllib.parse.urlunsplit((parts.scheme, host, parts.path, "", ""))


def _set_read_timeout(response: Any, seconds: float) -> None:
    """Make the next socket read of ``response`` wait at most ``seconds``."""
    sock = getattr(getattr(getattr(response, "fp", None), "raw", None), "_sock", None)
    if sock is not None:
        sock.settimeout(max(seconds, 0.001))


def _read_answer(response: Any, deadline: float, what: str) -> bytes:
    """The whole body, read before ``deadline`` (``time.monotonic()``) and not larger than
    :data:`MAX_ANSWER_BYTES`.

    Raises:
        TimeoutError: The deadline passed.
        RuntimeError: The answer is too large.
    """
    chunks: list[bytes] = []
    size = 0
    while True:
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            raise TimeoutError
        _set_read_timeout(response, remaining)
        chunk: bytes = response.read1(1 << 16)
        if not chunk:
            return b"".join(chunks)
        size += len(chunk)
        if size > MAX_ANSWER_BYTES:
            raise RuntimeError(
                f"{what} sent more than {MAX_ANSWER_BYTES >> 20} MiB, the most mapcv reads; "
                "use a smaller region or more specific tags for the classes"
            )
        chunks.append(chunk)


def _decode_answer(body: bytes, what: str) -> dict[str, Any]:
    """The JSON object of an Overpass answer, or a ``RuntimeError`` that says what came."""
    try:
        answer = json.loads(body.decode("utf-8"))
    except ValueError:  # also UnicodeDecodeError
        start = body[:60].decode("utf-8", errors="replace").strip()
        raise RuntimeError(
            f"{what} answered with something that is not Overpass JSON (it starts with "
            f"{start!r}); a busy server may send an error page instead. Try again later or "
            "set labels.osm.overpass_url to another instance"
        ) from None
    if not isinstance(answer, dict):
        raise RuntimeError(
            f"{what} answered with JSON that is not an Overpass answer (not an object); "
            "check labels.osm.overpass_url"
        )
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
    try:
        features = features_from_overpass(answer, source.classes)
    except ValueError as exc:
        raise RuntimeError(
            f"Overpass at {_server(source.overpass_url)} answered with JSON that is not an "
            f"Overpass answer ({exc}); check labels.osm.overpass_url"
        ) from None
    meta = answer.get("osm3s")
    collection = {
        "type": "FeatureCollection",
        "osm_base": meta.get("timestamp_osm_base") if isinstance(meta, dict) else None,
        "license": ATTRIBUTION,
        "query": query,
        "features": features,
    }
    folder.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temp.write_text(json.dumps(collection, separators=(",", ":")), encoding="utf-8")
    os.replace(temp, path)
    return path


def default_class_ids(source: OsmLabelsSource) -> dict[str, int]:
    """Class IDs in config order (1, 2, 3, ...) when ``labels.classes`` is not given."""
    return {entry.name: index for index, entry in enumerate(source.classes, start=1)}
