"""Find a Sentinel-2 product for a region in a STAC API (``imagery.search``).

A small client for the STAC API ``/search`` endpoint (POST, following ``next`` links),
standard library only. Of the items found, the one used is the least cloudy whose
footprint covers the whole region, ties going to the earlier acquisition and then the
item ID, so the same catalog state always gives the same product.
"""

from __future__ import annotations

import json
import re
import urllib.request
from dataclasses import dataclass
from typing import Any, Dict, Iterator, List, Optional, Tuple

from shapely.geometry import box, shape

from mapcv.config import StacSearchConfig

_TIMEOUT_S = 60
_MAX_PAGES = 50
_PAGE_SIZE = 100
_DATE = re.compile(r"\d{4}-\d{2}-\d{2}")


@dataclass(frozen=True)
class StacMatch:
    """The item chosen for a region."""

    item_id: str
    href: str
    cloud_cover: float
    datetime: str
    candidates: int


def _interval(value: str) -> str:
    """``datetime`` as the RFC 3339 interval the STAC API expects (whole days for dates)."""
    parts = value.split("/")
    if len(parts) == 1 and _DATE.fullmatch(parts[0]):
        parts = [parts[0], parts[0]]
    if len(parts) == 1:
        return parts[0]
    start, end = parts
    if _DATE.fullmatch(start):
        start += "T00:00:00Z"
    if _DATE.fullmatch(end):
        end += "T23:59:59Z"
    return f"{start}/{end}"


def _post(url: str, body: Dict[str, Any]) -> Dict[str, Any]:
    request = urllib.request.Request(
        url,
        data=json.dumps(body).encode("utf-8"),
        headers={"Content-Type": "application/json", "Accept": "application/geo+json"},
        method="POST",
    )
    with urllib.request.urlopen(request, timeout=_TIMEOUT_S) as response:  # noqa: S310 - https or loopback, checked by the config
        page: Dict[str, Any] = json.load(response)
    return page


def _get(url: str) -> Dict[str, Any]:
    with urllib.request.urlopen(url, timeout=_TIMEOUT_S) as response:  # noqa: S310
        page: Dict[str, Any] = json.load(response)
    return page


def search_items(
    search: StacSearchConfig, bbox: Tuple[float, float, float, float]
) -> Iterator[Dict[str, Any]]:
    """Every item of the search, page by page (at most 5,000)."""
    url: Optional[str] = f"{search.catalog}/search"
    body: Optional[Dict[str, Any]] = {
        "collections": [search.collection],
        "bbox": list(bbox),
        "datetime": _interval(search.datetime),
        "limit": _PAGE_SIZE,
    }
    method = "POST"
    for _ in range(_MAX_PAGES):
        assert url is not None
        page = _post(url, body or {}) if method == "POST" else _get(url)
        yield from page.get("features", [])
        following = next(
            (link for link in page.get("links", []) if link.get("rel") == "next"), None
        )
        if following is None:
            return
        url = following["href"]
        method = str(following.get("method", "GET")).upper()
        if method == "POST":
            # A next link may replace the body (often with a token) or merge into it.
            new_body = following.get("body") or {}
            body = {**(body or {}), **new_body} if following.get("merge") else (new_body or body)


def find_product(search: StacSearchConfig, bbox: Tuple[float, float, float, float]) -> StacMatch:
    """The item to use for ``bbox`` (west, south, east, north in degrees).

    Raises:
        ValueError: No item of the search covers the region with at most ``max_cloud``
            percent cloud, or the chosen item lacks the ``asset``.
        RuntimeError: The catalog cannot be reached or answers with an error.
    """
    region = box(*bbox)
    try:
        items = list(search_items(search, bbox))
    except OSError as exc:  # urllib's URLError and HTTPError are OSErrors
        raise RuntimeError(f"STAC search at {search.catalog} failed: {exc}") from exc
    candidates: List[Tuple[float, str, str, Dict[str, Any]]] = []
    covering = 0
    for item in items:
        geometry = item.get("geometry")
        if geometry is None or not shape(geometry).covers(region):
            continue
        covering += 1
        cloud = item.get("properties", {}).get("eo:cloud_cover")
        if cloud is None or float(cloud) > search.max_cloud:
            continue
        when = str(item.get("properties", {}).get("datetime") or "")
        candidates.append((float(cloud), when, str(item.get("id", "")), item))
    if not candidates:
        raise ValueError(
            f"no {search.collection} item in {search.catalog} covers the region in "
            f"{search.datetime} with at most {search.max_cloud:g}% cloud ({len(items)} found, "
            f"{covering} covering the whole region); widen imagery.search.datetime, raise "
            "max_cloud or shrink the region"
        )
    cloud, when, item_id, item = min(candidates, key=lambda c: (c[0], c[1], c[2]))
    asset = item.get("assets", {}).get(search.asset)
    if not asset or not asset.get("href"):
        raise ValueError(
            f"STAC item {item_id} has no '{search.asset}' asset (imagery.search.asset)"
        )
    return StacMatch(
        item_id=item_id,
        href=str(asset["href"]),
        cloud_cover=cloud,
        datetime=when,
        candidates=len(candidates),
    )
