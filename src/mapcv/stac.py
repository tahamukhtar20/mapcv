"""Find a Sentinel-2 product for a region in a STAC API (``imagery.search``).

A small client for the STAC API ``/search`` endpoint (POST, following ``next`` links),
standard library only. Of the items found, the one used is the least cloudy whose
footprint covers the whole region, ties going to the earlier acquisition and then the
item ID, so the same catalog state always gives the same product.
"""

from __future__ import annotations

import json
import re
import threading
import time
import urllib.request
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from urllib.parse import urljoin, urlsplit

from shapely.geometry import box, shape

from mapcv.config import StacSearchBase, StacSearchConfig, eopf_local_path

_TIMEOUT_S = 60
# One page of a STAC search: its size cap and how long it may take to arrive.
_MAX_PAGE_BYTES = 64 * 2**20
_DEADLINE_S = 120
_READ_SIZE = 1 << 16
_check_lock = threading.Lock()
_check_users: list[Callable[[Path], None]] = []
_local_path_check: Callable[[Path], None] | None = None
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


def _read_json(response: Any, url: str) -> dict[str, Any]:
    """The JSON object of a response, read within the size cap and the deadline."""
    deadline = time.monotonic() + _DEADLINE_S
    parts: list[bytes] = []
    size = 0
    while True:
        if time.monotonic() > deadline:
            raise RuntimeError(
                f"STAC catalog {_origin(url)} took longer than {_DEADLINE_S} s to send one "
                "page of results; try again later or use another catalog"
            )
        # read1 returns what one socket read gives, so a slow sender cannot hold a read.
        part = response.read1(_READ_SIZE)
        if not part:
            break
        size += len(part)
        if size > _MAX_PAGE_BYTES:
            raise RuntimeError(
                f"STAC catalog {_origin(url)} sent more than {_MAX_PAGE_BYTES // 2**20} MiB "
                "for one page of results, more than a STAC API page holds; check "
                "imagery.search.catalog"
            )
        parts.append(part)
    try:
        page = json.loads(b"".join(parts))
    except ValueError:
        raise RuntimeError(
            f"STAC catalog {_origin(url)} answered with something other than JSON; check "
            "imagery.search.catalog"
        ) from None
    if not isinstance(page, dict):
        raise RuntimeError(
            f"STAC catalog {_origin(url)} answered with JSON that is not a STAC search page"
        )
    return page


def _origin(url: str) -> str:
    """``scheme://host[:port]`` of a URL, for messages and for comparing hosts."""
    parts = urlsplit(url)
    return f"{parts.scheme}://{parts.netloc.rpartition('@')[2]}"


def _post(url: str, body: dict[str, Any]) -> dict[str, Any]:
    request = urllib.request.Request(
        url,
        data=json.dumps(body).encode("utf-8"),
        headers={"Content-Type": "application/json", "Accept": "application/geo+json"},
        method="POST",
    )
    with urllib.request.urlopen(request, timeout=_TIMEOUT_S) as response:
        return _read_json(response, url)


def _get(url: str) -> dict[str, Any]:
    with urllib.request.urlopen(url, timeout=_TIMEOUT_S) as response:
        return _read_json(response, url)


def _next_url(catalog: str, current: str, href: Any) -> str:
    """The URL of a ``next`` link, which must stay on the catalog's scheme and host:
    a catalog is remote input, and may not send mapcv to a file or another server."""
    following = urljoin(current, str(href))
    if _origin(following).lower() != _origin(catalog).lower():
        shown = _origin(following) if "://" in following else following.partition(":")[0] + ":"
        raise RuntimeError(
            f"STAC catalog {_origin(catalog)} links its next page to {shown}; mapcv follows "
            "next links on the catalog's own scheme and host only"
        )
    return following


@contextmanager
def local_paths_checked(check: Callable[[Path], None]) -> Iterator[None]:
    """Run ``check`` on every local file an item found inside the block points to.

    The MCP server uses it to keep a catalog (a local test catalog may name local
    files) from pointing outside its root. ``check`` raises to refuse a path.
    """
    global _local_path_check
    with _check_lock:
        _check_users.append(check)
        _local_path_check = check
    try:
        yield
    finally:
        with _check_lock:
            _check_users.remove(check)
            _local_path_check = _check_users[-1] if _check_users else None


def _check_local_assets(item: dict[str, Any]) -> None:
    """Refuse an item whose assets name a local file the active check rejects."""
    check = _local_path_check
    assets = item.get("assets")
    if check is None or not isinstance(assets, dict):
        return
    for asset in assets.values():
        href = asset.get("href") if isinstance(asset, dict) else None
        if isinstance(href, str):
            local = eopf_local_path(href)
            if local is not None:
                check(local if local.is_absolute() else Path.cwd() / local)


def search_items(
    search: StacSearchBase, bbox: tuple[float, float, float, float]
) -> Iterator[dict[str, Any]]:
    """Every item of the search, page by page (at most 5,000)."""
    url: str | None = f"{search.catalog}/search"
    body: dict[str, Any] | None = {
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
            (
                link
                for link in page.get("links") or []
                if isinstance(link, dict) and link.get("rel") == "next"
            ),
            None,
        )
        if following is None:
            return
        url = _next_url(search.catalog, url, following.get("href"))
        method = str(following.get("method", "GET")).upper()
        if method == "POST":
            # A next link may replace the body (often with a token) or merge into it.
            new_body = following.get("body") or {}
            body = {**(body or {}), **new_body} if following.get("merge") else (new_body or body)


def find_item(
    search: StacSearchBase, bbox: tuple[float, float, float, float]
) -> tuple[dict[str, Any], int]:
    """The item to use for ``bbox`` (west, south, east, north in degrees), and how many
    qualified.

    Raises:
        ValueError: No item of the search covers the region with at most ``max_cloud``
            percent cloud.
        RuntimeError: The catalog cannot be reached or answers with an error.
    """
    region = box(*bbox)
    try:
        items = list(search_items(search, bbox))
    except OSError as exc:  # urllib's URLError and HTTPError are OSErrors
        raise RuntimeError(f"STAC search at {search.catalog} failed: {exc}") from exc
    candidates: list[tuple[float, str, str, dict[str, Any]]] = []
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
    *_, item = min(candidates, key=lambda c: (c[0], c[1], c[2]))
    _check_local_assets(item)
    return item, len(candidates)


def find_product(search: StacSearchConfig, bbox: tuple[float, float, float, float]) -> StacMatch:
    """The EOPF product to use for ``bbox`` (see :func:`find_item`).

    Raises:
        ValueError: As :func:`find_item`, or the chosen item lacks the ``asset``.
        RuntimeError: As :func:`find_item`.
    """
    item, candidates = find_item(search, bbox)
    item_id = str(item.get("id", ""))
    asset = item.get("assets", {}).get(search.asset)
    if not asset or not asset.get("href"):
        raise ValueError(
            f"STAC item {item_id} has no '{search.asset}' asset (imagery.search.asset)"
        )
    properties = item.get("properties", {})
    return StacMatch(
        item_id=item_id,
        href=str(asset["href"]),
        cloud_cover=float(properties["eo:cloud_cover"]),
        datetime=str(properties.get("datetime") or ""),
        candidates=candidates,
    )
