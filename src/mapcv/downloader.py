"""Tile downloading: fetch XYZ tiles for a region and stitch them into one image."""

from __future__ import annotations

import hashlib
import logging
from collections.abc import Callable
from functools import cache
from io import BytesIO

import numpy as np
import numpy.typing as npt
from PIL import Image

from mapcv._mapcv_rs import (
    TileIndex,
    snap_bbox,
    tiles,
)
from mapcv._mapcv_rs import (
    fetch_tiles as fetch_tiles_rs,
)
from mapcv._mapcv_rs import (
    stitch_tiles as stitch_tiles_rs,
)
from mapcv._mapcv_rs import (
    tile_transform as tile_transform_rs,
)

URL_TEMPLATES: dict[str, str] = {
    # Esri
    "esri_satellite": "https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}",
    "esri_topo": "https://server.arcgisonline.com/ArcGIS/rest/services/World_Topo_Map/MapServer/tile/{z}/{y}/{x}",
    "esri_street": "https://server.arcgisonline.com/ArcGIS/rest/services/World_Street_Map/MapServer/tile/{z}/{y}/{x}",
    # CartoDB
    "cartodb_positron": "https://a.basemaps.cartocdn.com/light_all/{z}/{x}/{y}.png",
    "cartodb_dark_matter": "https://a.basemaps.cartocdn.com/dark_all/{z}/{x}/{y}.png",
}

# Tiles servers send with HTTP 200 where they have no imagery, by (size, SHA-256): they
# are NoData, not imagery. Esri's grey "Map data not yet available" tile, the same
# bytes for World_Imagery, World_Topo_Map and World_Street_Map at any zoom past
# their coverage.
PLACEHOLDER_TILES: frozenset[tuple[int, str]] = frozenset(
    {(2521, "9eafd300d61393184a4abc1d458564cfd1cd9b6f9c4e9c74687045c0a0e5b858")}
)

_log = logging.getLogger(__name__)


def is_placeholder_tile(payload: bytes) -> bool:
    """Whether a tile is a server's "no imagery here" placeholder (see PLACEHOLDER_TILES)."""
    return any(
        len(payload) == size and hashlib.sha256(payload).hexdigest() == digest
        for size, digest in PLACEHOLDER_TILES
    )


@cache
def _black_tile() -> bytes:
    buffer = BytesIO()
    Image.new("RGB", (256, 256)).save(buffer, "PNG")
    return buffer.getvalue()


def resolve_url_template(url_template: str | None, source: str | None) -> str:
    """Return a URL template string from an explicit template or a built-in source name."""
    if url_template:
        return url_template
    if source:
        if source not in URL_TEMPLATES:
            raise ValueError(f"Unknown tile source: {source}")
        return URL_TEMPLATES[source]
    raise ValueError("Provide url_template or source")


def download_region(
    west: float,
    south: float,
    east: float,
    north: float,
    zoom: int,
    url_template: str | None = None,
    source: str | None = None,
    max_connections: int = 16,
    policy: str = "lenient",
    snap_to_tiles: bool = True,
    max_failed_ratio: float = 0.05,
    progress: Callable[[int, int], None] | None = None,
) -> list[tuple[TileIndex, bytes]]:
    """Fetch all tiles for a bbox at the given zoom and return (tile, bytes) pairs.

    policy controls failure handling: "strict" raises on any failure, "lenient"
    skips failed tiles, "ignore" fills them with black NoData pixels.
    A server's "no imagery here" placeholder tile comes back black (NoData), as a
    failed tile does under "ignore", without counting as failed.
    snap_to_tiles expands the bbox outward to tile boundaries before fetching.
    ``progress(done, total)`` is called with the tiles finished so far. Prints
    nothing; the outcome is logged to the ``mapcv`` logger.
    Raises RuntimeError if the failed-tile fraction exceeds max_failed_ratio.
    """
    template = resolve_url_template(url_template, source)
    if snap_to_tiles:
        snapped = snap_bbox(west, south, east, north, zoom)
        west, south, east, north = snapped.west, snapped.south, snapped.east, snapped.north

    target_tiles = tiles(west, south, east, north, [zoom])
    total = len(target_tiles)

    if progress is not None:
        progress(0, total)
    results, failed_count, _ = fetch_tiles_rs(
        target_tiles,
        template,
        callback=(lambda done: progress(done, total)) if progress is not None else None,
        max_connections=max_connections,
        policy=policy,
        max_failed_ratio=max_failed_ratio,
    )

    placeholders = sum(1 for _, payload in results if is_placeholder_tile(payload))
    if placeholders:
        results = [
            (tile, _black_tile() if is_placeholder_tile(payload) else payload)
            for tile, payload in results
        ]
        _log.info(
            "%d tile(s) had no imagery (the server sent its placeholder tile) and are black",
            placeholders,
        )

    fetched = len(results)
    if policy == "ignore":
        _log.info(
            "%d/%d tiles returned (%d failures filled with NoData under 'ignore' policy)",
            fetched,
            total,
            failed_count,
        )
    else:
        _log.info("%d/%d tiles fetched, %d failed", fetched, total, failed_count)

    return results


def stitch_region(
    west: float,
    south: float,
    east: float,
    north: float,
    zoom: int,
    url_template: str | None = None,
    source: str | None = None,
    max_connections: int = 16,
    policy: str = "lenient",
    max_failed_ratio: float = 0.05,
    progress: Callable[[int, int], None] | None = None,
) -> tuple[npt.NDArray[np.uint8], tuple[float, float, float, float, float, float]]:
    """Fetch tiles, decode in parallel, and return a stitched (H, W, 3) image with its transform.

    Returns ``(image_array, transform)`` where ``transform`` is the
    ``(a, b, c, d, e, f)`` affine 6-tuple mapping pixel ``(col, row)`` to
    Web Mercator ``(x, y)`` in metres (rasterio ``Affine`` convention).
    """
    tile_data = download_region(
        west=west,
        south=south,
        east=east,
        north=north,
        zoom=zoom,
        url_template=url_template,
        source=source,
        max_connections=max_connections,
        policy=policy,
        snap_to_tiles=True,
        max_failed_ratio=max_failed_ratio,
        progress=progress,
    )
    image_array, min_x, min_y = stitch_tiles_rs(tile_data)
    transform = tile_transform_rs(min_x, min_y, zoom)
    return image_array, transform
