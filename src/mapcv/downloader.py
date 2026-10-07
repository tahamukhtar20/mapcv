"""Tile downloading: fetch XYZ tiles for a region and stitch them into one image."""

from __future__ import annotations

import logging
from collections.abc import Callable

import numpy as np
import numpy.typing as npt

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

_log = logging.getLogger(__name__)


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
