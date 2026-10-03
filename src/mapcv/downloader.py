"""Tile downloading: fetch XYZ tiles for a region and stitch them into one image."""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple, cast

import numpy as np
import numpy.typing as npt
from rich.console import Console
from rich.progress import (
    BarColumn,
    MofNCompleteColumn,
    Progress,
    SpinnerColumn,
    TextColumn,
    TimeElapsedColumn,
)

from mapcv._mapcv_rs import (
    PyTileIndex,
    fetch_tiles as fetch_tiles_rs,
    snap_bbox,
    stitch_tiles as stitch_tiles_rs,
    tile_transform as tile_transform_rs,
    tiles,
)

URL_TEMPLATES: Dict[str, str] = {
    # Esri
    "esri_satellite": "https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}",
    "esri_topo": "https://server.arcgisonline.com/ArcGIS/rest/services/World_Topo_Map/MapServer/tile/{z}/{y}/{x}",
    "esri_street": "https://server.arcgisonline.com/ArcGIS/rest/services/World_Street_Map/MapServer/tile/{z}/{y}/{x}",
    # CartoDB
    "cartodb_positron": "https://a.basemaps.cartocdn.com/light_all/{z}/{x}/{y}.png",
    "cartodb_dark_matter": "https://a.basemaps.cartocdn.com/dark_all/{z}/{x}/{y}.png",
}

_console = Console()


def _in_jupyter() -> bool:
    try:
        import IPython.core.getipython as _gip

        return cast(Any, _gip.get_ipython)() is not None
    except (ImportError, AttributeError):
        return False


def resolve_url_template(url_template: Optional[str], source: Optional[str]) -> str:
    """Return a URL template string from an explicit template or a built-in source name."""
    if url_template:
        return url_template
    if source:
        if source not in URL_TEMPLATES:
            raise ValueError(f"Unknown tile source: {source}")
        return URL_TEMPLATES[source]
    raise ValueError("Provide url_template or source")


def _make_progress() -> Progress:
    return Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
        "•",
        MofNCompleteColumn(),
        "•",
        TimeElapsedColumn(),
    )


def download_region(
    west: float,
    south: float,
    east: float,
    north: float,
    zoom: int,
    url_template: Optional[str] = None,
    source: Optional[str] = None,
    max_connections: int = 16,
    policy: str = "lenient",
    snap_to_tiles: bool = True,
    max_failed_ratio: float = 0.05,
) -> List[Tuple[PyTileIndex, bytes]]:
    """Fetch all tiles for a bbox at the given zoom and return (tile, bytes) pairs.

    policy controls failure handling: "strict" raises on any failure, "lenient"
    skips failed tiles, "ignore" fills them with black NoData pixels.
    snap_to_tiles expands the bbox outward to tile boundaries before fetching.
    Raises RuntimeError if the failed-tile fraction exceeds max_failed_ratio.
    """
    template = resolve_url_template(url_template, source)
    if snap_to_tiles:
        snapped = snap_bbox(west, south, east, north, zoom)
        west, south, east, north = snapped.west, snapped.south, snapped.east, snapped.north

    target_tiles = tiles(west, south, east, north, [zoom])
    total = len(target_tiles)

    if _in_jupyter():
        print(f"Fetching {total} tiles...", end=" ", flush=True)
        results, failed_count = fetch_tiles_rs(
            target_tiles,
            template,
            callback=lambda _: None,
            max_connections=max_connections,
            policy=policy,
            max_failed_ratio=max_failed_ratio,
        )
        print("done.")
    else:
        with _make_progress() as progress:
            task_id = progress.add_task(f"Fetching {total} tiles...", total=total)

            def progress_callback(completed: int) -> None:
                progress.update(task_id, completed=completed)

            results, failed_count = fetch_tiles_rs(
                target_tiles,
                template,
                callback=progress_callback,
                max_connections=max_connections,
                policy=policy,
                max_failed_ratio=max_failed_ratio,
            )
            progress.update(task_id, completed=total)

    fetched = len(results)
    if policy == "ignore":
        _console.print(
            f"[dim]{fetched}/{total} tiles returned"
            f" ({failed_count} failures filled with NoData under 'ignore' policy)[/dim]"
        )
    else:
        _console.print(f"[dim]{fetched}/{total} tiles fetched, {failed_count} failed[/dim]")

    return cast(List[Tuple[PyTileIndex, bytes]], results)


def stitch_region(
    west: float,
    south: float,
    east: float,
    north: float,
    zoom: int,
    url_template: Optional[str] = None,
    source: Optional[str] = None,
    max_connections: int = 16,
    policy: str = "lenient",
    max_failed_ratio: float = 0.05,
) -> Tuple[npt.NDArray[np.uint8], Tuple[float, float, float, float, float, float]]:
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
    )
    image_array, min_x, min_y = stitch_tiles_rs(tile_data)
    transform = tile_transform_rs(min_x, min_y, zoom)
    return image_array, transform
