"""Tile downloading: fetch, stitch, and strip-partition satellite tiles."""

from __future__ import annotations

from typing import Callable, Dict, List, Optional, Tuple

import numpy as np
import numpy.typing as npt
from rich.console import Console
from rich.progress import BarColumn, DownloadColumn, Progress, SpinnerColumn, TextColumn, TimeElapsedColumn

from mapcv._mapcv_rs import (
    PyTileIndex,
    fetch_tiles as fetch_tiles_rs,
    snap_bbox,
    stitch_tiles as stitch_tiles_rs,
    tile_transform as tile_transform_rs,
    tiles,
)

URL_TEMPLATES: Dict[str, str] = {
    # Google
    "google_satellite": "https://mt1.google.com/vt/lyrs=s&x={x}&y={y}&z={z}",
    # OpenStreetMap
    "osm": "https://tile.openstreetmap.org/{z}/{x}/{y}.png",
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
        return _gip.get_ipython() is not None  # type: ignore[no-untyped-call]
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


def iter_tile_strips(
    west: float,
    south: float,
    east: float,
    north: float,
    zoom: int,
    strip_rows: int,
) -> List[List[PyTileIndex]]:
    """Partition the tile grid for a bbox into horizontal strips of strip_rows tile rows each."""
    if strip_rows <= 0:
        raise ValueError("strip_rows must be positive")
    target_tiles = tiles(west, south, east, north, [zoom])
    target_tiles.sort(key=lambda t: (t.y, t.x))
    rows: Dict[int, List[PyTileIndex]] = {}
    for t in target_tiles:
        rows.setdefault(t.y, []).append(t)
    row_keys = sorted(rows.keys())
    strips: List[List[PyTileIndex]] = []
    for i in range(0, len(row_keys), strip_rows):
        strip_tiles: List[PyTileIndex] = []
        for y in row_keys[i : i + strip_rows]:
            strip_tiles.extend(rows[y])
        strips.append(strip_tiles)
    return strips


def _make_progress() -> Progress:
    return Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
        "•",
        DownloadColumn(),
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
        results: List[Tuple[PyTileIndex, bytes]] = fetch_tiles_rs(
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

            results = fetch_tiles_rs(
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
            " (failures are filled with NoData under 'ignore' policy)[/dim]"
        )
    else:
        _console.print(f"[dim]{fetched}/{total} tiles fetched, {total - fetched} failed[/dim]")

    return results


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


def download_region_strips(
    west: float,
    south: float,
    east: float,
    north: float,
    zoom: int,
    strip_rows: int,
    url_template: Optional[str] = None,
    source: Optional[str] = None,
    max_connections: int = 16,
    policy: str = "lenient",
    snap_to_tiles: bool = True,
    max_failed_ratio: float = 0.05,
) -> List[List[Tuple[PyTileIndex, bytes]]]:
    """Fetch tiles for a bbox in horizontal strips, caching tiles shared between strips.

    Returns one inner list per strip; each entry is a (tile, bytes) pair.
    strip_rows controls how many tile rows form a single strip.
    """
    template = resolve_url_template(url_template, source)
    if snap_to_tiles:
        snapped = snap_bbox(west, south, east, north, zoom)
        west, south, east, north = snapped.west, snapped.south, snapped.east, snapped.north

    strips = iter_tile_strips(west, south, east, north, zoom, strip_rows)
    total = sum(len(strip) for strip in strips)

    # shared across strips so overlapping tiles (stride-based patches) are fetched only once
    tile_cache: Dict[Tuple[int, int, int], bytes] = {}

    all_results: List[List[Tuple[PyTileIndex, bytes]]] = []
    total_fetched = 0
    total_failed = 0

    if _in_jupyter():
        print(f"Fetching {total} tiles ({len(strips)} strips)...", end=" ", flush=True)
        for strip in strips:
            to_fetch: List[PyTileIndex] = []
            cached_results: List[Tuple[PyTileIndex, bytes]] = []
            for t in strip:
                key = (t.x, t.y, t.z)
                if key in tile_cache:
                    cached_results.append((t, tile_cache[key]))
                else:
                    to_fetch.append(t)
            fresh: List[Tuple[PyTileIndex, bytes]] = []
            if to_fetch:
                fresh = fetch_tiles_rs(
                    to_fetch, template, callback=lambda _: None,
                    max_connections=max_connections, policy=policy,
                    max_failed_ratio=max_failed_ratio,
                )
                for t, b in fresh:
                    tile_cache[(t.x, t.y, t.z)] = b
            strip_results = cached_results + fresh
            strip_failed = 0 if policy == "ignore" else len(strip) - len(strip_results)
            total_fetched += len(strip_results)
            total_failed += strip_failed
            all_results.append(strip_results)
        print("done.")
    else:
        tiles_done = 0
        with _make_progress() as progress:
            task_id = progress.add_task("Fetching tiles (strips)...", total=total)

            for strip in strips:
                to_fetch2: List[PyTileIndex] = []
                cached_results2: List[Tuple[PyTileIndex, bytes]] = []
                for t in strip:
                    key = (t.x, t.y, t.z)
                    if key in tile_cache:
                        cached_results2.append((t, tile_cache[key]))
                    else:
                        to_fetch2.append(t)

                if cached_results2:
                    tiles_done += len(cached_results2)
                    progress.update(task_id, completed=tiles_done)

                fresh2: List[Tuple[PyTileIndex, bytes]] = []
                if to_fetch2:
                    _base = tiles_done

                    def _make_callback(base: int) -> Callable[[int], None]:
                        def _cb(completed: int) -> None:
                            progress.update(task_id, completed=base + completed)

                        return _cb

                    fresh2 = fetch_tiles_rs(
                        to_fetch2,
                        template,
                        callback=_make_callback(_base),
                        max_connections=max_connections,
                        policy=policy,
                        max_failed_ratio=max_failed_ratio,
                    )
                    for t, b in fresh2:
                        tile_cache[(t.x, t.y, t.z)] = b

                    tiles_done += len(to_fetch2)
                    progress.update(task_id, completed=tiles_done)

                strip_results2 = cached_results2 + fresh2
                # under 'ignore' policy len(strip) - len(strip_results2) is always 0 and misleading
                strip_failed2 = 0 if policy == "ignore" else len(strip) - len(strip_results2)
                total_fetched += len(strip_results2)
                total_failed += strip_failed2
                all_results.append(strip_results2)

    if policy == "ignore":
        _console.print(
            f"[dim]{total_fetched}/{total} tiles returned"
            " (failures are filled with NoData under 'ignore' policy)[/dim]"
        )
    else:
        _console.print(
            f"[dim]{total_fetched}/{total} tiles fetched, {total_failed} failed[/dim]"
        )

    return all_results
