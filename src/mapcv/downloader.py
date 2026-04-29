from typing import Callable, Dict, List, Optional, Tuple

from rich.console import Console
from rich.progress import BarColumn, DownloadColumn, Progress, SpinnerColumn, TextColumn, TimeElapsedColumn

from mapcv._mapcv_rs import PyTileIndex, fetch_tiles as fetch_tiles_rs, snap_bbox, tiles

URL_TEMPLATES: Dict[str, str] = {
    "google_satellite": "https://mt1.google.com/vt/lyrs=s&x={x}&y={y}&z={z}",
    "osm": "https://tile.openstreetmap.org/{z}/{x}/{y}.png",
}

_console = Console()


def resolve_url_template(url_template: Optional[str], source: Optional[str]) -> str:
    """Return the URL template to use, resolving built-in source names."""
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
    """Return tiles grouped into horizontal strips of at most `strip_rows` tile rows."""
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
    """Download all tiles covering a bounding box at the given zoom level.

    Uses the Rust fetcher for concurrent HTTP requests with a rich progress bar.
    Supply either a custom XYZ `url_template` (with ``{z}``, ``{x}``, ``{y}``)
    or a built-in ``source`` name (``"google_satellite"``, ``"osm"``).

    If ``snap_to_tiles`` is ``True`` the bbox is expanded outward to full tile
    boundaries before fetching.

    Raises ``RuntimeError`` if the fraction of failed tiles exceeds
    ``max_failed_ratio`` (default 5 %).
    """
    template = resolve_url_template(url_template, source)
    if snap_to_tiles:
        snapped = snap_bbox(west, south, east, north, zoom)
        west, south, east, north = snapped.west, snapped.south, snapped.east, snapped.north

    target_tiles = tiles(west, south, east, north, [zoom])
    total = len(target_tiles)

    with _make_progress() as progress:
        task_id = progress.add_task(f"Fetching {total} tiles...", total=total)

        def progress_callback(completed: int) -> None:
            progress.update(task_id, completed=completed)

        results: List[Tuple[PyTileIndex, bytes]] = fetch_tiles_rs(
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
    """Download tiles in horizontal strips of at most `strip_rows` tile rows.

    Tiles that appear in more than one strip (e.g. when stride-based patch
    sampling creates overlap in M5) are fetched only once and served from an
    in-memory cache for subsequent strips.

    Returns a list of strip results in row order.
    """
    template = resolve_url_template(url_template, source)
    if snap_to_tiles:
        snapped = snap_bbox(west, south, east, north, zoom)
        west, south, east, north = snapped.west, snapped.south, snapped.east, snapped.north

    strips = iter_tile_strips(west, south, east, north, zoom, strip_rows)
    total = sum(len(strip) for strip in strips)

    # (x, y, z) → bytes — shared across strips so overlapping tiles are
    # fetched only once.
    tile_cache: Dict[Tuple[int, int, int], bytes] = {}

    all_results: List[List[Tuple[PyTileIndex, bytes]]] = []
    total_fetched = 0
    total_failed = 0
    tiles_done = 0  # absolute progress counter for the progress bar

    with _make_progress() as progress:
        task_id = progress.add_task("Fetching tiles (strips)...", total=total)

        for strip in strips:
            to_fetch: List[PyTileIndex] = []
            cached_results: List[Tuple[PyTileIndex, bytes]] = []
            for t in strip:
                key = (t.x, t.y, t.z)
                if key in tile_cache:
                    cached_results.append((t, tile_cache[key]))
                else:
                    to_fetch.append(t)

            # Advance progress for cache hits immediately.
            if cached_results:
                tiles_done += len(cached_results)
                progress.update(task_id, completed=tiles_done)

            fresh: List[Tuple[PyTileIndex, bytes]] = []
            if to_fetch:
                _base = tiles_done

                def _make_callback(base: int) -> Callable[[int], None]:
                    def _cb(completed: int) -> None:
                        progress.update(task_id, completed=base + completed)

                    return _cb

                fresh = fetch_tiles_rs(
                    to_fetch,
                    template,
                    callback=_make_callback(_base),
                    max_connections=max_connections,
                    policy=policy,
                    max_failed_ratio=max_failed_ratio,
                )
                for t, b in fresh:
                    tile_cache[(t.x, t.y, t.z)] = b

                tiles_done += len(to_fetch)
                progress.update(task_id, completed=tiles_done)

            strip_results = cached_results + fresh
            # Under 'ignore' policy all tiles return (some as black NoData), so
            # len(strip) - len(strip_results) would always be 0 and is misleading.
            strip_failed = 0 if policy == "ignore" else len(strip) - len(strip_results)
            total_fetched += len(strip_results)
            total_failed += strip_failed
            all_results.append(strip_results)

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
