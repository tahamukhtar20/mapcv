import time
from typing import Optional, List, Tuple
from rich.progress import Progress, SpinnerColumn, BarColumn, TextColumn, TimeElapsedColumn, DownloadColumn
from mapcv._mapcv_rs import fetch_tiles as fetch_tiles_rs, PyTileIndex, tile, tiles

def download_region(
    west: float,
    south: float,
    east: float,
    north: float,
    zoom: int,
    url_template: str,
    max_connections: int = 16,
    policy: str = "lenient"
) -> List[Tuple[PyTileIndex, bytes]]:
    """
    Downloads a bounding box of tiles at the specified zoom level.
    Uses Rust under the hood for highly concurrent fetching and provides a rich progress bar.
    """
    target_tiles = tiles(west, south, east, north, [zoom])
    total = len(target_tiles)
    
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
        "•",
        DownloadColumn(),
        "•",
        TimeElapsedColumn(),
    ) as progress:
        task_id = progress.add_task(f"Fetching {total} tiles...", total=total)
        
        def progress_callback(completed: int) -> None:
            progress.update(task_id, completed=completed)
            
        results = fetch_tiles_rs(
            target_tiles,
            url_template,
            callback=progress_callback,
            max_connections=max_connections,
            policy=policy
        )
        
        progress.update(task_id, completed=total)
        
    return results
