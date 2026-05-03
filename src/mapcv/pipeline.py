"""Core pipeline logic: fetch -> rasterize -> patch -> write -> split."""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Optional

import numpy as np
import numpy.typing as npt
from rich.console import Console
from rich.progress import BarColumn, Progress, SpinnerColumn, TextColumn, TimeElapsedColumn

from mapcv._mapcv_rs import stitch_tiles as stitch_tiles_rs
from mapcv._mapcv_rs import tile_transform as tile_transform_rs
from mapcv.config import MapcvConfig
from mapcv.downloader import download_region_strips, resolve_url_template
from mapcv.labels import parse_geojson, parse_kml, transform_to_mercator
from mapcv.rasterizer import rasterize
from mapcv.sampler import SamplerConfig, sample_patches
from mapcv.splitter import SplitterConfig, split_dataset
from mapcv.writer import Manifest, WriterConfig, load_or_create_manifest, write_patches

_console = Console()

_MANIFEST_FILENAME = "manifest.json"
_SPLITS_SUBDIR = "splits"


def _strip_progress() -> Progress:
    return Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
        "•",
        TimeElapsedColumn(),
        console=_console,
    )


def run_generate(config: MapcvConfig) -> None:
    """Execute the full generate pipeline from *config*.

    Steps: parse labels -> fetch tiles in strips -> per-strip
    (stitch, rasterize, sample, write) -> save manifest.
    If ``config.split`` is set, also run the splitter afterwards.
    """
    staging = config.writer.staging_dir
    staging.mkdir(parents=True, exist_ok=True)
    manifest_path = staging / _MANIFEST_FILENAME

    geoms_with_class = []
    class_map: Dict[str, int] = {}
    if config.labels is not None:
        _console.print("[bold]Parsing labels...[/bold]")
        data = config.labels.path.read_bytes()
        suffix = config.labels.path.suffix.lower()
        if suffix == ".kml":
            raw, class_map = parse_kml(data, config.labels.label_field)
        else:
            raw, class_map = parse_geojson(data, config.labels.label_field)
        geoms_with_class = [(transform_to_mercator(g), c) for g, c in raw]
        n_classes = len(class_map) if class_map else (1 if geoms_with_class else 0)
        _console.print(f"[dim]  {len(geoms_with_class)} polygon(s), {n_classes} class(es)[/dim]")

    manifest: Manifest = load_or_create_manifest(manifest_path, class_map)

    template = resolve_url_template(config.tiles.url_template, config.tiles.source)
    all_strips = download_region_strips(
        config.region.west,
        config.region.south,
        config.region.east,
        config.region.north,
        config.region.zoom,
        config.tiles.strip_rows,
        url_template=template,
        max_connections=config.tiles.max_connections,
        policy=config.tiles.policy,
        max_failed_ratio=config.tiles.max_failed_ratio,
    )

    writer_cfg = WriterConfig(
        staging_dir=staging,
        image_format=config.writer.image_format,
        jpg_quality=config.writer.jpg_quality,
    )
    sampler_cfg = SamplerConfig(**config.sampler.model_dump())

    _console.print(f"[bold]Processing {len(all_strips)} strip(s)...[/bold]")
    with _strip_progress() as progress:
        task = progress.add_task("Strips", total=len(all_strips))
        for strip_idx, strip_tile_data in enumerate(all_strips):
            progress.update(task, description=f"Strip {strip_idx + 1}/{len(all_strips)}")
            if not strip_tile_data:
                progress.advance(task)
                continue

            image_array, min_x, min_y = stitch_tiles_rs(strip_tile_data)
            transform = tile_transform_rs(min_x, min_y, config.region.zoom)

            mask: Optional[npt.NDArray[np.uint8]] = None
            if geoms_with_class:
                h, w = image_array.shape[:2]
                mask = rasterize(
                    geoms_with_class,
                    (h, w),
                    transform,
                    config.labels.all_touched if config.labels else False,
                )

            imgs, msks, meta = sample_patches(image_array, mask, sampler_cfg)
            write_patches(imgs, msks, meta, writer_cfg, manifest, strip_index=strip_idx)
            progress.advance(task)

    manifest.save(manifest_path)
    total_patches = len(manifest.patches)
    _console.print(
        f"[green]Done.[/green] {total_patches} patch(es) written to [bold]{staging}[/bold]"
    )

    if config.split is not None:
        _console.print("[bold]Splitting dataset...[/bold]")
        splits_dir = staging / _SPLITS_SUBDIR
        split_dataset(manifest, config.split, splits_dir)
        _console.print(f"[green]Splits written to[/green] [bold]{splits_dir}[/bold]")


def run_split(
    staging_dir: Path,
    split_config: Optional[SplitterConfig] = None,
) -> None:
    """Split an existing dataset at *staging_dir* using its manifest.

    Writes split lists to ``staging_dir/splits/``.
    """
    manifest_path = staging_dir / _MANIFEST_FILENAME
    if not manifest_path.exists():
        raise FileNotFoundError(f"No manifest found at {manifest_path}")

    manifest = Manifest.load(manifest_path)
    cfg = split_config or SplitterConfig()
    splits_dir = staging_dir / _SPLITS_SUBDIR
    split_dataset(manifest, cfg, splits_dir)
    _console.print(f"[green]Splits written to[/green] [bold]{splits_dir}[/bold]")
