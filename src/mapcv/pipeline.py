"""Source-neutral dataset generation pipeline."""

from __future__ import annotations

from collections import defaultdict
from pathlib import Path
from typing import Any, DefaultDict, Dict, List, Optional, Tuple

import numpy as np
import numpy.typing as npt
from rich.console import Console
from shapely.geometry import box
from rich.progress import BarColumn, Progress, SpinnerColumn, TextColumn, TimeElapsedColumn

from mapcv._mapcv_rs import grid_sample_anchors, random_sample_anchors
from mapcv.config import MapcvConfig
from mapcv.imagery import (
    WindowedRasterSource,
    offset_transform,
    open_raster_source,
    transform_geometry_to_crs,
)
from mapcv.labels import GeomWithClass, parse_geojson, parse_kml, transform_to_mercator
from mapcv.rasterizer import rasterize
from mapcv.sampler import PatchMeta, SamplerConfig, sample_patches_at_anchors
from mapcv.splitter import SplitterConfig, split_dataset
from mapcv.writer import Manifest, load_or_create_manifest, write_patches

_console = Console()

_MANIFEST_FILENAME = "manifest.json"
_SPLITS_SUBDIR = "splits"


def _chunk_progress() -> Progress:
    return Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
        "•",
        TimeElapsedColumn(),
        console=_console,
    )


def _print_split_summary(counts: Dict[str, int], splits_dir: Path) -> None:
    total = counts["train"] + counts["val"] + counts["test"]
    shares = "  ".join(
        f"{name}={counts[name]} ({counts[name] / total:.0%})" if total else f"{name}=0"
        for name in ("train", "val", "test")
    )
    _console.print(f"[green]Splits written to[/green] [bold]{splits_dir}[/bold]: {shares}")
    if counts["dropped"]:
        _console.print(
            f"[dim]  {counts['dropped']} train/val patch(es) overlapping a held-out patch "
            "were left out[/dim]"
        )


def _global_anchors(height: int, width: int, config: SamplerConfig) -> List[Tuple[int, int]]:
    if config.mode == "random":
        return list(
            random_sample_anchors(
                height,
                width,
                config.patch_size,
                config.random_count,
                config.random_seed,
                config.edge_strategy,
            )
        )
    return list(
        grid_sample_anchors(
            height,
            width,
            config.patch_size,
            config.stride,
            config.edge_strategy,
        )
    )


def _group_anchors(anchors: List[Tuple[int, int]], chunk_rows: int) -> List[List[Tuple[int, int]]]:
    grouped: DefaultDict[int, List[Tuple[int, int]]] = defaultdict(list)
    for anchor in anchors:
        grouped[anchor[0] // chunk_rows].append(anchor)
    return [grouped[index] for index in sorted(grouped)]


def _parse_labels(
    config: MapcvConfig, destination_crs: str
) -> Tuple[List[GeomWithClass], Dict[str, int]]:
    if config.labels is None:
        return [], {}

    _console.print("[bold]Parsing labels...[/bold]")
    data = config.labels.path.read_bytes()
    labels = config.labels
    if labels.path.suffix.lower() == ".kml":
        raw, class_map = parse_kml(data, labels.label_field, labels.classes)
    else:
        raw, class_map = parse_geojson(data, labels.label_field, labels.classes)

    if destination_crs.upper() == "EPSG:3857":
        transformed = [(transform_to_mercator(geometry), class_id) for geometry, class_id in raw]
    else:
        transformed = [
            (transform_geometry_to_crs(geometry, destination_crs), class_id)
            for geometry, class_id in raw
        ]
    n_classes = len(class_map) if class_map else (1 if transformed else 0)
    _console.print(f"[dim]  {len(transformed)} polygon(s), {n_classes} class(es)[/dim]")
    return transformed, class_map


def _raster_bounds(source: WindowedRasterSource) -> Tuple[float, float, float, float]:
    a, _, c, _, e, f = source.metadata.transform
    xs = (c, c + a * source.metadata.width)
    ys = (f, f + e * source.metadata.height)
    return min(xs), min(ys), max(xs), max(ys)


def _warn_if_labels_miss_raster(
    geometries: List[GeomWithClass], source: WindowedRasterSource
) -> None:
    if not geometries:
        return
    extent = box(*_raster_bounds(source))
    if not any(geometry.intersects(extent) for geometry, _ in geometries):
        _console.print(
            "[yellow]Warning:[/yellow] no label polygon intersects the imagery extent, so "
            "every mask will be background. Check that labels are longitude/latitude "
            "(not swapped) and cover the configured region."
        )


def _process_anchor_chunk(
    source: WindowedRasterSource,
    anchors: List[Tuple[int, int]],
    config: MapcvConfig,
    geometries: List[GeomWithClass],
) -> Tuple[npt.NDArray[Any], Optional[npt.NDArray[np.uint8]], List[PatchMeta]]:
    patch_size = config.sampler.patch_size
    row_start = min(row for row, _ in anchors)
    row_stop = min(source.metadata.height, max(row + patch_size for row, _ in anchors))
    col_start = min(col for _, col in anchors)
    col_stop = min(source.metadata.width, max(col + patch_size for _, col in anchors))

    image, valid_mask = source.read_window(row_start, row_stop, col_start, col_stop)
    local_anchors = [(row - row_start, col - col_start) for row, col in anchors]
    mask: Optional[npt.NDArray[np.uint8]] = None
    if geometries:
        mask = rasterize(
            geometries,
            (image.shape[0], image.shape[1]),
            offset_transform(source.metadata.transform, row_start, col_start),
            config.labels.all_touched if config.labels else False,
        )

    images, masks, metadata = sample_patches_at_anchors(
        image,
        mask,
        local_anchors,
        config.sampler,
        row_offset=row_start,
        col_offset=col_start,
        valid_mask=valid_mask,
    )
    return images, masks, metadata


def run_generate(config: MapcvConfig) -> None:
    """Generate a patch dataset from the configured imagery source."""
    staging = config.writer.staging_dir
    staging.mkdir(parents=True, exist_ok=True)
    manifest_path = staging / _MANIFEST_FILENAME

    source = open_raster_source(config.region, config.imagery)
    try:
        geometries, class_map = _parse_labels(config, source.metadata.crs)
        _warn_if_labels_miss_raster(geometries, source)
        patch_shape = (
            [len(source.metadata.bands), config.sampler.patch_size, config.sampler.patch_size]
            if config.writer.image_format == "npy"
            else [config.sampler.patch_size, config.sampler.patch_size, 3]
        )
        manifest: Manifest = load_or_create_manifest(
            manifest_path,
            class_map,
            source_type=source.metadata.source_type,
            product_id=source.metadata.product_id,
            bands=source.metadata.bands,
            dtype=source.metadata.dtype,
            patch_shape=patch_shape,
            crs=source.metadata.crs,
            transform=source.metadata.transform,
            sampler=config.sampler.model_dump(mode="json"),
        )

        anchors = _global_anchors(source.metadata.height, source.metadata.width, config.sampler)
        completed_anchors = {(patch["row"], patch["col"]) for patch in manifest.patches}
        anchors = [anchor for anchor in anchors if anchor not in completed_anchors]
        chunks = _group_anchors(anchors, source.metadata.chunk_rows)
        _console.print(f"[bold]Processing {len(chunks)} imagery chunk(s)...[/bold]")
        with _chunk_progress() as progress:
            task = progress.add_task("Chunks", total=len(chunks))
            for chunk_index, chunk_anchors in enumerate(chunks):
                progress.update(task, description=f"Chunk {chunk_index + 1}/{len(chunks)}")
                images, masks, metadata = _process_anchor_chunk(
                    source, chunk_anchors, config, geometries
                )
                write_patches(
                    images,
                    masks,
                    metadata,
                    config.writer,
                    manifest,
                    strip_index=chunk_index,
                )
                # Persist after every chunk so an interrupted run resumes from here.
                manifest.save(manifest_path)
                progress.advance(task)

        manifest.save(manifest_path)
        requested = getattr(source, "tiles_requested", 0)
        if requested:
            failed = getattr(source, "tiles_failed", 0)
            _console.print(f"[dim]  {requested} tile(s) fetched, {failed} failed[/dim]")
    finally:
        source.close()

    total_patches = len(manifest.patches)
    _console.print(
        f"[green]Done.[/green] {total_patches} patch(es) written to [bold]{staging}[/bold]"
    )

    if config.split is not None:
        _console.print("[bold]Splitting dataset...[/bold]")
        splits_dir = staging / _SPLITS_SUBDIR
        counts = split_dataset(manifest, config.split, splits_dir)
        _print_split_summary(counts, splits_dir)


def run_split(
    staging_dir: Path,
    split_config: Optional[SplitterConfig] = None,
) -> None:
    """Split an existing version-1 or version-2 dataset manifest."""
    manifest_path = staging_dir / _MANIFEST_FILENAME
    if not manifest_path.exists():
        raise FileNotFoundError(f"No manifest found at {manifest_path}")

    manifest = Manifest.load(manifest_path)
    cfg = split_config or SplitterConfig()
    splits_dir = staging_dir / _SPLITS_SUBDIR
    counts = split_dataset(manifest, cfg, splits_dir)
    _print_split_summary(counts, splits_dir)
