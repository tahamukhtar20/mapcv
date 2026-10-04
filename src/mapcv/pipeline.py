"""Source-neutral dataset generation pipeline."""

from __future__ import annotations

import hashlib
import time
import warnings
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, DefaultDict, Dict, List, Optional, Tuple

import numpy as np
import numpy.typing as npt
from rich.console import Console
from shapely.geometry import box
from rich.progress import (
    BarColumn,
    MofNCompleteColumn,
    Progress,
    SpinnerColumn,
    TextColumn,
    TimeElapsedColumn,
    TimeRemainingColumn,
)

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


@dataclass
class GenerateResult:
    """Outcome of :func:`run_generate`."""

    staging_dir: Path
    manifest: Manifest
    new_patches: int
    split_counts: Optional[Dict[str, int]]
    tiles_requested: int
    tiles_failed: int
    seconds: float


def _chunk_progress(disable: bool = False) -> Progress:
    return Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        MofNCompleteColumn(),
        TextColumn("chunks •"),
        TimeElapsedColumn(),
        TextColumn("• eta"),
        TimeRemainingColumn(),
        console=_console,
        disable=disable,
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
    return transformed, class_map


LABELS_MISS_MESSAGE = (
    "no label polygon intersects the imagery extent, so every mask will be background. "
    "Check that labels are longitude/latitude (not swapped) and cover the configured region."
)
# Above this share of failed tiles a run is very likely misconfigured.
_FAILED_TILES_WARNING = 0.5


def _labels_fingerprint(config: MapcvConfig) -> Optional[Dict[str, Any]]:
    """Label settings plus a hash of the label file, so resume notices edits."""
    if config.labels is None:
        return None
    settings = config.labels.model_dump(mode="json", exclude={"path"})
    settings["sha256"] = hashlib.sha256(config.labels.path.read_bytes()).hexdigest()
    return settings


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
        warnings.warn(LABELS_MISS_MESSAGE, UserWarning, stacklevel=2)


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


def run_generate(config: MapcvConfig) -> GenerateResult:
    """Generate a patch dataset from the configured imagery source."""
    started = time.monotonic()
    staging = config.writer.staging_dir
    staging.mkdir(parents=True, exist_ok=True)
    manifest_path = staging / _MANIFEST_FILENAME

    with _console.status("Opening imagery…"):
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
            labels=_labels_fingerprint(config),
            writer=config.writer.model_dump(mode="json", exclude={"staging_dir"}),
        )

        resumed_patches = len(manifest.patches)
        anchors = _global_anchors(source.metadata.height, source.metadata.width, config.sampler)
        completed_anchors = {(patch["row"], patch["col"]) for patch in manifest.patches}
        # Number chunks over the whole raster so a resumed run records the same
        # chunk index for each patch as an uninterrupted one.
        chunks = []
        for chunk_index, group in enumerate(_group_anchors(anchors, source.metadata.chunk_rows)):
            remaining = [anchor for anchor in group if anchor not in completed_anchors]
            if remaining:
                chunks.append((chunk_index, remaining))
        to_go = sum(len(group) for _, group in chunks)
        if resumed_patches and not chunks:
            _console.print(
                f"[dim]Nothing left to do: all {resumed_patches} patch(es) are already "
                "written.[/dim]"
            )
        elif resumed_patches:
            _console.print(
                f"[dim]Resuming: {resumed_patches} patch(es) already written, {to_go} to go[/dim]"
            )
        with _chunk_progress(disable=not chunks) as progress:
            task = progress.add_task("Reading imagery and writing patches", total=len(chunks))
            for chunk_index, chunk_anchors in chunks:
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
        requested = int(getattr(source, "tiles_requested", 0))
        failed = int(getattr(source, "tiles_failed", 0))
        if requested and failed / requested > _FAILED_TILES_WARNING:
            warnings.warn(
                f"{failed} of {requested} tiles failed; check the tile URL, your network and "
                "imagery.policy (failed tiles are left empty or black).",
                UserWarning,
                stacklevel=2,
            )
        elif failed and config.sampler.max_empty_ratio >= 1.0:
            warnings.warn(
                f"{failed} tile(s) failed and were filled with black; patches that include them "
                "were kept, with labels over black pixels. Set sampler.max_empty_ratio below 1 "
                "(for example 0.5) to drop such patches.",
                UserWarning,
                stacklevel=2,
            )
    finally:
        source.close()

    split_counts: Optional[Dict[str, int]] = None
    if config.split is not None:
        split_counts = split_dataset(manifest, config.split, staging / _SPLITS_SUBDIR)

    return GenerateResult(
        staging_dir=staging,
        manifest=manifest,
        new_patches=len(manifest.patches) - resumed_patches,
        split_counts=split_counts,
        tiles_requested=requested,
        tiles_failed=failed,
        seconds=time.monotonic() - started,
    )


def run_split(
    staging_dir: Path,
    split_config: Optional[SplitterConfig] = None,
) -> Dict[str, int]:
    """Split an existing version-1 or version-2 dataset manifest; return split counts."""
    manifest_path = staging_dir / _MANIFEST_FILENAME
    if not manifest_path.exists():
        raise FileNotFoundError(f"No manifest found at {manifest_path}")

    manifest = Manifest.load(manifest_path)
    cfg = split_config or SplitterConfig()
    splits_dir = staging_dir / _SPLITS_SUBDIR
    return split_dataset(manifest, cfg, splits_dir)
