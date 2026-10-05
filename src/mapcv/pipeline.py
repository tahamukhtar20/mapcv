"""Source-neutral dataset generation pipeline."""

from __future__ import annotations

import time
import warnings
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, DefaultDict, Dict, List, Optional, Tuple

import numpy.typing as npt
from rich.console import Console
from rich.progress import (
    BarColumn,
    MofNCompleteColumn,
    Progress,
    SpinnerColumn,
    TextColumn,
    TimeElapsedColumn,
    TimeRemainingColumn,
)

from mapcv._mapcv_rs import grid_sample_anchors
from mapcv.config import MapcvConfig
from mapcv.imagery import WindowedRasterSource, offset_transform, open_raster_source
from mapcv.sampler import (
    PatchMeta,
    SamplerConfig,
    random_anchors_for,
    sample_annotated_patches,
)
from mapcv.splitter import SplitLists, SplitterConfig, split_dataset, split_manifest
from mapcv.targets import AnnotationBatch, Target, create_target
from mapcv.writer import Manifest, load_or_create_manifest
from mapcv.writers import create_writer

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
        return random_anchors_for(height, width, config)
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


# Above this share of failed tiles a run is very likely misconfigured.
_FAILED_TILES_WARNING = 0.5


def _process_anchor_chunk(
    source: WindowedRasterSource,
    anchors: List[Tuple[int, int]],
    sampler: SamplerConfig,
    target: Target,
) -> Tuple[npt.NDArray[Any], AnnotationBatch, List[PatchMeta]]:
    patch_size = sampler.patch_size
    row_start = min(row for row, _ in anchors)
    row_stop = min(source.metadata.height, max(row + patch_size for row, _ in anchors))
    col_start = min(col for _, col in anchors)
    col_stop = min(source.metadata.width, max(col + patch_size for _, col in anchors))

    image, valid_mask = source.read_window(row_start, row_stop, col_start, col_stop)
    local_anchors = [(row - row_start, col - col_start) for row, col in anchors]
    window = target.window(
        offset_transform(source.metadata.transform, row_start, col_start),
        image.shape[0],
        image.shape[1],
        valid_mask,
    )
    images, annotations, metadata = sample_annotated_patches(
        image,
        local_anchors,
        sampler,
        window,
        row_offset=row_start,
        col_offset=col_start,
        valid_mask=valid_mask,
    )
    return images, window.collate(annotations, patch_size), metadata


def run_generate(config: MapcvConfig) -> GenerateResult:
    """Generate a patch dataset from the configured imagery source.

    The pipeline is task-agnostic: the target (``create_target``) says what each
    patch is annotated with and the writer (``create_writer``) how it reaches disk.
    """
    started = time.monotonic()
    staging = config.writer.staging_dir
    staging.mkdir(parents=True, exist_ok=True)
    manifest_path = staging / _MANIFEST_FILENAME

    with _console.status("Opening imagery…"):
        source = open_raster_source(config.region, config.imagery)
    try:
        target = create_target(config)
        writer = create_writer(config.writer)
        target.prepare(source.metadata)
        manifest: Manifest = load_or_create_manifest(
            manifest_path,
            target.class_map,
            source_type=source.metadata.source_type,
            product_id=source.metadata.product_id,
            bands=source.metadata.bands,
            dtype=source.metadata.dtype,
            patch_shape=writer.patch_shape(source.metadata, config.sampler.patch_size),
            crs=source.metadata.crs,
            transform=source.metadata.transform,
            sampler=config.sampler.model_dump(mode="json"),
            labels=target.fingerprint(),
            writer=writer.fingerprint(),
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
                images, annotations, metadata = _process_anchor_chunk(
                    source, chunk_anchors, config.sampler, target
                )
                writer.write(images, annotations, metadata, manifest, chunk_index)
                # Persist after every chunk so an interrupted run resumes from here.
                manifest.save(manifest_path)
                progress.advance(task)

        manifest.save(manifest_path)
        requested = int(getattr(source, "tiles_requested", 0))
        failed = int(getattr(source, "tiles_failed", 0))
        reasons = str(getattr(source, "failure_reasons", "") or "")
        why = f" Causes: {reasons}." if reasons else ""
        if requested and failed / requested > _FAILED_TILES_WARNING:
            warnings.warn(
                f"{failed} of {requested} tiles failed; check the tile URL, your network and "
                f"imagery.policy (failed tiles are left empty or black).{why}",
                UserWarning,
                stacklevel=2,
            )
        elif failed and config.sampler.max_empty_ratio >= 1.0:
            warnings.warn(
                f"{failed} tile(s) failed and were filled with black; patches that include them "
                "were kept, with labels over black pixels. Set sampler.max_empty_ratio below 1 "
                f"(for example 0.5) to drop such patches.{why}",
                UserWarning,
                stacklevel=2,
            )
    finally:
        source.close()

    split_counts: Optional[Dict[str, int]] = None
    split_lists: Optional[SplitLists] = None
    if config.split is not None:
        split_counts, split_lists = split_manifest(manifest, config.split, staging / _SPLITS_SUBDIR)
    writer.finalize(manifest, split_lists)

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
