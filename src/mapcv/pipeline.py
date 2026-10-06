"""Source-neutral dataset generation pipeline."""

from __future__ import annotations

import time
import warnings
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, DefaultDict, Dict, List, Optional, Tuple

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
    TimeRemainingColumn,
)

from mapcv._mapcv_rs import grid_sample_anchors
from mapcv._patching import extract_array_patch
from mapcv.config import GeoTiffImageryConfig, MapcvConfig
from mapcv.footprints import FOOTPRINTS_FILENAME, write_footprints
from mapcv.imagery import (
    AlignedSource,
    WindowedRasterSource,
    grid_alignment,
    offset_transform,
    open_raster_source,
)
from mapcv.manifest import Manifest, SourceRecord, load_or_create_manifest, mapcv_version
from mapcv.sampler import (
    PatchMeta,
    SamplerConfig,
    random_anchors_for,
    sample_annotated_patches,
)
from mapcv.splitter import SplitLists, SplitterConfig, split_manifest
from mapcv.targets import AnnotationBatch, Target, create_target
from mapcv.writers import FilesWriter, check_compatible, create_writer, refresh_split_outputs

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
    others: Optional[Dict[str, AlignedSource]] = None,
) -> Tuple[npt.NDArray[Any], AnnotationBatch, List[PatchMeta], Dict[str, npt.NDArray[Any]]]:
    """Read one chunk's window and sample its patches.

    ``others`` are further sources read on ``source``'s grid. A pixel has imagery
    only where every source has it; the kept patches of every other source are
    returned by name, in the same order as ``source``'s.
    """
    patch_size = sampler.patch_size
    row_start = min(row for row, _ in anchors)
    row_stop = min(source.metadata.height, max(row + patch_size for row, _ in anchors))
    col_start = min(col for _, col in anchors)
    col_stop = min(source.metadata.width, max(col + patch_size for _, col in anchors))

    image, valid_mask = source.read_window(row_start, row_stop, col_start, col_stop)
    other_images: Dict[str, npt.NDArray[Any]] = {}
    for name, other in (others or {}).items():
        other_image, other_valid = other.read_window(row_start, row_stop, col_start, col_stop)
        other_images[name] = other_image
        valid_mask = valid_mask & other_valid
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
    other_patches: Dict[str, npt.NDArray[Any]] = {}
    for name, other_image in other_images.items():
        kept = [
            extract_array_patch(
                other_image,
                item["row"] - row_start,
                item["col"] - col_start,
                patch_size,
                sampler.pad_mode,
            )[0]
            for item in metadata
        ]
        other_patches[name] = (
            np.stack(kept, axis=0)
            if kept
            else np.empty((0, patch_size, patch_size, *other_image.shape[2:]), other_image.dtype)
        )
    return images, window.collate(annotations, patch_size), metadata, other_patches


def _open_sources(config: MapcvConfig) -> List[WindowedRasterSource]:
    """Open every imagery source in order, closing the opened ones if one fails."""
    opened: List[WindowedRasterSource] = []
    try:
        for imagery in config.sources:
            if isinstance(imagery, GeoTiffImageryConfig):
                # A GeoTIFF's bands and dtype must fit the output format; checked at open.
                opened.append(
                    open_raster_source(
                        config.region, imagery, image_format=config.writer.image_format
                    )
                )
            else:
                opened.append(open_raster_source(config.region, imagery))
    except BaseException:
        for source in opened:
            source.close()
        raise
    return opened


def run_generate(
    config: MapcvConfig, on_chunk: Optional[Callable[[int, int], None]] = None
) -> GenerateResult:
    """Generate a patch dataset from the configured imagery source.

    The pipeline is task-agnostic: the target (``create_target``) says what each
    patch is annotated with and the writer (``create_writer``) how it reaches disk.

    ``on_chunk(done, total)``, if given, is called with the chunks written so far and
    the chunks of this run: once before the first chunk and after every chunk (whose
    manifest is already saved). An exception it raises stops the run there; the
    finished chunks stay and the same call again resumes.
    """
    started = time.monotonic()
    staging = config.writer.staging_dir
    staging.mkdir(parents=True, exist_ok=True)
    manifest_path = staging / _MANIFEST_FILENAME

    target = create_target(config)
    names = config.source_names
    if config.multi_source:
        writer = create_writer(config.writer, target, names)
    else:
        writer = create_writer(config.writer, target)
    check_compatible(target, writer)
    with _console.status("Opening imagery…"):
        opened = _open_sources(config)
    source = opened[0]
    try:
        # Every further source is read on the first one's grid.
        others = {
            name: AlignedSource(other, grid_alignment(source.metadata, other.metadata, name))
            for name, other in zip(names[1:], opened[1:])
        }
        target.prepare(source.metadata)
        records: List[SourceRecord] = []
        for name, opened_source in zip(names, opened):
            meta = opened_source.metadata
            grid: Dict[str, Any] = {}
            aligned = others.get(name)
            if aligned is not None and not aligned.alignment.identity:
                # How the source's own grid maps onto the dataset's (the first source's).
                alignment = aligned.alignment
                grid = {
                    "factor": alignment.factor,
                    "offset": [alignment.row_offset, alignment.col_offset],
                }
            records.append(
                SourceRecord(
                    name=name,
                    source_type=meta.source_type,
                    product_id=meta.product_id,
                    bands=list(meta.bands),
                    dtype=meta.dtype,
                    crs=meta.crs,
                    transform=meta.transform,
                    patch_shape=writer.patch_shape(meta, config.sampler.patch_size),
                    fingerprint=meta.fingerprint,
                    **grid,
                )
            )
        expected = Manifest(
            mapcv_version=mapcv_version(),
            task=config.task,
            sources=records,
            target=target.record(),
            writer=writer.fingerprint(),
            sampler=config.sampler.model_dump(mode="json"),
        )
        manifest = load_or_create_manifest(manifest_path, expected)

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
            if on_chunk is not None:
                on_chunk(0, len(chunks))
            for done, (chunk_index, chunk_anchors) in enumerate(chunks, start=1):
                images, annotations, metadata, other_patches = _process_anchor_chunk(
                    source, chunk_anchors, config.sampler, target, others
                )
                if other_patches:
                    if not isinstance(writer, FilesWriter):  # pragma: no cover - create_writer
                        raise RuntimeError("several imagery sources need the files layout")
                    writer.write(
                        images, annotations, metadata, manifest, chunk_index, others=other_patches
                    )
                else:
                    writer.write(images, annotations, metadata, manifest, chunk_index)
                # Persist after every chunk so an interrupted run resumes from here.
                manifest.save(manifest_path)
                progress.advance(task)
                if on_chunk is not None:
                    on_chunk(done, len(chunks))

        # A finished dataset is left untouched (a 0.2 manifest stays version 2).
        if chunks or not manifest_path.exists():
            manifest.save(manifest_path)
        requested = sum(int(getattr(each, "tiles_requested", 0)) for each in opened)
        failed = sum(int(getattr(each, "tiles_failed", 0)) for each in opened)
        reasons = "; ".join(
            text for each in opened if (text := str(getattr(each, "failure_reasons", "") or ""))
        )
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
        for each in opened:
            each.close()

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
    """Split an existing dataset (manifest version 1, 2 or 3); return split counts.

    The manifest is read, never rewritten. Outputs that depend on the split
    (``patches.geojson``, a detection dataset's COCO files, YOLO image lists
    and ``dataset.yaml``, and a classification dataset's label tables) are rebuilt.
    """
    manifest_path = staging_dir / _MANIFEST_FILENAME
    if not manifest_path.exists():
        raise FileNotFoundError(f"No manifest found at {manifest_path}")

    manifest = Manifest.load(manifest_path)
    cfg = split_config or SplitterConfig()
    splits_dir = staging_dir / _SPLITS_SUBDIR
    counts, lists = split_manifest(manifest, cfg, splits_dir)
    footprints = staging_dir / FOOTPRINTS_FILENAME
    if footprints.exists():
        # Keep the footprint index's ``split`` property in step with the new lists.
        try:
            write_footprints(manifest, lists, footprints)
        except (RuntimeError, ValueError) as exc:
            warnings.warn(
                f"{FOOTPRINTS_FILENAME} was not updated: {exc}", UserWarning, stacklevel=2
            )
    refresh_split_outputs(manifest, staging_dir, lists)
    return counts
