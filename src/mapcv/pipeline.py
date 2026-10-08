"""Source-neutral dataset generation pipeline.

The library never prints: messages go to the ``mapcv`` logger (``logging``), problems
are ``warnings``, and progress is reported through callbacks. The CLI turns these into
its spinner, progress bar and styled output.
"""

from __future__ import annotations

import json
import logging
import os
import time
import warnings
from collections import Counter, defaultdict
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Union

import numpy as np
import numpy.typing as npt

from mapcv._mapcv_rs import grid_sample_anchors
from mapcv._patching import extract_array_patch
from mapcv.aoi import AreaOfInterest, column_clusters
from mapcv.config import GeoTiffImageryConfig, MapcvConfig
from mapcv.footprints import FOOTPRINTS_FILENAME, write_footprints
from mapcv.imagery import (
    AlignedSource,
    WindowedRasterSource,
    grid_alignment,
    offset_transform,
    open_raster_source,
)
from mapcv.locking import LOCK_FILENAME, StagingDirError, dataset_lock
from mapcv.manifest import Manifest, SourceRecord, load_or_create_manifest, mapcv_version
from mapcv.sampler import (
    PatchMeta,
    SamplerConfig,
    random_anchors_for,
    sample_annotated_patches,
)
from mapcv.splitter import SplitLists, SplitterConfig, split_manifest
from mapcv.targets import Target, create_target
from mapcv.targets.base import Annotation, WindowTarget
from mapcv.writers import FilesWriter, check_compatible, create_writer, refresh_split_outputs

_log = logging.getLogger(__name__)
# How often a running generation rewrites manifest.json (seconds).
_SAVE_EVERY_S = 10.0

_MANIFEST_FILENAME = "manifest.json"
_SPLITS_SUBDIR = "splits"


def _patches(count: int) -> str:
    """``1 patch``, ``1,234 patches``, for messages the CLI shows."""
    return f"{count:,} {'patch' if count == 1 else 'patches'}"


def _tiles(count: int) -> str:
    """``1 tile``, ``12 tiles``, for messages the CLI shows."""
    return f"{count:,} {'tile' if count == 1 else 'tiles'}"


@dataclass
class GenerateResult:
    """Outcome of :func:`run_generate`."""

    staging_dir: Path
    manifest: Manifest
    new_patches: int
    split_counts: dict[str, int] | None
    tiles_requested: int
    tiles_failed: int
    seconds: float
    # XYZ tiles read from the on-disk cache instead of downloaded.
    tiles_cached: int = 0
    # Patches left out of this run because they had no pixel of imagery at all.
    patches_without_imagery: int = 0


def _global_anchors(height: int, width: int, config: SamplerConfig) -> list[tuple[int, int]]:
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


def _group_anchors(anchors: list[tuple[int, int]], chunk_rows: int) -> list[list[tuple[int, int]]]:
    grouped: defaultdict[int, list[tuple[int, int]]] = defaultdict(list)
    for anchor in anchors:
        grouped[anchor[0] // chunk_rows].append(anchor)
    return [grouped[index] for index in sorted(grouped)]


# Above this share of failed tiles a run is very likely misconfigured.
_FAILED_TILES_WARNING = 0.5


def _process_anchor_chunk(
    source: WindowedRasterSource,
    anchors: list[tuple[int, int]],
    sampler: SamplerConfig,
    target: Target,
    others: dict[str, AlignedSource] | None = None,
    counts: Counter[str] | None = None,
) -> tuple[
    npt.NDArray[Any], list[Annotation], list[PatchMeta], dict[str, npt.NDArray[Any]], WindowTarget
]:
    """Read one chunk's window and sample its patches.

    ``others`` are further sources read on ``source``'s grid. A pixel has imagery
    only where every source has it; the kept patches of every other source are
    returned by name, in the same order as ``source``'s. The annotations are per
    patch; the returned window collates them for a writer.
    """
    patch_size = sampler.patch_size
    row_start = min(row for row, _ in anchors)
    row_stop = min(source.metadata.height, max(row + patch_size for row, _ in anchors))
    col_start = min(col for _, col in anchors)
    col_stop = min(source.metadata.width, max(col + patch_size for _, col in anchors))

    image, valid_mask = source.read_window(row_start, row_stop, col_start, col_stop)
    other_images: dict[str, npt.NDArray[Any]] = {}
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
        counts=counts,
    )
    other_patches: dict[str, npt.NDArray[Any]] = {}
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
    return images, annotations, metadata, other_patches, window


def _check_stackable(records: list[SourceRecord]) -> None:
    """Refuse sources that cannot share one (T, C, H, W) array (``writer.stack_sources``)."""
    first = records[0]
    for record in records[1:]:
        if record.dtype != first.dtype or len(record.bands) != len(first.bands):
            raise ValueError(
                f"writer.stack_sources needs every source to have the same bands and data type "
                f"to stack them: '{first.name}' has {len(first.bands)} band(s) of {first.dtype}, "
                f"'{record.name}' {len(record.bands)} of {record.dtype}; select matching bands "
                "or write the sources as separate files"
            )


def _open_sources(config: MapcvConfig) -> list[WindowedRasterSource]:
    """Open every imagery source in order, closing the opened ones if one fails."""
    opened: list[WindowedRasterSource] = []
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


@contextmanager
def _sources(
    config: MapcvConfig,
) -> Iterator[tuple[list[WindowedRasterSource], dict[str, AlignedSource]]]:
    """The opened sources and, by name, every further one aligned to the first's grid;
    all are closed on exit."""
    _log.debug("Opening imagery")
    opened = _open_sources(config)
    try:
        # Every further source is read on the first one's grid.
        others = {
            name: AlignedSource(other, grid_alignment(opened[0].metadata, other.metadata, name))
            for name, other in zip(config.source_names[1:], opened[1:])
        }
        yield opened, others
    finally:
        for each in opened:
            each.close()


def _area_of_interest(config: MapcvConfig, source: WindowedRasterSource) -> AreaOfInterest | None:
    if config.region.path is None:
        return None
    return AreaOfInterest(config.region, source.metadata.crs, source.metadata.transform)


# A chunk's window (every source's pixels, read at once) stays below this many bytes:
# a wider chunk is read as several column windows, so memory does not grow with the
# raster's width (a 10980-column, 12-band float32 Sentinel-2 strip is 540 MB).
_WINDOW_BYTES = 256 * 2**20


def _pixel_bytes(sources: list[WindowedRasterSource]) -> int:
    """Bytes of one pixel of every source together, as read."""
    return sum(
        max(1, len(each.metadata.bands)) * np.dtype(each.metadata.dtype).itemsize
        for each in sources
    )


def _max_window_width(rows: int, width: int, pixel_bytes: int, patch_size: int) -> int:
    """The widest window of ``rows`` rows read at once: ``width`` when it fits in
    :data:`_WINDOW_BYTES`, else what fits (but at least two patches wide)."""
    if rows * width * pixel_bytes <= _WINDOW_BYTES:
        return width
    return max(2 * patch_size, _WINDOW_BYTES // (rows * pixel_bytes))


def _column_windows(
    group: list[tuple[int, int]], patch_size: int, pixel_bytes: int
) -> list[list[tuple[int, int]]]:
    """``group`` as it is, or, when its window would pass :data:`_WINDOW_BYTES`, split
    into column ranges whose windows stay below it (left to right, order kept)."""
    rows = max(row for row, _ in group) + patch_size - min(row for row, _ in group)
    first = min(col for _, col in group)
    width = max(col for _, col in group) + patch_size - first
    max_width = _max_window_width(rows, width, pixel_bytes, patch_size)
    if max_width >= width:
        return [group]
    step = max_width - patch_size
    split: dict[int, list[tuple[int, int]]] = {}
    for anchor in group:
        split.setdefault((anchor[1] - first) // step, []).append(anchor)
    return [split[key] for key in sorted(split)]


def _anchor_groups(
    config: MapcvConfig,
    sources: list[WindowedRasterSource],
    aoi: AreaOfInterest | None,
) -> list[list[tuple[int, int]]]:
    """Every patch anchor of the raster, grouped into the chunks they are read in."""
    meta = sources[0].metadata
    patch_size = config.sampler.patch_size
    anchors = _global_anchors(meta.height, meta.width, config.sampler)
    groups = _group_anchors(anchors, meta.chunk_rows)
    if aoi is not None:
        # Only patches over the polygons, read in windows around them.
        anchors = aoi.keep(anchors, patch_size)
        groups = [
            cluster
            for group in _group_anchors(anchors, meta.chunk_rows)
            for cluster in column_clusters(group, patch_size)
        ]
    pixel_bytes = _pixel_bytes(sources)
    return [part for group in groups for part in _column_windows(group, patch_size, pixel_bytes)]


@dataclass
class Patch:
    """One patch of :func:`iter_patches`: its pixels, target and place on the raster.

    ``image`` is ``(H, W, C)`` (as read; ``(H, W, 3)`` uint8 for XYZ), ``target`` the
    task's annotation of the patch (a class mask for segmentation, boxes for detection,
    and so on; ``None`` without labels), ``row``/``col`` the patch's top-left pixel
    on the first source's grid, ``transform`` its affine transform in ``crs``, and
    ``others`` the patches of further imagery sources by name.
    """

    row: int
    col: int
    image: npt.NDArray[Any]
    target: Any
    transform: tuple[float, float, float, float, float, float]
    crs: str
    padded: bool
    others: dict[str, npt.NDArray[Any]]


ConfigLike = Union[MapcvConfig, str, "os.PathLike[str]"]


def _config(config: ConfigLike) -> MapcvConfig:
    return config if isinstance(config, MapcvConfig) else MapcvConfig.from_yaml(config)


def generate(
    config: ConfigLike, *, progress: Callable[[int, int], None] | None = None
) -> GenerateResult:
    """Build (or resume) the dataset a config describes; the library form of
    ``mapcv generate``.

    ``config`` is a :class:`~mapcv.MapcvConfig` or the path of a YAML file.
    ``progress(done, total)`` is called with the chunks written so far and the chunks
    of this run, before the first chunk and after each one; raising from it stops
    the run (finished chunks are kept, and calling again resumes). Nothing is
    printed: messages go to the ``mapcv`` logger and problems are warnings.
    """
    return run_generate(_config(config), progress)


def split(
    dataset: str | os.PathLike[str],
    config: SplitterConfig | None = None,
    **settings: Any,
) -> dict[str, int]:
    """Rewrite a dataset's split lists from its manifest; the library form of
    ``mapcv split``. Returns the patch count of each split and ``dropped``.

    Pass a :class:`~mapcv.SplitterConfig`, or its fields as keyword arguments
    (``split("dataset", strategy="random", test_ratio=0.25)``).
    """
    if config is not None and settings:
        raise TypeError("pass a SplitterConfig or keyword settings, not both")
    return run_split(Path(dataset), config or SplitterConfig(**settings))


def iter_patches(config: ConfigLike) -> Iterator[Patch]:
    """Yield every patch the config describes, in order, without writing anything.

    The same patches (and filters, such as ``sampler.max_empty_ratio`` and
    ``min_label_ratio``) as :func:`run_generate`, read chunk by chunk, so memory stays
    bounded; ``writer`` settings are not used. Useful to feed a model directly or to
    look at a config's output before generating it.
    """
    config = _config(config)
    target = create_target(config)
    with _sources(config) as (opened, others):
        source = opened[0]
        target.prepare(source.metadata)
        crs = source.metadata.crs
        for group in _anchor_groups(config, opened, _area_of_interest(config, source)):
            images, annotations, metadata, other_patches, _ = _process_anchor_chunk(
                source, group, config.sampler, target, others
            )
            for index, item in enumerate(metadata):
                yield Patch(
                    row=item["row"],
                    col=item["col"],
                    image=images[index],
                    target=annotations[index],
                    transform=offset_transform(source.metadata.transform, item["row"], item["col"]),
                    crs=crs,
                    padded=item["padded"],
                    others={name: patches[index] for name, patches in other_patches.items()},
                )


# Folders and files a run writes before its first manifest.json, or that are not the
# dataset's (a lock, a file browser's leftovers): a folder holding only these is a
# dataset folder whose first run stopped early.
_OWN_BEFORE_MANIFEST = frozenset(
    {
        "Images",
        "Masks",
        "images",
        "labels",
        "masks",
        "annotations",
        "A",
        "B",
        "label",
        "manifest.json.tmp",
        LOCK_FILENAME,
        ".DS_Store",
        "Thumbs.db",
        "desktop.ini",
    }
)


def _check_staging_dir(staging: Path) -> None:
    """Refuse a folder with files of its own and no manifest: mapcv would overwrite (and,
    for some layouts, remove) files such as ``train.txt`` or ``dataset.yaml`` there."""
    if not staging.is_dir() or (staging / _MANIFEST_FILENAME).exists():
        return
    foreign = sorted(
        entry.name for entry in staging.iterdir() if entry.name not in _OWN_BEFORE_MANIFEST
    )
    if foreign:
        raise StagingDirError(
            f"writer.staging_dir {staging} already holds files that mapcv did not write "
            f"(first: {foreign[0]}) and no manifest.json; generate into a new or empty "
            "folder, so none of them is overwritten or removed"
        )


def _first_broken_patch(staging: Path, manifest: Manifest) -> tuple[int, str] | None:
    """``(index, path)`` of the first patch with a missing or empty file, or ``None``.

    The index is the first patch of that patch's chunk: a resumed run writes the chunk
    again from there, numbering its files as an uninterrupted run would.
    """
    base = os.path.join(staging, "")  # plain strings: a Path per file is several times slower
    for index, entry in enumerate(manifest.patches):
        for rel in entry["files"].values():
            try:
                broken = os.stat(base + rel).st_size == 0
            except OSError:
                broken = True
            if broken:
                chunk = entry["chunk"]
                first = index
                while first > 0 and manifest.patches[first - 1]["chunk"] == chunk:
                    first -= 1
                return first, rel
    return None


def run_generate(
    config: MapcvConfig, on_chunk: Callable[[int, int], None] | None = None
) -> GenerateResult:
    """Generate a patch dataset from the configured imagery source.

    The pipeline is task-agnostic: the target (``create_target``) says what each
    patch is annotated with and the writer (``create_writer``) how it reaches disk.

    ``on_chunk(done, total)``, if given, is called with the chunks written so far and
    the chunks of this run: once before the first chunk and after every chunk. An
    exception it raises stops the run there; the finished chunks are saved and the
    same call again resumes.

    The manifest is saved every :data:`_SAVE_EVERY_S` seconds and whenever the run
    stops, normally or with an exception (Ctrl-C included); a process killed outright
    loses at most those seconds of chunks, which a resumed run writes again. It records
    ``complete: false`` until the run has finished, splits and annotations included. A
    resumed run also writes again the patches whose files went missing or empty.

    Raises:
        DatasetBusyError: Another mapcv run is writing to ``writer.staging_dir``.
        StagingDirError: ``writer.staging_dir`` holds other files and no manifest.
    """
    started = time.monotonic()
    staging = config.writer.staging_dir
    _check_staging_dir(staging)
    staging.mkdir(parents=True, exist_ok=True)
    with dataset_lock(staging):
        return _generate_locked(config, on_chunk, started)


def _generate_locked(
    config: MapcvConfig, on_chunk: Callable[[int, int], None] | None, started: float
) -> GenerateResult:
    staging = config.writer.staging_dir
    manifest_path = staging / _MANIFEST_FILENAME

    target = create_target(config)
    names = config.source_names
    if config.multi_source:
        writer = create_writer(config.writer, target, names)
    else:
        writer = create_writer(config.writer, target)
    check_compatible(target, writer)
    counts: Counter[str] = Counter()
    with _sources(config) as (opened, others):
        source = opened[0]
        target.prepare(source.metadata)
        records: list[SourceRecord] = []
        for name, opened_source in zip(names, opened):
            meta = opened_source.metadata
            grid: dict[str, Any] = {}
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
                    width=meta.width,
                    height=meta.height,
                    fingerprint=meta.fingerprint,
                    **grid,
                )
            )
        if config.writer.stack_sources:
            _check_stackable(records)
        aoi = _area_of_interest(config, source)
        # An area of interest is recorded so a resumed run notices other polygons.
        region_record: dict[str, Any] = {"region": aoi.record()} if aoi is not None else {}
        expected = Manifest(
            mapcv_version=mapcv_version(),
            complete=False,
            task=config.task,
            sources=records,
            target=target.record(),
            writer=writer.fingerprint(),
            sampler=config.sampler.model_dump(mode="json"),
            **region_record,
        )
        manifest = load_or_create_manifest(manifest_path, expected)
        # Whether the dataset on disk was finished (None: a manifest from before mapcv 0.3).
        was_complete = manifest.complete if manifest_path.exists() else False
        broken = _first_broken_patch(staging, manifest)
        if broken is not None:
            index, rel = broken
            warnings.warn(
                f"{rel} is missing or empty, so {_patches(len(manifest.patches) - index)} "
                "from its chunk on are written again",
                UserWarning,
                stacklevel=3,
            )
            del manifest.patches[index:]
            was_complete = False
        manifest.complete = False

        resumed_patches = len(manifest.patches)
        patch_size = config.sampler.patch_size
        groups = _anchor_groups(config, opened, aoi)
        completed_anchors = {(patch["row"], patch["col"]) for patch in manifest.patches}
        # Number chunks over the whole raster so a resumed run records the same
        # chunk index for each patch as an uninterrupted one.
        chunks = []
        for chunk_index, group in enumerate(groups):
            remaining = [anchor for anchor in group if anchor not in completed_anchors]
            if remaining:
                chunks.append((chunk_index, remaining))
        to_go = sum(len(group) for _, group in chunks)
        _log.debug(
            "%d chunk(s) of %d to process (%d anchors); source %s %s, %dx%d px, %s; "
            "window budget %d MB",
            len(chunks),
            len(groups),
            to_go,
            source.metadata.source_type,
            source.metadata.product_id,
            source.metadata.width,
            source.metadata.height,
            source.metadata.crs,
            _WINDOW_BYTES // 2**20,
        )
        if resumed_patches and not chunks:
            _log.info(
                "Nothing left to do: %s already written.",
                "the 1 patch is"
                if resumed_patches == 1
                else f"all {resumed_patches:,} patches are",
            )
        elif resumed_patches:
            _log.info(
                "Resuming: %s already written, %s to go.",
                _patches(resumed_patches),
                f"{to_go:,}",
            )
        if on_chunk is not None:
            on_chunk(0, len(chunks))
        saved = time.monotonic()
        complete = len(manifest.patches)  # entries of fully finished chunks
        try:
            for done, (chunk_index, chunk_anchors) in enumerate(chunks, start=1):
                chunk_started = time.perf_counter()
                images, per_patch, metadata, other_patches, window = _process_anchor_chunk(
                    source, chunk_anchors, config.sampler, target, others, counts
                )
                read_s = time.perf_counter() - chunk_started
                annotations = window.collate(per_patch, patch_size)
                if other_patches:
                    if not isinstance(writer, FilesWriter):  # pragma: no cover - create_writer
                        raise RuntimeError("several imagery sources need the files layout")
                    writer.write(
                        images, annotations, metadata, manifest, chunk_index, others=other_patches
                    )
                else:
                    writer.write(images, annotations, metadata, manifest, chunk_index)
                if aoi is not None:
                    for entry in manifest.patches[len(manifest.patches) - len(metadata) :]:
                        entry["summary"]["region"] = aoi.region_of(
                            entry["row"], entry["col"], patch_size
                        )
                complete = len(manifest.patches)
                _log.debug(
                    "chunk %d: %d anchor(s), %d patch(es) kept; read and annotate %.2f s, "
                    "write %.2f s",
                    chunk_index,
                    len(chunk_anchors),
                    len(metadata),
                    read_s,
                    time.perf_counter() - chunk_started - read_s,
                )
                # Persist every few seconds (rewriting a large manifest after every chunk
                # costs more than the chunk), and below whenever the run stops early.
                if time.monotonic() - saved >= _SAVE_EVERY_S:
                    manifest.save(manifest_path)
                    saved = time.monotonic()
                    _log.debug("manifest saved: %d patch(es)", len(manifest.patches))
                if on_chunk is not None:
                    on_chunk(done, len(chunks))
        except BaseException:
            # Interrupted (Ctrl-C, a failed chunk, a cancelling callback): keep the
            # finished chunks, so the same call resumes after them. A chunk stopped
            # part way is dropped; a resumed run writes it again over its files.
            _log.debug(
                "stopped: keeping %d patch(es) of finished chunks, dropping %d",
                complete,
                len(manifest.patches) - complete,
            )
            del manifest.patches[complete:]
            manifest.save(manifest_path)
            raise
        requested = sum(int(getattr(each, "tiles_requested", 0)) for each in opened)
        failed = sum(int(getattr(each, "tiles_failed", 0)) for each in opened)
        cached = sum(int(getattr(each, "tiles_cached", 0)) for each in opened)
        reasons = "; ".join(
            text for each in opened if (text := str(getattr(each, "failure_reasons", "") or ""))
        )
        why = f" Causes: {reasons}." if reasons else ""
        if requested and failed / requested > _FAILED_TILES_WARNING:
            warnings.warn(
                f"{failed:,} of {requested:,} tiles failed; check the tile URL, your network and "
                f"imagery.policy (failed tiles are left empty or black).{why}",
                UserWarning,
                stacklevel=3,
            )
        elif failed and config.sampler.max_empty_ratio >= 1.0:
            warnings.warn(
                f"{_tiles(failed)} failed and {'was' if failed == 1 else 'were'} filled with "
                f"black; patches that include them were kept{_failed_tile_masks(manifest)}. "
                "Set sampler.max_empty_ratio below 1 (for example 0.5) to drop such patches."
                f"{why}",
                UserWarning,
                stacklevel=3,
            )
    if counts["no_imagery"]:
        _log.info(
            "Left out %s without any imagery (failed tiles, NoData or outside the imagery).",
            _patches(counts["no_imagery"]),
        )

    split_counts: dict[str, int] | None = None
    split_lists: SplitLists | None = None
    try:
        # A run with nothing new keeps the dataset's splits, which `mapcv split` may have
        # changed since: re-splitting here would undo that (and break SHA256SUMS).
        kept = (
            _existing_splits(staging / _SPLITS_SUBDIR) if resumed_patches and not chunks else None
        )
        if kept is not None:
            split_counts, split_lists, settings = kept
            if config.split is not None and settings != config.split.model_dump(mode="json"):
                warnings.warn(
                    f"{_SPLITS_SUBDIR}/ was made with other split settings (by mapcv split) and "
                    f"is kept as it is; run `mapcv split {staging}` to apply other settings",
                    UserWarning,
                    stacklevel=3,
                )
        elif config.split is not None:
            split_counts, split_lists = split_manifest(
                manifest, config.split, staging / _SPLITS_SUBDIR
            )
        writer.finalize(manifest, split_lists)
    except BaseException:
        # Every patch is written, but splits or annotations may not be: the manifest says
        # the dataset is incomplete, and the same config writes them again.
        manifest.save(manifest_path)
        raise
    # A finished dataset is left untouched (a 0.2 manifest stays version 2).
    if chunks or was_complete is False:
        manifest.complete = True
        manifest.save(manifest_path)
    else:
        manifest.complete = was_complete

    return GenerateResult(
        staging_dir=staging,
        manifest=manifest,
        new_patches=len(manifest.patches) - resumed_patches,
        split_counts=split_counts,
        tiles_requested=requested,
        tiles_failed=failed,
        seconds=time.monotonic() - started,
        tiles_cached=cached,
        patches_without_imagery=counts["no_imagery"],
    )


def _failed_tile_masks(manifest: Manifest) -> str:
    """What the masks of patches with failed tiles hold there, for the warning."""
    if manifest.target is None or manifest.task == "classification":
        return ""
    if manifest.ignore_index is not None:
        return f"; their masks mark those pixels with labels.ignore_index ({manifest.ignore_index})"
    return ", with labels over black pixels (labels.ignore_index is null)"


def _existing_splits(
    splits_dir: Path,
) -> tuple[dict[str, int], SplitLists, dict[str, Any]] | None:
    """The split lists already in ``splits_dir`` (counts, lists, settings), or ``None``
    when they are incomplete."""
    record_path = splits_dir / "split.json"
    names = ("train", "val", "test")
    if not record_path.is_file() or not all((splits_dir / f"{n}.txt").is_file() for n in names):
        return None
    try:
        record = json.loads(record_path.read_text(encoding="utf-8"))
        lists = {n: (splits_dir / f"{n}.txt").read_text(encoding="utf-8").split() for n in names}
    except (OSError, ValueError):
        return None
    counts = {n: len(lists[n]) for n in names}
    counts["dropped"] = int(record.get("dropped", 0))
    settings = record.get("settings") if isinstance(record, dict) else None
    return counts, SplitLists(**lists), settings if isinstance(settings, dict) else {}


def run_split(
    staging_dir: Path,
    split_config: SplitterConfig | None = None,
) -> dict[str, int]:
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
    with dataset_lock(staging_dir):
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
