"""mapcv command-line interface.

The intended journey is ``mapcv init`` (write a config), ``mapcv plan`` (see
what it will cost), ``mapcv generate`` (build the dataset) and ``mapcv info``
(inspect the result).
"""

from __future__ import annotations

import json
import re
import sys
import warnings
from collections import Counter
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional, Set, Tuple, cast
from urllib.parse import urlsplit

import numpy as np
import typer
from pydantic import ValidationError
from rich.console import Console
from rich.markup import escape
from rich.panel import Panel
from rich.prompt import Confirm, IntPrompt, Prompt
from rich.table import Table

import mapcv
from mapcv import pipeline
from mapcv._mapcv_rs import parse_kml_rs
from mapcv.config import (
    EOPFZarrImageryConfig,
    GeoTiffImageryConfig,
    MapcvConfig,
    RasterLabelsConfig,
    eopf_local_path,
)
from mapcv.labels import (
    MAX_CLASS_ID,
    _normalize_label,
    VECTOR_LABEL_SUFFIXES,
    load_vector_labels,
    vector_attributes,
    vector_layers,
)
from mapcv.manifest import Manifest, ManifestMismatchError, patch_folders
from mapcv.pipeline import GenerateResult, run_generate, run_split
from mapcv.planning import Plan, ground_resolution_m, human_bytes
from mapcv.planning import plan as make_plan
from mapcv.splitter import SplitterConfig
from mapcv.writers.detection import categories

app = typer.Typer(
    name="mapcv",
    help=(
        "Turn a region and polygon labels into a ready-to-train segmentation, detection, instance or "
        "classification dataset.\n\n"
        "Start with [bold]mapcv init[/bold], check the cost with [bold]mapcv plan[/bold], "
        "then build with [bold]mapcv generate[/bold]."
    ),
    no_args_is_help=True,
    rich_markup_mode="rich",
    context_settings={"help_option_names": ["-h", "--help"]},
    pretty_exceptions_enable=False,
)


def _make_output_encodable() -> None:
    """Keep non-ASCII output (→ ✓ ⚠ box drawing) from crashing on legacy code pages.

    On Windows, redirected or piped output (``mapcv plan x.yaml > plan.txt``, CI
    logs) uses the locale code page such as cp1252, which cannot encode these
    characters. Redirected streams switch to UTF-8; a terminal on a legacy code
    page keeps it and shows ``?`` for characters it cannot display.
    """
    for stream in (sys.stdout, sys.stderr):
        encoding = (getattr(stream, "encoding", None) or "").lower().replace("-", "")
        reconfigure = getattr(stream, "reconfigure", None)
        if encoding in ("utf8", "utf8sig") or reconfigure is None:
            continue
        try:
            if stream.isatty():
                reconfigure(errors="replace")
            else:
                reconfigure(encoding="utf-8")
        except (OSError, ValueError):  # pragma: no cover - exotic stream objects
            continue


_make_output_encodable()
_console = Console()

_DOCS_URL = "https://tahamukhtar20.github.io/mapcv"
_PROVIDERS_URL = "https://github.com/tahamukhtar20/mapcv/blob/main/PROVIDERS.md"


# ── Shared helpers ───────────────────────────────────────────────────────────


def _version() -> str:
    return mapcv.__version__


def _version_callback(value: bool) -> None:
    if value:
        typer.echo(f"mapcv {_version()}")
        raise typer.Exit()


@app.callback()
def _main(
    version: bool = typer.Option(
        False,
        "--version",
        "-V",
        callback=_version_callback,
        is_eager=True,
        help="Show the mapcv version and exit.",
    ),
    quiet: bool = typer.Option(
        False, "--quiet", "-q", help="Hide progress output; still show errors and summaries."
    ),
) -> None:
    """Turn a region and polygon labels into a ready-to-train segmentation, detection, instance or
    classification dataset."""
    pipeline._console.quiet = quiet


def _format_validation_error(exc: ValidationError) -> List[str]:
    lines = []
    for error in exc.errors():
        location = ".".join(
            str(part)
            for part in error["loc"]
            if not str(part).startswith("function-") and part not in ("xyz", "eopf_zarr", "geotiff")
        )
        message = str(error["msg"]).removeprefix("Value error, ")
        lines.append(f"  • [bold]{location or 'config'}[/bold]: {message}")
    return lines


def _load_config(config_path: Path) -> MapcvConfig:
    if not config_path.exists():
        _console.print(f"[red]Config file not found:[/red] {config_path}")
        _console.print("[dim]Create one with [bold]mapcv init[/bold].[/dim]")
        raise typer.Exit(code=1)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        try:
            config = MapcvConfig.from_yaml(config_path)
        except ValidationError as exc:
            _console.print(f"[red]Config error[/red] in {config_path}:")
            for line in _format_validation_error(exc):
                _console.print(line)
            _console.print(
                "[dim]Fix the fields above, or start from a working config with "
                "[bold]mapcv init[/bold].[/dim]"
            )
            raise typer.Exit(code=1)
        except Exception as exc:
            _console.print(f"[red]Config error:[/red] {exc}")
            raise typer.Exit(code=1)
    for warning in caught:
        warnings.showwarning(warning.message, warning.category, warning.filename, warning.lineno)
    return config


def _show_warnings(caught: List[warnings.WarningMessage], shown: Set[str]) -> None:
    """Print captured warnings once each, in mapcv's style (also under --quiet)."""
    for warning in caught:
        message = str(warning.message)
        if message in shown:
            continue
        shown.add(message)
        _console.print(f"[yellow]⚠[/yellow]  {message}")


def _redact_url(url: str) -> str:
    """Show only scheme and host of a URL that may embed credentials."""
    parsed = urlsplit(url)
    if not parsed.scheme or not parsed.hostname:
        return url
    return f"{parsed.scheme}://{parsed.hostname}/..."


def _imagery_label(config: MapcvConfig) -> str:
    imagery = config.imagery
    if isinstance(imagery, EOPFZarrImageryConfig):
        return (
            f"Sentinel-2 EOPF {_redact_url(imagery.path)} · {imagery.resolution} m · "
            f"{len(imagery.bands)} bands"
        )
    if isinstance(imagery, GeoTiffImageryConfig):
        where = _redact_url(imagery.path) if "://" in imagery.path else imagery.path
        selected = f"bands {imagery.bands}" if imagery.bands else "all bands"
        overview = f" · overview {imagery.overview}" if imagery.overview else ""
        return f"GeoTIFF {where} · {selected}{overview}"
    source = imagery.source or _redact_url(imagery.url_template or "")
    return f"XYZ {source} · zoom {imagery.zoom}"


def _task_label(config: MapcvConfig) -> str:
    if config.task == "instance":
        instance = config.instance_options
        detail = (
            f"instance · COCO RLE masks · min_visible {instance.min_visible:g} · "
            f"masks ≥ {instance.min_area} px"
        )
        if instance.id_mask:
            detail += " · instance-ID PNGs"
        return detail
    if config.task == "classification":
        classification = config.classification_options
        return (
            f"classification · {classification.mode}-label · "
            f"min_fraction {classification.min_fraction:g} · empty: {classification.empty}"
        )
    if config.task != "detection":
        return config.task
    options = config.detection_options
    detail = (
        f"detection · {', '.join(options.formats)} · min_visible {options.min_visible:g} · "
        f"boxes ≥ {options.min_box_pixels:g} px"
    )
    if options.point_box_size is not None:
        detail += f" · points as {options.point_box_size:g} px boxes"
    return detail


def _settings_table(config: MapcvConfig) -> Table:
    table = Table.grid(padding=(0, 2))
    table.add_column(style="bold cyan", no_wrap=True)
    table.add_column()
    if config.task != "segmentation":
        table.add_row("Task", _task_label(config))
    region = config.region
    table.add_row(
        "Region", f"{region.west}, {region.south} → {region.east}, {region.north} (W, S → E, N)"
    )
    table.add_row("Imagery", _imagery_label(config))
    if config.labels is None:
        table.add_row("Labels", "none (image-only dataset)")
    elif isinstance(config.labels, RasterLabelsConfig):
        raster = config.labels
        where = _redact_url(raster.path) if "://" in raster.path else raster.path
        table.add_row(
            "Labels",
            f"{where} · raster band {raster.band} · {len(raster.class_map())} class(es)",
        )
    else:
        field = config.labels.label_field or "none — every polygon is class 1"
        layer = f" · layer: {config.labels.layer}" if config.labels.layer else ""
        table.add_row("Labels", f"{config.labels.path}{layer} · field: {field}")
    sampler = config.sampler
    table.add_row(
        "Patches",
        f"{sampler.patch_size} px · {sampler.mode} · stride {sampler.stride} · "
        f"edges: {sampler.edge_strategy}",
    )
    table.add_row("Output", f"{config.writer.staging_dir} · {config.writer.image_format}")
    if config.split is None:
        table.add_row("Split", "none")
    else:
        split = config.split
        train = 1 - split.test_ratio
        table.add_row(
            "Split",
            f"{split.strategy} · test {split.test_ratio:g} · val {split.val_ratio:g} of the "
            f"remaining {train:g}",
        )
    return table


def _plan_table(config: MapcvConfig, estimate: Plan) -> Table:
    table = Table.grid(padding=(0, 2))
    table.add_column(style="bold cyan", no_wrap=True)
    table.add_column()
    width_km, height_km = estimate.region_km
    region = config.region
    if config.task != "segmentation":
        table.add_row("Task", _task_label(config))
    table.add_row(
        "Region",
        f"{region.west}, {region.south} → {region.east}, {region.north}  "
        f"[dim](≈ {width_km:.1f} × {height_km:.1f} km)[/dim]",
    )
    table.add_row("Imagery", f"{estimate.imagery} [dim](≈ {estimate.resolution_m:.2f} m/px)[/dim]")
    width, height = estimate.raster_px
    raster = f"{width:,} × {height:,} px"
    if estimate.tiles is not None and estimate.download_bytes is not None:
        raster += (
            f" · {estimate.tiles:,} tiles [dim](≈ {human_bytes(estimate.download_bytes)} "
            "to download)[/dim]"
        )
    table.add_row("Raster", raster)
    labels = estimate.labels
    if labels is None:
        table.add_row("Labels", "none (image-only dataset)")
    else:
        classes = ", ".join(f"{name} → {cid}" for name, cid in sorted(labels.classes.items()))
        if labels.raster is not None:
            where = _redact_url(labels.path) if "://" in labels.path else labels.path
            detail = f"{labels.raster} · classes: {classes or 'none (all background)'}"
            table.add_row("Labels", f"{where} · {detail}")
        else:
            detail = f"{labels.polygons:,} polygon(s)"
            detail += f" · classes: {classes}" if classes else " · every polygon is class 1"
            table.add_row("Labels", f"{labels.path} · {detail}")
    table.add_row(
        "Patches",
        f"≈ {estimate.patches:,} × {estimate.patch_size} px "
        f"[dim]({config.sampler.mode}, stride {config.sampler.stride})[/dim]",
    )
    if estimate.objects is not None:
        what = "mask" if config.task == "instance" else "box"
        table.add_row(
            "Objects",
            f"≈ {estimate.objects:,} [dim](label features in the region; one {what} each, in "
            "every patch that shows enough of it)[/dim]",
        )
    table.add_row(
        "Output",
        f"{config.writer.staging_dir} · {config.writer.image_format} "
        f"[dim](≈ {human_bytes(estimate.output_bytes)})[/dim]",
    )
    if config.split is not None:
        table.add_row(
            "Split",
            f"{config.split.strategy} · test {config.split.test_ratio:g} · "
            f"val {config.split.val_ratio:g}",
        )
    table.add_row("Memory", f"≈ {human_bytes(estimate.chunk_memory_bytes)} per chunk")
    return table


def _print_plan(config_path: Path, config: MapcvConfig, estimate: Plan) -> None:
    _console.print(
        Panel(
            _plan_table(config, estimate),
            title=f"[bold]Plan for {config_path.name}[/bold]",
            title_align="left",
            border_style="cyan",
        )
    )
    for message in estimate.warnings:
        _console.print(f"[yellow]⚠[/yellow]  {message}")
    if isinstance(config.imagery, GeoTiffImageryConfig):
        _console.print("[dim]Your own imagery: mapcv reads it as it is, without resampling.[/dim]")
    elif not isinstance(config.imagery, EOPFZarrImageryConfig):
        _console.print(
            "[dim]Imagery terms are your responsibility: check the provider's license, "
            f"attribution and rate limits ({_PROVIDERS_URL}).[/dim]"
        )


def _raster_labels(manifest: Manifest) -> bool:
    """Whether the dataset's masks were read from a label raster."""
    target = manifest.target
    return target is not None and (target.labels or {}).get("type") == "raster"


def _class_names(manifest: Manifest) -> Dict[str, str]:
    names = {str(cid): name for name, cid in manifest.class_map.items()}
    names.setdefault("0", "background")
    ignore = manifest.ignore_index
    if ignore is not None:
        names.setdefault(
            str(ignore),
            "ignored (no imagery or label)" if _raster_labels(manifest) else "ignored (no imagery)",
        )
    if not manifest.class_map:
        names.setdefault("1", "labeled")
    return names


def _object_table(manifest: Manifest) -> Optional[Table]:
    """Objects and patches with objects per class, for detection and instance datasets."""
    objects: Counter[str] = Counter()
    patches: Counter[str] = Counter()
    for entry in manifest.patches:
        counts = entry["summary"].get("class_objects") or {}
        objects.update(counts)
        patches.update(counts.keys())
    total = sum(objects.values())
    if not total:
        return None
    names = {str(cid): name for cid, name in categories(manifest.class_map).items()}
    table = Table(box=None, padding=(0, 2), show_edge=False)
    table.add_column("class")
    table.add_column("id", justify="right")
    table.add_column("objects", justify="right")
    table.add_column("share", justify="right")
    table.add_column("patches", justify="right")
    for cid in sorted(objects, key=int):
        table.add_row(
            names.get(cid, f"class {cid}"),
            cid,
            f"{objects[cid]:,}",
            f"{objects[cid] / total:.1%}",
            f"{patches[cid]:,}",
        )
    return table


def _label_table(manifest: Manifest) -> Optional[Table]:
    """Patches per label, for classification datasets (a multi-label patch counts for each)."""
    patches: Counter[str] = Counter()
    for entry in manifest.patches:
        patches.update(str(cid) for cid in entry["summary"].get("labels") or [])
    total = len(manifest.patches)
    if not patches:
        return None
    names = {str(cid): name for cid, name in categories(manifest.class_map).items()}
    names["0"] = "background"
    table = Table(box=None, padding=(0, 2), show_edge=False)
    table.add_column("label")
    table.add_column("id", justify="right")
    table.add_column("patches", justify="right")
    table.add_column("share", justify="right")
    for cid in sorted(patches, key=int):
        table.add_row(
            names.get(cid, f"class {cid}"),
            cid,
            f"{patches[cid]:,}",
            f"{patches[cid] / total:.1%}",
        )
    return table


def _class_table(manifest: Manifest) -> Optional[Table]:
    if manifest.task in ("detection", "instance"):
        return _object_table(manifest)
    if manifest.task == "classification":
        return _label_table(manifest)
    totals: Counter[str] = Counter()
    for entry in manifest.patches:
        totals.update(entry["summary"].get("class_pixels") or {})
    pixels = sum(totals.values())
    if not pixels:
        return None
    names = _class_names(manifest)
    table = Table(box=None, padding=(0, 2), show_edge=False)
    table.add_column("class")
    table.add_column("id", justify="right")
    table.add_column("pixels", justify="right")
    for cid in sorted(totals, key=int):
        table.add_row(names.get(cid, f"class {cid}"), cid, f"{totals[cid] / pixels:.1%}")
    return table


def _split_line(counts: Dict[str, int]) -> str:
    total = counts["train"] + counts["val"] + counts["test"]
    parts = [
        f"{name} {counts[name]:,} ({counts[name] / total:.0%})" if total else f"{name} 0"
        for name in ("train", "val", "test")
    ]
    line = " · ".join(parts)
    if counts.get("dropped"):
        line += f" [dim]· {counts['dropped']:,} overlapping patch(es) left out[/dim]"
    return line


def _print_result(result: GenerateResult) -> None:
    manifest = result.manifest
    table = Table.grid(padding=(0, 2))
    table.add_column(style="bold green", no_wrap=True)
    table.add_column()
    total = len(manifest.patches)
    patches = f"{total:,}"
    if result.new_patches != total:
        patches += f" [dim]({result.new_patches:,} new this run)[/dim]"
    table.add_row("Patches", patches)
    source = manifest.source
    table.add_row("Source", f"{source.source_type} · {source.product_id or 'unknown product'}")
    shape = "×".join(str(dim) for dim in source.patch_shape) if source.patch_shape else "?"
    table.add_row("Shape", f"{shape} {source.dtype or ''}".strip())
    if result.tiles_requested:
        table.add_row(
            "Tiles", f"{result.tiles_requested:,} fetched · {result.tiles_failed:,} failed"
        )
    if result.split_counts is not None:
        table.add_row("Splits", _split_line(result.split_counts))
    minutes, seconds = divmod(int(result.seconds), 60)
    table.add_row("Time", f"{minutes}m {seconds:02d}s" if minutes else f"{seconds}s")
    written = [f"{folder}/" for folder in patch_folders(manifest)] or ["Images/"]
    written.append("manifest.json")
    if result.split_counts is not None:
        written.append("splits/")
    written.extend(
        name
        for name in ("annotations/", "labels/", "dataset.yaml")
        if manifest.task in ("detection", "instance") and (result.staging_dir / name).exists()
    )
    if manifest.task == "classification":
        written.extend(("labels.csv", "labels.json", "classes.txt"))
    table.add_row("Files", f"{result.staging_dir}/ ({', '.join(written)})")
    _console.print(
        Panel(table, title="[bold]Dataset ready[/bold]", title_align="left", border_style="green")
    )
    classes = _class_table(manifest)
    if classes is not None:
        _console.print(classes)
    guides = {
        "detection": "tutorials/object-detection/#train-a-detector",
        "instance": "tutorials/instance-segmentation/#train-a-model",
        "classification": "tutorials/classification/#train-a-classifier",
    }
    guide = guides.get(manifest.task, "guides/use-your-dataset/")
    _console.print(
        "\n[bold]Next[/bold]\n"
        f"  • Inspect it:      [cyan]mapcv info {result.staging_dir}[/cyan]\n"
        f"  • Re-split it:     [cyan]mapcv split {result.staging_dir} --strategy spatial[/cyan]\n"
        f"  • Train on it:     {_DOCS_URL}/{guide}"
    )


# ── init: templates and the guided wizard ────────────────────────────────────


class Template(str, Enum):
    """Ready-made starting configs."""

    xyz = "xyz"
    sentinel2 = "sentinel2"
    geotiff = "geotiff"
    detection = "detection"
    instance = "instance"
    classification = "classification"


_HEADER = f"""\
# mapcv config - docs: {_DOCS_URL}/reference/configuration/
# Check the cost first with `mapcv plan <this file>`, then run `mapcv generate <this file>`.
# You are responsible for the imagery provider's license, attribution and rate limits:
# {_PROVIDERS_URL}
"""

_XYZ_TEMPLATE = (
    _HEADER
    + """
region:                      # WGS-84 lon/lat bounding box
  west: 74.30
  south: 31.48
  east: 74.34
  north: 31.52

imagery:
  type: xyz
  zoom: 17                   # ~ 1 m/px here; each zoom level halves the pixel size
  source: esri_satellite     # or url_template: "https://.../{z}/{x}/{y}.png"
  max_connections: 4         # keep requests modest; respect the provider's limits
  policy: lenient            # strict | lenient | ignore - what to do with failed tiles
  max_failed_ratio: 0.05

# labels:                    # omit for an image-only dataset
#   path: buildings.geojson  # .geojson, .kml, .gpkg, .shp or .parquet; any CRS but GeoJSON/KML's lon/lat
#   layer: null              # the table of a .gpkg with several
#   label_field: null        # property holding the class; null = every polygon is class 1
#   classes: null            # optional fixed ids, e.g. {building: 1, road: 2}

sampler:
  patch_size: 256
  stride: 0                  # 0 = patch_size (no overlap); smaller = overlapping patches
  mode: grid                 # grid | random
  edge_strategy: pad         # pad | drop | shift - patches at the raster edge
  max_empty_ratio: 1.0       # skip patches with more empty (black/NoData) pixels than this

writer:
  staging_dir: ./dataset
  image_format: png          # png | jpg | tif (GeoTIFF)
  # mask_format: png        # png | npy | tif - masks as PNG, NumPy or GeoTIFF
  # world_files: false      # .pgw/.jgw next to PNG/JPG patches, for QGIS

split:                       # remove to skip splitting
  strategy: spatial          # spatial (no leakage between splits) | stratified | random
  test_ratio: 0.20
  val_ratio: 0.10
  labeled_ratios: [0.10, 0.20, 0.30]   # semi-supervised labeled subsets of train
  seed: 42
"""
)

_SENTINEL2_TEMPLATE = (
    _HEADER
    + """
# Needs the optional extra: pip install "mapcv[zarr]"   (Python 3.10-3.13)

region:                      # WGS-84 lon/lat bounding box inside the product
  west: 10.00
  south: 45.00
  east: 10.10
  north: 45.10

imagery:
  type: eopf_zarr
  # One Sentinel-2 L2A EOPF Zarr product: a local path, https:// or anonymous s3:// URL.
  # Browse products at https://stac.browser.user.eopf.eodc.eu/
  path: /path/to/S2X_MSIL2A_PRODUCT.zarr
  resolution: 10             # 10 | 20 | 60 metres
  bands: [b04, b03, b02, b08]   # order is kept in the output; omit for all 12 bands

# labels:
#   path: fields.geojson
#   label_field: crop

sampler:
  patch_size: 128
  stride: 0
  mode: grid
  edge_strategy: drop
  max_empty_ratio: 0.2       # NoData counts as empty

writer:
  staging_dir: ./dataset
  image_format: npy          # npy | tif; Sentinel-2 patches are float32, bands-first (bands, h, w)
  # mask_format: png        # png | npy | tif

split:
  strategy: spatial
  test_ratio: 0.20
  val_ratio: 0.10
"""
)

_GEOTIFF_TEMPLATE = (
    """\
# mapcv config - docs: {docs}/reference/configuration/
# Check the cost first with `mapcv plan <this file>`, then run `mapcv generate <this file>`.
# Your own GeoTIFF or Cloud Optimized GeoTIFF: you are responsible for its license.
""".format(docs=_DOCS_URL)
    + """
region:                      # WGS-84 lon/lat bounding box inside the file
  west: 2.30
  south: 48.85
  east: 2.32
  north: 48.87

imagery:
  type: geotiff
  # A local path (relative to this file), https:// URL or anonymous s3:// URL.
  # The file is read as it is: patches use its CRS and pixel grid, nothing is resampled.
  path: /path/to/ortho.tif
  # bands: [1, 2, 3]         # 1-based; default: every band, in file order
  # overview: 0              # 0 = full resolution; 1, 2, ... = reduced-resolution overviews
  # nodata: 0                # overrides the file's NoData value (patches over it are "empty")

# labels:                    # omit for an image-only dataset
#   path: buildings.geojson  # .geojson, .kml, .gpkg, .shp or .parquet: mapcv reprojects it into the imagery's CRS
#   layer: null              # the table of a .gpkg with several
#   label_field: null        # property holding the class; null = every polygon is class 1
#
# labels:                    # or a classified label raster (land cover, a model's output, ...)
#   type: raster             # any CRS and resolution: each pixel takes the label at its centre
#   path: landcover.tif      # local path, https:// or anonymous s3:// URL
#   classes:                 # raster value -> mask ID (0 = background), optionally with a name
#     10: {id: 1, name: tree_cover}
#     50: {id: 2, name: built_up}
#     80: {id: 3, name: water}
#   unmapped: background     # values not listed above: background | ignore

sampler:
  patch_size: 256
  stride: 0
  mode: grid
  edge_strategy: drop
  max_empty_ratio: 0.2       # NoData and the area outside the file count as empty

writer:
  staging_dir: ./dataset
  image_format: png          # png | jpg for 8-bit 1- or 3-band files; npy keeps any bands and dtype

split:
  strategy: spatial
  test_ratio: 0.20
  val_ratio: 0.10
"""
)

_DETECTION_TEMPLATE = (
    _HEADER
    + """
task: detection              # boxes (COCO + YOLO) instead of masks

region:                      # WGS-84 lon/lat bounding box
  west: 4.9375
  south: 52.3725
  east: 4.9515
  north: 52.3780

imagery:
  type: xyz
  zoom: 18
  source: esri_satellite     # or url_template: "https://.../{z}/{x}/{y}.png"
  max_connections: 4         # keep requests modest; respect the provider's limits

labels:
  path: buildings.geojson    # .geojson, .kml, .gpkg, .shp or .parquet; one object per feature
  label_field: null          # property holding the class; null = every feature is class 1

detection:
  min_visible: 0.3           # keep an object in a patch if >= 30% of its area is visible there
  min_box_pixels: 2          # drop boxes narrower or shorter than this (edge slivers)
  formats: [coco, yolo]      # annotations/instances_<split>.json and labels/*.txt + dataset.yaml
  # point_box_size: 16       # GeoJSON points become boxes of this many pixels

sampler:
  patch_size: 256
  stride: 0                  # 0 = patch_size (no overlap)
  mode: grid
  edge_strategy: drop        # pad | drop | shift (pad: boxes stop at the raster edge)

writer:
  staging_dir: ./dataset
  image_format: png          # png | jpg (Ultralytics cannot read npy)

split:                       # dataset.yaml for Ultralytics needs train and val lists
  strategy: spatial
  test_ratio: 0.20
  val_ratio: 0.10
  seed: 42
"""
)

_INSTANCE_TEMPLATE = (
    _HEADER
    + """
task: instance               # one mask per object (COCO RLE) instead of a class mask

region:                      # WGS-84 lon/lat bounding box
  west: 4.9375
  south: 52.3725
  east: 4.9515
  north: 52.3780

imagery:
  type: xyz
  zoom: 18
  source: esri_satellite     # or url_template: "https://.../{z}/{x}/{y}.png"
  max_connections: 4         # keep requests modest; respect the provider's limits

labels:
  path: buildings.geojson    # .geojson or .kml, in lon/lat; one instance per feature
  label_field: null          # property holding the class; null = every feature is class 1

instance:
  min_visible: 0.3           # keep an instance in a patch if >= 30% of its area is visible there
  min_area: 4                # drop masks with fewer pixels than this (edge slivers)
  id_mask: false             # true: also write a 16-bit instance-ID PNG per patch (masks/)

sampler:
  patch_size: 256
  stride: 0                  # 0 = patch_size (no overlap)
  mode: grid
  edge_strategy: drop        # pad | drop | shift (pad: masks stop at the raster edge)

writer:
  staging_dir: ./dataset
  image_format: png          # png | jpg

split:
  strategy: spatial
  test_ratio: 0.20
  val_ratio: 0.10
  seed: 42
"""
)

_CLASSIFICATION_TEMPLATE = (
    _HEADER
    + """
task: classification         # one label (or a set of labels) per patch, as a CSV, instead of masks

region:                      # WGS-84 lon/lat bounding box
  west: 4.9375
  south: 52.3725
  east: 4.9515
  north: 52.3780

imagery:
  type: xyz
  zoom: 18
  source: esri_satellite     # or url_template: "https://.../{z}/{x}/{y}.png"
  max_connections: 4         # keep requests modest; respect the provider's limits

labels:
  path: landuse.geojson      # .geojson, .kml, .gpkg, .shp or .parquet; or type: raster (a label raster)
  label_field: landuse       # property holding the class name

classification:
  mode: single               # single: the class covering most of the patch | multi: every class that qualifies
  min_fraction: 0.0          # share of the patch's valid pixels a class needs (0 = any labeled pixel)
  empty: skip                # skip: drop patches no class qualifies for | background: keep them as "background"

sampler:
  patch_size: 64
  stride: 0                  # 0 = patch_size (no overlap)
  mode: grid
  edge_strategy: drop        # pad | drop | shift

writer:
  staging_dir: ./dataset     # images/, labels.csv, labels.json, classes.txt
  image_format: png          # png | jpg

split:
  strategy: spatial
  test_ratio: 0.20
  val_ratio: 0.10
  seed: 42
"""
)

_TEMPLATES = {
    Template.xyz: _XYZ_TEMPLATE,
    Template.sentinel2: _SENTINEL2_TEMPLATE,
    Template.geotiff: _GEOTIFF_TEMPLATE,
    Template.detection: _DETECTION_TEMPLATE,
    Template.instance: _INSTANCE_TEMPLATE,
    Template.classification: _CLASSIFICATION_TEMPLATE,
}


def _yaml_str(value: str) -> str:
    """Single-quoted YAML scalar: backslashes (Windows paths) stay literal."""
    return "'" + value.replace("'", "''") + "'"


def _ask_layer(path: Path) -> Optional[str]:
    """For a GeoPackage with several layers, ask which one holds the features."""
    try:
        names = vector_layers(path)
    except ValueError:
        return None  # the caller reads the file next and shows why it cannot
    if len(names) < 2:
        return None
    _console.print(f"[dim]{path.name} has {len(names)} layers: {', '.join(names)}[/dim]")
    return Prompt.ask("Layer", choices=names, default=names[0], console=_console)


def _ask_bbox_or_file(
    default_bbox: Optional[Tuple[float, float, float, float]] = None,
) -> Tuple[Tuple[float, float, float, float], Optional[Path], Optional[str]]:
    default = ",".join(f"{value:.6f}" for value in default_bbox) if default_bbox else None
    while True:
        prompt = (
            "Area: a bounding box [dim]west,south,east,north[/dim] or a vector file "
            "[dim](.geojson, .kml, .gpkg, .shp, .parquet)[/dim]"
        )
        if default is not None:
            answer = Prompt.ask(
                prompt + " [dim](Enter = the whole file)[/dim]",
                default=default,
                show_default=False,
                console=_console,
            ).strip()
        else:
            answer = Prompt.ask(prompt, console=_console).strip()
        path = Path(answer).expanduser()
        if path.suffix.lower() in VECTOR_LABEL_SUFFIXES:
            if not path.exists():
                _console.print(f"[red]File not found:[/red] {path}")
                continue
            layer = _ask_layer(path)
            try:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    geometries, _ = load_vector_labels(path, layer=layer)
            except (ValueError, OSError) as exc:
                _console.print(f"[red]Cannot read that file:[/red] {escape(str(exc))}")
                continue
            if not geometries:
                _console.print("[red]No polygons in that file.[/red]")
                continue
            west = min(geom.bounds[0] for geom, _ in geometries)
            south = min(geom.bounds[1] for geom, _ in geometries)
            east = max(geom.bounds[2] for geom, _ in geometries)
            north = max(geom.bounds[3] for geom, _ in geometries)
            _console.print(
                f"[dim]Using the extent of {len(geometries):,} polygon(s): "
                f"{west:.5f}, {south:.5f} → {east:.5f}, {north:.5f}[/dim]"
            )
            return (west, south, east, north), path, layer
        try:
            west, south, east, north = (float(part) for part in answer.split(","))
        except ValueError:
            _console.print("[red]Enter four numbers like[/red] 74.30,31.48,74.34,31.52")
            continue
        if not (-180 <= west < east <= 180 and -90 <= south < north <= 90):
            _console.print("[red]Expected west < east and south < north, in lon/lat degrees.[/red]")
            continue
        return (west, south, east, north), None, None


def label_fields(
    path: Path, max_values: int = 5, layer: Optional[str] = None
) -> Dict[str, List[str]]:
    """Return candidate label fields of a vector label file with example values."""
    values: Dict[str, Counter[str]] = {}
    suffix = path.suffix.lower()
    if suffix == ".kml":
        data = path.read_bytes()
        text = data.decode("utf-8", errors="replace")
        names = set(re.findall(r'<(?:\w+:)?(?:Simple)?Data\s+name="([^"]+)"', text))
        for name in sorted(names):
            polygons, _ = parse_kml_rs(data, name)
            values[name] = Counter(label for _, label in polygons if label)
    elif suffix in (".geojson", ".json"):
        obj: Any = json.loads(path.read_bytes().decode("utf-8"))
        features = obj.get("features", [obj]) if isinstance(obj, dict) else []
        for feature in features:
            for key, value in (feature.get("properties") or {}).items():
                if value is not None and not isinstance(value, (dict, list)):
                    values.setdefault(key, Counter())[str(value)] += 1
    else:
        for name, column in vector_attributes(path, layer).items():
            counter = Counter(
                label
                for label in (
                    _normalize_label(value)
                    for value in column
                    if not isinstance(value, (bytes, dict, list))
                )
                if label is not None
            )
            values[name] = counter
    # Fields with more distinct values than a mask can hold (ids, names) can't be classes.
    return {
        name: [value for value, _ in counter.most_common(max_values)]
        for name, counter in values.items()
        if counter and len(counter) <= MAX_CLASS_ID
    }


def _ask_label_field(path: Path, layer: Optional[str] = None) -> Optional[str]:
    try:
        fields = label_fields(path, layer=layer)
    except (ValueError, OSError) as exc:
        _console.print(f"[yellow]Cannot read its attributes:[/yellow] {escape(str(exc))}")
        return None
    if not fields:
        _console.print("[dim]No attribute fields found: every polygon will be class 1.[/dim]")
        return None
    table = Table(box=None, padding=(0, 2), show_edge=False)
    table.add_column("field", style="bold")
    table.add_column("example values")
    for name, examples in fields.items():
        table.add_row(name, ", ".join(examples))
    _console.print(table)
    answer = Prompt.ask(
        "Which field holds the class? [dim](blank = every polygon is class 1)[/dim]",
        choices=[*fields, ""],
        default="",
        show_choices=False,
        console=_console,
    )
    return answer or None


class _GeoTiffAnswer:
    """What the wizard learned about the file the user pointed it at."""

    def __init__(
        self,
        imagery_lines: List[str],
        image_format: str,
        extent: Optional[Tuple[float, float, float, float]],
    ) -> None:
        self.imagery_lines = imagery_lines
        self.image_format = image_format
        self.extent = extent


def _geotiff_wgs84_extent(tif: Any) -> Optional[Tuple[float, float, float, float]]:
    """A lon/lat box (``west, south, east, north``) that lies inside the file, or ``None``.

    The file's own bounding box, projected to lon/lat, reaches a little outside the file
    when projected back (a lon/lat box is not a rectangle in UTM), so the box is shrunk
    in small steps until the whole of it maps into the file.
    """
    import math

    from pyproj import Transformer

    from mapcv.config import RegionConfig
    from mapcv.imagery import region_bounds_in_crs, region_pixel_window

    info = tif.info
    if info.epsg is None or info.transform is None:
        return None
    a, b, c, d, e, f = info.transform
    xs = [c + a * col + b * row for col in (0, info.width) for row in (0, info.height)]
    ys = [f + d * col + e * row for col in (0, info.width) for row in (0, info.height)]
    to_wgs84 = Transformer.from_crs(f"EPSG:{info.epsg}", "EPSG:4326", always_xy=True)
    west, south, east, north = to_wgs84.transform_bounds(
        min(xs), min(ys), max(xs), max(ys), densify_pts=21
    )
    width, height = east - west, north - south
    for step in range(100):
        shrink = step * 0.002
        edges = (
            math.ceil((west + width * shrink) * 1e6) / 1e6,
            math.ceil((south + height * shrink) * 1e6) / 1e6,
            math.floor((east - width * shrink) * 1e6) / 1e6,
            math.floor((north - height * shrink) * 1e6) / 1e6,
        )
        if edges[0] >= edges[2] or edges[1] >= edges[3]:
            break
        region = RegionConfig(west=edges[0], south=edges[1], east=edges[2], north=edges[3])
        bounds = region_bounds_in_crs(region, f"EPSG:{info.epsg}")
        if not region_pixel_window(bounds, info.transform, info.height, info.width)[4]:
            return edges
    return None


def _ask_geotiff() -> _GeoTiffAnswer:
    """Ask for a GeoTIFF/COG path or URL, show what is in it, and pick output defaults."""
    from mapcv.geotiff import GeoTiff
    from mapcv.imagery import geotiff_location

    _console.print("\n[dim]A local file, an https:// URL or a public s3:// object.[/dim]")
    while True:
        answer = Prompt.ask("GeoTIFF / COG path or URL", console=_console).strip().strip("'\"")
        try:
            tif = GeoTiff(
                geotiff_location(str(Path(answer).expanduser()) if "://" not in answer else answer)
            )
        except Exception as exc:  # noqa: BLE001 - any failure to open is shown and asked again
            _console.print(f"[red]Cannot read that file:[/red] {exc}")
            continue
        break
    info = tif.info
    table = Table.grid(padding=(0, 2))
    table.add_column(style="bold cyan", no_wrap=True)
    table.add_column()
    crs = f"EPSG:{info.epsg}" if info.epsg is not None else f"none usable ({info.crs_error})"
    table.add_row("CRS", crs)
    table.add_row(
        "Size", f"{info.width:,} × {info.height:,} px · {info.count} band(s) · {info.dtype}"
    )
    if info.transform is not None:
        table.add_row("Pixel", f"{abs(info.transform[0]):g} × {abs(info.transform[4]):g} CRS units")
    table.add_row("NoData", "none" if info.nodata is None else f"{info.nodata:g}")
    table.add_row("Overviews", str(len(info.overviews)))
    _console.print(table)
    if info.epsg is None:
        _console.print(
            "[yellow]mapcv needs a CRS given by an EPSG code; the config will be written, but "
            "generating from this file will fail until it is re-projected or re-tagged.[/yellow]"
        )
    path_text = answer if "://" in answer else str(Path(answer).expanduser())
    lines = ["  type: geotiff", f"  path: {_yaml_str(path_text)}"]
    image_format = "png" if info.dtype == np.uint8 and info.count in (1, 3) else "npy"
    if info.dtype == np.uint8 and info.count > 3:
        while True:
            picked = Prompt.ask(
                f"Bands to use [dim](1-based; e.g. 1,2,3 for RGB PNG; blank = all {info.count} "
                "as NPY)[/dim]",
                default="",
                show_default=False,
                console=_console,
            ).strip()
            try:
                numbers = [int(part) for part in picked.split(",")] if picked else []
            except ValueError:
                numbers = [0]
            if all(1 <= number <= info.count for number in numbers) and len(set(numbers)) == len(
                numbers
            ):
                break
            _console.print(f"[red]Enter distinct band numbers from 1 to {info.count}.[/red]")
        if numbers:
            lines.append(f"  bands: {numbers}")
            image_format = "png" if len(numbers) in (1, 3) else "npy"
    if image_format == "npy":
        _console.print("[dim]Patches will be written as NPY (bands, height, width).[/dim]")
    return _GeoTiffAnswer(lines, image_format, _geotiff_wgs84_extent(tif))


_RASTER_SUFFIXES = (".tif", ".tiff")
# The wizard lists a label raster's values from at most this many pixels.
_WIZARD_SAMPLE_PIXELS = 4_000_000


def _sample_label_values(
    tif: Any, bbox: Tuple[float, float, float, float]
) -> Tuple[Dict[int, int], bool]:
    """Pixel count per value of the label raster under ``bbox`` (lon/lat), and whether
    the values come from a reduced sample (an overview or a crop)."""
    from mapcv.config import RegionConfig
    from mapcv.imagery import region_bounds_in_crs, region_pixel_window

    info = tif.info
    region = RegionConfig(west=bbox[0], south=bbox[1], east=bbox[2], north=bbox[3])
    bounds = region_bounds_in_crs(region, f"EPSG:{info.epsg}")
    level, sampled = 0, False
    sizes = [(info.height, info.width), *info.overviews]
    while True:
        height, width = sizes[level]
        transform = info.overview_transform(level)
        row0, row1, col0, col1, _ = region_pixel_window(bounds, transform, height, width)
        pixels = (row1 - row0) * (col1 - col0)
        if pixels <= _WIZARD_SAMPLE_PIXELS or level == len(sizes) - 1:
            break
        level, sampled = level + 1, True
    if row0 >= row1 or col0 >= col1:
        return {}, sampled
    side = int(_WIZARD_SAMPLE_PIXELS**0.5)
    if pixels > _WIZARD_SAMPLE_PIXELS:  # no small enough overview: the centre of the area
        mid_row, mid_col = (row0 + row1) // 2, (col0 + col1) // 2
        row0, row1 = max(row0, mid_row - side // 2), min(row1, mid_row + side // 2)
        col0, col1 = max(col0, mid_col - side // 2), min(col1, mid_col + side // 2)
        sampled = True
    data, _ = tif.read_window(row0, row1, col0, col1, bands=[0], overview=level)
    values, counts = np.unique(data[..., 0], return_counts=True)
    return {int(value): int(count) for value, count in zip(values, counts)}, sampled


def _ask_label_raster(path_text: str, bbox: Tuple[float, float, float, float]) -> List[str]:
    """Describe a label raster, list its values and write ``labels`` lines that map them."""
    from mapcv.geotiff import GeoTiff
    from mapcv.imagery import geotiff_location
    from mapcv.targets.raster_labels import integer_nodata

    lines = ["labels:", "  type: raster", f"  path: {_yaml_str(path_text)}"]
    placeholder = lines + [
        "  classes:                   # raster value: {id: mask ID, name: class name}",
        "    1: {id: 1, name: class_1}",
    ]
    _console.print("[dim]A label raster: each pixel's value is its class.[/dim]")
    try:
        tif = GeoTiff(geotiff_location(path_text))
    except Exception as exc:  # noqa: BLE001 - shown, and the config is written for editing
        _console.print(f"[yellow]Cannot read that file:[/yellow] {exc}. Edit labels.classes.")
        return placeholder
    info = tif.info
    table = Table.grid(padding=(0, 2))
    table.add_column(style="bold cyan", no_wrap=True)
    table.add_column()
    crs = f"EPSG:{info.epsg}" if info.epsg is not None else f"none usable ({info.crs_error})"
    table.add_row("CRS", crs)
    table.add_row(
        "Size", f"{info.width:,} × {info.height:,} px · {info.count} band(s) · {info.dtype}"
    )
    if info.transform is not None:
        table.add_row("Pixel", f"{abs(info.transform[0]):g} × {abs(info.transform[4]):g} CRS units")
    table.add_row("NoData", "none" if info.nodata is None else f"{info.nodata:g}")
    _console.print(table)
    if info.epsg is None or info.transform is None or info.dtype.kind not in "iu":
        _console.print(
            "[yellow]mapcv reads integer label rasters with an EPSG CRS; the config will be "
            "written, but generating will fail until the file is fixed.[/yellow]"
        )
        return placeholder
    try:
        counts, sampled = _sample_label_values(tif, bbox)
    except Exception as exc:  # noqa: BLE001 - shown, and the config is written for editing
        _console.print(f"[yellow]Cannot read its values:[/yellow] {exc}. Edit labels.classes.")
        return placeholder
    nodata = integer_nodata(info.nodata, info.dtype)
    if nodata is not None:
        counts.pop(nodata, None)
    if not counts:
        _console.print("[yellow]No label values under the area.[/yellow] Edit labels.classes.")
        return placeholder
    total = sum(counts.values())
    values = sorted(counts)
    shown = Table(box=None, padding=(0, 2), show_edge=False)
    shown.add_column("value", justify="right")
    shown.add_column("share", justify="right")
    for value in values[:20]:
        shown.add_row(str(value), f"{counts[value] / total:.1%}")
    _console.print(shown)
    if len(values) > 20:
        _console.print(f"[dim]… and {len(values) - 20} more value(s).[/dim]")
    if sampled:
        _console.print("[dim]Values from a sample of the area (an overview or its centre).[/dim]")
    classes = [value for value in values if value != 0]
    if len(classes) > MAX_CLASS_ID - 1:
        _console.print(
            f"[yellow]{len(classes)} distinct values: too many for class IDs.[/yellow] "
            "Edit labels.classes."
        )
        return placeholder
    identity = all(1 <= value < MAX_CLASS_ID for value in classes)
    lines.append("  classes:                   # raster value: {id: mask ID, name: class name}")
    if 0 in counts:
        lines.append("    0: 0                     # background")
    for number, value in enumerate(classes, start=1):
        class_id = value if identity else number
        lines.append(f"    {value}: {{id: {class_id}, name: value_{value}}}")
    lines.append("  unmapped: background       # values not listed: background | ignore")
    _console.print(
        "[dim]Each value becomes a class; rename them (and merge values by giving them "
        "the same ID) in the config.[/dim]"
    )
    return lines


def _wizard() -> str:
    _console.print(
        Panel(
            "Answer a few questions to get a working config. Press Enter to accept a "
            "[bold]default[/bold].",
            title="[bold]mapcv init[/bold]",
            title_align="left",
            border_style="cyan",
        )
    )

    _console.print("\n[bold cyan]1/4 Imagery[/bold cyan]")
    _console.print(
        "  [bold]esri[/bold]       Esri World Imagery — sub-metre RGB (check Esri's terms)"
    )
    _console.print("  [bold]sentinel2[/bold]  Sentinel-2 L2A — open 10 m multispectral (EOPF Zarr)")
    _console.print("  [bold]custom[/bold]     your own XYZ tile URL")
    _console.print("  [bold]geotiff[/bold]    your own GeoTIFF / COG file or URL")
    kind = Prompt.ask(
        "Imagery",
        choices=["esri", "sentinel2", "custom", "geotiff"],
        default="esri",
        console=_console,
    )

    geotiff: Optional[_GeoTiffAnswer] = _ask_geotiff() if kind == "geotiff" else None

    _console.print("\n[bold cyan]2/4 Area[/bold cyan]")
    (west, south, east, north), area_file, area_layer = _ask_bbox_or_file(
        geotiff.extent if geotiff is not None else None
    )
    latitude = (south + north) / 2

    imagery_lines: List[str]
    if geotiff is not None:
        imagery_lines = geotiff.imagery_lines
        patch_default, image_format, edge = 256, geotiff.image_format, "drop"
    elif kind == "sentinel2":
        product = Prompt.ask(
            "Product path or URL [dim](local .zarr, https:// or s3://)[/dim]", console=_console
        )
        resolution = Prompt.ask(
            "Resolution in metres", choices=["10", "20", "60"], default="10", console=_console
        )
        bands = Prompt.ask("Bands", choices=["rgbn", "all"], default="rgbn", console=_console)
        imagery_lines = [
            "  type: eopf_zarr",
            f"  path: {_yaml_str(product)}",
            f"  resolution: {resolution}",
        ]
        if bands == "rgbn":
            imagery_lines.append("  bands: [b04, b03, b02, b08]   # red, green, blue, NIR")
        patch_default, image_format, edge = 128, "npy", "drop"
    else:
        table = Table(box=None, padding=(0, 2), show_edge=False)
        table.add_column("zoom", justify="right")
        table.add_column("pixel size")
        for zoom in range(15, 20):
            table.add_row(str(zoom), f"{ground_resolution_m(zoom, latitude):.2f} m")
        _console.print(table)
        zoom = IntPrompt.ask("Zoom", default=17, console=_console)
        if kind == "custom":
            template = Prompt.ask(
                "Tile URL with {z}, {x}, {y} [dim](keep API keys out of shared files)[/dim]",
                console=_console,
            )
            source_line = f"  url_template: {_yaml_str(template)}"
        else:
            source_line = "  source: esri_satellite"
        imagery_lines = [
            "  type: xyz",
            f"  zoom: {zoom}",
            source_line,
            "  max_connections: 4         # keep requests modest",
        ]
        patch_default, image_format, edge = 256, "png", "pad"

    _console.print("\n[bold cyan]3/4 Labels[/bold cyan]")
    labels_path: Optional[Path] = None
    labels_layer: Optional[str] = None
    raster_lines: List[str] = []
    if area_file is not None and Confirm.ask(
        f"Use {area_file.name} as the labels too?", default=True, console=_console
    ):
        labels_path = area_file
        labels_layer = area_layer
    elif area_file is None:
        answer = Prompt.ask(
            "Label file [dim](.geojson, .kml, .gpkg, .shp, .parquet, or a .tif label raster; "
            "blank for an image-only dataset)[/dim]",
            default="",
            show_default=False,
            console=_console,
        ).strip()
        if answer.lower().endswith(_RASTER_SUFFIXES):
            path_text = answer if "://" in answer else str(Path(answer).expanduser())
            raster_lines = _ask_label_raster(path_text, (west, south, east, north))
        elif answer:
            labels_path = Path(answer).expanduser()
            if labels_path.exists():
                labels_layer = _ask_layer(labels_path)
    # A label raster makes masks, so the task question is only asked for vector labels.
    label_lines: List[str] = raster_lines
    task_lines: List[str] = []
    detection_lines: List[str] = []
    if labels_path is not None:
        field = _ask_label_field(labels_path, labels_layer) if labels_path.exists() else None
        label_lines = ["labels:", f"  path: {_yaml_str(str(labels_path))}"]
        if labels_layer is not None:
            label_lines.append(f"  layer: {_yaml_str(labels_layer)}")
        label_lines.append(f"  label_field: {field}" if field else "  label_field: null")
        _console.print(
            "  [bold]segmentation[/bold]  a class mask per patch\n"
            "  [bold]detection[/bold]     a box per object (COCO and YOLO)\n"
            "  [bold]instance[/bold]      a mask per object (COCO RLE, optional instance-ID PNG)\n"
            "  [bold]classification[/bold]  a label (or set of labels) per patch, as a CSV"
        )
        task = Prompt.ask(
            "Task",
            choices=["segmentation", "detection", "instance", "classification"],
            default="segmentation",
            console=_console,
        )
        if task == "detection":
            formats = Prompt.ask(
                "Box formats", choices=["both", "coco", "yolo"], default="both", console=_console
            )
            chosen = "[coco, yolo]" if formats == "both" else f"[{formats}]"
            task_lines = ["task: detection", ""]
            detection_lines = [
                "detection:",
                "  min_visible: 0.3           # share of an object's area a patch must show",
                "  min_box_pixels: 2          # drop thinner boxes (slivers at patch edges)",
                f"  formats: {chosen}",
            ]
        elif task == "instance":
            id_mask = Confirm.ask(
                "Also write a 16-bit instance-ID PNG per patch?", default=False, console=_console
            )
            task_lines = ["task: instance", ""]
            detection_lines = [
                "instance:",
                "  min_visible: 0.3           # share of an instance's area a patch must show",
                "  min_area: 4                # drop masks with fewer pixels (edge slivers)",
                f"  id_mask: {'true' if id_mask else 'false'}",
            ]
        elif task == "classification":
            mode = Prompt.ask(
                "One label per patch, or every class present?",
                choices=["single", "multi"],
                default="single",
                console=_console,
            )
            task_lines = ["task: classification", ""]
            detection_lines = [
                "classification:",
                f"  mode: {mode}",
                "  min_fraction: 0.0          # share of the patch a class needs (0 = any pixel)",
                "  empty: skip                # skip | background (keep unlabeled patches)",
            ]

    _console.print("\n[bold cyan]4/4 Patches and output[/bold cyan]")
    patch_size = IntPrompt.ask("Patch size in pixels", default=patch_default, console=_console)
    staging = Prompt.ask("Output folder", default="./dataset", console=_console)
    do_split = Confirm.ask(
        "Split into train/val/test (spatial, no leakage)?", default=True, console=_console
    )

    lines = [
        _HEADER.rstrip(),
        "",
        *task_lines,
        "region:",
        f"  west: {west:.6f}",
        f"  south: {south:.6f}",
        f"  east: {east:.6f}",
        f"  north: {north:.6f}",
        "",
        "imagery:",
        *imagery_lines,
        "",
        *label_lines,
        *([""] if label_lines else []),
        *detection_lines,
        *([""] if detection_lines else []),
        "sampler:",
        f"  patch_size: {patch_size}",
        "  stride: 0                  # 0 = no overlap",
        "  mode: grid",
        f"  edge_strategy: {edge}",
        "",
        "writer:",
        f"  staging_dir: {_yaml_str(staging)}",
        f"  image_format: {image_format}",
    ]
    if do_split:
        lines += [
            "",
            "split:",
            "  strategy: spatial",
            "  test_ratio: 0.20",
            "  val_ratio: 0.10",
        ]
    return "\n".join(lines) + "\n"


@app.command(
    rich_help_panel="1. Build a dataset",
    epilog=(
        "Examples:\n\n"
        "  [cyan]mapcv init[/cyan]                         guided, writes mapcv.yaml\n\n"
        "  [cyan]mapcv init --template xyz --stdout[/cyan]   print a template\n\n"
        "  [cyan]mapcv init my.yaml --template sentinel2[/cyan]   a ready-made example\n\n"
        "  [cyan]mapcv init boxes.yaml --template detection[/cyan]   boxes for COCO and YOLO\n\n"
        "  [cyan]mapcv init masks.yaml --template instance[/cyan]   a mask per object (COCO RLE)\n\n"
        "  [cyan]mapcv init tiles.yaml --template classification[/cyan]   a label per patch (CSV)"
    ),
)
def init(
    output: Path = typer.Argument(
        Path("mapcv.yaml"), metavar="OUTPUT", help="Where to write the config."
    ),
    template: Optional[Template] = typer.Option(
        None, "--template", "-t", help="Write a ready-made example instead of asking."
    ),
    interactive: Optional[bool] = typer.Option(
        None,
        "--interactive/--no-interactive",
        help="Ask questions (default: when run in a terminal without --template).",
    ),
    force: bool = typer.Option(False, "--force", "-f", help="Overwrite an existing file."),
    stdout: bool = typer.Option(False, "--stdout", help="Print the template instead of writing."),
) -> None:
    """Create a config: answer a few questions, or start from a template."""
    guided = (
        interactive
        if interactive is not None
        else template is None and sys.stdin.isatty() and sys.stdout.isatty()
    )
    if stdout:
        typer.echo(_TEMPLATES[template or Template.xyz], nl=False)
        return
    target = output
    if target.exists() and not force and not guided:
        _console.print(f"[red]{target} already exists.[/red] Use [bold]--force[/bold].")
        raise typer.Exit(code=1)
    text = _wizard() if guided else _TEMPLATES[template or Template.xyz]
    if target.exists() and not force:
        if not (
            guided
            and Confirm.ask(f"{target} exists. Overwrite it?", default=False, console=_console)
        ):
            _console.print(f"[red]{target} already exists.[/red] Use [bold]--force[/bold].")
            raise typer.Exit(code=1)
    target.write_text(text, encoding="utf-8")
    try:
        MapcvConfig.from_yaml(target)
    except ValidationError as exc:
        _console.print(f"[yellow]Wrote {target}, but it needs edits:[/yellow]")
        for line in _format_validation_error(exc):
            _console.print(line)
        return
    _console.print(f"\n[green]✓[/green] Wrote [bold]{target}[/bold]")
    _console.print(
        "\n[bold]Next[/bold]\n"
        f"  1. See what it will cost:  [cyan]mapcv plan {target}[/cyan]\n"
        f"  2. Build the dataset:      [cyan]mapcv generate {target}[/cyan]\n"
        f"[dim]Imagery terms: {_PROVIDERS_URL}[/dim]"
    )


# ── Commands ─────────────────────────────────────────────────────────────────


@app.command(
    rich_help_panel="1. Build a dataset",
    epilog="Example: [cyan]mapcv plan mapcv.yaml[/cyan]",
)
def plan(
    config_path: Path = typer.Argument(..., metavar="CONFIG_PATH", help="Path to the YAML config."),
) -> None:
    """Estimate tiles, patches, disk and memory for a config [bold]without downloading[/bold]."""
    config = _load_config(config_path)
    try:
        estimate = make_plan(config)
    except (ValueError, RuntimeError, OSError) as exc:
        _console.print(f"[red]Cannot plan this config:[/red] {escape(str(exc))}")
        raise typer.Exit(code=1)
    _print_plan(config_path, config, estimate)
    _console.print(f"\nLooks right? Build it with [cyan]mapcv generate {config_path}[/cyan]")


@app.command(
    rich_help_panel="1. Build a dataset",
    epilog=(
        "Examples:\n\n"
        "  [cyan]mapcv generate mapcv.yaml[/cyan]\n\n"
        "  [cyan]mapcv generate mapcv.yaml --yes[/cyan]   (no prompt for large jobs, for CI)"
    ),
)
def generate(
    config_path: Path = typer.Argument(..., metavar="CONFIG_PATH", help="Path to the YAML config."),
    yes: bool = typer.Option(False, "--yes", "-y", help="Don't ask before large downloads."),
    dry_run: bool = typer.Option(
        False, "--dry-run", help="Only show the plan, like [bold]mapcv plan[/bold]."
    ),
) -> None:
    """Download imagery, rasterize labels and write patches, masks, manifest and splits.

    Re-running the same command resumes an interrupted run.
    """
    config = _load_config(config_path)
    try:
        estimate = make_plan(config)
    except (ValueError, RuntimeError, OSError) as exc:
        _console.print(f"[red]Cannot plan this config:[/red] {escape(str(exc))}")
        raise typer.Exit(code=1)
    _print_plan(config_path, config, estimate)
    if dry_run:
        return
    if estimate.is_large and not yes:
        if not sys.stdin.isatty():
            _console.print(
                "[red]This is a large job.[/red] Re-run with [bold]--yes[/bold] to confirm."
            )
            raise typer.Exit(code=2)
        if not Confirm.ask("This is a large job. Start it?", default=False, console=_console):
            raise typer.Exit(code=1)
    shown = set(estimate.warnings)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        try:
            result = run_generate(config)
        except ManifestMismatchError as exc:
            _show_warnings(caught, shown)
            _console.print(f"[red]Cannot resume:[/red] {exc}")
            raise typer.Exit(code=1)
        except KeyboardInterrupt:
            _console.print(
                "\n[yellow]Interrupted.[/yellow] Finished chunks are saved; run the same "
                "command again to resume."
            )
            raise typer.Exit(code=130)
        except Exception as exc:  # noqa: BLE001 - any failure gets the same resume advice
            _show_warnings(caught, shown)
            detail = escape(str(exc) or type(exc).__name__)
            _console.print(f"[red]Generation failed:[/red] {detail}")
            _console.print(
                "[dim]Fix the cause and run the same command again: finished chunks are "
                "kept and the run resumes.[/dim]"
            )
            raise typer.Exit(code=1)
    _show_warnings(caught, shown)
    if result is not None:
        _print_result(result)


@app.command(rich_help_panel="2. Use a dataset", epilog="Example: [cyan]mapcv info dataset/[/cyan]")
def info(
    staging_dir: Path = typer.Argument(
        ..., metavar="STAGING_DIR", help="Dataset directory containing manifest.json."
    ),
) -> None:
    """Summarize a generated dataset: source, shapes, class balance and splits."""
    manifest_path = staging_dir / "manifest.json"
    if not manifest_path.exists():
        _console.print(f"[red]No manifest found at[/red] {manifest_path}")
        raise typer.Exit(code=1)
    try:
        manifest = Manifest.load(manifest_path)
    except ManifestMismatchError as exc:
        _console.print(f"[red]{exc}[/red]")
        raise typer.Exit(code=1)
    table = Table.grid(padding=(0, 2))
    table.add_column(style="bold cyan", no_wrap=True)
    table.add_column()
    target = manifest.target
    task = manifest.task if target is not None else f"{manifest.task} · image only (no labels)"
    table.add_row("Task", task)
    source = manifest.source
    table.add_row("Source", f"{source.source_type} · {source.product_id or 'unknown product'}")
    if source.bands:
        table.add_row("Bands", ", ".join(source.bands))
    shape = "×".join(str(dim) for dim in source.patch_shape) or "?"
    table.add_row("Patches", f"{len(manifest.patches):,} · {shape} {source.dtype or ''}".strip())
    if source.crs:
        table.add_row("CRS", source.crs)
    if target is not None and target.ignore_index is not None:
        without = "imagery or label" if _raster_labels(manifest) else "imagery"
        if manifest.task == "classification":  # no masks: the value only decides what counts
            table.add_row("Ignore", f"pixels without {without} do not count towards coverage")
        else:
            table.add_row(
                "Ignore", f"mask value {target.ignore_index} marks pixels without {without}"
            )
    padded = sum(1 for entry in manifest.patches if entry["padded"])
    if padded:
        table.add_row("Padded", f"{padded:,} patch(es) touch the raster edge")
    splits_dir = staging_dir / "splits"
    if splits_dir.is_dir():
        counts = {}
        for name in ("train", "val", "test"):
            path = splits_dir / f"{name}.txt"
            text = path.read_text().strip() if path.exists() else ""
            counts[name] = len(text.splitlines()) if text else 0
        table.add_row("Splits", _split_line(counts))
    version = f"version {manifest.loaded_version}"
    if manifest.upgraded_from is not None:
        version += f" (mapcv 0.{manifest.upgraded_from}; read as version {manifest.version})"
    table.add_row("Manifest", version)
    _console.print(
        Panel(table, title=f"[bold]{staging_dir}[/bold]", title_align="left", border_style="cyan")
    )
    classes = _class_table(manifest)
    if classes is not None:
        _console.print(classes)


@app.command(
    rich_help_panel="2. Use a dataset",
    epilog=(
        "Examples:\n\n"
        "  [cyan]mapcv split dataset/[/cyan]\n\n"
        "  [cyan]mapcv split dataset/ --strategy spatial --block-size 2048 --test-ratio 0.15[/cyan]"
    ),
)
def split(
    staging_dir: Path = typer.Argument(
        ..., metavar="STAGING_DIR", help="Dataset directory containing manifest.json."
    ),
    test_ratio: float = typer.Option(0.20, help="Fraction of patches held out for testing."),
    val_ratio: float = typer.Option(0.10, help="Fraction of the remaining patches for validation."),
    labeled_ratios: Optional[List[float]] = typer.Option(
        None,
        help="Labeled fractions of train for semi-supervised lists (repeatable). "
        "Default: 0.1 0.2 0.3.",
    ),
    seed: int = typer.Option(42, help="Random seed."),
    strategy: str = typer.Option(
        "spatial", help="spatial (leakage-safe blocks) | stratified | random."
    ),
    block_size: Optional[int] = typer.Option(
        None, help="Spatial block size in pixels (default: 4 × patch size)."
    ),
    sample_limit: Optional[int] = typer.Option(None, help="Use at most this many patches."),
) -> None:
    """Re-split an existing dataset from its manifest; no images are read."""
    if not staging_dir.is_dir():
        _console.print(f"[red]Not a directory:[/red] {staging_dir}")
        raise typer.Exit(code=1)
    ratios = labeled_ratios if labeled_ratios is not None else [0.10, 0.20, 0.30]
    try:
        cfg = SplitterConfig(
            test_ratio=test_ratio,
            val_ratio=val_ratio,
            labeled_ratios=ratios,
            seed=seed,
            strategy=cast(Literal["spatial", "stratified", "random"], strategy),
            block_size=block_size,
            sample_limit=sample_limit,
        )
    except ValidationError as exc:
        _console.print("[red]Config error:[/red]")
        for line in _format_validation_error(exc):
            _console.print(line)
        raise typer.Exit(code=1)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        try:
            counts = run_split(staging_dir, cfg)
        except (FileNotFoundError, ManifestMismatchError) as exc:
            _console.print(f"[red]{exc}[/red]")
            raise typer.Exit(code=1)
    _show_warnings(caught, set())
    _console.print(
        f"[green]✓[/green] Splits written to [bold]{staging_dir / 'splits'}[/bold]: "
        f"{_split_line(counts)}"
    )


@app.command(
    rich_help_panel="3. Utilities", epilog="Example: [cyan]mapcv validate mapcv.yaml[/cyan]"
)
def validate(
    config_path: Path = typer.Argument(..., metavar="CONFIG_PATH", help="Path to the YAML config."),
) -> None:
    """Check a config without reading labels or imagery (use [bold]plan[/bold] for estimates)."""
    config = _load_config(config_path)
    _console.print(f"[green]✓[/green] {config_path} is a valid config.")
    _console.print(_settings_table(config))
    labels = config.labels
    if isinstance(labels, RasterLabelsConfig):
        label_file = eopf_local_path(labels.path)
        if label_file is not None and not label_file.exists():
            _console.print(f"[yellow]Warning:[/yellow] labels.path not found: {label_file}")
    elif labels is not None and not labels.path.exists():
        _console.print(f"[yellow]Warning:[/yellow] labels.path not found: {labels.path}")
    if isinstance(config.imagery, GeoTiffImageryConfig):
        local = eopf_local_path(config.imagery.path)
        if local is not None and not local.exists():
            _console.print(f"[yellow]Warning:[/yellow] imagery.path not found: {local}")


@app.command(
    "mcp",
    rich_help_panel="3. Utilities",
    epilog=(
        "Examples:\n\n"
        "  [cyan]mapcv mcp[/cyan]                         read-only, in this folder\n\n"
        "  [cyan]mapcv mcp --root ~/work --allow-write[/cyan]   may write datasets under ~/work"
    ),
)
def mcp_server(
    root: Path = typer.Option(
        Path("."),
        "--root",
        help="The only folder the server may read or write (default: the current folder).",
    ),
    allow_write: bool = typer.Option(
        False,
        "--allow-write",
        envvar="MAPCV_MCP_ALLOW_WRITE",
        help="Also offer the tools that write: write_config, generate and split.",
    ),
) -> None:
    """Run an MCP server over stdio so AI agents can build datasets with mapcv.

    Needs [bold]pip install "mapcv\\[mcp]"[/bold]. Without [bold]--allow-write[/bold] the
    server can only read, validate and plan.
    """
    try:
        from mapcv.mcp_server import serve
    except ImportError as exc:
        err = Console(stderr=True)
        err.print("[red]The MCP server needs the optional 'mcp' extra.[/red]")
        err.print('Install it with [bold]pip install "mapcv\\[mcp]"[/bold], then run this again.')
        err.print(f"[dim]{escape(str(exc))} (mapcv needs mcp 2.x)[/dim]")
        raise typer.Exit(code=1)
    try:
        serve(root, allow_write)
    except ValueError as exc:
        _console.print(f"[red]{exc}[/red]")
        raise typer.Exit(code=1)
