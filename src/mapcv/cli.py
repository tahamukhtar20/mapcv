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
from importlib.metadata import PackageNotFoundError
from importlib.metadata import version as package_version
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional, Tuple, cast
from urllib.parse import urlsplit

import typer
from pydantic import ValidationError
from rich.console import Console
from rich.panel import Panel
from rich.prompt import Confirm, IntPrompt, Prompt
from rich.table import Table

from mapcv import pipeline
from mapcv._mapcv_rs import parse_kml_rs
from mapcv.config import EOPFZarrImageryConfig, MapcvConfig
from mapcv.labels import parse_geojson, parse_kml
from mapcv.pipeline import GenerateResult, run_generate, run_split
from mapcv.planning import Plan, ground_resolution_m, human_bytes
from mapcv.planning import plan as make_plan
from mapcv.splitter import SplitterConfig
from mapcv.writer import Manifest

app = typer.Typer(
    name="mapcv",
    help=(
        "Turn a region and polygon labels into a ready-to-train segmentation dataset.\n\n"
        "Start with [bold]mapcv init[/bold], check the cost with [bold]mapcv plan[/bold], "
        "then build with [bold]mapcv generate[/bold]."
    ),
    no_args_is_help=True,
    rich_markup_mode="rich",
    context_settings={"help_option_names": ["-h", "--help"]},
    pretty_exceptions_enable=False,
)
_console = Console()

_DOCS_URL = "https://tahamukhtar20.github.io/mapcv"
_PROVIDERS_URL = "https://github.com/tahamukhtar20/mapcv/blob/main/PROVIDERS.md"


# ── Shared helpers ───────────────────────────────────────────────────────────


def _version() -> str:
    try:
        return package_version("mapcv")
    except PackageNotFoundError:  # pragma: no cover - only when run from a raw checkout
        return "unknown"


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
    """Turn a region and polygon labels into a ready-to-train segmentation dataset."""
    pipeline._console.quiet = quiet


def _format_validation_error(exc: ValidationError) -> List[str]:
    lines = []
    for error in exc.errors():
        location = ".".join(
            str(part)
            for part in error["loc"]
            if not str(part).startswith("function-") and part not in ("xyz", "eopf_zarr")
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
        warnings.simplefilter("always", FutureWarning)
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
        if issubclass(warning.category, FutureWarning):
            _console.print(f"[yellow]Deprecated:[/yellow] {warning.message}")
        else:
            warnings.showwarning(
                warning.message, warning.category, warning.filename, warning.lineno
            )
    return config


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
    source = imagery.source or _redact_url(imagery.url_template or "")
    return f"XYZ {source} · zoom {imagery.zoom}"


def _settings_table(config: MapcvConfig) -> Table:
    table = Table.grid(padding=(0, 2))
    table.add_column(style="bold cyan", no_wrap=True)
    table.add_column()
    region = config.region
    table.add_row(
        "Region", f"{region.west}, {region.south} → {region.east}, {region.north} (W, S → E, N)"
    )
    table.add_row("Imagery", _imagery_label(config))
    if config.labels is None:
        table.add_row("Labels", "none (image-only dataset)")
    else:
        field = config.labels.label_field or "none — every polygon is class 1"
        table.add_row("Labels", f"{config.labels.path} · field: {field}")
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
        detail = f"{labels.polygons:,} polygon(s)"
        detail += f" · classes: {classes}" if classes else " · every polygon is class 1"
        table.add_row("Labels", f"{labels.path} · {detail}")
    table.add_row(
        "Patches",
        f"≈ {estimate.patches:,} × {estimate.patch_size} px "
        f"[dim]({config.sampler.mode}, stride {config.sampler.stride})[/dim]",
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
    if not isinstance(config.imagery, EOPFZarrImageryConfig):
        _console.print(
            "[dim]Imagery terms are your responsibility: check the provider's license, "
            f"attribution and rate limits ({_PROVIDERS_URL}).[/dim]"
        )


def _class_names(manifest: Manifest) -> Dict[str, str]:
    names = {str(cid): name for name, cid in manifest.class_map.items()}
    names.setdefault("0", "background")
    if not manifest.class_map:
        names.setdefault("1", "labelled")
    return names


def _class_table(manifest: Manifest) -> Optional[Table]:
    totals: Counter[str] = Counter()
    for entry in manifest.patches:
        totals.update(entry["per_class_pixel_counts"])
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
    shape = "×".join(str(dim) for dim in manifest.patch_shape) if manifest.patch_shape else "?"
    table.add_row("Shape", f"{shape} {manifest.dtype or ''}".strip())
    if result.tiles_requested:
        table.add_row(
            "Tiles", f"{result.tiles_requested:,} fetched · {result.tiles_failed:,} failed"
        )
    if result.split_counts is not None:
        table.add_row("Splits", _split_line(result.split_counts))
    minutes, seconds = divmod(int(result.seconds), 60)
    table.add_row("Time", f"{minutes}m {seconds:02d}s" if minutes else f"{seconds}s")
    table.add_row("Files", f"{result.staging_dir}/ (Images/, Masks/, manifest.json, splits/)")
    _console.print(
        Panel(table, title="[bold]Dataset ready[/bold]", title_align="left", border_style="green")
    )
    classes = _class_table(manifest)
    if classes is not None:
        _console.print(classes)
    _console.print(
        "\n[bold]Next[/bold]\n"
        f"  • Inspect it:      [cyan]mapcv info {result.staging_dir}[/cyan]\n"
        f"  • Re-split it:     [cyan]mapcv split {result.staging_dir} --strategy spatial[/cyan]\n"
        f"  • Train on it:     {_DOCS_URL}/guides/use-your-dataset/"
    )


# ── init: templates and the guided wizard ────────────────────────────────────


class Template(str, Enum):
    """Ready-made starting configs."""

    xyz = "xyz"
    sentinel2 = "sentinel2"


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
#   path: buildings.geojson  # .geojson or .kml, in lon/lat
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
  image_format: png          # png | jpg

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
  image_format: npy          # Sentinel-2 patches are float32, bands-first (bands, h, w)

split:
  strategy: spatial
  test_ratio: 0.20
  val_ratio: 0.10
"""
)

_TEMPLATES = {Template.xyz: _XYZ_TEMPLATE, Template.sentinel2: _SENTINEL2_TEMPLATE}


def _ask_bbox_or_file() -> Tuple[Tuple[float, float, float, float], Optional[Path]]:
    while True:
        answer = Prompt.ask(
            "Area: a bounding box [dim]west,south,east,north[/dim] or a .geojson/.kml file",
            console=_console,
        ).strip()
        path = Path(answer).expanduser()
        if path.suffix.lower() in (".geojson", ".json", ".kml"):
            if not path.exists():
                _console.print(f"[red]File not found:[/red] {path}")
                continue
            data = path.read_bytes()
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                geometries, _ = (
                    parse_kml(data) if path.suffix.lower() == ".kml" else parse_geojson(data)
                )
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
            return (west, south, east, north), path
        try:
            west, south, east, north = (float(part) for part in answer.split(","))
        except ValueError:
            _console.print("[red]Enter four numbers like[/red] 74.30,31.48,74.34,31.52")
            continue
        if not (-180 <= west < east <= 180 and -90 <= south < north <= 90):
            _console.print("[red]Expected west < east and south < north, in lon/lat degrees.[/red]")
            continue
        return (west, south, east, north), None


def label_fields(path: Path, max_values: int = 5) -> Dict[str, List[str]]:
    """Return candidate label fields of a GeoJSON/KML file with example values."""
    data = path.read_bytes()
    values: Dict[str, Counter[str]] = {}
    if path.suffix.lower() == ".kml":
        text = data.decode("utf-8", errors="replace")
        names = set(re.findall(r'<(?:\w+:)?(?:Simple)?Data\s+name="([^"]+)"', text))
        for name in sorted(names):
            polygons, _ = parse_kml_rs(data, name)
            values[name] = Counter(label for _, label in polygons if label)
    else:
        obj: Any = json.loads(data.decode("utf-8"))
        features = obj.get("features", [obj]) if isinstance(obj, dict) else []
        for feature in features:
            for key, value in (feature.get("properties") or {}).items():
                if value is not None and not isinstance(value, (dict, list)):
                    values.setdefault(key, Counter())[str(value)] += 1
    return {
        name: [value for value, _ in counter.most_common(max_values)]
        for name, counter in values.items()
        if counter
    }


def _ask_label_field(path: Path) -> Optional[str]:
    fields = label_fields(path)
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
    kind = Prompt.ask(
        "Imagery", choices=["esri", "sentinel2", "custom"], default="esri", console=_console
    )

    _console.print("\n[bold cyan]2/4 Area[/bold cyan]")
    (west, south, east, north), area_file = _ask_bbox_or_file()
    latitude = (south + north) / 2

    imagery_lines: List[str]
    if kind == "sentinel2":
        product = Prompt.ask(
            "Product path or URL [dim](local .zarr, https:// or s3://)[/dim]", console=_console
        )
        resolution = Prompt.ask(
            "Resolution in metres", choices=["10", "20", "60"], default="10", console=_console
        )
        bands = Prompt.ask("Bands", choices=["rgbn", "all"], default="rgbn", console=_console)
        imagery_lines = [
            "  type: eopf_zarr",
            f'  path: "{product}"',
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
            source_line = f'  url_template: "{template}"'
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
    if area_file is not None and Confirm.ask(
        f"Use {area_file.name} as the labels too?", default=True, console=_console
    ):
        labels_path = area_file
    elif area_file is None:
        answer = Prompt.ask(
            "Label file [dim](.geojson/.kml; blank for an image-only dataset)[/dim]",
            default="",
            show_default=False,
            console=_console,
        ).strip()
        if answer:
            labels_path = Path(answer).expanduser()
    label_lines: List[str] = []
    if labels_path is not None:
        field = _ask_label_field(labels_path) if labels_path.exists() else None
        label_lines = ["labels:", f'  path: "{labels_path}"']
        label_lines.append(f"  label_field: {field}" if field else "  label_field: null")

    _console.print("\n[bold cyan]4/4 Patches and output[/bold cyan]")
    patch_size = IntPrompt.ask("Patch size in pixels", default=patch_default, console=_console)
    staging = Prompt.ask("Output folder", default="./dataset", console=_console)
    do_split = Confirm.ask(
        "Split into train/val/test (spatial, no leakage)?", default=True, console=_console
    )

    lines = [
        _HEADER.rstrip(),
        "",
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
        "sampler:",
        f"  patch_size: {patch_size}",
        "  stride: 0                  # 0 = no overlap",
        "  mode: grid",
        f"  edge_strategy: {edge}",
        "",
        "writer:",
        f'  staging_dir: "{staging}"',
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
        "  [cyan]mapcv init my.yaml --template sentinel2[/cyan]   a ready-made example"
    ),
)
def init(
    output: Optional[Path] = typer.Argument(
        None,
        help="Where to write the config (guided default: mapcv.yaml; otherwise stdout).",
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
) -> None:
    """Create a config: answer a few questions, or start from a template."""
    guided = (
        interactive
        if interactive is not None
        else template is None and sys.stdin.isatty() and sys.stdout.isatty()
    )
    text = _wizard() if guided else _TEMPLATES[template or Template.xyz]
    target = output if output is not None else (Path("mapcv.yaml") if guided else None)

    if target is None:
        typer.echo(text, nl=False)
        return
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
    config_path: Path = typer.Argument(..., help="Path to the YAML config."),
) -> None:
    """Estimate tiles, patches, disk and memory for a config [bold]without downloading[/bold]."""
    config = _load_config(config_path)
    try:
        estimate = make_plan(config)
    except (ValueError, RuntimeError, OSError) as exc:
        _console.print(f"[red]Cannot plan this config:[/red] {exc}")
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
    config_path: Path = typer.Argument(..., help="Path to the YAML config."),
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
        _console.print(f"[red]Cannot plan this config:[/red] {exc}")
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
    try:
        result = run_generate(config)
    except (ValueError, RuntimeError, OSError) as exc:
        _console.print(f"[red]Generation failed:[/red] {exc}")
        _console.print(
            "[dim]Fix the cause and run the same command again: finished chunks are kept "
            "and the run resumes.[/dim]"
        )
        raise typer.Exit(code=1)
    if result is not None:
        _print_result(result)


@app.command(rich_help_panel="2. Use a dataset", epilog="Example: [cyan]mapcv info dataset/[/cyan]")
def info(
    staging_dir: Path = typer.Argument(..., help="Dataset directory containing manifest.json."),
) -> None:
    """Summarize a generated dataset: source, shapes, class balance and splits."""
    manifest_path = staging_dir / "manifest.json"
    if not manifest_path.exists():
        _console.print(f"[red]No manifest found at[/red] {manifest_path}")
        raise typer.Exit(code=1)
    manifest = Manifest.load(manifest_path)
    table = Table.grid(padding=(0, 2))
    table.add_column(style="bold cyan", no_wrap=True)
    table.add_column()
    table.add_row("Source", f"{manifest.source_type} · {manifest.product_id or 'unknown product'}")
    if manifest.bands:
        table.add_row("Bands", ", ".join(manifest.bands))
    shape = "×".join(str(dim) for dim in manifest.patch_shape) or "?"
    table.add_row("Patches", f"{len(manifest.patches):,} · {shape} {manifest.dtype or ''}".strip())
    if manifest.crs:
        table.add_row("CRS", manifest.crs)
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
    table.add_row("Manifest", f"version {manifest.version}")
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
    staging_dir: Path = typer.Argument(..., help="Dataset directory containing manifest.json."),
    test_ratio: float = typer.Option(0.20, help="Fraction of patches held out for testing."),
    val_ratio: float = typer.Option(0.10, help="Fraction of the remaining patches for validation."),
    labeled_ratios: Optional[List[float]] = typer.Option(
        None, help="Labeled fractions of train for semi-supervised lists (repeatable)."
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
    try:
        run_split(staging_dir, cfg)
    except FileNotFoundError as exc:
        _console.print(f"[red]{exc}[/red]")
        raise typer.Exit(code=1)


@app.command(
    rich_help_panel="3. Utilities", epilog="Example: [cyan]mapcv validate mapcv.yaml[/cyan]"
)
def validate(
    config_path: Path = typer.Argument(..., help="Path to the YAML config."),
) -> None:
    """Check a config without reading labels or imagery (use [bold]plan[/bold] for estimates)."""
    config = _load_config(config_path)
    _console.print(f"[green]✓[/green] {config_path} is a valid config.")
    _console.print(_settings_table(config))
    if config.labels is not None and not config.labels.path.exists():
        _console.print(f"[yellow]Warning:[/yellow] labels.path not found: {config.labels.path}")
