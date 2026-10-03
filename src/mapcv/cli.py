"""mapcv command-line interface."""

from __future__ import annotations

import warnings
from pathlib import Path
from typing import List, Literal, Optional, cast
from urllib.parse import urlsplit

import typer
from rich.console import Console

from mapcv.config import EOPFZarrImageryConfig, MapcvConfig
from mapcv.pipeline import run_generate, run_split
from mapcv.splitter import SplitterConfig

app = typer.Typer(
    name="mapcv",
    help="A high-performance satellite imagery dataset creation tool for computer vision.",
    no_args_is_help=True,
)
_console = Console()

_PROVIDERS_URL = "https://github.com/tahamukhtar20/mapcv/blob/main/PROVIDERS.md"

_EXAMPLE_CONFIG = f"""\
# You are responsible for complying with the imagery provider's license,
# attribution, rate limits, and terms. mapcv grants no imagery rights:
# {_PROVIDERS_URL}
region:
  west: 74.20
  south: 31.40
  east: 74.40
  north: 31.60

imagery:
  type: xyz
  zoom: 16
  source: esri_satellite     # or an authorized url_template: "https://..."
  strip_rows: 4
  max_connections: 4         # keep requests modest; respect the provider's limits
  policy: lenient            # strict | lenient | ignore
  max_failed_ratio: 0.05

# labels:                    # omit for image-only datasets
#   path: labels.kml         # .kml or .geojson
#   label_field: null        # null -> all polygons get class 1
#   all_touched: false

sampler:
  patch_size: 256
  stride: 0                  # 0 = same as patch_size (non-overlapping)
  mode: grid                 # grid | random
  edge_strategy: pad         # pad | drop | shift
  pad_mode: zero             # zero | reflect
  max_empty_ratio: 1.0
  min_label_ratio: 0.0

writer:
  staging_dir: ./output
  image_format: png          # png | jpg; EOPF Zarr uses npy
  jpg_quality: 95

# split:                     # omit to skip splitting
#   test_ratio: 0.20
#   val_ratio: 0.10
#   labeled_ratios: [0.10, 0.20, 0.30]
#   seed: 42
#   strategy: spatial        # spatial (leakage-safe) | stratified | random
#   block_size: null         # spatial block in pixels; null = 4 x patch_size
"""


def _load_config(config_path: Path) -> MapcvConfig:
    if not config_path.exists():
        _console.print(f"[red]Config file not found:[/red] {config_path}")
        raise typer.Exit(code=1)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always", FutureWarning)
        try:
            config = MapcvConfig.from_yaml(config_path)
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


@app.command()
def generate(
    config_path: Path = typer.Argument(..., help="Path to YAML config file."),
) -> None:
    """Fetch tiles, rasterize labels, extract patches, and write a dataset."""
    config = _load_config(config_path)
    try:
        run_generate(config)
    except (ValueError, RuntimeError, OSError) as exc:
        _console.print(f"[red]Generation failed:[/red] {exc}")
        raise typer.Exit(code=1)


@app.command()
def split(
    staging_dir: Path = typer.Argument(..., help="Staging directory containing manifest.json."),
    test_ratio: float = typer.Option(0.20, help="Fraction of pool used for the test set."),
    val_ratio: float = typer.Option(0.10, help="Fraction of train+val pool used for val."),
    labeled_ratios: Optional[List[float]] = typer.Option(
        None, help="Labeled fractions (repeatable). Default: 0.10 0.20 0.30."
    ),
    seed: int = typer.Option(42, help="Random seed."),
    strategy: str = typer.Option("spatial", help="Split strategy: spatial | stratified | random."),
    block_size: Optional[int] = typer.Option(
        None, help="Spatial block size in pixels (default: 4 x patch size)."
    ),
    sample_limit: Optional[int] = typer.Option(None, help="Cap on total patches sampled."),
) -> None:
    """Split an existing dataset using its manifest (no images are re-read)."""
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
    except Exception as exc:
        _console.print(f"[red]Config error:[/red] {exc}")
        raise typer.Exit(code=1)
    try:
        run_split(staging_dir, cfg)
    except FileNotFoundError as exc:
        _console.print(f"[red]{exc}[/red]")
        raise typer.Exit(code=1)


@app.command()
def validate(
    config_path: Path = typer.Argument(..., help="Path to YAML config file."),
) -> None:
    """Validate a config file without fetching any data."""
    config = _load_config(config_path)

    _console.print("[green]Config is valid.[/green]")

    if config.labels is not None and not config.labels.path.exists():
        _console.print(f"[yellow]Warning:[/yellow] labels.path not found: {config.labels.path}")

    _console.print(
        f"  region : {config.region.west},{config.region.south} -> "
        f"{config.region.east},{config.region.north}"
    )
    if isinstance(config.imagery, EOPFZarrImageryConfig):
        _console.print(
            f"  imagery: EOPF Zarr {config.imagery.path}  "
            f"resolution={config.imagery.resolution}m  bands={len(config.imagery.bands)}"
        )
    else:
        source = config.imagery.source or _redact_url(config.imagery.url_template or "")
        _console.print(
            f"  imagery: XYZ {source}  zoom={config.imagery.zoom}  "
            f"strip_rows={config.imagery.strip_rows}"
        )
    _console.print(
        f"  sampler: patch_size={config.sampler.patch_size}  "
        f"mode={config.sampler.mode}  edge={config.sampler.edge_strategy}"
    )
    _console.print(f"  writer : {config.writer.staging_dir}  fmt={config.writer.image_format}")
    if config.split:
        _console.print(
            f"  split  : test={config.split.test_ratio}  "
            f"val={config.split.val_ratio}  strategy={config.split.strategy}"
        )


@app.command()
def init(
    output: Optional[Path] = typer.Argument(
        None, help="Write example config to this file. Prints to stdout if omitted."
    ),
) -> None:
    """Scaffold an example config file."""
    if output is None:
        _console.print(_EXAMPLE_CONFIG, highlight=False)
    else:
        output.write_text(_EXAMPLE_CONFIG)
        _console.print(f"[green]Example config written to[/green] [bold]{output}[/bold]")
    _console.print(f"[dim]Check your imagery provider's terms first: {_PROVIDERS_URL}[/dim]")
