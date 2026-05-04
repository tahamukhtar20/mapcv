"""mapcv command-line interface."""

from __future__ import annotations

from pathlib import Path
from typing import List, Literal, Optional, cast

import typer
from rich.console import Console

from mapcv.config import MapcvConfig
from mapcv.pipeline import run_generate, run_split
from mapcv.splitter import SplitterConfig

app = typer.Typer(
    name="mapcv",
    help="Satellite imagery dataset creation tool for segmentation.",
    no_args_is_help=True,
)
_console = Console()

_EXAMPLE_CONFIG = """\
region:
  west: 74.20
  south: 31.40
  east: 74.40
  north: 31.60
  zoom: 16

tiles:
  source: google_satellite   # or url_template: "https://..."
  strip_rows: 4
  max_connections: 16
  policy: lenient            # strict | lenient | ignore
  max_failed_ratio: 0.05

# labels:                    # omit for image-only datasets
#   path: labels.kml         # .kml or .geojson
#   label_field: null        # null -> all polygons get class 1
#   all_touched: false

sampler:
  patch_size: 256
  stride: 256                # 0 = same as patch_size (non-overlapping)
  mode: grid                 # grid | random
  edge_strategy: pad         # pad | drop | shift
  pad_mode: zero             # zero | reflect
  max_empty_ratio: 1.0
  min_label_ratio: 0.0

writer:
  staging_dir: ./output
  image_format: png          # png | jpg
  jpg_quality: 95

# split:                     # omit to skip splitting
#   test_ratio: 0.20
#   val_ratio: 0.10
#   labeled_ratios: [0.10, 0.20, 0.30]
#   seed: 42
#   strategy: stratified     # random | stratified
"""


@app.command()
def generate(
    config_path: Path = typer.Argument(..., help="Path to YAML config file."),
) -> None:
    """Fetch tiles, rasterize labels, extract patches, and write a dataset."""
    if not config_path.exists():
        _console.print(f"[red]Config file not found:[/red] {config_path}")
        raise typer.Exit(code=1)
    try:
        config = MapcvConfig.from_yaml(config_path)
    except Exception as exc:
        _console.print(f"[red]Config error:[/red] {exc}")
        raise typer.Exit(code=1)
    run_generate(config)


@app.command()
def split(
    staging_dir: Path = typer.Argument(..., help="Staging directory containing manifest.json."),
    test_ratio: float = typer.Option(0.20, help="Fraction of pool used for the test set."),
    val_ratio: float = typer.Option(0.10, help="Fraction of train+val pool used for val."),
    labeled_ratios: Optional[List[float]] = typer.Option(
        None, help="Labeled fractions (repeatable). Default: 0.10 0.20 0.30."
    ),
    seed: int = typer.Option(42, help="Random seed."),
    strategy: str = typer.Option("stratified", help="Sampling strategy: random | stratified."),
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
            strategy=cast(Literal["random", "stratified"], strategy),
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
    if not config_path.exists():
        _console.print(f"[red]Config file not found:[/red] {config_path}")
        raise typer.Exit(code=1)
    try:
        config = MapcvConfig.from_yaml(config_path)
    except Exception as exc:
        _console.print(f"[red]Config error:[/red] {exc}")
        raise typer.Exit(code=1)

    _console.print("[green]Config is valid.[/green]")

    if config.labels is not None and not config.labels.path.exists():
        _console.print(f"[yellow]Warning:[/yellow] labels.path not found: {config.labels.path}")

    _console.print(
        f"  region : {config.region.west},{config.region.south} -> "
        f"{config.region.east},{config.region.north}  zoom={config.region.zoom}"
    )
    source = config.tiles.source or config.tiles.url_template
    _console.print(f"  tiles  : {source}  strip_rows={config.tiles.strip_rows}")
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
