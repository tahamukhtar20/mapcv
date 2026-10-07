"""mapcv command-line interface.

The intended journey is ``mapcv init`` (write a config), ``mapcv plan`` (see
what it will cost), ``mapcv generate`` (build the dataset) and ``mapcv info``
(inspect the result).
"""

from __future__ import annotations

import datetime
import errno
import glob
import json
import logging
import os
import platform
import re
import shlex
import sys
import time
import warnings
from collections import Counter
from enum import Enum
from pathlib import Path
from typing import Any, Literal, NoReturn, cast
from urllib.parse import urlsplit

import numpy as np
import typer
import yaml
from pydantic import ValidationError
from rich.console import Console
from rich.markup import escape
from rich.panel import Panel
from rich.progress import (
    BarColumn,
    MofNCompleteColumn,
    Progress,
    SpinnerColumn,
    TextColumn,
    TimeElapsedColumn,
    TimeRemainingColumn,
)
from rich.prompt import Confirm, IntPrompt, Prompt
from rich.status import Status
from rich.table import Table
from rich.text import Text
from typer.core import TyperGroup

import mapcv
from mapcv import doctor
from mapcv._mapcv_rs import parse_kml as _parse_kml_bytes
from mapcv.config import (
    RASTER_LABEL_TYPES,
    UNION_TAGS,
    ContinuousLabelsConfig,
    EOPFZarrImageryConfig,
    GeoTiffImageryConfig,
    LabelsConfig,
    MapcvConfig,
    RasterLabelsConfig,
    StacCogImageryConfig,
    eopf_local_path,
)
from mapcv.labels import (
    MAX_CLASS_ID,
    VECTOR_LABEL_SUFFIXES,
    _normalize_label,
    load_vector_labels,
    vector_attributes,
    vector_layers,
)
from mapcv.manifest import Manifest, ManifestMismatchError, SourceRecord, patch_folders
from mapcv.pipeline import GenerateResult, run_generate, run_split
from mapcv.planning import Plan, ground_resolution_m, human_bytes
from mapcv.planning import plan as make_plan
from mapcv.splitter import SplitterConfig
from mapcv.writers.detection import categories


class _MapcvGroup(TyperGroup):
    """The command group: Ctrl-C in any command ends with one line and exit code 130
    (Typer alone exits silently, leaving the shell prompt after a half-written line)."""

    def invoke(self, ctx: Any) -> Any:
        try:
            return super().invoke(ctx)
        except KeyboardInterrupt:
            _err_console.print("\n[yellow]Interrupted.[/yellow]")
            raise typer.Exit(code=130) from None


app = typer.Typer(
    name="mapcv",
    cls=_MapcvGroup,
    help=(
        "Turn a region, imagery and labels into a ready-to-train segmentation, detection, "
        "instance segmentation, classification, change detection or regression dataset.\n\n"
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
# Results, plans and summaries go to stdout; errors (and Ctrl-C) to stderr, so
# ``mapcv plan x.yaml > plan.txt`` still shows a failure on the terminal.
_console = Console()
_err_console = Console(stderr=True)
# NO_COLOR as the environment set it; --no-color turns colours off for one run.
_NO_COLOR_FROM_ENV = _console.no_color
# --quiet: hide the spinner, progress bars and progress messages.
_quiet = False
# --debug: tracebacks after error messages, and mapcv's debug log on stderr.
_debug = False
_debug_handlers: list[logging.Handler] = []
_DEBUG_FORMAT = "%(asctime)s %(name)s %(levelname)s: %(message)s"


class _StderrDebugHandler(logging.Handler):
    """Debug-level records on stderr. Looks stderr up at each record (a progress bar
    replaces it while it runs) and leaves INFO and above to the CLI's own messages."""

    def emit(self, record: logging.LogRecord) -> None:
        if record.levelno >= logging.INFO and record.exc_info is None:
            return  # shown by the CLI already
        try:
            sys.stderr.write(self.format(record) + "\n")
        except (OSError, ValueError):  # pragma: no cover - a closed stderr
            pass


def _configure_debug(debug: bool, log_file: Path | None) -> None:
    """Install (or remove, for a later run in the same process) the debug handlers."""
    global _debug
    _debug = debug or log_file is not None
    logger = logging.getLogger("mapcv")
    for handler in _debug_handlers:
        logger.removeHandler(handler)
        handler.close()
    _debug_handlers.clear()
    if not _debug:
        logger.setLevel(logging.NOTSET)
        return
    formatter = logging.Formatter(_DEBUG_FORMAT)
    if debug:
        stderr = _StderrDebugHandler(logging.DEBUG)
        stderr.setFormatter(formatter)
        _debug_handlers.append(stderr)
    if log_file is not None:
        file_handler = logging.FileHandler(log_file, mode="w", encoding="utf-8")
        file_handler.setLevel(logging.DEBUG)
        file_handler.setFormatter(formatter)
        _debug_handlers.append(file_handler)
    for handler in _debug_handlers:
        logger.addHandler(handler)
    logger.setLevel(logging.DEBUG)
    logger.debug(
        "mapcv %s, Python %s, %s", mapcv.__version__, platform.python_version(), platform.platform()
    )


def _debug_traceback(exc: BaseException) -> None:
    """With --debug (or --debug-log), the full traceback after the friendly message."""
    if _debug:
        logging.getLogger("mapcv.cli").error("%s: %s", type(exc).__name__, exc, exc_info=exc)


def _live_terminal() -> bool:
    """Whether stdout is a terminal that can redraw a line (spinners, progress bars).

    Piped or redirected output, and ``TERM=dumb``, get plain lines instead: a bar
    there is only its last frame, with a spinner glyph and an ``eta -:--:--``.
    """
    return _console.is_terminal and not _console.is_dumb_terminal


def _duration(seconds: float) -> str:
    """``45s``, ``3m 07s``, ``2h 05m``: the same format everywhere."""
    seconds = int(seconds)
    hours, rest = divmod(seconds, 3600)
    minutes, secs = divmod(rest, 60)
    if hours:
        return f"{hours}h {minutes:02d}m"
    return f"{minutes}m {secs:02d}s" if minutes else f"{secs}s"


class _GenerateFeedback(logging.Handler):
    """The terminal side of a generation: a spinner while the imagery opens, a progress
    bar over the chunks (it is the ``on_chunk`` callback), and the library's log
    messages, dimmed. Output that is not a live terminal gets a plain line per tenth
    of the chunks instead. All of it is hidden under ``--quiet``."""

    def __init__(self) -> None:
        super().__init__(logging.INFO)
        self._status: Status | None = None
        self._progress: Progress | None = None
        self._task: Any | None = None
        self._level = logging.NOTSET
        self._live = _live_terminal()
        self._started = time.monotonic()
        self._reported = -1  # the last tenth printed in plain mode

    def __enter__(self) -> _GenerateFeedback:
        logger = logging.getLogger("mapcv")
        self._level = logger.level
        logger.setLevel(logging.DEBUG if _debug else logging.INFO)
        logger.addHandler(self)
        if not _quiet:
            if self._live:
                self._status = _console.status("Opening imagery…")
                self._status.start()
            else:
                _console.print("Opening imagery…")
        return self

    def __exit__(self, *exc: object) -> None:
        self._stop_status()
        if self._progress is not None:
            self._progress.stop()
        logger = logging.getLogger("mapcv")
        logger.removeHandler(self)
        logger.setLevel(self._level)

    def _stop_status(self) -> None:
        if self._status is not None:
            self._status.stop()
            self._status = None

    def emit(self, record: logging.LogRecord) -> None:
        if not _quiet:
            _console.print(f"[dim]{escape(record.getMessage())}[/dim]")

    def __call__(self, done: int, total: int) -> None:
        self._stop_status()
        if not self._live:
            tenth = done * 10 // total if total else 10
            if not _quiet and total and tenth > self._reported:
                self._reported = tenth
                _console.print(
                    f"Reading imagery and writing patches: {done:,}/{_plural(total, 'chunk')} "
                    f"({done / total:.0%}), {_duration(time.monotonic() - self._started)}",
                    soft_wrap=True,
                )
            return
        if self._progress is None and total and not _quiet:
            self._progress = Progress(
                SpinnerColumn(),
                TextColumn("[progress.description]{task.description}"),
                BarColumn(),
                MofNCompleteColumn(),
                TextColumn("chunks •"),
                TimeElapsedColumn(),
                TextColumn("• eta"),
                TimeRemainingColumn(),
                console=_console,
            )
            self._progress.start()
            self._task = self._progress.add_task("Reading imagery and writing patches", total=total)
        if self._progress is not None and self._task is not None:
            self._progress.update(self._task, completed=done)


_DOCS_URL = "https://tahamukhtar20.github.io/mapcv"
_PROVIDERS_URL = "https://github.com/tahamukhtar20/mapcv/blob/main/PROVIDERS.md"


# ── Shared helpers ───────────────────────────────────────────────────────────


def _version() -> str:
    return mapcv.__version__


def _plural(count: int, noun: str, nouns: str | None = None) -> str:
    """``1 patch``, ``1,234 patches``: a count with its noun, never ``patch(es)``."""
    return f"{count:,} {noun if count == 1 else nouns or noun + 's'}"


def _fail(message: str, hint: str | None = None, code: int = 1) -> NoReturn:
    """Print an error (Rich markup) and an optional dimmed next step on stderr, and exit.

    Every user error ends here, so they read alike: what went wrong, then what to do.
    """
    _err_console.print(message)
    if hint:
        _err_console.print(f"[dim]{hint}[/dim]")
    raise typer.Exit(code=code)


def _shell_path(path: Path | str) -> str:
    """A path as it would be typed in a command: quoted when it has spaces or other
    characters the shell would split on, so the suggested command can be copied."""
    text = str(path)
    if os.name == "nt":
        return f'"{text}"' if any(char in text for char in " \t&()^;,=") else text
    return shlex.quote(text)


def _coordinate(value: float) -> str:
    """A longitude or latitude to 6 decimals (about 0.1 m), without trailing zeros."""
    return f"{value:.6f}".rstrip("0").rstrip(".")


def _generate_os_error(exc: OSError) -> tuple[str, str | None]:
    """For a failed generation: a short reason for an operating-system error, without
    the ``[Errno 13]`` prefix, and the next step when there is a usual one."""
    where = f": {exc.filename}" if exc.filename else ""
    if exc.errno in (errno.EACCES, errno.EPERM, errno.EROFS):
        return (
            f"permission denied{where}",
            "Make that folder writable, or set writer.staging_dir to a folder you can write to.",
        )
    if exc.errno in (errno.ENOSPC, getattr(errno, "EDQUOT", errno.ENOSPC)):
        return (
            f"no space left on the disk{where}",
            "Free some space, or set writer.staging_dir to another disk and start again there.",
        )
    return f"{(exc.strerror or str(exc)).rstrip('.')}{where}", None


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
    no_color: bool = typer.Option(
        False, "--no-color", help="Print without colours, as NO_COLOR=1 does."
    ),
    debug: bool = typer.Option(
        False, "--debug", help="Show full tracebacks and mapcv's debug log (for bug reports)."
    ),
    debug_log: Path | None = typer.Option(
        None, "--debug-log", metavar="FILE", help="Also write the debug log to FILE."
    ),
) -> None:
    """Turn a region, imagery and labels into a ready-to-train segmentation, detection, instance
    segmentation, classification, change detection or regression dataset."""
    global _quiet
    _quiet = quiet
    _console.no_color = _err_console.no_color = no_color or _NO_COLOR_FROM_ENV
    _configure_debug(debug, debug_log)


def _validation_items(exc: ValidationError, options: bool = False) -> list[tuple[str, str]]:
    """``(field, message)`` per validation error; with ``options``, fields are named as
    the command-line options that set them (``--test-ratio``)."""
    items = []
    for error in exc.errors():
        location = ".".join(
            str(part)
            for part in error["loc"]
            if not str(part).startswith("function-") and part not in UNION_TAGS
        )
        if options and location:
            location = "--" + location.replace("_", "-")
        message = str(error["msg"]).removeprefix("Value error, ")
        items.append((location or "config", message))
    return items


def _print_items(console: Console, items: list[tuple[str, str]]) -> None:
    """``  • field: message`` lines whose wrapped text stays indented under the message
    (and, unlike a table, without spaces padding each line to the terminal width)."""
    width = max(console.width - 4, 20)
    for location, message in items:
        text = Text.assemble((location, "bold"), f": {message}")
        for index, line in enumerate(text.wrap(console, width)):
            line.rstrip()
            console.print(Text("  • " if index == 0 else "    ") + line, soft_wrap=True)


_INIT_HINT = "Fix the fields above, or start from a working config with [bold]mapcv init[/bold]."


def _load_config(config_path: Path) -> MapcvConfig:
    if not config_path.exists():
        _fail(
            f"[red]Config file not found:[/red] {escape(str(config_path))}",
            "Create one with [bold]mapcv init[/bold].",
        )
    if config_path.is_dir():
        _fail(
            f"[red]Not a config file:[/red] {escape(str(config_path))} is a folder.",
            "Pass the YAML file, for example [bold]mapcv.yaml[/bold].",
        )
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        try:
            config = MapcvConfig.from_yaml(config_path)
        except ValidationError as exc:
            _debug_traceback(exc)
            items = _validation_items(exc)
            if all(not error["loc"] and error["type"] == "model_type" for error in exc.errors()):
                _fail(
                    f"[red]Config error[/red] in {escape(str(config_path))}: it holds no "
                    "settings (expected YAML keys such as region:, imagery: and writer:).",
                    "Start from a working config with [bold]mapcv init[/bold].",
                )
            _err_console.print(f"[red]Config error[/red] in {escape(str(config_path))}:")
            _print_items(_err_console, items)
            _fail(f"[dim]{_INIT_HINT}[/dim]")
        except yaml.YAMLError as exc:
            _debug_traceback(exc)
            mark = getattr(exc, "problem_mark", None)
            where = f" at line {mark.line + 1}, column {mark.column + 1}" if mark else ""
            problem = getattr(exc, "problem", None) or str(exc)
            _fail(
                f"[red]Config error[/red] in {escape(str(config_path))}: not valid YAML{where} "
                f"({escape(str(problem))}).",
                "Check the indentation, colons and brackets there.",
            )
        except UnicodeDecodeError as exc:
            _debug_traceback(exc)
            _fail(
                f"[red]Config error[/red] in {escape(str(config_path))}: not a UTF-8 text file.",
                "Pass the YAML config, saved as UTF-8.",
            )
        except OSError as exc:
            _debug_traceback(exc)
            _fail(
                f"[red]Config error[/red] in {escape(str(config_path))}: cannot read it "
                f"({escape(exc.strerror or str(exc))})."
            )
        except Exception as exc:  # noqa: BLE001 - any other config failure is a user error
            _debug_traceback(exc)
            _fail(f"[red]Config error[/red] in {escape(str(config_path))}: {escape(str(exc))}")
    _show_warnings(caught, set())
    return config


def _show_warnings(caught: list[warnings.WarningMessage], shown: set[str]) -> None:
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
    if config.multi_source:
        return "; ".join(
            f"{name}: {_source_label(imagery)}"
            for name, imagery in zip(config.source_names, config.sources)
        )
    return _source_label(config.primary_imagery)


def _source_label(imagery: Any) -> str:
    if isinstance(imagery, StacCogImageryConfig):
        cog = imagery.search
        masked = f" · SCL mask {imagery.scl_mask}" if imagery.scl_mask else ""
        return (
            f"Sentinel-2 COGs · search {cog.collection} {cog.datetime} "
            f"≤ {cog.max_cloud:g}% cloud · bands {', '.join(imagery.bands)}{masked}"
        )
    if isinstance(imagery, EOPFZarrImageryConfig):
        if imagery.search is not None:
            search = imagery.search
            where = f"search {search.collection} {search.datetime} ≤ {search.max_cloud:g}% cloud"
        else:
            where = _redact_url(imagery.path or "")
        masked = f" · SCL mask {imagery.scl_mask}" if imagery.scl_mask else ""
        return (
            f"Sentinel-2 EOPF {where} · {imagery.resolution} m · {len(imagery.bands)} bands{masked}"
        )
    if isinstance(imagery, GeoTiffImageryConfig):
        where = _redact_url(imagery.path) if "://" in imagery.path else imagery.path
        selected = f"bands {imagery.bands}" if imagery.bands else "all bands"
        overview = f" · overview {imagery.overview}" if imagery.overview else ""
        return f"GeoTIFF {where} · {selected}{overview}"
    if imagery.earth_engine is not None:
        engine = imagery.earth_engine
        return f"Earth Engine {engine.image or engine.collection} · zoom {imagery.zoom}"
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
    if config.task == "change":
        change = config.change_options
        before, after = config.source_names[:2]
        source = (
            "before/after label sets" if change.before is not None else "labels mark the change"
        )
        return f"change · {before} → {after} · {source} · changed pixels = {change.change_value}"
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


def _region_box(config: MapcvConfig) -> str:
    region = config.region
    west, south, east, north = (
        _coordinate(value) for value in (region.west, region.south, region.east, region.north)
    )
    return f"{west}, {south} → {east}, {north}"


def _settings_table(config: MapcvConfig) -> Table:
    table = Table.grid(padding=(0, 2))
    table.add_column(style="bold cyan", no_wrap=True)
    table.add_column()
    if config.task != "segmentation":
        table.add_row("Task", _task_label(config))
    region = config.region
    area = f"{region.path} · " if region.path is not None else ""
    table.add_row(
        "Region",
        f"{area}{_region_box(config)} (W, S → E, N)",
    )
    table.add_row("Imagery", _imagery_label(config))
    change = config.change_options
    if config.task == "change" and change.before is not None and change.after is not None:
        table.add_row("Labels", f"before: {change.before.path} · after: {change.after.path}")
    elif config.labels is None:
        table.add_row("Labels", "none (image-only dataset)")
    elif isinstance(config.labels, RasterLabelsConfig):
        raster = config.labels
        where = _redact_url(raster.path) if "://" in raster.path else raster.path
        table.add_row(
            "Labels",
            f"{where} · raster band {raster.band} · "
            f"{_plural(len(raster.class_map()), 'class', 'classes')}",
        )
    elif isinstance(config.labels, ContinuousLabelsConfig):
        values = config.labels
        where = _redact_url(values.path) if "://" in values.path else values.path
        scaling = (
            f" · × {values.scale:g} + {values.offset:g}"
            if (values.scale, values.offset) != (1.0, 0.0)
            else ""
        )
        table.add_row("Labels", f"{where} · values of band {values.band}{scaling}")
    elif config.labels.osm is not None:
        names = ", ".join(entry.name for entry in config.labels.osm.classes)
        table.add_row("Labels", f"OpenStreetMap (Overpass) · {names}")
    elif config.labels.files is not None:
        for index, file in enumerate(config.labels.files):
            what = f"field: {file.label_field}" if file.label_field else f"class: {file.class_name}"
            layer = f" · layer: {file.layer}" if file.layer else ""
            table.add_row("Labels" if index == 0 else "", f"{file.path}{layer} · {what}")
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
    if config.task != "segmentation":
        table.add_row("Task", _task_label(config))
    table.add_row(
        "Region",
        f"{_region_box(config)}  [dim](≈ {width_km:.1f} × {height_km:.1f} km)[/dim]",
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
            detail = _plural(labels.polygons, "polygon")
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


def _make_plan(config: MapcvConfig) -> Plan:
    try:
        return make_plan(config)
    except (ValueError, RuntimeError, OSError) as exc:
        _debug_traceback(exc)
        _fail(f"[red]Cannot plan this config:[/red] {escape(str(exc))}")


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
    if config.multi_source:
        _console.print(
            "[dim]Several sources: each is read on the first source's grid (coarser ones are "
            "repeated, never interpolated).[/dim]"
        )
    elif isinstance(config.imagery, GeoTiffImageryConfig):
        _console.print("[dim]Your own imagery: mapcv reads it as it is, without resampling.[/dim]")


def _source_line(record: SourceRecord) -> str:
    """One source of a multi-source dataset: type, product, patch shape and its grid."""
    shape = "×".join(str(dim) for dim in record.patch_shape) or "?"
    line = (
        f"{record.source_type} · {record.product_id or 'unknown product'} · "
        f"{shape} {record.dtype or ''}".strip()
    )
    factor = (record.model_extra or {}).get("factor")
    if factor:
        line += f" · {factor}× coarser, repeated onto the grid"
    return line


def _raster_labels(manifest: Manifest) -> bool:
    """Whether the dataset's masks were read from a label raster."""
    target = manifest.target
    return target is not None and (target.labels or {}).get("type") == "raster"


def _class_names(manifest: Manifest) -> dict[str, str]:
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


def _object_table(manifest: Manifest) -> Table | None:
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


def _label_table(manifest: Manifest) -> Table | None:
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


def _value_table(manifest: Manifest) -> Table | None:
    """Target values over all patches, for regression datasets."""
    valid = 0
    total = 0.0
    low, high = float("inf"), float("-inf")
    for entry in manifest.patches:
        values = entry["summary"].get("values") or {}
        count = int(values.get("valid", 0))
        if not count:
            continue
        valid += count
        total += float(values["mean"]) * count
        low, high = min(low, float(values["min"])), max(high, float(values["max"]))
    if not valid:
        return None
    pixels = len(manifest.patches) * int((manifest.sampler or {}).get("patch_size", 0)) ** 2
    table = Table(box=None, padding=(0, 2), show_edge=False)
    table.add_column("target")
    table.add_column("value", justify="right")
    table.add_row(
        "pixels with a value", f"{valid:,}" + (f" ({valid / pixels:.1%})" if pixels else "")
    )
    table.add_row("min", f"{low:.6g}")
    table.add_row("mean", f"{total / valid:.6g}")
    table.add_row("max", f"{high:.6g}")
    return table


def _class_table(manifest: Manifest) -> Table | None:
    if manifest.task in ("detection", "instance"):
        return _object_table(manifest)
    if manifest.task == "classification":
        return _label_table(manifest)
    if manifest.task == "regression":
        return _value_table(manifest)
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


def _split_line(counts: dict[str, int]) -> str:
    total = counts["train"] + counts["val"] + counts["test"]
    parts = [
        f"{name} {counts[name]:,} ({counts[name] / total:.0%})" if total else f"{name} 0"
        for name in ("train", "val", "test")
    ]
    line = " · ".join(parts)
    if counts.get("dropped"):
        dropped = _plural(counts["dropped"], "overlapping patch", "overlapping patches")
        line += f" [dim]· {dropped} left out[/dim]"
    return line


# What generate may write next to the patch folders, in the order the summary lists it.
_DATASET_FILES = (
    "manifest.json",
    "splits/",
    "patches.geojson",
    # detection and instance: COCO, YOLO labels and image lists, the Ultralytics file
    "annotations/",
    "labels/",
    "train.txt",
    "val.txt",
    "test.txt",
    "dataset.yaml",
    # classification
    "labels.csv",
    "labels_train.csv",
    "labels_val.csv",
    "labels_test.csv",
    "labels.json",
    "classes.txt",
)


def _dataset_files(staging_dir: Path, manifest: Manifest) -> list[str]:
    """The folders and files of a generated dataset that are there, for the summary."""
    folders = [f"{folder}/" for folder in patch_folders(manifest)] or ["Images/"]
    return [
        name for name in (*folders, *_DATASET_FILES) if (staging_dir / name.rstrip("/")).exists()
    ]


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
    if len(manifest.sources) > 1:
        for record in manifest.sources:
            table.add_row(f"Source {record.name}", _source_line(record))
    else:
        source = manifest.source
        table.add_row("Source", f"{source.source_type} · {source.product_id or 'unknown product'}")
        shape = "×".join(str(dim) for dim in source.patch_shape) if source.patch_shape else "?"
        table.add_row("Shape", f"{shape} {source.dtype or ''}".strip())
    if result.tiles_requested or result.tiles_cached:
        cached = f" · {result.tiles_cached:,} from the cache" if result.tiles_cached else ""
        table.add_row(
            "Tiles",
            f"{result.tiles_requested:,} fetched{cached} · {result.tiles_failed:,} failed",
        )
    if result.split_counts is not None:
        table.add_row("Splits", _split_line(result.split_counts))
    table.add_row("Time", _duration(result.seconds))
    written = ", ".join(_dataset_files(result.staging_dir, manifest))
    table.add_row("Files", f"{escape(str(result.staging_dir))}/ ({written})")
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
    staging = escape(_shell_path(result.staging_dir))
    # Not wrapped: a URL broken over two lines can't be clicked or copied.
    _console.print(
        "\n[bold]Next[/bold]\n"
        f"  • Inspect it:      [cyan]mapcv info {staging}[/cyan]\n"
        f"  • Re-split it:     [cyan]mapcv split {staging} --strategy spatial[/cyan]\n"
        f"  • Band stats:      [cyan]mapcv stats {staging}[/cyan]\n"
        f"  • Train on it:     {_DOCS_URL}/{guide}",
        soft_wrap=True,
    )


# ── init: templates and the guided wizard ────────────────────────────────────


class Template(str, Enum):
    """Ready-made starting configs."""

    xyz = "xyz"
    sentinel2 = "sentinel2"
    geotiff = "geotiff"
    earth_engine = "earth-engine"
    detection = "detection"
    instance = "instance"
    classification = "classification"
    change = "change"
    regression = "regression"


_HEADER = f"""\
# mapcv config - docs: {_DOCS_URL}/reference/configuration/
# Check the cost first with `mapcv plan <this file>`, then run `mapcv generate <this file>`.
# Imagery sources, their licenses and credit lines: {_PROVIDERS_URL}
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
  strategy: spatial          # spatial (no leakage between splits) | stratified | random | region
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
    f"""\
# mapcv config - docs: {_DOCS_URL}/reference/configuration/
# Check the cost first with `mapcv plan <this file>`, then run `mapcv generate <this file>`.
# Your own GeoTIFF or Cloud Optimized GeoTIFF, read as it is.
"""
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

_EARTH_ENGINE_TEMPLATE = (
    _HEADER
    + """
# Google Earth Engine: pip install "mapcv[gee]", then log in once with
#   earthengine authenticate
# Requests run on your Earth Engine account and Cloud project; its terms, quotas and any
# billing are yours.
region:                      # WGS-84 lon/lat bounding box
  west: 4.9375
  south: 52.3725
  east: 4.9515
  north: 52.3780

imagery:
  type: xyz
  zoom: 15                   # ~ 3 m/px here; Earth Engine renders any zoom
  max_connections: 4
  earth_engine:
    # A cloud-free Sentinel-2 summer composite (10 m, worldwide since 2017):
    collection: COPERNICUS/S2_SR_HARMONIZED
    start: "2024-06-01"
    end: "2024-09-01"
    max_cloud: 40            # skip scenes with more than 40 % cloud
    cloud_score_plus: 0.6    # mask the cloudy pixels left (Sentinel-2 only)
    reducer: median          # median | mean | mosaic | min | max
    vis: {bands: [B4, B3, B2], min: 0, max: 3000, gamma: 1.2}
    project: YOUR-CLOUD-PROJECT
    # Landsat 8 (30 m, since 2013):
    #   collection: LANDSAT/LC08/C02/T1_L2, max_cloud: 20, no cloud_score_plus,
    #   vis: {bands: [SR_B4, SR_B3, SR_B2], min: 7300, max: 18000, gamma: 1.2}
    # NAIP (about 0.6 m, United States only):
    #   collection: USDA/NAIP/DOQQ, reducer: mosaic, no cloud settings,
    #   vis: {bands: [R, G, B], min: 0, max: 255}
    # One image instead of a collection: image: <asset id> (no start/end/cloud settings)

# labels:                    # omit for an image-only dataset
#   path: buildings.geojson
#   label_field: null

sampler:
  patch_size: 256
  edge_strategy: pad

writer:
  staging_dir: ./dataset
  image_format: png

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
  path: buildings.geojson    # any vector label format; one instance per feature
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

_CHANGE_TEMPLATE = (
    _HEADER
    + """
task: change                 # before/after image pairs and a change mask (A/, B/, label/)

region:                      # WGS-84 lon/lat bounding box
  west: 4.9375
  south: 52.3725
  east: 4.9515
  north: 52.3780

imagery:                     # two sources on one grid: the image before, then after
  - type: geotiff
    name: before
    path: city_2023.tif      # same CRS and pixel grid as the after image
  - type: geotiff
    name: after
    path: city_2025.tif

# Either labels that mark what changed ...
labels:
  path: changes.geojson      # every polygon is change
# ... or remove labels and compare two label sets (a pixel changed where they differ):
# change:
#   before: {path: buildings_2023.geojson}
#   after: {path: buildings_2025.geojson}

change:
  change_value: 1            # mask value of changed pixels (255 for loaders that expect 0/255)

sampler:
  patch_size: 256
  stride: 0                  # 0 = patch_size (no overlap)
  mode: grid
  edge_strategy: drop        # pad | drop | shift

writer:
  staging_dir: ./dataset     # A/ (before), B/ (after), label/ (change masks)
  image_format: png          # png | jpg need 1 or 3 uint8 bands; tif or npy keep any

split:
  strategy: spatial
  test_ratio: 0.20
  val_ratio: 0.10
  seed: 42
"""
)

_REGRESSION_TEMPLATE = (
    _HEADER
    + """
task: regression             # a float value per pixel (canopy height, biomass, ...) instead of classes

region:                      # WGS-84 lon/lat bounding box
  west: 4.9375
  south: 52.3725
  east: 4.9515
  north: 52.3780

imagery:
  type: geotiff
  path: ortho.tif            # local, https:// or anonymous s3://

labels:
  type: continuous
  path: canopy_height.tif    # any CRS and resolution: read at each image pixel's centre
  # band: 1
  # nodata: -9999            # default: the file's NoData
  # scale: 0.01              # target = value * scale + offset (e.g. centimetres to metres)
  # offset: 0.0
  # valid_min: 0             # raw values outside this range have no target (NaN)

sampler:
  patch_size: 256
  stride: 0                  # 0 = patch_size (no overlap)
  mode: grid
  edge_strategy: drop        # pad | drop | shift

writer:
  staging_dir: ./dataset     # Images/ and Masks/ (float32 targets, NaN = no value)
  image_format: tif          # tif | npy (png/jpg for 8-bit RGB imagery)
  mask_format: tif           # tif | npy: float32 targets

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
    Template.earth_engine: _EARTH_ENGINE_TEMPLATE,
    Template.detection: _DETECTION_TEMPLATE,
    Template.instance: _INSTANCE_TEMPLATE,
    Template.classification: _CLASSIFICATION_TEMPLATE,
    Template.change: _CHANGE_TEMPLATE,
    Template.regression: _REGRESSION_TEMPLATE,
}


def _yaml_str(value: str) -> str:
    """Single-quoted YAML scalar: backslashes (Windows paths) stay literal."""
    return "'" + value.replace("'", "''") + "'"


def _ask_layer(path: Path) -> str | None:
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
    default_bbox: tuple[float, float, float, float] | None = None,
) -> tuple[tuple[float, float, float, float], Path | None, str | None]:
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
                _console.print(f"[red]File not found:[/red] {escape(str(path))}")
                continue
            layer = _ask_layer(path)
            try:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    geometries, _ = load_vector_labels(path, layer=layer)
            except (ValueError, OSError) as exc:
                _debug_traceback(exc)
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
                f"[dim]Using the extent of {_plural(len(geometries), 'polygon')}: "
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


def label_fields(path: Path, max_values: int = 5, layer: str | None = None) -> dict[str, list[str]]:
    """Return candidate label fields of a vector label file with example values."""
    values: dict[str, Counter[str]] = {}
    suffix = path.suffix.lower()
    if suffix == ".kml":
        data = path.read_bytes()
        text = data.decode("utf-8", errors="replace")
        names = set(re.findall(r'<(?:\w+:)?(?:Simple)?Data\s+name="([^"]+)"', text))
        for name in sorted(names):
            polygons, _ = _parse_kml_bytes(data, name)
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


def _ask_label_field(path: Path, layer: str | None = None) -> str | None:
    try:
        fields = label_fields(path, layer=layer)
    except (ValueError, OSError) as exc:
        _debug_traceback(exc)
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
        show_default=False,
        show_choices=False,
        console=_console,
    )
    return answer or None


class _GeoTiffAnswer:
    """What the wizard learned about the file the user pointed it at."""

    def __init__(
        self,
        imagery_lines: list[str],
        image_format: str,
        extent: tuple[float, float, float, float] | None,
    ) -> None:
        self.imagery_lines = imagery_lines
        self.image_format = image_format
        self.extent = extent


def _geotiff_wgs84_extent(tif: Any) -> tuple[float, float, float, float] | None:
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
            _debug_traceback(exc)
            _console.print(f"[red]Cannot read that file:[/red] {escape(str(exc))}")
            continue
        break
    info = tif.info
    table = Table.grid(padding=(0, 2))
    table.add_column(style="bold cyan", no_wrap=True)
    table.add_column()
    crs = f"EPSG:{info.epsg}" if info.epsg is not None else f"none usable ({info.crs_error})"
    table.add_row("CRS", crs)
    table.add_row(
        "Size",
        f"{info.width:,} × {info.height:,} px · {_plural(info.count, 'band')} · {info.dtype}",
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
    tif: Any, bbox: tuple[float, float, float, float]
) -> tuple[dict[int, int], bool]:
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


def _ask_label_raster(path_text: str, bbox: tuple[float, float, float, float]) -> list[str]:
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
        _debug_traceback(exc)
        _console.print(
            f"[yellow]Cannot read that file:[/yellow] {escape(str(exc))}. Edit labels.classes."
        )
        return placeholder
    info = tif.info
    table = Table.grid(padding=(0, 2))
    table.add_column(style="bold cyan", no_wrap=True)
    table.add_column()
    crs = f"EPSG:{info.epsg}" if info.epsg is not None else f"none usable ({info.crs_error})"
    table.add_row("CRS", crs)
    table.add_row(
        "Size",
        f"{info.width:,} × {info.height:,} px · {_plural(info.count, 'band')} · {info.dtype}",
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
        _debug_traceback(exc)
        _console.print(
            f"[yellow]Cannot read its values:[/yellow] {escape(str(exc))}. Edit labels.classes."
        )
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
        _console.print(f"[dim]… and {_plural(len(values) - 20, 'more value')}.[/dim]")
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


# Earth Engine datasets the wizard offers, with colours that render them well.
_EE_PRESETS: dict[str, dict[str, Any]] = {
    "sentinel2": {
        "about": "Sentinel-2 surface reflectance, 10 m, worldwide since 2017",
        "collection": "COPERNICUS/S2_SR_HARMONIZED",
        "vis": "{bands: [B4, B3, B2], min: 0, max: 3000, gamma: 1.2}",
        "metres": 10.0,
        "max_cloud": 40,
        "cloud_score_plus": True,
        "reducer": "median",
    },
    "landsat": {
        "about": "Landsat 8 surface reflectance, 30 m, worldwide since 2013",
        "collection": "LANDSAT/LC08/C02/T1_L2",
        "vis": "{bands: [SR_B4, SR_B3, SR_B2], min: 7300, max: 18000, gamma: 1.2}",
        "metres": 30.0,
        "max_cloud": 20,
        "cloud_score_plus": False,
        "reducer": "median",
    },
    "naip": {
        "about": "NAIP aerial photos, about 0.6 m, United States only",
        "collection": "USDA/NAIP/DOQQ",
        "vis": "{bands: [R, G, B], min: 0, max: 255}",
        "metres": 0.6,
        "max_cloud": None,
        "cloud_score_plus": False,
        "reducer": "mosaic",
    },
}


def _zoom_for(metres: float, latitude: float) -> int:
    """The first zoom whose pixels are at least as fine as ``metres``."""
    for zoom in range(1, 23):
        if ground_resolution_m(zoom, latitude) <= metres:
            return zoom
    return 22


def _ask_earth_engine(latitude: float) -> list[str]:
    """Earth Engine questions: a dataset (or any asset), dates, clouds, project, zoom."""
    _console.print(
        '  [dim]Needs pip install "mapcv\\[gee]" and a one-time `earthengine authenticate`.'
        " Requests run on your Earth Engine account and Cloud project.[/dim]"
    )
    for name, preset in _EE_PRESETS.items():
        _console.print(f"  [bold]{name:<10}[/bold] {preset['about']}")
    _console.print(
        "  [bold]custom[/bold]     any image or collection from the Earth Engine catalog"
    )
    dataset = Prompt.ask(
        "Dataset", choices=[*_EE_PRESETS, "custom"], default="sentinel2", console=_console
    )
    last_year = datetime.datetime.now(tz=datetime.timezone.utc).year - 1
    lines = ["  type: xyz", "ZOOM", "  max_connections: 4", "  earth_engine:"]
    metres = 10.0
    if dataset == "custom":
        kind = Prompt.ask(
            "Is it one image or a collection of scenes?",
            choices=["image", "collection"],
            default="collection",
            console=_console,
        )
        asset = Prompt.ask(
            "Asset ID [dim](e.g. COPERNICUS/S2_SR_HARMONIZED)[/dim]", console=_console
        )
        lines.append(f"    {kind}: {_yaml_str(asset.strip())}")
        if kind == "collection":
            lines.append(
                f'    start: "{Prompt.ask("Start date", default=f"{last_year}-06-01", console=_console)}"'
            )
            lines.append(
                f'    end: "{Prompt.ask("End date", default=f"{last_year}-09-01", console=_console)}"'
            )
            reducer = Prompt.ask(
                "Combine the scenes with",
                choices=["median", "mosaic", "mean", "min", "max"],
                default="median",
                console=_console,
            )
            lines.append(f"    reducer: {reducer}")
        bands = Prompt.ask("Bands to show [dim](1 or 3, comma-separated)[/dim]", console=_console)
        low = Prompt.ask("Value shown as black", default="0", console=_console)
        high = Prompt.ask("Value shown as white", default="3000", console=_console)
        band_list = ", ".join(b.strip() for b in bands.split(",") if b.strip())
        lines.append(f"    vis: {{bands: [{band_list}], min: {low}, max: {high}}}")
        metres = float(Prompt.ask("Its pixel size in metres", default="10", console=_console))
    else:
        preset = _EE_PRESETS[dataset]
        metres = preset["metres"]
        lines.append(f"    collection: {preset['collection']}")
        if dataset == "naip":
            start = Prompt.ask("From", default=f"{last_year - 2}-01-01", console=_console)
            end = Prompt.ask("To", default=f"{last_year + 1}-01-01", console=_console)
        else:
            start = Prompt.ask("Start date", default=f"{last_year}-06-01", console=_console)
            end = Prompt.ask("End date", default=f"{last_year}-09-01", console=_console)
        lines += [f'    start: "{start}"', f'    end: "{end}"']
        if preset["max_cloud"] is not None:
            cloud = IntPrompt.ask(
                "Skip scenes with more cloud than (%)",
                default=preset["max_cloud"],
                console=_console,
            )
            lines.append(f"    max_cloud: {cloud}")
        if preset["cloud_score_plus"]:
            lines.append("    cloud_score_plus: 0.6    # mask the cloudy pixels left")
        lines += [f"    reducer: {preset['reducer']}", f"    vis: {preset['vis']}"]
    project = Prompt.ask(
        "Cloud project with Earth Engine enabled [dim](Enter to fill in later)[/dim]",
        default="",
        show_default=False,
        console=_console,
    ).strip()
    lines.append(f"    project: {_yaml_str(project) if project else 'YOUR-CLOUD-PROJECT'}")
    table = Table(box=None, padding=(0, 2), show_edge=False)
    table.add_column("zoom", justify="right")
    table.add_column("pixel size")
    suggested = _zoom_for(metres, latitude)
    for zoom in range(max(1, suggested - 2), min(22, suggested + 2) + 1):
        table.add_row(str(zoom), f"{ground_resolution_m(zoom, latitude):.2f} m")
    _console.print(table)
    zoom = IntPrompt.ask(
        f"Zoom [dim](the data is {metres:g} m)[/dim]", default=suggested, console=_console
    )
    return [f"  zoom: {zoom}" if line == "ZOOM" else line for line in lines]


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
    _console.print("  [bold]esri[/bold]       Esri World Imagery — sub-metre RGB")
    _console.print("  [bold]sentinel2[/bold]  Sentinel-2 L2A — open 10 m multispectral (EOPF Zarr)")
    _console.print("  [bold]custom[/bold]     your own XYZ tile URL")
    _console.print("  [bold]geotiff[/bold]    your own GeoTIFF / COG file or URL")
    _console.print(
        "  [bold]gee[/bold]        Google Earth Engine — Sentinel-2, Landsat, NAIP or any asset"
    )
    kind = Prompt.ask(
        "Imagery",
        choices=["esri", "sentinel2", "custom", "geotiff", "gee"],
        default="esri",
        console=_console,
    )

    geotiff: _GeoTiffAnswer | None = _ask_geotiff() if kind == "geotiff" else None

    _console.print("\n[bold cyan]2/4 Area[/bold cyan]")
    (west, south, east, north), area_file, area_layer = _ask_bbox_or_file(
        geotiff.extent if geotiff is not None else None
    )
    latitude = (south + north) / 2

    imagery_lines: list[str]
    if geotiff is not None:
        imagery_lines = geotiff.imagery_lines
        patch_default, image_format, edge = 256, geotiff.image_format, "drop"
    elif kind == "gee":
        imagery_lines = _ask_earth_engine(latitude)
        patch_default, image_format, edge = 256, "png", "pad"
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
    labels_path: Path | None = None
    labels_layer: str | None = None
    raster_lines: list[str] = []
    if area_file is not None and Confirm.ask(
        f"Use {area_file.name} as the labels too?", default=True, console=_console
    ):
        labels_path = area_file
        labels_layer = area_layer
    else:  # no area file, or labels from another file
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
    label_lines: list[str] = raster_lines
    task_lines: list[str] = []
    detection_lines: list[str] = []
    if labels_path is not None:
        field = _ask_label_field(labels_path, labels_layer) if labels_path.exists() else None
        label_lines = ["labels:", f"  path: {_yaml_str(str(labels_path))}"]
        if labels_layer is not None:
            label_lines.append(f"  layer: {_yaml_str(labels_layer)}")
        label_lines.append(f"  label_field: {field}" if field else "  label_field: null")
        _console.print(
            "  [bold]segmentation[/bold]    a class mask per patch\n"
            "  [bold]detection[/bold]       a box per object (COCO and YOLO)\n"
            "  [bold]instance[/bold]        a mask per object (COCO RLE, optional instance-ID PNG)\n"
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
    # Descriptions start in one column; each line fits 80 columns.
    epilog=(
        "Examples:\n\n"
        "  [cyan]mapcv init[/cyan]                                guided, writes mapcv.yaml\n\n"
        "  [cyan]mapcv init --template xyz --stdout[/cyan]        print a template\n\n"
        "  [cyan]mapcv init my.yaml -t sentinel2[/cyan]           a ready-made example\n\n"
        "  [cyan]mapcv init gee.yaml -t earth-engine[/cyan]       Sentinel-2 from Earth Engine\n\n"
        "  [cyan]mapcv init boxes.yaml -t detection[/cyan]        boxes for COCO and YOLO\n\n"
        "  [cyan]mapcv init masks.yaml -t instance[/cyan]         a mask per object (COCO RLE)\n\n"
        "  [cyan]mapcv init tiles.yaml -t classification[/cyan]   a label per patch (CSV)\n\n"
        "  [cyan]mapcv init pairs.yaml -t change[/cyan]           before/after pairs and change masks\n\n"
        "  [cyan]mapcv init heights.yaml -t regression[/cyan]     float targets from a raster"
    ),
)
def init(
    output: Path = typer.Argument(
        Path("mapcv.yaml"), metavar="OUTPUT", help="Where to write the config."
    ),
    template: Template | None = typer.Option(
        None,
        "--template",
        "-t",
        metavar="NAME",
        help="Write a ready-made example instead of asking: "
        + ", ".join(item.value for item in list(Template)[:-1])
        + f" or {list(Template)[-1].value}.",
    ),
    interactive: bool | None = typer.Option(
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
    exists = f"[red]{escape(str(target))} already exists.[/red]"
    overwrite = "Add [bold]--force[/bold] to overwrite it, or name another file."
    if target.is_dir():
        _fail(f"[red]{escape(str(target))} is a folder.[/red]", "Name the YAML file to write.")
    try:
        # Asked before the questions, so a "no" doesn't throw the answers away.
        if (
            target.exists()
            and not force
            and not (
                guided
                and Confirm.ask(f"{target} exists. Overwrite it?", default=False, console=_console)
            )
        ):
            _fail(exists, overwrite)
        text = _wizard() if guided else _TEMPLATES[template or Template.xyz]
    except KeyboardInterrupt:
        _fail("\n[yellow]Interrupted.[/yellow] Nothing was written.", code=130)
    except EOFError:
        _fail(
            "\n[red]The input ended before the last question.[/red] Nothing was written.",
            "Without a terminal, start from a template: [bold]mapcv init --template xyz[/bold].",
        )
    try:
        target.write_text(text, encoding="utf-8")
    except OSError as exc:
        _debug_traceback(exc)
        _fail(
            f"[red]Cannot write[/red] {escape(str(target))}: {escape(exc.strerror or str(exc))}.",
            "Check that its folder exists and that you can write to it.",
        )
    try:
        MapcvConfig.from_yaml(target)
    except ValidationError as exc:
        _debug_traceback(exc)
        _console.print(f"[yellow]⚠[/yellow]  Wrote {escape(str(target))}, but it needs edits:")
        _print_items(_console, _validation_items(exc))
        return
    _console.print(f"\n[green]✓[/green] Wrote [bold]{escape(str(target))}[/bold]", soft_wrap=True)
    path = escape(_shell_path(target))
    # Not wrapped: a URL broken over two lines can't be clicked or copied.
    _console.print(
        "\n[bold]Next[/bold]\n"
        f"  1. See what it will cost:  [cyan]mapcv plan {path}[/cyan]\n"
        f"  2. Build the dataset:      [cyan]mapcv generate {path}[/cyan]\n"
        f"[dim]Imagery terms: {_PROVIDERS_URL}[/dim]",
        soft_wrap=True,
    )


def _fit_title(text: str) -> str:
    """A panel title that fits the terminal: a long path loses its start, not its end."""
    room = max(_console.width - 6, 12)
    return text if len(text) <= room else "…" + text[-(room - 1) :]


def _require_dataset(staging_dir: Path) -> None:
    """Exit with the same message from every dataset command when there is no dataset."""
    manifest = staging_dir / "manifest.json"
    if not manifest.is_file():
        _fail(
            f"[red]No manifest found at[/red] {escape(str(manifest))}",
            "Pass the dataset folder that [bold]mapcv generate[/bold] wrote "
            "(the config's writer.staging_dir).",
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
    estimate = _make_plan(config)
    _print_plan(config_path, config, estimate)
    _console.print(
        "\nLooks right? Build it with "
        f"[cyan]mapcv generate {escape(_shell_path(config_path))}[/cyan]",
        soft_wrap=True,
    )


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
    estimate = _make_plan(config)
    _print_plan(config_path, config, estimate)
    if dry_run:
        return
    if estimate.is_large and not yes:
        if not sys.stdin.isatty():
            _fail(
                "[red]This is a large job.[/red] Re-run with [bold]--yes[/bold] to confirm.",
                code=2,
            )
        if not Confirm.ask("This is a large job. Start it?", default=False, console=_console):
            _fail("Not started.", "Re-run with [bold]--yes[/bold] to start without asking.")
    shown = set(estimate.warnings)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        try:
            with _GenerateFeedback() as feedback:
                result = run_generate(config, feedback)
        except ManifestMismatchError as exc:
            _debug_traceback(exc)
            _show_warnings(caught, shown)
            _fail(f"[red]Cannot resume:[/red] {escape(str(exc))}")
        except KeyboardInterrupt:
            _fail(
                "\n[yellow]Interrupted.[/yellow] Finished chunks are saved; run the same "
                "command again to resume.",
                code=130,
            )
        except Exception as exc:  # noqa: BLE001 - any failure gets the same resume advice
            _debug_traceback(exc)
            _show_warnings(caught, shown)
            hint = None
            if isinstance(exc, OSError) and exc.errno is not None:
                detail, hint = _generate_os_error(exc)
            else:
                detail = str(exc) or type(exc).__name__
            resume = "run the same command again: finished chunks are kept and the run resumes."
            _fail(
                f"[red]Generation failed:[/red] {escape(detail)}",
                f"{hint} Then {resume}" if hint else f"Fix the cause and {resume}",
            )
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
    _require_dataset(staging_dir)
    try:
        manifest = Manifest.load(staging_dir / "manifest.json")
    except (ValueError, OSError) as exc:  # ManifestMismatchError and bad JSON are ValueErrors
        _debug_traceback(exc)
        _fail(f"[red]Cannot read the dataset:[/red] {escape(str(exc))}")
    table = Table.grid(padding=(0, 2))
    table.add_column(style="bold cyan", no_wrap=True)
    table.add_column()
    target = manifest.target
    task = manifest.task if target is not None else f"{manifest.task} · image only (no labels)"
    table.add_row("Task", task)
    source = manifest.source
    if len(manifest.sources) > 1:
        for record in manifest.sources:
            bands = f" · bands {', '.join(record.bands)}" if record.bands else ""
            table.add_row(f"Source {record.name}", _source_line(record) + bands)
        table.add_row("Patches", f"{len(manifest.patches):,} per source")
    else:
        table.add_row("Source", f"{source.source_type} · {source.product_id or 'unknown product'}")
        if source.bands:
            table.add_row("Bands", ", ".join(source.bands))
        shape = "×".join(str(dim) for dim in source.patch_shape) or "?"
        table.add_row(
            "Patches", f"{len(manifest.patches):,} · {shape} {source.dtype or ''}".strip()
        )
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
        touch = "touches" if padded == 1 else "touch"
        table.add_row("Padded", f"{_plural(padded, 'patch', 'patches')} {touch} the raster edge")
    splits_dir = staging_dir / "splits"
    if splits_dir.is_dir():
        counts = {}
        for name in ("train", "val", "test"):
            path = splits_dir / f"{name}.txt"
            text = path.read_text(encoding="utf-8").strip() if path.exists() else ""
            counts[name] = len(text.splitlines()) if text else 0
        table.add_row("Splits", _split_line(counts))
    version = f"version {manifest.loaded_version}"
    if manifest.upgraded_from is not None:
        version += f" (mapcv 0.{manifest.upgraded_from}; read as version {manifest.version})"
    table.add_row("Manifest", version)
    _console.print(
        Panel(
            table,
            title=f"[bold]{escape(_fit_title(str(staging_dir)))}[/bold]",
            title_align="left",
            border_style="cyan",
        )
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
    labeled_ratios: list[float] | None = typer.Option(
        None,
        help="Labeled fractions of train for semi-supervised lists (repeatable). "
        "Default: 0.1 0.2 0.3.",
    ),
    seed: int = typer.Option(42, help="Random seed."),
    strategy: str = typer.Option(
        "spatial",
        help="spatial (leakage-safe blocks) | stratified | random | region (whole regions "
        "of an area of interest).",
    ),
    block_size: int | None = typer.Option(
        None, help="Spatial block size in pixels (default: 4 × patch size)."
    ),
    sample_limit: int | None = typer.Option(None, help="Use at most this many patches."),
) -> None:
    """Re-split an existing dataset from its manifest; no images are read."""
    ratios = labeled_ratios if labeled_ratios is not None else [0.10, 0.20, 0.30]
    try:
        cfg = SplitterConfig(
            test_ratio=test_ratio,
            val_ratio=val_ratio,
            labeled_ratios=ratios,
            seed=seed,
            strategy=cast(Literal["spatial", "stratified", "random", "region"], strategy),
            block_size=block_size,
            sample_limit=sample_limit,
        )
    except ValidationError as exc:
        _debug_traceback(exc)
        option, message = _validation_items(exc, options=True)[0]
        raise typer.BadParameter(message, param_hint=f"'{option}'") from None
    _require_dataset(staging_dir)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        try:
            counts = run_split(staging_dir, cfg)
        except (OSError, ValueError) as exc:  # ManifestMismatchError is a ValueError
            _debug_traceback(exc)
            _fail(f"[red]Cannot split the dataset:[/red] {escape(str(exc))}")
    _show_warnings(caught, set())
    _console.print(
        f"[green]✓[/green] Splits written to [bold]{escape(str(staging_dir / 'splits'))}[/bold]: "
        f"{_split_line(counts)}",
        soft_wrap=True,
    )


@app.command(
    rich_help_panel="2. Use a dataset",
    epilog=(
        "Examples:\n\n"
        "  [cyan]mapcv stats dataset/[/cyan]              train split (all patches without splits)\n\n"
        "  [cyan]mapcv stats dataset/ --split all[/cyan]"
    ),
)
def stats(
    staging_dir: Path = typer.Argument(
        ..., metavar="STAGING_DIR", help="Dataset directory containing manifest.json."
    ),
    split_name: str = typer.Option(
        "train", "--split", help="train, val, test or all: the patches to count."
    ),
) -> None:
    """Per-band mean and std, class balance and class weights, written to stats.json."""
    from mapcv.stats import write_stats

    if split_name not in ("train", "val", "test", "all"):
        raise typer.BadParameter("must be train, val, test or all.", param_hint="'--split'")
    _require_dataset(staging_dir)
    try:
        path, values = write_stats(staging_dir, split_name)
    except (OSError, ValueError) as exc:  # ManifestMismatchError is a ValueError
        _debug_traceback(exc)
        _fail(f"[red]Cannot compute the statistics:[/red] {escape(str(exc))}")
    table = Table(box=None, padding=(0, 2), show_edge=False)
    for column in ("source", "band", "mean", "std", "min", "max"):
        table.add_column(column, justify="left" if column in ("source", "band") else "right")
    for source, summary in values["sources"].items():
        for band, mean, std, low, high in zip(
            summary["bands"], summary["mean"], summary["std"], summary["min"], summary["max"]
        ):
            cells = [f"{v:.6g}" if v is not None else "-" for v in (mean, std, low, high)]
            table.add_row(source, band, *cells)
    _console.print(table)
    weights = (values.get("classes") or {}).get("median_frequency_weights")
    if weights:
        _console.print(
            "Class weights (median frequency): "
            + ", ".join(f"{name} {weight:.3g}" for name, weight in weights.items())
        )
    counted = _plural(values["patches"], "patch", "patches")
    _console.print(
        f"[green]✓[/green] {counted} of the [bold]{values['split']}[/bold] split → "
        f"[bold]{escape(str(path))}[/bold]",
        soft_wrap=True,
    )


@app.command(
    rich_help_panel="2. Use a dataset",
    epilog="Example: [cyan]mapcv card dataset/[/cyan]",
)
def card(
    staging_dir: Path = typer.Argument(
        ..., metavar="STAGING_DIR", help="Dataset directory containing manifest.json."
    ),
    force: bool = typer.Option(False, "--force", help="Replace an existing README.md."),
) -> None:
    """Write a dataset card (README.md with Hugging Face metadata) for sharing."""
    from mapcv.card import write_card

    _require_dataset(staging_dir)
    try:
        path = write_card(staging_dir, overwrite=force)
    except FileExistsError as exc:
        _debug_traceback(exc)
        _fail(f"[red]{escape(str(exc))}[/red]")
    except (OSError, ValueError) as exc:  # ManifestMismatchError is a ValueError
        _debug_traceback(exc)
        _fail(f"[red]Cannot write the dataset card:[/red] {escape(str(exc))}")
    _console.print(
        f"[green]✓[/green] Dataset card written to [bold]{escape(str(path))}[/bold]. Its licence "
        "is 'other' until you set it.",
        soft_wrap=True,
    )


@app.command(
    rich_help_panel="2. Use a dataset",
    epilog=(
        "Examples:\n\n"
        "  [cyan]mapcv verify dataset/[/cyan]\n\n"
        "  [cyan]mapcv verify dataset/ --deep --write-checksums[/cyan]"
    ),
)
def verify(
    staging_dir: Path = typer.Argument(
        ..., metavar="STAGING_DIR", help="Dataset directory containing manifest.json."
    ),
    deep: bool = typer.Option(False, "--deep", help="Also decode every image and check its shape."),
    write_checksums_: bool = typer.Option(
        False, "--write-checksums", help="Write SHA256SUMS after a successful check."
    ),
) -> None:
    """Check that every file is present and intact (and SHA256SUMS, when there is one)."""
    from mapcv.verify import CHECKSUMS_FILENAME, verify_dataset, write_checksums

    _require_dataset(staging_dir)
    report = verify_dataset(staging_dir, deep=deep)
    for note in report.notes:
        _console.print(f"[yellow]⚠[/yellow]  {escape(note)}")
    if not report.ok:
        for problem in report.problems[:20]:
            _console.print(f"[red]✗[/red] {escape(problem)}")
        if len(report.problems) > 20:
            _console.print(f"[red]… and {len(report.problems) - 20:,} more[/red]")
        _fail(
            f"[red]{_plural(len(report.problems), 'problem')} found.[/red]",
            "Copy the dataset again from where it came from, or generate it into a new folder.",
        )
    hashes = (
        f", {_plural(report.checked_hashes, 'hash', 'hashes')} match"
        if report.checked_hashes
        else ""
    )
    _console.print(
        f"[green]✓[/green] {_plural(report.patches, 'patch', 'patches')}, "
        f"{_plural(report.files, 'file')} present{hashes}."
    )
    if write_checksums_:
        try:
            path = write_checksums(staging_dir)
        except OSError as exc:
            _debug_traceback(exc)
            _fail(f"[red]Cannot write {CHECKSUMS_FILENAME}:[/red] {escape(str(exc))}")
        _console.print(
            f"[green]✓[/green] {CHECKSUMS_FILENAME} written to [bold]{escape(str(path))}[/bold]",
            soft_wrap=True,
        )


@app.command(
    rich_help_panel="3. Utilities", epilog="Example: [cyan]mapcv validate mapcv.yaml[/cyan]"
)
def validate(
    config_path: Path = typer.Argument(..., metavar="CONFIG_PATH", help="Path to the YAML config."),
) -> None:
    """Check a config without reading labels or imagery (use [bold]plan[/bold] for estimates)."""
    config = _load_config(config_path)
    _console.print(
        f"[green]✓[/green] {escape(str(config_path))} is a valid config.", soft_wrap=True
    )
    _console.print(_settings_table(config))
    labels = config.labels
    if isinstance(labels, RASTER_LABEL_TYPES):
        label_file = eopf_local_path(labels.path)
        if label_file is not None and not label_file.exists():
            _console.print(f"[yellow]⚠[/yellow]  labels.path not found: {label_file}")
    elif labels is not None:
        for key, path in labels.keyed_files():
            if not path.exists():
                _console.print(f"[yellow]⚠[/yellow]  {key} not found: {path}")
    area = labels.annotated_area if isinstance(labels, LabelsConfig) else None
    if area is not None and not area.exists():
        _console.print(f"[yellow]⚠[/yellow]  labels.annotated_area not found: {area}")
    change = config.change
    if change is not None:
        for key, label_set in (
            ("change.before.path", change.before),
            ("change.after.path", change.after),
        ):
            if label_set is not None:
                prefix = key.rsplit(".", 1)[0]
                for file_key, path in label_set.keyed_files(prefix):
                    if not path.exists():
                        _console.print(f"[yellow]⚠[/yellow]  {file_key} not found: {path}")
                set_area = label_set.annotated_area
                if set_area is not None and not set_area.exists():
                    _console.print(
                        f"[yellow]⚠[/yellow]  {prefix}.annotated_area not found: {set_area}"
                    )
    for name, imagery in zip(config.source_names, config.sources):
        if isinstance(imagery, GeoTiffImageryConfig):
            local = eopf_local_path(imagery.path)
            if imagery.is_pattern:
                # A mosaic pattern: missing only when it matches no file.
                local = None if glob.glob(str(local), recursive=True) else local
            if local is not None and not local.exists():
                where = f"imagery '{name}' path" if config.multi_source else "imagery.path"
                _console.print(f"[yellow]⚠[/yellow]  {where} not found: {local}")


@app.command(
    rich_help_panel="2. Use a dataset",
    epilog=(
        "Examples:\n\n"
        "  [cyan]mapcv export dataset/ --format hf-parquet --out dataset-hf/[/cyan]\n\n"
        "  [cyan]mapcv export dataset/ --format terratorch[/cyan]   writes dataset/terratorch.yaml"
    ),
)
def export(
    staging_dir: Path = typer.Argument(
        ..., metavar="STAGING_DIR", help="Dataset directory containing manifest.json."
    ),
    format_: str = typer.Option(
        ...,
        "--format",
        "-f",
        help="hf-parquet (Hugging Face), webdataset (tar shards), zarr (one store) or "
        "terratorch (a data config).",
    ),
    out: Path | None = typer.Option(
        None,
        "--out",
        "-o",
        help="hf-parquet, webdataset, zarr: a new folder (required). terratorch: the YAML "
        "file (default STAGING_DIR/terratorch.yaml).",
    ),
    shard_mb: int = typer.Option(
        1000, "--shard-mb", min=1, help="webdataset: largest shard size in MB."
    ),
) -> None:
    """Export a dataset: Hugging Face Parquet, WebDataset shards, Zarr or a TerraTorch config."""
    from mapcv.export import FORMATS, export_hf_parquet, export_terratorch
    from mapcv.shards import export_webdataset, export_zarr

    if format_ not in FORMATS:
        raise typer.BadParameter(
            f"must be one of: {', '.join(FORMATS)}.", param_hint="'--format' / '-f'"
        )
    if format_ != "terratorch" and out is None:
        raise typer.BadParameter(
            f"required with --format {format_}: the folder to write.", param_hint="'--out' / '-o'"
        )
    _require_dataset(staging_dir)
    shown = escape(str(out))
    try:
        if format_ == "webdataset":
            assert out is not None
            shards = export_webdataset(staging_dir, out, shard_mb * 1_000_000)
            message = f"{_plural(len(shards), 'tar shard')} and shards.json in [bold]{shown}[/bold]"
        elif format_ == "zarr":
            assert out is not None
            export_zarr(staging_dir, out)
            message = (
                f"Zarr store written to [bold]{shown}[/bold]; read it with "
                f'mapcv.data.MapcvDataset("{shown}", split="train").'
            )
        elif format_ == "hf-parquet":
            assert out is not None
            written = export_hf_parquet(staging_dir, out)
            message = (
                f"{_plural(len(written), 'Parquet file')} and a dataset card in "
                f'[bold]{shown}[/bold]. Load them with datasets.load_dataset("{shown}").'
            )
        else:
            path = export_terratorch(staging_dir, out)
            message = (
                f"TerraTorch data config written to [bold]{escape(str(path))}[/bold]; "
                "paste it into your training config."
            )
    except (OSError, ValueError, RuntimeError) as exc:
        _debug_traceback(exc)
        _fail(f"[red]Cannot export the dataset:[/red] {escape(str(exc))}")
    _console.print(f"[green]✓[/green] {message}", soft_wrap=True)


@app.command(
    "cache",
    rich_help_panel="3. Utilities",
    epilog=(
        "Examples:\n\n"
        "  [cyan]mapcv cache[/cyan]                       where it is and how big\n\n"
        "  [cyan]mapcv cache --clear --expired[/cyan]     delete only expired tiles"
    ),
)
def cache_command(
    clear: bool = typer.Option(False, "--clear", help="Delete the cached tiles."),
    expired: bool = typer.Option(
        False, "--expired", help="With --clear: delete only the tiles that have expired."
    ),
) -> None:
    """Show or clear the on-disk cache of downloaded XYZ tiles."""
    from mapcv import tile_cache

    if expired and not clear:
        raise typer.BadParameter("only applies with --clear.", param_hint="'--expired'")
    if clear:
        removed = tile_cache.clear(expired_only=expired)
        what = "expired cached tile" if expired else "cached tile"
        _console.print(
            f"[green]✓[/green] Deleted {_plural(removed, what)} from "
            f"{escape(str(tile_cache.tiles_dir()))}",
            soft_wrap=True,
        )
        return
    found = tile_cache.usage()
    _console.print(f"Tile cache: [bold]{escape(str(found.path))}[/bold]", soft_wrap=True)
    _console.print(
        f"  {_plural(found.tiles, 'tile')}, {human_bytes(found.bytes)}"
        + (f", {found.expired:,} expired" if found.expired else "")
    )
    _console.print(
        f"[dim]Set {tile_cache.CACHE_ENV} to move it, or imagery.cache: false to skip it.[/dim]"
    )


@app.command(
    "mcp",
    rich_help_panel="3. Utilities",
    # Typer keeps single line breaks of a docstring, so the help is one line per paragraph.
    help=(
        "Run an MCP server over stdio so AI agents can build datasets with mapcv.\n\n"
        'Needs [bold]pip install "mapcv\\[mcp]"[/bold]. Without [bold]--allow-write[/bold] the '
        "server can only read, validate and plan."
    ),
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
    """Run an MCP server over stdio so AI agents can build datasets with mapcv."""
    if not root.is_dir():
        raise typer.BadParameter(f"{root} is not a folder.", param_hint="'--root'")
    try:
        from mapcv.mcp_server import serve
    except ImportError as exc:
        _debug_traceback(exc)
        _fail(
            "[red]The MCP server needs the optional 'mcp' extra.[/red]\n"
            'Install it with [bold]pip install "mapcv\\[mcp]"[/bold], then run this again.',
            f"{escape(str(exc))} (mapcv needs mcp 2.x)",
        )
    try:
        serve(root, allow_write)
    except ValueError as exc:
        _debug_traceback(exc)
        _fail(f"[red]{escape(str(exc))}[/red]")


_STATUS_STYLE = {"ok": "green", "warn": "yellow", "fail": "bold red", "info": "dim"}
_STATUS_LABEL = {"ok": "ok", "warn": "warn", "fail": "FAIL", "info": "info"}


def _print_doctor_report(checks: list[doctor.Check]) -> None:
    """The checks as one small table per section: a word for the status (it reads the same
    without colour), the name, the detail, and the hint dimmed under it."""
    first = True
    for section in doctor.SECTIONS:
        rows = [check for check in checks if check.section == section]
        if not rows:
            continue
        _console.print(("" if first else "\n") + f"[bold]{escape(section)}[/bold]")
        first = False
        table = Table.grid(padding=(0, 2))
        table.add_column(width=4, no_wrap=True)
        table.add_column(width=24)
        table.add_column(ratio=1, overflow="fold")
        for check in rows:
            style = _STATUS_STYLE[check.status]
            table.add_row(
                f"[{style}]{_STATUS_LABEL[check.status]}[/{style}]",
                escape(check.name),
                escape(check.detail),
            )
            if check.hint:
                table.add_row("", "", f"[dim]→ {escape(check.hint)}[/dim]")
        _console.print(table)
    failed = sum(check.status == "fail" for check in checks)
    warned = sum(check.status == "warn" for check in checks)
    _console.print()
    if failed:
        _console.print(
            f"[red]{_plural(failed, 'check')} failed.[/red] Fix them first; mapcv can't work "
            "until then."
        )
    elif warned:
        _console.print(
            f"[yellow]{_plural(warned, 'warning')}.[/yellow] mapcv works; see the hints above."
        )
    else:
        _console.print("[green]All checks passed.[/green]")
    _console.print(
        "[dim]Paste this output, or the output of mapcv doctor --json, into a bug report.[/dim]"
    )


@app.command(
    "doctor",
    rich_help_panel="3. Utilities",
    help=(
        "Diagnose the installation, extras, tile cache and network.\n\n"
        "Takes a few seconds, for you and for bug reports. Changes nothing (it writes and "
        "deletes one small probe file in the tile cache folder) and prints no secrets. Exits "
        "with 1 if a check fails, such as a broken Rust extension or a tile cache folder that "
        "can't be written; a warning does not fail the run."
    ),
    epilog=(
        "Examples:\n\n"
        "  [cyan]mapcv doctor[/cyan]                      check this installation\n\n"
        "  [cyan]mapcv doctor --offline[/cyan]            skip the network checks\n\n"
        "  [cyan]mapcv doctor --json[/cyan]               for bug reports and scripts"
    ),
)
def doctor_command(
    json_output: bool = typer.Option(
        False, "--json", help="Print the checks as JSON (no colour), for bug reports and scripts."
    ),
    offline: bool = typer.Option(
        False, "--offline", help="Skip the network checks (no request leaves this machine)."
    ),
) -> None:
    """Diagnose the installation, extras, tile cache and network."""
    terminal = doctor.TerminalInfo(
        encoding=_console.encoding,
        is_tty=_console.is_terminal,
        width=_console.width,
        color_system=_console.color_system,
        no_color_env=_NO_COLOR_FROM_ENV,
        no_color_flag=_console.no_color and not _NO_COLOR_FROM_ENV,
    )
    checks = doctor.run_checks(terminal, offline=offline)
    if json_output:
        sys.stdout.write(doctor.to_json(checks, offline=offline))
        sys.stdout.flush()
    else:
        _print_doctor_report(checks)
    if doctor.has_failure(checks):
        raise typer.Exit(code=1)
