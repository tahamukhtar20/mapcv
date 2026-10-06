"""The tools behind ``mapcv mcp``, without any dependency on the MCP SDK.

Every tool is a plain function from a :class:`ToolState` and typed arguments to a
:class:`ToolResult` (structured data plus a short summary), or a :class:`ToolFailure`
for a mistake the agent can correct. ``mapcv.mcp_server`` wires them to the protocol.

Safety rules enforced here, not in the protocol layer:

* **Root.** Every path argument, and every path a config resolves to (labels, local
  imagery, ``writer.staging_dir``), must lie inside the root after symlinks are resolved.
* **Read-only by default.** Tools that write refuse to run unless ``allow_write`` is set.
* **Credentials.** Whatever a ``url_template`` carries (user info, query, path tokens)
  is removed from every result and error, as the CLI does in its output.
"""

from __future__ import annotations

import inspect
import json
import os
import re
import tempfile
import threading
import warnings
from collections import Counter
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import (
    Any,
    Callable,
    Dict,
    Iterator,
    List,
    Literal,
    Optional,
    Set,
    Tuple,
    Union,
    cast,
    get_args,
)
from urllib.parse import urlsplit

import yaml
from pydantic import ValidationError
from shapely.geometry import shape

from mapcv import planning
from mapcv._mapcv_rs import parse_kml_rs
from mapcv.cli import _class_names, _imagery_label, _raster_labels, _redact_url, _task_label
from mapcv.config import (
    PLANNED_TASKS,
    SUPPORTED_TASKS,
    EOPFZarrImageryConfig,
    GeoTiffImageryConfig,
    MapcvConfig,
    RasterLabelsConfig,
    XYZImageryConfig,
    _resolve_relative_paths,
    eopf_local_path,
)
from mapcv.downloader import URL_TEMPLATES
from mapcv.labels import (
    MAX_CLASS_ID,
    VECTOR_LABEL_SUFFIXES,
    _check_geojson_crs,
    _normalize_label,
    parse_kml,
)
from mapcv.manifest import Manifest, ManifestMismatchError, patch_folders
from mapcv.pipeline import GenerateResult, run_generate, run_split
from mapcv.planning import Plan, human_bytes
from mapcv.planning import plan as make_plan
from mapcv.splitter import SplitterConfig
from mapcv.vector_files import read_geoparquet, read_gpkg, read_shapefile, shapefile_files
from mapcv.writer import WriterConfig
from mapcv.writers.detection import categories

__all__ = [
    "GenerationCancelled",
    "Redactor",
    "Sandbox",
    "ToolFailure",
    "ToolResult",
    "ToolState",
]

_PROVIDERS_URL = "https://github.com/tahamukhtar20/mapcv/blob/main/PROVIDERS.md"

#: Config files and label files larger than this are refused instead of read into memory.
MAX_CONFIG_BYTES = 1024 * 1024
MAX_LABEL_BYTES = 256 * 1024 * 1024
#: Seconds a quick tool waits for a running generation to release the warning capture.
_WARNING_LOCK_WAIT = 15.0


class ToolFailure(Exception):
    """A mistake the agent can fix: the message says what to change.

    ``data`` is structured detail for the agent (for example the validation errors).
    """

    def __init__(self, message: str, data: Optional[Dict[str, Any]] = None) -> None:
        super().__init__(message)
        self.message = message
        self.data = data or {}


class GenerationCancelled(Exception):
    """Raised inside a generation when the client cancelled the call."""


@dataclass
class ToolResult:
    """What a tool returns: structured data and a short sentence for people."""

    summary: str
    data: Dict[str, Any] = field(default_factory=dict)


# ── Credentials ──────────────────────────────────────────────────────────────

_TEMPLATE_IN_TEXT = re.compile(r"""https?://[^\s"'<>]*\{[xyz]\}[^\s"'<>]*""")
_URL_IN_TEXT = re.compile(r"""https?://[^\s"'<>)\]]+""")
_MIN_QUERY_SECRET = 3
_MIN_PATH_SECRET = 6


def _secret_tokens(url: str) -> Set[str]:
    """The parts of a tile URL template that may be credentials."""
    parts = urlsplit(url)
    tokens: Set[str] = set()
    for value in (parts.username, parts.password, parts.fragment):
        if value:
            tokens.add(value)
    for pair in parts.query.split("&"):
        _, _, value = pair.partition("=")
        if len(value) >= _MIN_QUERY_SECRET and "{" not in value:
            tokens.add(value)
    for segment in parts.path.split("/"):
        if len(segment) >= _MIN_PATH_SECRET and "{" not in segment:
            tokens.add(segment)
    return tokens


class Redactor:
    """Removes the credentials of every ``url_template`` the server has seen from text.

    The CLI shows only ``scheme://host/...`` for a template. Results here are built
    from fields that never carry a template; this is the second line of defence for
    text the server does not control: error messages and warnings of the libraries.
    """

    def __init__(self) -> None:
        self._templates: Set[str] = set()
        self._hosts: Set[str] = set()
        self._tokens: Set[str] = set()
        self._lock = threading.Lock()

    def learn_url(self, template: str) -> None:
        """Remember a tile URL template so it is hidden from later output."""
        if not template:
            return
        parts = urlsplit(template)
        with self._lock:
            self._templates.add(template)
            if parts.hostname:
                self._hosts.add(parts.hostname)
            self._tokens |= _secret_tokens(template)

    def learn_text(self, text: str) -> None:
        """Remember every tile URL template written in a piece of config text."""
        for match in _TEMPLATE_IN_TEXT.finditer(text):
            self.learn_url(match.group(0).rstrip(",;"))

    def learn_data(self, data: Any) -> None:
        """Remember the template of a parsed (maybe invalid) config mapping."""
        imagery = data.get("imagery") if isinstance(data, dict) else None
        template = imagery.get("url_template") if isinstance(imagery, dict) else None
        if isinstance(template, str):
            self.learn_url(template)

    def scrub(self, text: str) -> str:
        """``text`` without known templates, their secrets, or credentials in URLs."""
        with self._lock:
            templates = sorted(self._templates, key=len, reverse=True)
            hosts = set(self._hosts)
            tokens = sorted(self._tokens, key=len, reverse=True)
        for template in templates:
            text = text.replace(template, _redact_url(template))

        def url(match: "re.Match[str]") -> str:
            found = match.group(0)
            parts = urlsplit(found)
            if parts.hostname in hosts or parts.username or parts.password or parts.query:
                return _redact_url(found)
            return found

        text = _URL_IN_TEXT.sub(url, text)
        for token in tokens:
            text = text.replace(token, "***")
        return text

    def scrub_data(self, data: Any) -> Any:
        """:meth:`scrub` applied to every string inside nested lists and dicts."""
        if isinstance(data, str):
            return self.scrub(data)
        if isinstance(data, dict):
            return {key: self.scrub_data(value) for key, value in data.items()}
        if isinstance(data, (list, tuple)):
            return [self.scrub_data(value) for value in data]
        return data


# ── Where the tools may read and write ───────────────────────────────────────


class Sandbox:
    """The folder tools may use, and whether they may write there."""

    def __init__(self, root: Union[str, "os.PathLike[str]"], allow_write: bool = False) -> None:
        resolved = Path(root).expanduser().resolve()
        if not resolved.is_dir():
            raise ValueError(f"--root {root} is not a folder")
        self.root = resolved
        self.allow_write = allow_write

    def resolve(self, value: str, what: str = "path") -> Path:
        """An absolute path inside the root; relative paths are relative to the root."""
        if not isinstance(value, str) or not value.strip() or "\x00" in value:
            raise ToolFailure(f"`{what}` must be a path inside {self.root}")
        candidate = Path(value)
        if not candidate.is_absolute():
            candidate = self.root / candidate
        try:
            resolved = candidate.resolve()
        except (OSError, RuntimeError, ValueError) as exc:
            raise ToolFailure(f"`{what}` {value!r} cannot be resolved: {exc}") from None
        return self.inside(resolved, what, value)

    def inside(self, resolved: Path, what: str, shown: Optional[str] = None) -> Path:
        """``resolved`` (symlinks already followed) if it is inside the root, else a failure."""
        try:
            resolved.relative_to(self.root)
        except ValueError:
            raise ToolFailure(
                f"`{what}` {shown or str(resolved)!r} is outside the folder this server may use "
                f"({self.root}). Use a path inside it; if the folder is wrong, restart the "
                "server with --root."
            ) from None
        return resolved

    def rel(self, path: Union[str, Path]) -> str:
        """``path`` relative to the root with ``/`` separators; ``.`` for the root itself."""
        try:
            relative = Path(path).resolve().relative_to(self.root)
        except (ValueError, OSError):
            return str(path)
        return relative.as_posix()

    def require_write(self, tool: str) -> None:
        """Fail unless the server was started with ``--allow-write``."""
        if not self.allow_write:
            raise ToolFailure(
                f"`{tool}` writes files, and this server is read-only. Ask the user to restart "
                "it with `mapcv mcp --allow-write`."
            )

    def check_config_paths(self, config: MapcvConfig) -> None:
        """Fail if a path the config resolves to (not a remote URL) leaves the root."""
        for what, path in config_paths(config):
            # What mapcv will open: a path still relative here is relative to the process.
            target = path if path.is_absolute() else Path.cwd() / path
            try:
                resolved = target.resolve()
            except (OSError, RuntimeError, ValueError) as exc:
                raise ToolFailure(f"{what} {str(path)!r} cannot be resolved: {exc}") from None
            self.inside(resolved, what, str(path))

    def check_tree(self, directory: Path, what: str) -> None:
        """Fail if something below an existing ``directory`` is a link that leaves the root.

        A symlink inside an output folder would let a write land elsewhere.
        """
        if not directory.is_dir():
            return
        stack = [directory]
        while stack:
            current = stack.pop()
            try:
                with os.scandir(current) as entries:
                    for entry in entries:
                        if entry.is_symlink():
                            self.inside(Path(entry.path).resolve(), what, entry.path)
                        elif entry.is_dir(follow_symlinks=False):
                            stack.append(Path(entry.path))
            except OSError:
                continue


def config_paths(config: MapcvConfig) -> List[Tuple[str, Path]]:
    """The local paths a config reads or writes, named by their config key."""
    found: List[Tuple[str, Path]] = []
    labels = config.labels
    if isinstance(labels, RasterLabelsConfig):
        local = eopf_local_path(labels.path)
        if local is not None:
            found.append(("labels.path", local))
    elif labels is not None:
        found.append(("labels.path", labels.path))
        if labels.path.suffix.lower() == ".shp" and labels.path.is_file():
            found.extend(
                ("labels.path (sidecar)", file) for file in shapefile_files(labels.path)[1:]
            )
    imagery = config.imagery
    if isinstance(imagery, (EOPFZarrImageryConfig, GeoTiffImageryConfig)):
        local = eopf_local_path(imagery.path)
        if local is not None:
            found.append(("imagery.path", local))
    found.append(("writer.staging_dir", config.writer.staging_dir))
    return found


# ── Shared state ─────────────────────────────────────────────────────────────

_WARNINGS_LOCK = threading.Lock()


@contextmanager
def capture_warnings(wait: float = _WARNING_LOCK_WAIT) -> Iterator[List[warnings.WarningMessage]]:
    """Record the warnings raised inside the block.

    ``warnings.catch_warnings`` changes process-wide state, so only one capture runs
    at a time; a tool that finds a generation running waits, then gives up with a message.
    """
    if not _WARNINGS_LOCK.acquire(timeout=wait):
        raise ToolFailure(
            "Another long operation (probably a running `generate`) is busy; try again when "
            "it has finished."
        )
    try:
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            yield caught
    finally:
        _WARNINGS_LOCK.release()


class _Jobs:
    """The staging folders a generation is writing to right now."""

    def __init__(self) -> None:
        self._active: Set[Path] = set()
        self._lock = threading.Lock()

    def acquire(self, staging: Path) -> None:
        with self._lock:
            if staging in self._active:
                raise ToolFailure(
                    f"A generation into {staging.name}/ is still running (or still stopping "
                    "after a cancel). Wait for it to finish before starting another."
                )
            self._active.add(staging)

    def release(self, staging: Path) -> None:
        with self._lock:
            self._active.discard(staging)


@dataclass
class ToolState:
    """Everything the tools share for the life of one server."""

    sandbox: Sandbox
    redactor: Redactor = field(default_factory=Redactor)
    jobs: _Jobs = field(default_factory=_Jobs)


def _warning_texts(caught: List[warnings.WarningMessage]) -> List[str]:
    texts: List[str] = []
    for warning in caught:
        message = str(warning.message)
        if message not in texts:
            texts.append(message)
    return texts


# ── Config loading ───────────────────────────────────────────────────────────


def format_validation_errors(exc: ValidationError) -> List[Dict[str, str]]:
    """Validation problems as ``{field, message}``: the ones ``mapcv validate`` lists."""
    errors: List[Dict[str, str]] = []
    for error in exc.errors():
        location = ".".join(
            str(part)
            for part in error["loc"]
            if not str(part).startswith("function-") and part not in ("xyz", "eopf_zarr", "geotiff")
        )
        message = str(error["msg"]).removeprefix("Value error, ")
        errors.append({"field": location or "config", "message": message})
    return errors


def _yaml_problem(exc: yaml.YAMLError) -> str:
    """A YAML error without the source snippet PyYAML adds (it could hold a template)."""
    mark = getattr(exc, "problem_mark", None)
    problem = getattr(exc, "problem", None) or "invalid YAML"
    where = f"line {mark.line + 1}, column {mark.column + 1}: " if mark is not None else ""
    return f"not valid YAML ({where}{problem})"


class ConfigInvalid(Exception):
    """A config that does not validate: ``errors`` are the problems, ``message`` the headline."""

    def __init__(self, message: str, errors: List[Dict[str, str]]) -> None:
        super().__init__(message)
        self.message = message
        self.errors = errors


def parse_config_text(state: ToolState, text: str, base: Path) -> MapcvConfig:
    """Validate config text; relative paths in it resolve against ``base`` as for a file."""
    state.redactor.learn_text(text)
    if len(text.encode("utf-8")) > MAX_CONFIG_BYTES:
        raise ToolFailure(f"The config is larger than {MAX_CONFIG_BYTES // 1024} KiB.")
    try:
        data = yaml.safe_load(text)
    except yaml.YAMLError as exc:
        problem = _yaml_problem(exc)
        raise ConfigInvalid(problem, [{"field": "config", "message": problem}]) from None
    state.redactor.learn_data(data)
    if not isinstance(data, dict):
        message = "the config must be a YAML mapping with region, imagery, sampler and writer"
        raise ConfigInvalid(message, [{"field": "config", "message": message}])
    _resolve_relative_paths(data, base)
    try:
        config = MapcvConfig.model_validate(data)
    except ValidationError as exc:
        errors = format_validation_errors(exc)
        raise ConfigInvalid("the config has errors", errors) from None
    imagery = config.imagery
    if isinstance(imagery, XYZImageryConfig) and imagery.url_template:
        state.redactor.learn_url(imagery.url_template)
    state.sandbox.check_config_paths(config)
    return config


def load_config(
    state: ToolState, config: Optional[str], yaml_text: Optional[str], arg: str = "config"
) -> Tuple[MapcvConfig, Optional[Path]]:
    """The config named by a path inside the root, or given as text (paths relative to the root)."""
    sandbox = state.sandbox
    if (config is None) == (yaml_text is None):
        raise ToolFailure(f"Give exactly one of `{arg}` (a path to a YAML file) or `yaml_text`.")
    if yaml_text is not None:
        return parse_config_text(state, yaml_text, sandbox.root), None
    assert config is not None
    path = sandbox.resolve(config, arg)
    if not path.is_file():
        raise ToolFailure(
            f"Config file not found: {sandbox.rel(path)}. Create one with `write_config`."
        )
    if path.stat().st_size > MAX_CONFIG_BYTES:
        raise ToolFailure(f"{sandbox.rel(path)} is larger than {MAX_CONFIG_BYTES // 1024} KiB.")
    try:
        text = path.read_text(encoding="utf-8")
    except (OSError, UnicodeDecodeError) as exc:
        raise ToolFailure(f"Cannot read {sandbox.rel(path)}: {exc}") from None
    return parse_config_text(state, text, path.parent), path


def _invalid_failure(exc: ConfigInvalid, where: str) -> ToolFailure:
    lines = "\n".join(f"  - {e['field']}: {e['message']}" for e in exc.errors)
    return ToolFailure(
        f"{where} is not a valid config:\n{lines}",
        {"valid": False, "errors": exc.errors},
    )


# ── describe_config_schema ───────────────────────────────────────────────────

_SCHEMA_CACHE: Dict[str, Any] = {}


def _base_config() -> Dict[str, Any]:
    return {
        "region": {"west": 10.0, "south": 50.0, "east": 10.1, "north": 50.1},
        "imagery": {"type": "xyz", "zoom": 15, "source": sorted(URL_TEMPLATES)[0]},
        "sampler": {"patch_size": 256},
        "writer": {"staging_dir": "dataset"},
    }


def _probe(config: Dict[str, Any]) -> Optional[str]:
    """``None`` if the config validates, else the first message ``mapcv validate`` gives."""
    try:
        MapcvConfig.model_validate(config)
    except ValidationError as exc:
        error = format_validation_errors(exc)[0]
        return f"{error['field']}: {error['message']}"
    return None


_PROBE_IMAGERY: Dict[str, Dict[str, Any]] = {
    "xyz": {"type": "xyz", "zoom": 15, "source": sorted(URL_TEMPLATES)[0]},
    "eopf_zarr": {"type": "eopf_zarr", "path": "S2.zarr"},
    "geotiff": {"type": "geotiff", "path": "ortho.tif"},
}
_PROBE_LABELS: Dict[str, Optional[Dict[str, Any]]] = {
    "none": None,
    "vector": {"type": "vector", "path": "labels.geojson"},
    "raster": {"type": "raster", "path": "landcover.tif", "classes": {1: 1}},
}


def _probed_rules() -> Dict[str, Any]:
    """Which combinations of task, labels, imagery and image format validate.

    Found by validating minimal configs with the real models, so the answer is
    whatever the validators say today and never a copy that can go stale.
    """
    invalid: List[Dict[str, str]] = []
    labels_per_task: Dict[str, List[str]] = {}
    for task in SUPPORTED_TASKS:
        labels_per_task[task] = []
        for kind, labels in _PROBE_LABELS.items():
            config = _base_config()
            config["task"] = task
            if labels is not None:
                config["labels"] = labels
            problem = _probe(config)
            if problem is None:
                labels_per_task[task].append(kind)
            else:
                invalid.append({"task": task, "labels": kind, "error": problem})
    formats = list(get_args(WriterConfig.model_fields["image_format"].annotation))
    formats_per_imagery: Dict[str, List[str]] = {}
    for kind, imagery in _PROBE_IMAGERY.items():
        formats_per_imagery[kind] = []
        for image_format in formats:
            config = _base_config()
            config["imagery"] = imagery
            config["writer"] = {"staging_dir": "dataset", "image_format": image_format}
            problem = _probe(config)
            if problem is None:
                formats_per_imagery[kind].append(image_format)
            else:
                invalid.append({"imagery": kind, "image_format": image_format, "error": problem})
    return {
        "tasks": list(SUPPORTED_TASKS),
        "planned_tasks_not_supported_yet": list(PLANNED_TASKS),
        "labels_allowed_per_task": labels_per_task,
        "task_options_block": [
            task for task in SUPPORTED_TASKS if task in MapcvConfig.model_fields
        ],
        "imagery_types": list(_PROBE_IMAGERY),
        "image_formats_per_imagery": formats_per_imagery,
        "xyz_sources": sorted(URL_TEMPLATES),
        "invalid_combinations": invalid,
        "relative_paths": inspect.getdoc(MapcvConfig.from_yaml),
    }


def _required_fields(schema: Dict[str, Any]) -> Dict[str, List[str]]:
    required = {"MapcvConfig": list(schema.get("required", []))}
    for name, definition in schema.get("$defs", {}).items():
        required[name] = list(definition.get("required", []))
    return required


def describe_config_schema(state: ToolState) -> ToolResult:
    """The JSON schema of the config, generated from the pydantic models, plus the rules."""
    if not _SCHEMA_CACHE:
        schema = MapcvConfig.model_json_schema()
        _SCHEMA_CACHE.update(
            schema=schema, required=_required_fields(schema), rules=_probed_rules()
        )
    data = dict(_SCHEMA_CACHE)
    data["notes"] = [
        "Unknown keys are errors. Region is a WGS-84 lon/lat box (west, south, east, north).",
        "Paths in a config file are relative to the file's folder.",
        "Imagery terms are the user's responsibility: " + _PROVIDERS_URL,
    ]
    rules = data["rules"]
    return ToolResult(
        f"Config schema: tasks {', '.join(rules['tasks'])}; imagery "
        f"{', '.join(rules['imagery_types'])}. `rules.invalid_combinations` lists what does "
        "not validate, with the exact message.",
        data,
    )


# ── validate_config ──────────────────────────────────────────────────────────


def _labels_summary(state: ToolState, config: MapcvConfig) -> Optional[Dict[str, Any]]:
    labels = config.labels
    if labels is None:
        return None
    if isinstance(labels, RasterLabelsConfig):
        where = _redact_url(labels.path) if "://" in labels.path else labels.path
        return {"type": "raster", "path": where, "band": labels.band, "classes": labels.class_map()}
    return {
        "type": "vector",
        "path": state.sandbox.rel(labels.path),
        "label_field": labels.label_field,
        "classes": labels.classes,
        "layer": labels.layer,
    }


def config_summary(state: ToolState, config: MapcvConfig) -> Dict[str, Any]:
    """The settings of a config, with credentials left out."""
    region = config.region
    sampler = config.sampler
    return {
        "task": config.task,
        "task_detail": _task_label(config),
        "region": {
            "west": region.west,
            "south": region.south,
            "east": region.east,
            "north": region.north,
        },
        "imagery": _imagery_label(config),
        "labels": _labels_summary(state, config),
        "sampler": {
            "patch_size": sampler.patch_size,
            "stride": sampler.stride,
            "mode": sampler.mode,
            "edge_strategy": sampler.edge_strategy,
        },
        "writer": {
            "staging_dir": state.sandbox.rel(config.writer.staging_dir),
            "image_format": config.writer.image_format,
        },
        "split": None if config.split is None else config.split.model_dump(mode="json"),
    }


def _missing_file_warnings(state: ToolState, config: MapcvConfig) -> List[str]:
    """The same missing-file warnings ``mapcv validate`` prints."""
    messages: List[str] = []
    labels = config.labels
    if isinstance(labels, RasterLabelsConfig):
        label_file = eopf_local_path(labels.path)
        if label_file is not None and not label_file.exists():
            messages.append(f"labels.path not found: {state.sandbox.rel(label_file)}")
    elif labels is not None and not labels.path.exists():
        messages.append(f"labels.path not found: {state.sandbox.rel(labels.path)}")
    if isinstance(config.imagery, GeoTiffImageryConfig):
        local = eopf_local_path(config.imagery.path)
        if local is not None and not local.exists():
            messages.append(f"imagery.path not found: {state.sandbox.rel(local)}")
    return messages


def validate_config(
    state: ToolState, path: Optional[str] = None, yaml_text: Optional[str] = None
) -> ToolResult:
    """Check a config like ``mapcv validate``: no labels or imagery are read."""
    try:
        config, file = load_config(state, path, yaml_text, "path")
    except ConfigInvalid as exc:
        lines = "; ".join(f"{e['field']}: {e['message']}" for e in exc.errors)
        return ToolResult(
            f"Invalid config: {lines}", {"valid": False, "errors": exc.errors, "warnings": []}
        )
    warns = _missing_file_warnings(state, config)
    where = state.sandbox.rel(file) if file is not None else "the config"
    suffix = f" Warnings: {'; '.join(warns)}." if warns else ""
    return ToolResult(
        f"{where} is a valid config.{suffix}",
        {
            "valid": True,
            "errors": [],
            "warnings": warns,
            "summary": config_summary(state, config),
        },
    )


# ── inspect_labels ───────────────────────────────────────────────────────────

_INSPECT_SUFFIXES = tuple(sorted(VECTOR_LABEL_SUFFIXES))


@dataclass
class _Scan:
    features: int = 0
    geometry: "Counter[str]" = field(default_factory=Counter)
    fields: Dict[str, "Counter[str]"] = field(default_factory=dict)
    with_field: Counter[str] = field(default_factory=Counter)
    bounds: Optional[List[float]] = None

    def extend(self, bounds: Tuple[float, float, float, float]) -> None:
        if self.bounds is None:
            self.bounds = list(bounds)
        else:
            self.bounds = [
                min(self.bounds[0], bounds[0]),
                min(self.bounds[1], bounds[1]),
                max(self.bounds[2], bounds[2]),
                max(self.bounds[3], bounds[3]),
            ]


_MAX_SHOWN = 64


def _clip(text: str) -> str:
    """A label value shortened for display: it is the file's content, not ours."""
    return text if len(text) <= _MAX_SHOWN else text[: _MAX_SHOWN - 1] + "…"


def _scan_geojson(data: bytes) -> _Scan:
    try:
        obj: Any = json.loads(data.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ToolFailure(f"The file is not valid UTF-8 JSON: {exc}") from None
    kind = obj.get("type", "") if isinstance(obj, dict) else ""
    if kind == "FeatureCollection":
        features: List[Any] = obj.get("features") or []
    elif kind == "Feature":
        features = [obj]
    else:
        raise ToolFailure(f"Expected a GeoJSON FeatureCollection or Feature, got {kind!r}.")
    try:
        _check_geojson_crs(obj)
    except ValueError as exc:
        raise ToolFailure(str(exc)) from None
    scan = _Scan()
    for feature in features:
        scan.features += 1
        geometry = feature.get("geometry")
        if geometry is None:
            scan.geometry["none"] += 1
        else:
            scan.geometry[str(geometry.get("type", "unknown"))] += 1
            try:
                parsed = shape(geometry)
            except (ValueError, TypeError, AttributeError, KeyError):
                continue
            if not parsed.is_empty:
                scan.extend(cast(Tuple[float, float, float, float], parsed.bounds))
        for key, value in (feature.get("properties") or {}).items():
            if value is None or isinstance(value, (dict, list)):
                continue
            label = _normalize_label(value)
            if label is None:
                continue
            scan.fields.setdefault(key, Counter())[label] += 1
            scan.with_field[key] += 1
    return scan


def _scan_kml(data: bytes) -> _Scan:
    text = data.decode("utf-8", errors="replace")
    names = sorted(set(re.findall(r'<(?:\w+:)?(?:Simple)?Data\s+name="([^"]+)"', text)))
    scan = _Scan()
    polygons, other = parse_kml_rs(data, None)
    scan.features = len(polygons) + other
    for group, _ in polygons:
        scan.geometry["MultiPolygon" if len(group) > 1 else "Polygon"] += 1
    if other:
        scan.geometry["points/lines"] += other
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        geometries, _ = parse_kml(data)
    for geometry, _ in geometries:
        scan.extend(cast(Tuple[float, float, float, float], geometry.bounds))
    for name in names:
        labeled, _ = parse_kml_rs(data, name)
        counter: Counter[str] = Counter()
        for _, label in labeled:
            normalized = _normalize_label(label)
            if normalized is not None:
                counter[normalized] += 1
        if counter:
            scan.fields[name] = counter
            scan.with_field[name] = sum(counter.values())
    return scan


def _scan_table(file: Path, layer: Optional[str]) -> _Scan:
    """GeoPackage, Shapefile and GeoParquet: features in lon/lat and their attributes."""
    suffix = file.suffix.lower()
    if suffix == ".gpkg":
        table = read_gpkg(file, layer)
    elif suffix == ".shp":
        table = read_shapefile(file)
    else:
        table = read_geoparquet(file)
    scan = _Scan()
    scan.features = len(table.geometries)
    for index, geometry in enumerate(table.geometries):
        if geometry is None or geometry.is_empty:
            scan.geometry["none"] += 1
        elif index in table.unreadable:
            scan.geometry["unreadable"] += 1
        else:
            scan.geometry[geometry.geom_type] += 1
            scan.extend(cast(Tuple[float, float, float, float], geometry.bounds))
    for name, column in table.columns.items():
        counter: Counter[str] = Counter()
        for value in column:
            if value is None or isinstance(value, (bytes, dict, list)):
                continue
            label = _normalize_label(value)
            if label is not None:
                counter[label] += 1
        if counter:
            scan.fields[name] = counter
            scan.with_field[name] = sum(counter.values())
    return scan


def inspect_labels(
    state: ToolState, path: str, max_values: int = 20, layer: Optional[str] = None
) -> ToolResult:
    """Fields, the values of each field, feature and geometry counts, and the extent."""
    sandbox = state.sandbox
    file = sandbox.resolve(path, "path")
    if not file.is_file():
        raise ToolFailure(f"Label file not found: {sandbox.rel(file)}")
    suffix = file.suffix.lower()
    if suffix in (".tif", ".tiff"):
        raise ToolFailure(
            "That is a raster. inspect_labels reads vector files; for a label raster set "
            "`labels: {type: raster, classes: ...}` and use `plan` to check it."
        )
    if suffix not in _INSPECT_SUFFIXES:
        raise ToolFailure(
            f"inspect_labels reads {', '.join(_INSPECT_SUFFIXES)} files, not '{file.name}'. "
            "Convert the labels to GeoJSON (EPSG:4326) first."
        )
    if layer is not None and suffix != ".gpkg":
        raise ToolFailure(
            f"`layer` picks a table of a GeoPackage (.gpkg); {file.name} has just one."
        )
    if suffix == ".shp":  # the files next to it are read too
        for sidecar in shapefile_files(file):
            sandbox.inside(sidecar.resolve(), "path", str(sidecar))
    if file.stat().st_size > MAX_LABEL_BYTES:
        raise ToolFailure(
            f"{sandbox.rel(file)} is larger than {MAX_LABEL_BYTES // 1024**2} MiB; "
            "use `plan` to check it instead."
        )
    max_values = max(1, min(int(max_values), 200))
    try:
        if suffix == ".kml":
            scan = _scan_kml(file.read_bytes())
        elif suffix in (".geojson", ".json"):
            scan = _scan_geojson(file.read_bytes())
        else:
            scan = _scan_table(file, layer)
    except ToolFailure:
        raise
    except (ValueError, RuntimeError, OSError) as exc:
        raise ToolFailure(f"Cannot read {sandbox.rel(file)}: {exc}") from None

    fields: List[Dict[str, Any]] = []
    for name in sorted(scan.fields):
        counter = scan.fields[name]
        distinct = len(counter)
        top = sorted(counter.items(), key=lambda item: (-item[1], item[0]))[:max_values]
        fields.append(
            {
                "name": name,
                "distinct": distinct,
                "labeled_features": scan.with_field[name],
                "missing": scan.features - scan.with_field[name],
                "usable_as_classes": 1 <= distinct <= MAX_CLASS_ID,
                "values": [{"value": _clip(value), "count": count} for value, count in top],
                "truncated": distinct > len(top),
            }
        )
    candidates = [f["name"] for f in fields if f["usable_as_classes"] and f["distinct"] >= 2]
    notes: List[str] = []
    extent: Optional[Dict[str, float]] = None
    if scan.bounds is not None:
        west, south, east, north = scan.bounds
        extent = {"west": west, "south": south, "east": east, "north": north}
        if not (-180 <= west and east <= 180 and -90 <= south and north <= 90):
            notes.append(
                "The coordinates are outside longitude/latitude ranges: the file is probably "
                "projected. mapcv reads labels as WGS-84 lon/lat; reproject it to EPSG:4326."
            )
    if not any(kind in scan.geometry for kind in ("Polygon", "MultiPolygon")):
        notes.append(
            "There are no polygons. Segmentation and instance tasks need polygons; detection "
            "also takes points with detection.point_box_size (GeoJSON)."
        )
    if candidates:
        notes.append(
            "Set labels.label_field to one of: " + ", ".join(candidates) + " (or omit it: every "
            "polygon is then class 1)."
        )
    data_out: Dict[str, Any] = {
        "path": sandbox.rel(file),
        "features": scan.features,
        "geometry_types": dict(scan.geometry),
        "extent": extent,
        "fields": fields,
        "label_field_candidates": candidates,
        "notes": notes,
    }
    where = (
        f"extent {extent['west']:.5f}, {extent['south']:.5f} to {extent['east']:.5f}, "
        f"{extent['north']:.5f}"
        if extent
        else "no extent"
    )
    return ToolResult(
        f"{sandbox.rel(file)}: {scan.features:,} feature(s), {len(fields)} field(s), {where}.",
        data_out,
    )


# ── plan ─────────────────────────────────────────────────────────────────────


def large_reason(estimate: Plan) -> Optional[str]:
    """Why :attr:`Plan.is_large` is true (``None`` when it is not), with the limits."""
    reasons: List[str] = []
    if (estimate.tiles or 0) > planning.LARGE_JOB_TILES:
        reasons.append(
            f"{estimate.tiles:,} tiles (the limit for a job without confirmation is "
            f"{planning.LARGE_JOB_TILES:,})"
        )
    total = (estimate.download_bytes or 0) + estimate.output_bytes
    if total > planning.LARGE_JOB_BYTES:
        reasons.append(
            f"about {human_bytes(total)} to download and write (the limit is "
            f"{human_bytes(planning.LARGE_JOB_BYTES)})"
        )
    return "; ".join(reasons) or None


def plan_data(state: ToolState, config: MapcvConfig, estimate: Plan) -> Dict[str, Any]:
    """A plan as structured data, with ``large`` and the reason."""
    labels = estimate.labels
    labels_data: Optional[Dict[str, Any]] = None
    if labels is not None:
        raster = labels.raster is not None
        labels_data = {
            "path": _redact_url(labels.path) if "://" in labels.path else labels.path,
            "type": "raster" if raster else "vector",
            "features": None if raster else labels.polygons,
            "features_in_region": None if raster else labels.in_region,
            "classes": labels.classes,
            "raster": labels.raster,
        }
        if not raster:
            labels_data["path"] = state.sandbox.rel(labels.path)
    large = estimate.is_large
    width_km, height_km = estimate.region_km
    return {
        "task": estimate.task,
        "imagery": estimate.imagery,
        "resolution_m": round(estimate.resolution_m, 3),
        "region_km": {"width": round(width_km, 2), "height": round(height_km, 2)},
        "raster_px": {"width": estimate.raster_px[0], "height": estimate.raster_px[1]},
        "tiles": estimate.tiles,
        "download_bytes": estimate.download_bytes,
        "patches": estimate.patches,
        "patch_size": estimate.patch_size,
        "objects": estimate.objects,
        "output_bytes": estimate.output_bytes,
        "chunk_memory_bytes": estimate.chunk_memory_bytes,
        "labels": labels_data,
        "warnings": list(estimate.warnings),
        "large": large,
        "large_reason": large_reason(estimate) if large else None,
        "large_limits": {
            "tiles": planning.LARGE_JOB_TILES,
            "download_plus_output_bytes": planning.LARGE_JOB_BYTES,
        },
        "output": state.sandbox.rel(config.writer.staging_dir),
    }


def _plan_summary(data: Dict[str, Any]) -> str:
    tiles = f"{data['tiles']:,} tiles, " if data["tiles"] is not None else ""
    text = (
        f"Plan: {tiles}about {data['patches']:,} patch(es) of {data['patch_size']} px, "
        f"{human_bytes(data['output_bytes'])} output."
    )
    if data["large"]:
        text += f" LARGE job: {data['large_reason']}. generate needs confirm_large=true."
    return text


def make_plan_for(state: ToolState, config: MapcvConfig) -> Tuple[Plan, List[str]]:
    """Estimate a config; also return the Python warnings raised while planning."""
    with capture_warnings() as caught:
        try:
            estimate = make_plan(config)
        except (ValueError, RuntimeError, OSError) as exc:
            raise ToolFailure(f"Cannot plan this config: {exc}") from None
    texts = _warning_texts(caught)
    extra = [text for text in texts if text not in estimate.warnings]
    return estimate, extra


def plan(
    state: ToolState, config: Optional[str] = None, yaml_text: Optional[str] = None
) -> ToolResult:
    """Estimate tiles, patches and sizes without downloading anything."""
    try:
        loaded, _ = load_config(state, config, yaml_text)
    except ConfigInvalid as exc:
        raise _invalid_failure(exc, "The config") from None
    estimate, extra = make_plan_for(state, loaded)
    data = plan_data(state, loaded, estimate)
    data["warnings"] = [*data["warnings"], *extra]
    return ToolResult(_plan_summary(data), data)


# ── write_config ─────────────────────────────────────────────────────────────


def write_config(
    state: ToolState, path: str, yaml_text: str, overwrite: bool = False
) -> ToolResult:
    """Validate config text, then write it to a ``.yaml`` file inside the root."""
    sandbox = state.sandbox
    sandbox.require_write("write_config")
    target = sandbox.resolve(path, "path")
    if target.suffix.lower() not in (".yaml", ".yml"):
        raise ToolFailure("A config file must end in .yaml or .yml.")
    if target.is_dir():
        raise ToolFailure(f"{sandbox.rel(target)} is a folder.")
    if target.exists() and not overwrite:
        raise ToolFailure(
            f"{sandbox.rel(target)} already exists. Pass overwrite=true to replace it."
        )
    try:
        config = parse_config_text(state, yaml_text, target.parent)
    except ConfigInvalid as exc:
        raise _invalid_failure(exc, "The config") from None
    text = yaml_text.replace("\r\n", "\n")
    if not text.endswith("\n"):
        text += "\n"
    target.parent.mkdir(parents=True, exist_ok=True)
    handle, temp_name = tempfile.mkstemp(dir=target.parent, prefix=".mapcv-", suffix=".tmp")
    try:
        with os.fdopen(handle, "w", encoding="utf-8", newline="\n") as stream:
            stream.write(text)
        os.replace(temp_name, target)
    except OSError as exc:
        Path(temp_name).unlink(missing_ok=True)
        raise ToolFailure(f"Cannot write {sandbox.rel(target)}: {exc}") from None
    return ToolResult(
        f"Wrote {sandbox.rel(target)}. Next: plan it, then generate.",
        {
            "written": sandbox.rel(target),
            "valid": True,
            "overwritten": overwrite,
            "summary": config_summary(state, config),
        },
    )


# ── generate ─────────────────────────────────────────────────────────────────


@dataclass
class GenerateJob:
    """A generation that passed every check and may start."""

    config: MapcvConfig
    config_path: Path
    plan: Dict[str, Any]
    extra_warnings: List[str]


def prepare_generate(state: ToolState, config: str, confirm_large: bool = False) -> GenerateJob:
    """Check a config and plan it; refuse a large job that is not confirmed."""
    sandbox = state.sandbox
    sandbox.require_write("generate")
    try:
        loaded, file = load_config(state, config, None)
    except ConfigInvalid as exc:
        raise _invalid_failure(exc, "The config") from None
    assert file is not None
    estimate, extra = make_plan_for(state, loaded)
    data = plan_data(state, loaded, estimate)
    data["warnings"] = [*data["warnings"], *extra]
    sandbox.check_tree(loaded.writer.staging_dir, "writer.staging_dir")
    if estimate.is_large and not confirm_large:
        raise ToolFailure(
            f"This is a large job: {data['large_reason']}. Nothing was started. Show the plan "
            "to the user and, if they agree, call generate again with confirm_large=true.",
            {"confirmation_required": True, "plan": data},
        )
    return GenerateJob(loaded, file, data, [])


def execute_generate(
    state: ToolState,
    job: GenerateJob,
    on_chunk: Callable[[int, int], None],
    cancel: threading.Event,
) -> ToolResult:
    """Run a prepared generation (call it from a worker thread). Resumes an earlier run."""
    config = job.config
    staging = config.writer.staging_dir.resolve()
    state.jobs.acquire(staging)

    def hook(done: int, total: int) -> None:
        if cancel.is_set():
            raise GenerationCancelled()
        on_chunk(done, total)

    try:
        with capture_warnings(wait=60.0) as caught:
            try:
                result = run_generate(config, hook)
            except GenerationCancelled:
                raise ToolFailure(
                    "Generation cancelled. Finished chunks are saved: call generate again with "
                    "the same config to resume."
                ) from None
            except ManifestMismatchError as exc:
                raise ToolFailure(f"Cannot resume: {exc}") from None
            except ToolFailure:
                raise
            except Exception as exc:  # noqa: BLE001 - the CLI gives every failure this advice
                detail = (str(exc) or type(exc).__name__).rstrip(". ")
                raise ToolFailure(
                    f"Generation failed: {detail}. Fix the cause and call generate again: "
                    "finished chunks are kept and the run resumes."
                ) from None
        warns = _warning_texts(caught)
    finally:
        state.jobs.release(staging)
    return _generate_result(state, job, result, warns)


def _generate_result(
    state: ToolState, job: GenerateJob, result: GenerateResult, warns: List[str]
) -> ToolResult:
    manifest = result.manifest
    sandbox = state.sandbox
    files = [f"{folder}/" for folder in patch_folders(manifest)] or ["Images/"]
    files.append("manifest.json")
    if result.split_counts is not None:
        files.append("splits/")
    files.extend(
        name
        for name in ("annotations/", "labels/", "dataset.yaml")
        if manifest.task in ("detection", "instance") and (result.staging_dir / name).exists()
    )
    dataset = sandbox.rel(result.staging_dir)
    data: Dict[str, Any] = {
        "dataset": dataset,
        "task": manifest.task,
        "patches": len(manifest.patches),
        "new_patches": result.new_patches,
        "tiles_requested": result.tiles_requested,
        "tiles_failed": result.tiles_failed,
        "splits": result.split_counts,
        "seconds": round(result.seconds, 1),
        "files": files,
        "warnings": [*job.plan["warnings"], *warns],
        "next": [f"info(dataset='{dataset}')", f"split(dataset='{dataset}', ...) to re-split"],
    }
    summary = f"Dataset ready in {dataset}/: {len(manifest.patches):,} patch(es)"
    if result.new_patches != len(manifest.patches):
        summary += f" ({result.new_patches:,} new this run)"
    if result.tiles_failed:
        summary += f"; {result.tiles_failed:,} of {result.tiles_requested:,} tiles failed"
    return ToolResult(summary + ".", data)


# ── info and split ───────────────────────────────────────────────────────────


def _class_rows(manifest: Manifest) -> List[Dict[str, Any]]:
    """Class balance: pixels per class (segmentation) or objects per class (the others)."""
    rows: List[Dict[str, Any]] = []
    if manifest.task in ("detection", "instance"):
        objects: Counter[str] = Counter()
        patches: Counter[str] = Counter()
        for entry in manifest.patches:
            counts = entry["summary"].get("class_objects") or {}
            objects.update(counts)
            patches.update(counts.keys())
        total = sum(objects.values())
        names = {str(cid): name for cid, name in categories(manifest.class_map).items()}
        for cid in sorted(objects, key=int):
            rows.append(
                {
                    "id": int(cid),
                    "name": names.get(cid, f"class {cid}"),
                    "objects": objects[cid],
                    "share": round(objects[cid] / total, 4),
                    "patches": patches[cid],
                }
            )
        return rows
    pixels: Counter[str] = Counter()
    for entry in manifest.patches:
        pixels.update(entry["summary"].get("class_pixels") or {})
    total = sum(pixels.values())
    names = _class_names(manifest)
    for cid in sorted(pixels, key=int):
        rows.append(
            {
                "id": int(cid),
                "name": names.get(cid, f"class {cid}"),
                "pixels": pixels[cid],
                "share": round(pixels[cid] / total, 4),
            }
        )
    return rows


def _split_counts(sandbox: Sandbox, dataset: Path) -> Optional[Dict[str, int]]:
    splits_dir = dataset / "splits"
    if not splits_dir.is_dir():
        return None
    counts: Dict[str, int] = {}
    for name in ("train", "val", "test"):
        path = splits_dir / f"{name}.txt"
        if path.exists():
            sandbox.inside(path.resolve(), "dataset", str(path))
        text = path.read_text().strip() if path.exists() else ""
        counts[name] = len(text.splitlines()) if text else 0
    return counts


def info(state: ToolState, dataset: str) -> ToolResult:
    """Summarize a generated dataset like ``mapcv info``."""
    sandbox = state.sandbox
    folder = sandbox.resolve(dataset, "dataset")
    manifest_path = folder / "manifest.json"
    if not manifest_path.exists():
        raise ToolFailure(f"No manifest found at {sandbox.rel(manifest_path)}")
    sandbox.inside(manifest_path.resolve(), "dataset", dataset)
    try:
        manifest = Manifest.load(manifest_path)
    except ManifestMismatchError as exc:
        raise ToolFailure(str(exc)) from None
    source = manifest.source
    target = manifest.target
    ignore = target.ignore_index if target is not None else None
    splits = _split_counts(sandbox, folder)
    data: Dict[str, Any] = {
        "dataset": sandbox.rel(folder),
        "task": manifest.task,
        "image_only": target is None,
        "source": {
            "type": source.source_type,
            "product": source.product_id,
            "bands": list(source.bands),
            "crs": source.crs,
            "dtype": source.dtype,
            "patch_shape": list(source.patch_shape),
        },
        "patches": len(manifest.patches),
        "padded_patches": sum(1 for entry in manifest.patches if entry["padded"]),
        "ignore_index": ignore,
        "raster_labels": _raster_labels(manifest),
        "classes": manifest.class_map,
        "class_balance": _class_rows(manifest),
        "splits": splits,
        "manifest_version": manifest.loaded_version,
        "mapcv_version": manifest.mapcv_version,
    }
    split_text = (
        ", ".join(f"{name} {count:,}" for name, count in splits.items())
        if splits is not None
        else "no splits"
    )
    return ToolResult(
        f"{sandbox.rel(folder)}: {manifest.task} dataset, {len(manifest.patches):,} patch(es), "
        f"{split_text}.",
        data,
    )


def split(
    state: ToolState,
    dataset: str,
    test_ratio: float = 0.20,
    val_ratio: float = 0.10,
    labeled_ratios: Optional[List[float]] = None,
    seed: int = 42,
    strategy: str = "spatial",
    block_size: Optional[int] = None,
    sample_limit: Optional[int] = None,
) -> ToolResult:
    """Re-split an existing dataset from its manifest; no images are read."""
    sandbox = state.sandbox
    sandbox.require_write("split")
    folder = sandbox.resolve(dataset, "dataset")
    if not folder.is_dir():
        raise ToolFailure(f"Not a folder: {sandbox.rel(folder)}")
    try:
        config = SplitterConfig(
            test_ratio=test_ratio,
            val_ratio=val_ratio,
            labeled_ratios=labeled_ratios if labeled_ratios is not None else [0.10, 0.20, 0.30],
            seed=seed,
            strategy=cast(Literal["spatial", "stratified", "random"], strategy),
            block_size=block_size,
            sample_limit=sample_limit,
        )
    except ValidationError as exc:
        errors = format_validation_errors(exc)
        lines = "; ".join(f"{e['field']}: {e['message']}" for e in errors)
        raise ToolFailure(f"Invalid split settings: {lines}", {"errors": errors}) from None
    sandbox.check_tree(folder, "dataset")
    with capture_warnings() as caught:
        try:
            counts = run_split(folder, config)
        except (FileNotFoundError, ManifestMismatchError) as exc:
            raise ToolFailure(str(exc)) from None
    parts = ", ".join(f"{name} {counts[name]:,}" for name in ("train", "val", "test"))
    dropped = counts.get("dropped", 0)
    return ToolResult(
        f"Splits written to {sandbox.rel(folder)}/splits: {parts}.",
        {
            "dataset": sandbox.rel(folder),
            "splits": counts,
            "dropped_overlapping": dropped,
            "strategy": config.strategy,
            "warnings": _warning_texts(caught),
        },
    )
