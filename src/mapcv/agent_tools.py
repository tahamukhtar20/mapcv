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

import copy
import difflib
import ipaddress
import json
import os
import tempfile
import threading
import warnings
from collections import Counter
from collections.abc import Callable
from contextlib import AbstractContextManager, nullcontext
from dataclasses import dataclass, field
from pathlib import Path
from typing import (
    Any,
    Literal,
    cast,
    get_args,
)
from urllib.parse import urlsplit

import yaml
from pydantic import ValidationError
from shapely.geometry import shape

from mapcv import planning
from mapcv._confine import first_outside_path
from mapcv._mapcv_rs import kml_fields as _kml_fields
from mapcv._mapcv_rs import parse_kml as _parse_kml_bytes
from mapcv._net import is_internal_host, public_addresses_only
from mapcv._redact import Redactor
from mapcv._redact import redact_url as _redact_url
from mapcv._warnings import capture as _capture
from mapcv.cli import _class_names, _imagery_label, _raster_labels, _task_label
from mapcv.config import (
    MULTI_SOURCE_TASKS,
    PLANNED_TASKS,
    RASTER_LABEL_TYPES,
    SUPPORTED_TASKS,
    UNION_TAGS,
    ContinuousLabelsConfig,
    EOPFZarrImageryConfig,
    GeoTiffImageryConfig,
    LabelsConfig,
    MapcvConfig,
    RasterLabelsConfig,
    StacCogImageryConfig,
    XYZImageryConfig,
    _resolve_relative_paths,
    eopf_local_path,
    load_yaml,
)
from mapcv.downloader import URL_TEMPLATES
from mapcv.labels import (
    MAX_CLASS_ID,
    VECTOR_LABEL_SUFFIXES,
    _check_geojson_crs,
    _normalize_label,
    kml_to_utf8,
    parse_kml,
)
from mapcv.locking import DatasetBusyError, StagingDirError
from mapcv.manifest import Manifest, ManifestMismatchError, SourceRecord, patch_folders
from mapcv.pipeline import GenerateResult, run_generate, run_split
from mapcv.planning import Plan, human_bytes
from mapcv.planning import plan as make_plan
from mapcv.splitter import SplitterConfig
from mapcv.stac import local_paths_checked
from mapcv.stats import dataset_stats, write_stats
from mapcv.vector_files import read_geoparquet, read_gpkg, read_shapefile, shapefile_files
from mapcv.verify import verify_dataset, write_checksums
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


class ToolFailure(Exception):
    """A mistake the agent can fix: the message says what to change.

    ``data`` is structured detail for the agent (for example the validation errors).
    """

    def __init__(self, message: str, data: dict[str, Any] | None = None) -> None:
        super().__init__(message)
        self.message = message
        self.data = data or {}


class GenerationCancelled(Exception):
    """Raised inside a generation when the client cancelled the call."""


@dataclass
class ToolResult:
    """What a tool returns: structured data and a short sentence for people."""

    summary: str
    data: dict[str, Any] = field(default_factory=dict)


# ── Where the tools may read and write ───────────────────────────────────────


class Sandbox:
    """The folder tools may use, and whether they may write there."""

    def __init__(
        self,
        root: str | os.PathLike[str],
        allow_write: bool = False,
        allow_local_urls: bool = False,
    ) -> None:
        resolved = Path(root).expanduser().resolve()
        if not resolved.is_dir():
            raise ValueError(f"--root {root} is not a folder")
        self.root = resolved
        self.allow_write = allow_write
        self.allow_local_urls = allow_local_urls

    def network(self) -> AbstractContextManager[None]:
        """Where requests may connect while a tool reads imagery: public addresses only,
        redirects and the addresses names resolve to included, unless the server was
        started with ``--allow-local-urls``."""
        return nullcontext() if self.allow_local_urls else public_addresses_only()

    def url_problems(self, config: MapcvConfig) -> list[dict[str, str]]:
        """An error per URL of ``config`` that names this machine or a private network,
        when this server connects to public addresses only. Only the URL's text is
        judged here, so nothing is resolved or sent."""
        if self.allow_local_urls:
            return []
        return [
            {
                "field": key,
                "message": f"{_shown_origin(url)} is on this machine or a private network, and "
                "this server connects to public addresses only. Use a public URL, or ask the "
                "user to restart the server with `mapcv mcp --allow-local-urls`",
            }
            for key, url in config_urls(config)
            if is_internal_host(urlsplit(url).hostname)
        ]

    def resolve(self, value: str, what: str = "path") -> Path:
        """An absolute path inside the root; relative paths are relative to the root."""
        if not isinstance(value, str) or not value.strip() or "\x00" in value:
            raise ToolFailure(f"`{what}` must be a path inside {self.root}")
        candidate = Path(value)
        if not candidate.is_absolute():
            candidate = self.root / candidate
        try:
            resolved = candidate.resolve()
        except (OSError, RuntimeError, ValueError) as exc:  # pragma: no cover - exotic paths
            raise ToolFailure(f"`{what}` {value!r} cannot be resolved: {exc}") from None
        return self.inside(resolved, what, value)

    def inside(self, resolved: Path, what: str, shown: str | None = None) -> Path:
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

    def rel(self, path: str | Path) -> str:
        """``path`` relative to the root with ``/`` separators; ``.`` for the root itself."""
        try:
            relative = Path(path).resolve().relative_to(self.root)
        except (ValueError, OSError):  # pragma: no cover - exotic paths
            return str(path)
        return relative.as_posix()

    def check_found_path(self, path: Path) -> None:
        """Fail if a local file found at run time (a STAC asset) is outside the root."""
        self.inside(path.resolve(), "a file named by the STAC catalog", str(path))

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
            except (OSError, RuntimeError, ValueError) as exc:  # pragma: no cover - exotic paths
                raise ToolFailure(f"{what} {str(path)!r} cannot be resolved: {exc}") from None
            self.inside(resolved, what, str(path))

    def check_tree(self, directory: Path, what: str) -> None:
        """Fail if something below an existing ``directory`` would let a write leave the root.

        A symlink that resolves outside the root would let a write land elsewhere, and so
        would a hard link: a file with a second name, which may be outside the root.
        """
        if not directory.is_dir():
            return
        stack = [directory]
        while stack:
            current = stack.pop()
            try:
                with os.scandir(current) as entries:
                    found = list(entries)
            except OSError:
                continue
            for entry in found:
                if entry.is_symlink():
                    self.inside(Path(entry.path).resolve(), what, entry.path)
                elif entry.is_dir(follow_symlinks=False):
                    stack.append(Path(entry.path))
                else:
                    try:
                        links = os.lstat(entry.path).st_nlink  # DirEntry.stat has none on Windows
                    except OSError:
                        continue
                    if links > 1:
                        raise ToolFailure(
                            f"`{what}` holds {self.rel(entry.path)!r}, a hard link: the same "
                            "file has another name, possibly outside the folder this server "
                            f"may use ({self.root}), and writing it would change that file "
                            "too. Ask the user to replace it with a copy, or use another folder."
                        )


def _vector_label_paths(key: str, path: Path) -> list[tuple[str, Path]]:
    """A vector label file and, for a Shapefile, its sidecar files."""
    found: list[tuple[str, Path]] = [(key, path)]
    if path.suffix.lower() == ".shp" and path.is_file():
        found.extend((f"{key} (sidecar)", file) for file in shapefile_files(path)[1:])
    return found


def _shown_origin(url: str) -> str:
    """``scheme://host[:port]`` of a URL: no user info, path or query."""
    parts = urlsplit(url)
    return f"{parts.scheme}://{parts.netloc.rpartition('@')[2]}"


def config_urls(config: MapcvConfig) -> list[tuple[str, str]]:
    """The http(s) URLs a config makes mapcv connect to, named by their config key."""
    found: list[tuple[str, str]] = []

    def add(key: str, value: str | None) -> None:
        if value is not None and urlsplit(value).scheme in ("http", "https"):
            found.append((key, value))

    if isinstance(config.labels, RASTER_LABEL_TYPES):
        add("labels.path", config.labels.path)
    for name, imagery in zip(config.source_names, config.sources):
        where = f"imagery '{name}' " if config.multi_source else "imagery."
        if isinstance(imagery, XYZImageryConfig):
            add(f"{where}url_template", imagery.url_template)
        if isinstance(imagery, (EOPFZarrImageryConfig, GeoTiffImageryConfig)):
            add(f"{where}path", imagery.path)
        search = getattr(imagery, "search", None)
        if search is not None:
            add(f"{where}search.catalog", search.catalog)
    return found


def config_paths(config: MapcvConfig) -> list[tuple[str, Path]]:
    """The local paths a config reads or writes, named by their config key."""
    found: list[tuple[str, Path]] = []
    labels = config.labels
    if isinstance(labels, RASTER_LABEL_TYPES):
        local = eopf_local_path(labels.path)
        if local is not None:
            found.append(("labels.path", local))
    elif labels is not None:
        # Every label file is read: each one must be inside the root.
        for key, path in labels.keyed_files():
            found.extend(_vector_label_paths(key, path))
        if labels.annotated_area is not None:
            found.append(("labels.annotated_area", labels.annotated_area))
    # Change detection may compare two label sets instead: both are read.
    change = config.change
    if change is not None:
        for key, label_set in (
            ("change.before.path", change.before),
            ("change.after.path", change.after),
        ):
            if label_set is not None:
                prefix = key.rsplit(".", 1)[0]
                for file_key, path in label_set.keyed_files(prefix):
                    found.extend(_vector_label_paths(file_key, path))
                if label_set.annotated_area is not None:
                    found.append((f"{prefix}.annotated_area", label_set.annotated_area))
    for name, imagery in zip(config.source_names, config.sources):
        if isinstance(imagery, (EOPFZarrImageryConfig, GeoTiffImageryConfig)):
            # A searched product is found at run time, in a remote catalog.
            local = eopf_local_path(imagery.path) if imagery.path is not None else None
            if local is not None:
                where = f"imagery '{name}' path" if config.multi_source else "imagery.path"
                found.append((where, local))
                if isinstance(imagery, GeoTiffImageryConfig) and imagery.is_pattern:
                    # A mosaic reads every file its pattern matches, links included.
                    try:
                        matches = imagery.files()
                    except FileNotFoundError:
                        matches = []
                    found.extend((where, Path(match)) for match in matches)
    if config.region.path is not None:
        found.extend(_vector_label_paths("region.path", config.region.path))
    found.append(("writer.staging_dir", config.writer.staging_dir))
    return found


# ── Shared state ─────────────────────────────────────────────────────────────


def capture_warnings(broad: bool = False) -> AbstractContextManager[list[warnings.WarningMessage]]:
    """Record the warnings raised inside the block; captures of different tools can overlap.

    ``broad`` also takes warnings from threads the block started (a generation's workers).
    """
    return _capture(broad)


class _Jobs:
    """The staging folders a generation is writing to right now."""

    def __init__(self) -> None:
        self._active: set[Path] = set()
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


def _unique(texts: list[str]) -> list[str]:
    """The texts in order, each once (the plan and the run raise the same warning)."""
    return list(dict.fromkeys(texts))


def _warning_texts(caught: list[warnings.WarningMessage]) -> list[str]:
    texts: list[str] = []
    for warning in caught:
        message = str(warning.message)
        if message not in texts:
            texts.append(message)
    return texts


# ── Config loading ───────────────────────────────────────────────────────────


def _format_error(error: Any) -> dict[str, str]:
    location = ".".join(
        str(part)
        for part in error["loc"]
        if not str(part).startswith("function-") and part not in UNION_TAGS
    )
    message = str(error["msg"]).removeprefix("Value error, ")
    return {"field": location or "config", "message": message}


def format_validation_errors(exc: ValidationError) -> list[dict[str, str]]:
    """Validation problems as ``{field, message}``: the ones ``mapcv validate`` lists."""
    return [_format_error(error) for error in exc.errors()]


def _json_schema() -> dict[str, Any]:
    if "schema" not in _SCHEMA_CACHE:
        _SCHEMA_CACHE["schema"] = MapcvConfig.model_json_schema()
    schema: dict[str, Any] = _SCHEMA_CACHE["schema"]
    return schema


def _schema_variants(node: dict[str, Any], defs: dict[str, Any]) -> list[dict[str, Any]]:
    while "$ref" in node:
        node = defs[node["$ref"].rsplit("/", 1)[-1]]
    for key in ("anyOf", "oneOf"):
        if key in node:
            return [found for sub in node[key] for found in _schema_variants(sub, defs)]
    return [node]


def _known_keys(path: list[str | int], parent: dict[str, Any]) -> list[str]:
    """The keys the config accepts in the mapping at ``path`` (``parent`` is that mapping)."""
    schema = _json_schema()
    defs = schema.get("$defs", {})
    nodes = _schema_variants(schema, defs)
    for part in path:
        found: list[dict[str, Any]] = []
        for node in nodes:
            if isinstance(part, int):
                if "items" in node:
                    found.extend(_schema_variants(node["items"], defs))
            elif part in node.get("properties", {}):
                found.extend(_schema_variants(node["properties"][part], defs))
        nodes = found
    kind = parent.get("type")
    typed = [n for n in nodes if n.get("properties", {}).get("type", {}).get("const") == kind]
    names: set[str] = set()
    for node in typed or nodes:
        names.update(node.get("properties", {}))
    return sorted(names)


def _walk(data: Any, loc: tuple[Any, ...]) -> tuple[Any, list[str | int]]:
    """The container that holds the last key of an error ``loc`` and the data path to it;
    the union tags pydantic adds to a ``loc`` are skipped."""
    current = data
    path: list[str | int] = []
    for part in loc[:-1]:
        in_dict = isinstance(current, dict) and part in current
        in_list = isinstance(current, list) and isinstance(part, int) and part < len(current)
        if in_dict or in_list:
            current = current[part]
            path.append(part)
    return current, path


def _all_validation_errors(data: dict[str, Any], first: ValidationError) -> list[dict[str, str]]:
    """Every problem of an invalid config that can be found in one pass.

    Pydantic runs the checks that compare several fields only when each field is valid on
    its own, so one misspelt key hides them. Misspelt keys are reported (with the closest
    valid key) and left out, then the config is checked again.
    """
    work = copy.deepcopy(data)
    errors: list[dict[str, str]] = []
    exc: ValidationError | None = first
    for _ in range(6):
        if exc is None:
            break
        dropped = 0
        for error in exc.errors():
            item = _format_error(error)
            if error["type"] == "extra_forbidden":
                parent, path = _walk(work, error["loc"])
                key = error["loc"][-1]
                if isinstance(parent, dict) and key in parent:
                    close = difflib.get_close_matches(str(key), _known_keys(path, parent), n=1)
                    if close:
                        item["message"] += f". Did you mean '{close[0]}'?"
                    del parent[key]
                    dropped += 1
            if item not in errors:
                errors.append(item)
        if not dropped:
            break
        try:
            MapcvConfig.model_validate(work)
            exc = None
        except ValidationError as again:
            exc = again
    return errors


def _yaml_problem(exc: yaml.YAMLError) -> str:
    """A YAML error without the source snippet PyYAML adds (it could hold a template)."""
    mark = getattr(exc, "problem_mark", None)
    problem = getattr(exc, "problem", None) or "invalid YAML"
    where = f"line {mark.line + 1}, column {mark.column + 1}: " if mark is not None else ""
    return f"not valid YAML ({where}{problem})"


class ConfigInvalid(Exception):
    """A config that does not validate: ``errors`` are the problems, ``message`` the headline."""

    def __init__(self, message: str, errors: list[dict[str, str]]) -> None:
        super().__init__(message)
        self.message = message
        self.errors = errors


def parse_config_text(state: ToolState, text: str, base: Path) -> MapcvConfig:
    """Validate config text; relative paths in it resolve against ``base`` as for a file."""
    state.redactor.learn_text(text)
    if len(text.encode("utf-8")) > MAX_CONFIG_BYTES:
        raise ToolFailure(f"The config is larger than {MAX_CONFIG_BYTES // 1024} KiB.")
    try:
        data = load_yaml(text)
    except yaml.YAMLError as exc:
        problem = _yaml_problem(exc)
        raise ConfigInvalid(problem, [{"field": "config", "message": problem}]) from None
    state.redactor.learn_data(data)
    if not isinstance(data, dict):
        message = "the config must be a YAML mapping with region, imagery, sampler and writer"
        raise ConfigInvalid(message, [{"field": "config", "message": message}])
    _resolve_relative_paths(data, base)
    # Validation reads region.path (an area of interest) for its bounds: check it, and
    # a Shapefile's sidecars (.prj, .cpg, ...), are inside the root first, so a config
    # cannot make the server open a file outside it.
    region = data.get("region")
    if isinstance(region, dict) and isinstance(region.get("path"), str):
        target = Path(region["path"])
        target = target if target.is_absolute() else Path.cwd() / target
        for what, path in _vector_label_paths("region.path", target):
            state.sandbox.inside(path.resolve(), what, str(path))
    try:
        config = MapcvConfig.model_validate(data)
    except ValidationError as exc:
        errors = _all_validation_errors(data, exc)
        raise ConfigInvalid("the config has errors", errors) from None
    # Every source's template: a second source's credentials must be redacted too.
    for imagery in config.sources:
        if isinstance(imagery, XYZImageryConfig) and imagery.url_template:
            state.redactor.learn_url(imagery.url_template)
    state.sandbox.check_config_paths(config)
    return config


def load_config(
    state: ToolState, config: str | None, yaml_text: str | None, arg: str = "config"
) -> tuple[MapcvConfig, Path | None]:
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

_SCHEMA_CACHE: dict[str, Any] = {}


def _base_config() -> dict[str, Any]:
    return {
        "region": {"west": 10.0, "south": 50.0, "east": 10.1, "north": 50.1},
        "imagery": {"type": "xyz", "zoom": 15, "source": min(URL_TEMPLATES)},
        "sampler": {"patch_size": 256},
        "writer": {"staging_dir": "dataset"},
    }


def _probe(config: dict[str, Any]) -> str | None:
    """``None`` if the config validates, else the first message ``mapcv validate`` gives."""
    try:
        MapcvConfig.model_validate(config)
    except ValidationError as exc:
        error = format_validation_errors(exc)[0]
        return f"{error['field']}: {error['message']}"
    return None


_PROBE_IMAGERY: dict[str, dict[str, Any]] = {
    "xyz": {"type": "xyz", "zoom": 15, "source": min(URL_TEMPLATES)},
    "eopf_zarr": {"type": "eopf_zarr", "path": "S2.zarr"},
    "geotiff": {"type": "geotiff", "path": "ortho.tif"},
    "stac_cog": {"type": "stac_cog", "search": {"datetime": "2025-06-01/2025-06-30"}},
}
_PROBE_LABELS: dict[str, dict[str, Any] | None] = {
    "none": None,
    "vector": {"type": "vector", "path": "labels.geojson"},
    "raster": {"type": "raster", "path": "landcover.tif", "classes": {1: 1}},
    "continuous": {"type": "continuous", "path": "canopy_height.tif"},
}


def _probed_rules() -> dict[str, Any]:
    """Which combinations of task, labels, imagery and image format validate.

    Found by validating minimal configs with the real models, so the answer is
    whatever the validators say today and never a copy that can go stale.
    """
    invalid: list[dict[str, str]] = []
    labels_per_task: dict[str, list[str]] = {}
    for task in SUPPORTED_TASKS:
        labels_per_task[task] = []
        for kind, labels in _PROBE_LABELS.items():
            config = _base_config()
            config["task"] = task
            if task == "change":
                # Change detection compares two images: probe it with a before/after pair.
                one = config["imagery"]
                config["imagery"] = [{**one, "name": "before"}, {**one, "name": "after"}]
            if labels is not None:
                config["labels"] = labels
            problem = _probe(config)
            if problem is None:
                labels_per_task[task].append(kind)
            else:
                invalid.append({"task": task, "labels": kind, "error": problem})
    formats = list(get_args(WriterConfig.model_fields["image_format"].annotation))
    formats_per_imagery: dict[str, list[str]] = {}
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
        # imagery as a list of named sources on one grid: these tasks; change needs exactly 2.
        "multi_source_tasks": list(MULTI_SOURCE_TASKS),
        "labels_allowed_per_task": labels_per_task,
        "task_options_block": [
            task for task in SUPPORTED_TASKS if task in MapcvConfig.model_fields
        ],
        "imagery_types": list(_PROBE_IMAGERY),
        "image_formats_per_imagery": formats_per_imagery,
        "xyz_sources": sorted(URL_TEMPLATES),
        "invalid_combinations": invalid,
        "relative_paths": _RELATIVE_PATHS,
    }


_RELATIVE_PATHS = (
    "Relative paths in a config file are relative to that file's folder; in `yaml_text` "
    "they are relative to the server's root folder."
)


def _compact_type(node: dict[str, Any]) -> str:
    """A field's type in a few words: ``int 1..255``, ``list[LabelFile]``, ``'png'|'jpg'``."""
    if "$ref" in node:
        return str(node["$ref"]).rsplit("/", 1)[-1]
    if "const" in node:
        return repr(node["const"])
    if "enum" in node:
        return "|".join(repr(value) for value in node["enum"])
    for key in ("anyOf", "oneOf"):
        if key in node:
            parts: list[str] = []
            for sub in node[key]:
                text = _compact_type(sub)
                if text != "null" and text not in parts:
                    parts.append(text)
            return " | ".join(parts)
    kind = node.get("type")
    if kind == "array":
        return f"list[{_compact_type(node.get('items', {}))}]"
    if kind == "object":
        extra = node.get("additionalProperties")
        return f"map[{_compact_type(extra)}]" if isinstance(extra, dict) else "object"
    text = str(kind or "any")
    low, high = node.get("minimum", node.get("exclusiveMinimum")), node.get("maximum")
    if low is not None or high is not None:
        text += f" {'' if low is None else low}..{'' if high is None else high}"
    return text


def _compact_model(name: str, node: dict[str, Any]) -> dict[str, Any]:
    required = set(node.get("required", []))
    fields: dict[str, str] = {}
    for field_name, prop in node.get("properties", {}).items():
        text = _compact_type(prop)
        if field_name in required:
            text += " (required)"
        elif prop.get("default") is not None:
            text += f" = {json.dumps(prop['default'], separators=(',', ':'))}"
        fields[field_name] = text
    summary = (node.get("description") or "").strip().split("\n\n")[0].replace("\n", " ")
    model: dict[str, Any] = {"fields": fields}
    if summary:
        model["about"] = summary if len(summary) <= 300 else summary[:299] + "…"
    return model


def _section_models(schema: dict[str, Any], section: str) -> set[str]:
    """The model names reachable from a top-level key (or the model named ``section``)."""
    defs = schema.get("$defs", {})
    start: Any = schema["properties"].get(section) if section in schema["properties"] else None
    if start is None:
        if section not in defs and section != "MapcvConfig":
            raise ToolFailure(
                f"Unknown section '{section}'. Use a top-level key "
                f"({', '.join(schema['properties'])}) or a model name."
            )
        start = {"$ref": f"#/$defs/{section}"} if section in defs else schema
    seen: set[str] = set()

    def visit(node: Any) -> None:
        if isinstance(node, dict):
            ref = node.get("$ref")
            if isinstance(ref, str):
                name = ref.rsplit("/", 1)[-1]
                if name not in seen:
                    seen.add(name)
                    visit(defs[name])
            for value in node.values():
                visit(value)
        elif isinstance(node, list):
            for value in node:
                visit(value)

    visit(start)
    return seen


def describe_config_schema(
    state: ToolState, section: str | None = None, full_schema: bool = False
) -> ToolResult:
    """A short description of the config (each model's fields with type and default) and the
    rules between fields. ``section`` narrows it to one top-level key or model; ``full_schema``
    adds the raw JSON schema, generated from the pydantic models."""
    schema = _json_schema()
    if "rules" not in _SCHEMA_CACHE:
        _SCHEMA_CACHE["rules"] = _probed_rules()
    defs = schema.get("$defs", {})
    wanted = _section_models(schema, section) if section else None
    models: dict[str, Any] = {}
    if wanted is None or section == "MapcvConfig":
        models["MapcvConfig"] = _compact_model("MapcvConfig", schema)
    for name, node in defs.items():
        if wanted is None or name in wanted:
            models[name] = _compact_model(name, node)
    rules = _SCHEMA_CACHE["rules"]
    data: dict[str, Any] = {
        "models": models,
        "required": {
            name: list(node.get("required", []))
            for name, node in {"MapcvConfig": schema, **defs}.items()
            if wanted is None or name in models
        },
        "rules": rules,
        "notes": [
            "Unknown keys are errors. Region is a WGS-84 lon/lat box (west, south, east, north).",
            _RELATIVE_PATHS,
            (
                "To name the class of a label file without a label_field, list it under "
                "labels.files with `class: <name>`."
            ),
            (
                "labels.osm is accepted by validation, but `plan` and `generate` refuse it over "
                "MCP: use the CLI, or save an OSM extract as GeoJSON."
            ),
            "Imagery sources, their licenses and credit lines: " + _PROVIDERS_URL,
        ],
        "more": "describe_config_schema(section='labels') narrows this to one key; "
        "full_schema=true adds the raw JSON schema.",
    }
    if full_schema:
        data["schema"] = schema
    return ToolResult(
        f"Config: tasks {', '.join(rules['tasks'])}; imagery {', '.join(rules['imagery_types'])}. "
        "`rules.invalid_combinations` lists what does not validate, with the exact message.",
        data,
    )


# ── validate_config ──────────────────────────────────────────────────────────


def _labels_summary(state: ToolState, config: MapcvConfig) -> dict[str, Any] | None:
    labels = config.labels
    if labels is None:
        return None
    if isinstance(labels, RasterLabelsConfig):
        where = _redact_url(labels.path) if "://" in labels.path else labels.path
        return {"type": "raster", "path": where, "band": labels.band, "classes": labels.class_map()}
    if isinstance(labels, ContinuousLabelsConfig):
        where = _redact_url(labels.path) if "://" in labels.path else labels.path
        return {
            "type": "continuous",
            "path": where,
            "band": labels.band,
            "scale": labels.scale,
            "offset": labels.offset,
        }
    if labels.files is not None:
        return {
            "type": "vector",
            "files": [
                {
                    **file.model_dump(mode="json", by_alias=True),
                    "path": state.sandbox.rel(file.path),
                }
                for file in labels.files
            ],
            "classes": labels.classes,
        }
    if labels.osm is not None:
        return {
            "type": "osm",
            "classes": [entry.model_dump(mode="json") for entry in labels.osm.classes],
            "class_ids": labels.classes,
        }
    first = labels.first_path
    assert first is not None
    return {
        "type": "vector",
        "path": state.sandbox.rel(first),
        "label_field": labels.label_field,
        "classes": labels.classes,
        "layer": labels.layer,
    }


def config_summary(state: ToolState, config: MapcvConfig) -> dict[str, Any]:
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


def _missing_files(state: ToolState, config: MapcvConfig) -> list[tuple[str, Path]]:
    """Local inputs the config names that do not exist, as ``(config key, path)``."""
    missing: list[tuple[str, Path]] = []

    def check(key: str, path: Path | None) -> None:
        if path is not None and not path.exists():
            missing.append((key, path))

    labels = config.labels
    if isinstance(labels, RASTER_LABEL_TYPES):
        check("labels.path", eopf_local_path(labels.path))
    elif labels is not None:
        for key, path in labels.keyed_files():
            check(key, path)
        check("labels.annotated_area", labels.annotated_area)
    change = config.change
    if change is not None:
        for prefix, label_set in (("change.before", change.before), ("change.after", change.after)):
            if label_set is not None:
                for file_key, path in label_set.keyed_files(prefix):
                    check(file_key, path)
                check(f"{prefix}.annotated_area", label_set.annotated_area)
    for name, imagery in zip(config.source_names, config.sources):
        if isinstance(imagery, GeoTiffImageryConfig):
            local = eopf_local_path(imagery.path)
            if imagery.is_pattern:
                # A mosaic pattern: missing only when it matches no file.
                local = None if imagery.matching_files() else local
            check(f"imagery '{name}' path" if config.multi_source else "imagery.path", local)
    return missing


#: Label files larger than this are not opened to check `label_field` while validating.
_FIELD_CHECK_BYTES = 64 * 1024 * 1024


def _file_fields(file: Path, layer: str | None) -> list[str] | None:
    """The attributes that hold values in a vector label file, or ``None`` when the file
    cannot be read here (too big, an unknown format, or a read error that `plan` will name)."""
    suffix = file.suffix.lower()
    try:
        if not file.is_file() or file.stat().st_size > _FIELD_CHECK_BYTES:
            return None
        if suffix == ".kml":
            scan = _scan_kml(file.read_bytes())
        elif suffix in (".geojson", ".json"):
            scan = _scan_geojson(file.read_bytes())
        elif suffix in (".gpkg", ".shp", ".parquet", ".geoparquet"):
            scan = _scan_table(file, layer if suffix == ".gpkg" else None)
        else:
            return None
    except (ToolFailure, ValueError, RuntimeError, OSError, ImportError):
        return None
    return sorted(scan.fields)


def _label_field_problems(state: ToolState, config: MapcvConfig) -> list[dict[str, str]]:
    """A ``label_field`` the label file does not have, with the closest field name."""
    problems: list[dict[str, str]] = []
    sets: list[tuple[str, LabelsConfig]] = []
    if isinstance(config.labels, LabelsConfig):
        sets.append(("labels", config.labels))
    if config.change is not None:
        sets.extend(
            (prefix, label_set)
            for prefix, label_set in (
                ("change.before", config.change.before),
                ("change.after", config.change.after),
            )
            if label_set is not None
        )
    for prefix, labels in sets:
        for index, entry in enumerate(labels.label_files):
            if entry.label_field is None:
                continue
            fields = _file_fields(entry.path, entry.layer)
            if fields is None or entry.label_field in fields:
                continue
            key = (
                f"{prefix}.files[{index}].label_field" if labels.files else f"{prefix}.label_field"
            )
            close = difflib.get_close_matches(entry.label_field, fields, n=1, cutoff=0.6)
            close = close or [f for f in fields if f.lower() == entry.label_field.lower()]
            hint = f" Did you mean '{close[0]}'?" if close else ""
            have = ", ".join(fields[:12]) + (", ..." if len(fields) > 12 else "")
            problems.append(
                {
                    "field": key,
                    "message": (
                        f"'{entry.label_field}' is not an attribute with values in "
                        f"{state.sandbox.rel(entry.path)} (it has: {have or 'none'}).{hint}"
                    ),
                }
            )
    return problems


def _declared_bands(imagery: Any) -> tuple[int | None, str | None]:
    """Band count and data type of a source as far as its config alone says."""
    if isinstance(imagery, XYZImageryConfig):
        return 3, "uint8"
    if isinstance(imagery, StacCogImageryConfig):
        return len(imagery.bands), "uint16"
    if isinstance(imagery, EOPFZarrImageryConfig):
        return len(imagery.bands), "float32"
    if isinstance(imagery, GeoTiffImageryConfig) and imagery.bands:
        return len(imagery.bands), None
    return None, None


def _stack_problems(config: MapcvConfig) -> list[dict[str, str]]:
    """``writer.stack_sources`` with sources whose declared bands or data types differ."""
    if not config.writer.stack_sources:
        return []
    known = [
        (name, *_declared_bands(imagery))
        for name, imagery in zip(config.source_names, config.sources)
    ]
    counted = [(name, count, dtype) for name, count, dtype in known if count is not None]
    problems: list[dict[str, str]] = []
    for name, count, dtype in counted[1:]:
        first, first_count, first_dtype = counted[0]
        if count != first_count or (dtype and first_dtype and dtype != first_dtype):
            problems.append(
                {
                    "field": "writer.stack_sources",
                    "message": (
                        "needs every source to have the same bands and data type to stack them: "
                        f"'{first}' has {first_count} band(s), '{name}' {count}; select matching "
                        "`bands` or write the sources as separate files"
                    ),
                }
            )
            break
    return problems


_OSM_WARNING = (
    "labels.osm is fetched from Overpass and cached outside this server's root, so `plan` and "
    "`generate` refuse it over MCP: run `mapcv plan` / `mapcv generate` in a terminal, or save "
    "an OSM extract as GeoJSON and point labels.path at it"
)


def config_problems(
    state: ToolState, config: MapcvConfig
) -> tuple[list[dict[str, str]], list[str]]:
    """What is wrong with a config that validates: ``(errors, warnings)``.

    Errors would make `plan` or `generate` fail (a missing input, a ``label_field`` the file
    does not have, sources that cannot be stacked); warnings are about this server.
    """
    errors = [
        {
            "field": key,
            "message": f"file not found: {state.sandbox.rel(path)}. Fix the path or create "
            "the file first",
        }
        for key, path in _missing_files(state, config)
    ]
    errors.extend(_label_field_problems(state, config))
    errors.extend(_stack_problems(config))
    errors.extend(state.sandbox.url_problems(config))
    label_sets = [config.labels] + (
        [config.change.before, config.change.after] if config.change is not None else []
    )
    uses_osm = any(isinstance(item, LabelsConfig) and item.osm is not None for item in label_sets)
    return errors, [_OSM_WARNING] if uses_osm else []


def _problem_lines(errors: list[dict[str, str]]) -> str:
    return "; ".join(f"{e['field']}: {e['message']}" for e in errors)


# Cloud metadata services that are not in a link-local range or have a name.
_METADATA_HOSTS = frozenset(
    {"metadata.google.internal", "metadata", "instance-data", "100.100.100.200", "fd00:ec2::254"}
)


def _metadata_host_warnings(config: MapcvConfig) -> list[str]:
    """A warning per ``url_template`` on a link-local or cloud metadata address.

    Tiles are fetched from the machine running this server; such a host is not a tile
    server (a misread or injected URL), though a refusal would also block odd setups.
    """
    messages: list[str] = []
    for name, imagery in zip(config.source_names, config.sources):
        if not isinstance(imagery, XYZImageryConfig) or not imagery.url_template:
            continue
        host = (urlsplit(imagery.url_template).hostname or "").lower().rstrip(".")
        try:
            link_local = ipaddress.ip_address(host).is_link_local
        except ValueError:
            link_local = False
        if link_local or host in _METADATA_HOSTS:
            where = (
                f"imagery '{name}' url_template" if config.multi_source else "imagery.url_template"
            )
            messages.append(
                f"{where} points to {host}, a link-local or cloud metadata address rather "
                "than a tile server; check the URL with the user before generating"
            )
    return messages


def validate_config(
    state: ToolState, path: str | None = None, yaml_text: str | None = None
) -> ToolResult:
    """Check a config like ``mapcv validate``, and that the files it names exist and have
    the ``label_field`` it asks for. Imagery is not read."""
    try:
        config, file = load_config(state, path, yaml_text, "path")
    except ConfigInvalid as exc:
        return ToolResult(
            f"Invalid config: {_problem_lines(exc.errors)}",
            {"valid": False, "errors": exc.errors, "warnings": []},
        )
    errors, warns = config_problems(state, config)
    warns = [*warns, *_metadata_host_warnings(config)]
    where = state.sandbox.rel(file) if file is not None else "the config"
    data = {
        "valid": not errors,
        "errors": errors,
        "warnings": warns,
        "summary": config_summary(state, config),
    }
    if errors:
        return ToolResult(f"Invalid config: {_problem_lines(errors)}", data)
    suffix = f" Warnings: {'; '.join(warns)}." if warns else ""
    return ToolResult(f"{where} is a valid config.{suffix}", data)


# ── inspect_labels ───────────────────────────────────────────────────────────

_INSPECT_SUFFIXES = tuple(sorted(VECTOR_LABEL_SUFFIXES))


@dataclass
class _Scan:
    features: int = 0
    geometry: Counter[str] = field(default_factory=Counter)
    fields: dict[str, Counter[str]] = field(default_factory=dict)
    with_field: Counter[str] = field(default_factory=Counter)
    bounds: list[float] | None = None

    def extend(self, bounds: tuple[float, float, float, float]) -> None:
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
        features: list[Any] = obj.get("features") or []
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
                scan.extend(cast(tuple[float, float, float, float], parsed.bounds))
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
    data = kml_to_utf8(data)
    scan = _Scan()
    polygons, other = _parse_kml_bytes(data, None)
    scan.features = len(polygons) + other
    for group, _ in polygons:
        scan.geometry["MultiPolygon" if len(group) > 1 else "Polygon"] += 1
    if other:
        scan.geometry["points/lines"] += other
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        geometries, _ = parse_kml(data)
    for geometry, _ in geometries:
        scan.extend(cast(tuple[float, float, float, float], geometry.bounds))
    # Every field of every polygon in one pass (one parse per field name is quadratic).
    counters: dict[str, Counter[str]] = {}
    for fields in _kml_fields(data):
        for name, label in fields.items():
            normalized = _normalize_label(label)
            if normalized is not None:
                counters.setdefault(name, Counter())[normalized] += 1
    for name in sorted(counters):
        scan.fields[name] = counters[name]
        scan.with_field[name] = sum(counters[name].values())
    return scan


def _scan_table(file: Path, layer: str | None) -> _Scan:
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
            scan.extend(cast(tuple[float, float, float, float], geometry.bounds))
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
    state: ToolState, path: str, max_values: int = 20, layer: str | None = None
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

    fields: list[dict[str, Any]] = []
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
    notes: list[str] = []
    extent: dict[str, float] | None = None
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
    data_out: dict[str, Any] = {
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


def large_reason(estimate: Plan) -> str | None:
    """Why :attr:`Plan.is_large` is true (``None`` when it is not), with the limits."""
    reasons: list[str] = []
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
    if estimate.patches > planning.LARGE_JOB_PATCHES:
        reasons.append(
            f"about {estimate.patches:,} patches (the limit is {planning.LARGE_JOB_PATCHES:,})"
        )
    return "; ".join(reasons) or None


def _plan_notes(config: MapcvConfig, estimate: Plan) -> list[str]:
    """What this plan does not say, so a missing number is not read as zero."""
    notes: list[str] = []
    if estimate.download_bytes is None:
        notes.append(
            "download_bytes is null: imagery read from files or a catalog is not estimated "
            "(only the windows the patches cover are read, so it is at most the file sizes)."
        )
    if any(
        isinstance(source, StacCogImageryConfig)
        or (isinstance(source, EOPFZarrImageryConfig) and source.search is not None)
        for source in config.sources
    ):
        notes.append(
            "The Sentinel-2 scene is chosen when `generate` runs (the least cloudy that covers "
            "the region); `info` then shows its id in source.product."
        )
    if config.task == "classification":
        notes.append("patches is an upper bound: patches no class qualifies for are skipped.")
    return notes


def plan_data(state: ToolState, config: MapcvConfig, estimate: Plan) -> dict[str, Any]:
    """A plan as structured data, with ``large`` and the reason."""
    labels = estimate.labels
    labels_data: dict[str, Any] | None = None
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
        "blocking": list(estimate.blocking),
        "notes": _plan_notes(config, estimate),
        "large": large,
        "large_reason": large_reason(estimate) if large else None,
        "large_limits": {
            "tiles": planning.LARGE_JOB_TILES,
            "download_plus_output_bytes": planning.LARGE_JOB_BYTES,
            "patches": planning.LARGE_JOB_PATCHES,
        },
        "output": state.sandbox.rel(config.writer.staging_dir),
    }


def _plan_summary(data: dict[str, Any]) -> str:
    tiles = f"{data['tiles']:,} tiles, " if data["tiles"] is not None else ""
    text = (
        f"Plan: {tiles}about {data['patches']:,} patch(es) of {data['patch_size']} px, "
        f"{human_bytes(data['output_bytes'])} output."
    )
    if data["large"]:
        text += f" LARGE job: {data['large_reason']}. generate needs confirm_large=true."
    return text


def make_plan_for(state: ToolState, config: MapcvConfig) -> tuple[Plan, list[str]]:
    """Estimate a config; also return the Python warnings raised while planning."""
    change = config.change
    label_sets = [config.labels] + ([change.before, change.after] if change is not None else [])
    if any(isinstance(labels, LabelsConfig) and labels.osm is not None for labels in label_sets):
        # Plan and generate fetch them, caching the answer outside the server's root.
        raise ToolFailure(
            "labels.osm downloads OpenStreetMap labels and caches them outside this server's "
            "root, so plan and generate are not available for it here. Run `mapcv plan` / "
            "`mapcv generate` in a terminal, or point labels.path at an OSM extract."
        )
    problems = state.sandbox.url_problems(config)
    if problems:
        raise ToolFailure(
            f"Cannot plan this config: {_problem_lines(problems)}", {"errors": problems}
        )
    with capture_warnings() as caught:
        try:
            with local_paths_checked(state.sandbox.check_found_path), state.sandbox.network():
                estimate = make_plan(config)
        except (ValueError, RuntimeError, OSError) as exc:
            raise ToolFailure(f"Cannot plan this config: {exc}") from None
    texts = _warning_texts(caught) + _metadata_host_warnings(config)
    extra = [text for text in texts if text not in estimate.warnings]
    return estimate, extra


def plan(state: ToolState, config: str | None = None, yaml_text: str | None = None) -> ToolResult:
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
    errors, warns = config_problems(state, config)
    # The file is saved either way (a draft may name labels not downloaded yet), but `plan`
    # and `generate` fail until the problems are fixed.
    text_out = f"Wrote {sandbox.rel(target)}. Next: plan it, then generate."
    if errors:
        text_out = (
            f"Wrote {sandbox.rel(target)}, but it has problems that make plan and generate "
            f"fail: {_problem_lines(errors)}"
        )
    elif warns:
        text_out += f" Warnings: {'; '.join(warns)}."
    return ToolResult(
        text_out,
        {
            "written": sandbox.rel(target),
            "valid": not errors,
            "errors": errors,
            "warnings": warns,
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
    plan: dict[str, Any]
    extra_warnings: list[str]


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
    if estimate.blocking:
        raise ToolFailure(
            f"{estimate.blocking[0]}. Nothing was started.", {"errors": estimate.blocking}
        )
    if estimate.patches == 0:
        raise ToolFailure(
            "This config would write no patches. Nothing was started. Enlarge the region, "
            "lower sampler.patch_size or set sampler.edge_strategy: pad.",
            {"plan": data},
        )
    sandbox.check_tree(loaded.writer.staging_dir, "writer.staging_dir")
    if estimate.is_large and not confirm_large:
        raise ToolFailure(
            f"This is a large job: {data['large_reason']}. Nothing was started. Show the plan "
            "to the user and, if they agree, call generate again with confirm_large=true.",
            {"confirmation_required": True, "plan": data},
        )
    return GenerateJob(_without_tile_cache(loaded), file, data, [])


def _without_tile_cache(config: MapcvConfig) -> MapcvConfig:
    """The config with the on-disk tile cache off: it lives outside the sandbox root."""

    def off(source: Any) -> Any:
        if isinstance(source, XYZImageryConfig):
            return source.model_copy(update={"cache": False})
        return source

    imagery = config.imagery
    update = [off(each) for each in imagery] if isinstance(imagery, list) else off(imagery)
    return config.model_copy(update={"imagery": update})


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
        with capture_warnings(broad=True) as caught:
            try:
                with local_paths_checked(state.sandbox.check_found_path), state.sandbox.network():
                    result = run_generate(config, hook)
            except GenerationCancelled:
                raise ToolFailure(
                    "Generation cancelled. Finished chunks are saved: call generate again with "
                    "the same config to resume."
                ) from None
            except ManifestMismatchError as exc:
                raise ToolFailure(f"Cannot resume: {exc}") from None
            except (DatasetBusyError, StagingDirError) as exc:
                raise ToolFailure(f"Cannot generate: {exc}") from None
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
    state: ToolState, job: GenerateJob, result: GenerateResult, warns: list[str]
) -> ToolResult:
    manifest = result.manifest
    sandbox = state.sandbox
    if not manifest.patches:
        raise ToolFailure(
            "No patches were written, so there is no dataset. The region may be smaller than "
            "one patch, or every patch was dropped by sampler.max_empty_ratio or "
            "sampler.min_label_ratio (see the warnings). Check the numbers with plan.",
            {"warnings": _unique([*job.plan["warnings"], *warns])},
        )
    files = [f"{folder}/" for folder in patch_folders(manifest)] or ["Images/"]
    files.append("manifest.json")
    if result.split_counts is not None:
        files.append("splits/")
    if (result.staging_dir / "patches.geojson").is_file():
        files.append("patches.geojson")
    files.extend(
        name
        for name in ("annotations/", "labels/", "dataset.yaml")
        if manifest.task in ("detection", "instance") and (result.staging_dir / name).exists()
    )
    if manifest.task == "classification":
        files.extend(("labels.csv", "labels.json", "classes.txt"))
    dataset = sandbox.rel(result.staging_dir)
    data: dict[str, Any] = {
        "dataset": dataset,
        "task": manifest.task,
        "patches": len(manifest.patches),
        "new_patches": result.new_patches,
        "tiles_requested": result.tiles_requested,
        "tiles_failed": result.tiles_failed,
        "splits": result.split_counts,
        "seconds": round(result.seconds, 1),
        "files": files,
        "warnings": _unique([*job.plan["warnings"], *warns]),
        "next": [f"info(dataset='{dataset}')", f"split(dataset='{dataset}', ...) to re-split"],
    }
    summary = f"Dataset ready in {dataset}/: {len(manifest.patches):,} patch(es)"
    if result.new_patches != len(manifest.patches):
        summary += f" ({result.new_patches:,} new this run)"
    if result.tiles_failed:
        summary += f"; {result.tiles_failed:,} of {result.tiles_requested:,} tiles failed"
    return ToolResult(summary + ".", data)


# ── info and split ───────────────────────────────────────────────────────────


def _class_rows(manifest: Manifest) -> list[dict[str, Any]]:
    """Class balance: pixels per class (segmentation), objects per class (detection and
    instance) or patches per label (classification)."""
    rows: list[dict[str, Any]] = []
    if manifest.task == "classification":
        patches_per_label: Counter[str] = Counter()
        for entry in manifest.patches:
            patches_per_label.update(str(cid) for cid in entry["summary"].get("labels") or [])
        label_names = {str(cid): name for cid, name in categories(manifest.class_map).items()}
        label_names["0"] = "background"
        for cid in sorted(patches_per_label, key=int):
            rows.append(
                {
                    "id": int(cid),
                    "name": label_names.get(cid, f"class {cid}"),
                    "patches": patches_per_label[cid],
                    "share": round(patches_per_label[cid] / len(manifest.patches), 4),
                }
            )
        return rows
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


def _split_counts(sandbox: Sandbox, dataset: Path) -> dict[str, int] | None:
    splits_dir = dataset / "splits"
    if not splits_dir.is_dir():
        return None
    counts: dict[str, int] = {}
    for name in ("train", "val", "test"):
        path = splits_dir / f"{name}.txt"
        if path.exists():
            sandbox.inside(path.resolve(), "dataset", str(path))
        text = path.read_text().strip() if path.exists() else ""
        counts[name] = len(text.splitlines()) if text else 0
    return counts


def _source_data(record: SourceRecord) -> dict[str, Any]:
    return {
        "name": record.name,
        "type": record.source_type,
        "product": record.product_id,
        "bands": list(record.bands),
        "crs": record.crs,
        "dtype": record.dtype,
        "patch_shape": list(record.patch_shape),
    }


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
    target = manifest.target
    ignore = target.ignore_index if target is not None else None
    splits = _split_counts(sandbox, folder)
    sources = [_source_data(record) for record in manifest.sources] or [
        _source_data(manifest.source)
    ]
    data: dict[str, Any] = {
        "dataset": sandbox.rel(folder),
        "task": manifest.task,
        "image_only": target is None,
        # The first source's grid is the dataset's; `sources` lists every one (change
        # detection and multi-source datasets have several).
        "source": sources[0],
        "sources": sources,
        "patches": len(manifest.patches),
        # False: generate stopped before it finished; None: made before mapcv recorded it.
        "complete": manifest.complete,
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
    unfinished = (
        " Incomplete: generate stopped before it finished; call generate again with the same "
        "config to finish it."
        if manifest.complete is False
        else ""
    )
    text = (
        f"{sandbox.rel(folder)}: {manifest.task} dataset, {len(manifest.patches):,} patch(es), "
        f"{split_text}.{unfinished}"
    )
    return ToolResult(text, data)


def split(
    state: ToolState,
    dataset: str,
    test_ratio: float = 0.20,
    val_ratio: float = 0.10,
    labeled_ratios: list[float] | None = None,
    seed: int = 42,
    strategy: str = "spatial",
    block_size: int | None = None,
    sample_limit: int | None = None,
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
        except (FileNotFoundError, ManifestMismatchError, DatasetBusyError) as exc:
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


# ── stats and verify ─────────────────────────────────────────────────────────

_MAX_PROBLEMS_SHOWN = 50


def _checked_dataset(state: ToolState, dataset: str) -> tuple[Path, Manifest]:
    """The dataset folder and its manifest, after checking that nothing the dataset names
    (a patch file, a link in the folder) leads outside the root."""
    sandbox = state.sandbox
    folder = sandbox.resolve(dataset, "dataset")
    manifest_path = folder / "manifest.json"
    if not manifest_path.is_file():
        raise ToolFailure(f"No manifest found at {sandbox.rel(manifest_path)}")
    sandbox.inside(manifest_path.resolve(), "dataset", dataset)
    try:
        manifest = Manifest.load(manifest_path)
    except (ManifestMismatchError, ValueError, OSError) as exc:
        raise ToolFailure(f"Cannot read the dataset: {exc}") from None
    sandbox.check_tree(folder, "dataset")
    listed = [rel for entry in manifest.patches for rel in entry["files"].values()]
    checksums = folder / "SHA256SUMS"
    if checksums.is_file():
        try:
            for line in checksums.read_text(encoding="utf-8").splitlines():
                if line.strip():
                    listed.append(line.partition("  ")[2])
        except (OSError, UnicodeDecodeError):
            pass
    outside = first_outside_path(listed)
    if outside is not None:
        raise ToolFailure(
            f"The dataset lists a file path that leaves its folder ({outside[:80]!r}); "
            "mapcv does not read it."
        )
    return folder, manifest


def stats(state: ToolState, dataset: str, split: str = "train", save: bool = False) -> ToolResult:
    """Band mean/std, class balance and class weights of a dataset, like ``mapcv stats``.

    Read-only: ``stats.json`` is written only with ``save`` (which needs ``--allow-write``).
    """
    if split not in ("train", "val", "test", "all"):
        raise ToolFailure("`split` must be train, val, test or all.")
    sandbox = state.sandbox
    if save:
        sandbox.require_write("stats with save=true")
    folder, _ = _checked_dataset(state, dataset)
    try:
        with capture_warnings() as caught:
            if save:
                path, values = write_stats(folder, split)
            else:
                path, values = None, dataset_stats(folder, split)
    except (OSError, ValueError) as exc:  # ManifestMismatchError is a ValueError
        raise ToolFailure(f"Cannot compute the statistics: {exc}") from None
    notes = _warning_texts(caught)
    weights = (values.get("classes") or {}).get("median_frequency_weights")
    text = (
        f"{sandbox.rel(folder)}: statistics of {values['patches']:,} patch(es), "
        f"split {values['split']}"
    )
    if weights:
        text += "; class weights " + ", ".join(f"{k} {v:.3g}" for k, v in weights.items())
    out = {
        "dataset": sandbox.rel(folder),
        **values,
        "saved": sandbox.rel(path) if path else None,
        "warnings": notes,
    }
    return ToolResult(" ".join([text + ".", *notes]), out)


def verify(
    state: ToolState, dataset: str, deep: bool = False, write_sums: bool = False
) -> ToolResult:
    """Check that every file of a dataset is present and intact, like ``mapcv verify``.

    Read-only: ``SHA256SUMS`` is written only with ``write_sums`` (needs ``--allow-write``).
    """
    sandbox = state.sandbox
    if write_sums:
        sandbox.require_write("verify with write_sums=true")
    folder, _ = _checked_dataset(state, dataset)
    report = verify_dataset(folder, deep=deep)
    written: str | None = None
    if write_sums and (report.ok or report.only_rewritten):
        # Files that split, stats and card rewrite are recorded with their new hashes.
        try:
            written = sandbox.rel(write_checksums(folder))
        except OSError as exc:
            raise ToolFailure(f"Cannot write SHA256SUMS: {exc}") from None
        if not report.ok:
            report = verify_dataset(folder, deep=deep)
    out = {
        "dataset": sandbox.rel(folder),
        "ok": report.ok,
        "patches": report.patches,
        "files": report.files,
        "hashes_checked": report.checked_hashes,
        "deep": deep,
        "problem_count": len(report.problems),
        "problems": report.problems[:_MAX_PROBLEMS_SHOWN],
        "notes": report.notes,
        "incomplete": getattr(report, "incomplete", False),
        "checksums_written": written,
    }
    where = sandbox.rel(folder)
    if report.ok:
        summary = f"{where}: {report.patches:,} patch(es), {report.files:,} file(s) present"
        if report.checked_hashes:
            summary += f", {report.checked_hashes:,} hash(es) match"
        return ToolResult(summary + ".", out)
    return ToolResult(
        f"{where}: {len(report.problems):,} problem(s), first: {report.problems[0]}", out
    )
