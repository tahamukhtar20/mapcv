"""``mapcv doctor``: diagnose an installation in one command, for users and for bug reports.

Each ``*_checks`` function returns a list of :class:`Check` records; :func:`run_checks`
joins them, and the CLI renders them (a table, or JSON with ``--json``). A check only
reads. The one thing it writes is a probe file in the tile cache folder, which it deletes
at once. Nothing it prints is a secret: a credentials file is looked for, never opened
(only its place is shown), and proxy settings are shown by variable name, not by value,
because a proxy URL can hold a password.
"""

from __future__ import annotations

import importlib
import importlib.metadata
import importlib.util
import json
import os
import platform
import re
import socket
import ssl
import sys
import tempfile
import time
import urllib.error
import urllib.request
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor, wait
from dataclasses import dataclass
from pathlib import Path
from typing import Literal
from urllib.parse import urlsplit

import mapcv
from mapcv import tile_cache

Status = Literal["ok", "warn", "fail", "info"]

SECTIONS: tuple[str, ...] = ("mapcv", "Core", "Extras", "Tile cache", "Terminal", "Network")
NETWORK_TIMEOUT_S = 5.0
ISSUE_ZARR_PY314 = "https://github.com/tahamukhtar20/mapcv/issues/212"

# The runtime dependencies of pyproject.toml, used only when the package metadata is
# missing (mapcv run from a source checkout that was never installed).
_FALLBACK_CORE_DEPENDENCIES = (
    "Pillow",
    "numpy",
    "shapely",
    "pydantic",
    "typer",
    "rich",
    "pyyaml",
    "typing_extensions",
    "pyproj",
    "pyshp",
)
_UNREACHABLE_HINT = (
    "Check your connection. Behind a proxy or firewall, set HTTPS_PROXY and allow this "
    "host. Not needed if you don't use this service."
)
_PROXY_VARIABLES = ("HTTPS_PROXY", "HTTP_PROXY", "ALL_PROXY", "NO_PROXY")


@dataclass(frozen=True)
class Check:
    """One line of the report.

    ``status`` is ``ok`` (fine), ``warn`` (works, but check it), ``fail`` (mapcv can't
    work) or ``info`` (a fact, or an optional part that isn't installed). ``hint`` says
    what to do next, when there is something to do.
    """

    section: str
    name: str
    status: Status
    detail: str
    hint: str | None = None

    def to_dict(self) -> dict[str, str | None]:
        return {
            "section": self.section,
            "name": self.name,
            "status": self.status,
            "detail": self.detail,
            "hint": self.hint,
        }


@dataclass(frozen=True)
class TerminalInfo:
    """What the CLI knows about its output, passed in so the checks stay pure."""

    encoding: str | None
    is_tty: bool
    width: int
    color_system: str | None
    no_color_env: bool
    no_color_flag: bool


@dataclass(frozen=True)
class Endpoint:
    """A service mapcv talks to by default and the URL to probe it with."""

    name: str
    url: str
    # Reached means any answer below 500 (a root URL that is a 404 still proves the way
    # through), instead of a success or redirect.
    accept_client_error: bool = False


def _display_path(path: str | Path) -> str:
    """The path with the home folder shortened to ``~`` (shorter, and no user name)."""
    text = str(path)
    try:
        home = str(Path.home())
    except (RuntimeError, KeyError):  # pragma: no cover - no home folder at all
        return text
    if home and home not in ("/", "\\") and (text == home or text.startswith(home + os.sep)):
        return "~" + text[len(home) :]
    return text


def _tiles_and_size(tiles: int, size: float) -> str:
    """``1 tile, 31.6 KB``: the format of ``mapcv cache`` (``planning.human_bytes``), kept
    here so the doctor imports nothing that may be the broken part."""
    count = f"{tiles:,} {'tile' if tiles == 1 else 'tiles'}"
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if size < 1000 or unit == "TB":
            return f"{count}, {size:.0f} {unit}" if unit == "B" else f"{count}, {size:.1f} {unit}"
        size /= 1000
    return count  # pragma: no cover - the loop always returns


def _installed(module: str) -> bool:
    """Whether ``module`` can be imported, without importing it."""
    try:
        return importlib.util.find_spec(module) is not None
    except (ImportError, ValueError):
        return False


def _dist_version(dist: str) -> str | None:
    try:
        return importlib.metadata.version(dist)
    except importlib.metadata.PackageNotFoundError:
        return None


def _major(version: str) -> int | None:
    match = re.match(r"\d+", version)
    return int(match.group()) if match else None


# 1. mapcv --------------------------------------------------------------------------


def _system_description() -> str:
    system = platform.system() or "Unknown OS"
    release = platform.release()
    if system == "Darwin":
        system, release = "macOS", platform.mac_ver()[0] or release
    text = f"{system} {release} {platform.machine()}".replace("  ", " ").strip()
    if system == "Linux":
        libc, libc_version = platform.libc_ver()
        if libc:
            text += f" ({libc} {libc_version})".replace(" )", ")")
    return text


def mapcv_checks() -> list[Check]:
    section = "mapcv"
    version = mapcv.__version__
    if version.startswith("0+unknown"):
        first = Check(
            section,
            "mapcv",
            "warn",
            "version unknown: mapcv is not installed as a package",
            "Install it with pip install mapcv (or pip install -e . in a source checkout).",
        )
    else:
        first = Check(section, "mapcv", "ok", version)
    return [
        first,
        Check(
            section,
            "Python",
            "ok",
            f"{platform.python_version()} ({platform.python_implementation()})",
        ),
        Check(section, "Interpreter", "info", _display_path(sys.executable or "unknown")),
        Check(section, "System", "info", _system_description()),
        Check(section, "Location", "info", _display_path(Path(mapcv.__file__).resolve().parent)),
    ]


# 2. Core ---------------------------------------------------------------------------


def _core_dependencies() -> list[str]:
    """The names of mapcv's runtime dependencies (those without an ``extra`` marker)."""
    try:
        requirements = importlib.metadata.requires("mapcv")
    except importlib.metadata.PackageNotFoundError:
        requirements = None
    if not requirements:
        return list(_FALLBACK_CORE_DEPENDENCIES)
    names = []
    for requirement in requirements:
        if "extra ==" in requirement or "extra==" in requirement:
            continue
        match = re.match(r"[A-Za-z0-9][A-Za-z0-9._-]*", requirement)
        if match:
            names.append(match.group())
    return names or list(_FALLBACK_CORE_DEPENDENCIES)


def core_checks() -> list[Check]:
    section = "Core"
    checks: list[Check] = []
    reinstall = "Reinstall it with: pip install --force-reinstall --no-cache-dir mapcv"
    try:
        extension = importlib.import_module("mapcv._mapcv_rs")
    except Exception as exc:  # noqa: BLE001 - a missing, wrong-platform or broken binary
        checks.append(
            Check(
                section,
                "Rust extension",
                "fail",
                f"can't be loaded ({type(exc).__name__}: {exc})",
                f"{reinstall}. If this repeats, send this report with your bug report.",
            )
        )
    else:
        built = getattr(extension, "__version__", None)
        if built is None:
            checks.append(
                Check(section, "Rust extension", "ok", "loaded (it does not report a version)")
            )
        elif built != mapcv.__version__:
            checks.append(
                Check(
                    section,
                    "Rust extension",
                    "fail",
                    f"is version {built} but the Python package is {mapcv.__version__}",
                    f"{reinstall}. (In a source checkout: maturin develop.)",
                )
            )
        else:
            checks.append(Check(section, "Rust extension", "ok", f"loaded, version {built}"))
    for name in _core_dependencies():
        found = _dist_version(name)
        if found is None:
            checks.append(
                Check(section, name, "fail", "not installed", f"{reinstall} to restore it.")
            )
        else:
            checks.append(Check(section, name, "ok", found))
    return checks


# 3. Extras -------------------------------------------------------------------------


def _install_hint(extra: str) -> str:
    return f'pip install "mapcv[{extra}]"'


def _earth_engine_credentials() -> list[str]:
    """Where Earth Engine credentials were found: file paths and variable names, never
    their contents."""
    found = []
    home = Path.home()
    earthengine = home / ".config" / "earthengine" / "credentials"
    if earthengine.is_file():
        found.append(_display_path(earthengine))
    variable = os.environ.get("GOOGLE_APPLICATION_CREDENTIALS")
    if variable and Path(variable).expanduser().is_file():
        found.append("GOOGLE_APPLICATION_CREDENTIALS")
    gcloud_dir = os.environ.get("CLOUDSDK_CONFIG")
    if gcloud_dir:
        gcloud = Path(gcloud_dir)
    elif sys.platform == "win32" and os.environ.get("APPDATA"):
        gcloud = Path(os.environ["APPDATA"]) / "gcloud"
    else:
        gcloud = home / ".config" / "gcloud"
    adc = gcloud / "application_default_credentials.json"
    if adc.is_file():
        found.append(_display_path(adc))
    return found


def _zarr_check(section: str) -> Check:
    name = "zarr"
    purpose = "Sentinel-2 EOPF imagery"
    if not (_installed("xarray_eopf") and _installed("zarr")):
        python = sys.version_info[:2]
        if python >= (3, 14):
            return Check(
                section,
                name,
                "info",
                f"not available on Python {python[0]}.{python[1]} yet ({purpose})",
                f"Use Python 3.13 or older for it, or follow {ISSUE_ZARR_PY314}",
            )
        if _installed("xarray_eopf") or _installed("zarr"):
            return Check(
                section,
                name,
                "warn",
                f"only part of the extra is installed ({purpose})",
                f"Install the rest with: {_install_hint('zarr')}",
            )
        return Check(
            section, name, "info", f"not installed ({purpose})", f"Install: {_install_hint('zarr')}"
        )
    eopf = _dist_version("xarray-eopf") or "unknown"
    zarr_version = _dist_version("zarr") or "unknown"
    detail = f"xarray-eopf {eopf}, zarr {zarr_version}"
    if (_major(zarr_version) or 0) >= 3:
        return Check(
            section,
            name,
            "warn",
            f"{detail}; the EOPF reader needs zarr 2.18",
            f"Fix it with: {_install_hint('zarr')}",
        )
    return Check(section, name, "ok", detail)


def _simple_extra(
    section: str,
    name: str,
    module: str,
    dist: str,
    extra: str,
    purpose: str,
    hint: str | None = None,
) -> Check:
    if not _installed(module):
        return Check(
            section,
            name,
            "info",
            f"not installed ({purpose})",
            f"Install: {hint or _install_hint(extra)}",
        )
    return Check(section, name, "ok", f"{dist} {_dist_version(dist) or 'unknown'}")


def extras_checks() -> list[Check]:
    section = "Extras"
    checks = [_zarr_check(section)]
    gee = _simple_extra(
        section, "gee", "ee", "earthengine-api", "gee", "Google Earth Engine imagery"
    )
    checks.append(gee)
    if gee.status == "ok":
        credentials = _earth_engine_credentials()
        if credentials:
            checks.append(
                Check(section, "Earth Engine credentials", "ok", "found: " + ", ".join(credentials))
            )
        else:
            checks.append(
                Check(
                    section,
                    "Earth Engine credentials",
                    "info",
                    "none found",
                    "Log in once with: earthengine authenticate",
                )
            )
    checks.append(
        _simple_extra(
            section,
            "parquet / export",
            "pyarrow",
            "pyarrow",
            "parquet",
            "GeoParquet labels and Hugging Face export",
            hint=f"{_install_hint('parquet')} for GeoParquet labels, or "
            f"{_install_hint('export')} for the Hugging Face export",
        )
    )
    mcp = _simple_extra(section, "mcp", "mcp", "mcp", "mcp", "the MCP server for AI agents")
    if mcp.status == "ok" and (_major(mcp.detail.split()[-1]) or 0) != 2:
        mcp = Check(
            section,
            "mcp",
            "warn",
            f"{mcp.detail}; mapcv needs mcp 2.x",
            'Upgrade it with: pip install -U "mapcv[mcp]"',
        )
    checks.append(mcp)
    return checks


# 4. Tile cache ---------------------------------------------------------------------


def _probe_folder(folder: Path) -> tuple[Path | None, str | None]:
    """Write and delete a probe file where ``folder`` is or would be created.

    Returns the existing folder that was probed, and the reason if it can't be written.
    """
    probe_in = folder
    while not probe_in.exists():
        if probe_in.parent == probe_in:
            break
        probe_in = probe_in.parent
    if not probe_in.is_dir():
        return probe_in, f"{_display_path(probe_in)} is a file, not a folder"
    try:
        descriptor, name = tempfile.mkstemp(prefix=".mapcv-doctor-", dir=probe_in)
    except OSError as exc:
        return probe_in, exc.strerror or str(exc)
    os.close(descriptor)
    try:
        os.unlink(name)
    except OSError:  # pragma: no cover - removed by something else already
        pass
    return probe_in, None


def cache_checks() -> list[Check]:
    section = "Tile cache"
    folder = tile_cache.tiles_dir()
    origin = f" (from {tile_cache.CACHE_ENV})" if os.environ.get(tile_cache.CACHE_ENV) else ""
    shown = _display_path(folder)
    _, problem = _probe_folder(folder)
    if problem is not None:
        return [
            Check(
                section,
                "Folder",
                "fail",
                f"{shown}{origin} can't be written: {problem}",
                f"Set {tile_cache.CACHE_ENV} to a folder you can write to, "
                "or set imagery.cache: false in your config.",
            )
        ]
    checks = [
        Check(
            section,
            "Folder",
            "ok",
            f"{shown}{origin}, writable" + ("" if folder.is_dir() else " (created on first use)"),
        )
    ]
    if folder.is_dir():
        found = tile_cache.usage()
        detail = _tiles_and_size(found.tiles, found.bytes)
        if found.expired:
            checks.append(
                Check(
                    section,
                    "Contents",
                    "info",
                    f"{detail}, {found.expired:,} expired",
                    "Delete the expired ones with: mapcv cache --clear --expired",
                )
            )
        else:
            checks.append(Check(section, "Contents", "info", detail))
    else:
        checks.append(Check(section, "Contents", "info", "empty"))
    return checks


# 5. Terminal -----------------------------------------------------------------------


def terminal_checks(info: TerminalInfo) -> list[Check]:
    section = "Terminal"
    encoding = info.encoding or "unknown"
    normalized = encoding.lower().replace("-", "").replace("_", "")
    if normalized in ("utf8", "utf8sig"):
        checks = [Check(section, "Encoding", "ok", encoding)]
    else:
        checks = [
            Check(
                section,
                "Encoding",
                "warn",
                f"{encoding}: symbols such as arrows may show as ?",
                "Use a UTF-8 terminal, or set PYTHONUTF8=1.",
            )
        ]
    checks.append(
        Check(
            section,
            "Output",
            "info",
            "an interactive terminal" if info.is_tty else "not a terminal (piped or redirected)",
        )
    )
    if info.no_color_env:
        colour = "off (NO_COLOR is set)"
    elif info.no_color_flag:
        colour = "off (--no-color)"
    elif info.color_system is None:
        colour = "off (not a colour terminal)"
    else:
        colour = f"on ({info.color_system})"
    checks.append(Check(section, "Colour", "info", colour))
    checks.append(Check(section, "Width", "info", f"{info.width} columns"))
    return checks


# 6. Network ------------------------------------------------------------------------


def default_endpoints() -> list[Endpoint]:
    """The services mapcv uses by default. URLs come from the code that uses them."""
    from mapcv.config import CogSearchConfig, StacSearchConfig
    from mapcv.downloader import URL_TEMPLATES

    esri = URL_TEMPLATES["esri_satellite"].format(z=0, x=0, y=0)
    eopf = StacSearchConfig.model_fields["catalog"].default
    earth_search = CogSearchConfig.model_fields["catalog"].default
    endpoints = [
        Endpoint("Esri World Imagery tiles", esri),
        Endpoint("EOPF STAC catalog", str(eopf)),
        Endpoint("Earth Search STAC", str(earth_search)),
    ]
    if _installed("ee"):
        endpoints.append(Endpoint("Earth Engine", "https://earthengine.googleapis.com/", True))
    return endpoints


def _failure_reason(exc: BaseException, timeout: float) -> str:
    reason = exc.reason if isinstance(exc, urllib.error.URLError) else exc
    if isinstance(reason, (TimeoutError, socket.timeout)):
        return f"no answer within {timeout:g} s"
    if isinstance(reason, socket.gaierror):
        return "the host name can't be resolved"
    if isinstance(reason, ConnectionRefusedError):
        return "connection refused"
    if isinstance(reason, ssl.SSLCertVerificationError):
        return "its TLS certificate can't be verified"
    if isinstance(reason, ssl.SSLError):
        return "TLS handshake failed"
    text = str(reason) or type(reason).__name__
    return text[:120]


def _probe(endpoint: Endpoint, timeout: float) -> Check:
    section = "Network"
    host = urlsplit(endpoint.url).hostname or endpoint.url
    hint = _UNREACHABLE_HINT
    request = urllib.request.Request(
        endpoint.url, headers={"User-Agent": f"mapcv-doctor/{mapcv.__version__}"}
    )
    started = time.perf_counter()
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            status = int(response.status)
    except urllib.error.HTTPError as exc:  # an answer, if an unfriendly one
        status = exc.code
    except (urllib.error.URLError, OSError, ValueError) as exc:
        return Check(
            section,
            endpoint.name,
            "warn",
            f"{host}: unreachable, {_failure_reason(exc, timeout)}",
            hint,
        )
    milliseconds = (time.perf_counter() - started) * 1000
    answer = f"HTTP {status} in {milliseconds:,.0f} ms"
    detail = f"{host}: {answer}"
    if status == 407:
        return Check(
            section,
            endpoint.name,
            "warn",
            f"{detail} (the proxy wants a login)",
            "Put the credentials in HTTPS_PROXY, e.g. http://user:password@proxy:8080.",
        )
    if status >= 500:
        return Check(
            section,
            endpoint.name,
            "warn",
            f"{detail} (server error)",
            "The service has a problem. Try again later.",
        )
    if status >= 400 and not endpoint.accept_client_error:
        return Check(
            section,
            endpoint.name,
            "warn",
            f"{detail} (the request was refused)",
            f"Check that {host} is not blocked for you. Behind a proxy or firewall, set HTTPS_PROXY.",
        )
    if status >= 400:
        answer += " (it answered)"
    return Check(section, endpoint.name, "ok", answer)


def _proxy_check() -> Check:
    section = "Network"
    names = [
        name for name in _PROXY_VARIABLES if os.environ.get(name) or os.environ.get(name.lower())
    ]
    if names:
        return Check(section, "Proxy", "info", "set by " + ", ".join(names) + " (values not shown)")
    if any(scheme != "no" for scheme in urllib.request.getproxies()):
        return Check(section, "Proxy", "info", "taken from the system settings")
    return Check(section, "Proxy", "info", "none")


def network_checks(
    endpoints: list[Endpoint] | None = None, timeout: float = NETWORK_TIMEOUT_S
) -> list[Check]:
    """Probe the endpoints at once. An unreachable one is a warning: you may not need it."""
    todo = default_endpoints() if endpoints is None else endpoints
    checks = [_proxy_check()]
    if not todo:
        return checks
    executor = ThreadPoolExecutor(max_workers=len(todo), thread_name_prefix="mapcv-doctor")
    try:
        futures = [executor.submit(_probe, endpoint, timeout) for endpoint in todo]
        # A host name lookup is not covered by the socket timeout, so wait a little longer
        # than it, then report what is still pending as unanswered.
        wait(futures, timeout=timeout + 3)
        for endpoint, future in zip(todo, futures, strict=True):
            if future.done():
                checks.append(future.result())
            else:
                checks.append(
                    Check(
                        "Network",
                        endpoint.name,
                        "warn",
                        f"{urlsplit(endpoint.url).hostname}: no answer within {timeout:g} s",
                        _UNREACHABLE_HINT,
                    )
                )
    finally:
        executor.shutdown(wait=False, cancel_futures=True)
    return checks


# All together ----------------------------------------------------------------------


def run_checks(terminal: TerminalInfo, offline: bool = False) -> list[Check]:
    """Every check, in section order. A check that crashes becomes a warning, not a
    traceback: the report is what a user pastes into a bug report."""
    steps: list[tuple[str, Callable[[], list[Check]]]] = [
        ("mapcv", mapcv_checks),
        ("Core", core_checks),
        ("Extras", extras_checks),
        ("Tile cache", cache_checks),
        ("Terminal", lambda: terminal_checks(terminal)),
    ]
    if offline:
        steps.append(
            (
                "Network",
                lambda: [
                    _proxy_check(),
                    Check("Network", "Endpoints", "info", "skipped (--offline)"),
                ],
            )
        )
    else:
        steps.append(("Network", network_checks))
    checks: list[Check] = []
    for section, step in steps:
        try:
            checks.extend(step())
        except Exception as exc:  # noqa: BLE001 - one broken check must not hide the rest
            checks.append(
                Check(
                    section,
                    "Checks",
                    "warn",
                    f"could not run: {type(exc).__name__}: {exc}",
                    "Please report this at https://github.com/tahamukhtar20/mapcv/issues",
                )
            )
    return checks


def has_failure(checks: list[Check]) -> bool:
    return any(check.status == "fail" for check in checks)


def to_json(checks: list[Check], offline: bool = False) -> str:
    """The report as JSON: ``ok`` (no check failed), ``offline`` and ``checks``, each with
    ``section``, ``name``, ``status``, ``detail`` and ``hint`` (``null`` if none)."""
    document = {
        "ok": not has_failure(checks),
        "offline": offline,
        "checks": [check.to_dict() for check in checks],
    }
    return json.dumps(document, indent=2) + "\n"
