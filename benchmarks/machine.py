"""Facts about the machine and the mapcv build that a result file should carry."""

from __future__ import annotations

import datetime
import os
import platform
import subprocess
import sys
from importlib import metadata
from typing import Any

from benchmarks.measure import ROOT

try:
    import psutil
except ImportError:
    psutil = None

LIBRARIES = ("numpy", "Pillow", "shapely", "pydantic", "typer", "rich", "rasterio", "pyproj")


def _cpu_model() -> str:
    try:
        if sys.platform.startswith("linux"):
            with open("/proc/cpuinfo", encoding="utf-8") as handle:
                for line in handle:
                    if line.lower().startswith(("model name", "hardware", "cpu model")):
                        return line.split(":", 1)[1].strip()
        if sys.platform == "darwin":
            return subprocess.run(
                ["sysctl", "-n", "machdep.cpu.brand_string"],
                capture_output=True,
                text=True,
                check=True,
            ).stdout.strip()
    except (OSError, subprocess.SubprocessError, IndexError):
        pass
    return platform.processor() or platform.machine()


def _ram_gb() -> float | None:
    if psutil is not None:
        return round(float(psutil.virtual_memory().total) / 2**30, 1)
    try:
        return round(os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES") / 2**30, 1)
    except (ValueError, OSError, AttributeError):
        return None


def _git(*args: str) -> str | None:
    try:
        result = subprocess.run(
            ["git", "-C", str(ROOT), *args], capture_output=True, text=True, check=True
        )
    except (OSError, subprocess.SubprocessError):
        return None
    return result.stdout.strip()


def _version(package: str) -> str | None:
    try:
        return metadata.version(package)
    except metadata.PackageNotFoundError:
        return None


def machine_info() -> dict[str, Any]:
    """Hardware, OS and interpreter."""
    return {
        "os": platform.platform(),
        "machine": platform.machine(),
        "cpu_model": _cpu_model(),
        "cpu_count_logical": os.cpu_count(),
        "cpu_count_physical": psutil.cpu_count(logical=False) if psutil is not None else None,
        "ram_gb": _ram_gb(),
        "python": platform.python_version(),
        "python_implementation": platform.python_implementation(),
    }


def load_average() -> float | None:
    """One-minute load average (None on Windows); high values mean other work disturbs timings."""
    try:
        return round(os.getloadavg()[0], 2)
    except (OSError, AttributeError):
        return None


def mapcv_info() -> dict[str, Any]:
    """The mapcv under test: version, where it is imported from, git state, key libraries."""
    import mapcv

    status = _git("status", "--porcelain")
    return {
        "version": _version("mapcv"),
        "module_path": str(mapcv.__file__),
        "git_commit": _git("rev-parse", "HEAD"),
        "git_branch": _git("rev-parse", "--abbrev-ref", "HEAD"),
        "git_dirty": None if status is None else bool(status),
        "libraries": {name: _version(name) for name in LIBRARIES},
    }


def timestamp() -> str:
    """Current UTC time, ISO 8601."""
    return datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds")
