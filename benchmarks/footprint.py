"""Install footprint: what ``pip install mapcv`` costs in a fresh virtual environment.

Needs network access (and, by default, a published mapcv release). Pass a
different requirement to measure something else, for example a local wheel
(``dist/mapcv-0.2.0-cp310-abi3-manylinux_2_17_x86_64.whl``).
"""

from __future__ import annotations

import json
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import Any, Dict, List

from benchmarks.measure import child_env

# Present in every fresh venv or added by pip itself; not mapcv's dependencies.
_TOOLING = {"pip", "setuptools", "wheel"}


def _run(command: List[str], cwd: Path) -> str:
    result = subprocess.run(
        command, cwd=cwd, capture_output=True, text=True, env=child_env(), timeout=1800
    )
    if result.returncode != 0:
        raise RuntimeError(f"{' '.join(command[:4])} failed:\n{result.stderr.strip()[-800:]}")
    return result.stdout


def _dir_mb(path: Path) -> float:
    return round(sum(f.stat().st_size for f in path.rglob("*") if f.is_file()) / 2**20, 1)


def measure_footprint(requirement: str = "mapcv") -> Dict[str, Any]:
    """Wheel size, dependency count, install time and disk size for ``pip install REQUIREMENT``."""
    with tempfile.TemporaryDirectory(prefix="mapcv-footprint-") as tmp:
        work = Path(tmp)
        env_dir = work / "venv"
        _run([sys.executable, "-m", "venv", str(env_dir)], work)
        bin_dir = env_dir / ("Scripts" if sys.platform == "win32" else "bin")
        python = str(bin_dir / "python")

        wheel_dir = work / "wheel"
        local = Path(requirement)
        if local.is_file():
            wheels = [local]
        else:
            _run(
                [python, "-m", "pip", "download", "--no-deps", "--no-cache-dir", "-q"]
                + ["-d", str(wheel_dir), requirement],
                work,
            )
            wheels = sorted(wheel_dir.iterdir())

        site = Path(
            _run(
                [python, "-c", "import sysconfig; print(sysconfig.get_path('purelib'))"], work
            ).strip()
        )
        empty_mb = _dir_mb(site)  # pip itself, present before mapcv

        started = time.perf_counter()
        _run([python, "-m", "pip", "install", "--no-cache-dir", "-q", requirement], work)
        install_s = time.perf_counter() - started

        listing = json.loads(_run([python, "-m", "pip", "list", "--format=json"], work))
        installed = {item["name"].lower(): item["version"] for item in listing}
        direct = _run(
            [
                python,
                "-c",
                "from importlib.metadata import requires; "
                "print(chr(10).join(r for r in requires('mapcv') or [] if 'extra ==' not in r))",
            ],
            work,
        ).splitlines()
        names = sorted(set(installed) - _TOOLING - {"mapcv"})
        return {
            "requirement": requirement,
            "mapcv_version": installed.get("mapcv"),
            "wheel_files": [w.name for w in wheels],
            "wheel_mb": round(sum(w.stat().st_size for w in wheels) / 2**20, 2),
            "direct_dependencies": len([line for line in direct if line.strip()]),
            "total_dependencies": len(names),
            "dependencies": {name: installed[name] for name in names},
            "install_s": round(install_s, 1),
            "installed_mb": round(_dir_mb(site) - empty_mb, 1),
            "python": sys.version.split()[0],
            "note": "install time is network-bound and measured once with --no-cache-dir",
        }
