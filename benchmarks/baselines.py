"""Comparison baselines: other ways of building the same dataset.

A baseline is anything that turns the same inputs mapcv gets into image/mask
patches: a rasterio + geopandas script, TorchGeo, leafmap, and so on. Register
one here and ``python -m benchmarks run --baselines NAME`` runs it on every
selected scenario, against the same tile server, with the same repeat count,
and records wall time, peak memory (whole process tree) and stage-free
summaries next to mapcv's under ``scenarios.<name>.baselines.<NAME>``.

The harness only measures a baseline; it does not judge its output. A baseline
that writes mapcv's layout (``Images/``, ``Masks/``, ``manifest.json``,
``splits/``) can be passed to :func:`benchmarks.checks.check_dataset` by its
own wrapper; for fairness, give it the same fetch concurrency as mapcv
(``Workload.max_connections``).

Registered (scripts in ``baseline_scripts/``):

- ``rasterio-script``: the hand-rolled rasterio + Pillow pipeline.
- ``gdal-cli``: GDAL's programs (TMS driver fetch, ``gdal_rasterize``, ``gdal_translate
  -srcwin`` per patch). Its programs come from ``GDAL_BIN`` or ``PATH``.
- ``torchgeo``: ``RasterDataset`` & ``VectorDataset`` sampled by ``GridGeoSampler``. Runs
  with the Python in ``MAPCV_BENCH_TORCHGEO_PYTHON`` (default: this one).
- ``leafmap``: ``map_tiles_to_geotiff``, then rasterio windows. Runs with the Python in
  ``MAPCV_BENCH_LEAFMAP_PYTHON`` (needs GDAL's ``osgeo`` bindings).

Every one's output is compared with mapcv's patch by patch
(:func:`benchmarks.checks.compare_with_mapcv`). ``rasterio-script`` and ``gdal-cli``
are *references*: they burn the labels with GDAL's rules, so data that differs from
theirs is a mapcv problem and fails the run. For the other tools a difference is a
finding about that tool, reported next to its time. A baseline whose tool is not
installed is skipped with the reason.

Example of another one::

    class MyScript:
        name = "my-script"
        reference = False

        def command(self, work: Workload) -> list[str]:
            return [sys.executable, "my_baseline.py", work.tile_url, str(work.labels),
                    str(work.output_dir)]

        def missing(self) -> str | None:
            return None

    register(MyScript())
"""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol


@dataclass(frozen=True)
class Workload:
    """Everything a baseline needs to do the same job as ``mapcv generate``."""

    scenario: str
    tile_url: str  # XYZ template with {z}, {x}, {y}, served by the local tile server
    zoom: int
    region: tuple[float, float, float, float]  # west, south, east, north in degrees
    labels: Path  # GeoJSON, property "class" holds the class name
    patch_size: int
    stride: int
    max_connections: int  # fetch concurrency mapcv uses; give baselines the same
    output_dir: Path  # empty, writable; the baseline writes its dataset here


class Baseline(Protocol):
    """A comparison implementation, run as one child process per repeat."""

    name: str
    # Data that differs from a reference baseline's is a mapcv problem.
    reference: bool

    def command(self, work: Workload) -> list[str]:
        """The command line to run (cwd is ``work.output_dir``'s parent)."""
        ...

    def missing(self) -> str | None:
        """Why the baseline cannot run here (its tool is not installed), or ``None``."""
        ...


BASELINES: dict[str, Baseline] = {}


def register(baseline: Baseline) -> None:
    """Make ``baseline`` selectable with ``--baselines``."""
    BASELINES[baseline.name] = baseline


_SCRIPTS = Path(__file__).resolve().parent / "baseline_scripts"


def _arguments(work: Workload) -> list[str]:
    """The arguments every baseline script takes, in order."""
    west, south, east, north = work.region
    return [
        work.tile_url,
        str(work.zoom),
        repr(west),
        repr(south),
        repr(east),
        repr(north),
        str(work.labels),
        str(work.patch_size),
        str(work.stride),
        str(work.max_connections),
        str(work.output_dir),
    ]


def _missing_modules(python: str, modules: list[str]) -> str | None:
    """Why ``python`` cannot import ``modules``, or ``None``."""
    if shutil.which(python) is None and not os.path.exists(python):
        return f"{python} not found"
    probe = subprocess.run(
        [python, "-c", "import " + ", ".join(modules)],
        capture_output=True,
        text=True,
        check=False,
    )
    if probe.returncode != 0:
        lines = probe.stderr.strip().splitlines()
        return f"{python} cannot import {', '.join(modules)}: {lines[-1] if lines else ''}"
    return None


class _PythonScript:
    """A baseline script run by an interpreter that has its tool installed."""

    name = ""
    script = ""
    modules: list[str] = []
    python_variable = ""
    reference = False

    @property
    def python(self) -> str:
        return os.environ.get(self.python_variable, sys.executable)

    def command(self, work: Workload) -> list[str]:
        return [self.python, str(_SCRIPTS / self.script), *_arguments(work)]

    def missing(self) -> str | None:
        return _missing_modules(self.python, self.modules)


class RasterioScript(_PythonScript):
    """A well-written rasterio + Pillow script (see ``baseline_scripts/rasterio_script.py``)."""

    name = "rasterio-script"
    script = "rasterio_script.py"
    modules = ["rasterio", "mercantile", "shapely", "pyproj"]
    reference = True


class GdalCli(_PythonScript):
    """GDAL's command-line programs (see ``baseline_scripts/gdal_cli.py``)."""

    name = "gdal-cli"
    script = "gdal_cli.py"
    reference = True

    def missing(self) -> str | None:
        folder = os.environ.get("GDAL_BIN")
        for program in ("gdal_translate", "gdal_rasterize", "ogr2ogr"):
            if (shutil.which(program, path=folder) if folder else shutil.which(program)) is None:
                return f"{program} not found (install GDAL or set GDAL_BIN)"
        return None


class TorchGeo(_PythonScript):
    """TorchGeo's datasets and grid sampler (see ``baseline_scripts/torchgeo_script.py``)."""

    name = "torchgeo"
    script = "torchgeo_script.py"
    modules = ["torchgeo", "mercantile"]
    python_variable = "MAPCV_BENCH_TORCHGEO_PYTHON"


class Leafmap(_PythonScript):
    """leafmap's tile download (see ``baseline_scripts/leafmap_script.py``)."""

    name = "leafmap"
    script = "leafmap_script.py"
    modules = ["leafmap", "osgeo.gdal", "mercantile", "rasterio"]
    python_variable = "MAPCV_BENCH_LEAFMAP_PYTHON"


for _baseline in (RasterioScript(), GdalCli(), TorchGeo(), Leafmap()):
    register(_baseline)
