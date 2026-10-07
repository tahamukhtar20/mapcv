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

Registered: ``rasterio-script`` (``baseline_scripts/rasterio_script.py``), the
hand-rolled rasterio + Pillow pipeline. Its output is compared with mapcv's patch by
patch (:func:`benchmarks.checks.compare_with_mapcv`); a baseline whose data differs is
reported as such next to its time.

Example of another one::

    class MyScript:
        name = "rasterio-script"

        def command(self, work: Workload) -> list[str]:
            return [sys.executable, "my_baseline.py", work.tile_url, str(work.labels),
                    str(work.output_dir)]

    register(MyScript())
"""

from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Protocol, Tuple


@dataclass(frozen=True)
class Workload:
    """Everything a baseline needs to do the same job as ``mapcv generate``."""

    scenario: str
    tile_url: str  # XYZ template with {z}, {x}, {y}, served by the local tile server
    zoom: int
    region: Tuple[float, float, float, float]  # west, south, east, north in degrees
    labels: Path  # GeoJSON, property "class" holds the class name
    patch_size: int
    stride: int
    max_connections: int  # fetch concurrency mapcv uses; give baselines the same
    output_dir: Path  # empty, writable; the baseline writes its dataset here


class Baseline(Protocol):
    """A comparison implementation, run as one child process per repeat."""

    name: str

    def command(self, work: Workload) -> List[str]:
        """The command line to run (cwd is ``work.output_dir``'s parent)."""
        ...


BASELINES: Dict[str, Baseline] = {}


def register(baseline: Baseline) -> None:
    """Make ``baseline`` selectable with ``--baselines``."""
    BASELINES[baseline.name] = baseline


_SCRIPTS = Path(__file__).resolve().parent / "baseline_scripts"


class RasterioScript:
    """A well-written rasterio + Pillow script (see ``baseline_scripts/rasterio_script.py``)."""

    name = "rasterio-script"

    def command(self, work: Workload) -> List[str]:
        west, south, east, north = work.region
        return [
            sys.executable,
            str(_SCRIPTS / "rasterio_script.py"),
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


register(RasterioScript())
