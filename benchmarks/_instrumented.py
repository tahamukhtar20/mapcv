"""Run ``mapcv generate`` with per-stage timers; used as a child process by the harness.

    python -m benchmarks._instrumented STAGES_JSON generate CONFIG --yes

The real command line is executed in-process (same code path as the ``mapcv``
script); only wall-clock timers are wrapped around the pipeline's building
blocks, so the overhead is a few ``perf_counter`` calls per chunk. Stage
seconds are written to STAGES_JSON when the process exits, also after Ctrl-C.

Stages (seconds, summed over the run):

``plan``      the up-front plan and estimate that ``mapcv generate`` prints
``open``      opening the imagery source
``labels``    parsing, reprojecting and indexing labels
``fetch``     HTTP requests for tiles (Rust fetcher, includes network wait)
``decode``    decoding fetched tiles into the strip, excluding ``fetch``
``rasterize`` burning label polygons into the strip
``sample``    cutting patches out of the strip
``write``     encoding and writing patch images and masks (Rust writer)
``manifest``  saving ``manifest.json`` after each chunk
``split``     the train/val/test split
``total``     everything from just before the CLI starts to exit; the harness's
              wall time additionally includes interpreter start-up and imports
"""

from __future__ import annotations

import atexit
import json
import sys
import time
from collections import defaultdict
from functools import wraps
from pathlib import Path
from typing import Any, Callable, DefaultDict, List, TypeVar

F = TypeVar("F", bound=Callable[..., Any])

_stages: DefaultDict[str, float] = defaultdict(float)


def _timed(stage: str, func: F) -> F:
    @wraps(func)
    def wrapper(*args: Any, **kwargs: Any) -> Any:
        started = time.perf_counter()
        try:
            return func(*args, **kwargs)
        finally:
            _stages[stage] += time.perf_counter() - started

    return wrapper  # type: ignore[return-value]


def _install() -> None:
    from mapcv import cli, pipeline
    from mapcv.imagery import XYZRasterSource
    from mapcv.targets import segmentation
    from mapcv.writer import Manifest
    from mapcv.writers import FilesWriter

    cli.make_plan = _timed("plan", cli.make_plan)
    pipeline.open_raster_source = _timed("open", pipeline.open_raster_source)
    segmentation.SegmentationTarget.prepare = _timed(  # type: ignore[method-assign]
        "labels", segmentation.SegmentationTarget.prepare
    )
    segmentation.rasterize = _timed("rasterize", segmentation.rasterize)
    pipeline.sample_annotated_patches = _timed("sample", pipeline.sample_annotated_patches)
    FilesWriter.write = _timed("write", FilesWriter.write)  # type: ignore[method-assign]
    pipeline.split_manifest = _timed("split", pipeline.split_manifest)
    Manifest.save = _timed("manifest", Manifest.save)  # type: ignore[method-assign]
    # read_window includes the fetch; the fetch is subtracted afterwards to get decode time.
    XYZRasterSource.read_window = _timed(  # type: ignore[method-assign]
        "read_window", XYZRasterSource.read_window
    )
    XYZRasterSource._fetch = _timed("fetch", XYZRasterSource._fetch)  # type: ignore[method-assign]


def _dump(path: Path, started: float) -> None:
    stages = dict(_stages)
    if "read_window" in stages:
        stages["decode"] = max(0.0, stages.pop("read_window") - stages.get("fetch", 0.0))
    stages["total"] = time.perf_counter() - started
    path.write_text(json.dumps({k: round(v, 4) for k, v in stages.items()}), encoding="utf-8")


def main(argv: List[str]) -> None:
    """``argv`` is the stage file path followed by the ``mapcv`` arguments."""
    stages_path = Path(argv[0])
    started = time.perf_counter()
    _install()
    atexit.register(_dump, stages_path, started)
    from mapcv.cli import app

    app(args=argv[1:], prog_name="mapcv")


if __name__ == "__main__":
    main(sys.argv[1:])
