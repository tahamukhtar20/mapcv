"""Optional online run: the repository's quickstart example against Esri World Imagery.

Reported separately from the offline scenarios because it measures the network
and the provider as much as mapcv. It is small (77 tiles, polite concurrency),
runs once, never in CI, and only checks that the run succeeds and writes a
consistent dataset; pixels cannot be verified against a live service.

You are responsible for the provider's terms (see PROVIDERS.md).
"""

from __future__ import annotations

import json
import shutil
import tempfile
from pathlib import Path
from typing import Any

from benchmarks.measure import ROOT, mapcv_generate_command, run_process

QUICKSTART = ROOT / "examples" / "quickstart"


def run_online(workdir: Path | None = None) -> dict[str, Any]:
    """Run ``examples/quickstart`` once and return timings and a few sanity numbers."""
    if not QUICKSTART.is_dir():
        return {"status": "skipped", "problems": [], "reason": f"{QUICKSTART} not found"}
    temporary = workdir is None
    root = Path(tempfile.mkdtemp(prefix="mapcv-online-")) if workdir is None else workdir
    run_dir = root / "online"
    shutil.rmtree(run_dir, ignore_errors=True)
    shutil.copytree(QUICKSTART, run_dir)
    try:
        stages = run_dir / "stages.json"
        result = run_process(
            mapcv_generate_command(run_dir / "mapcv.yaml", stages),
            run_dir,
            run_dir / "generate.log",
            stages_file=stages,
        )
        problems = []
        patches = 0
        if result.exit_code != 0:
            problems.append(f"exited {result.exit_code}:\n    {result.log_tail(8)}")
        else:
            manifest = json.loads((run_dir / "dataset" / "manifest.json").read_text("utf-8"))
            patches = len(manifest["patches"])
            images = len(list((run_dir / "dataset" / "Images").iterdir()))
            if patches == 0 or images != patches:
                problems.append(f"{patches} patches in the manifest, {images} images on disk")
        return {
            "status": "failed" if problems else "ok",
            "problems": problems,
            "source": "esri_satellite, examples/quickstart/mapcv.yaml",
            "run": result.to_json(),
            "patches": patches,
            "note": "includes real network latency and Esri's response times; run once",
        }
    finally:
        if temporary:
            shutil.rmtree(root, ignore_errors=True)
