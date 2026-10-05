"""Run a command as a child process; measure wall time and peak memory of its process tree."""

from __future__ import annotations

import json
import os
import signal
import statistics
import subprocess
import sys
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence

try:
    import psutil
except ImportError:  # peak memory is then reported as null
    psutil = None

ROOT = Path(__file__).resolve().parent.parent
POLL_SECONDS = 0.02


@dataclass
class ProcessResult:
    """What one child process did."""

    wall_s: float
    peak_rss_mb: Optional[float]
    exit_code: int
    interrupted: bool = False
    stages: Dict[str, float] = field(default_factory=dict)
    log: Path = Path()

    def log_tail(self, lines: int = 12) -> str:
        try:
            text = self.log.read_text(encoding="utf-8", errors="replace")
        except OSError:
            return ""
        return "\n".join(text.strip().splitlines()[-lines:])

    def to_json(self) -> Dict[str, Any]:
        return {
            "wall_s": round(self.wall_s, 3),
            "peak_rss_mb": self.peak_rss_mb,
            "exit_code": self.exit_code,
            "interrupted": self.interrupted,
            "stages": self.stages,
        }


def child_env() -> Dict[str, str]:
    """Environment for children: importable ``benchmarks``, plain unstyled output."""
    env = dict(os.environ)
    python_path = env.get("PYTHONPATH")
    env["PYTHONPATH"] = f"{ROOT}{os.pathsep}{python_path}" if python_path else str(ROOT)
    env.update(COLUMNS="100", TERM="dumb", NO_COLOR="1", PYTHONUNBUFFERED="1")
    return env


def _tree_rss(root: "psutil.Process") -> int:
    total = root.memory_info().rss
    for child in root.children(recursive=True):
        try:
            total += child.memory_info().rss
        except psutil.Error:
            pass  # exited between listing and reading
    return int(total)


def run_process(
    command: Sequence[str],
    cwd: Path,
    log: Path,
    *,
    stages_file: Optional[Path] = None,
    interrupt_when: Optional[Callable[[], bool]] = None,
    timeout_s: float = 3600,
) -> ProcessResult:
    """Run ``command`` to completion, sampling the summed RSS of it and its children.

    With ``interrupt_when``, SIGINT is sent the first time that callable returns
    True (it is polled every 20 ms while the process runs).
    """
    peak = 0
    stop = threading.Event()
    with open(log, "w", encoding="utf-8") as handle:
        process = subprocess.Popen(
            list(command),
            cwd=cwd,
            stdout=handle,
            stderr=subprocess.STDOUT,
            env=child_env(),
        )
        started = time.perf_counter()

        def poll_memory() -> None:
            nonlocal peak
            while not stop.is_set():
                try:
                    peak = max(peak, _tree_rss(psutil.Process(process.pid)))
                except psutil.Error:
                    pass
                time.sleep(POLL_SECONDS)

        sampler = None
        if psutil is not None:
            sampler = threading.Thread(target=poll_memory, daemon=True)
            sampler.start()
        interrupted = False
        try:
            while process.poll() is None:
                if time.perf_counter() - started > timeout_s:
                    process.kill()
                    break
                if interrupt_when is not None and not interrupted and interrupt_when():
                    process.send_signal(signal.SIGINT)
                    interrupted = True
                time.sleep(POLL_SECONDS / 2)
            exit_code = process.wait()
        finally:
            wall = time.perf_counter() - started
            stop.set()
            if sampler is not None:
                sampler.join()
    stages: Dict[str, float] = {}
    if stages_file is not None and stages_file.exists():
        try:
            stages = json.loads(stages_file.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            stages = {}
    return ProcessResult(
        wall_s=wall,
        peak_rss_mb=round(peak / 2**20, 1) if psutil is not None else None,
        exit_code=exit_code,
        interrupted=interrupted,
        stages=stages,
        log=log,
    )


def mapcv_generate_command(config: Path, stages_file: Path) -> List[str]:
    """``mapcv generate CONFIG --yes`` run in-process by the stage-timing wrapper."""
    return [
        sys.executable,
        "-m",
        "benchmarks._instrumented",
        str(stages_file),
        "generate",
        str(config),
        "--yes",
    ]


def summarise(values: Sequence[float]) -> Dict[str, float]:
    """Median and spread of repeated measurements (``stdev`` is 0 for a single run)."""
    if not values:
        return {}
    return {
        "n": len(values),
        "median": round(statistics.median(values), 3),
        "min": round(min(values), 3),
        "max": round(max(values), 3),
        "stdev": round(statistics.stdev(values), 3) if len(values) > 1 else 0.0,
    }
