"""Run scenarios against the local tile server and collect results."""

from __future__ import annotations

import json
import shutil
import subprocess
import sys
import tempfile
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

from shapely.geometry import Polygon

from benchmarks import baselines as baseline_registry
from benchmarks.checks import CheckReport, check_dataset, tree_hash
from benchmarks.machine import load_average, machine_info, mapcv_info, timestamp
from benchmarks.measure import (
    ProcessResult,
    child_env,
    mapcv_generate_command,
    run_process,
    summarise,
)
from benchmarks.scenarios import (
    PATCH,
    SCENARIOS,
    X0,
    Y0,
    ZOOM,
    Scenario,
    config_yaml,
    make_labels,
    region_bounds,
)

SCHEMA_VERSION = 1
MB_PER_PATCH = 0.11  # PNG image + mask of the synthetic tiles, measured; JPEG output is smaller
INTERRUPT_AT_FRACTION = 0.4  # resume scenarios send Ctrl-C once this share is written


class TileServerProcess:
    """The synthetic tile server in its own process, so it does not share a GIL with the harness."""

    def __init__(self) -> None:
        self._process = subprocess.Popen(
            [sys.executable, "-u", "-m", "benchmarks.tileserver"],
            stdout=subprocess.PIPE,
            env=child_env(),
            text=True,
        )
        assert self._process.stdout is not None
        line = self._process.stdout.readline()
        if not line.strip().isdigit():
            self.close()
            raise RuntimeError("the tile server did not report a port")
        self.port = int(line)

    def warm(self, scenario: Scenario) -> None:
        """Fetch each tile once so the timed runs do not pay for the server's PNG encoding."""

        def fetch(position: Tuple[int, int]) -> None:
            url = (
                f"http://127.0.0.1:{self.port}/{ZOOM}/{position[0]}/{position[1]}"
                f".{scenario.tile_format}"
            )
            try:
                urllib.request.urlopen(url, timeout=30).read()
            except urllib.error.URLError:
                pass

        positions = [(X0 + x, Y0 + y) for y in range(scenario.ny) for x in range(scenario.nx)]
        with ThreadPoolExecutor(max_workers=16) as pool:
            list(pool.map(fetch, positions))

    def close(self) -> None:
        self._process.terminate()
        try:
            self._process.wait(timeout=10)
        except subprocess.TimeoutExpired:
            self._process.kill()
        if self._process.stdout is not None:
            self._process.stdout.close()

    def __enter__(self) -> "TileServerProcess":
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()


@dataclass
class Context:
    """Settings shared by all scenarios of one suite run."""

    workdir: Path
    server: TileServerProcess
    repeat: int = 3
    baselines: Sequence[str] = ()
    log: Callable[[str], None] = field(default=lambda message: None)


def _prepare(
    scenario: Scenario, directory: Path, port: int
) -> Tuple[Path, List[Tuple[Polygon, int]]]:
    """Fresh directory with labels and config; returns the config path and label geometries."""
    shutil.rmtree(directory, ignore_errors=True)
    directory.mkdir(parents=True)
    geometries = make_labels(scenario, directory / "labels.geojson")
    config = directory / "mapcv.yaml"
    config.write_text(config_yaml(scenario, port), encoding="utf-8")
    return config, geometries


def _generate(config: Path, tag: str, **kwargs: Any) -> ProcessResult:
    directory = config.parent
    return run_process(
        mapcv_generate_command(config, directory / f"stages-{tag}.json"),
        directory,
        directory / f"generate-{tag}.log",
        stages_file=directory / f"stages-{tag}.json",
        **kwargs,
    )


def _tail(result: ProcessResult) -> str:
    return "\n    ".join(result.log_tail(8).splitlines())


def _failure(result: ProcessResult, what: str) -> str:
    return f"{what}: exited {result.exit_code}\n    {_tail(result)}"


def _stage_medians(results: Sequence[ProcessResult]) -> Dict[str, float]:
    names = sorted({name for result in results for name in result.stages})
    return {
        name: summarise([r.stages[name] for r in results if name in r.stages])["median"]
        for name in names
    }


def _timing_summary(
    results: Sequence[ProcessResult], tiles: int, patches: Optional[int]
) -> Dict[str, Any]:
    wall = summarise([r.wall_s for r in results])
    summary: Dict[str, Any] = {"wall_s": wall, "stages_median_s": _stage_medians(results)}
    rss = [r.peak_rss_mb for r in results if r.peak_rss_mb is not None]
    if rss:
        summary["peak_rss_mb"] = summarise(rss)
    if wall:
        summary["tiles_per_s"] = round(tiles / wall["median"], 1)
        if patches:
            summary["patches_per_s"] = round(patches / wall["median"], 1)
    return summary


def _check_strict(result: ProcessResult) -> List[str]:
    problems = []
    if result.exit_code == 0:
        problems.append("strict policy: mapcv exited 0 although tiles failed")
    log = result.log.read_text(encoding="utf-8", errors="replace")
    if "Traceback" in log:
        problems.append("strict policy: failure ended in a Python traceback, not a clean message")
    if "fail" not in log.lower():
        problems.append("strict policy: the output never says that tiles failed")
    return problems


def _dataset_mb(dataset: Path) -> float:
    return round(sum(f.stat().st_size for f in dataset.rglob("*") if f.is_file()) / 2**20, 1)


def _run_generate_scenario(scenario: Scenario, ctx: Context) -> Dict[str, Any]:
    run_dir = ctx.workdir / scenario.name
    config, geometries = _prepare(scenario, run_dir, ctx.server.port)
    ctx.server.warm(scenario)
    results: List[ProcessResult] = []
    problems: List[str] = []
    report = CheckReport()
    reference_hash: Optional[str] = None
    output_mb: Optional[float] = None
    for repeat in range(ctx.repeat):
        shutil.rmtree(run_dir / "dataset", ignore_errors=True)
        result = _generate(config, str(repeat))
        results.append(result)
        ctx.log(
            f"    run {repeat + 1}/{ctx.repeat}: {result.wall_s:.2f} s, "
            f"{result.peak_rss_mb} MB peak, exit {result.exit_code}"
        )
        if scenario.kind == "strict":
            problems.extend(_check_strict(result))
            continue
        if result.exit_code != 0:
            problems.append(_failure(result, f"run {repeat + 1}"))
            continue
        if repeat == 0:
            report = check_dataset(scenario, run_dir / "dataset", geometries)
            if ctx.repeat > 1:
                reference_hash = tree_hash(run_dir / "dataset")
            output_mb = _dataset_mb(run_dir / "dataset")
        elif reference_hash is not None and tree_hash(run_dir / "dataset") != reference_hash:
            problems.append(f"run {repeat + 1} produced different output than run 1")
    problems.extend(report.problems)

    outcome: Dict[str, Any] = {
        "runs": [result.to_json() for result in results],
        "checks": report.stats,
        "skipped": report.skipped,
    }
    patches = report.stats.get("patches")
    outcome["summary"] = _timing_summary(results, scenario.tiles, patches)
    if output_mb is not None:
        outcome["summary"]["output_mb"] = output_mb
    if ctx.baselines and scenario.kind == "generate":
        outcome["baselines"], baseline_problems = _run_baselines(scenario, ctx, run_dir)
        problems.extend(baseline_problems)
    outcome["problems"] = problems
    return outcome


def _run_baselines(
    scenario: Scenario, ctx: Context, run_dir: Path
) -> Tuple[Dict[str, Any], List[str]]:
    west, south, east, north = region_bounds(scenario)
    tile_url = (
        f"http://127.0.0.1:{ctx.server.port}/{scenario.tile_prefix()}"
        f"{{z}}/{{x}}/{{y}}.{scenario.tile_format}"
    )
    out: Dict[str, Any] = {}
    problems: List[str] = []
    for name in ctx.baselines:
        results: List[ProcessResult] = []
        for repeat in range(ctx.repeat):
            output = run_dir / f"baseline-{name}"
            shutil.rmtree(output, ignore_errors=True)
            output.mkdir()
            work = baseline_registry.Workload(
                scenario=scenario.name,
                tile_url=tile_url,
                zoom=ZOOM,
                region=(west, south, east, north),
                labels=run_dir / "labels.geojson",
                patch_size=PATCH,
                stride=scenario.stride,
                max_connections=16,
                output_dir=output,
            )
            command = baseline_registry.BASELINES[name].command(work)
            result = run_process(command, run_dir, run_dir / f"baseline-{name}-{repeat}.log")
            results.append(result)
            if result.exit_code != 0:
                problems.append(_failure(result, f"baseline {name}"))
        out[name] = {
            "runs": [r.to_json() for r in results],
            "summary": _timing_summary(results, scenario.tiles, None),
        }
    return out, problems


def _written_patches(manifest: Path) -> int:
    try:
        return len(json.loads(manifest.read_text(encoding="utf-8"))["patches"])
    except (OSError, ValueError, KeyError):
        return 0  # not written yet, or caught mid-replace


def _run_resume_scenario(scenario: Scenario, ctx: Context) -> Dict[str, Any]:
    """An uninterrupted run, then an interrupted + resumed one that must match it exactly."""
    if sys.platform == "win32":
        return {"skipped": ["resume needs SIGINT delivery, which Windows does not support"]}
    root = ctx.workdir / scenario.name
    shutil.rmtree(root, ignore_errors=True)
    problems: List[str] = []
    identical = False
    ctx.server.warm(scenario)

    reference_config, geometries = _prepare(scenario, root / "reference", ctx.server.port)
    reference = _generate(reference_config, "reference")
    ctx.log(f"    uninterrupted: {reference.wall_s:.2f} s, exit {reference.exit_code}")
    if reference.exit_code != 0:
        return {"problems": [_failure(reference, "uninterrupted run")]}
    report = check_dataset(scenario, root / "reference" / "dataset", geometries)
    problems.extend(report.problems)
    reference_hash = tree_hash(root / "reference" / "dataset")

    config, _ = _prepare(scenario, root / "resumed", ctx.server.port)
    manifest = root / "resumed" / "dataset" / "manifest.json"
    target = int(report.stats.get("patches", scenario.expected_patches()) * INTERRUPT_AT_FRACTION)
    first = _generate(
        config, "interrupted", interrupt_when=lambda: _written_patches(manifest) >= target
    )
    partial = _written_patches(manifest)
    ctx.log(f"    interrupted at {partial} patches: exit {first.exit_code}")
    if not first.interrupted or partial >= scenario.expected_patches():
        problems.append("the run finished before it could be interrupted; nothing was tested")
    if first.exit_code == 0:
        problems.append("an interrupted run exited 0")
    if "Traceback" in first.log.read_text(encoding="utf-8", errors="replace"):
        problems.append("Ctrl-C ended in a Python traceback, not a clean message")

    second = _generate(config, "resumed")
    ctx.log(f"    resumed: {second.wall_s:.2f} s, exit {second.exit_code}")
    if second.exit_code != 0:
        problems.append(_failure(second, "resumed run"))
    else:
        resumed = check_dataset(scenario, root / "resumed" / "dataset", geometries)
        problems.extend(f"resumed run: {problem}" for problem in resumed.problems)
        identical = tree_hash(root / "resumed" / "dataset") == reference_hash
        if not identical:
            problems.append("the resumed dataset differs from the uninterrupted one")

    return {
        "runs": {
            "uninterrupted": reference.to_json(),
            "interrupted": first.to_json(),
            "resumed": second.to_json(),
        },
        "summary": {
            "patches_at_interrupt": partial,
            "uninterrupted_wall_s": round(reference.wall_s, 3),
            "resumed_wall_s": round(second.wall_s, 3),
            "identical_to_uninterrupted": identical,
        },
        "checks": report.stats,
        "skipped": report.skipped,
        "problems": problems,
    }


def _disk_problem(scenario: Scenario, ctx: Context) -> Optional[str]:
    """A message if the work directory is too small for the scenario's output, else None."""
    copies = 2 if scenario.kind == "resume" else 1
    needed = scenario.expected_patches() * MB_PER_PATCH * copies * 1.3
    free = shutil.disk_usage(ctx.workdir).free / 2**20
    if free >= needed:
        return None
    return (
        f"needs about {needed / 1024:.1f} GB of disk in {ctx.workdir} but only "
        f"{free / 1024:.1f} GB is free (a temp dir can be a small RAM-backed tmpfs); "
        "pass --workdir on a larger disk"
    )


def run_scenario(scenario: Scenario, ctx: Context) -> Dict[str, Any]:
    """Run one scenario; the returned dict is its entry in the results file."""
    started = time.perf_counter()
    no_room = _disk_problem(scenario, ctx)
    if no_room:
        outcome: Dict[str, Any] = {"problems": [no_room]}
    elif scenario.kind == "resume":
        outcome = _run_resume_scenario(scenario, ctx)
    else:
        outcome = _run_generate_scenario(scenario, ctx)
    outcome.setdefault("problems", [])
    outcome.setdefault("skipped", [])
    outcome["definition"] = scenario.to_json()
    outcome["description"] = scenario.description
    outcome["duration_s"] = round(time.perf_counter() - started, 1)
    outcome["status"] = "failed" if outcome["problems"] else "ok"
    return outcome


def run_suite(
    names: Sequence[str],
    *,
    repeat: int,
    label: str = "",
    mode: str = "custom",
    baselines: Sequence[str] = (),
    workdir: Optional[Path] = None,
    keep: bool = False,
    log: Callable[[str], None] = lambda message: None,
) -> Dict[str, Any]:
    """Run the named scenarios and return the complete results document."""
    unknown = [name for name in names if name not in SCENARIOS]
    if unknown:
        raise ValueError(f"unknown scenario(s) {unknown}; known: {sorted(SCENARIOS)}")
    missing = [name for name in baselines if name not in baseline_registry.BASELINES]
    if missing:
        registered = sorted(baseline_registry.BASELINES) or "none registered yet"
        raise ValueError(f"unknown baseline(s) {missing}; available: {registered}")

    temporary = workdir is None
    root = Path(tempfile.mkdtemp(prefix="mapcv-bench-")) if workdir is None else workdir
    root.mkdir(parents=True, exist_ok=True)
    document: Dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "created": timestamp(),
        "label": label,
        "mode": mode,
        "repeat": repeat,
        "machine": {**machine_info(), "load_avg_1m_at_start": load_average()},
        "mapcv": mapcv_info(),
        "tile_server": {
            "kind": "local synthetic XYZ server on 127.0.0.1, run in its own process",
            "note": "the server shares the machine's CPU with mapcv; tiles are pre-encoded",
        },
        "scenarios": {},
    }
    try:
        with TileServerProcess() as server:
            ctx = Context(root, server, repeat=repeat, baselines=baselines, log=log)
            for name in names:
                scenario = SCENARIOS[name]
                log(f"== {name}: {scenario.description}")
                document["scenarios"][name] = run_scenario(scenario, ctx)
                entry = document["scenarios"][name]
                log(f"   {entry['status']} in {entry['duration_s']} s")
    finally:
        if temporary and not keep:
            shutil.rmtree(root, ignore_errors=True)
    document["machine"]["load_avg_1m_at_end"] = load_average()
    document["workdir"] = str(root) if (keep or not temporary) else None
    document["problems"] = [
        f"{name}: {problem}"
        for name, entry in document["scenarios"].items()
        for problem in entry["problems"]
    ]
    document["ok"] = not document["problems"]
    return document
