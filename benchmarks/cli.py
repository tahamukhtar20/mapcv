"""Command line: ``python -m benchmarks run|list``."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

from benchmarks import baselines as baseline_registry
from benchmarks.runner import run_suite
from benchmarks.scenarios import SCENARIOS, Group, names_in_group

GROUPS: Dict[str, Group] = {"quick": "quick", "standard": "standard", "large": "large"}
DEFAULT_REPEAT = 3


def expand(selection: Sequence[str]) -> List[str]:
    """Scenario names, with ``quick``/``standard``/``large``/``all`` expanded to their members."""
    names: List[str] = []
    for item in selection:
        if item == "all":
            expanded = list(SCENARIOS)
        elif item in GROUPS:
            expanded = names_in_group(GROUPS[item])
        elif item in SCENARIOS:
            expanded = [item]
        else:
            raise SystemExit(
                f"unknown scenario or group '{item}'; run `python -m benchmarks list` to see them"
            )
        names.extend(name for name in expanded if name not in names)
    return names


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m benchmarks",
        description="Reproducible mapcv benchmarks: offline, checked, repeated.",
    )
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("list", help="show scenarios and registered baselines")
    compare = commands.add_parser(
        "compare", help="a table of mapcv against the baselines from a results file"
    )
    compare.add_argument("results", type=Path, metavar="FILE")
    compare.add_argument(
        "--markdown", action="store_true", help="a Markdown table (for the docs Performance page)"
    )

    run = commands.add_parser("run", help="run scenarios and write a JSON results file")
    run.add_argument(
        "--scenarios",
        nargs="+",
        metavar="NAME",
        help="scenario names or groups (quick, standard, large, all); default: standard",
    )
    run.add_argument(
        "--quick",
        action="store_true",
        help="a few seconds: the tiny 'quick' scenarios, one repeat (what the pytest smoke test runs)",
    )
    run.add_argument(
        "--repeat",
        type=int,
        default=None,
        metavar="N",
        help=f"runs per scenario, for median and spread (default {DEFAULT_REPEAT}, 1 with --quick)",
    )
    run.add_argument("--out", type=Path, default=Path("benchmark-results.json"), metavar="FILE")
    run.add_argument(
        "--label", default="", help="free text stored in the results, e.g. 'release build'"
    )
    run.add_argument(
        "--workdir", type=Path, help="keep inputs and outputs here (default: temp dir)"
    )
    run.add_argument("--keep", action="store_true", help="do not delete the temp work directory")
    run.add_argument("--baselines", nargs="+", default=[], metavar="NAME")
    run.add_argument(
        "--online",
        action="store_true",
        help="also run the quickstart against Esri (needs network; reported separately)",
    )
    run.add_argument(
        "--footprint",
        nargs="?",
        const="mapcv",
        metavar="REQUIREMENT",
        help="also measure `pip install REQUIREMENT` (default mapcv) in a fresh venv (needs network)",
    )
    return parser


def _row(name: str, entry: Dict[str, Any]) -> str:
    summary = entry.get("summary", {})
    if "uninterrupted_wall_s" in summary:
        timing = (
            f"uninterrupted {summary['uninterrupted_wall_s']:>6.2f} s, resumed "
            f"{summary['resumed_wall_s']:.2f} s from {summary['patches_at_interrupt']} patches"
        )
    elif summary.get("wall_s"):
        wall, rss = summary["wall_s"], summary.get("peak_rss_mb", {})
        timing = (
            f"{wall['median']:>8.2f} s (±{wall['stdev']:.2f}, n={wall['n']})  "
            f"{summary.get('tiles_per_s', 0):>7.1f} tiles/s  "
            f"{rss.get('median', float('nan')):>7.0f} MB"
        )
    else:
        timing = "-"
    return f"  {name:<12} {entry['status']:<7} {timing}"


def print_report(document: Dict[str, Any], out: Path) -> None:
    print("\nscenario     status  wall time (median)  throughput  peak RSS")
    for name, entry in document["scenarios"].items():
        print(_row(name, entry))
    online = document.get("online")
    if online:
        wall = online.get("run", {}).get("wall_s")
        print(
            f"  online       {online['status']:<7} {wall} s (Esri quickstart, reported separately)"
        )
    footprint = document.get("footprint")
    if footprint and "error" not in footprint:
        print(
            f"  footprint    {footprint['wheel_mb']} MB wheel, {footprint['total_dependencies']} "
            f"dependencies, {footprint['installed_mb']} MB installed, "
            f"{footprint['install_s']} s to install"
        )
    machine = document["machine"]
    load, cpus = machine.get("load_avg_1m_at_start"), machine["cpu_count_logical"]
    if load is not None and cpus and load > cpus / 2:
        print(
            f"\nNOTE: the load average was {load} on {cpus} CPUs when the run started; "
            "other work on this machine adds noise to the timings."
        )
    skipped = sorted(
        {note for entry in document["scenarios"].values() for note in entry["skipped"]}
    )
    for note in skipped:
        print(f"\nNOTE: {note}")
    if document["problems"]:
        print(f"\nFAILED, {len(document['problems'])} problem(s):")
        for problem in document["problems"]:
            print(f"  - {problem}")
    else:
        print("\nAll correctness checks passed.")
    print(f"Results written to {out}")


def compare_table(document: Dict[str, Any], markdown: bool = False) -> str:
    """mapcv and each baseline per scenario: median wall time, peak memory, and whether
    the baseline's data matched mapcv's (a time is only comparable when it did)."""
    rows = [("scenario", "tool", "wall time", "peak RSS", "vs mapcv", "same data")]
    for name, entry in document.get("scenarios", {}).items():
        summary = entry.get("summary", {})
        if not summary.get("wall_s"):
            continue
        ours = summary["wall_s"]["median"]
        rows.append(
            (
                name,
                "mapcv",
                f"{ours:.2f} s",
                f"{summary['peak_rss_mb']['median']:.0f} MB",
                "1.00×",
                "—",
            )
        )
        for tool, result in entry.get("baselines", {}).items():
            if "skipped" in result:
                rows.append((name, tool, "skipped", "—", "—", "—"))
                continue
            theirs = result["summary"]["wall_s"]["median"]
            comparison = result.get("comparison", {})
            same = comparison.get("same_data")
            ratio = f"{theirs / ours:.2f}×" if same else "not comparable"
            rows.append(
                (
                    name,
                    tool,
                    f"{theirs:.2f} s",
                    f"{result['summary']['peak_rss_mb']['median']:.0f} MB",
                    ratio,
                    "yes" if same else "no" if same is False else "unchecked",
                )
            )
    if markdown:
        lines = ["| " + " | ".join(rows[0]) + " |", "|" + " --- |" * len(rows[0])]
        lines += ["| " + " | ".join(row) + " |" for row in rows[1:]]
        return "\n".join(lines)
    widths = [max(len(row[i]) for row in rows) for i in range(len(rows[0]))]
    return "\n".join(
        "  ".join(cell.ljust(w) for cell, w in zip(row, widths)).rstrip() for row in rows
    )


def main(argv: Optional[Sequence[str]] = None) -> int:
    """Entry point; returns the process exit code (non-zero if any check failed)."""
    args = build_parser().parse_args(argv)
    if args.command == "compare":
        print(compare_table(json.loads(args.results.read_text(encoding="utf-8")), args.markdown))
        return 0
    if args.command == "list":
        for group in GROUPS:
            print(f"{group}:")
            for name in names_in_group(GROUPS[group]):
                print(f"  {name:<12} {SCENARIOS[name].description}")
        print("baselines:", ", ".join(sorted(baseline_registry.BASELINES)) or "none registered")
        return 0

    if args.quick:
        names = names_in_group("quick")
        mode = "quick"
    else:
        names = expand(args.scenarios or ["standard"])
        mode = "custom" if args.scenarios else "standard"
    repeat = args.repeat or (1 if args.quick else DEFAULT_REPEAT)
    if repeat < 1:
        raise SystemExit("--repeat must be at least 1")

    def log(message: str) -> None:
        print(message, flush=True)

    try:
        document = run_suite(
            names,
            repeat=repeat,
            label=args.label,
            mode=mode,
            baselines=args.baselines,
            workdir=args.workdir,
            keep=args.keep,
            log=log,
        )
    except ValueError as error:
        raise SystemExit(str(error))

    if args.online:
        from benchmarks.online import run_online

        log("== online: examples/quickstart against Esri (see PROVIDERS.md for the terms)")
        document["online"] = run_online()
        document["problems"] += [f"online: {p}" for p in document["online"]["problems"]]
    if args.footprint:
        from benchmarks.footprint import measure_footprint

        log(f"== footprint: pip install {args.footprint} in a fresh venv")
        try:
            document["footprint"] = measure_footprint(args.footprint)
        except (RuntimeError, OSError) as error:
            document["footprint"] = {"error": str(error)}
            document["problems"].append(f"footprint: {error}")
    document["ok"] = not document["problems"]

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(document, indent=2) + "\n", encoding="utf-8")
    print_report(document, args.out)
    return 0 if document["ok"] else 1
