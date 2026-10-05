"""Smoke test for the benchmark harness in benchmarks/, so it does not rot.

Runs the ``--quick`` suite (a few seconds): the synthetic tile server, the real
``mapcv generate`` in a child process, all correctness checks, and the JSON
results. The checks that need rasterio, pyproj and psutil (the ``bench``
dependency group, part of ``dev``) skip this module when they are missing, and so
does a source distribution, which leaves benchmarks/ out.
"""

from __future__ import annotations

import importlib.util
import json
import shutil
import sys
from pathlib import Path
from typing import Any, Dict, Iterator, List, Tuple

import numpy as np
import pytest
from PIL import Image
from shapely.geometry import Polygon

if importlib.util.find_spec("benchmarks") is None:
    pytest.skip("benchmarks/ is not shipped in the source distribution", allow_module_level=True)

from benchmarks import baselines, checks  # noqa: E402
from benchmarks.checks import CheckReport, check_dataset, tree_hash  # noqa: E402
from benchmarks.cli import expand, main  # noqa: E402
from benchmarks.measure import summarise  # noqa: E402
from benchmarks.runner import run_suite  # noqa: E402
from benchmarks.scenarios import SCENARIOS, make_labels, names_in_group  # noqa: E402
from benchmarks.tileserver import is_failing  # noqa: E402

pytestmark = [
    pytest.mark.skipif(
        any(importlib.util.find_spec(name) is None for name in ("psutil", "rasterio", "pyproj")),
        reason="needs the bench dependency group: uv sync --group bench",
    ),
    pytest.mark.skipif(
        sys.platform == "win32", reason="the harness is only validated on Linux and macOS"
    ),
]


@pytest.fixture(scope="module")
def quick_run(
    tmp_path_factory: pytest.TempPathFactory,
) -> Iterator[Tuple[int, Dict[str, Any], Path]]:
    """``python -m benchmarks run --quick``, once for the module: (exit code, results, workdir)."""
    base = tmp_path_factory.mktemp("bench")
    out = base / "results.json"
    code = main(["run", "--quick", "--out", str(out), "--workdir", str(base / "work")])
    yield code, json.loads(out.read_text(encoding="utf-8")), base / "work"


def test_quick_suite_passes_all_checks(quick_run: Tuple[int, Dict[str, Any], Path]) -> None:
    code, results, _ = quick_run

    assert results["problems"] == []
    assert code == 0
    assert results["ok"] is True
    assert set(results["scenarios"]) == set(names_in_group("quick"))
    for entry in results["scenarios"].values():
        assert entry["status"] == "ok"
        assert entry["skipped"] == []


def test_results_carry_machine_and_stage_info(quick_run: Tuple[int, Dict[str, Any], Path]) -> None:
    _, results, _ = quick_run

    assert results["schema_version"] == 1
    assert results["machine"]["cpu_count_logical"] >= 1
    assert results["machine"]["ram_gb"] > 0
    assert results["mapcv"]["version"]
    assert {"git_commit", "git_dirty", "libraries"} <= set(results["mapcv"])
    q = results["scenarios"]["Q"]
    assert q["summary"]["wall_s"]["n"] == 1
    assert q["summary"]["peak_rss_mb"]["median"] > 0
    assert {"fetch", "decode", "rasterize", "sample", "write", "split"} <= set(
        q["summary"]["stages_median_s"]
    )
    resume = results["scenarios"]["Q-resume"]["summary"]
    assert resume["identical_to_uninterrupted"] is True
    assert 0 < resume["patches_at_interrupt"] < SCENARIOS["Q-resume"].expected_patches()


@pytest.fixture()
def dataset_copy(
    quick_run: Tuple[int, Dict[str, Any], Path], tmp_path: Path
) -> Tuple[Path, List[Tuple[Polygon, int]]]:
    """A scratch copy of Q's dataset, safe to damage, with the label geometries."""
    _, _, work = quick_run
    shutil.copytree(work / "Q" / "dataset", tmp_path / "dataset")
    geometries = make_labels(SCENARIOS["Q"], tmp_path / "labels.geojson")
    return tmp_path / "dataset", geometries


def test_checks_pass_on_an_intact_dataset(
    dataset_copy: Tuple[Path, List[Tuple[Polygon, int]]],
) -> None:
    dataset, geometries = dataset_copy

    report = check_dataset(SCENARIOS["Q"], dataset, geometries)

    assert report.problems == []
    assert report.stats["patches_checked"] == 36


def test_checks_catch_a_changed_pixel(
    dataset_copy: Tuple[Path, List[Tuple[Polygon, int]]],
) -> None:
    dataset, geometries = dataset_copy
    path = next((dataset / "Images").iterdir())
    pixels = np.array(Image.open(path))
    pixels[10, 10, 0] ^= 1
    Image.fromarray(pixels).save(path)

    report = check_dataset(SCENARIOS["Q"], dataset, geometries)

    assert any("differs from the served tiles" in problem for problem in report.problems)


def test_checks_catch_a_changed_mask(
    dataset_copy: Tuple[Path, List[Tuple[Polygon, int]]],
) -> None:
    dataset, geometries = dataset_copy
    path = next((dataset / "Masks").iterdir())
    mask = np.array(Image.open(path))
    mask[:, :] = 3 - (mask[:, :] % 3)
    Image.fromarray(mask).save(path)

    report = check_dataset(SCENARIOS["Q"], dataset, geometries)

    assert any("rasterio" in problem or "class counts" in problem for problem in report.problems)


def test_mask_check_is_skipped_with_a_message_when_rasterio_is_missing(
    dataset_copy: Tuple[Path, List[Tuple[Polygon, int]]], monkeypatch: pytest.MonkeyPatch
) -> None:
    dataset, geometries = dataset_copy
    monkeypatch.setattr(checks, "HAVE_MASK_REFERENCE", False)

    report = check_dataset(SCENARIOS["Q"], dataset, geometries)

    assert report.problems == []
    assert report.skipped == [checks.MASK_REFERENCE_HINT]
    assert "mask_disagree_pct" not in report.stats


def test_checks_catch_a_patch_in_two_splits(
    dataset_copy: Tuple[Path, List[Tuple[Polygon, int]]],
) -> None:
    dataset, geometries = dataset_copy
    test_names = (dataset / "splits" / "test.txt").read_text(encoding="utf-8").splitlines()
    with open(dataset / "splits" / "train.txt", "a", encoding="utf-8") as handle:
        handle.write(test_names[0] + "\n")

    report = check_dataset(SCENARIOS["Q"], dataset, geometries)

    assert any("share" in problem for problem in report.problems)


def test_tree_hash_changes_with_content(
    dataset_copy: Tuple[Path, List[Tuple[Polygon, int]]],
) -> None:
    dataset, _ = dataset_copy
    before = tree_hash(dataset)

    next((dataset / "Images").iterdir()).write_bytes(b"x")

    assert tree_hash(dataset) != before


def test_main_exits_non_zero_when_a_check_fails(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    import benchmarks.runner

    def broken(*args: object, **kwargs: object) -> Any:
        report = CheckReport()
        report.problems.append("planted failure")
        return report

    monkeypatch.setattr(benchmarks.runner, "check_dataset", broken)
    out = tmp_path / "results.json"

    code = main(["run", "--scenarios", "Q", "--out", str(out), "--workdir", str(tmp_path / "w")])

    assert code == 1
    assert json.loads(out.read_text(encoding="utf-8"))["problems"] == ["Q: planted failure"]


def test_baseline_hook_measures_a_registered_baseline(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    class Noop:
        name = "noop"

        def command(self, work: baselines.Workload) -> List[str]:
            return [sys.executable, "-c", "pass"]

    monkeypatch.setitem(baselines.BASELINES, "noop", Noop())

    results = run_suite(["Q"], repeat=1, baselines=["noop"], workdir=tmp_path)

    assert results["ok"], results["problems"]
    assert results["scenarios"]["Q"]["baselines"]["noop"]["summary"]["wall_s"]["n"] == 1


def test_unknown_names_are_rejected() -> None:
    with pytest.raises(SystemExit):
        expand(["nope"])
    with pytest.raises(ValueError, match="unknown baseline"):
        run_suite(["Q"], repeat=1, baselines=["nope"])


def test_groups_expand_without_duplicates() -> None:
    assert expand(["quick", "Q", "S"]) == names_in_group("quick") + ["S"]
    assert set(expand(["all"])) == set(SCENARIOS)


def test_summarise_gives_median_and_spread() -> None:
    summary = summarise([1.0, 2.0, 9.0])

    assert summary["median"] == 2.0
    assert summary["min"] == 1.0
    assert summary["max"] == 9.0
    assert summary["n"] == 3
    assert summary["stdev"] > 0
    assert summarise([3.0])["stdev"] == 0.0


def test_failure_injection_is_deterministic_and_about_the_requested_rate() -> None:
    failing = [is_failing(x, 5, 50) for x in range(10_000)]

    assert failing == [is_failing(x, 5, 50) for x in range(10_000)]
    assert 100 < sum(failing) < 300
    assert not is_failing(1, 1, 0)
