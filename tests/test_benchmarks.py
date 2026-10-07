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
from typing import Any, Dict, Iterator, List, Optional, Tuple

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


def test_a_work_directory_without_room_fails_early_with_a_message(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    class Full:
        free = 1024

    monkeypatch.setattr(shutil, "disk_usage", lambda path: Full())

    results = run_suite(["M"], repeat=1, workdir=tmp_path)

    assert results["ok"] is False
    assert "--workdir" in results["problems"][0]
    assert results["scenarios"]["M"]["status"] == "failed"


def test_baseline_hook_measures_a_registered_baseline(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    class Noop:
        name = "noop"
        reference = False

        def command(self, work: baselines.Workload) -> List[str]:
            return [sys.executable, "-c", "pass"]

        def missing(self) -> Optional[str]:
            return None

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


def test_the_rasterio_script_makes_mapcvs_data(tmp_path: Path) -> None:
    """The registered baseline writes the same patches as mapcv on a quick scenario, and a
    broken copy of its output is reported as different data."""
    results = run_suite(["Q"], repeat=1, baselines=["rasterio-script"], workdir=tmp_path)
    assert results["ok"], results["problems"]
    comparison = results["scenarios"]["Q"]["baselines"]["rasterio-script"]["comparison"]
    assert comparison["same_data"] and comparison["images_differing"] == 0
    assert comparison["patches_mapcv"] == comparison["patches_baseline"] > 0

    run_dir = tmp_path / "Q"
    images = sorted((run_dir / "baseline-rasterio-script" / "images").glob("*.png"))
    pixels = np.asarray(Image.open(images[0])).copy()
    pixels[0, 0] ^= 1
    Image.fromarray(pixels).save(images[0])
    images[1].unlink()
    broken = checks.compare_with_mapcv(run_dir / "dataset", run_dir / "baseline-rasterio-script")
    assert broken["images_differing"] == 1 and broken["only_mapcv"] == 1
    assert not broken["same_data"]


def test_compare_table_marks_baselines_with_other_data() -> None:
    from benchmarks.cli import compare_table

    timing = {"wall_s": {"median": 2.0}, "peak_rss_mb": {"median": 100}}
    document = {
        "scenarios": {
            "M": {
                "summary": timing,
                "baselines": {
                    "same": {
                        "summary": {"wall_s": {"median": 3.0}, "peak_rss_mb": {"median": 50}},
                        "comparison": {"same_data": True},
                    },
                    "other": {
                        "summary": {"wall_s": {"median": 1.0}, "peak_rss_mb": {"median": 50}},
                        "comparison": {"same_data": False},
                    },
                },
            }
        }
    }
    text = compare_table(document)
    assert "1.50×" in text and "not comparable" in text
    markdown = compare_table(document, markdown=True)
    assert (
        markdown.splitlines()[0].startswith("| scenario | tool |")
        and "| M | other | 1.00 s | 50 MB | not comparable | no |" in markdown
    )


def test_every_baseline_is_registered() -> None:
    assert {"rasterio-script", "gdal-cli", "torchgeo", "leafmap"} <= set(baselines.BASELINES)
    assert baselines.BASELINES["rasterio-script"].reference
    assert baselines.BASELINES["gdal-cli"].reference
    assert not baselines.BASELINES["torchgeo"].reference


def test_missing_tools_are_named(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    class NeedsNothing(baselines._PythonScript):
        modules = ["json"]

    class NeedsMissing(baselines._PythonScript):
        modules = ["json", "surely_not_installed_mapcv"]
        python_variable = "MAPCV_BENCH_TEST_PYTHON"

    assert NeedsNothing().missing() is None
    assert "surely_not_installed_mapcv" in str(NeedsMissing().missing())
    monkeypatch.setenv("MAPCV_BENCH_TEST_PYTHON", str(tmp_path / "no-python"))
    assert "not found" in str(NeedsMissing().missing())
    monkeypatch.setenv("GDAL_BIN", str(tmp_path))
    assert "GDAL_BIN" in str(baselines.BASELINES["gdal-cli"].missing())


def _fake_baseline(name: str, reference: bool, missing: Optional[str] = None) -> Any:
    """A baseline that writes one black patch where mapcv writes many."""
    script = (
        "import sys, pathlib; from PIL import Image; out = pathlib.Path(sys.argv[1]);"
        "[(out / d).mkdir() for d in ('images', 'masks')];"
        "Image.new('RGB', (256, 256)).save(out / 'images' / 'r0_c0.png');"
        "Image.new('L', (256, 256)).save(out / 'masks' / 'r0_c0.png')"
    )

    class Fake:
        def command(self, work: baselines.Workload) -> List[str]:
            return [sys.executable, "-c", script, str(work.output_dir)]

        def missing(self) -> Optional[str]:
            return missing

    fake = Fake()
    fake.name = name  # type: ignore[attr-defined]
    fake.reference = reference  # type: ignore[attr-defined]
    return fake


def test_other_data_fails_only_against_a_reference(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setitem(baselines.BASELINES, "tool", _fake_baseline("tool", reference=False))
    monkeypatch.setitem(baselines.BASELINES, "ref", _fake_baseline("ref", reference=True))
    monkeypatch.setitem(baselines.BASELINES, "absent", _fake_baseline("absent", False, "no x"))

    results = run_suite(["Q"], repeat=1, baselines=["tool", "absent"], workdir=tmp_path / "a")
    assert results["ok"], results["problems"]
    tool = results["scenarios"]["Q"]["baselines"]["tool"]
    assert tool["comparison"]["same_data"] is False and "not comparable" in tool["finding"]
    assert results["scenarios"]["Q"]["baselines"]["absent"] == {"skipped": "no x"}

    results = run_suite(["Q"], repeat=1, baselines=["ref"], workdir=tmp_path / "b")
    assert not results["ok"]
    assert any("baseline ref produced different data" in p for p in results["problems"])


def test_compare_table_shows_skipped_baselines() -> None:
    from benchmarks.cli import compare_table

    timing = {"wall_s": {"median": 2.0}, "peak_rss_mb": {"median": 100}}
    document = {
        "scenarios": {"M": {"summary": timing, "baselines": {"gone": {"skipped": "no GDAL"}}}}
    }
    assert "| M | gone | skipped | — | — | — |" in compare_table(document, markdown=True)
