"""``plan`` counts grid patches without building them, and ``generate`` refuses grids it
could not hold, with a clear message, before anything is written.

The configs below have tens of millions to 10^18 grid patches. Before the count was done
per axis, ``plan`` built every anchor as a Python tuple and was killed by the system
after a few seconds and some gigabytes. Each run is in a fresh interpreter whose resident
memory is watched (Linux only), so a regression fails the test instead of the machine.
"""

from __future__ import annotations

import json
import sys
import textwrap
from pathlib import Path

import pytest
from test_crafted_inputs import _bounded

from mapcv._mapcv_rs import MAX_ANCHORS, grid_anchor_count, grid_sample_anchors
from mapcv.sampler import (
    MAX_PATCHES,
    SamplerConfig,
    TooManyPatchesError,
    check_patch_limit,
    grid_axes_for,
    grid_patch_count,
)

pytestmark = pytest.mark.skipif(
    not sys.platform.startswith("linux"), reason="the subprocess is watched through /proc"
)

#: Far below what building the anchors took (several GB), well above a normal plan: the
#: interpreter with mapcv, numpy, shapely and pydantic loaded is about 150 MiB.
_PLAN_MEMORY = 600 << 20

#: A 0.03 degree square at zoom 18: 5,888 x 8,960 px, 52,756,480 patches at stride 1.
_FINE = """
region: {west: 10.0, south: 50.0, east: 10.03, north: 50.03}
imagery: {type: xyz, zoom: 18, url_template: "http://127.0.0.1:9/{z}/{x}/{y}.png"}
sampler: {patch_size: 64, stride: 1, mode: grid}
writer: {staging_dir: out}
"""

#: The whole world at zoom 22: about 10^18 patches.
_WORLD = """
region: {west: -179.9, south: -85, east: 179.9, north: 85}
imagery: {type: xyz, zoom: 22, url_template: "http://127.0.0.1:9/{z}/{x}/{y}.png"}
sampler: {patch_size: 64, stride: 1, mode: grid}
writer: {staging_dir: out}
"""

#: The same with a built-in source: the MCP server connects to public addresses only.
_PUBLIC_WORLD = _WORLD.replace(
    'url_template: "http://127.0.0.1:9/{z}/{x}/{y}.png"', "source: esri_satellite"
)


def _run(code: str) -> str:
    result = _bounded(code, _PLAN_MEMORY)
    assert result.returncode == 0, f"exit {result.returncode}: {result.stderr[-1500:]}"
    return result.stdout


@pytest.mark.parametrize("config", [_FINE, _WORLD], ids=["fine-grid", "world"])
def test_plan_counts_a_huge_grid_in_bounded_memory(tmp_path: Path, config: str) -> None:
    (tmp_path / "mapcv.yaml").write_text(config)
    out = _run(
        f"""
        from mapcv.config import MapcvConfig
        from mapcv.planning import plan

        estimate = plan(MapcvConfig.from_yaml({str(tmp_path / "mapcv.yaml")!r}))
        width, height = estimate.raster_px
        print(estimate.patches, width * height, len(estimate.blocking))
        """
    )
    patches, pixels, blocking = (int(value) for value in out.split())
    # Stride 1 with edge_strategy "pad": one patch for every pixel.
    assert patches == pixels
    assert patches > MAX_PATCHES
    assert blocking == 1  # generate refuses it, and the plan says so


def test_the_plan_command_shows_the_count_and_why_generate_would_refuse(tmp_path: Path) -> None:
    (tmp_path / "mapcv.yaml").write_text(_FINE)
    out = _run(
        f"""
        import os
        os.environ["COLUMNS"] = "200"
        os.chdir({str(tmp_path)!r})
        from typer.testing import CliRunner
        from mapcv.cli import app

        result = CliRunner().invoke(app, ["plan", "mapcv.yaml"])
        print(result.exit_code)
        print(" ".join(result.output.split()))
        """
    )
    code, _, output = out.partition("\n")
    assert code == "1"
    assert "52,756,480" in output
    assert "8,960 x 5,888 = 52,756,480 patches, more than the limit of 50,000,000" in output
    assert "Raise sampler.stride" in output
    assert "shrink the region or split it" in output


def test_the_mcp_plan_tool_counts_a_huge_grid_in_bounded_memory(tmp_path: Path) -> None:
    out = _run(
        f"""
        import json
        from pathlib import Path
        from mapcv.agent_tools import Sandbox, ToolState, plan

        result = plan(ToolState(Sandbox(Path({str(tmp_path)!r}))), yaml_text={_PUBLIC_WORLD!r})
        data = result.data
        print(json.dumps({{
            "patches": data["patches"],
            "pixels": data["raster_px"]["width"] * data["raster_px"]["height"],
            "blocking": data["blocking"],
            "summary": result.summary,
        }}))
        """
    )
    data = json.loads(out)
    assert data["patches"] == data["pixels"] > MAX_PATCHES
    assert "more than the limit of 50,000,000" in data["blocking"][0]
    assert "generate will refuse this config" in data["summary"]


def test_generate_refuses_a_grid_over_the_limit_before_writing(tmp_path: Path) -> None:
    staging = tmp_path / "dataset"
    out = _run(
        f"""
        from mapcv.config import MapcvConfig
        from mapcv.pipeline import run_generate
        from mapcv.sampler import TooManyPatchesError

        config = MapcvConfig.model_validate({{
            "region": {{"west": 10.0, "south": 50.0, "east": 10.03, "north": 50.03}},
            "imagery": {{"type": "xyz", "zoom": 18,
                         "url_template": "http://127.0.0.1:9/{{z}}/{{x}}/{{y}}.png"}},
            "sampler": {{"patch_size": 64, "stride": 1}},
            "writer": {{"staging_dir": {str(staging)!r}}},
        }})
        try:
            run_generate(config)
        except TooManyPatchesError as exc:
            print("refused:", exc)
        else:
            print("accepted")
        """
    )
    assert "refused:" in out, out
    assert "the grid has 8,960 x 5,888 = 52,756,480 patches" in out
    assert "more than the limit of 50,000,000 for one run" in out
    assert "Raise sampler.stride" in out
    assert not (staging / "manifest.json").exists()


def test_the_generate_command_stops_on_a_grid_over_the_limit(tmp_path: Path) -> None:
    (tmp_path / "mapcv.yaml").write_text(_FINE)
    out = _run(
        f"""
        import os
        os.environ["COLUMNS"] = "200"
        os.chdir({str(tmp_path)!r})
        from typer.testing import CliRunner
        from mapcv.cli import app

        result = CliRunner().invoke(app, ["generate", "mapcv.yaml", "--yes"])
        print(result.exit_code)
        print(" ".join(result.output.split()))
        """
    )
    code, _, output = out.partition("\n")
    assert code == "1"
    assert "This config cannot be built" in output
    assert "more than the limit of 50,000,000" in output
    assert "Generation failed" not in output
    assert not (tmp_path / "out" / "manifest.json").exists()


def test_a_random_count_over_the_limit_is_refused_too(tmp_path: Path) -> None:
    config = _FINE.replace("stride: 1, mode: grid", "mode: random, random_count: 51000000")
    config = config.replace("staging_dir: out", f"staging_dir: {tmp_path / 'out'}")
    (tmp_path / "mapcv.yaml").write_text(config)
    out = _run(
        f"""
        from mapcv.config import MapcvConfig
        from mapcv.pipeline import run_generate
        from mapcv.planning import plan
        from mapcv.sampler import TooManyPatchesError

        config = MapcvConfig.from_yaml({str(tmp_path / "mapcv.yaml")!r})
        estimate = plan(config)
        print(estimate.patches, len(estimate.blocking))
        try:
            run_generate(config)
        except TooManyPatchesError as exc:
            print("refused:", exc)
        """
    )
    assert out.startswith("51000000 1\n")
    assert "sampler.random_count asks for 51,000,000 patches" in out
    assert "more than the limit of 50,000,000" in out


def test_a_grid_just_within_the_limit_is_not_refused() -> None:
    config = SamplerConfig(patch_size=1, stride=1)
    count = grid_patch_count(5_000, 10_000, config)
    assert count == MAX_PATCHES == int(MAX_ANCHORS) == 50_000_000
    check_patch_limit(5_000, 10_000, config)
    with pytest.raises(TooManyPatchesError, match="more than the limit of 50,000,000"):
        check_patch_limit(5_000, 10_001, config)
    rows, cols = grid_axes_for(5_000, 10_000, config)
    assert len(rows) * len(cols) == count
    with pytest.raises(ValueError, match="more than the limit of 50000000"):
        grid_sample_anchors(5_000, 10_001, 1, 1, "pad")
    assert issubclass(TooManyPatchesError, ValueError)


#: Above what a run near the limit holds for its anchors (a grid as its rows and columns,
#: random anchors 8 bytes each), far below the 7 GB that 50,000,000 anchors took as
#: Python tuples of about 150 bytes.
_ANCHOR_MEMORY = 900 << 20

_ANCHOR_GROUPS = """
import resource
from types import SimpleNamespace
from mapcv import pipeline
from mapcv.config import MapcvConfig

config = MapcvConfig.model_validate({{
    "region": {{"west": 10.0, "south": 50.0, "east": 10.03, "north": 50.03}},
    "imagery": {{"type": "xyz", "zoom": 18,
                 "url_template": "http://127.0.0.1:9/{{z}}/{{x}}/{{y}}.png"}},
    "sampler": {sampler},
    "writer": {{"staging_dir": "out"}},
}})
meta = SimpleNamespace(height={height}, width={width}, chunk_rows=1024, bands=[1, 2, 3],
                       dtype="uint8")
chunks = pipeline._anchor_groups(config, [SimpleNamespace(metadata=meta)], None)
print(sum(len(chunk) for chunk in chunks), len(chunks))
"""


@pytest.mark.parametrize(
    ("sampler", "height", "width", "expected"),
    [
        ({"patch_size": 64, "stride": 1}, 7_000, 7_100, 49_700_000),
        (
            {"patch_size": 64, "mode": "random", "random_count": 20_000_000},
            6_000,
            6_000,
            20_000_000,
        ),
    ],
    ids=["grid", "random"],
)
def test_a_run_near_the_limit_holds_its_anchors_in_bounded_memory(
    sampler: dict[str, object], height: int, width: int, expected: int
) -> None:
    # Before anchors were held compactly these were refused (over 3,000,000), and a
    # grid of this size took several GB as Python tuples.
    out = _bounded(
        _ANCHOR_GROUPS.format(sampler=sampler, height=height, width=width), _ANCHOR_MEMORY
    )
    assert out.returncode == 0, f"exit {out.returncode}: {out.stderr[-1500:]}"
    total, chunks = (int(value) for value in out.stdout.split())
    assert total == expected
    assert chunks == -(-height // 1024)


# ── The per-axis count equals the enumeration ───────────────────────────────────────


def _loop_axis(dim: int, patch: int, stride: int, strategy: str) -> int:
    """Anchors along one axis by stepping, as the first grid sampler did."""
    positions = []
    p = 0
    if strategy == "pad":
        while p < dim:
            positions.append(p)
            p += stride
        return len(positions)
    while p + patch <= dim:
        positions.append(p)
        p += stride
    if strategy == "shift":
        if dim >= patch:
            if not positions or positions[-1] < dim - patch:
                positions.append(dim - patch)
        elif not positions:
            positions.append(0)
    return len(positions)


@pytest.mark.parametrize("strategy", ["pad", "drop", "shift"])
def test_the_arithmetic_count_equals_the_enumeration(strategy: str) -> None:
    sizes = [(1, 1), (7, 40), (37, 11), (64, 64), (100, 73), (130, 31)]
    for height, width in sizes:
        for patch in (1, 3, 8, 32, 64, 120):
            for stride in (1, 2, 7, 16, 33, 64, 500):
                rows, cols, total = grid_anchor_count(height, width, patch, stride, strategy)
                anchors = grid_sample_anchors(height, width, patch, stride, strategy)
                where = f"{strategy} {height}x{width} patch={patch} stride={stride}"
                assert total == len(anchors), where
                assert rows == _loop_axis(height, patch, stride, strategy), where
                assert cols == _loop_axis(width, patch, stride, strategy), where
                assert len(set(anchors)) == len(anchors), where
                config = SamplerConfig(patch_size=patch, stride=stride, edge_strategy=strategy)  # type: ignore[arg-type]
                assert grid_patch_count(height, width, config) == total, where


def test_the_count_of_a_grid_too_big_to_build_is_exact() -> None:
    rows, cols, total = grid_anchor_count(2**40, 2**41, 64, 3, "drop")
    assert (rows, cols) == ((2**40 - 64) // 3 + 1, (2**41 - 64) // 3 + 1)
    assert total == rows * cols > 2**64
    with pytest.raises(ValueError, match="stride must be > 0"):
        grid_anchor_count(10, 10, 4, 0)


def test_the_message_for_a_grid_over_the_limit_says_how_to_fix_it() -> None:
    config = SamplerConfig(patch_size=64, stride=1)
    with pytest.raises(TooManyPatchesError) as raised:
        check_patch_limit(8_000, 8_000, config)
    text = textwrap.dedent(str(raised.value))
    assert "8,000 x 8,000 = 64,000,000 patches" in text
    assert "limit of 50,000,000" in text
    assert "sampler.stride" in text
    check_patch_limit(100, 100, config)  # within the limit: no error
