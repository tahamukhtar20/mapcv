"""MV: pixel-level validation of rasterizer against rasterio golden fixtures."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, cast

import numpy as np
import numpy.typing as npt
import pytest
from shapely.wkt import loads as wkt_loads

from mapcv.rasterizer import rasterize

_NPZ = Path(__file__).parent / "golden" / "rasterize_golden.npz"
_META_FILE = Path(__file__).parent / "golden" / "rasterize_golden_meta.json"

# The fixtures are committed: a missing one is an error, never a skipped test.
if not _NPZ.exists() or not _META_FILE.exists():
    raise FileNotFoundError(
        "rasterize_golden.npz / rasterize_golden_meta.json not found - regenerate with: uv run --group golden python tests/generate_golden.py"
    )


def _load() -> tuple[Any, list[dict[str, Any]]]:
    npz = np.load(str(_NPZ))
    meta = json.loads(_META_FILE.read_text())
    return npz, meta


_NPZ_DATA, _META = _load()


def _run_case(entry: dict[str, Any]) -> tuple[npt.NDArray[np.uint8], npt.NDArray[np.uint8]]:
    geom = wkt_loads(entry["wkt"])
    class_id: int = entry["class_id"]
    transform = cast(tuple[float, float, float, float, float, float], tuple(entry["transform"]))
    h, w = entry["out_shape"]
    all_touched = bool(entry.get("all_touched", False))
    our = rasterize([(geom, class_id)], (h, w), transform, all_touched=all_touched)
    golden = _NPZ_DATA[f"case_{entry['case']}"]
    return our, golden


@pytest.mark.parametrize("entry", _META)
def test_rasterize_matches_rasterio(entry: dict[str, Any]) -> None:
    """Our rasterize() must match rasterio pixel-for-pixel, in both all_touched modes."""
    our, golden = _run_case(entry)
    mismatch = int(np.sum(our != golden))
    assert mismatch == 0, (
        f"case {entry['case']} (all_touched={entry.get('all_touched', False)}): "
        f"{mismatch}/{golden.size} pixel mismatches "
        f"(our nonzero={int(np.count_nonzero(our))}, ref nonzero={entry['nonzero']})"
    )
