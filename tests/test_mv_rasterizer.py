"""MV: pixel-level validation of rasterizer against rasterio golden fixtures."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Tuple, cast

import numpy as np
import pytest
from shapely.wkt import loads as wkt_loads

from mapcv.rasterizer import rasterize

_NPZ = Path(__file__).parent / "golden" / "rasterize_golden.npz"
_META_FILE = Path(__file__).parent / "golden" / "rasterize_golden_meta.json"

if not _NPZ.exists() or not _META_FILE.exists():
    pytest.skip(
        "rasterize_golden.npz / rasterize_golden_meta.json not found - run tests/generate_golden.py first",
        allow_module_level=True,
    )


def _load() -> Tuple[Any, List[Dict[str, Any]]]:
    npz = np.load(str(_NPZ))
    meta = json.loads(_META_FILE.read_text())
    return npz, meta


_NPZ_DATA, _META = _load()

# case 5 is the thin diagonal - allow up to 1% edge-pixel disagreement
_THIN_DIAGONAL_CASE = 5
_THIN_DIAGONAL_TOL = 0.01


def _run_case(entry: Dict[str, Any]) -> Tuple[np.ndarray, np.ndarray]:
    geom = wkt_loads(entry["wkt"])
    class_id: int = entry["class_id"]
    transform = cast(Tuple[float, float, float, float, float, float], tuple(entry["transform"]))
    h, w = entry["out_shape"]
    our = rasterize([(geom, class_id)], (h, w), transform)
    golden = _NPZ_DATA[f"case_{entry['case']}"]
    return our, golden


@pytest.mark.parametrize("entry", _META)
def test_rasterize_matches_rasterio(entry: Dict[str, Any]) -> None:
    """Our rasterize() must match rasterio pixel-for-pixel (case 5: <=1% tolerance)."""
    our, golden = _run_case(entry)
    mismatch = int(np.sum(our != golden))
    total = golden.size
    case = entry["case"]

    if case == _THIN_DIAGONAL_CASE:
        rate = mismatch / total
        assert rate <= _THIN_DIAGONAL_TOL, (
            f"case {case} (thin diagonal): mismatch rate {rate:.3%} > {_THIN_DIAGONAL_TOL:.1%} "
            f"({mismatch}/{total} pixels)"
        )
    else:
        assert mismatch == 0, (
            f"case {case}: {mismatch}/{total} pixel mismatches "
            f"(our nonzero={int(np.count_nonzero(our))}, ref nonzero={entry['nonzero']})"
        )
