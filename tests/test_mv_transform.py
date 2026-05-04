"""MV: validate coordinate transform against pyproj golden fixtures."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List

import pytest
from shapely.geometry import Point

from mapcv._mapcv_rs import xy as rust_xy
from mapcv.labels import transform_to_mercator

_GOLDEN = Path(__file__).parent / "golden" / "transform_golden.json"

if not _GOLDEN.exists():
    pytest.skip(
        "transform_golden.json not found - run tests/generate_golden.py first",
        allow_module_level=True,
    )


def _load() -> List[Dict[str, Any]]:
    result: List[Dict[str, Any]] = json.loads(_GOLDEN.read_text())
    return result


_DATA = _load()

# 1 mm in meters - tolerance for Web Mercator coordinate comparison
_TOL_M = 1e-3


def test_xy_matches_pyproj() -> None:
    """Rust xy() must agree with pyproj EPSG:4326->3857 within 1 mm."""
    max_dx = max_dy = 0.0
    failures: List[str] = []
    for i, entry in enumerate(_DATA):
        lng, lat = entry["lng"], entry["lat"]
        our_x, our_y = rust_xy(lng, lat)
        ref_x, ref_y = entry["mx"], entry["my"]
        dx = abs(our_x - ref_x)
        dy = abs(our_y - ref_y)
        max_dx = max(max_dx, dx)
        max_dy = max(max_dy, dy)
        if dx > _TOL_M or dy > _TOL_M:
            failures.append(f"[{i}] lng={lng:.4f} lat={lat:.4f}: dx={dx:.2e} dy={dy:.2e}")
    assert not failures, (
        f"{len(failures)} failures (max_dx={max_dx:.2e} max_dy={max_dy:.2e}):\n"
        + "\n".join(failures[:10])
    )


def test_transform_to_mercator_matches_pyproj() -> None:
    """Python transform_to_mercator() must agree with pyproj within 1 mm."""
    max_dx = max_dy = 0.0
    failures: List[str] = []
    for i, entry in enumerate(_DATA):
        lng, lat = entry["lng"], entry["lat"]
        pt = transform_to_mercator(Point(lng, lat))
        our_x, our_y = pt.x, pt.y
        ref_x, ref_y = entry["mx"], entry["my"]
        dx = abs(our_x - ref_x)
        dy = abs(our_y - ref_y)
        max_dx = max(max_dx, dx)
        max_dy = max(max_dy, dy)
        if dx > _TOL_M or dy > _TOL_M:
            failures.append(f"[{i}] lng={lng:.4f} lat={lat:.4f}: dx={dx:.2e} dy={dy:.2e}")
    assert not failures, (
        f"{len(failures)} failures (max_dx={max_dx:.2e} max_dy={max_dy:.2e}):\n"
        + "\n".join(failures[:10])
    )
