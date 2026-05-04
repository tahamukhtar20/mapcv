"""MV: validate tile math functions against mercantile golden fixtures."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List

import pytest

from mapcv._mapcv_rs import bounds, tile, tiles, xy, xy_bounds

_GOLDEN = Path(__file__).parent / "golden" / "tile_math_golden.json"

if not _GOLDEN.exists():
    pytest.skip(
        "tile_math_golden.json not found - run tests/generate_golden.py first",
        allow_module_level=True,
    )


def _load() -> Dict[str, Any]:
    result: Dict[str, Any] = json.loads(_GOLDEN.read_text())
    return result


_DATA = _load()


def test_xy_matches_mercantile() -> None:
    """Our xy() must agree with mercantile.xy within 1e-5 m."""
    failures: List[str] = []
    for i, entry in enumerate(_DATA["xy"]):
        lng, lat = entry["lng"], entry["lat"]
        our_x, our_y = xy(lng, lat)
        ref_x, ref_y = entry["mx"], entry["my"]
        dx = abs(our_x - ref_x)
        dy = abs(our_y - ref_y)
        if dx > 1e-5 or dy > 1e-5:
            failures.append(f"[{i}] lng={lng} lat={lat}: dx={dx:.2e} dy={dy:.2e}")
    assert not failures, f"{len(failures)} failures:\n" + "\n".join(failures[:10])


def test_tile_matches_mercantile() -> None:
    """Our tile() must return the exact same (x, y, z) as mercantile.tile."""
    failures: List[str] = []
    for i, entry in enumerate(_DATA["tile"]):
        lng, lat, zoom = entry["lng"], entry["lat"], entry["zoom"]
        our = tile(lng, lat, zoom)
        if our.x != entry["x"] or our.y != entry["y"] or our.z != entry["z"]:
            failures.append(
                f"[{i}] lng={lng} lat={lat} z={zoom}: "
                f"ours=({our.x},{our.y},{our.z}) ref=({entry['x']},{entry['y']},{entry['z']})"
            )
    assert not failures, f"{len(failures)} failures:\n" + "\n".join(failures[:10])


def test_xy_bounds_matches_mercantile() -> None:
    """Our xy_bounds() must agree with mercantile.xy_bounds within 1e-5 m."""
    failures: List[str] = []
    for i, entry in enumerate(_DATA["xy_bounds"]):
        our = xy_bounds(entry["x"], entry["y"], entry["z"])
        for key in ("west", "south", "east", "north"):
            diff = abs(getattr(our, key) - entry[key])
            if diff > 1e-5:
                failures.append(
                    f"[{i}] tile=({entry['x']},{entry['y']},{entry['z']}) {key}: diff={diff:.2e}"
                )
    assert not failures, f"{len(failures)} failures:\n" + "\n".join(failures[:10])


def test_bounds_matches_mercantile() -> None:
    """Our bounds() must agree with mercantile.bounds within 1e-6 degrees."""
    failures: List[str] = []
    for i, entry in enumerate(_DATA["bounds"]):
        our = bounds(entry["x"], entry["y"], entry["z"])
        for key in ("west", "south", "east", "north"):
            diff = abs(getattr(our, key) - entry[key])
            if diff > 1e-6:
                failures.append(
                    f"[{i}] tile=({entry['x']},{entry['y']},{entry['z']}) {key}: diff={diff:.2e}"
                )
    assert not failures, f"{len(failures)} failures:\n" + "\n".join(failures[:10])


def test_tiles_matches_mercantile() -> None:
    """Our tiles() must return the exact same tile set as mercantile.tiles."""
    failures: List[str] = []
    for i, entry in enumerate(_DATA["tiles"]):
        w, s, e, n, zoom = (
            entry["west"],
            entry["south"],
            entry["east"],
            entry["north"],
            entry["zoom"],
        )
        our_list = tiles(w, s, e, n, [zoom])
        our_set = {(t.x, t.y, t.z) for t in our_list}
        ref_set = {(row[0], row[1], row[2]) for row in entry["tiles"]}
        if our_set != ref_set:
            only_ours = our_set - ref_set
            only_ref = ref_set - our_set
            failures.append(
                f"[{i}] bbox=({w:.3f},{s:.3f},{e:.3f},{n:.3f}) z={zoom}: "
                f"+{len(only_ours)} extra, -{len(only_ref)} missing"
            )
    assert not failures, f"{len(failures)} failures:\n" + "\n".join(failures[:10])
