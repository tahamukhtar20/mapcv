"""The Rust extension's Python surface: its type stubs match the compiled module, and
its small classes behave like values (equality, hashing, repr, pickling)."""

from __future__ import annotations

import copy
import importlib.util
import pickle
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from mapcv import _mapcv_rs
from mapcv._mapcv_rs import BBox, GeoTiff, TileIndex


@pytest.mark.skipif(importlib.util.find_spec("mypy") is None, reason="needs mypy (dev extra)")
def test_the_stubs_match_the_compiled_module() -> None:
    result = subprocess.run(
        [sys.executable, "-m", "mypy.stubtest", "mapcv._mapcv_rs"],
        capture_output=True,
        check=False,
        text=True,
        timeout=300,
    )
    assert result.returncode == 0, result.stdout + result.stderr


def test_tile_index_is_a_value() -> None:
    tile = TileIndex(3, 5, 12)
    assert (tile.x, tile.y, tile.z) == (3, 5, 12)
    assert tile == TileIndex(3, 5, 12) and tile != TileIndex(3, 5, 13)
    assert tile != (3, 5, 12)
    assert len({tile, TileIndex(3, 5, 12), TileIndex(5, 3, 12)}) == 2
    assert repr(tile) == "TileIndex(x=3, y=5, z=12)"
    assert type(tile).__module__ == "mapcv._mapcv_rs" and type(tile).__name__ == "TileIndex"
    for clone in (pickle.loads(pickle.dumps(tile)), copy.copy(tile), copy.deepcopy(tile)):
        assert clone == tile and type(clone) is TileIndex
    with pytest.raises(AttributeError):
        tile.x = 4  # type: ignore[misc]


def test_bbox_is_a_value() -> None:
    box = BBox(-1.5, 2.0, 3.25, 4.0)
    assert (box.west, box.south, box.east, box.north) == (-1.5, 2.0, 3.25, 4.0)
    assert box == BBox(-1.5, 2.0, 3.25, 4.0) and box != BBox(-1.5, 2.0, 3.25, 4.5)
    # -0.0 equals 0.0, so the two must hash alike.
    assert BBox(0.0, -0.0, 1.0, 1.0) == BBox(-0.0, 0.0, 1.0, 1.0)
    assert hash(BBox(0.0, -0.0, 1.0, 1.0)) == hash(BBox(-0.0, 0.0, 1.0, 1.0))
    assert repr(box) == "BBox(west=-1.5, south=2.0, east=3.25, north=4.0)"
    assert pickle.loads(pickle.dumps(box)) == box
    with pytest.raises(AttributeError):
        box.west = 0.0  # type: ignore[misc]


def test_functions_return_the_value_classes() -> None:
    assert _mapcv_rs.tile(0.0, 0.0, 1) == TileIndex(1, 1, 1)
    assert set(_mapcv_rs.tiles(-1.0, -1.0, 1.0, 1.0, [1])) == {
        TileIndex(x, y, 1) for x in (0, 1) for y in (0, 1)
    }
    world = _mapcv_rs.bounds(0, 0, 0)
    assert world == BBox(-180.0, world.south, 180.0, world.north) and world.north > 85
    # The names before 0.3 still work.
    assert _mapcv_rs.PyTileIndex is TileIndex and _mapcv_rs.PyBBox is BBox


def test_geotiff_repr_names_the_file(tmp_path: Path) -> None:
    folder = tmp_path / "it's here"
    folder.mkdir()
    _mapcv_rs.write_geotiffs(
        np.zeros(4, dtype=np.uint8),
        "uint8",
        (1, 2, 2, 1),
        [[10.0, 1.0, 0.0, 50.0, 0.0, -1.0]],  # GDAL order: x0, dx, 0, y0, 0, -dy
        ["a.tif"],
        str(folder),
        4326,
        True,
    )
    path = str(folder / "a.tif")
    assert repr(GeoTiff(path)) == f"GeoTiff({path!r})"
