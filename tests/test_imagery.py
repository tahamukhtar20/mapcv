"""Tests for the source-neutral imagery adapters."""

from __future__ import annotations

from io import BytesIO
from types import SimpleNamespace
from typing import Any, List, Tuple

import numpy as np
import pytest
from PIL import Image

from mapcv._mapcv_rs import TileIndex
from mapcv.config import RegionConfig, XYZImageryConfig
from mapcv.imagery import (
    XYZRasterSource,
    _snap_bounds_to_grid,
    offset_transform,
)


def _region() -> RegionConfig:
    return RegionConfig(west=10.0, south=45.0, east=10.15, north=45.15)


def test_snap_bounds_to_grid_aligns_to_pixel_edges() -> None:
    x_values = np.array([500005.0, 500015.0, 500025.0])
    y_values = np.array([4999995.0, 4999985.0, 4999975.0])
    snapped = _snap_bounds_to_grid(
        (500003.2, 4999971.0, 500021.7, 4999999.9), x_values, y_values, 10
    )
    assert snapped == pytest.approx((500000.0, 4999970.0, 500030.0, 5000000.0))
    # Bounds already on pixel edges are left unchanged.
    assert _snap_bounds_to_grid(snapped, x_values, y_values, 10) == pytest.approx(snapped)


def _png_tile(value: int) -> bytes:
    buffer = BytesIO()
    Image.fromarray(np.full((256, 256, 3), value, dtype=np.uint8)).save(buffer, format="PNG")
    return buffer.getvalue()


def test_xyz_source_exposes_window_contract(monkeypatch: pytest.MonkeyPatch) -> None:
    transform = (2.0, 0.0, 100.0, 0.0, -2.0, 200.0)
    tile = TileIndex(3, 4, 12)
    monkeypatch.setattr(
        "mapcv.imagery.snap_bbox",
        lambda *args, **kwargs: SimpleNamespace(west=0, south=0, east=1, north=1),
    )
    monkeypatch.setattr("mapcv.imagery.tiles", lambda *args, **kwargs: [tile])
    monkeypatch.setattr(
        "mapcv.imagery.fetch_tiles",
        lambda requested, *args, **kwargs: (
            [(t, _png_tile(7), None) for t in requested],
            0,
            ([], None),
        ),
    )
    monkeypatch.setattr("mapcv.imagery.tile_transform", lambda *args: transform)
    source = XYZRasterSource(
        _region(), XYZImageryConfig(zoom=12, source="esri_satellite", strip_rows=2)
    )

    window, valid = source.read_window(20, 60, 30, 80)

    assert window.shape == (40, 50, 3)
    assert np.all(window == 7)
    assert valid.all()
    assert source.metadata.crs == "EPSG:3857"
    assert source.metadata.chunk_rows == 512


def test_xyz_source_marks_black_pixels_invalid(monkeypatch: pytest.MonkeyPatch) -> None:
    tile = TileIndex(3, 4, 12)
    monkeypatch.setattr(
        "mapcv.imagery.snap_bbox",
        lambda *args, **kwargs: SimpleNamespace(west=0, south=0, east=1, north=1),
    )
    monkeypatch.setattr("mapcv.imagery.tiles", lambda *args, **kwargs: [tile])
    # The fetcher black-fills failed tiles under the lenient/ignore policies.
    monkeypatch.setattr(
        "mapcv.imagery.fetch_tiles",
        lambda requested, *args, **kwargs: (
            [(t, _png_tile(0), None) for t in requested],
            0,
            ([], None),
        ),
    )
    source = XYZRasterSource(_region(), XYZImageryConfig(zoom=12, source="esri_satellite"))

    _, valid = source.read_window(0, 256, 0, 256)

    assert not valid.any()


def test_xyz_custom_template_product_id_keeps_only_hostname(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    tile = TileIndex(3, 4, 12)
    monkeypatch.setattr(
        "mapcv.imagery.snap_bbox",
        lambda *args, **kwargs: SimpleNamespace(west=0, south=0, east=1, north=1),
    )
    monkeypatch.setattr("mapcv.imagery.tiles", lambda *args, **kwargs: [tile])
    monkeypatch.setattr(
        "mapcv.imagery.fetch_tiles",
        lambda requested, *args, **kwargs: (
            [(t, _png_tile(7), None) for t in requested],
            0,
            ([], None),
        ),
    )
    template = "https://tiles.example.com/wmts/SECRET-INSTANCE/{z}/{x}/{y}.png?key=SECRET"
    source = XYZRasterSource(_region(), XYZImageryConfig(zoom=12, url_template=template))

    assert source.metadata.product_id == "custom-xyz:tiles.example.com"


def test_offset_transform_uses_global_pixel_origin() -> None:
    transform = (10.0, 0.0, 300.0, 0.0, -10.0, 500.0)
    assert offset_transform(transform, row=4, col=3) == (
        10.0,
        0.0,
        330.0,
        0.0,
        -10.0,
        460.0,
    )


def test_xyz_source_fetches_lazily_per_window_and_evicts(monkeypatch: pytest.MonkeyPatch) -> None:
    grid = [TileIndex(x, y, 12) for y in range(10, 14) for x in range(5, 7)]
    calls: List[List[Tuple[int, int]]] = []

    def fake_fetch(requested: List[Any], *args: Any, **kwargs: Any) -> Tuple[List[Any], int, Any]:
        calls.append([(t.x, t.y) for t in requested])
        return (
            [(t, _png_tile(9), None) for t in requested if (t.x, t.y) != (6, 11)],
            1,
            ([("HTTP 503", 1)], "HTTP 503 for URL: x"),
        )

    monkeypatch.setattr(
        "mapcv.imagery.snap_bbox",
        lambda *args, **kwargs: SimpleNamespace(west=0, south=0, east=1, north=1),
    )
    monkeypatch.setattr("mapcv.imagery.tiles", lambda *args, **kwargs: grid)
    monkeypatch.setattr("mapcv.imagery.fetch_tiles", fake_fetch)
    source = XYZRasterSource(_region(), XYZImageryConfig(zoom=12, source="esri_satellite"))
    assert calls == []  # nothing is downloaded until a window is read

    source.read_window(0, 512, 0, 512)  # tile rows 10-11
    source.read_window(256, 768, 0, 512)  # rows 11-12: row 11 is reused, row 10 evicted
    assert calls == [[(5, 10), (6, 10), (5, 11), (6, 11)], [(5, 12), (6, 12)]]
    assert {key[1] for key in source._tiles} == {11, 12}
    assert source.tiles_requested == 6
    assert source.tiles_failed == 2
    assert source.failure_reasons == "2 x HTTP 503 (e.g. HTTP 503 for URL: x)"
