"""Tests for the source-neutral imagery adapters."""

from __future__ import annotations

from io import BytesIO
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest
import xarray as xr
from PIL import Image
from shapely.geometry import Point

from mapcv._mapcv_rs import PyTileIndex
from mapcv.config import EOPFZarrImageryConfig, RegionConfig, XYZImageryConfig
from mapcv.imagery import (
    EOPFZarrRasterSource,
    XYZRasterSource,
    offset_transform,
    transform_geometry_to_crs,
)


def _dataset() -> xr.Dataset:
    b08 = np.arange(16, dtype=np.float64).reshape(4, 4)
    b04 = b08 + 100
    return xr.Dataset(
        data_vars={"b08": (("y", "x"), b08), "b04": (("y", "x"), b04)},
        coords={
            "x": np.array([10.00, 10.05, 10.10, 10.15]),
            "y": np.array([45.15, 45.10, 45.05, 45.00]),
        },
        attrs={"crs": "EPSG:4326"},
    ).chunk({"x": 2, "y": 2})


def _region() -> RegionConfig:
    return RegionConfig(west=10.0, south=45.0, east=10.15, north=45.15)


def test_eopf_source_preserves_band_order_and_casts_float32(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    product = tmp_path / "S2_TEST.zarr"
    product.mkdir()
    captured: list[dict[str, Any]] = []

    def fake_open_dataset(path: str, **kwargs: Any) -> xr.Dataset:
        captured.append(kwargs)
        return _dataset()

    monkeypatch.setattr(xr, "open_dataset", fake_open_dataset)
    source = EOPFZarrRasterSource(
        _region(),
        EOPFZarrImageryConfig(path=str(product), bands=["b08", "b04"], resolution=10),
    )

    image, valid = source.read_window(0, source.metadata.height, 0, source.metadata.width)

    assert len(captured) == 2
    assert captured[0]["engine"] == "eopf-zarr"
    assert captured[0]["variables"] == ["b08", "b04"]
    assert "bbox" not in captured[0]
    assert captured[1]["bbox"] == pytest.approx([10.0, 45.0, 10.15, 45.15])
    assert captured[1]["crs"] == "EPSG:4326"
    assert source.metadata.product_id == "S2_TEST.zarr"
    assert source.metadata.bands == ["b08", "b04"]
    assert image.dtype == np.float32
    assert image.shape[-1] == 2
    np.testing.assert_array_equal(image[:, :, 1], image[:, :, 0] + 100)
    assert valid.all()
    source.close()


def test_eopf_source_reports_missing_variables(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    product = tmp_path / "S2_TEST.zarr"
    product.mkdir()
    monkeypatch.setattr(xr, "open_dataset", lambda *args, **kwargs: _dataset())

    with pytest.raises(ValueError, match="b11.*Available variables"):
        EOPFZarrRasterSource(_region(), EOPFZarrImageryConfig(path=str(product), bands=["b11"]))


def test_eopf_source_rejects_secret_bearing_url() -> None:
    with pytest.raises(ValueError, match="must not contain"):
        EOPFZarrRasterSource(
            _region(),
            EOPFZarrImageryConfig(path="https://example.com/S2.zarr?token=secret", bands=["b04"]),
        )


def test_eopf_source_reports_out_of_bounds_region(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    product = tmp_path / "S2_TEST.zarr"
    product.mkdir()
    monkeypatch.setattr(xr, "open_dataset", lambda *args, **kwargs: _dataset())

    with pytest.raises(ValueError, match="does not intersect"):
        EOPFZarrRasterSource(
            RegionConfig(west=-80, south=-40, east=-79, north=-39),
            EOPFZarrImageryConfig(path=str(product), bands=["b04"]),
        )


def test_eopf_source_wraps_remote_open_errors(monkeypatch: pytest.MonkeyPatch) -> None:
    def fail_open(*args: Any, **kwargs: Any) -> xr.Dataset:
        raise OSError("public store unavailable")

    monkeypatch.setattr(xr, "open_dataset", fail_open)
    with pytest.raises(RuntimeError, match="S2_TEST.zarr.*public store unavailable"):
        EOPFZarrRasterSource(
            _region(),
            EOPFZarrImageryConfig(path="https://example.com/S2_TEST.zarr", bands=["b04"]),
        )


def test_eopf_source_rejects_missing_local_product(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match="not found"):
        EOPFZarrRasterSource(
            _region(), EOPFZarrImageryConfig(path=str(tmp_path / "missing.zarr"), bands=["b04"])
        )


def _png_tile(value: int) -> bytes:
    buffer = BytesIO()
    Image.fromarray(np.full((256, 256, 3), value, dtype=np.uint8)).save(buffer, format="PNG")
    return buffer.getvalue()


def test_xyz_source_exposes_window_contract(monkeypatch: pytest.MonkeyPatch) -> None:
    transform = (2.0, 0.0, 100.0, 0.0, -2.0, 200.0)
    tile = PyTileIndex(3, 4, 12)
    monkeypatch.setattr(
        "mapcv.imagery.snap_bbox",
        lambda *args, **kwargs: SimpleNamespace(west=0, south=0, east=1, north=1),
    )
    monkeypatch.setattr("mapcv.imagery.tiles", lambda *args, **kwargs: [tile])
    monkeypatch.setattr(
        "mapcv.imagery.download_region", lambda *args, **kwargs: [(tile, _png_tile(7))]
    )
    monkeypatch.setattr("mapcv.imagery.tile_transform", lambda *args: transform)
    source = XYZRasterSource(_region(), XYZImageryConfig(zoom=12, source="osm", strip_rows=2))

    window, valid = source.read_window(20, 60, 30, 80)

    assert window.shape == (40, 50, 3)
    assert np.all(window == 7)
    assert valid.all()
    assert source.metadata.crs == "EPSG:3857"
    assert source.metadata.chunk_rows == 512


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


def test_transform_geometry_to_projected_crs() -> None:
    from pyproj import Transformer

    point = Point(10.0, 45.0)
    transformed = transform_geometry_to_crs(point, "EPSG:32632")
    expected_x, expected_y = Transformer.from_crs(
        "EPSG:4326", "EPSG:32632", always_xy=True
    ).transform(10.0, 45.0)

    assert transformed.x == pytest.approx(expected_x)
    assert transformed.y == pytest.approx(expected_y)
