"""Tests for the EOPF Zarr imagery source (needs the optional zarr extra)."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pytest
from shapely.geometry import Point

from mapcv.config import EOPFZarrImageryConfig, RegionConfig
from mapcv.imagery import (
    EOPFZarrRasterSource,
    transform_geometry_to_crs,
)

pytest.importorskip("xarray", reason="EOPF support needs the zarr extra")
pytest.importorskip("pyproj", reason="EOPF support needs the zarr extra")

import xarray as xr  # noqa: E402


def _dataset() -> xr.Dataset:
    b08 = np.arange(16, dtype=np.float64).reshape(4, 4)
    b04 = b08 + 100
    dataset: xr.Dataset = xr.Dataset(
        data_vars={"b08": (("y", "x"), b08), "b04": (("y", "x"), b04)},
        coords={
            "x": np.array([10.00, 10.05, 10.10, 10.15]),
            "y": np.array([45.15, 45.10, 45.05, 45.00]),
        },
        attrs={"crs": "EPSG:4326"},
    ).chunk({"x": 2, "y": 2})
    return dataset


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
    # The bbox is expanded to pixel edges of the 0.05-degree product grid.
    assert captured[1]["bbox"] == pytest.approx([9.975, 44.975, 10.175, 45.175])
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


def test_transform_geometry_to_projected_crs() -> None:
    from pyproj import Transformer

    point = Point(10.0, 45.0)
    transformed = transform_geometry_to_crs(point, "EPSG:32632")
    expected_x, expected_y = Transformer.from_crs(
        "EPSG:4326", "EPSG:32632", always_xy=True
    ).transform(10.0, 45.0)

    assert transformed.x == pytest.approx(expected_x)
    assert transformed.y == pytest.approx(expected_y)


def test_eopf_band_that_comes_back_empty_fails_the_read(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    product = tmp_path / "S2_TEST.zarr"
    product.mkdir()

    def dataset_with_failed_band(*args: Any, **kwargs: Any) -> xr.Dataset:
        dataset = _dataset()
        dataset["b04"] = dataset["b04"] * np.nan  # e.g. a timed-out chunk read
        return dataset

    monkeypatch.setattr(xr, "open_dataset", dataset_with_failed_band)
    source = EOPFZarrRasterSource(
        _region(), EOPFZarrImageryConfig(path=str(product), bands=["b08", "b04"])
    )
    with pytest.raises(RuntimeError, match="band b04 returned no data"):
        source.read_window(0, source.metadata.height, 0, source.metadata.width)


def test_eopf_pixel_is_invalid_when_any_band_is_missing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    product = tmp_path / "S2_TEST.zarr"
    product.mkdir()

    def dataset_with_one_hole(*args: Any, **kwargs: Any) -> xr.Dataset:
        dataset = _dataset().compute()
        dataset["b04"][0, 0] = np.nan
        return dataset

    monkeypatch.setattr(xr, "open_dataset", dataset_with_one_hole)
    source = EOPFZarrRasterSource(
        _region(), EOPFZarrImageryConfig(path=str(product), bands=["b08", "b04"])
    )
    _, valid = source.read_window(0, source.metadata.height, 0, source.metadata.width)
    assert not valid[0, 0] and valid.sum() == valid.size - 1
