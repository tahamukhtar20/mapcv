"""Opt-in public EOPF integration test (never required by ordinary PR CI)."""

from __future__ import annotations

import os
from urllib.parse import urlsplit

import numpy as np
import pytest

from mapcv.config import EOPFZarrImageryConfig, RegionConfig
from mapcv.imagery import EOPFZarrRasterSource, _dataset_crs


@pytest.mark.eopf_live
def test_anonymous_public_eopf_product() -> None:
    url = os.environ.get("MAPCV_EOPF_URL")
    if not url:
        pytest.skip("set MAPCV_EOPF_URL to an anonymous public Sentinel-2 L2A product")

    import xarray as xr
    from pyproj import Transformer

    storage_options = {"anon": True} if urlsplit(url).scheme == "s3" else None
    discovery = xr.open_dataset(
        url,
        engine="eopf-zarr",
        op_mode="analysis",
        variables=["b04"],
        resolution=60,
        chunks={},
        storage_options=storage_options,
    )
    try:
        crs = _dataset_crs(discovery)
        x = np.asarray(discovery.coords["x"].values)
        y = np.asarray(discovery.coords["y"].values)
        center_x = float(x[len(x) // 2])
        center_y = float(y[len(y) // 2])
        to_wgs84 = Transformer.from_crs(crs, "EPSG:4326", always_xy=True)
        west, south, east, north = to_wgs84.transform_bounds(
            center_x - 120,
            center_y - 120,
            center_x + 120,
            center_y + 120,
            densify_pts=21,
        )
    finally:
        discovery.close()

    source = EOPFZarrRasterSource(
        RegionConfig(west=west, south=south, east=east, north=north),
        EOPFZarrImageryConfig(path=url, bands=["b04"], resolution=60, chunk_rows=4),
    )
    try:
        image, valid = source.read_window(
            0,
            min(4, source.metadata.height),
            0,
            min(4, source.metadata.width),
        )
        assert image.dtype == np.float32
        assert image.shape[-1] == 1
        assert valid.any()
        assert source.metadata.crs != "EPSG:3857"
    finally:
        source.close()
