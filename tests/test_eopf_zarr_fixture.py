"""A real (tiny) EOPF Sentinel-2 L2A Zarr product, opened through xarray-eopf.

The other EOPF tests mock ``xr.open_dataset``; these build a product with the EOPF
layout (``measurements/reflectance/r10m`` and ``r20m``, ``uint16`` digital numbers with
``scale_factor``/``add_offset``/``_FillValue``, the CRS in ``stac_discovery``) and run
the real ``eopf-zarr`` engine, so a broken engine install or an upstream change in how
products are read fails here. 10 m bands must come through exactly (decoded, with
``NaN`` where the product has no data); 20 m bands are resampled by xarray-eopf.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import numpy.typing as npt
import pytest

xr = pytest.importorskip("xarray", reason="needs the zarr extra")
pytest.importorskip("xarray_eopf", reason="needs the zarr extra")
from pyproj import Transformer

from mapcv.config import EOPFZarrImageryConfig, MapcvConfig, RegionConfig
from mapcv.imagery import EOPFZarrRasterSource
from mapcv.manifest import Manifest
from mapcv.pipeline import run_generate

X0, Y0 = 600000.0, 5400000.0
WIDTH, HEIGHT = 120, 100  # at 10 m
EPSG = 32631
NAME = "S2B_MSIL2A_20250101T103021_N0511_R108_T31UFT_20250101T120000.zarr"


def _digital_numbers(width: int, height: int, seed: int) -> npt.NDArray[np.uint16]:
    values = np.random.default_rng(seed).integers(1000, 4000, size=(height, width))
    values[:3, :5] = 0  # no data
    return values.astype(np.uint16)


def _decoded(values: npt.NDArray[np.uint16]) -> npt.NDArray[np.float64]:
    return np.where(values == 0, np.nan, values * 0.0001 - 0.1)


def _group(resolution: int, bands: dict[str, npt.NDArray[np.uint16]]) -> Any:
    height, width = next(iter(bands.values())).shape
    attrs = {"scale_factor": 0.0001, "add_offset": -0.1, "_FillValue": 0, "proj:epsg": EPSG}
    return xr.Dataset(
        {
            name: xr.DataArray(values, dims=("y", "x"), attrs=attrs)
            for name, values in bands.items()
        },
        coords={
            "x": X0 + resolution * (np.arange(width) + 0.5),
            "y": Y0 - resolution * (np.arange(height) + 0.5),
        },
    )


@pytest.fixture(scope="module")
def product(tmp_path_factory: pytest.TempPathFactory) -> dict[str, object]:
    r10 = {
        name: _digital_numbers(WIDTH, HEIGHT, seed)
        for seed, name in enumerate(["b02", "b03", "b04", "b08"])
    }
    r20 = {
        name: _digital_numbers(WIDTH // 2, HEIGHT // 2, seed + 10)
        for seed, name in enumerate(["b05", "b11"])
    }
    tree = xr.DataTree.from_dict(
        {
            "/": xr.Dataset(
                attrs={"stac_discovery": {"properties": {"proj:code": f"EPSG:{EPSG}"}}}
            ),
            "/measurements/reflectance/r10m": _group(10, r10),
            "/measurements/reflectance/r20m": _group(20, r20),
        }
    )
    path = tmp_path_factory.mktemp("eopf") / NAME
    tree.to_zarr(path, mode="w")
    return {"path": path, "r10": r10, "r20": r20}


def _region(col0: int, row0: int, col1: int, row1: int) -> dict[str, float]:
    """A lon/lat box just inside the given 10 m pixel span (the grid snaps outward)."""
    to_lonlat = Transformer.from_crs(f"EPSG:{EPSG}", "EPSG:4326", always_xy=True)
    corners = [
        to_lonlat.transform(X0 + 10 * c + dx, Y0 - 10 * r - dy)
        for c, r, dx, dy in (
            (col0, row1, 1, -1),
            (col1, row0, -1, 1),
            (col0, row0, 1, 1),
            (col1, row1, -1, -1),
        )
    ]
    lons, lats = zip(*corners)
    return {"west": max(min(lons), -180), "south": min(lats), "east": max(lons), "north": max(lats)}


def test_the_eopf_engine_is_installed() -> None:
    # xarray-eopf registers the engine on import of all its modes; a missing
    # undeclared dependency (pystac-client) makes it vanish silently.
    assert "eopf-zarr" in xr.backends.list_engines()


def test_the_source_reads_the_product_exactly(product: dict[str, object]) -> None:
    config = EOPFZarrImageryConfig(
        type="eopf_zarr", path=str(product["path"]), bands=["b04", "b08", "b11"]
    )
    source = EOPFZarrRasterSource(RegionConfig.model_validate(_region(10, 10, 100, 80)), config)
    try:
        meta = source.metadata
        assert meta.crs == f"EPSG:{EPSG}" and meta.dtype == "float32"
        col0 = round((meta.transform[2] - X0) / 10)
        row0 = round((Y0 - meta.transform[5]) / 10)
        assert meta.transform[0] == 10.0 and meta.transform[4] == -10.0
        image, valid = source.read_window(0, meta.height, 0, meta.width)
        r10 = product["r10"]
        assert isinstance(r10, dict)
        for index, band in enumerate(["b04", "b08"]):
            expected = _decoded(r10[band])[row0 : row0 + meta.height, col0 : col0 + meta.width]
            np.testing.assert_allclose(
                image[:, :, index], expected.astype(np.float32), equal_nan=True
            )
        assert not valid[np.isnan(image).any(axis=-1)].any()
        b11 = image[:, :, 2][valid]
        assert np.all((b11 >= 0.0) & (b11 <= 0.3))  # resampled from 20 m, within the data range
    finally:
        source.close()


def test_a_dataset_from_the_product(product: dict[str, object], tmp_path: Path) -> None:
    config = MapcvConfig.model_validate(
        {
            "region": _region(10, 10, 74, 74),
            "imagery": {"type": "eopf_zarr", "path": str(product["path"]), "bands": ["b02", "b04"]},
            "sampler": {"patch_size": 32, "edge_strategy": "drop", "max_empty_ratio": 1.0},
            "writer": {"staging_dir": str(tmp_path / "dataset"), "image_format": "npy"},
        }
    )
    run_generate(config)
    manifest = Manifest.load(tmp_path / "dataset" / "manifest.json")
    assert manifest.source.source_type == "eopf_zarr" and len(manifest.patches) >= 4
    r10 = product["r10"]
    assert isinstance(r10, dict)
    for entry in manifest.patches:
        _a, _, c, _, _e, f = manifest.patch_transform(entry)
        col, row = round((c - X0) / 10), round((Y0 - f) / 10)
        patch = np.load(tmp_path / "dataset" / entry["files"]["image"])
        for index, band in enumerate(["b02", "b04"]):
            expected = _decoded(r10[band])[row : row + 32, col : col + 32].astype(np.float32)
            np.testing.assert_allclose(patch[index], expected, equal_nan=True)
