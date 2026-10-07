"""Regression datasets (``task: regression``): float targets from a continuous raster.

Target patches are compared with what rasterio gives for the same raster: a direct
window read when it is on the imagery grid, and ``rasterio.warp.reproject`` with
nearest neighbour (``tolerance=0``, exact) onto each patch's grid otherwise, then
``value * scale + offset`` with NaN where the raster has no valid value.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import numpy.typing as npt
import pytest
from pydantic import ValidationError
from typer.testing import CliRunner

pytest.importorskip("rasterio", reason="regression tests write their rasters with rasterio")
import rasterio
import rasterio.warp
from rasterio.crs import CRS
from rasterio.enums import Resampling
from rasterio.transform import Affine
from test_multi_source import (
    EPSG,
    PATCH,
    reference_transform,
    region_inside,
    write_raster,
)

from mapcv.cli import app
from mapcv.config import MapcvConfig
from mapcv.manifest import Manifest
from mapcv.pipeline import run_generate
from mapcv.planning import plan

runner = CliRunner()
WIDTH, HEIGHT = 448, 384


def write_values(
    path: Path,
    transform: Affine,
    width: int,
    height: int,
    *,
    dtype: str = "float32",
    nodata: float | None = None,
    epsg: int = EPSG,
    seed: int = 4,
) -> npt.NDArray[Any]:
    """A continuous raster with a NoData block (when ``nodata`` is given)."""
    rng = np.random.default_rng(seed)
    if np.dtype(dtype).kind == "f":
        data = rng.uniform(0.0, 40.0, size=(height, width)).astype(dtype)
    else:
        data = rng.integers(0, 3000, size=(height, width)).astype(dtype)
    if nodata is not None:
        data[100:140, 50:200] = nodata
    profile: dict[str, Any] = {
        "driver": "GTiff",
        "height": height,
        "width": width,
        "count": 1,
        "dtype": dtype,
        "crs": CRS.from_epsg(epsg),
        "transform": transform,
        "tiled": True,
        "blockxsize": 128,
        "blockysize": 128,
        "compress": "deflate",
    }
    if nodata is not None:
        profile["nodata"] = nodata
    with rasterio.open(path, "w", **profile) as dst:
        dst.write(data, 1)
    return data


@pytest.fixture
def scene(tmp_path: Path) -> dict[str, Any]:
    ref = reference_transform()
    write_raster(tmp_path / "image.tif", ref, WIDTH, HEIGHT, seed=1)
    return {"region": region_inside(ref, WIDTH, HEIGHT), "ref": ref}


def regression_config(
    tmp_path: Path,
    region: dict[str, float],
    labels: dict[str, Any],
    *,
    staging: str = "dataset",
    **writer: Any,
) -> MapcvConfig:
    return MapcvConfig.model_validate(
        {
            "task": "regression",
            "region": region,
            "imagery": {"type": "geotiff", "path": str(tmp_path / "image.tif")},
            "labels": {"type": "continuous", **labels},
            "sampler": {"patch_size": PATCH, "edge_strategy": "drop"},
            "writer": {"staging_dir": str(tmp_path / staging), "image_format": "tif", **writer},
        }
    )


def expected_target(
    path: Path,
    patch: Affine,
    *,
    scale: float = 1.0,
    offset: float = 0.0,
    valid_min: float | None = None,
    valid_max: float | None = None,
) -> npt.NDArray[np.float32]:
    """rasterio's nearest-neighbour value at every patch pixel centre, scaled, NaN if invalid."""
    with rasterio.open(path) as src:
        data = src.read(1).astype(np.float64)
        nodata = src.nodata
        valid = np.isfinite(data)
        if nodata is not None:
            valid &= data != nodata
        if valid_min is not None:
            valid &= data >= valid_min
        if valid_max is not None:
            valid &= data <= valid_max
        marked = np.where(valid, data, -1e30)
        out = np.full((PATCH, PATCH), -2e30, dtype=np.float64)
        rasterio.warp.reproject(
            marked,
            out,
            src_transform=src.transform,
            src_crs=src.crs,
            dst_transform=patch,
            dst_crs=CRS.from_epsg(EPSG),
            resampling=Resampling.nearest,
            src_nodata=None,
            dst_nodata=None,
            init_dest_nodata=False,
            tolerance=0,
        )
    target = (out * scale + offset).astype(np.float32)
    target[(out <= -1e30)] = np.nan  # invalid values and pixels outside the raster
    return target


def _target(staging: Path, entry: Any) -> npt.NDArray[np.float32]:
    path = staging / entry["files"]["mask"]
    if path.suffix == ".npy":
        loaded: npt.NDArray[np.float32] = np.load(path)
        return loaded
    with rasterio.open(path) as patch:
        values: npt.NDArray[np.float32] = patch.read(1)
        assert patch.dtypes[0] == "float32" and np.isnan(patch.nodata)
    return values


def _assert_targets(config: MapcvConfig, manifest: Manifest, path: Path, **scaling: Any) -> int:
    staging = config.writer.staging_dir
    compared = 0
    for entry in manifest.patches:
        want = expected_target(path, Affine(*manifest.patch_transform(entry)), **scaling)
        got = _target(staging, entry)
        np.testing.assert_array_equal(got, want, err_msg=f"{entry['row']},{entry['col']}")
        values = entry["summary"]["values"]
        finite = want[np.isfinite(want)].astype(np.float64)
        assert values["valid"] == finite.size
        if finite.size:
            assert values["min"] == pytest.approx(finite.min())
            assert values["max"] == pytest.approx(finite.max())
            assert values["mean"] == pytest.approx(finite.mean())
        compared += int(finite.size > 0)
    assert compared > 0
    return compared


# ── Datasets ─────────────────────────────────────────────────────────────────


def test_targets_on_the_imagery_grid_with_nodata(tmp_path: Path, scene: dict[str, Any]) -> None:
    write_values(tmp_path / "height.tif", scene["ref"], WIDTH, HEIGHT, nodata=-9999.0)
    config = regression_config(tmp_path, scene["region"], {"path": str(tmp_path / "height.tif")})
    assert config.writer.mask_format == "tif"  # floats do not fit PNG: GeoTIFF by default
    manifest = run_generate(config).manifest
    assert manifest.task == "regression"
    assert manifest.target is not None and manifest.target.dtype == "float32"
    _assert_targets(config, manifest, tmp_path / "height.tif")
    assert any(entry["summary"]["values"]["valid"] < PATCH * PATCH for entry in manifest.patches)


def test_scaled_integer_values_on_another_crs_and_grid(
    tmp_path: Path, scene: dict[str, Any]
) -> None:
    # Decimetres in an int16 raster, in lon/lat at about 3 m, NoData -1.
    ref = scene["ref"]
    west, south, east, north = rasterio.warp.transform_bounds(
        CRS.from_epsg(EPSG), CRS.from_epsg(4326), ref.c, ref.f - HEIGHT, ref.c + WIDTH, ref.f
    )
    step = 3e-5
    lonlat = Affine(step, 0.0, west - 10 * step, 0.0, -step, north + 10 * step)
    cols, rows = int((east - west) / step) + 20, int((north - south) / step) + 20
    write_values(tmp_path / "dem.tif", lonlat, cols, rows, dtype="int16", nodata=-1, epsg=4326)
    labels = {"path": str(tmp_path / "dem.tif"), "scale": 0.1, "offset": 100.0, "valid_max": 2500}
    config = regression_config(tmp_path, scene["region"], labels, mask_format="npy")
    manifest = run_generate(config).manifest
    _assert_targets(config, manifest, tmp_path / "dem.tif", scale=0.1, offset=100.0, valid_max=2500)


def test_pixels_without_imagery_have_no_target(tmp_path: Path, scene: dict[str, Any]) -> None:
    ref = scene["ref"]
    write_raster(tmp_path / "image.tif", ref, WIDTH, HEIGHT, seed=1)
    with rasterio.open(tmp_path / "image.tif", "r+") as dst:
        dst.nodata = 0
        data = dst.read()
        data[:, 200:260, 100:300] = 0  # imagery NoData
        dst.write(data)
    write_values(tmp_path / "height.tif", ref, WIDTH, HEIGHT)
    config = regression_config(
        tmp_path, scene["region"], {"path": str(tmp_path / "height.tif")}, staging="holes"
    )
    config.sampler.max_empty_ratio = 1.0
    manifest = run_generate(config).manifest
    holes = 0
    for entry in manifest.patches:
        patch = Affine(*manifest.patch_transform(entry))
        col, row = (round(v) for v in ~ref * (patch.c, patch.f))
        no_imagery = np.all(data[:, row : row + PATCH, col : col + PATCH] == 0, axis=0)
        target = _target(config.writer.staging_dir, entry)
        assert np.isnan(target[no_imagery]).all()
        assert np.isfinite(target[~no_imagery]).all()
        holes += int(no_imagery.any())
    assert holes > 0


def test_min_label_ratio_counts_pixels_with_a_value(tmp_path: Path, scene: dict[str, Any]) -> None:
    write_values(tmp_path / "height.tif", scene["ref"], WIDTH, HEIGHT, nodata=-9999.0)
    config = regression_config(
        tmp_path, scene["region"], {"path": str(tmp_path / "height.tif")}, staging="ratio"
    )
    config.sampler.min_label_ratio = 0.999
    manifest = run_generate(config).manifest
    for entry in manifest.patches:
        assert entry["summary"]["values"]["valid"] >= 0.999 * PATCH * PATCH


def test_resume_plan_info_and_several_sources(tmp_path: Path, scene: dict[str, Any]) -> None:
    write_values(tmp_path / "height.tif", scene["ref"], WIDTH, HEIGHT, nodata=-9999.0)
    config = regression_config(tmp_path, scene["region"], {"path": str(tmp_path / "height.tif")})
    full = run_generate(config).manifest
    staging = config.writer.staging_dir
    files = {path: path.read_bytes() for path in staging.rglob("*.tif")}
    cut = Manifest.load(staging / "manifest.json")
    cut.patches = cut.patches[: len(cut.patches) // 2]
    cut.save(staging / "manifest.json")
    assert run_generate(config).manifest.patches == full.patches
    assert {path: path.read_bytes() for path in staging.rglob("*.tif")} == files
    changed = regression_config(
        tmp_path, scene["region"], {"path": str(tmp_path / "height.tif"), "scale": 2.0}
    )
    with pytest.raises(ValueError, match="labels"):
        run_generate(changed)

    estimate = plan(config)
    assert estimate.task == "regression" and estimate.labels is not None
    assert estimate.labels.raster is not None and "height.tif" in estimate.labels.raster
    result = runner.invoke(app, ["info", str(staging)], env={"COLUMNS": "200"})
    assert result.exit_code == 0, result.output
    assert "pixels with a value" in result.output and "mean" in result.output

    # Two image sources with one target: the multi-source layout applies.
    write_raster(tmp_path / "image2.tif", scene["ref"], WIDTH, HEIGHT, seed=5)
    two = MapcvConfig.model_validate(
        {
            **config.model_dump(mode="json", exclude={"imagery", "writer"}),
            "imagery": [
                {"type": "geotiff", "name": "a", "path": str(tmp_path / "image.tif")},
                {"type": "geotiff", "name": "b", "path": str(tmp_path / "image2.tif")},
            ],
            "writer": {"staging_dir": str(tmp_path / "two"), "image_format": "tif"},
        }
    )
    entry = run_generate(two).manifest.patches[0]
    assert set(entry["files"]) == {"a", "b", "mask"}


# ── Config ───────────────────────────────────────────────────────────────────


XYZ = {"type": "xyz", "zoom": 18, "source": "esri_satellite"}
BASE: dict[str, Any] = {
    "region": {"west": 4.9, "south": 52.3, "east": 4.91, "north": 52.31},
    "imagery": XYZ,
    "sampler": {"patch_size": 256},
    "writer": {"staging_dir": "out", "image_format": "tif"},
}
VALUES = {"type": "continuous", "path": "chm.tif"}


@pytest.mark.parametrize(
    ("changes", "message"),
    [
        ({"task": "regression"}, "task: regression needs labels.type: continuous"),
        (
            {"task": "regression", "labels": {"path": "a.geojson"}},
            "task: regression needs labels.type: continuous",
        ),
        ({"labels": VALUES}, "labels.type: continuous holds values to predict"),
        ({"task": "detection", "labels": VALUES}, "set task: regression (task is 'detection')"),
        ({"task": "regression", "labels": {**VALUES, "scale": 0}}, "labels.scale must be"),
        (
            {"task": "regression", "labels": {**VALUES, "valid_min": 5, "valid_max": 1}},
            "valid_min must not be above",
        ),
        (
            {
                "task": "regression",
                "labels": VALUES,
                "writer": {"staging_dir": "o", "mask_format": "png"},
            },
            "PNG cannot hold",
        ),
    ],
)
def test_config_refuses_bad_regression_settings(changes: dict[str, Any], message: str) -> None:
    with pytest.raises(ValidationError) as raised:
        MapcvConfig.model_validate({**BASE, **changes})
    assert message in str(raised.value)


def test_unknown_tasks_list_every_supported_task() -> None:
    with pytest.raises(ValidationError) as raised:
        MapcvConfig.model_validate({**BASE, "task": "forecast"})
    assert (
        "supported: segmentation, detection, instance, classification, change, regression"
        in str(raised.value)
    )
    assert "planned" not in str(raised.value)
