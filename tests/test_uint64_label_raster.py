"""A label raster of 64-bit unsigned integers keeps class values beyond int64."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from pyproj import Transformer

from mapcv.config import MapcvConfig
from mapcv.planning import plan


def test_a_uint64_label_raster_with_values_beyond_int64_is_classified(tmp_path: Path) -> None:
    pytest.importorskip("rasterio")
    import rasterio
    from rasterio.crs import CRS
    from rasterio.transform import Affine

    from mapcv.config import RasterLabelsConfig
    from mapcv.targets.raster_labels import LabelRasterSampler

    top = 2**64 - 1
    big = 2**63 + 5
    values = np.array([[0, 7, big, top], [top, big, 7, 0]], dtype=np.uint64)
    path = tmp_path / "labels.tif"
    transform = Affine(1.0, 0.0, 500_000.0, 0.0, -1.0, 5_400_000.0)
    with rasterio.open(
        path,
        "w",
        driver="GTiff",
        height=2,
        width=4,
        count=1,
        dtype="uint64",
        crs=CRS.from_epsg(32631),
        transform=transform,
    ) as dst:
        dst.write(values, 1)
    config = RasterLabelsConfig.model_validate(
        {"type": "raster", "path": str(path), "classes": {top: 1, big: 2, 7: 3}}
    )
    sampler = LabelRasterSampler(config, "EPSG:32631")
    mask = sampler.sample((1.0, 0.0, 500_000.0, 0.0, -1.0, 5_400_000.0), 2, 4)
    # 0 is unmapped (background); every other value maps through the class table.
    np.testing.assert_array_equal(mask, [[0, 3, 2, 1], [1, 2, 3, 0]])

    # `plan` reads the raster too: it used to stop with a traceback (OverflowError).
    imagery = tmp_path / "image.tif"
    with rasterio.open(
        imagery,
        "w",
        driver="GTiff",
        height=2,
        width=4,
        count=3,
        dtype="uint8",
        crs=CRS.from_epsg(32631),
        transform=transform,
    ) as dst:
        dst.write(np.full((3, 2, 4), 100, np.uint8))
    west, south = 500_000.0, 5_400_000.0 - 2.0
    lon0, lat0 = Transformer.from_crs(32631, 4326, always_xy=True).transform(west, south)
    lon1, lat1 = Transformer.from_crs(32631, 4326, always_xy=True).transform(west + 4, south + 2)
    estimate = plan(
        MapcvConfig.model_validate(
            {
                "region": {"west": lon0, "south": lat0, "east": lon1, "north": lat1},
                "imagery": {"type": "geotiff", "path": str(imagery)},
                "labels": {"type": "raster", "path": str(path), "classes": {top: 1}},
                "sampler": {"patch_size": 2, "stride": 0, "edge_strategy": "drop"},
                "writer": {"staging_dir": str(tmp_path / "out")},
            }
        )
    )
    assert estimate.labels is not None and estimate.labels.classes
