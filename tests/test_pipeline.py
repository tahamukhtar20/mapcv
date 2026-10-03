"""End-to-end tests for the source-neutral generation pipeline."""

from __future__ import annotations

from pathlib import Path
from typing import Any, List, Tuple

import numpy as np
import numpy.typing as npt
import pytest

from mapcv.config import MapcvConfig
from mapcv.imagery import RasterMetadata
from mapcv.pipeline import run_generate
from mapcv.writer import Manifest


class FakeRasterSource:
    """Small chunked raster used to exercise global anchor behavior."""

    def __init__(self) -> None:
        values = np.arange(7 * 5 * 2, dtype=np.float32)
        self.image = values.reshape(7, 5, 2)
        self.valid = np.ones((7, 5), dtype=np.bool_)
        self.windows: List[Tuple[int, int, int, int]] = []
        self.closed = False
        self.metadata = RasterMetadata(
            source_type="eopf_zarr",
            product_id="S2_TEST.zarr",
            width=5,
            height=7,
            bands=["b08", "b04"],
            dtype="float32",
            crs="EPSG:32632",
            transform=(10.0, 0.0, 500000.0, 0.0, -10.0, 5000000.0),
            chunk_rows=2,
        )

    def read_window(
        self, row_start: int, row_stop: int, col_start: int, col_stop: int
    ) -> Tuple[npt.NDArray[np.float32], npt.NDArray[np.bool_]]:
        self.windows.append((row_start, row_stop, col_start, col_stop))
        return (
            self.image[row_start:row_stop, col_start:col_stop],
            self.valid[row_start:row_stop, col_start:col_stop],
        )

    def close(self) -> None:
        self.closed = True


def _config(tmp_path: Path) -> MapcvConfig:
    return MapcvConfig.model_validate(
        {
            "region": {"west": 9.0, "south": 45.0, "east": 9.1, "north": 45.1},
            "imagery": {
                "type": "eopf_zarr",
                "path": str(tmp_path / "unused.zarr"),
                "bands": ["b08", "b04"],
                "chunk_rows": 2,
            },
            "sampler": {
                "patch_size": 3,
                "stride": 2,
                "edge_strategy": "drop",
            },
            "writer": {"staging_dir": str(tmp_path / "dataset"), "image_format": "npy"},
        }
    )


def test_generate_keeps_global_anchors_across_chunk_seams_and_resumes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    sources: List[FakeRasterSource] = []

    def open_source(*args: Any, **kwargs: Any) -> FakeRasterSource:
        source = FakeRasterSource()
        sources.append(source)
        return source

    monkeypatch.setattr("mapcv.pipeline.open_raster_source", open_source)
    config = _config(tmp_path)

    run_generate(config)

    manifest_path = config.writer.staging_dir / "manifest.json"
    manifest = Manifest.load(manifest_path)
    coordinates = [(entry["row"], entry["col"]) for entry in manifest.patches]
    assert coordinates == [(0, 0), (0, 2), (2, 0), (2, 2), (4, 0), (4, 2)]
    assert sources[0].windows == [(0, 3, 0, 5), (2, 5, 0, 5), (4, 7, 0, 5)]
    assert sources[0].closed
    assert manifest.version == 2
    assert manifest.source_type == "eopf_zarr"
    assert manifest.bands == ["b08", "b04"]
    assert manifest.dtype == "float32"
    assert manifest.patch_shape == [2, 3, 3]
    assert manifest.crs == "EPSG:32632"
    assert manifest.transform == (10.0, 0.0, 500000.0, 0.0, -10.0, 5000000.0)

    stored = np.load(config.writer.staging_dir / "Images" / "patch_0000000.npy")
    assert stored.shape == (2, 3, 3)
    assert stored.dtype == np.float32

    run_generate(config)

    resumed = Manifest.load(manifest_path)
    assert [(entry["row"], entry["col"]) for entry in resumed.patches] == coordinates
    assert sources[1].windows == []
    assert sources[1].closed
