"""``mapcv.infer.predict_raster``: tiled prediction stitched into one GeoTIFF.

A predictor that returns a pixel's own value must give back the raster itself, for any
patch size, overlap and strip layout: blending averages a pixel's predictions with
weights, so equal predictions survive exactly and any misplaced patch shows. The output
is read back with rasterio and compared with the source raster and its grid.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import numpy.typing as npt
import pytest
import yaml

pytest.importorskip("rasterio", reason="the output is checked with rasterio")
import rasterio
from test_multi_source import EPSG, reference_transform, region_inside

from mapcv.config import MapcvConfig
from mapcv.infer import blend_window, predict_raster

WIDTH, HEIGHT = 300, 230


def _raster(tmp_path: Path, nodata_rows: int = 0) -> npt.NDArray[np.float32]:
    rng = np.random.default_rng(5)
    data = rng.uniform(0, 100, size=(3, HEIGHT, WIDTH)).astype(np.float32)
    data[:, :nodata_rows, :] = -9999
    with rasterio.open(
        tmp_path / "in.tif",
        "w",
        driver="GTiff",
        height=HEIGHT,
        width=WIDTH,
        count=3,
        dtype="float32",
        crs=f"EPSG:{EPSG}",
        transform=reference_transform(),
        nodata=-9999 if nodata_rows else None,
    ) as dst:
        dst.write(data)
    return data


def _config(tmp_path: Path, patch: int, chunk_rows: int = 64) -> MapcvConfig:
    return MapcvConfig.model_validate(
        {
            "region": region_inside(reference_transform(), WIDTH, HEIGHT, margin=0.0),
            "imagery": {
                "type": "geotiff",
                "path": str(tmp_path / "in.tif"),
                "chunk_rows": chunk_rows,
            },
            "sampler": {"patch_size": patch},
            "writer": {"staging_dir": str(tmp_path / "unused")},
        }
    )


def _read(path: Path) -> tuple[npt.NDArray[Any], Any]:
    with rasterio.open(path) as src:
        return src.read(), src


def _window(path: Path) -> tuple[int, int, int, int]:
    """Where the output sits on the input grid: (row, col, height, width)."""
    with rasterio.open(path) as src:
        transform, height, width = src.transform, src.height, src.width
        assert src.crs.to_epsg() == EPSG
    ref = reference_transform()
    col = round((transform.c - ref.c) / ref.a)
    row = round((transform.f - ref.f) / ref.e)
    assert (transform.a, transform.e) == (ref.a, ref.e)
    return row, col, height, width


def band0(batch: npt.NDArray[Any]) -> npt.NDArray[np.float32]:
    return batch[:, 0].astype(np.float32)


@pytest.mark.parametrize(
    ("patch", "overlap", "chunk_rows"),
    [(64, 0.5, 64), (64, 0.0, 1024), (50, 0.25, 40), (37, 0.75, 100), (128, 0.5, 64)],
)
def test_a_pixel_value_predictor_gives_back_the_raster(
    tmp_path: Path, patch: int, overlap: float, chunk_rows: int
) -> None:
    data = _raster(tmp_path)
    out = predict_raster(
        _config(tmp_path, patch, chunk_rows), band0, tmp_path / "out" / "p.tif", overlap=overlap
    )
    values, _ = _read(out)
    row, col, height, width = _window(out)
    assert values.shape == (1, height, width) and values.dtype == np.float32
    np.testing.assert_allclose(values[0], data[0, row : row + height, col : col + width], rtol=1e-5)


def test_argmax_classes_and_pixels_without_imagery(tmp_path: Path) -> None:
    data = _raster(tmp_path, nodata_rows=30)

    def one_hot(batch: npt.NDArray[Any]) -> npt.NDArray[np.float32]:
        classes = np.floor(batch[:, 0]).astype(np.int64) % 3
        return np.stack([(classes == k).astype(np.float32) for k in range(3)], axis=1)

    out = predict_raster(_config(tmp_path, 48), one_hot, tmp_path / "classes.tif", overlap=0.5)
    values, _ = _read(out)
    row, col, height, width = _window(out)
    expected = (np.floor(data[0]).astype(np.int64) % 3).astype(np.uint8)[
        row : row + height, col : col + width
    ]
    missing = (data[0] == -9999)[row : row + height, col : col + width]
    assert values.dtype == np.uint8 and missing.any()
    np.testing.assert_array_equal(values[0][~missing], expected[~missing])
    assert (values[0][missing] == 255).all()
    with rasterio.open(out) as src:
        assert src.nodata == 255

    scores = predict_raster(
        _config(tmp_path, 48), one_hot, tmp_path / "scores.tif", output="scores"
    )
    values, _ = _read(scores)
    assert values.shape[0] == 3 and values.dtype == np.float32
    assert np.isnan(values[:, missing]).all()
    np.testing.assert_allclose(values[:, ~missing].sum(axis=0), 1.0, rtol=1e-5)


def test_batches_progress_yaml_configs_and_small_rasters(tmp_path: Path) -> None:
    data = _raster(tmp_path)
    sizes: list[int] = []
    calls: list[tuple[int, int]] = []

    def record(batch: npt.NDArray[Any]) -> npt.NDArray[np.float32]:
        sizes.append(batch.shape[0])
        assert batch.dtype == np.float32 and batch.shape[1] == 3
        return band0(batch)

    config = _config(tmp_path, 64)
    path = tmp_path / "config.yaml"
    path.write_text(yaml.safe_dump(config.model_dump(mode="json")))
    out = predict_raster(
        path, record, tmp_path / "p.tif", batch_size=5, progress=lambda d, t: calls.append((d, t))
    )
    assert max(sizes) == 5 and calls[0][0] == 0 and calls[-1][0] == calls[-1][1] == len(calls) - 1
    # A patch larger than the raster: one zero-padded patch, cropped back.
    big = predict_raster(_config(tmp_path, 512), band0, tmp_path / "big.tif")
    values, _ = _read(big)
    row, col, height, width = _window(big)
    np.testing.assert_allclose(values[0], data[0, row : row + height, col : col + width], rtol=1e-5)
    assert out.exists()


def test_refusals(tmp_path: Path) -> None:
    _raster(tmp_path)
    config = _config(tmp_path, 64)
    with pytest.raises(ValueError, match="overlap"):
        predict_raster(config, band0, tmp_path / "x.tif", overlap=1.0)
    with pytest.raises(ValueError, match="output"):
        predict_raster(config, band0, tmp_path / "x.tif", output="logits")  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="batch_size"):
        predict_raster(config, band0, tmp_path / "x.tif", batch_size=0)
    with pytest.raises(ValueError, match=r"fn must return \(N, K, 64, 64\)"):
        predict_raster(config, lambda b: b[:, 0, :10], tmp_path / "x.tif")
    with pytest.raises(ValueError, match="at most 255 classes"):
        predict_raster(
            config,
            lambda b: np.ones((b.shape[0], 300, 64, 64)) * np.arange(300)[None, :, None, None],
            tmp_path / "x.tif",
        )


def test_blend_window_is_symmetric_and_never_zero() -> None:
    weights = blend_window(8)
    assert weights.shape == (8, 8) and (weights > 0).all()
    np.testing.assert_allclose(weights, weights.T)
    np.testing.assert_allclose(weights, weights[::-1, ::-1])
    assert weights[3, 3] == weights.max() and weights[0, 0] == weights.min()


@pytest.mark.parametrize(
    ("patch", "overlap", "chunk_rows"), [(64, 0.5, 64), (40, 0.6, 50), (64, 0.0, 30)]
)
def test_disagreeing_patches_blend_like_one_pass_over_the_raster(
    tmp_path: Path, patch: int, overlap: float, chunk_rows: int
) -> None:
    """Each patch predicts its own mean everywhere, so patches disagree; the strip-wise
    result must equal a brute-force weighted average over the whole raster at once."""
    from mapcv._mapcv_rs import grid_sample_anchors

    data = _raster(tmp_path)

    def patch_mean(batch: npt.NDArray[Any]) -> npt.NDArray[np.float64]:
        means = batch[:, 0].reshape(batch.shape[0], -1).mean(axis=1)
        return np.broadcast_to(means[:, None, None], (batch.shape[0], patch, patch)).copy()

    out = predict_raster(
        _config(tmp_path, patch, chunk_rows), patch_mean, tmp_path / "m.tif", overlap=overlap
    )
    values, _ = _read(out)
    row, col, height, width = _window(out)
    image = data[0, row : row + height, col : col + width].astype(np.float64)
    stride = max(1, int(round(patch * (1 - overlap))))
    weights = blend_window(patch)
    sums = np.zeros((height, width))
    total = np.zeros((height, width))
    for r, c in grid_sample_anchors(height, width, patch, stride, "shift"):
        piece = np.zeros((patch, patch))
        h, w = min(patch, height - r), min(patch, width - c)
        piece[:h, :w] = image[r : r + h, c : c + w]
        sums[r : r + h, c : c + w] += piece.mean() * weights[:h, :w]
        total[r : r + h, c : c + w] += weights[:h, :w]
    np.testing.assert_allclose(values[0], sums / total, rtol=1e-5)
