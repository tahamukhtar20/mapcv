"""Tiled inference over a whole region, stitched back into one GeoTIFF.

:func:`predict_raster` reads the config's imagery (its first source, over its region)
patch by patch, hands batches to a numpy-in, numpy-out callable, and blends the
overlapping predictions with a cosine (Hann) window, so patch borders, where models
are least reliable, count least and leave no seams. The result is written as a
GeoTIFF on the imagery's grid: no torch, GDAL or rasterio needed.

Memory: the source is read in strips of rows (``chunk_rows`` of the imagery), and
only the strip's running sums are kept besides the output raster itself
(``height × width`` values of the output type).
"""

from __future__ import annotations

import math
import os
from pathlib import Path
from typing import Any, Callable, List, Literal, Optional, Tuple, Union

import numpy as np
import numpy.typing as npt

from mapcv._georef import epsg_code, is_geographic
from mapcv._mapcv_rs import grid_sample_anchors, write_geotiffs
from mapcv.config import GeoTiffImageryConfig, MapcvConfig
from mapcv.imagery import open_raster_source

Predictor = Callable[[npt.NDArray[Any]], npt.NDArray[Any]]
ARGMAX_NODATA = 255


def blend_window(size: int) -> npt.NDArray[np.float64]:
    """The ``size × size`` Hann weights: highest in the middle, small but never 0 at the
    edges (``sin²`` sampled at pixel centres), so every pixel keeps a weight."""
    ramp = np.sin(np.pi * (np.arange(size) + 0.5) / size) ** 2
    return np.outer(ramp, ramp)


def _stride(patch_size: int, overlap: float) -> int:
    if not 0.0 <= overlap < 1.0:
        raise ValueError(f"overlap must be in [0, 1), got {overlap}")
    return max(1, int(round(patch_size * (1.0 - overlap))))


def predict_raster(
    config: Union[MapcvConfig, str, "os.PathLike[str]"],
    fn: Predictor,
    out: Union[str, "os.PathLike[str]"],
    *,
    overlap: float = 0.5,
    batch_size: int = 16,
    output: Literal["argmax", "scores"] = "argmax",
    progress: Optional[Callable[[int, int], None]] = None,
) -> Path:
    """Predict every pixel of the config's region with ``fn`` and write a GeoTIFF.

    Args:
        config: A :class:`~mapcv.MapcvConfig` or the path of its YAML; its first imagery
            source, ``region`` and ``sampler.patch_size`` are used.
        fn: Called with a batch ``(N, C, P, P)`` of patches as read (the source's
            dtype; normalise inside ``fn``), returns ``(N, K, P, P)`` scores (logits or
            probabilities) or ``(N, P, P)`` values.
        out: The GeoTIFF to write.
        overlap: Fraction of a patch that neighbouring patches share (0 to < 1); more
            overlap, smoother blending, more calls to ``fn``.
        batch_size: Patches per call of ``fn``.
        output: ``"argmax"`` writes the class with the highest blended score as
            ``uint8`` (255 where there is no imagery); ``"scores"`` writes the blended
            scores as ``float32`` bands (``NaN`` where there is no imagery).
            ``(N, P, P)`` predictions always give one ``float32`` band.
        progress: ``progress(done, total)``, called with the strips finished so far.

    Returns:
        The written file.
    """
    if not isinstance(config, MapcvConfig):
        config = MapcvConfig.from_yaml(config)
    if output not in ("argmax", "scores"):
        raise ValueError(f"output must be 'argmax' or 'scores', got {output!r}")
    if batch_size < 1:
        raise ValueError("batch_size must be at least 1")
    imagery = config.primary_imagery
    if isinstance(imagery, GeoTiffImageryConfig):
        # Any band count and data type: prediction reads raw arrays.
        source = open_raster_source(config.region, imagery, image_format="npy")
    else:
        source = open_raster_source(config.region, imagery)
    try:
        return _predict(
            source, config.sampler.patch_size, fn, Path(out), overlap, batch_size, output, progress
        )
    finally:
        source.close()


def _predict(
    source: Any,
    patch: int,
    fn: Predictor,
    out: Path,
    overlap: float,
    batch_size: int,
    output: str,
    progress: Optional[Callable[[int, int], None]],
) -> Path:
    meta = source.metadata
    height, width = meta.height, meta.width
    if meta.crs is None:
        raise ValueError("the imagery has no CRS, so the prediction cannot be georeferenced")
    anchors = grid_sample_anchors(height, width, patch, _stride(patch, overlap), "shift")
    rows = sorted({row for row, _ in anchors})
    by_row = {row: [col for r, col in anchors if r == row] for row in rows}
    weights = blend_window(patch)

    # Strips of patch rows: each strip reads the rows its patches cover and is done
    # with every output row above the next strip's first patch row.
    chunk = max(patch, meta.chunk_rows)
    strips: List[List[int]] = []
    for row in rows:
        if strips and row - strips[-1][0] < chunk:
            strips[-1].append(row)
        else:
            strips.append([row])

    result: Optional[npt.NDArray[Any]] = None
    channels = 0
    single = False
    valid_out = np.zeros((height, width), dtype=bool)
    carry: Optional[Tuple[int, npt.NDArray[np.float64], npt.NDArray[np.float64]]] = None
    if progress is not None:
        progress(0, len(strips))
    for done, strip in enumerate(strips, start=1):
        top = strip[0]
        bottom = min(height, strip[-1] + patch)
        image, valid = source.read_window(top, bottom, 0, width)
        valid_out[top:bottom] |= valid
        sums: Optional[npt.NDArray[np.float64]] = None
        total = np.zeros((bottom - top, width), dtype=np.float64)
        if carry is not None:
            carry_top, carry_sums, carry_total = carry
            offset = carry_top - top
            total[offset : offset + carry_total.shape[0]] += carry_total
        positions = [(row, col) for row in strip for col in by_row[row]]
        for start in range(0, len(positions), batch_size):
            batch_positions = positions[start : start + batch_size]
            batch = np.stack([_patch(image, row - top, col, patch) for row, col in batch_positions])
            predicted = np.asarray(fn(np.moveaxis(batch, -1, 1)))
            if predicted.ndim == 3:
                predicted = predicted[:, np.newaxis]
            if (
                predicted.ndim != 4
                or predicted.shape[0] != len(batch_positions)
                or predicted.shape[2:] != (patch, patch)
            ):
                raise ValueError(
                    f"fn must return (N, K, {patch}, {patch}) scores or (N, {patch}, {patch}) "
                    f"values for a batch of N={len(batch_positions)}, got {predicted.shape}"
                )
            if sums is None:
                channels = predicted.shape[1]
                single = channels == 1
                sums = np.zeros((channels, bottom - top, width), dtype=np.float64)
                if carry is not None:
                    carry_top, carry_sums, _ = carry
                    offset = carry_top - top
                    sums[:, offset : offset + carry_sums.shape[1]] += carry_sums
            for (row, col), scores in zip(batch_positions, predicted):
                r0 = row - top
                h = min(patch, height - row)
                w = min(patch, width - col)
                sums[:, r0 : r0 + h, col : col + w] += scores[:, :h, :w] * weights[:h, :w]
                total[r0 : r0 + h, col : col + w] += weights[:h, :w]
        assert sums is not None
        # Rows above the next strip's first patch are final.
        final_stop = (strips[done][0] if done < len(strips) else height) - top
        if result is None:
            result = (
                np.full((height, width), ARGMAX_NODATA, dtype=np.uint8)
                if output == "argmax" and not single
                else np.full((channels, height, width), np.nan, dtype=np.float32)
            )
        blended = sums[:, :final_stop] / np.maximum(total[:final_stop], 1e-12)
        if result.ndim == 2:
            classes = np.argmax(blended, axis=0)
            if classes.size and int(classes.max()) >= ARGMAX_NODATA:
                raise ValueError(f"argmax output holds at most {ARGMAX_NODATA} classes")
            result[top : top + final_stop] = classes.astype(np.uint8)
        else:
            result[:, top : top + final_stop] = blended.astype(np.float32)
        carry = (
            (top + final_stop, sums[:, final_stop:], total[final_stop:])
            if final_stop < bottom - top
            else None
        )
        if progress is not None:
            progress(done, len(strips))

    assert result is not None
    if result.ndim == 2:
        result[~valid_out] = ARGMAX_NODATA
        bands, data, dtype, nodata = 1, result[np.newaxis], "uint8", float(ARGMAX_NODATA)
    else:
        result[:, ~valid_out] = np.nan
        bands, data, dtype, nodata = result.shape[0], result, "float32", math.nan
    out.parent.mkdir(parents=True, exist_ok=True)
    code = epsg_code(meta.crs)
    pixels = np.ascontiguousarray(np.moveaxis(data, 0, -1))  # (H, W, bands)
    write_geotiffs(
        pixels.reshape(-1).view(np.uint8),
        dtype,
        (1, height, width, bands),
        [list(meta.transform)],
        [out.name],
        str(out.parent),
        code,
        is_geographic(code),
        nodata,
        None,
    )
    return out


def _patch(image: npt.NDArray[Any], row: int, col: int, patch: int) -> npt.NDArray[Any]:
    """The ``patch × patch`` window at ``(row, col)``, zero-padded past the edges."""
    piece = image[row : row + patch, col : col + patch]
    if piece.shape[:2] == (patch, patch):
        return piece
    padded = np.zeros((patch, patch) + image.shape[2:], dtype=image.dtype)
    padded[: piece.shape[0], : piece.shape[1]] = piece
    return padded
