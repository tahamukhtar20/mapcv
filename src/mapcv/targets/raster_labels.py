"""Semantic segmentation from a classified label raster (GeoTIFF / COG).

The label raster may differ from the imagery in CRS, resolution and origin. Every
imagery pixel takes the value of the label pixel that contains its **centre**
(nearest neighbour: categorical values are never averaged or interpolated):

1. The centres ``(col + 0.5, row + 0.5)`` of an imagery window's pixels are mapped
   to the imagery CRS with the window's affine transform. Transforms are
   corner-based, in rasterio's ``(a, b, c, d, e, f)`` order, so pixel ``(0, 0)``
   spans ``[0, 1) x [0, 1)`` and its centre is ``(0.5, 0.5)``.
2. When the label raster has another CRS, the centres are projected into it with
   pyproj, exactly, every point on its own (in parallel threads; the coordinates are
   bit for bit those of one call per point). (GDAL's warper approximates the
   transformation with an error of up to 0.125 pixel by default; mapcv does not.)
3. The inverse of the label raster's affine transform gives fractional label pixel
   coordinates. The transform is corner-based for ``PixelIsPoint`` files too: the
   reader shifts it by half a pixel, as GDAL and rasterio do.
4. The label pixel is ``floor(coordinate + 1e-10)``. A centre that falls exactly on
   the boundary between two label pixels (possible when the label grid is finer
   than the imagery's or offset by half a pixel) takes the right / lower one, the
   same tie rule as GDAL's nearest-neighbour warp kernel.

Pixels whose centre falls outside the label raster, on its NoData value or on one
of ``labels.ignore_values`` get ``labels.ignore_index``; values missing from
``labels.classes`` get background or ``ignore_index`` (``labels.unmapped``).

When the label raster has the imagery's CRS and pixel grid (up to a whole-pixel
offset), which is the usual case for a raster made for the imagery, the window is
copied directly instead; the result is identical to the general path.
"""

from __future__ import annotations

import math
import os
import warnings
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Dict, Optional, Tuple

import numpy as np
import numpy.typing as npt

from mapcv._patching import MaskWindow
from mapcv.config import ContinuousLabelsConfig, RasterLabelsConfig
from mapcv.geotiff import GeoTiff
from mapcv.imagery import (
    RasterMetadata,
    _safe_product_id,
    geotiff_fingerprint,
    geotiff_location,
)
from mapcv.labels import ClassMap
from mapcv.manifest import TargetRecord
from mapcv.targets.base import Transform, WindowTarget

# Added before flooring a fractional label pixel coordinate, as GDAL's warp kernel does.
TIE_EPSILON = 1e-10
# Fast path: the imagery and label grids may differ by this much (scale, in pixels per
# pixel; offset, in pixels) and still be treated as the same grid.
_SAME_SCALE_TOLERANCE = 1e-12
_WHOLE_PIXEL_TOLERANCE = 1e-9
# Pixels transformed at a time on the general path, to bound memory.
_BLOCK_PIXELS = 1 << 20
# Label pixels read around the window estimated from its perimeter.
_MARGIN = 1
# Cross-CRS projection runs in this many threads, on pieces of at least ``_MIN_PIECE``
# points (smaller arrays stay on the calling thread). PROJ transforms one point after
# another, whatever the array size, and pyproj releases the GIL while it does.
_PROJ_THREADS = min(8, os.cpu_count() or 1)
_MIN_PIECE = 1 << 15
_proj_pool: Optional[ThreadPoolExecutor] = None


def _proj_executor() -> ThreadPoolExecutor:
    global _proj_pool
    if _proj_pool is None:
        _proj_pool = ThreadPoolExecutor(max_workers=_PROJ_THREADS, thread_name_prefix="mapcv-proj")
    return _proj_pool


def _forget_proj_pool() -> None:
    # A forked child (multiprocessing's ``fork`` start method, a DataLoader worker) inherits
    # the executor without its threads, and ``submit`` would wait for them forever.
    global _proj_pool
    _proj_pool = None


if hasattr(os, "register_at_fork"):  # pragma: no branch - Windows has no fork
    os.register_at_fork(after_in_child=_forget_proj_pool)


LABEL_RASTER_MISS_MESSAGE = (
    "the label raster does not overlap the imagery, so every mask pixel will be {value}. "
    "Check labels.path and the region"
)


def integer_nodata(nodata: Optional[float], dtype: np.dtype[Any]) -> Optional[int]:
    """The file's NoData value when an integer raster can hold it, else ``None``."""
    if nodata is None or not math.isfinite(nodata) or nodata != int(nodata):
        return None
    info = np.iinfo(dtype)
    return int(nodata) if info.min <= nodata <= info.max else None


class _Classifier:
    """Maps raw label values to mask values: class IDs, background or the ignore value."""

    def __init__(
        self,
        dtype: np.dtype[Any],
        mapping: Dict[int, int],
        ignored: Tuple[int, ...],
        unmapped: int,
        ignore: int,
    ) -> None:
        self._dtype = dtype
        info = np.iinfo(dtype)
        fits = {value: code for value, code in mapping.items() if info.min <= value <= info.max}
        fits.update({value: ignore for value in ignored if info.min <= value <= info.max})
        self._unmapped = unmapped
        self._lut: Optional[npt.NDArray[np.uint8]] = None
        if dtype.itemsize <= 2:
            # 8- and 16-bit rasters: one table lookup per pixel, through the unsigned view.
            unsigned = np.dtype(f"u{dtype.itemsize}")
            lut = np.full(1 << (8 * dtype.itemsize), unmapped, dtype=np.uint8)
            for value, code in fits.items():
                lut[int(np.array(value, dtype=dtype).view(unsigned))] = code
            self._lut = lut
            self._unsigned = unsigned
        keys = sorted(fits)
        self._keys = np.asarray(keys, dtype=np.int64)
        self._codes = np.asarray([fits[key] for key in keys], dtype=np.uint8)

    def __call__(self, values: npt.NDArray[Any]) -> npt.NDArray[np.uint8]:
        if self._lut is not None:
            codes: npt.NDArray[np.uint8] = self._lut[values.view(self._unsigned)]
            return codes
        out = np.full(values.shape, self._unmapped, dtype=np.uint8)
        if not len(self._keys):
            return out
        wide = values.astype(np.int64)
        position = np.clip(np.searchsorted(self._keys, wide), 0, len(self._keys) - 1)
        hit = self._keys[position] == wide
        out[hit] = self._codes[position[hit]]
        return out


class _GridSampler:
    """Reads a raster at the pixel centres of imagery windows (nearest neighbour).

    ``imagery_crs`` is the CRS the window transforms are expressed in. Subclasses set
    how raw values become output values (``_decode``), the output type and the value of
    pixels outside the raster (``_fill``).
    """

    _out_dtype: "np.dtype[Any]"
    _fill: Any

    def __init__(self, path: str, band: int, imagery_crs: str) -> None:
        name = _safe_product_id(path)
        self.location = geotiff_location(path)
        self._tif = GeoTiff(self.location)
        info = self._tif.info
        if info.epsg is None:
            raise ValueError(
                f"label raster '{name}' has no usable CRS: "
                f"{info.crs_error or 'no CRS in the file'}. mapcv reads files whose CRS is an "
                "EPSG code; re-project or re-tag the file."
            )
        if info.transform is None:
            raise ValueError(
                f"label raster '{name}' has no georeferencing (no pixel size/origin tags)"
            )
        if band > info.count:
            raise ValueError(f"labels.band is {band}, but '{name}' has {info.count} band(s)")
        a, b, _, d, e, _ = info.transform
        if a * e - b * d == 0:
            raise ValueError(f"label raster '{name}' has a degenerate pixel transform")

        self.name = name
        self.info = info
        self.crs = f"EPSG:{info.epsg}"
        self.transform: Transform = info.transform
        self._band = band - 1
        self._same_crs = self.crs == imagery_crs.upper()
        self._to_label: Any = None
        if not self._same_crs:
            from pyproj import Transformer

            self._to_label = Transformer.from_crs(imagery_crs, self.crs, always_xy=True)

    def _decode(self, values: npt.NDArray[Any]) -> npt.NDArray[Any]:
        raise NotImplementedError  # pragma: no cover - every sampler defines it

    # ── reading ──────────────────────────────────────────────────────────────

    def _read_codes(self, row0: int, row1: int, col0: int, col1: int) -> npt.NDArray[Any]:
        """Output values of label pixels ``[row0, row1) x [col0, col1)``; the fill outside."""
        data, inside = self._tif.read_window(row0, row1, col0, col1, bands=[self._band])
        codes = self._decode(np.ascontiguousarray(data[..., 0]))
        codes[~inside] = self._fill
        return codes

    def _indices(self, fractional: npt.NDArray[np.float64], size: int) -> npt.NDArray[np.int64]:
        """Label pixel index of each fractional coordinate; -1 outside the raster."""
        finite = np.isfinite(fractional)
        index = np.floor(np.where(finite, fractional, -1.0) + TIE_EPSILON)
        index[~finite | (index < 0) | (index >= size)] = -1
        return index.astype(np.int64)

    # ── geometry ─────────────────────────────────────────────────────────────

    def _composed(self, transform: Transform) -> Tuple[float, float, float, float, float, float]:
        """``(pu, pv, p0, qu, qv, q0)``: label column ``pu*u + pv*v + p0`` and row
        ``qu*u + qv*v + q0`` of the imagery pixel position ``(u, v)``, same CRS only.

        Origins are subtracted before scaling, which keeps large projected
        coordinates from costing precision.
        """
        a, b, c, d, e, f = transform
        la, lb, lc, ld, le, lf = self.transform
        det = la * le - lb * ld
        dc, df = c - lc, f - lf
        return (
            (le * a - lb * d) / det,
            (le * b - lb * e) / det,
            (le * dc - lb * df) / det,
            (la * d - ld * a) / det,
            (la * e - ld * b) / det,
            (la * df - ld * dc) / det,
        )

    def _fractional(
        self, transform: Transform, u: npt.NDArray[np.float64], v: npt.NDArray[np.float64]
    ) -> Tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
        """Fractional label ``(row, col)`` of imagery pixel positions ``(u, v)`` (broadcast)."""
        if self._same_crs:
            pu, pv, p0, qu, qv, q0 = self._composed(transform)
            return qu * u + qv * v + q0, pu * u + pv * v + p0
        a, b, c, d, e, f = transform
        lx, ly = self._project(a * u + b * v + c, d * u + e * v + f)
        la, lb, lc, ld, le, lf = self.transform
        det = la * le - lb * ld
        return (la * (ly - lf) - ld * (lx - lc)) / det, (le * (lx - lc) - lb * (ly - lf)) / det

    def _project(
        self, x: npt.NDArray[np.float64], y: npt.NDArray[np.float64]
    ) -> Tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
        """Imagery-CRS coordinates (broadcast) projected into the label CRS.

        PROJ transforms the points of an array one after another, so the array is cut
        into pieces that run in parallel threads (pyproj releases the GIL and gives each
        thread its own PROJ context). Every point goes through the same operation whatever
        piece it falls in: the coordinates are bit for bit those of a call per point
        (``tests/test_raster_label_projection.py`` checks this for several CRS pairs).
        """
        x, y = np.broadcast_arrays(x, y)
        shape = x.shape
        # Copies, projected in place.
        lx = np.array(x, dtype=np.float64, order="C").reshape(-1)
        ly = np.array(y, dtype=np.float64, order="C").reshape(-1)
        transformer = self._to_label

        def piece(lo: int, hi: int) -> None:
            transformer.transform(lx[lo:hi], ly[lo:hi], inplace=True)

        pieces = min(_PROJ_THREADS, len(lx) // _MIN_PIECE)
        if pieces < 2:
            piece(0, len(lx))
        else:
            bounds = [len(lx) * i // pieces for i in range(pieces + 1)]
            executor = _proj_executor()
            futures = [executor.submit(piece, bounds[i], bounds[i + 1]) for i in range(pieces)]
            for future in futures:
                future.result()
        return lx.reshape(shape), ly.reshape(shape)

    def grid_offset(self, transform: Transform) -> Optional[Tuple[int, int]]:
        """``(row, col)`` of the window's first pixel in the label raster when both share
        a CRS and a pixel grid (up to a whole-pixel offset), else ``None``."""
        if not self._same_crs:
            return None
        pu, pv, p0, qu, qv, q0 = self._composed(transform)
        if pv != 0.0 or qu != 0.0:
            return None
        if abs(pu - 1.0) > _SAME_SCALE_TOLERANCE or abs(qv - 1.0) > _SAME_SCALE_TOLERANCE:
            return None
        row, col = round(q0), round(p0)
        if abs(q0 - row) > _WHOLE_PIXEL_TOLERANCE or abs(p0 - col) > _WHOLE_PIXEL_TOLERANCE:
            return None
        return int(row), int(col)

    # ── sampling ─────────────────────────────────────────────────────────────

    def sample(
        self, transform: Transform, height: int, width: int, *, fast: bool = True
    ) -> npt.NDArray[Any]:
        """Output values of a ``height`` x ``width`` imagery window (nearest neighbour).

        ``fast=False`` skips the direct copy for identical grids (for tests).
        """
        offset = self.grid_offset(transform) if fast else None
        if offset is not None:
            row, col = offset
            return self._copy(row, col, height, width)
        if self._same_crs:
            pu, pv, _, qu, _, _ = self._composed(transform)
            if pv == 0.0 and qu == 0.0:
                return self._separable(transform, height, width)
        return self._general(transform, height, width)

    def _copy(self, row: int, col: int, height: int, width: int) -> npt.NDArray[Any]:
        info = self.info
        if row >= info.height or col >= info.width or row + height <= 0 or col + width <= 0:
            return np.full((height, width), self._fill, dtype=self._out_dtype)
        return self._read_codes(row, row + height, col, col + width)

    def _separable(self, transform: Transform, height: int, width: int) -> npt.NDArray[Any]:
        """Axis-aligned grids in one CRS: label rows depend on imagery rows only, and
        columns on columns, so two 1-D lookups index the label window."""
        u = np.arange(width, dtype=np.float64) + 0.5
        v = np.arange(height, dtype=np.float64) + 0.5
        pu, _, p0, _, qv, q0 = self._composed(transform)
        cols = self._indices(pu * u + p0, self.info.width)
        rows = self._indices(qv * v + q0, self.info.height)
        out = np.full((height, width), self._fill, dtype=self._out_dtype)
        row_hit, col_hit = rows >= 0, cols >= 0
        if not row_hit.any() or not col_hit.any():
            return out
        rows_in, cols_in = rows[row_hit], cols[col_hit]
        r0, c0 = int(rows_in.min()), int(cols_in.min())
        codes = self._read_codes(r0, int(rows_in.max()) + 1, c0, int(cols_in.max()) + 1)
        out[np.ix_(row_hit, col_hit)] = codes[np.ix_(rows_in - r0, cols_in - c0)]
        return out

    def _general(self, transform: Transform, height: int, width: int) -> npt.NDArray[Any]:
        """Any grids: every pixel centre is mapped on its own, a block of rows at a time."""
        out = np.full((height, width), self._fill, dtype=self._out_dtype)
        cache = self._perimeter_window(transform, height, width)
        codes = (
            self._read_codes(*cache)
            if cache is not None
            else np.empty((0, 0), dtype=self._out_dtype)
        )
        u = np.arange(width, dtype=np.float64)[None, :] + 0.5
        block = max(1, _BLOCK_PIXELS // max(width, 1))
        for start in range(0, height, block):
            stop = min(height, start + block)
            v = np.arange(start, stop, dtype=np.float64)[:, None] + 0.5
            frac_rows, frac_cols = self._fractional(transform, u, v)
            rows = self._indices(frac_rows, self.info.height)
            cols = self._indices(frac_cols, self.info.width)
            hit = (rows >= 0) & (cols >= 0)
            if not hit.any():
                continue
            rows_in, cols_in = rows[hit], cols[hit]
            need = (
                int(rows_in.min()),
                int(rows_in.max()) + 1,
                int(cols_in.min()),
                int(cols_in.max()) + 1,
            )
            if cache is None or not (
                cache[0] <= need[0]
                and need[1] <= cache[1]
                and cache[2] <= need[2]
                and need[3] <= cache[3]
            ):
                # The perimeter missed part of the window (a strongly curved projection):
                # read what this block needs.
                cache = (
                    need
                    if cache is None
                    else (
                        min(cache[0], need[0]),
                        max(cache[1], need[1]),
                        min(cache[2], need[2]),
                        max(cache[3], need[3]),
                    )
                )
                codes = self._read_codes(*cache)
            target = out[start:stop]
            target[hit] = codes[rows_in - cache[0], cols_in - cache[2]]
        return out

    def _perimeter_window(
        self, transform: Transform, height: int, width: int
    ) -> Optional[Tuple[int, int, int, int]]:
        """The label pixels under the window, estimated from its outermost pixel centres
        and widened by a margin; ``None`` when the window misses the label raster."""
        cols = np.arange(width, dtype=np.float64) + 0.5
        rows = np.arange(height, dtype=np.float64) + 0.5
        u = np.concatenate([cols, cols, np.full(height, 0.5), np.full(height, width - 0.5)])
        v = np.concatenate([np.full(width, 0.5), np.full(width, height - 0.5), rows, rows])
        frac_rows, frac_cols = self._fractional(transform, u, v)
        finite = np.isfinite(frac_rows) & np.isfinite(frac_cols)
        if not finite.any():
            return None
        info = self.info
        r0 = max(0, math.floor(float(frac_rows[finite].min())) - _MARGIN)
        r1 = min(info.height, math.floor(float(frac_rows[finite].max())) + 1 + _MARGIN)
        c0 = max(0, math.floor(float(frac_cols[finite].min())) - _MARGIN)
        c1 = min(info.width, math.floor(float(frac_cols[finite].max())) + 1 + _MARGIN)
        if r0 >= r1 or c0 >= c1:
            return None
        return r0, r1, c0, c1

    def overlaps(self, source: RasterMetadata) -> bool:
        """Whether the label raster covers any of the imagery's pixel centres (sampled)."""
        steps = 64
        u = np.linspace(0.5, max(source.width - 0.5, 0.5), steps)[None, :]
        v = np.linspace(0.5, max(source.height - 0.5, 0.5), steps)[:, None]
        frac_rows, frac_cols = self._fractional(source.transform, u, v)
        rows = self._indices(frac_rows, self.info.height)
        cols = self._indices(frac_cols, self.info.width)
        if bool(((rows >= 0) & (cols >= 0)).any()):
            return True
        # A label raster smaller than the sampling step: test its centre too.
        return self._centre_inside(source)

    def _centre_inside(self, source: RasterMetadata) -> bool:
        la, lb, lc, ld, le, lf = self.transform
        u, v = self.info.width / 2, self.info.height / 2
        x, y = la * u + lb * v + lc, ld * u + le * v + lf
        if not self._same_crs:
            from pyproj import Transformer

            back = Transformer.from_crs(self.crs, source.crs, always_xy=True)
            x, y = back.transform(x, y)
        a, b, c, d, e, f = source.transform
        det = a * e - b * d
        col = (e * (x - c) - b * (y - f)) / det
        row = (a * (y - f) - d * (x - c)) / det
        return bool(0 <= col <= source.width and 0 <= row <= source.height)


class LabelRasterSampler(_GridSampler):
    """Reads a classified label raster at imagery pixel centres as mask values."""

    def __init__(self, labels: RasterLabelsConfig, imagery_crs: str) -> None:
        super().__init__(labels.path, labels.band, imagery_crs)
        info = self.info
        if info.dtype.kind not in "iu":
            raise ValueError(
                f"label raster '{self.name}' holds {info.dtype} values; label rasters must hold "
                "integer class values (uint8, uint16, int16, ...)"
            )
        self.nodata = (
            labels.nodata if labels.nodata is not None else integer_nodata(info.nodata, info.dtype)
        )
        ignore = labels.ignore_index if labels.ignore_index is not None else 0
        self.ignore = ignore
        ignored = tuple(labels.ignore_values) + (() if self.nodata is None else (self.nodata,))
        self._classify = _Classifier(
            info.dtype,
            {value: target.id for value, target in labels.classes.items()},
            ignored,
            ignore if labels.unmapped == "ignore" else 0,
            ignore,
        )
        self._out_dtype = np.dtype(np.uint8)
        self._fill = ignore

    def _decode(self, values: npt.NDArray[Any]) -> npt.NDArray[Any]:
        return self._classify(values)


class ValueRasterSampler(_GridSampler):
    """Reads a continuous raster at imagery pixel centres as float32 targets.

    The target is ``value * labels.scale + labels.offset``, computed in float64. Pixels
    outside the raster, on its NoData, ``NaN`` or outside ``valid_min``..``valid_max``
    (raw values) are ``NaN``.
    """

    def __init__(self, labels: ContinuousLabelsConfig, imagery_crs: str) -> None:
        super().__init__(labels.path, labels.band, imagery_crs)
        nodata = labels.nodata if labels.nodata is not None else self.info.nodata
        self.nodata: Optional[float] = None if nodata is None else float(nodata)
        self._labels = labels
        self._out_dtype = np.dtype(np.float32)
        self._fill = np.float32(np.nan)

    def _decode(self, values: npt.NDArray[Any]) -> npt.NDArray[Any]:
        labels = self._labels
        raw = values.astype(np.float64)
        invalid = ~np.isfinite(raw)
        if self.nodata is not None:
            invalid |= np.isnan(raw) if math.isnan(self.nodata) else raw == self.nodata
        if labels.valid_min is not None:
            invalid |= raw < labels.valid_min
        if labels.valid_max is not None:
            invalid |= raw > labels.valid_max
        target = (raw * labels.scale + labels.offset).astype(np.float32)
        target[invalid] = np.nan
        return target


class RasterSegmentationTarget:
    """Class masks read from a classified label raster, on the imagery's pixel grid."""

    def __init__(self, labels: RasterLabelsConfig) -> None:
        self._labels = labels
        self._class_map: ClassMap = labels.class_map()
        self._sampler: Optional[LabelRasterSampler] = None
        self._fingerprint: Optional[Dict[str, Any]] = None

    @property
    def type(self) -> Optional[str]:
        return "segmentation"

    @property
    def class_map(self) -> ClassMap:
        return self._class_map

    @property
    def sampler(self) -> LabelRasterSampler:
        """The label raster reader; valid after :meth:`prepare`."""
        if self._sampler is None:
            raise RuntimeError("RasterSegmentationTarget.prepare() must run first")
        return self._sampler

    def prepare(self, source: RasterMetadata) -> None:
        sampler = LabelRasterSampler(self._labels, source.crs)
        self._sampler = sampler
        self._fingerprint = geotiff_fingerprint(sampler.location)
        if not sampler.overlaps(source):
            value = (
                f"labels.ignore_index ({self._labels.ignore_index})"
                if self._labels.ignore_index is not None
                else "background"
            )
            # Attribute the warning to the caller of prepare().
            warnings.warn(LABEL_RASTER_MISS_MESSAGE.format(value=value), UserWarning, stacklevel=3)

    def record(self) -> Optional[TargetRecord]:
        """Class map and ignore value, plus the label settings and a fingerprint of the
        label raster (as for GeoTIFF imagery), so a resumed run notices another file."""
        if self._fingerprint is None:
            raise RuntimeError("RasterSegmentationTarget.prepare() must run first")
        settings = self._labels.model_dump(mode="json", exclude={"path", "ignore_index"})
        settings["fingerprint"] = self._fingerprint
        return TargetRecord(
            type="segmentation",
            class_map=self.class_map,
            ignore_index=self._labels.ignore_index,
            dtype="uint8",
            labels=settings,
            options={},
        )

    def window(
        self,
        transform: Transform,
        height: int,
        width: int,
        valid_mask: Optional[npt.NDArray[np.bool_]],
    ) -> WindowTarget:
        mask = self.sampler.sample(transform, height, width)
        return MaskWindow(mask, self._labels.ignore_index)
