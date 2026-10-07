"""Windowed imagery sources: XYZ tiles, EOPF Sentinel-2 Zarr products and GeoTIFF/COG files."""

from __future__ import annotations

import hashlib
import math
import os
import sys
import time
import urllib.request
import warnings
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from functools import lru_cache
from io import BytesIO
from pathlib import Path
from collections import Counter
from typing import Any, Dict, List, Optional, Protocol, Set, Tuple
from urllib.parse import quote, urlsplit

import numpy as np
import numpy.typing as npt
from PIL import Image
from shapely.geometry.base import BaseGeometry
from shapely.ops import transform as shapely_transform

from mapcv._mapcv_rs import (
    TileIndex,
    decode_tile_window,
    fetch_tiles,
    snap_bbox,
    tile_transform,
    tiles,
)
from mapcv.config import (
    EOPFZarrImageryConfig,
    GeoTiffImageryConfig,
    RegionConfig,
    StacCogImageryConfig,
    XYZImageryConfig,
    _LOOPBACK_HOSTS,
    _validate_eopf_path,
    eopf_local_path,
)
from mapcv.downloader import resolve_url_template
from mapcv.geotiff import GeoTiff
from mapcv.tile_cache import TileCache


Transform = Tuple[float, float, float, float, float, float]

# Tiles Rust leaves to Pillow (JPEG, ...) are decoded in these threads; Pillow
# releases the GIL in its decoders. Created on first use and reused across windows.
_PILLOW_THREADS = min(8, os.cpu_count() or 1)
_pillow_pool: Optional[ThreadPoolExecutor] = None


def _pillow_executor() -> ThreadPoolExecutor:
    global _pillow_pool
    if _pillow_pool is None:
        _pillow_pool = ThreadPoolExecutor(
            max_workers=_PILLOW_THREADS, thread_name_prefix="mapcv-pillow"
        )
    return _pillow_pool


@dataclass(frozen=True)
class RasterMetadata:
    """Spatial and channel metadata shared by all imagery sources."""

    source_type: str
    product_id: str
    width: int
    height: int
    bands: List[str]
    dtype: str
    crs: str
    transform: Transform
    chunk_rows: int
    #: Identifies the exact input file(s), for sources whose input can change under the same
    #: name (a GeoTIFF); recorded in the manifest so a resumed run refuses a different file.
    fingerprint: Optional[Dict[str, Any]] = None


class WindowedRasterSource(Protocol):
    """Internal source contract consumed by the generation pipeline."""

    metadata: RasterMetadata

    def read_window(
        self, row_start: int, row_stop: int, col_start: int, col_stop: int
    ) -> Tuple[npt.NDArray[Any], npt.NDArray[np.bool_]]:
        """Read one channels-last pixel window and its validity mask."""

    def close(self) -> None:
        """Release source resources."""


def offset_transform(transform: Transform, row: int, col: int) -> Transform:
    """Translate an affine transform to a pixel-window origin."""
    a, b, c, d, e, f = transform
    return (a, b, c + col * a + row * b, d, e, f + col * d + row * e)


# Tolerances of grid alignment: relative, for pixel sizes; in pixels, for grid origins.
_SCALE_TOLERANCE = 1e-9
_ORIGIN_TOLERANCE_PX = 1e-6


@dataclass(frozen=True)
class GridAlignment:
    """Where another source's pixels sit on a reference source's pixel grid.

    The other source's pixel ``(0, 0)`` starts at reference pixel
    ``(row_offset, col_offset)`` and each of its pixels covers ``factor`` x ``factor``
    reference pixels: reference pixel ``(r, c)`` lies in its pixel
    ``((r - row_offset) // factor, (c - col_offset) // factor)``.
    """

    factor: int
    row_offset: int
    col_offset: int

    @property
    def identity(self) -> bool:
        """Whether both grids are the same (same pixel size and origin)."""
        return self.factor == 1 and self.row_offset == 0 and self.col_offset == 0


def _same_crs(first: str, second: str) -> bool:
    if first == second:
        return True
    from pyproj import CRS
    from pyproj.exceptions import CRSError

    try:
        return bool(CRS.from_user_input(first).equals(CRS.from_user_input(second)))
    except CRSError:
        return False


def grid_alignment(reference: RasterMetadata, other: RasterMetadata, name: str) -> GridAlignment:
    """How source ``name`` (``other``) lines up with the ``reference`` source's grid.

    Both must share a CRS, and ``other``'s pixels must be the reference pixels or a
    whole number of them across, on a grid whose origin falls on a reference pixel
    corner. Then every reference pixel lies in exactly one pixel of ``other``, and
    reading ``other`` on the reference grid (nearest neighbour) is exact.

    Raises:
        ValueError: The grids do not line up that way; the message says how.
    """
    if not _same_crs(reference.crs, other.crs):
        raise ValueError(
            f"imagery '{name}' is in {other.crs}, but the first source is in {reference.crs}; "
            "every source must be in the first source's CRS (mapcv does not reproject imagery)"
        )
    a, b, c, d, e, f = reference.transform
    oa, ob, oc, od, oe, of = other.transform
    size = math.hypot(a, d)
    ratio = math.hypot(oa, od) / size
    factor = round(ratio)
    linear = (a, b, d, e)
    scaled = (oa, ob, od, oe)
    tolerance = _SCALE_TOLERANCE * factor * max(abs(value) for value in linear)
    if factor < 1 or any(abs(o - factor * r) > tolerance for o, r in zip(scaled, linear)):
        if ratio < 1 - _SCALE_TOLERANCE:
            hint = (
                "; list the source with the finest pixels first (the first source's grid is "
                "the dataset's grid, and coarser sources are repeated onto it)"
            )
        else:
            hint = " or a whole number of them, on axes parallel to the first source's"
        raise ValueError(
            f"imagery '{name}' has pixels of {math.hypot(oa, od):g} x {math.hypot(ob, oe):g} "
            f"CRS units, which are not the first source's ({size:g} x {math.hypot(b, e):g}){hint}"
        )
    det = a * e - b * d
    dx, dy = oc - c, of - f
    col = (e * dx - b * dy) / det
    row = (a * dy - d * dx) / det
    col_offset, row_offset = round(col), round(row)
    if abs(col - col_offset) > _ORIGIN_TOLERANCE_PX or abs(row - row_offset) > _ORIGIN_TOLERANCE_PX:
        # Adding 0.0 turns a negative zero into a plain one for the message.
        raise ValueError(
            f"imagery '{name}' is on a grid shifted by ({row - math.floor(row) + 0.0:.4f}, "
            f"{col - math.floor(col) + 0.0:.4f}) pixels from the first source's grid; sources must "
            "share pixel corners (resample one onto the other's grid first)"
        )
    return GridAlignment(factor, row_offset, col_offset)


class AlignedSource:
    """A source read on a reference source's pixel grid (see :class:`GridAlignment`).

    A coarser source's pixels are repeated (nearest neighbour, exact: every reference
    pixel lies inside one of them). Reference pixels outside the source's raster read as
    zero and invalid.
    """

    def __init__(self, source: WindowedRasterSource, alignment: GridAlignment) -> None:
        self.source = source
        self.alignment = alignment
        self.metadata = source.metadata
        self._empty: Optional[Tuple[Tuple[int, ...], "np.dtype[Any]"]] = None

    def read_window(
        self, row_start: int, row_stop: int, col_start: int, col_stop: int
    ) -> Tuple[npt.NDArray[Any], npt.NDArray[np.bool_]]:
        """Read reference pixels ``[row_start, row_stop) x [col_start, col_stop)``."""
        meta = self.metadata
        k, row_offset, col_offset = (
            self.alignment.factor,
            self.alignment.row_offset,
            self.alignment.col_offset,
        )
        if self.alignment.identity and row_stop <= meta.height and col_stop <= meta.width:
            return self.source.read_window(row_start, row_stop, col_start, col_stop)
        rows = (np.arange(row_start, row_stop) - row_offset) // k
        cols = (np.arange(col_start, col_stop) - col_offset) // k
        row_in = (rows >= 0) & (rows < meta.height)
        col_in = (cols >= 0) & (cols < meta.width)
        height, width = len(rows), len(cols)
        if not row_in.any() or not col_in.any():
            channels, dtype = self._empty_like()
            return (
                np.zeros((height, width, *channels), dtype=dtype),
                np.zeros((height, width), dtype=np.bool_),
            )
        # Rows and columns ascend, so the first and last inside ones bound the read.
        first_row, last_row = int(rows[row_in][0]), int(rows[row_in][-1])
        first_col, last_col = int(cols[col_in][0]), int(cols[col_in][-1])
        data, valid = self.source.read_window(first_row, last_row + 1, first_col, last_col + 1)
        row_index = np.clip(rows - first_row, 0, last_row - first_row)
        col_index = np.clip(cols - first_col, 0, last_col - first_col)
        image = data[row_index][:, col_index]
        inside = row_in[:, np.newaxis] & col_in[np.newaxis, :]
        mask = valid[row_index][:, col_index] & inside
        if not inside.all():
            image[~inside] = 0
        return image, mask

    def _empty_like(self) -> Tuple[Tuple[int, ...], "np.dtype[Any]"]:
        """Trailing shape and dtype of this source's windows (read once, from one pixel)."""
        if self._empty is None:
            sample, _ = self.source.read_window(0, 1, 0, 1)
            self._empty = (tuple(sample.shape[2:]), sample.dtype)
        return self._empty

    def close(self) -> None:
        """Close the underlying source."""
        self.source.close()


@lru_cache(maxsize=8)
def _wgs84_transformer(destination_crs: str) -> Any:
    from pyproj import Transformer

    return Transformer.from_crs("EPSG:4326", destination_crs, always_xy=True)


def transform_geometry_to_crs(geometry: BaseGeometry, destination_crs: str) -> BaseGeometry:
    """Transform a WGS-84 geometry into ``destination_crs``."""
    return shapely_transform(_wgs84_transformer(destination_crs).transform, geometry)


def region_bounds_in_crs(region: RegionConfig, crs: str) -> Tuple[float, float, float, float]:
    """``(left, bottom, right, top)`` of the WGS-84 ``region`` in ``crs``.

    The region's edges are densified before projecting, so the box also covers the
    bulge of a curved edge (a lon/lat box is not a rectangle in UTM).
    """
    left, bottom, right, top = _wgs84_transformer(crs).transform_bounds(
        region.west, region.south, region.east, region.north, densify_pts=21
    )
    return float(left), float(bottom), float(right), float(top)


# Tolerance for float noise when snapping to a grid, in grid steps.
_GRID_EPS = 1e-9


def snap_interval_to_grid(
    low: float, high: float, origin: float, step: float, eps: float = _GRID_EPS
) -> Tuple[float, float]:
    """Expand ``[low, high]`` outward to the nearest grid lines ``origin + k * step``.

    ``eps`` (in steps) keeps an edge that sits on a grid line, up to float noise, from
    growing by a whole pixel.
    """
    return (
        origin + math.floor((low - origin) / step + eps) * step,
        origin + math.ceil((high - origin) / step - eps) * step,
    )


def _safe_product_id(path_or_url: str) -> str:
    local = eopf_local_path(path_or_url)
    if local is not None:
        return local.name or "imagery-product"
    parsed = urlsplit(path_or_url)
    name = Path(parsed.path.rstrip("/")).name
    return name or parsed.hostname or "imagery-product"


def _xyz_product_id(url_template: str) -> str:
    # Custom templates can embed secrets in the path or query (e.g. an instance
    # ID), so only the hostname is recorded.
    return f"custom-xyz:{urlsplit(url_template).hostname or 'unknown'}"


class XYZRasterSource:
    """Windowed view over cached XYZ tiles without a full-region allocation."""

    def __init__(self, region: RegionConfig, config: XYZImageryConfig) -> None:
        snapped = snap_bbox(
            region.west,
            region.south,
            region.east,
            region.north,
            config.zoom,
        )
        target_tiles = tiles(
            snapped.west,
            snapped.south,
            snapped.east,
            snapped.north,
            [config.zoom],
        )
        if not target_tiles:
            raise ValueError("XYZ imagery returned no tiles for the requested region")

        self._config = config
        self._template = resolve_url_template(config.url_template, config.source)
        self._zoom = config.zoom
        # Tiles are fetched lazily per window and evicted once windows move
        # past them, so memory stays bounded to about one chunk of tiles and
        # a resumed run only downloads the chunks it still needs.
        self._tiles: Dict[Tuple[int, int], bytes] = {}
        self._attempted: Set[Tuple[int, int]] = set()
        self._cache = TileCache(self._template) if config.cache else None
        self.tiles_requested = 0
        self.tiles_cached = 0
        self.tiles_failed = 0
        self.failure_causes: Counter[str] = Counter()
        self.failure_example: Optional[str] = None
        self._min_x = min(tile.x for tile in target_tiles)
        self._max_x = max(tile.x for tile in target_tiles)
        self._min_y = min(tile.y for tile in target_tiles)
        self._max_y = max(tile.y for tile in target_tiles)

        self.metadata = RasterMetadata(
            source_type="xyz",
            product_id=config.source or _xyz_product_id(self._template),
            width=(self._max_x - self._min_x + 1) * 256,
            height=(self._max_y - self._min_y + 1) * 256,
            bands=["red", "green", "blue"],
            dtype="uint8",
            crs="EPSG:3857",
            transform=tile_transform(self._min_x, self._min_y, config.zoom),
            chunk_rows=config.strip_rows * 256,
        )

    def read_window(
        self, row_start: int, row_stop: int, col_start: int, col_stop: int
    ) -> Tuple[npt.NDArray[np.uint8], npt.NDArray[np.bool_]]:
        height = max(0, row_stop - row_start)
        width = max(0, col_stop - col_start)
        if height == 0 or width == 0:
            return (
                np.zeros((height, width, 3), dtype=np.uint8),
                np.zeros((height, width), dtype=np.bool_),
            )

        tile_col_start = self._min_x + col_start // 256
        tile_col_stop = self._min_x + (col_stop - 1) // 256
        tile_row_start = self._min_y + row_start // 256
        tile_row_stop = self._min_y + (row_stop - 1) // 256
        self._evict_rows_above(tile_row_start)
        self._fetch(
            [
                (tile_x, tile_y)
                for tile_y in range(tile_row_start, tile_row_stop + 1)
                for tile_x in range(tile_col_start, tile_col_stop + 1)
            ]
        )
        payloads = [
            (tile_x, tile_y, payload)
            for tile_y in range(tile_row_start, tile_row_stop + 1)
            for tile_x in range(tile_col_start, tile_col_stop + 1)
            if (payload := self._tiles.get((tile_x, tile_y))) is not None
        ]
        # Rust decodes the tiles it can match to Pillow exactly (PNG, WebP, GIF)
        # on all cores without the GIL and builds the validity mask. Failed tiles
        # are black-filled by the fetcher and some providers serve black NoData,
        # so all-zero pixels count as empty, as in mapcv 0.1.
        window, valid, undecoded = decode_tile_window(
            payloads, self._min_x, self._min_y, row_start, row_stop, col_start, col_stop
        )
        if undecoded:
            self._decode_with_pillow(
                window, valid, undecoded, row_start, row_stop, col_start, col_stop
            )
        return window, valid

    def _decode_with_pillow(
        self,
        window: npt.NDArray[np.uint8],
        valid: npt.NDArray[np.bool_],
        undecoded: List[Tuple[int, int]],
        row_start: int,
        row_stop: int,
        col_start: int,
        col_stop: int,
    ) -> None:
        """Decode the tiles Rust left to Pillow (JPEG, 16-bit PNG, rare formats).

        The tiles are decoded in a thread pool and written into the window in
        order, so the first undecodable tile raises exactly as it did serially.
        """
        if len(undecoded) > 1 and _PILLOW_THREADS > 1:
            decoded = _pillow_executor().map(lambda key: self._decode_tile(*key), undecoded)
        else:
            decoded = (self._decode_tile(*key) for key in undecoded)
        for (tile_x, tile_y), tile_image in zip(undecoded, decoded):
            global_row = (tile_y - self._min_y) * 256
            global_col = (tile_x - self._min_x) * 256
            source_row_start = max(0, row_start - global_row)
            source_col_start = max(0, col_start - global_col)
            source_row_stop = min(256, row_stop - global_row)
            source_col_stop = min(256, col_stop - global_col)
            target_row_start = global_row + source_row_start - row_start
            target_col_start = global_col + source_col_start - col_start
            target_row_stop = target_row_start + source_row_stop - source_row_start
            target_col_stop = target_col_start + source_col_stop - source_col_start
            target = (
                slice(target_row_start, target_row_stop),
                slice(target_col_start, target_col_stop),
            )
            window[target] = tile_image[
                source_row_start:source_row_stop, source_col_start:source_col_stop
            ]
            valid[target] = np.any(window[target] != 0, axis=-1)

    def _decode_tile(self, tile_x: int, tile_y: int) -> npt.NDArray[np.uint8]:
        payload = self._tiles[(tile_x, tile_y)]
        try:
            with Image.open(BytesIO(payload)) as image:
                tile_image = np.asarray(image.convert("RGB"), dtype=np.uint8)
        except Exception as exc:
            raise RuntimeError(f"Unable to decode XYZ tile {tile_x}/{tile_y}") from exc
        if tile_image.shape != (256, 256, 3):
            raise ValueError("XYZ tile sources must return 256x256 RGB-compatible images")
        return tile_image

    def _evict_rows_above(self, tile_row: int) -> None:
        for key in [key for key in self._tiles if key[1] < tile_row]:
            del self._tiles[key]

    def _fetch(self, keys: List[Tuple[int, int]]) -> None:
        missing = [key for key in keys if key not in self._attempted]
        if not missing:
            return
        self._attempted.update(missing)
        cache = self._cache
        if cache is not None:
            for x, y in missing:
                cached = cache.get(x, y, self._zoom)
                if cached is not None:
                    self._tiles[(x, y)] = cached
            self.tiles_cached += sum(1 for key in missing if key in self._tiles)
            missing = [key for key in missing if key not in self._tiles]
            if not missing:
                return
        config = self._config
        results, failed, (causes, example) = fetch_tiles(
            [TileIndex(x, y, self._zoom) for x, y in missing],
            self._template,
            max_connections=config.max_connections,
            policy=config.policy,
            max_failed_ratio=config.max_failed_ratio,
            cache_headers=True,
        )
        self.tiles_requested += len(missing)
        self.tiles_failed += failed
        self.failure_causes.update(dict(causes))
        if example and self.failure_example is None:
            self.failure_example = example
        for tile, payload, headers in results:
            self._tiles[(tile.x, tile.y)] = payload
            # Black fills of failed tiles have no headers and are never cached.
            if cache is not None and headers is not None:
                cache.put(tile.x, tile.y, tile.z, payload, headers)

    @property
    def failure_reasons(self) -> str:
        """Failed tiles so far by cause, most common first, with one example."""
        if not self.failure_causes:
            return ""
        counts = ", ".join(f"{n} x {cause}" for cause, n in self.failure_causes.most_common())
        return f"{counts} (e.g. {self.failure_example})" if self.failure_example else counts

    def close(self) -> None:
        """Drop cached tiles."""
        self._tiles.clear()


def _dataset_crs(dataset: Any) -> str:
    candidates: List[Any] = []
    for key in ("crs", "crs_wkt", "spatial_ref"):
        candidates.append(dataset.attrs.get(key))

    for coordinate in dataset.coords.values():
        for key in ("crs", "crs_wkt", "spatial_ref"):
            candidates.append(coordinate.attrs.get(key))

    try:
        rio_crs = dataset.rio.crs
    except (AttributeError, RuntimeError):
        rio_crs = None
    candidates.append(rio_crs)

    try:
        from pyproj import CRS
    except ImportError as exc:  # pragma: no cover - optional dependency guard
        raise RuntimeError("Install EOPF support with 'pip install mapcv[zarr]'.") from exc

    for candidate in candidates:
        if candidate:
            try:
                return str(CRS.from_user_input(candidate).to_string())
            except Exception:
                continue
    raise ValueError("EOPF dataset does not expose a readable projected CRS")


def _coordinate_transform(dataset: Any, resolution: int) -> Transform:
    x_values = np.asarray(dataset.coords["x"].values, dtype=np.float64)
    y_values = np.asarray(dataset.coords["y"].values, dtype=np.float64)
    if x_values.size == 0 or y_values.size == 0:
        raise ValueError("requested region does not intersect the EOPF product")
    x_step = float(x_values[1] - x_values[0]) if x_values.size > 1 else float(resolution)
    y_step = float(y_values[1] - y_values[0]) if y_values.size > 1 else -float(resolution)
    return (
        x_step,
        0.0,
        float(x_values[0] - x_step / 2.0),
        0.0,
        y_step,
        float(y_values[0] - y_step / 2.0),
    )


def _snap_bounds_to_grid(
    bounds: Tuple[float, float, float, float],
    x_values: npt.NDArray[Any],
    y_values: npt.NDArray[Any],
    resolution: int,
) -> Tuple[float, float, float, float]:
    """Expand ``bounds`` outward to pixel edges of the product grid.

    The reader builds its output grid from the bbox origin, so an unsnapped bbox
    is offset by a sub-pixel amount and every band would be resampled.
    """
    x_res = abs(float(x_values[1] - x_values[0])) if x_values.size > 1 else float(resolution)
    y_res = abs(float(y_values[1] - y_values[0])) if y_values.size > 1 else float(resolution)
    x_edge = float(np.min(x_values)) - x_res / 2.0
    y_edge = float(np.min(y_values)) - y_res / 2.0
    left, bottom, right, top = bounds
    snapped_left, snapped_right = snap_interval_to_grid(left, right, x_edge, x_res)
    snapped_bottom, snapped_top = snap_interval_to_grid(bottom, top, y_edge, y_res)
    return snapped_left, snapped_bottom, snapped_right, snapped_top


class BandGapError(RuntimeError):
    """One band came back empty while the others had data (a failed read)."""


# Remote EOPF reads (object store over HTTPS) time out now and then; a window is
# read up to this many times, waiting the listed seconds between attempts.
EOPF_READ_ATTEMPTS = 4
EOPF_RETRY_DELAYS = (2.0, 5.0, 10.0)
# Errors that mean the request or the configuration is wrong; retrying cannot help.
_NOT_RETRYABLE = (ValueError, TypeError, KeyError, IndexError, NotImplementedError)


def _check_band_coverage(
    finite: npt.NDArray[np.bool_], bands: List[str], row_start: int, row_stop: int
) -> None:
    """Fail when a band is empty where other bands have data.

    Remote Zarr reads that time out can come back as all-NaN chunks for one
    band while the others are fine; writing such patches would silently corrupt
    the dataset. Real NoData (outside the swath) is empty in every band at once.
    """
    has_any = np.any(finite, axis=-1)
    if not has_any.any():
        return
    for index, band in enumerate(bands):
        if not np.any(finite[..., index] & has_any):
            raise BandGapError(
                f"band {band} returned no data for rows {row_start}-{row_stop} while other "
                "bands did; the read probably failed (e.g. a network timeout). Run the same "
                "command again to resume from this chunk."
            )


class EOPFZarrRasterSource:
    """Lazy window reader for one Sentinel-2 L2A EOPF Zarr product."""

    def __init__(self, region: RegionConfig, config: EOPFZarrImageryConfig) -> None:
        fingerprint: Dict[str, Any] = {}
        if config.search is not None:
            from mapcv.stac import find_product

            match = find_product(
                config.search, (region.west, region.south, region.east, region.north)
            )
            try:
                path = _validate_eopf_path(match.href)
            except ValueError as exc:
                raise ValueError(f"STAC item {match.item_id}: {exc}") from None
            catalog_host = urlsplit(config.search.catalog).hostname or ""
            if eopf_local_path(path) is not None and catalog_host not in _LOOPBACK_HOSTS:
                # A catalog is remote input: it may name remote products only, never a
                # file on this machine (outside, say, the MCP server's root).
                raise ValueError(
                    f"STAC item {match.item_id} points to a local file ({match.href}); a "
                    "catalog may only point to https:// or s3:// products"
                )
            fingerprint["stac"] = {
                "catalog": config.search.catalog,
                "collection": config.search.collection,
                "item": match.item_id,
            }
        else:
            assert config.path is not None  # EOPFZarrImageryConfig: path or search
            path = config.path
        if config.scl_mask is not None:
            fingerprint["scl_mask"] = list(config.scl_mask)
        # URL safety rules are enforced by EOPFZarrImageryConfig validation (and above
        # for a found product).
        local_path = eopf_local_path(path)
        if local_path is not None and not local_path.exists():
            raise FileNotFoundError(f"EOPF Zarr product not found: {local_path}")

        try:
            import xarray as xr
        except ImportError as exc:
            if sys.version_info >= (3, 14):
                raise RuntimeError(
                    "Sentinel-2 (EOPF Zarr) support needs Python 3.10-3.13: its zarr dependency "
                    "has no Python 3.14 wheels, so the mapcv[zarr] extra installs nothing "
                    "on 3.14. Use a Python 3.13 environment for Sentinel-2."
                ) from exc
            raise RuntimeError(
                "EOPF Zarr support is optional; install it with 'pip install mapcv[zarr]'."
            ) from exc

        storage_options = {"anon": True} if urlsplit(path).scheme == "s3" else None
        variables = list(config.bands) + (["scl"] if config.scl_mask is not None else [])

        def open_dataset(**spatial_options: Any) -> Any:
            return xr.open_dataset(
                path,
                engine="eopf-zarr",
                op_mode="analysis",
                variables=variables,
                resolution=config.resolution,
                chunks={},
                storage_options=storage_options,
                **spatial_options,
            )

        try:
            discovery = open_dataset()
        except Exception as exc:
            raise RuntimeError(
                f"Unable to open anonymous EOPF product '{_safe_product_id(path)}': {exc}"
            ) from exc

        missing = [band for band in variables if band not in discovery.data_vars]
        if missing:
            available = ", ".join(sorted(str(name) for name in discovery.data_vars))
            discovery.close()
            raise ValueError(
                f"EOPF variables not found: {', '.join(missing)}. Available variables: {available}"
            )
        if "x" not in discovery.coords or "y" not in discovery.coords:
            discovery.close()
            raise ValueError(
                "EOPF analysis dataset must expose one-dimensional x and y coordinates"
            )

        try:
            crs = _dataset_crs(discovery)
        except Exception:
            discovery.close()
            raise
        left, bottom, right, top = region_bounds_in_crs(region, crs)
        x_values = np.asarray(discovery.coords["x"].values)
        y_values = np.asarray(discovery.coords["y"].values)
        if x_values.size == 0 or y_values.size == 0:
            discovery.close()
            raise ValueError("EOPF product exposes an empty spatial grid")
        product_left, product_right = float(np.min(x_values)), float(np.max(x_values))
        product_bottom, product_top = float(np.min(y_values)), float(np.max(y_values))
        intersects = not (
            right < product_left
            or left > product_right
            or top < product_bottom
            or bottom > product_top
        )
        if intersects:
            left, bottom, right, top = _snap_bounds_to_grid(
                (left, bottom, right, top), x_values, y_values, config.resolution
            )
        discovery.close()
        if not intersects:
            raise ValueError("requested region does not intersect the EOPF product")

        try:
            dataset = open_dataset(bbox=[left, bottom, right, top], crs=crs)
        except Exception as exc:
            raise RuntimeError(
                f"Unable to crop anonymous EOPF product '{_safe_product_id(path)}': {exc}"
            ) from exc

        height = int(dataset.sizes.get("y", 0))
        width = int(dataset.sizes.get("x", 0))
        if height == 0 or width == 0:
            dataset.close()
            raise ValueError("requested region does not intersect the EOPF product")

        self._dataset = dataset
        self._bands = list(config.bands)
        self._scl_mask = list(config.scl_mask) if config.scl_mask is not None else None
        self.metadata = RasterMetadata(
            source_type="eopf_zarr",
            product_id=_safe_product_id(path),
            width=width,
            height=height,
            bands=list(config.bands),
            dtype="float32",
            crs=crs,
            transform=_coordinate_transform(dataset, config.resolution),
            chunk_rows=config.chunk_rows,
            fingerprint=fingerprint or None,
        )

    def read_window(
        self, row_start: int, row_stop: int, col_start: int, col_stop: int
    ) -> Tuple[npt.NDArray[np.float32], npt.NDArray[np.bool_]]:
        """Read a window, retrying transient failures (timeouts, empty bands)."""
        for attempt in range(1, EOPF_READ_ATTEMPTS + 1):
            try:
                return self._read_window_once(row_start, row_stop, col_start, col_stop)
            except _NOT_RETRYABLE:
                raise
            except Exception as exc:  # noqa: BLE001 - network stacks raise many types
                if attempt == EOPF_READ_ATTEMPTS:
                    raise RuntimeError(
                        f"reading rows {row_start}-{row_stop} failed {attempt} times; last "
                        f"error: {exc}"
                    ) from exc
                time.sleep(EOPF_RETRY_DELAYS[min(attempt, len(EOPF_RETRY_DELAYS)) - 1])
        raise AssertionError("unreachable")  # pragma: no cover

    def _read_window_once(
        self, row_start: int, row_stop: int, col_start: int, col_stop: int
    ) -> Tuple[npt.NDArray[np.float32], npt.NDArray[np.bool_]]:
        window = self._dataset[self._bands].isel(
            y=slice(row_start, row_stop), x=slice(col_start, col_stop)
        )
        array_data = window.to_array(dim="band").transpose("y", "x", "band").data
        if hasattr(array_data, "compute"):
            array_data = array_data.compute()
        image = np.asarray(array_data, dtype=np.float32)
        if image.ndim != 3 or image.shape[-1] != len(self._bands):
            raise ValueError("selected EOPF variables must resolve to two-dimensional y/x rasters")
        finite = np.isfinite(image)
        _check_band_coverage(finite, self._bands, row_start, row_stop)
        # A pixel is only usable when every requested band has data.
        valid = np.all(finite, axis=-1)
        if self._scl_mask is not None:
            scl = np.asarray(
                self._dataset["scl"].isel(
                    y=slice(row_start, row_stop), x=slice(col_start, col_stop)
                )
            )
            # Masked classes, and anything outside 0-11 (the fill at product edges), have
            # no usable imagery.
            masked = np.isin(scl, self._scl_mask) | (scl > 11)
            valid &= ~masked
            image[masked] = np.nan
        return image, valid

    def close(self) -> None:
        """Close the underlying xarray dataset."""
        self._dataset.close()


# ── GeoTIFF / COG ────────────────────────────────────────────────────────────

# The head (and, for local files, tail) of a file that is hashed for its fingerprint.
_FINGERPRINT_BYTES = 64 * 1024
_FINGERPRINT_TIMEOUT_S = 15.0
# Tolerance, in pixels, for a region edge that sits on the file's border.
_EDGE_EPS_PX = 1e-6


def geotiff_location(path: str) -> str:
    """What the reader opens for ``imagery.path``: a path, or the URL itself."""
    local = eopf_local_path(path)
    return str(local) if local is not None else path


def _remote_http_url(url: str) -> str:
    """The ``http(s)`` URL behind ``url`` (``s3://bucket/key`` maps as in the Rust reader)."""
    parsed = urlsplit(url)
    if parsed.scheme != "s3":
        return url
    bucket, key = parsed.netloc, parsed.path.lstrip("/")
    host = (
        f"https://s3.amazonaws.com/{bucket}/"
        if "." in bucket
        else f"https://{bucket}.s3.amazonaws.com/"
    )
    return host + quote(key, safe="/%")


def geotiff_fingerprint(location: str) -> Dict[str, Any]:
    """A cheap identity of a GeoTIFF, so a resumed run notices a different file.

    Local files: size and the SHA-256 of the first and last 64 KiB (where the TIFF headers
    and, for non-COG files, the directory live). The modification time is left out on
    purpose: it changes when a file is copied, touched or rsynced, which would refuse a
    resume of an unchanged file.
    URLs: the URL (credentials are rejected up front, so it is safe to record) plus the
    ``ETag``, total size and SHA-256 of the first 64 KiB, taken from one ranged request.
    Nothing reads the whole file, whatever its size.
    """
    if "://" not in location:
        path = Path(location)
        stat = path.stat()
        digest = hashlib.sha256()
        with path.open("rb") as handle:
            digest.update(handle.read(_FINGERPRINT_BYTES))
            if stat.st_size > _FINGERPRINT_BYTES:
                handle.seek(max(_FINGERPRINT_BYTES, stat.st_size - _FINGERPRINT_BYTES))
                digest.update(handle.read(_FINGERPRINT_BYTES))
        return {
            "kind": "file",
            "size": stat.st_size,
            "sha256_head_tail": digest.hexdigest(),
        }
    fingerprint: Dict[str, Any] = {"kind": "url", "url": location}
    request = urllib.request.Request(
        _remote_http_url(location),
        headers={"Range": f"bytes=0-{_FINGERPRINT_BYTES - 1}", "User-Agent": "mapcv"},
    )
    try:
        with urllib.request.urlopen(request, timeout=_FINGERPRINT_TIMEOUT_S) as response:  # noqa: S310
            head = response.read(_FINGERPRINT_BYTES)
            etag = response.headers.get("ETag")
            content_range = response.headers.get("Content-Range", "")
            total = content_range.rpartition("/")[2]
            fingerprint["size"] = int(total) if total.isdigit() else None
            fingerprint["etag"] = etag.strip() if etag else None
            fingerprint["sha256_head"] = hashlib.sha256(head).hexdigest()
    except OSError:
        # The reader reports unreachable files itself; a missing fingerprint only
        # weakens the resume check to the URL.
        pass
    return fingerprint


def _nodata_mask(data: npt.NDArray[Any], nodata: Optional[float]) -> npt.NDArray[np.bool_]:
    """Pixels where every band is NoData (or non-finite, for float rasters)."""
    empty = _all_equal(data, nodata)
    if np.issubdtype(data.dtype, np.floating):
        empty |= np.all(~np.isfinite(data), axis=-1)
    return empty


def _all_equal(data: npt.NDArray[Any], nodata: Optional[float]) -> npt.NDArray[np.bool_]:
    if nodata is None or np.isnan(nodata):
        return np.zeros(data.shape[:2], dtype=np.bool_)
    if np.issubdtype(data.dtype, np.integer):
        info = np.iinfo(data.dtype)
        if nodata != int(nodata) or not info.min <= nodata <= info.max:
            return np.zeros(data.shape[:2], dtype=np.bool_)
    return np.asarray(np.all(data == data.dtype.type(nodata), axis=-1), dtype=np.bool_)


def _json_nodata(nodata: Optional[float]) -> Optional[Any]:
    """``nodata`` as a JSON-safe, self-equal value (``NaN != NaN`` would break resuming)."""
    if nodata is None:
        return None
    return "nan" if np.isnan(nodata) else nodata


def region_pixel_window(
    bounds: Tuple[float, float, float, float], transform: Transform, height: int, width: int
) -> Tuple[int, int, int, int, bool]:
    """Pixel window ``(row0, row1, col0, col1)`` covering ``bounds``, clipped to the raster.

    The CRS box is mapped through the inverse of the (possibly rotated) pixel transform
    and its pixel extent is snapped outward to whole pixels. The last value says whether
    the box reaches beyond the raster.
    """
    left, bottom, right, top = bounds
    a, b, c, d, e, f = transform
    det = a * e - b * d
    if det == 0:
        raise ValueError("the GeoTIFF's pixel transform is degenerate")
    cols: List[float] = []
    rows: List[float] = []
    for x, y in ((left, bottom), (left, top), (right, bottom), (right, top)):
        cols.append((e * (x - c) - b * (y - f)) / det)
        rows.append((a * (y - f) - d * (x - c)) / det)
    low_col, high_col = min(cols), max(cols)
    low_row, high_row = min(rows), max(rows)
    extends = (
        low_col < -_EDGE_EPS_PX
        or low_row < -_EDGE_EPS_PX
        or high_col > width + _EDGE_EPS_PX
        or high_row > height + _EDGE_EPS_PX
    )
    snapped_col0, snapped_col1 = snap_interval_to_grid(low_col, high_col, 0.0, 1.0)
    snapped_row0, snapped_row1 = snap_interval_to_grid(low_row, high_row, 0.0, 1.0)
    return (
        max(0, int(snapped_row0)),
        min(height, int(snapped_row1)),
        max(0, int(snapped_col0)),
        min(width, int(snapped_col1)),
        extends,
    )


def _extent_text(transform: Transform, height: int, width: int) -> str:
    a, b, c, d, e, f = transform
    xs = [c + a * col + b * row for col in (0, width) for row in (0, height)]
    ys = [f + d * col + e * row for col in (0, width) for row in (0, height)]
    return f"{min(xs):.2f}, {min(ys):.2f} to {max(xs):.2f}, {max(ys):.2f}"


class GeoTiffRasterSource:
    """Windowed reader over one GeoTIFF or COG, on the file's own pixel grid.

    The region (WGS-84) is projected into the file's CRS and snapped outward to whole
    pixels, so pixels are read exactly as stored: no warping, no resampling. The window
    is clipped to the raster. Patches, masks and transforms are expressed in the file's
    CRS and grid, so labels are rasterized onto the same pixels.
    """

    def __init__(
        self,
        region: RegionConfig,
        config: GeoTiffImageryConfig,
        *,
        image_format: Optional[str] = None,
    ) -> None:
        # URL safety rules are enforced by GeoTiffImageryConfig validation.
        location = geotiff_location(config.path)
        self._tif = GeoTiff(location)
        info = self._tif.info
        name = _safe_product_id(config.path)

        if info.epsg is None:
            raise ValueError(
                f"GeoTIFF '{name}' has no usable CRS: {info.crs_error or 'no CRS in the file'}. "
                "mapcv reads files whose CRS is an EPSG code; re-project or re-tag the file."
            )
        if info.transform is None:
            raise ValueError(f"GeoTIFF '{name}' has no georeferencing (no pixel size/origin tags)")
        if config.overview > len(info.overviews):
            raise ValueError(
                f"imagery.overview is {config.overview}, but '{name}' has "
                f"{len(info.overviews)} overview level(s) (0 is the full resolution)"
            )
        file_transform = info.overview_transform(config.overview)
        assert file_transform is not None
        level_height, level_width = (
            (info.height, info.width)
            if config.overview == 0
            else info.overviews[config.overview - 1]
        )

        selected = (
            list(config.bands) if config.bands is not None else list(range(1, info.count + 1))
        )
        if max(selected) > info.count:
            raise ValueError(
                f"imagery.bands asks for band {max(selected)}, but '{name}' has {info.count} band(s)"
            )
        self._bands = [band - 1 for band in selected]
        self._expand_gray = False
        if image_format is not None and image_format not in ("npy", "tif"):
            if info.dtype != np.uint8 or len(selected) not in (1, 3):
                raise ValueError(
                    f"'{name}' has {len(selected)} selected band(s) of {info.dtype}; "
                    f"writer.image_format '{image_format}' writes 1 or 3 bands of uint8. "
                    "Use writer.image_format: npy or tif (keep band count and dtype), or select "
                    "1 or 3 bands of a uint8 file with imagery.bands"
                )
            self._expand_gray = len(selected) == 1
        names = [f"b{band}" for band in selected]
        if self._expand_gray:
            names = names * 3

        crs = f"EPSG:{info.epsg}"
        left, bottom, right, top = region_bounds_in_crs(region, crs)
        row0, row1, col0, col1, extends = region_pixel_window(
            (left, bottom, right, top), file_transform, level_height, level_width
        )
        if row0 >= row1 or col0 >= col1:
            raise ValueError(
                f"requested region does not intersect the GeoTIFF '{name}' "
                f"(file extent in {crs}: {_extent_text(file_transform, level_height, level_width)})"
            )
        if extends:
            warnings.warn(
                f"the region extends beyond the GeoTIFF '{name}'; only the part inside the file "
                "is used (patches at its edge are padded or dropped by sampler.edge_strategy)",
                UserWarning,
                stacklevel=3,
            )

        self._overview = config.overview
        self._row0, self._col0 = row0, col0
        self._nodata = config.nodata if config.nodata is not None else info.nodata
        effective_nodata = _json_nodata(self._nodata)
        self._dtype = info.dtype
        self.metadata = RasterMetadata(
            source_type="geotiff",
            product_id=name,
            width=col1 - col0,
            height=row1 - row0,
            bands=names,
            dtype=str(info.dtype),
            crs=crs,
            transform=offset_transform(file_transform, row0, col0),
            chunk_rows=config.chunk_rows,
            fingerprint={
                **geotiff_fingerprint(location),
                "overview": config.overview,
                "bands": selected,
                "nodata": effective_nodata,
            },
        )

    def read_window(
        self, row_start: int, row_stop: int, col_start: int, col_stop: int
    ) -> Tuple[npt.NDArray[Any], npt.NDArray[np.bool_]]:
        data, inside = self._tif.read_window(
            self._row0 + row_start,
            self._row0 + row_stop,
            self._col0 + col_start,
            self._col0 + col_stop,
            bands=self._bands,
            overview=self._overview,
        )
        valid = inside & ~_nodata_mask(data, self._nodata)
        if self._expand_gray:
            data = np.repeat(data, 3, axis=-1)
        return data, valid

    def close(self) -> None:
        """Nothing to release: the reader holds no open handles between reads."""


class StacCogRasterSource:
    """Sentinel-2 bands as separate COGs of a STAC item, stacked on the finest band's grid.

    The item is found with :func:`mapcv.stac.find_item`. Each band asset is read with
    :class:`GeoTiffRasterSource` (mapcv's own reader, range requests over https); bands
    of coarser resolution, and the scene classification for ``scl_mask``, are placed on
    the finest band's grid by :class:`AlignedSource` (pixels repeated, exact). A pixel
    has imagery when every band has it and, with ``scl_mask``, its scene class is not
    masked (values outside 0-11 count as no data).
    """

    def __init__(self, region: RegionConfig, config: StacCogImageryConfig) -> None:
        from mapcv.stac import find_item

        bbox = (region.west, region.south, region.east, region.north)
        item, _ = find_item(config.search, bbox)
        item_id = str(item.get("id", ""))
        assets = item.get("assets", {})
        loopback = (urlsplit(config.search.catalog).hostname or "") in _LOOPBACK_HOSTS
        wanted = list(config.bands) + ([config.scl_asset] if config.scl_mask is not None else [])

        def href(key: str) -> str:
            asset = assets.get(key)
            if not asset or not asset.get("href"):
                available = ", ".join(sorted(assets))
                raise ValueError(f"STAC item {item_id} has no '{key}' asset; it has: {available}")
            location = str(asset["href"])
            if eopf_local_path(location) is not None and not loopback:
                raise ValueError(
                    f"STAC item {item_id} points to a local file ({location}); a catalog may "
                    "only point to https:// or s3:// files"
                )
            return location

        locations = {key: href(key) for key in wanted}
        opened: Dict[str, WindowedRasterSource] = {}
        try:
            for key, location in locations.items():
                opened[key] = GeoTiffRasterSource(
                    region,
                    GeoTiffImageryConfig(
                        type="geotiff", path=location, bands=[1], chunk_rows=config.chunk_rows
                    ),
                    image_format="npy",
                )
            # The finest band's grid is the source's grid.
            reference_key = min(
                config.bands, key=lambda key: abs(opened[key].metadata.transform[0])
            )
            reference = opened[reference_key]
            dtypes = {opened[key].metadata.dtype for key in config.bands}
            if len(dtypes) != 1:
                raise ValueError(
                    f"imagery.bands of STAC item {item_id} have different data types "
                    f"({', '.join(sorted(str(d) for d in dtypes))}); select bands of one type"
                )
            self._aligned = {
                key: AlignedSource(source, grid_alignment(reference.metadata, source.metadata, key))
                for key, source in opened.items()
            }
        except BaseException:
            for source in opened.values():
                source.close()
            raise
        self._opened = opened
        self._bands = list(config.bands)
        self._scl = (
            (self._aligned[config.scl_asset], list(config.scl_mask))
            if config.scl_mask is not None
            else None
        )
        fingerprint: Dict[str, Any] = {
            "stac": {
                "catalog": config.search.catalog,
                "collection": config.search.collection,
                "item": item_id,
            }
        }
        if config.scl_mask is not None:
            fingerprint["scl_mask"] = list(config.scl_mask)
        meta = reference.metadata
        self.metadata = RasterMetadata(
            source_type="stac_cog",
            product_id=item_id,
            width=meta.width,
            height=meta.height,
            bands=list(config.bands),
            dtype=meta.dtype,
            crs=meta.crs,
            transform=meta.transform,
            chunk_rows=config.chunk_rows,
            fingerprint=fingerprint,
        )

    def read_window(
        self, row_start: int, row_stop: int, col_start: int, col_stop: int
    ) -> Tuple[npt.NDArray[Any], npt.NDArray[np.bool_]]:
        layers = []
        valid: Optional[npt.NDArray[np.bool_]] = None
        for key in self._bands:
            data, band_valid = self._aligned[key].read_window(
                row_start, row_stop, col_start, col_stop
            )
            layers.append(data[:, :, 0])
            valid = band_valid if valid is None else valid & band_valid
        assert valid is not None
        if self._scl is not None:
            scl_source, masked_classes = self._scl
            classes, scl_valid = scl_source.read_window(row_start, row_stop, col_start, col_stop)
            classes = classes[:, :, 0]
            valid = valid & scl_valid & ~(np.isin(classes, masked_classes) | (classes > 11))
        image = np.stack(layers, axis=-1)
        return image, valid

    def close(self) -> None:
        for source in self._opened.values():
            source.close()


def open_raster_source(
    region: RegionConfig,
    imagery: XYZImageryConfig | EOPFZarrImageryConfig | GeoTiffImageryConfig | StacCogImageryConfig,
    *,
    image_format: Optional[str] = None,
) -> WindowedRasterSource:
    """Construct the configured raster source.

    ``image_format`` (``writer.image_format``) lets a GeoTIFF source check that its
    bands and dtype fit the output; other sources ignore it.
    """
    if isinstance(imagery, XYZImageryConfig):
        return XYZRasterSource(region, imagery)
    if isinstance(imagery, GeoTiffImageryConfig):
        return GeoTiffRasterSource(region, imagery, image_format=image_format)
    if isinstance(imagery, StacCogImageryConfig):
        return StacCogRasterSource(region, imagery)
    return EOPFZarrRasterSource(region, imagery)
