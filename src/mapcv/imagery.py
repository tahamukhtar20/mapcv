"""Windowed imagery sources: XYZ tiles, EOPF Sentinel-2 Zarr products and GeoTIFF/COG files."""

from __future__ import annotations

import hashlib
import logging
import math
import os
import sys
import threading
import time
import urllib.request
import warnings
from collections import Counter
from collections.abc import Iterator
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from dataclasses import dataclass
from functools import lru_cache
from io import BytesIO
from pathlib import Path
from typing import Any, Protocol
from urllib.parse import quote, urlsplit

import numpy as np
import numpy.typing as npt
import shapely
from PIL import Image
from shapely.geometry.base import BaseGeometry

from mapcv._mapcv_rs import (
    TileIndex,
    decode_tile_window,
    fetch_tiles,
    snap_bbox,
    tile_transform,
    tiles,
)
from mapcv._net import fsspec_options, linked_url_refusal
from mapcv._net import urlopen as _urlopen
from mapcv.config import (
    _LOOPBACK_HOSTS,
    EOPFZarrImageryConfig,
    GeoTiffImageryConfig,
    RegionConfig,
    StacCogImageryConfig,
    XYZImageryConfig,
    _validate_eopf_path,
    eopf_local_path,
)
from mapcv.downloader import resolve_url_template
from mapcv.geotiff import GeoTiff
from mapcv.tile_cache import CacheHeaders, TileCache

Transform = tuple[float, float, float, float, float, float]

# Tiles Rust leaves to Pillow (JPEG, ...) are decoded in these threads; Pillow
# releases the GIL in its decoders. Created on first use and reused across windows.
_PILLOW_THREADS = min(8, os.cpu_count() or 1)
_pillow_pool: ThreadPoolExecutor | None = None


def _pillow_executor() -> ThreadPoolExecutor:
    global _pillow_pool
    if _pillow_pool is None:
        _pillow_pool = ThreadPoolExecutor(
            max_workers=_PILLOW_THREADS, thread_name_prefix="mapcv-pillow"
        )
    return _pillow_pool


_log = logging.getLogger(__name__)


@dataclass(frozen=True)
class RasterMetadata:
    """Spatial and channel metadata shared by all imagery sources."""

    source_type: str
    product_id: str
    width: int
    height: int
    bands: list[str]
    dtype: str
    crs: str
    transform: Transform
    chunk_rows: int
    #: Identifies the exact input file(s), for sources whose input can change under the same
    #: name (a GeoTIFF); recorded in the manifest so a resumed run refuses a different file.
    fingerprint: dict[str, Any] | None = None


class WindowedRasterSource(Protocol):
    """Internal source contract consumed by the generation pipeline."""

    metadata: RasterMetadata

    def read_window(
        self, row_start: int, row_stop: int, col_start: int, col_stop: int
    ) -> tuple[npt.NDArray[Any], npt.NDArray[np.bool_]]:
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
        self._empty: tuple[tuple[int, ...], np.dtype[Any]] | None = None

    def read_window(
        self, row_start: int, row_stop: int, col_start: int, col_stop: int
    ) -> tuple[npt.NDArray[Any], npt.NDArray[np.bool_]]:
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

    def _empty_like(self) -> tuple[tuple[int, ...], np.dtype[Any]]:
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
    """Transform a WGS-84 geometry into ``destination_crs`` (Z is dropped)."""
    transformer = _wgs84_transformer(destination_crs)

    def apply(coords: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
        x, y = transformer.transform(coords[:, 0], coords[:, 1])
        return np.column_stack((x, y))

    result: BaseGeometry = shapely.transform(geometry, apply)
    return result


def region_bounds_in_crs(region: RegionConfig, crs: str) -> tuple[float, float, float, float]:
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
) -> tuple[float, float]:
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


def _tile_range_bounds(
    min_x: int, min_y: int, max_x: int, max_y: int, zoom: int
) -> tuple[float, float, float, float]:
    """(west, south, east, north) in degrees of the XYZ tiles ``min_x..max_x`` by
    ``min_y..max_y``: their outer edges, a meridian or parallel each."""
    count = 2.0**zoom

    def lon(x: int) -> float:
        return x / count * 360.0 - 180.0

    def lat(y: int) -> float:
        return math.degrees(math.atan(math.sinh(math.pi * (1.0 - 2.0 * y / count))))

    return lon(min_x), lat(max_y + 1), lon(max_x + 1), lat(min_y)


_ALPHA_MODES = frozenset({"RGBA", "LA", "PA", "RGBa", "La"})


def _rgb_array(image: Image.Image) -> npt.NDArray[np.uint8]:
    """The RGB pixels of a tile image.

    16-bit grayscale is scaled to 8 bits by its high byte, as 16-bit RGB already is (a
    plain ``convert("RGB")`` would clip every value above 255 to white). Fully
    transparent pixels come out black: they have no imagery, and all-black pixels count
    as empty.
    """
    if image.mode.startswith("I"):  # I;16, I;16L, I;16B, I (32-bit)
        gray = np.clip(np.asarray(image).astype(np.int64) >> 8, 0, 255).astype(np.uint8)
        return np.repeat(gray[..., np.newaxis], 3, axis=-1)
    if image.mode in _ALPHA_MODES or "transparency" in image.info:
        rgba = np.asarray(image.convert("RGBA"), dtype=np.uint8)
        rgb = np.ascontiguousarray(rgba[..., :3])
        rgb[rgba[..., 3] == 0] = 0
        return rgb
    return np.asarray(image.convert("RGB"), dtype=np.uint8)


def _xyz_product_id(url_template: str) -> str:
    # Custom templates can embed secrets in the path or query (e.g. an instance
    # ID), so only the hostname is recorded.
    return f"custom-xyz:{urlsplit(url_template).hostname or 'unknown'}"


# A slow, salted hash of a custom URL template: a resumed run notices another layer, year
# or style on the same host, while a key in the template cannot be read back from the
# manifest (and guessing a weak one costs this many SHA-256 rounds per guess).
_TEMPLATE_HASH_SALT = b"mapcv url_template"
_TEMPLATE_HASH_ROUNDS = 100_000


def url_template_digest(url_template: str) -> str:
    """The hex PBKDF2-SHA256 digest of a URL template, recorded instead of the template."""
    digest = hashlib.pbkdf2_hmac(
        "sha256", url_template.encode("utf-8"), _TEMPLATE_HASH_SALT, _TEMPLATE_HASH_ROUNDS
    )
    return digest.hex()


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
        self._min_x = min(tile.x for tile in target_tiles)
        self._max_x = max(tile.x for tile in target_tiles)
        self._min_y = min(tile.y for tile in target_tiles)
        self._max_y = max(tile.y for tile in target_tiles)
        engine = config.earth_engine
        fingerprint: dict[str, Any] | None = None
        if engine is not None:
            from mapcv import earth_engine

            # A collection is filtered to the scenes over the tiles the raster covers (not
            # just the region), so every pixel gets all its scenes. Which scenes go into
            # the composite decides the rendered pixels, so those bounds are part of the
            # cache key: tiles rendered for another area are never reused.
            bounds = _tile_range_bounds(
                self._min_x, self._min_y, self._max_x, self._max_y, config.zoom
            )
            # A fresh map URL each time; it holds a short-lived map ID, so the cache and
            # the manifest key on the image and its rendering instead. The manifest needs
            # no bounds: the region and the grid it records decide them.
            self._template = earth_engine.tile_url(engine, bounds)
            cache_key = "earth-engine:" + engine.model_dump_json(exclude={"project"})
            if engine.collection is not None:
                cache_key += f"|bounds:{','.join(repr(value) for value in bounds)}"
            product_id = earth_engine.product_id(engine)
            fingerprint = {"earth_engine": engine.model_dump(mode="json", exclude={"project"})}
        else:
            self._template = resolve_url_template(config.url_template, config.source)
            cache_key = self._template
            product_id = config.source or _xyz_product_id(self._template)
            if config.url_template:
                fingerprint = {"url_template_pbkdf2": url_template_digest(self._template)}
        self._zoom = config.zoom
        # Tiles are fetched lazily per window and evicted once windows move
        # past them, so memory stays bounded to about one chunk of tiles and
        # a resumed run only downloads the chunks it still needs.
        self._tiles: dict[tuple[int, int], bytes] = {}
        self._attempted: set[tuple[int, int]] = set()
        self._cache = TileCache(cache_key) if config.cache else None
        # Tiles fetched but not yet decoded: cached only once they decode (see read_window).
        self._uncached: dict[tuple[int, int], CacheHeaders] = {}
        self.tiles_requested = 0
        self.tiles_cached = 0
        self.tiles_failed = 0
        self.failure_causes: Counter[str] = Counter()
        self.failure_example: str | None = None

        self.metadata = RasterMetadata(
            source_type="xyz",
            product_id=product_id,
            width=(self._max_x - self._min_x + 1) * 256,
            height=(self._max_y - self._min_y + 1) * 256,
            bands=["red", "green", "blue"],
            dtype="uint8",
            crs="EPSG:3857",
            transform=tile_transform(self._min_x, self._min_y, config.zoom),
            chunk_rows=config.strip_rows * 256,
            fingerprint=fingerprint,
        )

    def read_window(
        self, row_start: int, row_stop: int, col_start: int, col_stop: int
    ) -> tuple[npt.NDArray[np.uint8], npt.NDArray[np.bool_]]:
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
        self._cache_decoded()
        return window, valid

    def _cache_decoded(self) -> None:
        """Cache the fetched tiles of a window that has decoded, and only those: a tile a
        server answered with something that is not a usable image would otherwise be
        served from the cache on every later run."""
        cache = self._cache
        pending, self._uncached = self._uncached, {}
        if cache is None:
            return
        for (x, y), headers in pending.items():
            payload = self._tiles.get((x, y))
            if payload is not None:
                cache.put(x, y, self._zoom, payload, headers)

    def _decode_with_pillow(
        self,
        window: npt.NDArray[np.uint8],
        valid: npt.NDArray[np.bool_],
        undecoded: list[tuple[int, int]],
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
        name = f"{self._zoom}/{tile_x}/{tile_y}"
        try:
            with Image.open(BytesIO(payload)) as image:
                tile_image = _rgb_array(image)
        except Exception as exc:
            raise RuntimeError(
                f"Unable to decode XYZ tile {name}: the server did not send a usable image."
                f"{self._forget_cached(tile_x, tile_y)}"
            ) from exc
        if tile_image.shape != (256, 256, 3):
            height, width = tile_image.shape[:2]
            raise ValueError(
                f"XYZ tile sources must return 256x256 RGB-compatible images; tile {name} is "
                f"{width}x{height}.{self._forget_cached(tile_x, tile_y)}"
            )
        return tile_image

    def _forget_cached(self, tile_x: int, tile_y: int) -> str:
        """Drop a tile that did not decode from the cache (an earlier run may have kept
        it); the sentence to add to the error when it was there."""
        cache = self._cache
        if cache is not None and cache.discard(tile_x, tile_y, self._zoom):
            return " The cached copy was removed; run the same command again to download it."
        return ""

    def _evict_rows_above(self, tile_row: int) -> None:
        for key in [key for key in self._tiles if key[1] < tile_row]:
            del self._tiles[key]

    def _fetch(self, keys: list[tuple[int, int]]) -> None:
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
            hits = sum(1 for key in missing if key in self._tiles)
            self.tiles_cached += hits
            _log.debug("tile cache: %d of %d tile(s) found", hits, len(missing))
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
        _log.debug(
            "fetched %d tile(s) at zoom %d, %d failed%s",
            len(missing),
            self._zoom,
            failed,
            f" ({', '.join(f'{n} x {kind}' for kind, n in causes)})" if failed else "",
        )
        self.failure_causes.update(dict(causes))
        if example and self.failure_example is None:
            self.failure_example = example
        for tile, payload, headers in results:
            self._tiles[(tile.x, tile.y)] = payload
            # Black fills of failed tiles have no headers and are never cached; the others
            # are once they have decoded.
            if cache is not None and headers is not None:
                self._uncached[(tile.x, tile.y)] = headers

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
    candidates: list[Any] = []
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
        from pyproj.exceptions import CRSError
    except ImportError as exc:  # pragma: no cover - optional dependency guard
        raise RuntimeError("Install EOPF support with 'pip install mapcv[zarr]'.") from exc

    for candidate in candidates:
        if candidate:
            try:
                return str(CRS.from_user_input(candidate).to_string())
            except (CRSError, TypeError, ValueError):
                continue  # not a CRS pyproj reads; try the next attribute
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
    bounds: tuple[float, float, float, float],
    x_values: npt.NDArray[Any],
    y_values: npt.NDArray[Any],
    resolution: int,
) -> tuple[float, float, float, float]:
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
    finite: npt.NDArray[np.bool_], bands: list[str], row_start: int, row_stop: int
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


_fsspec_config_lock = threading.Lock()


@contextmanager
def _eopf_guards(path: str, *, trust_host: bool = True) -> Iterator[None]:
    """The rules for opening an EOPF product: no ``pickle`` codec (a product names its
    own codecs, and that one runs code), and, over https, connections to public
    addresses only (:mod:`mapcv._net`), redirects included.

    xarray-eopf passes no options to fsspec for an https product, so the connection
    rules go in through fsspec's configuration while the product is opened; the file
    system created then keeps them for the reads that follow. ``trust_host=False`` for
    a product a STAC catalog named: its host is judged by its addresses too.
    """
    from mapcv._zarr_safety import pickle_codec_refused

    https = urlsplit(path).scheme == "https"
    options = fsspec_options(path, trust_host=trust_host) if https else {}
    with pickle_codec_refused():
        if not options:
            yield
            return
        import fsspec.config

        with _fsspec_config_lock:
            saved = fsspec.config.conf.get("https")
            fsspec.config.conf["https"] = {**(saved or {}), **options}
            try:
                yield
            finally:
                if saved is None:
                    fsspec.config.conf.pop("https", None)
                else:
                    fsspec.config.conf["https"] = saved


class EOPFZarrRasterSource:
    """Lazy window reader for one Sentinel-2 L2A EOPF Zarr product."""

    def __init__(self, region: RegionConfig, config: EOPFZarrImageryConfig) -> None:
        fingerprint: dict[str, Any] = {}
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
            refusal = linked_url_refusal(config.search.catalog, path)
            if refusal is not None:
                raise ValueError(f"STAC item {match.item_id}: {refusal}")
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
            with _eopf_guards(path, trust_host=config.search is None):
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
    ) -> tuple[npt.NDArray[np.float32], npt.NDArray[np.bool_]]:
        """Read a window, retrying transient failures (timeouts, empty bands)."""
        for attempt in range(1, EOPF_READ_ATTEMPTS + 1):
            try:
                return self._read_window_once(row_start, row_stop, col_start, col_stop)
            except _NOT_RETRYABLE:
                raise
            except Exception as exc:
                if attempt == EOPF_READ_ATTEMPTS:
                    raise RuntimeError(
                        f"reading rows {row_start}-{row_stop} failed {attempt} times; last "
                        f"error: {exc}"
                    ) from exc
                time.sleep(EOPF_RETRY_DELAYS[min(attempt, len(EOPF_RETRY_DELAYS)) - 1])
        raise AssertionError("unreachable")  # pragma: no cover

    def _read_window_once(
        self, row_start: int, row_stop: int, col_start: int, col_stop: int
    ) -> tuple[npt.NDArray[np.float32], npt.NDArray[np.bool_]]:
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


def geotiff_fingerprint(location: str, *, trust_host: bool = True) -> dict[str, Any]:
    """A cheap identity of a GeoTIFF, so a resumed run notices a different file.

    Local files: size, modification time (in nanoseconds) and the SHA-256 of the first and
    last 64 KiB (where the TIFF headers and, for non-COG files, the directory live). The
    modification time catches pixels edited in place, which leave the size and both ends
    unchanged; hashing the whole file would cost a full read of every file on every run.
    A copy that does not keep modification times (``cp`` without ``-p``) therefore counts
    as another file, which refuses a resume rather than mixing two versions.
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
            "mtime_ns": stat.st_mtime_ns,
            "sha256_head_tail": digest.hexdigest(),
        }
    fingerprint: dict[str, Any] = {"kind": "url", "url": location}
    request = urllib.request.Request(
        _remote_http_url(location),
        headers={"Range": f"bytes=0-{_FINGERPRINT_BYTES - 1}", "User-Agent": "mapcv"},
    )
    try:
        with _urlopen(request, timeout=_FINGERPRINT_TIMEOUT_S, trust_host=trust_host) as response:
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


def _nodata_mask(data: npt.NDArray[Any], nodata: float | None) -> npt.NDArray[np.bool_]:
    """Pixels where every band is NoData (or non-finite, for float rasters)."""
    empty = _all_equal(data, nodata)
    if np.issubdtype(data.dtype, np.floating):
        empty |= np.all(~np.isfinite(data), axis=-1)
    return empty


def _all_equal(data: npt.NDArray[Any], nodata: float | None) -> npt.NDArray[np.bool_]:
    if nodata is None or np.isnan(nodata):
        return np.zeros(data.shape[:2], dtype=np.bool_)
    if np.issubdtype(data.dtype, np.integer):
        info = np.iinfo(data.dtype)
        if nodata != int(nodata) or not info.min <= nodata <= info.max:
            return np.zeros(data.shape[:2], dtype=np.bool_)
    return np.asarray(np.all(data == data.dtype.type(nodata), axis=-1), dtype=np.bool_)


def _warn_nodata_unreachable(nodata: float | None, dtype: Any, where: str) -> None:
    """Warn when ``imagery.nodata`` is a value the files' data type cannot hold: no pixel
    would ever equal it, so it would silently do nothing."""
    if nodata is None or math.isnan(nodata):
        return
    kind = np.dtype(dtype)
    if np.issubdtype(kind, np.integer):
        info = np.iinfo(kind)
        fits = nodata == int(nodata) and info.min <= nodata <= info.max
        holds = f"whole numbers from {info.min} to {info.max}"
    elif np.issubdtype(kind, np.floating):
        limit = float(np.finfo(kind).max)
        fits = abs(nodata) <= limit
        holds = f"numbers from -{limit:g} to {limit:g}"
    else:
        return
    if not fits:
        warnings.warn(
            f"imagery.nodata {nodata:g} cannot occur in {where}, whose {kind} pixels hold "
            f"{holds}; no pixel is treated as NoData because of it",
            UserWarning,
            stacklevel=4,
        )


def _json_nodata(nodata: float | None) -> Any | None:
    """``nodata`` as a JSON-safe, self-equal value (``NaN != NaN`` would break resuming)."""
    if nodata is None:
        return None
    return "nan" if np.isnan(nodata) else nodata


def region_pixel_window(
    bounds: tuple[float, float, float, float], transform: Transform, height: int, width: int
) -> tuple[int, int, int, int, bool]:
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
    cols: list[float] = []
    rows: list[float] = []
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


def _region_miss_hint(
    region: RegionConfig, crs: str, transform: Transform, height: int, width: int
) -> str:
    """Where the raster is in lon/lat, and whether the region has its axes swapped."""
    a, b, c, d, e, f = transform
    xs = [c + a * col + b * row for col in (0, width) for row in (0, height)]
    ys = [f + d * col + e * row for col in (0, width) for row in (0, height)]
    try:
        from pyproj import Transformer

        to_lonlat = Transformer.from_crs(crs, "EPSG:4326", always_xy=True)
        west, south, east, north = to_lonlat.transform_bounds(
            min(xs), min(ys), max(xs), max(ys), densify_pts=21
        )
    except Exception:  # noqa: BLE001 - the CRS units are shown instead
        return ""
    hint = f"; the file covers lon {west:.4f} to {east:.4f}, lat {south:.4f} to {north:.4f}"
    swapped = (region.south, region.west, region.north, region.east)  # as lon, lat, lon, lat
    if (
        -90.0 <= region.west <= 90.0
        and -90.0 <= region.east <= 90.0
        and swapped[0] < east
        and swapped[2] > west
        and swapped[1] < north
        and swapped[3] > south
    ):
        hint += (
            ". The region looks like it has latitude and longitude swapped: west and east "
            "are longitudes, south and north latitudes"
        )
    return hint


def _require_horizontal_crs(epsg: int, name: str) -> None:
    """Reject a file whose CRS is not a 2-D map CRS (a geocentric or vertical one has no
    map coordinates to put a region on)."""
    from pyproj import CRS
    from pyproj.exceptions import CRSError

    try:
        crs = CRS.from_epsg(epsg)
    except CRSError:
        return  # an unknown code: the region/extent checks report what they can
    if not (crs.is_projected or crs.is_geographic):
        raise ValueError(
            f"GeoTIFF '{name}' is tagged with EPSG:{epsg} ({crs.name}), which is not a map CRS "
            "(it is geocentric, vertical or engineering, not projected or geographic); "
            "re-tag the file with the horizontal CRS of its pixels"
        )


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
        image_format: str | None = None,
        linked: bool = False,
    ) -> None:
        # URL safety rules are enforced by GeoTiffImageryConfig validation. A file a STAC
        # catalog named (``linked``) is not the user's host: it is judged by its addresses.
        location = geotiff_location(config.path)
        self._tif = GeoTiff(location, trust_host=not linked)
        info = self._tif.info
        name = _safe_product_id(config.path)

        if info.epsg is None:
            raise ValueError(
                f"GeoTIFF '{name}' has no usable CRS: {info.crs_error or 'no CRS in the file'}. "
                "mapcv reads files whose CRS is an EPSG code; re-project or re-tag the file."
            )
        _require_horizontal_crs(info.epsg, name)
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
                + _region_miss_hint(region, crs, file_transform, level_height, level_width)
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
        _warn_nodata_unreachable(config.nodata, info.dtype, f"'{name}'")
        self._nodata = config.nodata if config.nodata is not None else info.nodata
        effective_nodata = _json_nodata(self._nodata)
        self._dtype = info.dtype
        # Where the file the fingerprint describes is, for a message about its times.
        self.file_locations = {"": location}
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
                **geotiff_fingerprint(location, trust_host=not linked),
                "overview": config.overview,
                "bands": selected,
                "nodata": effective_nodata,
            },
        )

    def read_window(
        self, row_start: int, row_stop: int, col_start: int, col_stop: int
    ) -> tuple[npt.NDArray[Any], npt.NDArray[np.bool_]]:
        data, inside = self._tif.read_window(
            self._row0 + row_start,
            self._row0 + row_stop,
            self._col0 + col_start,
            self._col0 + col_stop,
            bands=self._bands,
            overview=self._overview,
        )
        _log.debug(
            "GeoTIFF window rows %d:%d, cols %d:%d (%.1f MB, %d%% inside the file)",
            self._row0 + row_start,
            self._row0 + row_stop,
            self._col0 + col_start,
            self._col0 + col_stop,
            data.nbytes / 2**20,
            round(100 * float(inside.mean())) if inside.size else 0,
        )
        valid = inside & ~_nodata_mask(data, self._nodata)
        if self._expand_gray:
            data = np.repeat(data, 3, axis=-1)
        return data, valid

    def close(self) -> None:
        """Nothing to release: the reader holds no open handles between reads."""


# Mosaic files must share their pixel size to this relative tolerance, and their grids
# must line up to this fraction of a pixel.
_MOSAIC_SCALE_TOLERANCE = 1e-9
_MOSAIC_ALIGN_TOLERANCE_PX = 1e-6


def _same_size(first: float, second: float) -> bool:
    return abs(first - second) <= _MOSAIC_SCALE_TOLERANCE * abs(second)


class GeoTiffMosaicSource:
    """Windowed reader over several GeoTIFFs that together cover an area, as one raster.

    The files must share a CRS, data type, band count, NoData value (unless
    ``imagery.nodata`` sets one), north-up pixel size and pixel grid; their extents may
    leave gaps (read as no imagery) or overlap (the first file in sorted order wins, as in
    ``rasterio.merge``). Only headers are read up front; each window reads just the files
    it overlaps, so memory does not grow with the number of files.
    """

    def __init__(
        self,
        region: RegionConfig,
        config: GeoTiffImageryConfig,
        files: list[str],
        *,
        image_format: str | None = None,
    ) -> None:
        pattern = _safe_product_id(config.path)
        self._tifs = [GeoTiff(geotiff_location(path)) for path in files]
        names = [Path(path).name for path in files]
        reference = self._tifs[0].info
        for tif, name in zip(self._tifs, names):
            info = tif.info
            if info.epsg is None:
                raise ValueError(
                    f"GeoTIFF '{name}' has no usable CRS: "
                    f"{info.crs_error or 'no CRS in the file'}. mapcv reads files whose CRS is "
                    "an EPSG code; re-project or re-tag the file."
                )
            _require_horizontal_crs(info.epsg, name)
            if info.transform is None:
                raise ValueError(f"GeoTIFF '{name}' has no georeferencing")
            if config.overview > len(info.overviews):
                raise ValueError(
                    f"imagery.overview is {config.overview}, but '{name}' has "
                    f"{len(info.overviews)} overview level(s) (0 is the full resolution)"
                )
            for what, mine, theirs in (
                ("CRS", info.epsg, reference.epsg),
                ("data type", info.dtype, reference.dtype),
                ("band count", info.count, reference.count),
            ):
                if mine != theirs:
                    raise ValueError(
                        f"imagery.path: '{name}' has {what} {mine} but '{names[0]}' has "
                        f"{theirs}; the files of a mosaic must match"
                    )
            if config.nodata is None and _json_nodata(info.nodata) != _json_nodata(
                reference.nodata
            ):
                raise ValueError(
                    f"imagery.path: '{name}' has NoData {info.nodata} but '{names[0]}' has "
                    f"{reference.nodata}; set imagery.nodata to the value to use for all files"
                )

        transforms = []
        sizes = []
        for tif in self._tifs:
            transform = tif.info.overview_transform(config.overview)
            assert transform is not None
            transforms.append(transform)
            sizes.append(
                (tif.info.height, tif.info.width)
                if config.overview == 0
                else tif.info.overviews[config.overview - 1]
            )
        a, b, c, d, e, f = transforms[0]
        offsets = []
        full_a, _, _, _, full_e, _ = self._tifs[0].info.transform or (0.0,) * 6
        for (ta, tb, tc, td, te, tf), name, tif in zip(transforms, names, self._tifs):
            if tb or td or b or d:
                raise ValueError(
                    f"imagery.path: '{name}' is rotated; a mosaic needs north-up files"
                )
            file_a, _, _, _, file_e, _ = tif.info.transform or (0.0,) * 6
            if not (_same_size(file_a, full_a) and _same_size(file_e, full_e)):
                raise ValueError(
                    f"imagery.path: '{name}' has pixel size {file_a} x {-file_e} but "
                    f"'{names[0]}' has {full_a} x {-full_e}; resample the files to one pixel "
                    "size first"
                )
            if not (_same_size(ta, a) and _same_size(te, e)):
                # Same pixels, but the overviews were not built on a common grid (a file
                # whose size is not a multiple of the overview factor is rounded).
                raise ValueError(
                    f"imagery.path: overview {config.overview} of '{name}' has pixel size "
                    f"{ta} x {-te} but that of '{names[0]}' has {a} x {-e}, although the files "
                    "have the same pixel size: their overviews do not share one grid. Use "
                    "imagery.overview: 0, or build the overviews from one combined file"
                )
            col, row = (tc - c) / a, (tf - f) / e
            if (
                abs(col - round(col)) > _MOSAIC_ALIGN_TOLERANCE_PX
                or abs(row - round(row)) > _MOSAIC_ALIGN_TOLERANCE_PX
            ):
                raise ValueError(
                    f"imagery.path: '{name}' is not on the same pixel grid as '{names[0]}' "
                    f"(offset by {col - round(col):+.3f}, {row - round(row):+.3f} pixel); "
                    "align the files first (for example gdalwarp -tap)"
                )
            offsets.append((round(row), round(col)))
        top = min(row for row, _ in offsets)
        left = min(col for _, col in offsets)
        bottom = max(row + h for (row, _), (h, _) in zip(offsets, sizes))
        right = max(col + w for (_, col), (_, w) in zip(offsets, sizes))
        # Each file's place in the mosaic, as (row0, row1, col0, col1).
        self._places = [
            (row - top, row - top + h, col - left, col - left + w)
            for (row, col), (h, w) in zip(offsets, sizes)
        ]
        mosaic_transform: Transform = (a, 0.0, c + left * a, 0.0, e, f + top * e)
        height, width = bottom - top, right - left

        selected = (
            list(config.bands) if config.bands is not None else list(range(1, reference.count + 1))
        )
        if max(selected) > reference.count:
            raise ValueError(
                f"imagery.bands asks for band {max(selected)}, but the files have "
                f"{reference.count} band(s)"
            )
        self._bands = [band - 1 for band in selected]
        self._expand_gray = False
        if image_format is not None and image_format not in ("npy", "tif"):
            if reference.dtype != np.uint8 or len(selected) not in (1, 3):
                raise ValueError(
                    f"the files have {len(selected)} selected band(s) of {reference.dtype}; "
                    f"writer.image_format '{image_format}' writes 1 or 3 bands of uint8. "
                    "Use writer.image_format: npy or tif (keep band count and dtype), or select "
                    "1 or 3 bands of uint8 files with imagery.bands"
                )
            self._expand_gray = len(selected) == 1
        band_names = [f"b{band}" for band in selected]
        if self._expand_gray:
            band_names = band_names * 3

        crs = f"EPSG:{reference.epsg}"
        bounds = region_bounds_in_crs(region, crs)
        row0, row1, col0, col1, extends = region_pixel_window(
            bounds, mosaic_transform, height, width
        )
        if row0 >= row1 or col0 >= col1:
            raise ValueError(
                f"requested region does not intersect the {len(files)} GeoTIFFs of {pattern} "
                f"(mosaic extent in {crs}: {_extent_text(mosaic_transform, height, width)})"
                + _region_miss_hint(region, crs, mosaic_transform, height, width)
            )
        if extends:
            warnings.warn(
                f"the region extends beyond the {len(files)} GeoTIFFs of {pattern}; only the "
                "part they cover is used (patches at its edge are padded or dropped by "
                "sampler.edge_strategy)",
                UserWarning,
                stacklevel=3,
            )
        self._overview = config.overview
        self._row0, self._col0 = row0, col0
        _warn_nodata_unreachable(config.nodata, reference.dtype, f"the files of {pattern}")
        self._nodata = config.nodata if config.nodata is not None else reference.nodata
        self._file_nodata = [
            config.nodata if config.nodata is not None else tif.info.nodata for tif in self._tifs
        ]
        self._dtype = np.dtype(reference.dtype)
        _log.debug("GeoTIFF mosaic of %d file(s), %dx%d px in %s", len(files), width, height, crs)
        self.file_locations = {name: str(path) for name, path in zip(names, files)}
        self.metadata = RasterMetadata(
            source_type="geotiff",
            product_id=f"{pattern} ({len(files)} files)",
            width=col1 - col0,
            height=row1 - row0,
            bands=band_names,
            dtype=str(reference.dtype),
            crs=crs,
            transform=offset_transform(mosaic_transform, row0, col0),
            chunk_rows=config.chunk_rows,
            fingerprint={
                "kind": "mosaic",
                "files": [
                    {"name": name, **geotiff_fingerprint(geotiff_location(path))}
                    for name, path in zip(names, files)
                ],
                "overview": config.overview,
                "bands": selected,
                "nodata": _json_nodata(self._nodata),
            },
        )

    def _fill_value(self) -> Any:
        nodata = self._nodata
        if nodata is None:
            return 0
        if np.issubdtype(self._dtype, np.integer):
            info = np.iinfo(self._dtype)
            if math.isnan(nodata) or nodata != int(nodata) or not info.min <= nodata <= info.max:
                return 0
        return nodata

    def read_window(
        self, row_start: int, row_stop: int, col_start: int, col_stop: int
    ) -> tuple[npt.NDArray[Any], npt.NDArray[np.bool_]]:
        r0, r1 = self._row0 + row_start, self._row0 + row_stop
        c0, c1 = self._col0 + col_start, self._col0 + col_stop
        height, width = max(0, r1 - r0), max(0, c1 - c0)
        data = np.full((height, width, len(self._bands)), self._fill_value(), dtype=self._dtype)
        valid = np.zeros((height, width), dtype=np.bool_)
        read = 0
        for tif, nodata, (f_r0, f_r1, f_c0, f_c1) in zip(
            self._tifs, self._file_nodata, self._places
        ):
            top, bottom = max(r0, f_r0), min(r1, f_r1)
            left, right = max(c0, f_c0), min(c1, f_c1)
            if top >= bottom or left >= right:
                continue
            part, inside = tif.read_window(
                top - f_r0,
                bottom - f_r0,
                left - f_c0,
                right - f_c0,
                bands=self._bands,
                overview=self._overview,
            )
            read += 1
            usable = inside & ~_nodata_mask(part, nodata)
            target = (slice(top - r0, bottom - r0), slice(left - c0, right - c0))
            # The first file with data wins where files overlap.
            fresh = usable & ~valid[target]
            data[target][fresh] = part[fresh]
            valid[target] |= fresh
        _log.debug(
            "GeoTIFF mosaic window rows %d:%d, cols %d:%d from %d file(s)", r0, r1, c0, c1, read
        )
        if self._expand_gray:
            data = np.repeat(data, 3, axis=-1)
        return data, valid

    def close(self) -> None:
        """Nothing to release: the readers hold no open handles between reads."""


def open_geotiff_source(
    region: RegionConfig, config: GeoTiffImageryConfig, *, image_format: str | None = None
) -> GeoTiffRasterSource | GeoTiffMosaicSource:
    """A GeoTIFF source: one file as it is, or several (a glob pattern) as a mosaic."""
    files = config.files()
    if len(files) == 1 and files[0] == config.path:
        return GeoTiffRasterSource(region, config, image_format=image_format)
    return GeoTiffMosaicSource(region, config, files, image_format=image_format)


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
            refusal = linked_url_refusal(config.search.catalog, location)
            if refusal is not None:
                raise ValueError(f"STAC item {item_id} asset '{key}': {refusal}")
            return location

        locations = {key: href(key) for key in wanted}
        opened: dict[str, WindowedRasterSource] = {}
        try:
            for key, location in locations.items():
                opened[key] = GeoTiffRasterSource(
                    region,
                    GeoTiffImageryConfig(
                        type="geotiff", path=location, bands=[1], chunk_rows=config.chunk_rows
                    ),
                    image_format="npy",
                    linked=True,
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
        fingerprint: dict[str, Any] = {
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
    ) -> tuple[npt.NDArray[Any], npt.NDArray[np.bool_]]:
        layers = []
        valid: npt.NDArray[np.bool_] | None = None
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
    image_format: str | None = None,
) -> WindowedRasterSource:
    """Construct the configured raster source.

    ``image_format`` (``writer.image_format``) lets a GeoTIFF source check that its
    bands and dtype fit the output; other sources ignore it.
    """
    if isinstance(imagery, XYZImageryConfig):
        return XYZRasterSource(region, imagery)
    if isinstance(imagery, GeoTiffImageryConfig):
        return open_geotiff_source(region, imagery, image_format=image_format)
    if isinstance(imagery, StacCogImageryConfig):
        return StacCogRasterSource(region, imagery)
    return EOPFZarrRasterSource(region, imagery)
