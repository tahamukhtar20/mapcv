"""Windowed imagery sources for XYZ tiles and EOPF Sentinel-2 Zarr products."""

from __future__ import annotations

import math
import sys
import time
from dataclasses import dataclass
from functools import lru_cache
from io import BytesIO
from pathlib import Path
from collections import Counter
from typing import Any, Dict, List, Optional, Protocol, Set, Tuple
from urllib.parse import urlsplit

import numpy as np
import numpy.typing as npt
from PIL import Image
from shapely.geometry.base import BaseGeometry
from shapely.ops import transform as shapely_transform

from mapcv._mapcv_rs import (
    PyTileIndex,
    decode_tile_window,
    fetch_tiles,
    snap_bbox,
    tile_transform,
    tiles,
)
from mapcv.config import (
    EOPFZarrImageryConfig,
    RegionConfig,
    XYZImageryConfig,
    eopf_local_path,
)
from mapcv.downloader import resolve_url_template


Transform = Tuple[float, float, float, float, float, float]


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


@lru_cache(maxsize=8)
def _wgs84_transformer(destination_crs: str) -> Any:
    try:
        from pyproj import Transformer
    except ImportError as exc:  # pragma: no cover - exercised by optional-extra smoke tests
        raise RuntimeError(
            "Non-Web-Mercator imagery requires the Zarr dependencies. "
            "Install them with 'pip install mapcv[zarr]'."
        ) from exc
    return Transformer.from_crs("EPSG:4326", destination_crs, always_xy=True)


def transform_geometry_to_crs(geometry: BaseGeometry, destination_crs: str) -> BaseGeometry:
    """Transform a WGS-84 geometry into ``destination_crs``."""
    return shapely_transform(_wgs84_transformer(destination_crs).transform, geometry)


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
        self.tiles_requested = 0
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
        """Decode the tiles Rust left to Pillow (JPEG, 16-bit PNG, rare formats)."""
        for tile_x, tile_y in undecoded:
            tile_image = self._decode_tile(tile_x, tile_y)
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
        config = self._config
        results, failed, (causes, example) = fetch_tiles(
            [PyTileIndex(x, y, self._zoom) for x, y in missing],
            self._template,
            max_connections=config.max_connections,
            policy=config.policy,
            max_failed_ratio=config.max_failed_ratio,
        )
        self.tiles_requested += len(missing)
        self.tiles_failed += failed
        self.failure_causes.update(dict(causes))
        if example and self.failure_example is None:
            self.failure_example = example
        for tile, payload in results:
            self._tiles[(tile.x, tile.y)] = payload

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
    eps = 1e-9
    left, bottom, right, top = bounds
    return (
        x_edge + math.floor((left - x_edge) / x_res + eps) * x_res,
        y_edge + math.floor((bottom - y_edge) / y_res + eps) * y_res,
        x_edge + math.ceil((right - x_edge) / x_res - eps) * x_res,
        y_edge + math.ceil((top - y_edge) / y_res - eps) * y_res,
    )


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
        # URL safety rules are enforced by EOPFZarrImageryConfig validation.
        local_path = eopf_local_path(config.path)
        if local_path is not None and not local_path.exists():
            raise FileNotFoundError(f"EOPF Zarr product not found: {local_path}")

        try:
            import xarray as xr
            from pyproj import Transformer
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

        storage_options = {"anon": True} if urlsplit(config.path).scheme == "s3" else None

        def open_dataset(**spatial_options: Any) -> Any:
            return xr.open_dataset(
                config.path,
                engine="eopf-zarr",
                op_mode="analysis",
                variables=config.bands,
                resolution=config.resolution,
                chunks={},
                storage_options=storage_options,
                **spatial_options,
            )

        try:
            discovery = open_dataset()
        except Exception as exc:
            raise RuntimeError(
                f"Unable to open anonymous EOPF product '{_safe_product_id(config.path)}': {exc}"
            ) from exc

        missing = [band for band in config.bands if band not in discovery.data_vars]
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
        transformer = Transformer.from_crs("EPSG:4326", crs, always_xy=True)
        left, bottom, right, top = transformer.transform_bounds(
            region.west,
            region.south,
            region.east,
            region.north,
            densify_pts=21,
        )
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
                f"Unable to crop anonymous EOPF product '{_safe_product_id(config.path)}': {exc}"
            ) from exc

        height = int(dataset.sizes.get("y", 0))
        width = int(dataset.sizes.get("x", 0))
        if height == 0 or width == 0:
            dataset.close()
            raise ValueError("requested region does not intersect the EOPF product")

        self._dataset = dataset
        self._bands = list(config.bands)
        self.metadata = RasterMetadata(
            source_type="eopf_zarr",
            product_id=_safe_product_id(config.path),
            width=width,
            height=height,
            bands=list(config.bands),
            dtype="float32",
            crs=crs,
            transform=_coordinate_transform(dataset, config.resolution),
            chunk_rows=config.chunk_rows,
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
        return image, valid

    def close(self) -> None:
        """Close the underlying xarray dataset."""
        self._dataset.close()


def open_raster_source(
    region: RegionConfig,
    imagery: XYZImageryConfig | EOPFZarrImageryConfig,
) -> WindowedRasterSource:
    """Construct the configured raster source."""
    if isinstance(imagery, XYZImageryConfig):
        return XYZRasterSource(region, imagery)
    return EOPFZarrRasterSource(region, imagery)
