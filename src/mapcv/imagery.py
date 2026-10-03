"""Windowed imagery sources for XYZ tiles and EOPF Sentinel-2 Zarr products."""

from __future__ import annotations

from dataclasses import dataclass
from io import BytesIO
from pathlib import Path
from typing import Any, List, Protocol, Tuple
from urllib.parse import urlsplit

import numpy as np
import numpy.typing as npt
from PIL import Image
from shapely.geometry.base import BaseGeometry
from shapely.ops import transform as shapely_transform

from mapcv._mapcv_rs import snap_bbox, tile_transform, tiles
from mapcv.config import EOPFZarrImageryConfig, RegionConfig, XYZImageryConfig
from mapcv.downloader import download_region, resolve_url_template


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


def transform_geometry_to_crs(geometry: BaseGeometry, destination_crs: str) -> BaseGeometry:
    """Transform a WGS-84 geometry into ``destination_crs``."""
    try:
        from pyproj import Transformer
    except ImportError as exc:  # pragma: no cover - exercised by optional-extra smoke tests
        raise RuntimeError(
            "Non-Web-Mercator imagery requires the Zarr dependencies. "
            "Install them with 'pip install mapcv[zarr]'."
        ) from exc

    transformer = Transformer.from_crs("EPSG:4326", destination_crs, always_xy=True)
    return shapely_transform(transformer.transform, geometry)


def _safe_product_id(path_or_url: str) -> str:
    parsed = urlsplit(path_or_url)
    name = Path(parsed.path.rstrip("/")).name
    return name or parsed.hostname or "imagery-product"


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

        downloaded = download_region(
            region.west,
            region.south,
            region.east,
            region.north,
            config.zoom,
            url_template=config.url_template,
            source=config.source,
            max_connections=config.max_connections,
            policy=config.policy,
            max_failed_ratio=config.max_failed_ratio,
        )
        self._tiles = {(tile.x, tile.y): payload for tile, payload in downloaded}
        self._min_x = min(tile.x for tile in target_tiles)
        self._max_x = max(tile.x for tile in target_tiles)
        self._min_y = min(tile.y for tile in target_tiles)
        self._max_y = max(tile.y for tile in target_tiles)

        template = resolve_url_template(config.url_template, config.source)
        self.metadata = RasterMetadata(
            source_type="xyz",
            product_id=config.source or _safe_product_id(template),
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
        window = np.zeros((height, width, 3), dtype=np.uint8)
        valid = np.zeros((height, width), dtype=np.bool_)
        if height == 0 or width == 0:
            return window, valid

        tile_col_start = self._min_x + col_start // 256
        tile_col_stop = self._min_x + (col_stop - 1) // 256
        tile_row_start = self._min_y + row_start // 256
        tile_row_stop = self._min_y + (row_stop - 1) // 256
        for tile_y in range(tile_row_start, tile_row_stop + 1):
            for tile_x in range(tile_col_start, tile_col_stop + 1):
                payload = self._tiles.get((tile_x, tile_y))
                if payload is None:
                    continue
                try:
                    with Image.open(BytesIO(payload)) as image:
                        tile_image = np.asarray(image.convert("RGB"), dtype=np.uint8)
                except Exception as exc:
                    raise RuntimeError(f"Unable to decode XYZ tile {tile_x}/{tile_y}") from exc
                if tile_image.shape != (256, 256, 3):
                    raise ValueError("XYZ tile sources must return 256x256 RGB-compatible images")

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
                window[target_row_start:target_row_stop, target_col_start:target_col_stop] = (
                    tile_image[
                        source_row_start:source_row_stop,
                        source_col_start:source_col_stop,
                    ]
                )
                valid[target_row_start:target_row_stop, target_col_start:target_col_stop] = True
        return window, valid

    def close(self) -> None:
        """XYZ sources hold no external resources."""


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
                return CRS.from_user_input(candidate).to_string()
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


class EOPFZarrRasterSource:
    """Lazy window reader for one Sentinel-2 L2A EOPF Zarr product."""

    def __init__(self, region: RegionConfig, config: EOPFZarrImageryConfig) -> None:
        parsed = urlsplit(config.path)
        if parsed.scheme not in ("", "file", "http", "https", "s3"):
            raise ValueError("EOPF path must be local, file://, HTTPS, or anonymous s3://")
        if parsed.username or parsed.password or parsed.query or parsed.fragment:
            raise ValueError("EOPF URLs must not contain credentials, query strings, or fragments")
        if parsed.scheme in ("", "file"):
            local_path = Path(parsed.path if parsed.scheme == "file" else config.path)
            if not local_path.exists():
                raise FileNotFoundError(f"EOPF Zarr product not found: {local_path}")

        try:
            import xarray as xr
            from pyproj import Transformer
        except ImportError as exc:
            raise RuntimeError(
                "EOPF Zarr support is optional; install it with 'pip install mapcv[zarr]'."
            ) from exc

        storage_options = {"anon": True} if parsed.scheme == "s3" else None

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
        window = self._dataset[self._bands].isel(
            y=slice(row_start, row_stop), x=slice(col_start, col_stop)
        )
        array_data = window.to_array(dim="band").transpose("y", "x", "band").data
        if hasattr(array_data, "compute"):
            array_data = array_data.compute()
        image = np.asarray(array_data, dtype=np.float32)
        if image.ndim != 3 or image.shape[-1] != len(self._bands):
            raise ValueError("selected EOPF variables must resolve to two-dimensional y/x rasters")
        valid = np.any(np.isfinite(image), axis=-1)
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
