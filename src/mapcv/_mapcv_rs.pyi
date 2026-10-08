"""Type stubs for mapcv's Rust extension (``src/python.rs``, ``src/tile_decoder.rs``).

tests/test_extension_stubs.py checks these against the compiled module with
``mypy.stubtest``; keep them in step with the ``#[pyfunction]`` signatures.
"""

from collections.abc import Callable, Sequence
from typing import (
    Final,
    Literal,
    TypeAlias,
    TypedDict,
    final,
    overload,
)

import numpy as np
import numpy.typing as npt

__version__: str

__all__ = [
    "BBox",
    "GeoTiff",
    "PyBBox",
    "PyTileIndex",
    "TileIndex",
    "__version__",
    "bounds",
    "decode_tile_window",
    "fetch_tiles",
    "grid_sample_anchors",
    "parse_kml",
    "random_anchor_capacity",
    "random_sample_anchors",
    "rasterize",
    "set_public_only",
    "snap_bbox",
    "stitch_tiles",
    "tile",
    "tile_transform",
    "tiles",
    "write_geotiffs",
    "write_patches",
    "xy",
    "xy_bounds",
]

_Transform: TypeAlias = tuple[float, float, float, float, float, float]
# A ring is a list of (x, y) vertices; a polygon is its exterior ring, then holes.
_Ring: TypeAlias = list[tuple[float, float]]

@final
class TileIndex:
    """An XYZ tile index; equal tiles compare and hash alike, and pickle."""

    @property
    def x(self) -> int: ...
    @property
    def y(self) -> int: ...
    @property
    def z(self) -> int: ...
    def __new__(cls, x: int, y: int, z: int) -> TileIndex: ...
    def __eq__(self, value: object, /) -> bool: ...
    def __hash__(self) -> int: ...
    def __getnewargs__(self) -> tuple[int, int, int]: ...

@final
class BBox:
    """A bounding box (degrees, or Web Mercator metres from ``xy_bounds``)."""

    @property
    def west(self) -> float: ...
    @property
    def south(self) -> float: ...
    @property
    def east(self) -> float: ...
    @property
    def north(self) -> float: ...
    def __new__(cls, west: float, south: float, east: float, north: float) -> BBox: ...
    def __eq__(self, value: object, /) -> bool: ...
    def __hash__(self) -> int: ...
    def __getnewargs__(self) -> tuple[float, float, float, float]: ...

# The class names before mapcv 0.3.
PyTileIndex: Final = TileIndex
PyBBox: Final = BBox

class _GeoTiffMetadata(TypedDict):
    """What ``GeoTiff.metadata()`` returns."""

    width: int
    height: int
    count: int
    dtype: str
    epsg: int | None
    crs_error: str | None
    crs_citation: str | None
    transform: _Transform | None
    raster_type: Literal["area", "point"]
    nodata: float | None
    tiled: bool
    block_size: tuple[int, int]
    overviews: list[tuple[int, int]]
    compression: str
    predictor: int
    planar: bool
    photometric: str
    byte_order: Literal["little", "big"]
    bigtiff: bool

@final
class GeoTiff:
    """A local or remote (``https://``, ``s3://``) GeoTIFF, read by window."""

    def __new__(cls, path: str, cache_bytes: int = ...) -> GeoTiff: ...
    def metadata(self) -> _GeoTiffMetadata: ...
    def read_window(
        self,
        row0: int,
        row1: int,
        col0: int,
        col1: int,
        bands: Sequence[int] | None = None,
        overview: int = 0,
    ) -> tuple[npt.NDArray[np.generic], npt.NDArray[np.bool_]]: ...

def xy(lng: float, lat: float) -> tuple[float, float]: ...
def tile(lng: float, lat: float, zoom: int) -> TileIndex: ...
def tiles(
    west: float, south: float, east: float, north: float, zooms: Sequence[int]
) -> list[TileIndex]: ...
def xy_bounds(x: int, y: int, z: int) -> BBox: ...
def bounds(x: int, y: int, z: int) -> BBox: ...
def snap_bbox(west: float, south: float, east: float, north: float, zoom: int) -> BBox: ...

# Failed-tile counts per cause, and one example message.
_FailureCauses: TypeAlias = tuple[list[tuple[str, int]], str | None]
# Cache-Control, Expires, Date and Age of a response (None when absent).
_CacheHeaders: TypeAlias = tuple[str | None, str | None, str | None, str | None]

@overload
def fetch_tiles(
    tiles: Sequence[TileIndex],
    url_template: str,
    callback: Callable[[int], object] | None = None,
    max_connections: int = 16,
    policy: str = "lenient",
    max_failed_ratio: float = 0.05,
    cache_headers: Literal[False] = False,
) -> tuple[list[tuple[TileIndex, bytes]], int, _FailureCauses]: ...
@overload
def fetch_tiles(
    tiles: Sequence[TileIndex],
    url_template: str,
    callback: Callable[[int], object] | None = None,
    max_connections: int = 16,
    policy: str = "lenient",
    max_failed_ratio: float = 0.05,
    *,
    cache_headers: Literal[True],
) -> tuple[list[tuple[TileIndex, bytes, _CacheHeaders | None]], int, _FailureCauses]: ...
def grid_sample_anchors(
    height: int, width: int, patch_size: int, stride: int, edge_strategy: str = "pad"
) -> list[tuple[int, int]]: ...
def random_sample_anchors(
    height: int,
    width: int,
    patch_size: int,
    count: int,
    seed: int = 42,
    edge_strategy: str = "pad",
) -> list[tuple[int, int]]: ...
def random_anchor_capacity(
    height: int, width: int, patch_size: int, edge_strategy: str = "pad"
) -> int: ...
def rasterize(
    polygons: Sequence[tuple[Sequence[Sequence[tuple[float, float]]], int]],
    height: int,
    width: int,
    transform: _Transform,
    all_touched: bool = False,
) -> npt.NDArray[np.uint8]: ...
def stitch_tiles(
    tile_data: Sequence[tuple[TileIndex, bytes]],
) -> tuple[npt.NDArray[np.uint8], int, int]: ...
def tile_transform(min_x: int, min_y: int, zoom: int) -> _Transform: ...
def decode_tile_window(
    tiles: Sequence[tuple[int, int, bytes]],
    origin_x: int,
    origin_y: int,
    row_start: int,
    row_stop: int,
    col_start: int,
    col_stop: int,
) -> tuple[npt.NDArray[np.uint8], npt.NDArray[np.bool_], list[tuple[int, int]]]: ...
def write_patches(
    image_patches: npt.NDArray[np.uint8],
    mask_patches: npt.NDArray[np.uint8] | None,
    meta: Sequence[tuple[int, int, bool]],
    start_idx: int,
    strip_index: int,
    images_dir: str,
    masks_dir: str,
    image_format: str = "png",
    jpg_quality: int = 95,
    jpg_subsampling: str = "4:2:0",
) -> list[tuple[str, str | None, int, int, bool, int, list[tuple[int, int]], float]]: ...
def write_geotiffs(
    data: npt.NDArray[np.uint8],
    dtype: str,
    shape: tuple[int, int, int, int],
    transforms: Sequence[Sequence[float]],
    names: Sequence[str],
    directory: str,
    epsg: int,
    geographic: bool,
    nodata: float | None = None,
    band_names: Sequence[str] | None = None,
    level: int | None = None,
) -> None: ...
def parse_kml(
    data: bytes, label_field: str | None = None
) -> tuple[list[tuple[list[_Ring], str | None]], int]: ...
def set_public_only(on: bool) -> None: ...
