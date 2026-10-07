"""Type stubs for mapcv's Rust extension (``src/python.rs``, ``src/tile_decoder.rs``).

tests/test_extension_stubs.py checks these against the compiled module with
``mypy.stubtest``; keep them in step with the ``#[pyfunction]`` signatures.
"""

from typing import (
    Callable,
    Final,
    List,
    Literal,
    Optional,
    Sequence,
    Tuple,
    TypedDict,
    final,
    overload,
)

import numpy as np
import numpy.typing as npt

__all__ = [
    "xy",
    "tile",
    "tiles",
    "xy_bounds",
    "bounds",
    "snap_bbox",
    "fetch_tiles",
    "rasterize",
    "grid_sample_anchors",
    "random_sample_anchors",
    "random_anchor_capacity",
    "stitch_tiles",
    "tile_transform",
    "decode_tile_window",
    "write_patches",
    "write_geotiffs",
    "parse_kml",
    "TileIndex",
    "BBox",
    "PyTileIndex",
    "PyBBox",
    "GeoTiff",
]

_Transform = Tuple[float, float, float, float, float, float]
# A ring is a list of (x, y) vertices; a polygon is its exterior ring, then holes.
_Ring = List[Tuple[float, float]]

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
    def __getnewargs__(self) -> Tuple[int, int, int]: ...

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
    def __getnewargs__(self) -> Tuple[float, float, float, float]: ...

# The class names before mapcv 0.3.
PyTileIndex: Final = TileIndex
PyBBox: Final = BBox

class _GeoTiffMetadata(TypedDict):
    """What ``GeoTiff.metadata()`` returns."""

    width: int
    height: int
    count: int
    dtype: str
    epsg: Optional[int]
    crs_error: Optional[str]
    crs_citation: Optional[str]
    transform: Optional[_Transform]
    raster_type: Literal["area", "point"]
    nodata: Optional[float]
    tiled: bool
    block_size: Tuple[int, int]
    overviews: List[Tuple[int, int]]
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
        bands: Optional[Sequence[int]] = None,
        overview: int = 0,
    ) -> Tuple[npt.NDArray[np.generic], npt.NDArray[np.bool_]]: ...

def xy(lng: float, lat: float) -> Tuple[float, float]: ...
def tile(lng: float, lat: float, zoom: int) -> TileIndex: ...
def tiles(
    west: float, south: float, east: float, north: float, zooms: Sequence[int]
) -> List[TileIndex]: ...
def xy_bounds(x: int, y: int, z: int) -> BBox: ...
def bounds(x: int, y: int, z: int) -> BBox: ...
def snap_bbox(west: float, south: float, east: float, north: float, zoom: int) -> BBox: ...

# Failed-tile counts per cause, and one example message.
_FailureCauses = Tuple[List[Tuple[str, int]], Optional[str]]
# Cache-Control, Expires, Date and Age of a response (None when absent).
_CacheHeaders = Tuple[Optional[str], Optional[str], Optional[str], Optional[str]]

@overload
def fetch_tiles(
    tiles: Sequence[TileIndex],
    url_template: str,
    callback: Optional[Callable[[int], object]] = None,
    max_connections: int = 16,
    policy: str = "lenient",
    max_failed_ratio: float = 0.05,
    cache_headers: Literal[False] = False,
) -> Tuple[List[Tuple[TileIndex, bytes]], int, _FailureCauses]: ...
@overload
def fetch_tiles(
    tiles: Sequence[TileIndex],
    url_template: str,
    callback: Optional[Callable[[int], object]] = None,
    max_connections: int = 16,
    policy: str = "lenient",
    max_failed_ratio: float = 0.05,
    *,
    cache_headers: Literal[True],
) -> Tuple[List[Tuple[TileIndex, bytes, Optional[_CacheHeaders]]], int, _FailureCauses]: ...
def grid_sample_anchors(
    height: int, width: int, patch_size: int, stride: int, edge_strategy: str = "pad"
) -> List[Tuple[int, int]]: ...
def random_sample_anchors(
    height: int,
    width: int,
    patch_size: int,
    count: int,
    seed: int = 42,
    edge_strategy: str = "pad",
) -> List[Tuple[int, int]]: ...
def random_anchor_capacity(
    height: int, width: int, patch_size: int, edge_strategy: str = "pad"
) -> int: ...
def rasterize(
    polygons: Sequence[Tuple[Sequence[Sequence[Tuple[float, float]]], int]],
    height: int,
    width: int,
    transform: _Transform,
    all_touched: bool = False,
) -> npt.NDArray[np.uint8]: ...
def stitch_tiles(
    tile_data: Sequence[Tuple[TileIndex, bytes]],
) -> Tuple[npt.NDArray[np.uint8], int, int]: ...
def tile_transform(min_x: int, min_y: int, zoom: int) -> _Transform: ...
def decode_tile_window(
    tiles: Sequence[Tuple[int, int, bytes]],
    origin_x: int,
    origin_y: int,
    row_start: int,
    row_stop: int,
    col_start: int,
    col_stop: int,
) -> Tuple[npt.NDArray[np.uint8], npt.NDArray[np.bool_], List[Tuple[int, int]]]: ...
def write_patches(
    image_patches: npt.NDArray[np.uint8],
    mask_patches: Optional[npt.NDArray[np.uint8]],
    meta: Sequence[Tuple[int, int, bool]],
    start_idx: int,
    strip_index: int,
    images_dir: str,
    masks_dir: str,
    image_format: str = "png",
    jpg_quality: int = 95,
    jpg_subsampling: str = "4:2:0",
) -> List[Tuple[str, Optional[str], int, int, bool, int, List[Tuple[int, int]], float]]: ...
def write_geotiffs(
    data: npt.NDArray[np.uint8],
    dtype: str,
    shape: Tuple[int, int, int, int],
    transforms: Sequence[Sequence[float]],
    names: Sequence[str],
    directory: str,
    epsg: int,
    geographic: bool,
    nodata: Optional[float] = None,
    band_names: Optional[Sequence[str]] = None,
    level: Optional[int] = None,
) -> None: ...
def parse_kml(
    data: bytes, label_field: Optional[str] = None
) -> Tuple[List[Tuple[List[_Ring], Optional[str]]], int]: ...
