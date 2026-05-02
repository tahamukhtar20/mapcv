# API Reference

All public symbols are importable directly from `mapcv`:

```python
import mapcv
# or
from mapcv import stitch_region, parse_geojson, rasterize, ...
```

---

## Downloading

### `stitch_region`

```python
def stitch_region(
    west: float, south: float, east: float, north: float,
    zoom: int,
    url_template: str | None = None,
    source: str | None = None,
    max_connections: int = 16,
    policy: str = "lenient",
    max_failed_ratio: float = 0.05,
) -> tuple[np.ndarray, tuple[float, float, float, float, float, float]]
```

Fetch all tiles for a bounding box, decode them in parallel, stitch into
one image, and return `(image, transform)`.

- `image` is a `(H, W, 3)` uint8 NumPy array.
- `transform` is the affine 6-tuple `(a, b, c, d, e, f)` mapping pixel
  `(col, row)` to Web Mercator `(x, y)` in metres (rasterio convention).

The bbox is automatically snapped outward to tile boundaries before fetching.

---

### `download_region`

```python
def download_region(
    west, south, east, north, zoom,
    url_template=None, source=None,
    max_connections=16, policy="lenient",
    snap_to_tiles=True, max_failed_ratio=0.05,
) -> list[tuple[PyTileIndex, bytes]]
```

Lower-level: fetch tiles and return raw `(tile_index, png_bytes)` pairs
without stitching.

---

### `download_region_strips`

```python
def download_region_strips(
    west, south, east, north, zoom, strip_rows,
    url_template=None, source=None,
    max_connections=16, policy="lenient",
    snap_to_tiles=True, max_failed_ratio=0.05,
) -> list[list[tuple[PyTileIndex, bytes]]]
```

Like `download_region` but splits the tile grid into horizontal strips of
`strip_rows` tile rows each. Tiles shared between strips are cached and
fetched only once. Returns one inner list per strip.

Use this for large regions that would not fit in RAM if stitched all at once.

---

## Labels

### `parse_geojson`

```python
def parse_geojson(
    data: bytes,
    label_field: str | None = None,
) -> tuple[list[tuple[Geometry, int]], dict[str, int]]
```

Parse GeoJSON bytes (FeatureCollection or single Feature) into
`(geometry, class_id)` pairs.

- `label_field`: property name used for class labels. If `None`, all
  polygons receive `class_id = 1`.
- Returns `(geometries, class_map)` where `class_map` maps each class name
  string to its assigned integer ID (assigned by encounter order, starting
  at 1).

Non-polygon geometries are silently skipped.

---

### `parse_kml`

```python
def parse_kml(
    data: bytes,
    label_field: str | None = None,
) -> tuple[list[tuple[Geometry, int]], dict[str, int]]
```

Same interface as `parse_geojson` but accepts KML bytes.
Supports nested `<Folder>` elements and `<MultiGeometry>`.

---

### `transform_to_mercator`

```python
def transform_to_mercator(geom: BaseGeometry) -> BaseGeometry
```

Project a Shapely geometry from WGS-84 (EPSG:4326) to Web Mercator
(EPSG:3857). Required before passing geometries to `rasterize`.

---

## Rasterization

### `rasterize`

```python
def rasterize(
    geometries: Sequence[tuple[BaseGeometry, int]],
    out_shape: tuple[int, int],
    transform: tuple[float, float, float, float, float, float],
    all_touched: bool = False,
) -> np.ndarray  # uint8, shape (H, W)
```

Burn `(geometry, class_id)` pairs into a uint8 mask of shape `(height, width)`.

- Background pixels are `0`.
- Geometries later in the list overwrite earlier ones.
- `transform` must be in Web Mercator (use `transform_to_mercator` first).
- `class_id` must be in `1..=255`.

---

## Patch sampling

### `sample_patches`

```python
def sample_patches(
    image: np.ndarray,   # (H, W, 3) uint8
    mask: np.ndarray | None,  # (H, W) uint8 or None
    config: SamplerConfig,
) -> tuple[np.ndarray, np.ndarray | None, list[dict]]
```

Extract patches from `image` (and optionally `mask`) according to `config`.

Returns `(patch_images, patch_masks, meta)`:

- `patch_images`: `(N, patch_size, patch_size, 3)` uint8
- `patch_masks`: `(N, patch_size, patch_size)` uint8, or `None` if no mask given
- `meta`: list of dicts with keys `row`, `col`, `padded`, `empty_ratio`,
  `per_class_pixel_counts`

---

### `SamplerConfig`

```python
class SamplerConfig(BaseModel):
    patch_size: int
    stride: int = 0             # 0 = same as patch_size
    mode: str = "grid"          # "grid" | "random"
    edge_strategy: str = "pad"  # "pad" | "drop" | "shift"
    pad_mode: str = "zero"      # "zero" | "reflect"
    max_empty_ratio: float = 1.0
    min_label_ratio: float = 0.0
    random_count: int | None = None
    random_seed: int | None = None
```

---

## Writing

### `write_patches`

```python
def write_patches(
    images: np.ndarray,
    masks: np.ndarray | None,
    meta: list[dict],
    config: WriterConfig,
    manifest: Manifest,
    strip_index: int | None = None,
) -> None
```

Write patch images (and masks) to disk and append entries to `manifest`.
Skips files that already exist on disk (resume-safe).

---

### `WriterConfig`

```python
class WriterConfig(BaseModel):
    staging_dir: Path
    image_format: str = "png"   # "png" | "jpg"
    jpg_quality: int = 95       # 1 -- 100
```

---

### `Manifest`

```python
class Manifest:
    version: int
    class_map: dict[str, int]
    patches: list[ManifestEntry]

    def save(self, path: Path) -> None: ...

    @classmethod
    def load(cls, path: Path) -> "Manifest": ...
```

---

### `load_or_create_manifest`

```python
def load_or_create_manifest(
    path: Path,
    class_map: dict[str, int],
) -> Manifest
```

Load an existing manifest from `path`, or create a new one with `class_map`
if the file does not exist.

---

## Splitting

### `split_dataset`

```python
def split_dataset(
    manifest: Manifest,
    config: SplitterConfig,
    staging_dir: Path,
) -> None
```

Split the manifest into train / val / test JSON files written to `staging_dir`.

---

### `SplitterConfig`

```python
class SplitterConfig(BaseModel):
    test_ratio: float = 0.20
    val_ratio: float = 0.10
    labeled_ratios: list[float] = [0.10, 0.20, 0.30]
    seed: int = 42
    strategy: str = "stratified"  # "random" | "stratified"
    sample_limit: int | None = None
```

---

## Built-in tile sources

```python
from mapcv import URL_TEMPLATES
print(list(URL_TEMPLATES.keys()))
# ['google_satellite', 'osm', 'esri_satellite', 'esri_topo',
#  'esri_street', 'cartodb_positron', 'cartodb_dark_matter']
```
