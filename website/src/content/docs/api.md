---
title: API Reference
description: Full Python API reference for mapcv.
---

All public symbols are importable directly from `mapcv`:

```python
from mapcv import stitch_region, parse_geojson, rasterize, sample_patches, ...
```

---

## Downloading

### `stitch_region`

```python
mapcv.stitch_region(
    west: float,
    south: float,
    east: float,
    north: float,
    zoom: int,
    url_template: str | None = None,
    source: str | None = None,
    max_connections: int = 16,
    policy: str = "lenient",
    max_failed_ratio: float = 0.05,
) -> tuple[np.ndarray, tuple[float, float, float, float, float, float]]
```

Fetch all tiles for a bounding box, decode in parallel, stitch into one image, and return `(image, transform)`.

**Parameters**

- **`west`** (`float`) - Western boundary longitude (WGS-84).
- **`south`** (`float`) - Southern boundary latitude (WGS-84).
- **`east`** (`float`) - Eastern boundary longitude (WGS-84).
- **`north`** (`float`) - Northern boundary latitude (WGS-84).
- **`zoom`** (`int`) - XYZ tile zoom level.
- **`url_template`** (`str`, optional) - Custom XYZ URL with `{z}`, `{x}`, `{y}` placeholders. Mutually exclusive with `source`.
- **`source`** (`str`, optional) - Built-in source name. Mutually exclusive with `url_template`.
- **`max_connections`** (`int`, optional, defaults to `16`) - Number of parallel tile fetches.
- **`policy`** (`str`, optional, defaults to `"lenient"`) - Failure policy: `"strict"`, `"lenient"`, or `"ignore"`.
- **`max_failed_ratio`** (`float`, optional, defaults to `0.05`) - Abort if this fraction of tiles fail. Only used when `policy="lenient"`.

**Returns**

`(image, transform)` where:
- `image` is a `(H, W, 3)` `uint8` NumPy array.
- `transform` is the affine 6-tuple `(a, b, c, d, e, f)` mapping pixel `(col, row)` to Web Mercator `(x, y)` in metres.

The bounding box is snapped outward to tile boundaries before fetching.

**Example**

```python
import mapcv

image, transform = mapcv.stitch_region(
    west=4.883, south=52.371, east=4.896, north=52.378,
    zoom=17,
    source="esri_satellite",
)
print(image.shape)  # (1280, 1536, 3)
```

---

### `download_region`

```python
mapcv.download_region(
    west: float,
    south: float,
    east: float,
    north: float,
    zoom: int,
    url_template: str | None = None,
    source: str | None = None,
    max_connections: int = 16,
    policy: str = "lenient",
    snap_to_tiles: bool = True,
    max_failed_ratio: float = 0.05,
) -> list[tuple[PyTileIndex, bytes]]
```

Lower-level fetch that returns raw `(tile_index, png_bytes)` pairs without stitching.

**Parameters**

Same as `stitch_region`, plus:

- **`snap_to_tiles`** (`bool`, optional, defaults to `True`) - Expand the bbox outward to tile boundaries before fetching.

**Returns**

`list[tuple[PyTileIndex, bytes]]`, one entry per tile.

---

### `download_region_strips`

```python
mapcv.download_region_strips(
    west: float,
    south: float,
    east: float,
    north: float,
    zoom: int,
    strip_rows: int,
    url_template: str | None = None,
    source: str | None = None,
    max_connections: int = 16,
    policy: str = "lenient",
    snap_to_tiles: bool = True,
    max_failed_ratio: float = 0.05,
) -> list[list[tuple[PyTileIndex, bytes]]]
```

Like `download_region` but divides the tile grid into horizontal strips of `strip_rows` tile rows. Tiles shared between strips are cached and fetched only once.

**Parameters**

Same as `download_region`, plus:

- **`strip_rows`** (`int`) - Number of tile rows per strip.

**Returns**

`list[list[tuple[PyTileIndex, bytes]]]`, one inner list per strip.

---

## Labels

### `parse_geojson`

```python
mapcv.parse_geojson(
    data: bytes,
    label_field: str | None = None,
) -> tuple[list[tuple[Geometry, int]], dict[str, int]]
```

Parse GeoJSON bytes into `(geometry, class_id)` pairs.

**Parameters**

- **`data`** (`bytes`) - Raw GeoJSON bytes (FeatureCollection or single Feature).
- **`label_field`** (`str`, optional, defaults to `None`) - Property name used for class labels. If `None`, all polygons receive `class_id = 1`.

**Returns**

`(geometries, class_map)` where `class_map` maps each class name to its integer ID (assigned by encounter order, starting at 1). Non-polygon geometries are silently skipped.

**Example**

```python
data = Path("labels.geojson").read_bytes()
geoms, class_map = mapcv.parse_geojson(data, label_field="class")
# class_map: {"residential": 1, "water": 2, "park": 3}
```

---

### `parse_kml`

```python
mapcv.parse_kml(
    data: bytes,
    label_field: str | None = None,
) -> tuple[list[tuple[Geometry, int]], dict[str, int]]
```

Same interface as `parse_geojson` but accepts KML bytes. Supports nested `<Folder>` elements and `<MultiGeometry>`.

---

### `transform_to_mercator`

```python
mapcv.transform_to_mercator(geom: BaseGeometry) -> BaseGeometry
```

Project a Shapely geometry from WGS-84 (EPSG:4326) to Web Mercator (EPSG:3857).

**Parameters**

- **`geom`** (`BaseGeometry`) - A Shapely geometry in WGS-84.

**Returns**

The same geometry reprojected to Web Mercator. Required before passing geometries to `rasterize`.

---

## Rasterization

### `rasterize`

```python
mapcv.rasterize(
    geometries: Sequence[tuple[BaseGeometry, int]],
    out_shape: tuple[int, int],
    transform: tuple[float, float, float, float, float, float],
    all_touched: bool = False,
) -> np.ndarray
```

Burn `(geometry, class_id)` pairs into a `uint8` mask.

**Parameters**

- **`geometries`** (`Sequence[tuple[BaseGeometry, int]]`) - Pairs of Web Mercator geometry and class ID. Geometries later in the list overwrite earlier ones.
- **`out_shape`** (`tuple[int, int]`) - Output mask shape as `(height, width)`.
- **`transform`** (`tuple[float, float, float, float, float, float]`) - Affine 6-tuple mapping pixel `(col, row)` to Web Mercator `(x, y)`. Use the transform returned by `stitch_region`.
- **`all_touched`** (`bool`, optional, defaults to `False`) - If `True`, all pixels touched by the polygon boundary are burned, not just pixel centres.

**Returns**

`np.ndarray` of shape `(H, W)` and dtype `uint8`. Background pixels are `0`; class pixels are `1..255`.

**Example**

```python
merc_geoms = [(mapcv.transform_to_mercator(g), cid) for g, cid in geoms]
mask = mapcv.rasterize(merc_geoms, out_shape=(H, W), transform=transform)
```

---

## Patch Sampling

### `SamplerConfig`

```python
class mapcv.SamplerConfig(
    patch_size: int,
    stride: int = 0,
    mode: str = "grid",
    edge_strategy: str = "pad",
    pad_mode: str = "zero",
    max_empty_ratio: float = 1.0,
    min_label_ratio: float = 0.0,
    random_seed: int = 42,
    random_count: int = 100,
)
```

**Parameters**

- **`patch_size`** (`int`) - Side length of each square patch in pixels.
- **`stride`** (`int`, optional, defaults to `0`) - Step between patch centres. `0` uses `patch_size` as stride.
- **`mode`** (`str`, optional, defaults to `"grid"`) - `"grid"` scans row-by-row. `"random"` draws random positions - requires `random_count`.
- **`edge_strategy`** (`str`, optional, defaults to `"pad"`) - `"pad"` fills incomplete edge patches. `"drop"` discards them. `"shift"` slides the last patch inward.
- **`pad_mode`** (`str`, optional, defaults to `"zero"`) - `"zero"` fills with black. `"reflect"` mirrors the border. Only used when `edge_strategy="pad"`.
- **`max_empty_ratio`** (`float`, optional, defaults to `1.0`) - Discard patches with more than this fraction of black pixels.
- **`min_label_ratio`** (`float`, optional, defaults to `0.0`) - Discard patches with less than this fraction of labeled pixels.
- **`random_seed`** (`int`, optional, defaults to `42`) - Seed for reproducible random sampling. Only used when `mode="random"`.
- **`random_count`** (`int`, optional, defaults to `100`) - Number of random patches to draw. Only used when `mode="random"`.

---

### `sample_patches`

```python
mapcv.sample_patches(
    strip_image: np.ndarray,
    strip_mask: np.ndarray | None,
    config: SamplerConfig,
) -> tuple[np.ndarray, np.ndarray | None, list[dict]]
```

Extract fixed-size patches from `strip_image` and optionally `strip_mask`.

**Parameters**

- **`strip_image`** (`np.ndarray`) - `(H, W, 3)` `uint8` array.
- **`strip_mask`** (`np.ndarray | None`) - `(H, W)` `uint8` array, or `None` for image-only datasets.
- **`config`** (`SamplerConfig`) - Sampling configuration.

**Returns**

`(patch_images, patch_masks, meta)` where:
- `patch_images`: `(N, patch_size, patch_size, 3)` `uint8`
- `patch_masks`: `(N, patch_size, patch_size)` `uint8`, or `None` if no mask given
- `meta`: list of dicts with keys `row`, `col`, `padded`

---

## Writing

### `WriterConfig`

```python
class mapcv.WriterConfig(
    staging_dir: Path,
    image_format: str = "png",
    jpg_quality: int = 95,
)
```

**Parameters**

- **`staging_dir`** (`Path`) - Directory where `Images/`, `Masks/`, and `manifest.json` are written.
- **`image_format`** (`str`, optional, defaults to `"png"`) - `"png"` for lossless, `"jpg"` for smaller lossy files.
- **`jpg_quality`** (`int`, optional, defaults to `95`) - JPEG quality from 1 to 100. Only used when `image_format="jpg"`.

---

### `Manifest`

```python
class mapcv.Manifest(
    version: int,
    class_map: dict[str, int],
    patches: list[ManifestEntry],
)
```

**Methods**

- **`save(path: Path) -> None`** - Write the manifest to disk as JSON.
- **`Manifest.load(path: Path) -> Manifest`** - Load a manifest from a JSON file.

---

### `ManifestEntry`

Each entry in `manifest.patches` is a `TypedDict` with the following fields:

| Field | Type | Description |
|---|---|---|
| `filename` | `str` | Relative path to the patch image inside the staging directory. |
| `mask_filename` | `str \| None` | Relative path to the mask image, or `None` for unlabeled datasets. |
| `row` | `int` | Top-left row of the patch in the stitched strip image. |
| `col` | `int` | Top-left column of the patch in the stitched strip image. |
| `padded` | `bool` | `True` if the patch was zero-padded to reach `patch_size`. |
| `strip_index` | `int` | Index of the strip this patch was sampled from. |
| `per_class_pixel_counts` | `dict[str, int]` | Pixel count for each class ID string (e.g. `{"1": 1024, "2": 512}`). Empty for unlabeled patches. |
| `empty_ratio` | `float` | Fraction of black (zero) pixels in the image patch. |

---

### `load_or_create_manifest`

```python
mapcv.load_or_create_manifest(
    path: Path,
    class_map: dict[str, int],
) -> Manifest
```

Load an existing manifest from `path`, or create a new one with `class_map` if the file does not exist.

---

### `write_patches`

```python
mapcv.write_patches(
    image_patches: np.ndarray,
    mask_patches: np.ndarray | None,
    meta: list[dict],
    config: WriterConfig,
    manifest: Manifest,
    strip_index: int = 0,
) -> None
```

Write patch images and masks to disk and append entries to `manifest`. Skips files that already exist (resume-safe).

**Parameters**

- **`image_patches`** (`np.ndarray`) - `(N, patch_size, patch_size, 3)` `uint8` array.
- **`mask_patches`** (`np.ndarray | None`) - `(N, patch_size, patch_size)` `uint8` array, or `None`.
- **`meta`** (`list[dict]`) - Patch metadata from `sample_patches`.
- **`config`** (`WriterConfig`) - Writer configuration.
- **`manifest`** (`Manifest`) - Manifest to append patch entries to.
- **`strip_index`** (`int`, optional, defaults to `0`) - Strip index, used to generate unique patch filenames across strips.

---

## Splitting

### `SplitterConfig`

```python
class mapcv.SplitterConfig(
    test_ratio: float = 0.20,
    val_ratio: float = 0.10,
    labeled_ratios: list[float] = [0.10, 0.20, 0.30],
    seed: int = 42,
    strategy: str = "stratified",
    sample_limit: int | None = None,
)
```

**Parameters**

- **`test_ratio`** (`float`, optional, defaults to `0.20`) - Fraction of patches reserved for the test set.
- **`val_ratio`** (`float`, optional, defaults to `0.10`) - Fraction of non-test patches used for validation.
- **`labeled_ratios`** (`list[float]`, optional, defaults to `[0.10, 0.20, 0.30]`) - For each ratio, writes `<ratio>/labeled.txt` and `<ratio>/unlabeled.txt` for semi-supervised workflows.
- **`seed`** (`int`, optional, defaults to `42`) - Random seed for reproducible splits.
- **`strategy`** (`str`, optional, defaults to `"stratified"`) - `"stratified"` preserves class distribution. `"random"` splits without balancing.
- **`sample_limit`** (`int`, optional) - Cap on total patches sampled before splitting.

---

### `split_dataset`

```python
mapcv.split_dataset(
    manifest: Manifest,
    config: SplitterConfig,
    output_dir: Path,
) -> None
```

Split the manifest into `train.txt`, `val.txt`, and `test.txt` files written to `output_dir`.

**Parameters**

- **`manifest`** (`Manifest`) - The dataset manifest to split.
- **`config`** (`SplitterConfig`) - Split configuration.
- **`output_dir`** (`Path`) - Directory where split `.txt` files are written.

---

## Built-in tile sources

```python
from mapcv import URL_TEMPLATES
print(list(URL_TEMPLATES.keys()))
# ['google_satellite', 'osm', 'esri_satellite', 'esri_topo',
#  'esri_street', 'cartodb_positron', 'cartodb_dark_matter']
```
