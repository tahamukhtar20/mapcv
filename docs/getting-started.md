# Getting Started

## Installation

```bash
pip install mapcv
```

Requires Python >= 3.10. Pre-built wheels cover Linux, macOS, and Windows.

### Building from source

You need [Rust](https://rustup.rs/) and `maturin`:

```bash
git clone https://github.com/tahamukhtar20/mapcv
cd mapcv
pip install maturin
maturin develop --release
```

---

## Your first dataset in 3 steps

### 1. Scaffold a config

```bash
mapcv init my_dataset.yaml
```

This writes an annotated example config. Open it and fill in your bounding box,
zoom level, and label file path.

### 2. Edit the config

```yaml title="my_dataset.yaml"
region:
  west:  4.883
  south: 52.371
  east:  4.896
  north: 52.378
  zoom:  17

tiles:
  source: esri_satellite
  max_connections: 16

labels:
  path: labels.geojson      # .kml or .geojson
  label_field: class        # property name for multiclass labels

sampler:
  patch_size: 256
  edge_strategy: drop

writer:
  staging_dir: ./output

split:
  test_ratio: 0.20
  val_ratio:  0.10
```

### 3. Generate the dataset

```bash
mapcv generate my_dataset.yaml
```

When it finishes you will find:

```
output/
  Images/
    patch_0000000.png
    patch_0000001.png
    ...
  Masks/
    patch_0000000.png
    ...
  manifest.json
  train.json
  val.json
  test.json
```

---

## Config reference

### `region`

| Field   | Type  | Description                              |
|---------|-------|------------------------------------------|
| `west`  | float | Western longitude (WGS-84)               |
| `south` | float | Southern latitude (WGS-84)               |
| `east`  | float | Eastern longitude (WGS-84)               |
| `north` | float | Northern latitude (WGS-84)               |
| `zoom`  | int   | XYZ tile zoom level (typically 15 -- 18) |

### `tiles`

| Field              | Type   | Default    | Description                                      |
|--------------------|--------|------------|--------------------------------------------------|
| `source`           | str    | --         | Built-in source name (see below) or omit in favour of `url_template` |
| `url_template`     | str    | --         | Custom XYZ URL, e.g. `https://tile.openstreetmap.org/{z}/{x}/{y}.png` |
| `strip_rows`       | int    | 4          | Tile rows per download strip (memory control)    |
| `max_connections`  | int    | 16         | Parallel HTTP connections                        |
| `policy`           | str    | `lenient`  | `strict` / `lenient` / `ignore`                 |
| `max_failed_ratio` | float  | 0.05       | Abort if this fraction of tiles fail             |

**Built-in sources**

| Name                 | Description              |
|----------------------|--------------------------|
| `esri_satellite`     | Esri World Imagery       |
| `esri_topo`          | Esri World Topo Map      |
| `esri_street`        | Esri World Street Map    |
| `google_satellite`   | Google Satellite         |
| `google_roadmap`     | Google Roadmap           |
| `google_hybrid`      | Google Hybrid            |
| `osm`                | OpenStreetMap            |
| `cartodb_positron`   | CartoDB Positron (light) |
| `cartodb_dark_matter`| CartoDB Dark Matter      |

!!! warning "Tile size"
    Most sources serve 256 x 256 px tiles. Mapbox High-Res serves 512 x 512 px
    tiles -- using it as a source will produce images at double the expected
    resolution.

### `labels` (optional)

Omit this section entirely for image-only (unlabeled) datasets.

| Field         | Type   | Default | Description                                    |
|---------------|--------|---------|------------------------------------------------|
| `path`        | str    | --      | Path to `.kml` or `.geojson` file              |
| `label_field` | str    | null    | Property name for class labels; `null` assigns class 1 to all polygons |
| `all_touched` | bool   | false   | Rasterize all pixels the polygon touches (not just centres) |

### `sampler`

| Field             | Type   | Default | Description                                         |
|-------------------|--------|---------|-----------------------------------------------------|
| `patch_size`      | int    | --      | Square patch side length in pixels                  |
| `stride`          | int    | 0       | Step between patches; `0` = same as `patch_size`    |
| `mode`            | str    | `grid`  | `grid` or `random`                                  |
| `edge_strategy`   | str    | `pad`   | `pad` / `drop` / `shift`                           |
| `pad_mode`        | str    | `zero`  | `zero` or `reflect` (only used when `edge_strategy = pad`) |
| `max_empty_ratio` | float  | 1.0     | Drop patches where this fraction of image pixels are black |
| `min_label_ratio` | float  | 0.0     | Drop patches with fewer labeled pixels than this    |

### `writer`

| Field          | Type   | Default | Description                         |
|----------------|--------|---------|-------------------------------------|
| `staging_dir`  | str    | --      | Output directory                    |
| `image_format` | str    | `png`   | `png` or `jpg`                      |
| `jpg_quality`  | int    | 95      | JPEG quality (1 -- 100)             |

### `split` (optional)

Omit to skip splitting.

| Field             | Type          | Default             | Description                            |
|-------------------|---------------|---------------------|----------------------------------------|
| `test_ratio`      | float         | 0.20                | Fraction reserved for test set         |
| `val_ratio`       | float         | 0.10                | Fraction of remainder used for val     |
| `labeled_ratios`  | list[float]   | [0.10, 0.20, 0.30]  | Labeled sub-fractions of train split   |
| `seed`            | int           | 42                  | Random seed                            |
| `strategy`        | str           | `stratified`        | `random` or `stratified`               |

---

## Python API quick reference

```python
from mapcv import (
    # Downloading
    stitch_region,
    download_region,
    download_region_strips,

    # Labels
    parse_kml,
    parse_geojson,
    transform_to_mercator,

    # Rasterization
    rasterize,

    # Patch sampling
    sample_patches,
    SamplerConfig,

    # Writing
    write_patches,
    WriterConfig,
    Manifest,
    load_or_create_manifest,

    # Splitting
    split_dataset,
    SplitterConfig,
)
```

See the [API Reference](api.md) for full signatures and descriptions.
