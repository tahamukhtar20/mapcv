# CLI Reference

The `mapcv` command-line tool exposes four subcommands.

```
Usage: mapcv [OPTIONS] COMMAND [ARGS]...

  Satellite imagery dataset creation tool for segmentation.

Commands:
  generate   Fetch tiles, rasterize labels, extract patches, write dataset.
  split      Split an existing dataset using its manifest.
  validate   Validate a config file without fetching any data.
  init       Scaffold an example config file.
```

---

## `mapcv generate`

Runs the full pipeline: tile download, label rasterization, patch sampling,
and dataset writing.

```bash
mapcv generate CONFIG_PATH
```

**Arguments**

| Argument      | Description                  |
|---------------|------------------------------|
| `CONFIG_PATH` | Path to the YAML config file |

**Example**

```bash
mapcv generate my_dataset.yaml
```

---

## `mapcv split`

Splits an already-generated dataset into train / val / test using its
`manifest.json`. No image files are re-read; only the manifest is processed.

```bash
mapcv split [OPTIONS] STAGING_DIR
```

**Arguments**

| Argument      | Description                                       |
|---------------|---------------------------------------------------|
| `STAGING_DIR` | Directory containing `manifest.json`              |

**Options**

| Option              | Default      | Description                                        |
|---------------------|--------------|----------------------------------------------------|
| `--test-ratio`      | 0.2          | Fraction of all patches reserved for test          |
| `--val-ratio`       | 0.1          | Fraction of train+val pool used for validation     |
| `--labeled-ratios`  | 0.10 0.20 0.30 | Labeled sub-fractions of the train split (repeatable) |
| `--seed`            | 42           | Random seed                                        |
| `--strategy`        | `stratified` | `random` or `stratified`                           |
| `--sample-limit`    | --           | Cap on total patches sampled                       |

**Example**

```bash
mapcv split ./output --test-ratio 0.15 --val-ratio 0.10
```

---

## `mapcv validate`

Parses and validates a config file without downloading any tiles or writing
any files. Useful for catching config errors before a long run.

```bash
mapcv validate CONFIG_PATH
```

---

## `mapcv init`

Scaffolds an annotated example config file. Without an argument it prints to
stdout; with a path argument it writes to that file.

```bash
mapcv init [OUTPUT]
```

**Example**

```bash
# print to terminal
mapcv init

# write to file
mapcv init my_dataset.yaml
```

**Output**

```yaml
region:
  west: 74.20
  south: 31.40
  east: 74.40
  north: 31.60
  zoom: 16

tiles:
  source: google_satellite   # or url_template: "https://..."
  strip_rows: 4
  max_connections: 16
  policy: lenient            # strict | lenient | ignore
  max_failed_ratio: 0.05

# labels:                    # omit for image-only datasets
#   path: labels.kml         # .kml or .geojson
#   label_field: null        # null -> all polygons get class 1
#   all_touched: false

sampler:
  patch_size: 256
  stride: 256                # 0 = same as patch_size (non-overlapping)
  mode: grid                 # grid | random
  edge_strategy: pad         # pad | drop | shift
  pad_mode: zero             # zero | reflect
  max_empty_ratio: 1.0
  min_label_ratio: 0.0

writer:
  staging_dir: ./output
  image_format: png          # png | jpg
  jpg_quality: 95

# split:                     # omit to skip splitting
#   test_ratio: 0.20
#   val_ratio: 0.10
#   labeled_ratios: [0.10, 0.20, 0.30]
#   seed: 42
#   strategy: stratified     # random | stratified
```
