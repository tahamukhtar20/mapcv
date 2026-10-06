<p align="center">
  <img src="https://raw.githubusercontent.com/tahamukhtar20/mapcv/main/assets/logo.svg" alt="mapcv logo" width="180"/>
</p>

<h1 align="center">mapcv</h1>

<p align="center">
    <em>A high-performance satellite imagery dataset creation tool for computer vision.</em>
</p>

<p align="center">
    <a href="https://pypi.org/project/mapcv/" target="_blank">
        <img src="https://img.shields.io/pypi/v/mapcv?color=%2334D058&label=pypi%20package" alt="Package version">
    </a>
    <a href="https://pypi.org/project/mapcv/" target="_blank">
        <img src="https://img.shields.io/pypi/pyversions/mapcv.svg?color=%2334D058" alt="Supported Python versions">
    </a>
    <a href="https://github.com/tahamukhtar20/mapcv/blob/main/LICENSE" target="_blank">
        <img src="https://img.shields.io/github/license/tahamukhtar20/mapcv.svg?color=%2334D058" alt="License">
    </a>
    <a href="https://pepy.tech/project/mapcv" target="_blank">
        <img src="https://static.pepy.tech/badge/mapcv" alt="Downloads">
    </a>
    <a href="https://tahamukhtar20.github.io/mapcv/" target="_blank">
        <img src="https://img.shields.io/badge/docs-online-blue" alt="Documentation">
    </a>
    <a href="https://scorecard.dev/viewer/?uri=github.com/tahamukhtar20/mapcv" target="_blank">
        <img src="https://api.scorecard.dev/projects/github.com/tahamukhtar20/mapcv/badge" alt="OpenSSF Scorecard">
    </a>
    <a href="https://github.com/astral-sh/ruff" target="_blank">
        <img src="https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json" alt="Ruff">
    </a>
</p>

---

## Statement of Need

Creating machine learning datasets from satellite imagery is traditionally a frustrating experience. Wrestling with heavy, notoriously complex GIS libraries like GDAL is a massive pain point for researchers who just want to train models.

Existing geospatial ecosystems are heavily **analysis-first**. **mapcv** is different. It is explicitly designed as a **data creation-first** tool. It provides a fast, end-to-end pipeline written in Python and Rust specifically optimized for fetching map tiles, rasterizing complex labels (GeoJSON, KML, GeoPackage, Shapefile, GeoParquet), and splitting areas into uniform, ML-ready patches. The target audience includes computer vision researchers, data scientists, and ML engineers who need an efficient and reliable way to prepare high-quality satellite datasets for training segmentation models without the traditional GIS headaches.

## Installation

```bash
pip install mapcv
```

Requires Python 3.10 or newer. One pre-built wheel per platform (Linux x86-64 and aarch64 with glibc or musl, macOS, Windows) covers every supported Python version; other platforms build from source and need a Rust toolchain.

EOPF Sentinel-2 L2A Zarr support is optional:

```bash
pip install "mapcv[zarr]"
```

The `zarr` extra currently supports Python 3.10–3.13: its zarr 2 dependency has no Python 3.14 wheels yet.

## Quick start

### 1. Scaffold a config file

```bash
mapcv init my_dataset.yaml
```

Open the file and fill in your region, imagery source, and (optionally) label path. Everything else has sensible defaults.

> **Note:** XYZ sources must return standard **256x256 pixel** tiles. You are responsible for complying with the imagery provider's license, attribution, rate limits, and terms. mapcv does not grant imagery rights.

### 2. Generate the dataset

```bash
mapcv generate my_dataset.yaml
```

This fetches tiles, rasterizes labels, extracts patches, and writes everything to the output directory specified in your config.

### 3. Re-split an existing dataset (optional)

```bash
mapcv split ./output --test-ratio 0.15 --val-ratio 0.10
```

Re-runs the train/val/test split from the existing `manifest.json` without re-downloading anything.

### 4. Prepare it for training and sharing (optional)

```bash
mapcv stats ./output     # per-band mean/std and class weights over train → stats.json
mapcv card ./output      # README.md dataset card with Hugging Face metadata
mapcv verify ./output --write-checksums
```

## EOPF Sentinel-2 Zarr

Version 0.2.0 can read one local or anonymous public EOPF Sentinel-2 L2A product per run. It preserves the product's projected CRS, harmonizes selected bands with the EOPF backend, and writes bands-first `float32` NPY patches.

```yaml
region:
  west: 10.0
  south: 45.0
  east: 10.2
  north: 45.2

imagery:
  type: eopf_zarr
  path: /data/S2_L2A_PRODUCT.zarr
  resolution: 10
  bands: [b01, b02, b03, b04, b05, b06, b07, b08, b8a, b09, b11, b12]
  chunk_rows: 1024

sampler:
  patch_size: 256

writer:
  staging_dir: ./dataset
  image_format: npy
```

Private-store credentials, STAC discovery, mosaicking, cloud masking, GeoTIFF output, and Google Earth Engine integration are outside the 0.2.0 scope. See the [migration guide](https://github.com/tahamukhtar20/mapcv/blob/main/MIGRATION.md) and [provider guidance](https://github.com/tahamukhtar20/mapcv/blob/main/PROVIDERS.md).

## GeoTIFF and COG imagery

Your own GeoTIFF or Cloud Optimized GeoTIFF (a local file, an `https://` URL or a public `s3://` object) can be the imagery. mapcv reads it without GDAL, keeps its CRS, pixel grid, bands and data type, and reprojects WGS-84 labels onto it. 8-bit RGB can be written as PNG/JPG; any other layout needs `image_format: npy`.

```yaml
imagery:
  type: geotiff
  path: ortho/scene.tif      # or https://.../scene.tif
  # bands: [1, 2, 3]         # 1-based, default: all
  # overview: 0              # reduced-resolution level
  # nodata: 0                # override the file's NoData
```

## AI agents

`pip install "mapcv[mcp]"` adds `mapcv mcp`, an MCP server that lets Claude Code, Claude Desktop, Cursor or VS Code inspect labels, validate, plan and generate datasets. It is read-only by default, confined to one folder (`--root`), and refuses large jobs until you confirm them. In Claude Code, `/plugin marketplace add tahamukhtar20/mapcv` then `/plugin install mapcv@mapcv` also installs the agent skill. See [Use mapcv with AI agents](https://tahamukhtar20.github.io/mapcv/guides/use-with-ai-agents/).

## Documentation

Full documentation including configuration reference, CLI reference, and API reference is available at **[tahamukhtar20.github.io/mapcv](https://tahamukhtar20.github.io/mapcv)**.

## Contributing

Contributions are welcome. Please review the [Contributing Guide](https://github.com/tahamukhtar20/mapcv/blob/main/CONTRIBUTING.md) and [Code of Conduct](https://github.com/tahamukhtar20/mapcv/blob/main/CODE_OF_CONDUCT.md) before opening a pull request.

## Citation

*(Citation information will be added after publication.)*

## Author

**Muhammad Taha Mukhtar** · [tahamukhtar20+mapcv@gmail.com](mailto:tahamukhtar20+mapcv@gmail.com)

## License

This project is licensed under the MIT License — see the [LICENSE](https://github.com/tahamukhtar20/mapcv/blob/main/LICENSE) file for details.
