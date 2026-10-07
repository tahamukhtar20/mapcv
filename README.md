<p align="center">
  <img src="https://raw.githubusercontent.com/tahamukhtar20/mapcv/main/assets/logo.svg" alt="mapcv logo" width="180"/>
</p>

<h1 align="center">mapcv</h1>

<p align="center">
    <em>Turn a region, imagery and labels into a ready-to-train remote-sensing dataset. GDAL-free, exact, resumable.</em>
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
    <a href="https://www.bestpractices.dev/projects/15276" target="_blank">
        <img src="https://www.bestpractices.dev/projects/15276/badge" alt="OpenSSF Best Practices">
    </a>
    <a href="https://scorecard.dev/viewer/?uri=github.com/tahamukhtar20/mapcv" target="_blank">
        <img src="https://api.scorecard.dev/projects/github.com/tahamukhtar20/mapcv/badge" alt="OpenSSF Scorecard">
    </a>
    <a href="https://github.com/astral-sh/ruff" target="_blank">
        <img src="https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json" alt="Ruff">
    </a>
</p>

---

<p align="center">
  <img src="https://raw.githubusercontent.com/tahamukhtar20/mapcv/main/website/src/assets/figures/segmentation-patches.webp" alt="Aerial patches over Amsterdam and the same patches with mapcv's building masks" width="760"/>
  <br/><sub>Patches and building masks made by mapcv. Imagery: Beeldmateriaal Nederland (CC BY 4.0); labels: © OpenStreetMap contributors.</sub>
</p>

## Statement of Need

Building a machine-learning dataset from geospatial data usually means a one-off script: fetch or read imagery, burn label polygons, cut patches, split. Done carefully, it handles coordinate systems, pixel-centre rules, NoData, memory, crashes and spatial leakage between train and test; done quickly, it silently gets them wrong. Existing geospatial tools are **analysis-first** or sample data at training time. **mapcv** is **data-creation-first**: one YAML config, and the result is plain files on disk, checked against reference implementations.

- **Imagery:** XYZ tiles, your own GeoTIFF/COG (local, `https://`, `s3://`), Sentinel-2 (EOPF Zarr, or COGs found in a STAC catalog, with cloud masking), Google Earth Engine; several sources on one grid.
- **Labels:** GeoJSON, KML, GeoPackage, Shapefile, GeoParquet, OpenStreetMap, or a label raster.
- **Tasks:** semantic segmentation, object detection (COCO/YOLO), instance segmentation (COCO RLE), patch classification, change detection and regression.
- **Output:** PNG/JPEG/GeoTIFF/NPY patches, a manifest, leakage-safe splits, and exports to Hugging Face and TerraTorch; `mapcv.data.MapcvDataset` loads any of them for PyTorch.
- **Correct and fast:** masks follow GDAL's exact rules; a careful rasterio script, GDAL's tools and leafmap produce the same patches, 1.5–34× more slowly ([benchmarks](https://tahamukhtar20.github.io/mapcv/project/performance/)). Interrupted runs resume byte for byte; memory stays bounded on any region size.

Why not a script, TorchGeo, Raster Vision or geoai? See [Why mapcv](https://tahamukhtar20.github.io/mapcv/why-mapcv/).

## Installation

```bash
pip install mapcv
```

Requires Python 3.10 or newer. One pre-built wheel per platform (Linux x86-64 and aarch64 with glibc or musl, macOS, Windows) covers every supported Python version; other platforms build from source and need a Rust toolchain.

Optional extras: `mapcv[zarr]` (Sentinel-2 EOPF Zarr, Python 3.10–3.13), `mapcv[gee]` (Google Earth Engine), `mapcv[parquet]` (GeoParquet labels), `mapcv[export]` (Hugging Face Parquet) and `mapcv[mcp]` (the MCP server for AI agents). XYZ tiles, GeoTIFF/COG and STAC need none of them.

## Quick start

### 1. Scaffold a config file

```bash
mapcv init my_dataset.yaml
```

Open the file and fill in your region, imagery source, and (optionally) label path. Everything else has sensible defaults.

> **Note:** XYZ sources must return standard **256x256 pixel** tiles. mapcv does not provide imagery; see [PROVIDERS.md](PROVIDERS.md) for how it treats imagery and how to credit each source.

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

## Bring your own imagery

Your own GeoTIFF or Cloud Optimized GeoTIFF (a local file, an `https://` URL, a public `s3://` object, or a folder of tiles read as one mosaic) can be the imagery; no tile server is involved. mapcv reads it without GDAL, keeps its CRS, pixel grid, bands and data type, and reprojects the labels onto it. Patches can be written as PNG/JPEG (8-bit RGB), NPY or georeferenced GeoTIFF.

```yaml
imagery:
  type: geotiff
  path: ortho/scene.tif      # or https://.../scene.tif, s3://bucket/scene.tif, or tiles/*.tif (a mosaic)
  # bands: [1, 2, 3]         # 1-based, default: all
  # overview: 0              # reduced-resolution level
  # nodata: 0                # override the file's NoData
```

Sentinel-2 can come from an EOPF Zarr product (`type: eopf_zarr`) or be found for your region and dates in a STAC catalog (`type: stac_cog`, with SCL cloud masking); Earth Engine images use `imagery.earth_engine`. Every option: [configuration reference](https://tahamukhtar20.github.io/mapcv/reference/configuration/).

## AI agents

`pip install "mapcv[mcp]"` adds `mapcv mcp`, an MCP server that lets Claude Code, Claude Desktop, Cursor or VS Code inspect labels, validate, plan and generate datasets. It is read-only by default, confined to one folder (`--root`), and refuses large jobs until you confirm them. In Claude Code, `/plugin marketplace add tahamukhtar20/mapcv` then `/plugin install mapcv@mapcv` also installs the agent skill. See [Use mapcv with AI agents](https://tahamukhtar20.github.io/mapcv/guides/use-with-ai-agents/).

## Documentation

Full documentation including configuration reference, CLI reference, and API reference is available at **[tahamukhtar20.github.io/mapcv](https://tahamukhtar20.github.io/mapcv)**.

## Contributing

Contributions are welcome. Please review the [Contributing Guide](https://github.com/tahamukhtar20/mapcv/blob/main/CONTRIBUTING.md) and [Code of Conduct](https://github.com/tahamukhtar20/mapcv/blob/main/CODE_OF_CONDUCT.md) before opening a pull request.

## Citation

Every release is archived on Zenodo: [10.5281/zenodo.23149438](https://doi.org/10.5281/zenodo.23149438) resolves to the latest version and lists the DOI of each one. GitHub's "Cite this repository" button gives APA and BibTeX from [`CITATION.cff`](CITATION.cff); see [Citing mapcv](https://tahamukhtar20.github.io/mapcv/project/citing/) for crediting the data sources too.

## Author

**Muhammad Taha Mukhtar** · [tahamukhtar20+mapcv@gmail.com](mailto:tahamukhtar20+mapcv@gmail.com)

## License

This project is licensed under the MIT License — see the [LICENSE](https://github.com/tahamukhtar20/mapcv/blob/main/LICENSE) file for details.
