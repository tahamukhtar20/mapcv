<p align="center">
  <img src="assets/logo.svg" alt="mapcv logo" width="180"/>
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
    <a href="https://github.com/astral-sh/ruff" target="_blank">
        <img src="https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json" alt="Ruff">
    </a>
</p>

---

## Statement of Need

Creating machine learning datasets from satellite imagery is traditionally a frustrating experience. Wrestling with heavy, notoriously complex GIS libraries like GDAL is a massive pain point for researchers who just want to train models.

Existing geospatial ecosystems are heavily **analysis-first**. **mapcv** is different. It is explicitly designed as a **data creation-first** tool. It provides a fast, end-to-end pipeline written in Python and Rust specifically optimized for fetching map tiles, rasterizing complex labels (KML/GeoJSON), and splitting areas into uniform, ML-ready patches. The target audience includes computer vision researchers, data scientists, and ML engineers who need an efficient and reliable way to prepare high-quality satellite datasets for training segmentation models without the traditional GIS headaches.

## Installation

```bash
pip install mapcv
```

Requires Python 3.10 or higher. Pre-built wheels cover Linux, macOS, and Windows.

## Quick start

### 1. Scaffold a config file

```bash
mapcv init my_dataset.yaml
```

Open the file and fill in your region, tile source, and (optionally) label path. Everything else has sensible defaults.

> **Note:** mapcv supports tile sources that serve standard **256x256 pixel** tiles (OpenStreetMap, Esri, Google Satellite, CartoDB, and most providers). Sources serving 512x512 tiles are not supported and will produce incorrectly scaled patches.

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

## Documentation

Full documentation including configuration reference, CLI reference, and API reference is available at **[tahamukhtar20.github.io/mapcv](https://tahamukhtar20.github.io/mapcv)**.

## Contributing

Contributions are welcome. Please review the [Contributing Guide](CONTRIBUTING.md) and [Code of Conduct](CODE_OF_CONDUCT.md) before opening a pull request.

## Citation

*(Citation information will be added after publication.)*

## License

This project is licensed under the MIT License — see the [LICENSE](LICENSE) file for details.
