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
</p>

---

## Statement of Need

Creating machine learning datasets from satellite imagery is traditionally a frustrating experience. Wrestling with heavy, notoriously complex GIS libraries like GDAL is a massive pain point for researchers who just want to train models. Furthermore, trying to extract data from platforms like Google Earth Engine (GEE) often forces users into clunky pipelines, such as having to export data to Google Drive first before finally downloading it locally.

Existing geospatial ecosystems are heavily **analysis-first**. **mapcv** is different. It is explicitly designed as a **data creation-first** tool. It provides a blazingly fast, end-to-end pipeline written in Python and Rust specifically optimized for fetching map tiles, rasterizing complex labels (KML/GeoJSON), and seamlessly splitting areas into uniform, ML-ready patches. The target audience includes computer vision researchers, data scientists, and ML engineers who need an efficient and reliable way to prepare high-quality satellite datasets for training segmentation models without the traditional GIS headaches.

## Installation

You can install `mapcv` directly from PyPI using pip:

```bash
pip install mapcv
```

> **Note**: `mapcv` requires Python 3.10 or higher.

## Quick-start

`mapcv` provides a simple command-line interface to build your datasets.

### 1. Initialize a Project
Create a configuration file in your directory:
```bash
mapcv init
```
This generates a `mapcv.yaml` file where you can define your bounding box, zoom level, and label files (e.g., `labels.kml`).

> **Warning:** `mapcv` currently only supports tile sources that serve standard **256x256 pixel** map tiles (which is the default for Google Earth Engine, OpenStreetMap, Esri, and most providers). Sources that serve 512x512 tiles (such as **Mapbox High-Res `@2x` endpoints**) or others, are not supported yet, and will result in incorrectly scaled patches.

### 2. Generate the Dataset
Fetch the tiles, rasterize the labels, and generate image/mask patches:
```bash
mapcv generate --config mapcv.yaml
```

### 3. Split the Dataset
Split the generated patches into Train, Validation, and Test sets:
```bash
mapcv split --config mapcv.yaml
```

## Documentation

Full documentation, including API references and advanced usage tutorials, is coming soon.

## Community & Contributing

We welcome contributions! Please review our:
- [Contributing Guide](CONTRIBUTING.md) for information on setting up your development environment and submitting pull requests.
- [Code of Conduct](CODE_OF_CONDUCT.md) to understand our community standards.

## Citation

*(Citation information will be added later)*

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
