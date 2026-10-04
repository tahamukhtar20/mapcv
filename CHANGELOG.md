# Changelog

All notable changes to this project will be documented in this file.

## [0.2.0] - 2026-10-04

mapcv 0.2 adds Sentinel-2 imagery, leakage-safe dataset splits and Python 3.14 support, and fixes several ways 0.1 could silently produce wrong datasets. See [MIGRATION.md](https://github.com/tahamukhtar20/mapcv/blob/main/MIGRATION.md) before upgrading: configs and datasets need small changes.

### Highlights

- **EOPF Sentinel-2 L2A Zarr input** (`pip install "mapcv[zarr]"`): lazy windowed reads, band selection, 10/20/60 m harmonization on the product's own pixel grid, labels aligned in the native UTM CRS, and bands-first `float32` NPY patches.
- **Leakage-safe splits by default.** The new `spatial` strategy keeps whole raster blocks in one split and leaves out train/val patches that overlap a held-out patch, so test scores are not inflated by shared pixels.
- **Python 3.14 and one wheel per platform.** Stable-ABI (abi3) wheels for Linux x86-64, macOS and Windows cover Python 3.10 and newer.
- **Bounded memory for large regions.** XYZ tiles are fetched per chunk instead of all at once; labels and resume compatibility are checked before the first download, and a resumed run fetches only the chunks it still needs. Labels are reprojected in one pass and each chunk rasterizes only the polygons near it.
- **Stable class IDs.** Integer labels are used as mask values, other labels are numbered in sorted order, and `labels.classes` pins IDs explicitly.
- **A CLI that walks you through it.** `mapcv init` asks about imagery, area, zoom and labels and writes a commented config; `mapcv plan` estimates tiles, patches, download size and memory without downloading; `generate` previews the plan, asks before large jobs and ends with a summary of files, classes and splits; `mapcv info` describes an existing dataset.
- **Manifest version 2** records the source, product, bands, dtype, patch shape, CRS, affine transform and sampler settings.

### Breaking changes

- New `imagery:` block (`type: xyz` or `eopf_zarr`); `imagery.type` is required.
- Unknown config keys are errors instead of being ignored, and `source` and `url_template` cannot both be set.
- Relative paths in a config resolve against the config file's folder, not the working directory.
- `mapcv init` writes `mapcv.yaml` unless given a path; `--stdout` prints the config instead.
- The built-in `google_satellite` and `osm` tile presets are removed (provider terms forbid bulk downloading); configs using them fail validation with an explanation.
- The default split strategy is `spatial` instead of `stratified`.
- Class IDs no longer follow feature order; set `labels.classes` to reproduce 0.1 IDs.
- `sampler.mode: random` draws `random_count` patches in total rather than per strip, and grid patches may cross former strip seams.
- 0.1 datasets (manifest v1) can still be split but not resumed; resuming any dataset requires the same imagery, label file, label settings, sampler and writer settings.
- `max_failed_ratio` applies to each chunk of tiles rather than the whole region.
- Removed Python APIs: `download_region_strips`, `iter_tile_strips`, `TilesConfig` and `hello()`. `sample_patches` remains, and its metadata now includes `empty_ratio`.
- The `zarr` extra supports Python 3.10–3.13 until its zarr 2 dependency ships Python 3.14 wheels.

### Deprecations

- `region.zoom` and the `tiles` block are accepted with a deprecation notice and will be removed in 0.3.0.

### Fixes

**Datasets and resume**
- Save the manifest after every chunk and overwrite orphaned files from interrupted runs, so a resume can no longer pair entries with the wrong files.
- Count black or failed XYZ tiles as empty for `max_empty_ratio` again.
- Refuse to resume when the label file or the writer settings changed, and refuse manifests written by a newer mapcv.
- A resumed run records the same chunk index for each patch as an uninterrupted one; Ctrl-C during `generate` says how to resume.
- Warn when failed tiles were filled with black inside kept patches (`sampler.max_empty_ratio: 1`), where masks still carry labels.
- Record how a split was made in `splits/split.json`; split lists end with a newline.
- `stratified` splits now stratify the train/val/test assignment (by labeled fraction and dominant class), not only the subsample; labeled-ratio folders are named exactly (`0.29` → `29/`) and stale split lists are overwritten.

**Labels**
- More than 255 classes is an error instead of wrapping KML IDs to background or reusing class 1; `3` and `3.0` are one class.
- Read QGIS/ogr2ogr `<SimpleData>` labels, tolerate spaces after commas in coordinates, keep polygons inside GeometryCollections, and put the outer ring first.
- Reject projected GeoJSON `crs` members, truncated KML, non-finite coordinates and unsupported label files (`.kmz`, `.shp`) instead of producing empty masks.
- Warn with counts when features are skipped and when no label intersects the imagery.

**Tile downloads**
- Retry truncated responses and apply the failure policy instead of aborting the run; count HTML error pages and empty bodies as failed tiles.
- Honour `Retry-After` with jittered backoff, identify requests with a contact URL, and enforce `max_failed_ratio` for any capitalization of `lenient`.
- Validate `url_template` placeholders when the config loads (for example, a leftover `{s}`).

**EOPF**
- A band that fails to read is an error instead of a silent all-NaN band, and a pixel is valid only when every band has data.
- Accept Windows drive and `file:///C:/` paths; reject URLs with credentials, query strings or fragments at validation.

**Security**
- Update quick-xml to 0.42, fixing quadratic parse time on KML with many duplicate attributes (RUSTSEC-2026-0194); also update rustls (RUSTSEC-2026-0285) and crossbeam-epoch (RUSTSEC-2026-0204).
- Build the image crate with only the PNG, JPEG, WebP and GIF codecs, dropping 56 dependencies including the unmaintained `paste`.

**CLI and security**
- Print each warning once, also under `--quiet`; warn when most tiles fail under `policy: ignore`; say why a resume was refused.
- `mapcv.__version__`, and uppercase argument names in `--help` with Typer 0.27.
- Keep secrets out of manifests and `mapcv validate` output; show deprecation notices; report generation errors without a traceback; `mapcv init` links the provider guidance.

### Documentation

- A rebuilt documentation site organised around getting from zero to a dataset: quickstart, tutorials, how-to guides, a full CLI and configuration reference, and a new logo.
- `examples/`: an Esri + OpenStreetMap quickstart, Sentinel-2 land cover, notebooks for exploring a dataset and training a model, and scripts for the Python API and a PyTorch dataset.
- `PROVIDERS.md` on imagery licensing, provider terms and credentials; `MIGRATION.md`.

### Packaging and CI

- PyO3 0.29, committed `Cargo.lock` and `--locked` builds; project URLs, keywords and dependency floors on PyPI; a 0.2 MB sdist (was 4.5 MB).
- CI tests Python 3.10–3.14 with branch coverage; releases are cut only from `main`, test every wheel and the sdist, wait for approval before publishing, and use this changelog as release notes.
- Release artifacts carry build provenance (`gh attestation verify <file> --repo tahamukhtar20/mapcv`); `SECURITY.md` describes how to report vulnerabilities privately.
- PR-title policy, grouped Dependabot updates, an EOPF smoke-test workflow and issue/PR templates.

## [0.1.0] - 2026-05-04

### Features

- Web Mercator tile math in Rust with Python bindings.
- Async tile fetcher with reqwest and a parallel Rust tile stitcher.
- KML (Rust quick-xml) and GeoJSON label parsing with CRS transforms.
- Scanline polygon rasterizer in Rust.
- Grid and random patch sampler.
- Parallel Rust patch writer and dataset manifest.
- Manifest-driven train/val/test and labeled/unlabeled splitter.
- `mapcv` CLI (`generate`, `split`, `validate`, `init`) with YAML configuration.
- Golden-file validation suite against the GDAL stack.
- Built-in Esri and CartoDB basemap sources.
- Documentation website.

### Bug Fixes

- Clamp tile math inputs and outputs to mercantile behavior.
- Handle non-conformant Google Earth KML.
- Python 3.10 typing compatibility (`typing_extensions.TypedDict`, `npt.NDArray` arguments).
- Use rustls for reqwest to avoid OpenSSL in manylinux builds.

### CI/CD

- Cross-platform release workflow with a version-consistency check and trusted PyPI publishing.
- git-cliff changelog generation.
