# Changelog

All notable changes to this project will be documented in this file.

## [0.3.0] - 2026-10-14

mapcv 0.3 turns it from a segmentation tool for XYZ tiles into a dataset compiler for remote sensing: six tasks, your own GeoTIFFs and COGs, Sentinel-2 from STAC, Google Earth Engine, many label formats and OpenStreetMap, and tools to train on, check and share the result. See [MIGRATION.md](https://github.com/tahamukhtar20/mapcv/blob/main/MIGRATION.md) before upgrading: the manifest is now version 3, and masks mark pixels without imagery.

### Highlights

- **Six tasks** (`task:`): segmentation, object detection (COCO and YOLO), instance segmentation (COCO RLE masks, optional instance-ID masks), patch classification (single and multi-label), change detection (before/after pairs in the LEVIR-CD layout) and regression (float targets from a raster of values).
- **Your own imagery** (`type: geotiff`): GeoTIFF and Cloud Optimized GeoTIFF from local files, `https://` and anonymous `s3://`, read fast in the file's own CRS and pixel grid, any data type and band count; a glob pattern (`tiles/*.tif`) reads a folder of tiles as one mosaic.
- **More imagery sources:** Sentinel-2 band COGs found in a STAC catalog (`type: stac_cog`), EOPF products found by search, SCL cloud masking, Google Earth Engine images and cloud-filtered composites (`imagery.earth_engine`, `pip install "mapcv[gee]"`), and several sources sampled on one grid (dates, sensors, a DEM), optionally stacked into one file per patch.
- **More labels:** GeoPackage, Shapefile and GeoParquet besides GeoJSON and KML, label rasters (same grid or reprojected), labels straight from OpenStreetMap through Overpass, several label files with a class each, buffered lines and points, and annotated areas (outside them, pixels are ignored rather than background).
- **Areas of interest:** `region.path` with polygons samples only where they are, and `split.strategy: region` holds out whole areas.
- **Outputs:** georeferenced GeoTIFF and NPY images and masks, world files, `patches.geojson` footprints for QGIS, and exports to Hugging Face Parquet, TerraTorch, WebDataset shards and Zarr.
- **Train, check and share:** `mapcv.data.MapcvDataset` loads any task for PyTorch; `mapcv stats` (band means and class weights), `mapcv card` (a Hugging Face dataset card), `mapcv verify` (with `SHA256SUMS` over every file), tiled prediction over a region into one GeoTIFF (`mapcv.infer.predict_raster`), and `mapcv doctor` for the installation.
- **AI agents:** `mapcv mcp` (`pip install "mapcv[mcp]"`) lets Claude, Cursor and other MCP clients inspect labels, validate, plan and generate, read-only by default and confined to one folder; plus an agent skill and `llms.txt`.
- **Datasets that stay consistent:** `generate` and `split` lock the dataset folder, an interrupted run is marked incomplete in the manifest (`complete`) and resumes where it stopped, `info`, `verify`, `stats` and `card` say when a dataset is unfinished and the exports refuse one, and a GeoTIFF changed since the last run (size or modification time) stops a resume.
- **Exact and fast:** the rasterizer matches GDAL's rules (checked against rasterio), XYZ tiles are decoded in Rust in parallel, downloaded tiles are cached on disk as the server's caching headers allow, and each chunk's memory is bounded. A benchmark suite checks correctness at scale; see the Performance page.

### Breaking changes

- **Manifest version 3** (`task`, `sources`, `target`, per-patch `files` and `summary`). mapcv 0.3 reads versions 1–3; 0.2 cannot read version 3. Code that reads `manifest.json` directly needs the new keys (MIGRATION.md has a table and a snippet).
- **Pixels without imagery** (padding, failed tiles, NoData, outside a GeoTIFF) are `labels.ignore_index` (255 by default) in masks instead of background `0`. Pass it to your loss, or set `labels.ignore_index: null` for 0.2's behaviour (needed to resume a 0.2 dataset).
- **`tiles:` and `region.zoom`**, deprecated in 0.2, are removed and fail validation with a pointer to MIGRATION.md.
- **JPEG patches** use 4:2:0 chroma subsampling by default, as Pillow does (up to half the size); `writer.jpg_subsampling: "4:4:4"` resumes a 0.2 JPEG dataset.
- **`sampler.mode: random`** draws positions without repeats and covers the edges; 0.2 random datasets cannot be resumed.
- **Patches without any imagery** (all tiles failed, wholly in NoData or outside a GeoTIFF) are no longer written, whatever `sampler.max_empty_ratio` is; `generate` says how many were left out.
- **Stricter configs:** a key written twice in the same mapping is an error, a number field refuses a YAML boolean (`patch_size: true`), and `writer.staging_dir` must be new, empty or a mapcv dataset.
- **One run holds at most 3,000,000 patches** (grid positions or `random_count`); `plan` counts any grid and `generate` refuses a larger one with what to change. Jobs over 1,000,000 patches still ask for confirmation.
- **Runs that write nothing** leave no dataset: a plan with no patch stops before it starts, and a run whose every patch is dropped writes no manifest, split lists or footprints.
- **Tile helpers** (`mapcv.bounds`, `mapcv.snap_bbox` and the tile functions of the extension) raise `ValueError` for a coordinate that is not a number, a zoom above 32 or a tile that does not exist, instead of clamping.
- **Library output:** `run_generate` and the download functions no longer print or draw progress bars; messages go to the `mapcv` logger and progress to callbacks. The manifest classes, `load_or_create_manifest` and `write_patches` signatures changed (see MIGRATION.md).

### Security

- Redirects are checked like the configured URL, and requests send no `Referer` header. A host the config does not name, reached through a redirect or a STAC catalog, may not be on this machine or a private network, judged by the addresses its name resolves to when connecting.
- `mapcv mcp` connects to public addresses only unless started with `--allow-local-urls`, and its URL errors read the same whatever listens at the address.
- `MapcvDataset` reads Zarr stores of numbers with plain compressors only and refuses others (such as `pickle`) before opening any array; EOPF products are opened with the `pickle` codec disabled.
- Dataset folders with symbolic links that point outside are refused, manifests with absolute or `..` patch paths do not load, and credentials and query values in URLs are redacted from messages and manifests in one place.
- Crafted inputs are bounded: GeoParquet, GeoPackage and GeoTIFF readers cap the memory a small file can claim, KML field inspection is linear, and a pipe or device given as an input file is refused instead of hanging.

### Fixes

- Rasterization follows GDAL's scanline and `all_touched` rules exactly, so mask edges match rasterio.
- Rust entry points validate array shapes instead of panicking; the GeoTIFF, KML and tile decoders are fuzzed.
- Degenerate and antimeridian-crossing boxes are handled; the region is range-checked.
- Class counts in the manifest are deterministic.
- User errors print one message with the next step instead of a traceback: corrupt manifests, unreadable patches, region splits without areas, and configs that could never work are refused at `validate`. `--debug` shows the traceback when you need it.
- `plan` counts grid patches without building them, so a fine stride or a huge region no longer exhausts memory (also through the MCP `plan` tool).
- KML files in UTF-16, ISO-8859-1 and windows-1252 are read; GeoPackages in read-only folders open; multi-part features spanning UTM zones are buffered per zone; `plan` runs the same class-ID checks as `generate`; uint64 label rasters keep their values.
- Label files: numbers sort by value, a `label_field` no feature has is an error, OSM ways and relations are assembled robustly, and Overpass failures (including answers cut off mid-transfer) give a clean message.
- XYZ fetching retries and reports failed tiles without URL paths; the `mapcv init` wizard checks each answer; progress lines fit narrow terminals.
- Change detection with annotated areas no longer counts unlabeled pixels as change; Earth Engine composites are filtered on the exact tile area and cached per area.

### Install

- `pip install mapcv`; extras: `zarr` (EOPF Sentinel-2, Python 3.10–3.13), `gee` (Earth Engine), `parquet` (GeoParquet labels), `export` (Hugging Face Parquet), `mcp` (MCP server). `mapcv doctor` checks what is installed. GeoTIFF/COG, STAC COGs and the other label formats need no extra.

## [0.2.0] - 2026-10-04

mapcv 0.2 adds Sentinel-2 imagery, leakage-safe dataset splits and Python 3.14 support, and fixes several ways 0.1 could silently produce wrong datasets. See [MIGRATION.md](https://github.com/tahamukhtar20/mapcv/blob/main/MIGRATION.md) before upgrading: configs and datasets need small changes.

### Highlights

- **EOPF Sentinel-2 L2A Zarr input** (`pip install "mapcv[zarr]"`): lazy windowed reads, band selection, 10/20/60 m harmonization on the product's own pixel grid, labels aligned in the native UTM CRS, and bands-first `float32` NPY patches.
- **Leakage-safe splits by default.** The new `spatial` strategy keeps whole raster blocks in one split and leaves out train/val patches that overlap a held-out patch, so test scores are not inflated by shared pixels.
- **Python 3.14 and one wheel per platform.** Stable-ABI (abi3) wheels cover Python 3.10 and newer on Linux (x86-64 and aarch64, glibc and musl), macOS and Windows, so Docker on Apple Silicon, AWS Graviton, Raspberry Pi and Alpine install without a Rust toolchain.
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
- Say why tiles failed: errors and warnings group failures by cause (HTTP status, timeout, non-image response) with one example URL.
- Retry truncated responses and apply the failure policy instead of aborting the run; count HTML error pages and empty bodies as failed tiles.
- Honour `Retry-After` with jittered backoff, identify requests with a contact URL, and enforce `max_failed_ratio` for any capitalization of `lenient`.
- Validate `url_template` placeholders when the config loads (for example, a leftover `{s}`).

**EOPF**
- Retry Sentinel-2 window reads up to 4 times when the object store times out or a band comes back empty, instead of failing the run on the first error.
- A band that fails to read is an error instead of a silent all-NaN band, and a pixel is valid only when every band has data.
- Accept Windows drive and `file:///C:/` paths; reject URLs with credentials, query strings or fragments at validation.

**Security**
- Update quick-xml to 0.42, fixing quadratic parse time on KML with many duplicate attributes (RUSTSEC-2026-0194); also update rustls (RUSTSEC-2026-0285) and crossbeam-epoch (RUSTSEC-2026-0204).
- Build the image crate with only the PNG, JPEG, WebP and GIF codecs, dropping 56 dependencies including the unmaintained `paste`.

**CLI and security**
- On Windows, redirected or piped output (`mapcv plan x.yaml > plan.txt`, CI logs) no longer crashes with `UnicodeEncodeError`.
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
