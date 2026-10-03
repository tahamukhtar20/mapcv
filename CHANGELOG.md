# Changelog

All notable changes to this project will be documented in this file.

## [0.2.0] - Unreleased

See [MIGRATION.md](https://github.com/tahamukhtar20/mapcv/blob/main/MIGRATION.md) for upgrade steps.

### Features

- Add optional EOPF Sentinel-2 L2A Zarr input (`pip install "mapcv[zarr]"`) with lazy window reads, band selection, resolution harmonization, native-CRS label alignment, and bands-first `float32` NPY output.
- Add a discriminated `imagery` configuration block (`xyz` | `eopf_zarr`).
- Add `labels.classes` for explicit label → class ID mapping, and read KML `<SimpleData>` labels (QGIS/ogr2ogr exports).
- Upgrade manifests to version 2, recording source, product, bands, dtype, patch shape, CRS, affine transform, and sampler settings.
- Compute sampling anchors over the whole raster, so patches cross former strip seams without loss or duplication.
- Add a leakage-safe `spatial` split strategy (now the default) that keeps whole raster blocks in one split and leaves out train/val patches overlapping a held-out patch.

### Breaking changes

- Remove the built-in `google_satellite` preset; configs that use it fail validation.
- `sampler.mode: random` draws `random_count` patches in total rather than per strip.
- 0.1.x (manifest v1) datasets can be split but not resumed by `mapcv generate`.
- Resuming requires the same imagery, labels, and sampler settings.
- `imagery.type` is required.
- The default split strategy is `spatial` instead of `stratified`.
- Class IDs no longer depend on feature order: integer labels are used as IDs, other labels are numbered in sorted order.

### Deprecations

- `region.zoom` and the `tiles` block, replaced by `imagery` with `type: xyz`. Both are removed in 0.3.0.

### Bug Fixes

- Save the manifest after every chunk and overwrite orphaned files from interrupted runs.
- Snap EOPF reads to the product pixel grid so native-resolution bands are not resampled.
- Accept Windows drive and `file:///C:/` EOPF paths.
- Treat black-filled failed XYZ tiles as empty for `max_empty_ratio`.
- Keep secrets out of manifests and `mapcv validate` output; reject EOPF URLs with credentials, query strings, or fragments at validation.
- Report generation errors without a traceback.
- `stratified` splits now stratify the train/val/test assignment by labeled fraction and dominant class, not only the subsample.
- Name labeled-ratio folders exactly (`0.29` → `29/`, `0.125` → `12.5/`) and reject ratios outside (0, 1] or duplicates.
- Overwrite stale split lists when splitting an empty manifest.
- Reject more than 255 classes instead of wrapping KML class IDs to 0 or reusing IDs, and treat `3` and `3.0` as one class.
- Reject non-WGS-84 GeoJSON `crs` members, truncated KML, and non-finite coordinates; accept spaces after commas in KML coordinates and keep polygons inside GeometryCollections.
- Warn with counts when features are skipped (no polygon, no label, or unmapped label), and reject unsupported label file types up front.
- Retry truncated tile responses and apply the failure policy instead of aborting the whole download; count non-image responses (HTML error pages, empty bodies) as failed tiles.
- Honour `Retry-After` on 429/5xx with jittered backoff, identify requests with a contact URL in the User-Agent, and raise `ValueError` for an invalid policy or connection count.
- Enforce `max_failed_ratio` for any capitalization of `lenient`, and validate `url_template` placeholders up front.
- Fetch XYZ tiles per chunk instead of holding the whole region in memory, so labels and resume compatibility are checked before any download, memory stays bounded to about one chunk of tiles, and a resumed run downloads only the chunks it still needs. `max_failed_ratio` now applies to each chunk.
- Warn when no label polygon intersects the imagery (for example swapped longitude/latitude).

### Documentation

- Add `PROVIDERS.md` on imagery licensing and credentials, and link it from `mapcv init`.
- Add `MIGRATION.md` and matching website pages.

### CI/CD

- Support Python 3.14 by upgrading PyO3 to 0.29, and ship stable-ABI (abi3) wheels: one wheel per platform for Python 3.10 and newer.
- Test Python 3.10–3.14, enforce branch coverage, and check wheel metadata before publishing.
- Add a PR-title policy, grouped Dependabot updates, an EOPF smoke-test workflow, and issue/PR templates.
- Publish project URLs and keywords on PyPI, set minimum versions for runtime dependencies, and slim the sdist from 4.5 MB to 0.2 MB.

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
