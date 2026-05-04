# Changelog

All notable changes to this project will be documented in this file.

## [Unreleased]

### Bug Fixes

- Clamp tile math inputs to mercantile defaults
- Harden tile math API edges
- Clamp tile math outputs consistently
- Address Copilot review comments
- Address Copilot review comments on M3
- Address Copilot review comments on M3 (z param, docstrings, Rust docs)
- Address Copilot review comments on M4
- Address code review comments on M5
- Resolve clippy pedantic errors (items_after_statements, manual_is_multiple_of)
- Remove unused struct import in test_stitcher
- Pass strict=False to fastkml.KML.parse to handle non-conformant Google Earth KML
- Add npt.NDArray type args to satisfy mypy on Python 3.10 (CI)
- Use typing_extensions.TypedDict for pydantic compatibility on Python < 3.12
- Address codebase review findings (bugs, types, conventions)

### CI/CD

- Add rasterio to mypy ignore_missing_imports for cross-validation test
- Improve release workflow with multi-Python, ARM, test gate, and GitHub Release
- Add version consistency check to release workflow (#25)
- Add git-cliff changelog generation (#26)

### Documentation

- Add warning regarding unsupported 512x512 tile sources to README
- Fix API inaccuracies, remove Amsterdam demo, add deploy workflow
- Update documentation, simplify CONTRIBUTING instructions, and refine README quick-start guide
- Fix QA-identified inaccuracies in CLI, config, API, and examples
- Add downloads, docs, and ruff badges to README (#22)
- Add author contact to README

### Features

- Implement core Web Mercator tile math in Rust with Python bindings
- Implement core Web Mercator tile math in Rust
- Implement tile fetcher with reqwest
- Implement core Web Mercator tile math in Rust
- Restore fetcher wrapper and tests
- Complete M2 tile fetcher
- Implement M3 label parsing (KML, GeoJSON, CRS transform)
- Implement M4 scanline rasterizer in Rust
- Implement M5 patch sampler (Rust anchors + Python wrapper)
- Implement M6 patch writer and manifest
- M6.5 - Rust tile stitcher with parallel PNG decode
- Rust parallel patch writer (rayon, 6.6x faster than ThreadPoolExecutor)
- Replace fastkml KML parser with Rust quick-xml implementation
- M7 dataset splitter (manifest-driven train/val/test + labeled/unlabeled)
- M8 CLI + config (generate, split, validate, init commands)
- MV validation suite (golden-file cross-validation against GDAL stack)
- Expand built-in tile sources with Esri and CartoDB basemaps
- Add docs scaffold, OSM label data, and Jupyter-aware downloader
- Add MkDocs Material website with hero, animations, and full docs

### Refactoring

- Migrate documentation to Astro Starlight and improve class count logging in pipeline


