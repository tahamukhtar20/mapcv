# Agent guide for mapcv

Instructions for coding agents (Jules, Claude Code, Codex, Copilot) and anyone new to
the codebase. Human-facing setup lives in [CONTRIBUTING.md](CONTRIBUTING.md); this file is the short,
strict version.

## What mapcv is

A GDAL-free Python + Rust library and CLI that turns a region, imagery and labels into
ready-to-train remote-sensing datasets.

- **Imagery:** XYZ tiles, Sentinel-2 EOPF Zarr, GeoTIFF/COG (local, https, s3).
- **Labels:** GeoJSON, KML, GeoPackage, Shapefile or GeoParquet features, or label rasters.
- **Tasks:** segmentation; detection with COCO/YOLO output; instance segmentation with COCO RLE masks.
- **Output:** image/mask patches, `manifest.json` (v3) and leakage-safe `splits/`.

## Setup

```bash
uv sync --locked --all-extras                          # Python 3.10–3.14; the zarr extra needs <3.14
PYO3_PYTHON=$PWD/.venv/bin/python .venv/bin/maturin develop --uv -q   # build the Rust extension
```

- Rebuild after every change under `src/*.rs`.
- If imports look stale, delete `src/mapcv/_mapcv_rs*.so` and rebuild.
- The Rust toolchain is pinned in `rust-toolchain.toml` (1.99.0).

## Checks (all must pass before a PR; CI runs the same)

```bash
cargo fmt --check
cargo clippy --locked --all-targets -- -D warnings      # clippy pedantic and missing_docs are deny
cargo test --locked
.venv/bin/ruff format --check . && .venv/bin/ruff check .
.venv/bin/mypy src tests                                # strict
.venv/bin/pytest -q
PATH=$PWD/.venv/bin:$PATH .venv/bin/python tests/e2e/journey.py --version "$(grep -m1 '^version' pyproject.toml | cut -d'"' -f2)"
.venv/bin/python -m benchmarks run --quick              # large-scale correctness smoke test
```

Pre-commit runs the same hooks; commit with the venv on `PATH`:

```bash
env PATH=$PWD/.venv/bin:$PATH UV_NO_SYNC=1 PYO3_PYTHON=$PWD/.venv/bin/python git commit
```

## Code map

| Area | Where |
|---|---|
| CLI (`init`, `validate`, `plan`, `generate`, `info`, `split`) | `src/mapcv/cli.py` |
| Config (pydantic, `extra="forbid"`) | `src/mapcv/config.py` |
| Imagery sources (`WindowedRasterSource`) | `src/mapcv/imagery.py`, `src/mapcv/geotiff.py` |
| Pipeline: anchors, then per-chunk window, annotate, write | `src/mapcv/pipeline.py` |
| Targets: what each task attaches to a patch | `src/mapcv/targets/` |
| Writers: output layouts; `finalize` runs after splitting | `src/mapcv/writers/`, `src/mapcv/writer.py` |
| Manifest v3 (reads v1/v2) | `src/mapcv/manifest.py` |
| Splits (spatial by default, no leakage) | `src/mapcv/splitter.py` |
| Rust: rasterizer (a port of GDAL), tile fetch/decode, GeoTIFF read/write, KML, patch writer | `src/*.rs`, `src/geotiff/` |
| Benchmarks and correctness at scale | `benchmarks/` |

Read the docstrings of the protocol in `targets/base.py` and `writers/base.py` before adding a task or
an output format.

## Rules

- **Correctness is checked against references, not by eye.**
  - Rasterization and resampling are compared with rasterio, and boxes with an independent shapely computation.
  - Changes that shouldn't alter outputs must prove byte-identical files (images, masks, splits) against `main`.
  - Silent misalignment, such as a half-pixel shift or a wrong CRS axis order, is the worst class of bug.
- **No panics on user input.** Every Rust entry point validates its input and raises `ValueError`.
  Malformed files must error, never hang or allocate unboundedly.
- **Determinism.** The same config gives the same bytes. Use seeded randomness and ordered maps.
  Text files are written with `\n` on every OS.
- **User-facing messages** say what went wrong and what to do next, with no tracebacks for user errors.
  Credentials in `url_template` never appear in output, manifests or logs.
- **Formats are contracts.** Manifest fields, config keys and output layouts are public. A breaking change
  needs a `!` in the PR title, a `BREAKING CHANGE:` footer and a MIGRATION.md entry.
- **Licensing.** Code derived from another project carries its notice in the source and in
  `THIRD_PARTY_NOTICES.md`. New crates and packages must be MIT, Apache-2.0 or BSD-compatible.
- **Dependencies.** No GDAL, and no new runtime dependency without a reason in the PR.
  Test-only tools go in dev or test groups.

## Workflow

- **Branches and PRs.** One focused branch per change, squash-merged to `main`.
  The PR title must match `^(feat|fix|perf|refactor|docs|ci|chore|test)(\([a-z0-9-]+\))?!?: .+`.
  The body has a short summary and a `## Test plan` with the numbers you measured.
- **What not to touch.** Don't edit `CHANGELOG.md`; it is written at release time.
  Don't add planning, roadmap or paper documents to the repo.
- **Docs.** Docs live in `website/` (Astro Starlight). Update the pages your change affects.
  `cd website && npm ci && npm run build` must pass, including its link check.
- **Temporary files.** Keep large ones (venvs, datasets, benchmark output) inside your checkout,
  out of shared `/tmp`, and delete them when done.
- **Merging.** Agents don't merge; a maintainer reviews and merges.
