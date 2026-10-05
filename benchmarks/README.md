# mapcv benchmarks

A reproducible, offline benchmark suite for `mapcv generate`. It doubles as a large-scale end-to-end test: every run is checked against values computed independently of mapcv, and any mismatch fails the run with a non-zero exit code.

It measures mapcv only. Comparisons with other tools are not part of this suite yet; see [Baselines](#baselines-extension-point).

## Quick start

```bash
uv sync --all-extras          # the dev group includes the benchmark dependencies
uv run maturin develop --uv --release   # benchmark a release build, see below

uv run python -m benchmarks run --quick                       # a few seconds
uv run python -m benchmarks run                               # standard scenarios, 3 repeats
uv run python -m benchmarks run --scenarios M L --repeat 5 --out results.json
uv run python -m benchmarks list                              # scenarios and groups
```

Without uv: `pip install -e . psutil rasterio pyproj`, then `python -m benchmarks run ...` from the repository root.

**Always benchmark a release build.** `maturin develop` without `--release` builds the Rust extension unoptimised and is many times slower; the numbers would describe the debug build. The results file records the git commit but cannot tell the two builds apart, so say which one you used with `--label "release build"`.

The benchmark dependencies (`psutil`, `rasterio`, `pyproj`) live in the `bench` dependency group in `pyproject.toml`. They are not dependencies of mapcv. Without `psutil` peak memory is reported as `null`; without `rasterio` or `pyproj` the mask check is skipped with a message (printed at the end of the run and stored in the results), and everything else still runs.

## Scenarios

`run` without `--scenarios` runs the **standard** group. Names and groups can be mixed: `--scenarios quick S L`. Use `all` for everything.

| Group | Scenario | What it is | What it exercises |
| --- | --- | --- | --- |
| quick | `Q` | 6x6 tiles, 60 polygons | the whole pipeline in about a second |
| quick | `Q-failures` | `Q`, about 1 tile in 7 answers 404 | lenient policy: failed tiles become black, masks get the ignore value there |
| quick | `Q-strict` | same under `policy: strict` | must exit non-zero with a message, no traceback |
| quick | `Q-resume` | `Q`, Ctrl-C part-way, run again | the resumed dataset must equal an uninterrupted one |
| standard | `S` | 10x10 tiles (100 patches), 300 polygons | small job, mostly start-up |
| standard | `M` | 32x32 tiles (1,024 patches), 3,000 polygons | the reference scenario |
| standard | `M-jpg` | `M` written as JPEG | JPEG encoding in the writer, lossy output check |
| standard | `M-jpgtiles` | `M`, the server sends JPEG tiles | tile decode cost |
| standard | `M-overlap` | `M` with stride 128 (3,969 patches) | overlapping patches; the spatial split must not leak across splits |
| standard | `M-failures` | `M`, about 2 % of tiles answer 404 | lenient policy at scale |
| standard | `M-strict` | `M-failures` under `policy: strict` | clean failure |
| standard | `M-resume` | `M`, Ctrl-C at about 40 %, run again | resume at scale, byte-for-byte |
| large | `L` | 100x100 tiles (10,000 patches), 30,000 polygons | throughput and memory at dataset scale |
| large | `XL` | 200x200 tiles (40,000 patches), 30,000 polygons | memory must stay bounded as the area grows |
| large | `M-polys100k` | `M`'s raster with 100,000 polygons | label parsing, indexing and rasterization |

Every scenario is deterministic: the labels are jittered, non-overlapping rectangles in three classes drawn with a fixed seed over a block of zoom-18 tiles near Amsterdam, and the tiles are synthetic (below). Sizes are in `scenarios.py`.

Failed tiles use HTTP 404, which mapcv does not retry. A 5xx or 429 answer is retried with a 0.5 s-step backoff (three retries), so scenarios with 5xx tiles would mostly measure sleeping.

## How a scenario runs

1. A **synthetic XYZ tile server** (`tileserver.py`) starts on `127.0.0.1` in its own process. Pixel (X, Y) of the zoom-18 raster has a colour that is a pure function of its coordinates, so any output patch can be verified exactly, and no imagery licence, network or cache is involved. Before each scenario the harness requests every tile once, so the timed runs do not pay for the server's PNG encoding. The server shares the machine's CPU with mapcv; its per-request cost is small but not zero.
2. The harness writes the labels and a mapcv YAML config into a fresh directory, wipes the previous output, and starts `mapcv generate CONFIG --yes` in a **child process** (`_instrumented.py`, which runs the real CLI in-process with timers around the pipeline stages). This is repeated `--repeat` times (default 3).
3. While the child runs, the harness samples the resident memory (RSS) of the child and its descendants every 20 ms and keeps the peak.
4. After the first repeat the output is checked (below). Later repeats must produce byte-identical output to the first.

## What is measured

Per repeat:

- `wall_s`: wall-clock time of the whole process, from start to exit, including interpreter start-up and imports (about 0.2 s).
- `peak_rss_mb`: peak of the summed RSS of the process tree. Pages shared between processes are counted once per process, so this slightly overstates a multi-process tree; mapcv runs as one process.
- `stages`: seconds spent in each pipeline stage, summed over the run. These are wall-clock times of the wrapped functions, so `fetch` includes network wait and retries.

| Stage | Meaning |
| --- | --- |
| `plan` | the plan and estimate printed before generating |
| `open` | opening the imagery source |
| `labels` | parsing, reprojecting and indexing labels |
| `fetch` | HTTP requests for tiles (Rust fetcher) |
| `decode` | decoding tiles into the strip, excluding `fetch` |
| `rasterize` | burning labels into the strip |
| `sample` | cutting patches out of the strip |
| `write` | encoding and writing images and masks (Rust writer) |
| `manifest` | saving `manifest.json` after each chunk |
| `split` | the train/val/test split |
| `total` | from just before the CLI starts to exit |

Per scenario, `summary` has the **median, min, max and standard deviation** over the repeats for `wall_s` and `peak_rss_mb`, the median of each stage, `tiles_per_s` and `patches_per_s` (from the median wall time) and `output_mb`.

### Reading results

- Compare medians, and look at the spread: a standard deviation that is more than a few percent of the median means the machine was busy. Close other programs, plug a laptop in, and use `--repeat 5` or more.
- Numbers are only comparable on the same machine, the same build (release!) and the same Python. The results file records CPU, RAM, OS, Python, library versions, the mapcv version and git commit (and whether the tree was dirty) so that a number can be traced back.
- `S` is dominated by start-up; use `M` and larger for throughput. Throughput is `tiles / wall time`, which includes start-up and the final split.
- `peak_rss_mb` for `L` and `XL` should be close to each other: mapcv holds a few tile rows at a time (`imagery.strip_rows`), not the area.
- The tiles are small (a few KB; real aerial tiles are typically much larger) and the server is on loopback with no latency, so fetch and decode costs are optimistic compared with a real provider. Use `--online` for a reality check.
- Resume scenarios report the uninterrupted and the resumed wall time and how many patches were written when Ctrl-C arrived. They run once regardless of `--repeat`, and are not performance numbers.

## Correctness checks

A failed check is added to `problems`, printed, stored in the results file, and makes the exit code 1.

- **Images**: every patch (or, for `L`, `XL` and `M-polys100k`, a seeded sample of 300 to 400) equals the tiles the server sent: exactly for PNG output, within a small tolerance for JPEG output. Failed tiles must be black.
- **Masks**: equal to `rasterio.features.rasterize` on the same labels reprojected with `pyproj`; pixels without imagery (failed tiles) carry `labels.ignore_index`. Needs rasterio and pyproj, otherwise skipped with a message.
- **Manifest**: patch count matches the grid, patches lie on the stride grid with no duplicates, `transform` and `crs` are the tile grid's, the class map is right, Images/ and Masks/ hold exactly the listed files, per-class pixel counts equal the mask files, no temporary files are left.
- **Splits**: the lists are well formed (trailing newline), disjoint, name only known patches, and no train or validation patch shares a pixel with a test patch (or a train one with a validation patch).
- **Failure handling**: `strict` exits non-zero, with a message and without a traceback.
- **Resume**: the interrupted run exits non-zero without a traceback, and after resuming the dataset (images, masks, manifest content, splits) is identical to an uninterrupted run. The manifest is compared as parsed JSON, because the key order of `per_class_pixel_counts` differs between runs.
- **Determinism**: all repeats of a scenario produce identical output.

## Results file

`--out FILE` (default `benchmark-results.json`, git-ignored) is a single JSON document, `schema_version` 1:

```text
created, label, mode, repeat, ok, problems[]
machine:   os, cpu_model, cpu_count_logical/physical, ram_gb, python,
           load_avg_1m_at_start/at_end
mapcv:     version, module_path, git_commit, git_branch, git_dirty, libraries{}
scenarios: NAME -> definition, status, problems[], skipped[],
           runs[]    (wall_s, peak_rss_mb, exit_code, stages{} per repeat),
           summary   (wall_s{n,median,min,max,stdev}, peak_rss_mb{...},
                      stages_median_s{}, tiles_per_s, patches_per_s, output_mb),
           checks    (what was verified and the measured differences),
           baselines (only with --baselines)
online:    present only with --online
footprint: present only with --footprint
```

## Optional runs

**`--online`** runs `examples/quickstart` (77 Esri World Imagery tiles, four connections) once and records its time and memory under `online`, separate from the offline scenarios. It checks that the run succeeds and writes a consistent dataset, not pixels. You are responsible for the provider's terms ([PROVIDERS.md](../PROVIDERS.md)); do not put it in CI.

**`--footprint [REQUIREMENT]`** creates a fresh venv and measures `pip install REQUIREMENT` (default `mapcv`, i.e. the latest release on PyPI): wheel size, number of direct and total dependencies, size on disk and install time without pip's cache. It needs network access. Pass a local wheel (`--footprint dist/mapcv-*.whl`) to measure an unpublished build. Install time depends on your network; compare it on the same connection only.

## Baselines (extension point)

Comparisons with other tools are out of scope for now, but the harness has a hook. A baseline is an object with a `name` and a `command(workload)` method returning the command line of a script that does the same job (same tile server, labels, region, patch size, and fetch concurrency) and writes to `workload.output_dir`. Register it in `baselines.py` (see the example in that file) and run:

```bash
python -m benchmarks run --scenarios M --baselines my-script
```

Each baseline runs once per repeat in a fresh child process, measured exactly like mapcv (wall time and peak RSS of the process tree), and appears under `scenarios.M.baselines.my-script`. The harness only measures baselines; checking their output against the same references is up to the baseline's author (`benchmarks.checks.check_dataset` works on any output in mapcv's layout). Give baselines the same fetch concurrency as mapcv (`workload.max_connections`).

## Tests and CI

`tests/test_benchmarks.py` runs `--quick` and the checks' negative cases (a changed pixel, a changed mask, a patch in two splits must all be caught) in about ten seconds. It skips itself if `psutil`, `rasterio` or `pyproj` are missing and on Windows, where the harness is not validated (resume scenarios need POSIX signals). `benchmarks/` is excluded from the source distribution.

The harness patches a few private functions of `mapcv.pipeline` to time stages (`_instrumented.py`). If a refactor renames one, the smoke test fails with the missing name; update the wrapper along with the refactor.
