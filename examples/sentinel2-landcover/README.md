# Sentinel-2 land cover near Amerongen (open data end to end)

Build a 4-band, 4-class land-cover dataset from one public Sentinel-2 L2A product in ESA's
EOPF Zarr format and OpenStreetMap land-use polygons. The region is about 6.9 × 6.6 km
around Amerongen and Leersum in the province of Utrecht (Netherlands): the forests of the
Utrechtse Heuvelrug, farmland, villages and the Nederrijn river.

## Install

Sentinel-2 support is an optional extra. It currently needs **Python 3.10–3.13** (its zarr 2
dependency has no Python 3.14 wheels yet):

```bash
pip install "mapcv[zarr]"
```

## Run

```bash
cd examples/sentinel2-landcover   # paths in mapcv.yaml are relative to this folder

mapcv plan mapcv.yaml             # no network: estimate from the region and resolution
mapcv generate mapcv.yaml         # reads 4 bands of the product over HTTPS, anonymously
mapcv info dataset
```

`mapcv plan` prints:

```text
╭─ Plan for mapcv.yaml ────────────────────────────────────────────────────────╮
│ Region   5.4, 51.975 → 5.5, 52.035  (≈ 6.9 × 6.6 km)                         │
│ Imagery  Sentinel-2 L2A (EOPF) · 4 bands (≈ 10.00 m/px)                      │
│ Raster   709 × 692 px                                                        │
│ Labels   landuse.geojson · 643 polygon(s) · classes: built_up → 1, farmland  │
│          → 2, forest → 3, water → 4                                          │
│ Patches  ≈ 25 × 128 px (grid, stride 128)                                    │
│ Output   dataset · npy (≈ 6.6 MB)                                            │
│ Split    spatial · test 0.2 · val 0.1                                        │
│ Memory   ≈ 16.7 MB per chunk                                                 │
╰──────────────────────────────────────────────────────────────────────────────╯
```

`generate` opens the remote product (this alone can take 30 s), then reads only the Zarr
chunks that cover the region: two 1830 × 1830 px chunks (about 4 MB each) per band. A
completed run printed:

```text
╭─ Dataset ready ──────────────────────────────────────────────────────────────╮
│ Patches  25                                                                  │
│ Source   eopf_zarr ·                                                         │
│          S2A_MSIL2A_20250513T104041_N0511_R008_T31UFT_20250513T143716.zarr   │
│ Shape    4×128×128 float32                                                   │
│ Splits   train 18 (72%) · val 2 (8%) · test 5 (20%)                          │
│ Time     11m 07s                                                             │
│ Files    dataset/ (Images/, Masks/, manifest.json, splits/)                  │
╰──────────────────────────────────────────────────────────────────────────────╯
  class         id    pixels
  background     0     25.7%
  built_up       1      7.8%
  farmland       2     19.0%
  forest         3     42.9%
  water          4      4.7%
```

### If the download fails or bands come back empty

The EOPF Sentinel Zarr Samples service is free and best-effort. While this example was built
(on a slow, ~85 KB/s connection), it often answered chunk requests with **HTTP 408 Request
Time-out**: three of four `generate` runs stopped with that error, on different chunks. Run the
same command again; finished chunks are kept and the run resumes. A faster connection makes
this much less likely.

Failed reads are retried: a chunk is read up to four times before the run stops with
`reading rows … failed 4 times`. A read that times out can also come back as an empty band
while the other bands have data; mapcv checks every chunk for that and stops with
`band b02 returned no data for rows … while other bands did` instead of writing broken
patches. In both cases, run the same command again to resume from that chunk.


## What you get

```text
dataset/
  Images/patch_0000000.npy ...   float32, shape (4, 128, 128): bands b04, b03, b02, b08
  Masks/patch_0000000.png ...    uint8 class ids: 0 background, 1 built_up, 2 farmland,
                                 3 forest, 4 water
  manifest.json                  product id, bands, dtype, patch shape, CRS (EPSG:32631),
                                 affine transform, per-patch positions and pixel counts
  splits/                        train.txt, val.txt, test.txt, 10|20|30/labeled.txt ...
  patches.geojson                footprints of the patches, to open in QGIS
```

**Bands-first `float32`.** Unlike the PNG/JPG patches of XYZ imagery, every NPY patch has
shape `(bands, height, width)`, in exactly the order of `imagery.bands` (here red, green,
blue, near-infrared). The values are decoded surface reflectance (the product's scale and
offset applied: mostly 0–1, bright surfaces can exceed 1), not raw digital numbers, so `np.load(path)` gives a
model-ready tensor layout with no transposing:

```python
import numpy as np

patch = np.load("dataset/Images/patch_0000000.npy")  # (4, 128, 128) float32
red, green, blue, nir = patch
ndvi = (nir - red) / (nir + red)
```

**NoData is NaN.** Pixels outside the product's valid data (swath edges, no-data areas) are
`NaN`, never 0, because 0 can be a valid reflectance. They count as empty for
`sampler.max_empty_ratio` (here 0.2: patches with more than 20 % NaN are skipped). Replace
them before training, e.g. `np.nan_to_num(patch, nan=0.0)`, and mask them out of the loss if
your region touches an edge. This region lies well inside the product's valid data, so any
NaN you see here comes from a failed download (see above), not from the scene.

The patches stay in the product's own projection (UTM zone 31N); mapcv reprojects the label
polygons to it instead of resampling the imagery.

The notebooks in [`../notebooks/`](../notebooks/) work on this dataset too: set
`DATASET = Path("../sentinel2-landcover/dataset")`.

## Labels

`landuse.geojson` has one `class` property per polygon, derived from OSM tags:

| class | id | OSM tags |
| --- | --- | --- |
| `built_up` | 1 | `landuse=residential, commercial, industrial, retail` |
| `farmland` | 2 | `landuse=farmland, meadow, orchard` |
| `forest` | 3 | `landuse=forest`, `natural=wood` |
| `water` | 4 | `natural=water`, `landuse=reservoir, basin` |

`labels.classes` in `mapcv.yaml` pins these ids. Polygons of each class were merged,
clipped to the query box, simplified by ~10 m (about one pixel) and stripped of parts under
0.25 ha, which keeps the file small without changing what a 10 m pixel can see. Features are
ordered built_up → farmland → forest → water because later polygons are drawn on top
(e.g. a lake inside a residential area stays water). Areas without any of these tags
(roads, grass, heath) are background, class 0.

## Data and attribution

**Imagery** — Copernicus Sentinel-2 L2A, free and open data. Contains modified Copernicus
Sentinel data 2025. Product `S2A_MSIL2A_20250513T104041_N0511_R008_T31UFT_20250513T143716`
(Sentinel-2A, 13 May 2025, MGRS tile 31UFT, 0.03 % cloud cover), read from ESA's EOPF
Sentinel Zarr Samples service. It was found with the EOPF STAC API:

```text
https://stac.core.eopf.eodc.eu/collections/sentinel-2-l2a/items/S2A_MSIL2A_20250513T104041_N0511_R008_T31UFT_20250513T143716
```

To use another scene, search the same API (or browse
<https://stac.browser.user.eopf.eodc.eu/>), copy the item's `product` asset href into
`imagery.path`, and pick a region inside its footprint. The sample service may retire old
products; if this URL stops working, that is the fix.

**Labels** — © OpenStreetMap contributors, available under the
[Open Database Licence (ODbL) 1.0](https://opendatacommons.org/licenses/odbl/1-0/); see
<https://www.openstreetmap.org/copyright>. Downloaded on 2026-10-04 (OSM data as of
2026-10-04T10:53:06Z) from the Overpass API with this query, and converted by
[`../scripts/fetch_osm_labels.py`](../scripts/fetch_osm_labels.py):

```text
[out:json][timeout:90];
(
  way["landuse"~"^(farmland|meadow|orchard|forest|residential|commercial|industrial|retail|reservoir|basin)$"](51.972,5.397,52.038,5.503);
  relation["landuse"~"^(farmland|meadow|orchard|forest|residential|commercial|industrial|retail|reservoir|basin)$"]["type"="multipolygon"](51.972,5.397,52.038,5.503);
  way["natural"~"^(water|wood)$"](51.972,5.397,52.038,5.503);
  relation["natural"~"^(water|wood)$"]["type"="multipolygon"](51.972,5.397,52.038,5.503);
);
out geom;
```

Regenerate it with `python ../scripts/fetch_osm_labels.py sentinel2-landcover`.
