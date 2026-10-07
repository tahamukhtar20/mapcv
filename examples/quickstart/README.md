# Quickstart: buildings in Amsterdam

Build a small building-segmentation dataset from Esri World Imagery and OpenStreetMap
building footprints in three commands. It covers about 1.0 × 0.6 km of Amsterdam's Eastern
Docklands (KNSM-eiland, Java-eiland and Borneo-Sporenburg) at zoom 18, roughly 0.36 m per
pixel, and takes well under a minute.

```bash
pip install mapcv
cd examples/quickstart          # paths in mapcv.yaml are relative to this folder

mapcv plan mapcv.yaml           # what will it download and write? (no network)
mapcv generate mapcv.yaml       # fetch 77 tiles, rasterize labels, write patches and splits
mapcv info dataset              # summarize the result
```

## Expected output

`mapcv plan mapcv.yaml`:

```text
╭─ Plan for mapcv.yaml ────────────────────────────────────────────────────────╮
│ Region   4.9375, 52.3725 → 4.9515, 52.378  (≈ 1.0 × 0.6 km)                  │
│ Imagery  esri_satellite · zoom 18 (≈ 0.36 m/px)                              │
│ Raster   2,816 × 1,792 px · 77 tiles (≈ 1.9 MB to download)                  │
│ Labels   buildings.geojson · 967 polygon(s) · classes: building → 1          │
│ Patches  ≈ 77 × 256 px (grid, stride 256)                                    │
│ Output   dataset · png (≈ 8.6 MB)                                            │
│ Split    spatial · test 0.2 · val 0.1                                        │
│ Memory   ≈ 28.8 MB per chunk                                                 │
╰──────────────────────────────────────────────────────────────────────────────╯
```

The end of `mapcv generate mapcv.yaml` (timings depend on your connection):

```text
╭─ Dataset ready ──────────────────────────────────────────────────────────────╮
│ Patches  77                                                                  │
│ Source   xyz · esri_satellite                                                │
│ Shape    256×256×3 uint8                                                     │
│ Tiles    77 fetched · 0 failed                                               │
│ Splits   train 52 (68%) · val 8 (10%) · test 17 (22%)                        │
│ Time     12s                                                                 │
│ Files    dataset/ (Images/, Masks/, manifest.json, splits/)                  │
╰──────────────────────────────────────────────────────────────────────────────╯
  class         id    pixels
  background     0     81.0%
  building       1     19.0%
```

The result:

```text
dataset/
  Images/patch_0000000.png ...    256×256 RGB imagery
  Masks/patch_0000000.png ...     256×256 class ids: 0 = background, 1 = building
  manifest.json                   source, class map, CRS, per-patch positions and pixel counts
  splits/train.txt val.txt test.txt, splits/10|20|30/labeled.txt unlabeled.txt
  patches.geojson                 footprints of the patches, to open in QGIS
```

The split is `spatial`: whole blocks of the raster go to train, val or test, so neighbouring
patches never leak across the train/test boundary. Running `generate` again resumes an
interrupted run instead of starting over.

## Next

* Explore the dataset: [`../notebooks/01-explore-a-dataset.ipynb`](../notebooks/01-explore-a-dataset.ipynb)
* Train a model on it: [`../notebooks/02-train-a-segmentation-model.ipynb`](../notebooks/02-train-a-segmentation-model.ipynb)
* Load it in PyTorch: [`../scripts/torch_dataset.py`](../scripts/torch_dataset.py)
* Do the same from Python instead of YAML: [`../scripts/python_api.py`](../scripts/python_api.py)
* One class per OSM building type (house, apartments, houseboat, ...): set
  `labels.label_field: building` and use a new `writer.staging_dir`.

## Data and attribution

**Labels** — `buildings.geojson` holds 967 building footprints (ways and multipolygon
relations tagged `building=*`) from OpenStreetMap.
© OpenStreetMap contributors, available under the
[Open Database Licence (ODbL) 1.0](https://opendatacommons.org/licenses/odbl/1-0/); see
<https://www.openstreetmap.org/copyright>. Each feature keeps its OSM id and `building` tag.
It was downloaded on 2026-10-04 (OSM data as of 2026-10-04T10:57:06Z) from the Overpass API
with this query, and converted by
[`../scripts/fetch_osm_labels.py`](../scripts/fetch_osm_labels.py) (which also records the
query inside the file):

```text
[out:json][timeout:60];
(
  way["building"](52.3715,4.936,52.379,4.953);
  relation["building"]["type"="multipolygon"](52.3715,4.936,52.379,4.953);
);
out geom;
```

Regenerate it with `python ../scripts/fetch_osm_labels.py quickstart`. A dataset or model you
derive from these labels is subject to the ODbL's share-alike terms for databases.

**Imagery** — Esri World Imagery (sources: Esri, Maxar, Earthstar Geographics, and the GIS
user community), proprietary, under Esri's terms of use; see
[PROVIDERS.md](../../PROVIDERS.md). It is not included in this repository, so don't commit the
generated `dataset/` folder.
