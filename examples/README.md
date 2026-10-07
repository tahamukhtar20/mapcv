# mapcv examples

Small, reproducible examples that take you from a config file to a trained model. Each one
runs in minutes on a laptop and uses openly licensed labels from OpenStreetMap.

| Example | What it shows | Needs | Runtime |
| --- | --- | --- | --- |
| [`quickstart/`](quickstart/) | XYZ imagery + building polygons → PNG patches, masks and a leakage-free spatial split, with `mapcv plan`, `generate` and `info` | `pip install mapcv` | < 1 min, 77 tiles (~2 MB) |
| [`sentinel2-landcover/`](sentinel2-landcover/) | A public Sentinel-2 L2A EOPF Zarr product + land-use polygons → 4-band `float32` NPY patches with 4 land-cover classes | `pip install "mapcv[zarr]"` (Python 3.10–3.13) | a few minutes, ~32 MB read; the free EOPF sample service often times out (see its README) |
| [`notebooks/01-explore-a-dataset.ipynb`](notebooks/01-explore-a-dataset.ipynb) | Read `manifest.json` and the split lists; plot the spatial split, class balance and image/mask overlays | `matplotlib` | seconds |
| [`notebooks/02-train-a-segmentation-model.ipynb`](notebooks/02-train-a-segmentation-model.ipynb) | Train a tiny U-Net in PyTorch for two epochs and report test mIoU | `torch` (optional, not a mapcv dependency) | ~1 min on CPU |
| [`scripts/python_api.py`](scripts/python_api.py) | The quickstart from Python: `MapcvConfig.model_validate`, `mapcv.plan`, `mapcv.generate` and the `GenerateResult` | `pip install mapcv` | < 1 min |
| [`scripts/torch_dataset.py`](scripts/torch_dataset.py) | A minimal, copy-paste dataset class that reads PNG/JPG/NPY patches, masks, `splits/*.txt` and `manifest.json`. mapcv itself ships a complete one, [`mapcv.data.MapcvDataset`](https://tahamukhtar20.github.io/mapcv/guides/use-your-dataset/#a-pytorch-dataset), that reads every task and format | numpy, Pillow; PyTorch optional | — |
| [`scripts/fetch_osm_labels.py`](scripts/fetch_osm_labels.py) | How the label files were made: one Overpass query each, converted to compact GeoJSON | network | seconds |

## Run them

Every example folder holds a `mapcv.yaml` whose paths are relative to that folder, so run
the CLI from inside it:

```bash
pip install mapcv
cd examples/quickstart
mapcv plan mapcv.yaml        # estimate tiles, patches, disk and memory; downloads nothing
mapcv generate mapcv.yaml    # build ./dataset
mapcv info dataset           # summarize it
```

Then open the notebooks (they read `../quickstart/dataset` by default):

```bash
pip install jupyterlab matplotlib            # plus torch for notebook 02
jupyter lab examples/notebooks/
```

The scripts run from anywhere:

```bash
python examples/scripts/python_api.py --plan-only
python examples/scripts/torch_dataset.py examples/quickstart/dataset
```

Generated `dataset/` folders are ignored by git. Keep it that way: they contain third-party
imagery.

## Data and licences

| File | Content | Source | Licence |
| --- | --- | --- | --- |
| `quickstart/buildings.geojson` | 967 building footprints, Amsterdam Eastern Docklands | OpenStreetMap via Overpass API, OSM data as of 2026-10-04T10:57:06Z | ODbL 1.0 |
| `sentinel2-landcover/landuse.geojson` | Land use/land cover near Amerongen (Utrecht), dissolved into 4 classes | OpenStreetMap via Overpass API, OSM data as of 2026-10-04T10:53:06Z | ODbL 1.0 |

The exact Overpass queries are in each example's README, inside each GeoJSON file (the
top-level `"osm"` member) and in [`scripts/fetch_osm_labels.py`](scripts/fetch_osm_labels.py),
which rebuilds the files. They replace the unattributed `examples/data/amsterdam_*.geojson`
files and the old `01_amsterdam_demo.ipynb` notebook of mapcv 0.1.

**Attribution:** label data © OpenStreetMap contributors, available under the
[Open Database Licence (ODbL) 1.0](https://opendatacommons.org/licenses/odbl/1-0/) —
<https://www.openstreetmap.org/copyright>. If you publish a dataset or a model trained on
these labels, keep this attribution; a derived database is subject to the ODbL's share-alike
terms.

**Imagery is not included.** The examples download it when you run them:

* Esri World Imagery (quickstart) is proprietary, under Esri's terms of use.
* Sentinel-2 data are free and open: "Contains modified Copernicus Sentinel data 2025",
  read from ESA's EOPF Sentinel Zarr Samples service.

Licenses and credit lines per source: [PROVIDERS.md](../PROVIDERS.md).
