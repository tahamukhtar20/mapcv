"""Hand a mapcv dataset to other tools (``mapcv export``).

``hf-parquet``
    One Parquet file per split in ``data/`` plus a dataset card, the layout the
    Hugging Face Hub and ``datasets.load_dataset("parquet", ...)`` read. Every patch
    file becomes a ``{bytes, path}`` column, exactly as written (PNG/JPEG columns are
    typed as ``datasets.Image``; NPY/GeoTIFF columns hold the file's bytes), with the
    patch's place (``row``, ``col``, ``crs``, ``transform``) and, for classification,
    its labels. Needs pyarrow (``pip install "mapcv[export]"``).
``webdataset``, ``zarr``
    Few large files instead of one per patch: see :mod:`mapcv.shards`.
``terratorch``
    ``terratorch.yaml``: the ``data:`` section of a TerraTorch training config
    (``GenericNonGeoSegmentationDataModule``, or the pixel-wise regression one) pointing
    at the dataset's folders and split lists, with the band means and standard
    deviations of the train split.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional

from mapcv.data import splits_of
from mapcv.manifest import Manifest, ManifestEntry

FORMATS = ("hf-parquet", "terratorch", "webdataset", "zarr")
_IMAGE_SUFFIXES = (".png", ".jpg", ".jpeg")


def _manifest(root: Path) -> Manifest:
    path = root / "manifest.json"
    if not path.exists():
        raise FileNotFoundError(f"No manifest found at {path}")
    return Manifest.load(path)


def _entries_by_split(root: Path, manifest: Manifest) -> Dict[str, List[ManifestEntry]]:
    splits = splits_of(root)
    if splits == ("all",):
        return {"all": list(manifest.patches)}
    by_name = {manifest.patch_name(entry): entry for entry in manifest.patches}
    result: Dict[str, List[ManifestEntry]] = {}
    for split in splits:
        names = (root / "splits" / f"{split}.txt").read_text(encoding="utf-8").split()
        result[split] = [by_name[name] for name in names if name in by_name]
    return result


def _stats(root: Path) -> Dict[str, Any]:
    path = root / "stats.json"
    if path.exists():
        stats: Dict[str, Any] = json.loads(path.read_text(encoding="utf-8"))
        if stats.get("split") in ("train", "all"):
            return stats
    from mapcv.stats import dataset_stats

    return dataset_stats(root, "train")


# ── Hugging Face Parquet ─────────────────────────────────────────────────────


def export_hf_parquet(root: Path, out: Path) -> List[Path]:
    """Write ``out/data/<split>-00000-of-00001.parquet`` and ``out/README.md``; returns the
    Parquet files. ``out`` must not be the dataset folder."""
    try:
        import pyarrow as pa
        import pyarrow.parquet as pq
    except ImportError as exc:
        raise RuntimeError(
            "the hf-parquet export needs pyarrow: pip install 'mapcv[export]'"
        ) from exc
    root, out = root.resolve(), out.resolve()
    if out == root:
        raise ValueError("export to a new folder, not the dataset folder itself")
    manifest = _manifest(root)
    by_split = _entries_by_split(root, manifest)
    keys = list(manifest.patches[0]["files"]) if manifest.patches else []
    blob = pa.struct([("bytes", pa.binary()), ("path", pa.string())])
    fields = [
        pa.field("name", pa.string()),
        *(pa.field(key, blob) for key in keys),
        pa.field("row", pa.int64()),
        pa.field("col", pa.int64()),
        pa.field("crs", pa.string()),
        pa.field("transform", pa.list_(pa.float64(), 6)),
    ]
    features: Dict[str, Any] = {"name": {"dtype": "string", "_type": "Value"}}
    for key in keys:
        suffix = Path(manifest.patches[0]["files"][key]).suffix.lower()
        features[key] = (
            {"_type": "Image"}
            if suffix in _IMAGE_SUFFIXES
            else {
                "bytes": {"dtype": "binary", "_type": "Value"},
                "path": {"dtype": "string", "_type": "Value"},
            }
        )
    features.update(
        row={"dtype": "int64", "_type": "Value"},
        col={"dtype": "int64", "_type": "Value"},
        crs={"dtype": "string", "_type": "Value"},
        transform={
            "feature": {"dtype": "float64", "_type": "Value"},
            "length": 6,
            "_type": "Sequence",
        },
    )
    classification = manifest.task == "classification"
    if classification:
        fields.append(pa.field("labels", pa.list_(pa.int64())))
        # Class IDs as in the manifest (the card lists their names).
        features["labels"] = {"feature": {"dtype": "int64", "_type": "Value"}, "_type": "Sequence"}
    schema = pa.schema(
        fields, metadata={"huggingface": json.dumps({"info": {"features": features}})}
    )

    written: List[Path] = []
    (out / "data").mkdir(parents=True, exist_ok=True)
    for split, entries in by_split.items():
        columns: Dict[str, List[Any]] = {field.name: [] for field in fields}
        for entry in entries:
            columns["name"].append(manifest.patch_name(entry))
            for key in keys:
                rel = entry["files"][key]
                columns[key].append({"bytes": (root / rel).read_bytes(), "path": Path(rel).name})
            columns["row"].append(entry["row"])
            columns["col"].append(entry["col"])
            columns["crs"].append(manifest.source.crs)
            columns["transform"].append(list(manifest.patch_transform(entry)))
            if classification:
                columns["labels"].append(
                    [int(label) for label in entry["summary"].get("labels") or []]
                )
        table = pa.table(columns, schema=schema)
        path = out / "data" / f"{split}-00000-of-00001.parquet"
        pq.write_table(table, path)
        written.append(path)
    (out / "README.md").write_text(_hf_card(root, list(by_split)), encoding="utf-8", newline="\n")
    return written


def _hf_card(root: Path, splits: List[str]) -> str:
    from mapcv.card import card_text

    card = card_text(root)
    configs = ["configs:", "- config_name: default", "  data_files:"]
    for split in splits:
        configs += [f"  - split: {split}", f"    path: data/{split}-*"]
    end = card.index("\n---", 3)
    return card[:end] + "\n" + "\n".join(configs) + card[end:]


# ── TerraTorch ───────────────────────────────────────────────────────────────


def terratorch_config(root: Path, batch_size: int = 8, num_workers: int = 4) -> str:
    """The ``data:`` section of a TerraTorch config for the dataset (see the module docs).

    Raises:
        ValueError: The dataset is not a single-source segmentation or regression
            dataset, or its patches are NPY files (TerraTorch reads them with GDAL).
    """
    root = root.resolve()
    manifest = _manifest(root)
    if manifest.task not in ("segmentation", "regression") or len(manifest.sources) != 1:
        raise ValueError(
            "the TerraTorch export covers single-source segmentation and regression datasets; "
            f"this one is {manifest.task} with {len(manifest.sources)} source(s)"
        )
    if not manifest.patches:
        raise ValueError("the dataset has no patches")
    files = manifest.patches[0]["files"]
    image_suffix = Path(files["image"]).suffix
    mask_suffix = Path(files.get("mask", "")).suffix
    if image_suffix == ".npy" or mask_suffix == ".npy":
        raise ValueError(
            "TerraTorch reads patches with GDAL, which cannot read NPY: generate the dataset "
            "with writer.image_format and writer.mask_format tif"
        )
    stats = _stats(root)["sources"][manifest.source.name]
    splits = splits_of(root)
    regression = manifest.task == "regression"
    module = (
        "GenericNonGeoPixelwiseRegressionDataModule"
        if regression
        else "GenericNonGeoSegmentationDataModule"
    )
    images, masks = root / "Images", root / "Masks"
    lines = [
        "# The data section of a TerraTorch config, written by mapcv export.",
        f"# Dataset: {root}",
    ]
    if not regression:
        ignore = manifest.ignore_index
        lines.append(
            "# Pixels without imagery or label are "
            + (
                f"{ignore} in the masks; GeoTIFF masks declare it as NoData, which TerraTorch "
                "reads as missing and replaces with no_label_replace (-1 below): set the "
                "task's ignore_index to -1."
                if ignore is not None and mask_suffix == ".tif"
                else f"{ignore} in the masks: set the task's ignore_index to {ignore}."
                if ignore is not None
                else "not marked (no ignore index)."
            )
        )
    lines += [
        "data:",
        f"  class_path: terratorch.datamodules.{module}",
        "  init_args:",
        f"    batch_size: {batch_size}",
        f"    num_workers: {num_workers}",
    ]
    if not regression:
        num_classes = max([0, *manifest.class_map.values(), 1]) + 1
        lines.append(f"    num_classes: {num_classes}")
    for split in ("train", "val", "test"):
        source = split if split in splits else None
        lines += [
            f"    {split}_data_root: {json.dumps(str(images))}",
            f"    {split}_label_data_root: {json.dumps(str(masks))}",
        ]
        if source is not None:
            lines.append(f"    {split}_split: {json.dumps(str(root / 'splits' / f'{split}.txt'))}")
    lines += [
        f'    img_grep: "*{image_suffix}"',
        f'    label_grep: "*{mask_suffix}"',
        f"    means: {json.dumps([_round(v) for v in stats['mean']])}",
        f"    stds: {json.dumps([_round(v) for v in stats['std']])}",
        "    no_data_replace: 0",
    ]
    if not regression:
        lines.append("    no_label_replace: -1")
    if splits == ("all",):
        lines.insert(
            2, "# The dataset has no split lists: every split reads every patch; run mapcv split."
        )
    return "\n".join(lines) + "\n"


def _round(value: Optional[float]) -> float:
    return round(float(value), 6) if value is not None else 0.0


def export_terratorch(root: Path, out: Optional[Path] = None) -> Path:
    """Write :func:`terratorch_config` to ``out`` (default ``<dataset>/terratorch.yaml``)."""
    path = out or root / "terratorch.yaml"
    path.write_text(terratorch_config(root), encoding="utf-8", newline="\n")
    return path
