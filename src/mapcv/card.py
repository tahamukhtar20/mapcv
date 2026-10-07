"""Dataset cards (``mapcv card``): a ``README.md`` that says what a dataset is.

The card has Hugging Face dataset-card YAML front matter (license, task, tags, size)
and describes the imagery (type, product, CRS, ground sampling distance, bands), the
classes, the splits, the files and how the dataset was made, from the manifest and,
when it exists, ``stats.json``. Its licence is ``other`` with a reminder: the imagery
provider's terms decide what may be shared, and mapcv cannot know them.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional

from mapcv.manifest import Manifest, SourceRecord
from mapcv.stats import STATS_FILENAME

CARD_FILENAME = "README.md"

# Hugging Face task categories by mapcv task.
_TASK_CATEGORIES = {
    "segmentation": ["image-segmentation"],
    "change": ["image-segmentation"],
    "instance": ["image-segmentation"],
    "detection": ["object-detection"],
    "classification": ["image-classification"],
    "regression": ["depth-estimation"],
}
_TASK_NAMES = {
    "segmentation": "semantic segmentation",
    "change": "change detection",
    "instance": "instance segmentation",
    "detection": "object detection",
    "classification": "patch classification",
    "regression": "pixel regression",
}
_PROVIDERS = {
    "esri_satellite": "Esri World Imagery",
    "esri_topo": "Esri World Topographic Map",
    "esri_street": "Esri World Street Map",
    "cartodb_positron": "CARTO Positron basemap",
    "cartodb_dark_matter": "CARTO Dark Matter basemap",
}

# Credit lines the providers ask for (their terms remain the authority).
_ATTRIBUTION = {
    "esri_satellite": "Source: Esri, Maxar, Earthstar Geographics, and the GIS User Community",
    "esri_topo": "Sources: Esri and the GIS User Community",
    "esri_street": "Sources: Esri and the GIS User Community",
    "cartodb_positron": "© OpenStreetMap contributors © CARTO",
    "cartodb_dark_matter": "© OpenStreetMap contributors © CARTO",
}


def _attribution(record: SourceRecord) -> Optional[str]:
    if record.source_type == "eopf_zarr":
        return "Contains modified Copernicus Sentinel data"
    return _ATTRIBUTION.get(record.product_id or "")


def _size_category(count: int) -> str:
    for limit, name in ((1_000, "n<1K"), (10_000, "1K<n<10K"), (100_000, "10K<n<100K")):
        if count < limit:
            return name
    return "100K<n<1M" if count < 1_000_000 else "n>1M"


def _gsd(record: SourceRecord) -> Optional[float]:
    if record.crs is None or record.transform is None:
        return None
    from mapcv.planning import _pixel_size_m

    try:
        return _pixel_size_m(record.crs, record.transform)
    except Exception:  # noqa: BLE001 - an unknown CRS just leaves the GSD out
        return None


def _source_rows(manifest: Manifest) -> List[str]:
    rows = [
        "| Source | Type | Product | CRS | Pixel size | Bands |",
        "| --- | --- | --- | --- | --- | --- |",
    ]
    for record in manifest.sources:
        gsd = _gsd(record)
        product = _PROVIDERS.get(record.product_id or "", record.product_id or "unknown")
        rows.append(
            f"| `{record.name}` | {record.source_type} | {product} | {record.crs or '?'} | "
            f"{f'≈ {gsd:.2f} m' if gsd else '?'} | {', '.join(record.bands) or '?'} ({record.dtype or '?'}) |"
        )
    return rows


def _split_counts(staging_dir: Path) -> Dict[str, int]:
    counts: Dict[str, int] = {}
    for split in ("train", "val", "test"):
        path = staging_dir / "splits" / f"{split}.txt"
        if path.exists():
            counts[split] = len(path.read_text(encoding="utf-8").split())
    return counts


def card_text(staging_dir: Path) -> str:
    """The card of the dataset in ``staging_dir`` (see the module docs)."""
    manifest = Manifest.load(staging_dir / "manifest.json")
    stats: Dict[str, Any] = {}
    stats_path = staging_dir / STATS_FILENAME
    if stats_path.exists():
        stats = json.loads(stats_path.read_text(encoding="utf-8"))
    task = manifest.task if manifest.target is not None else "image-only"
    patches = len(manifest.patches)
    patch_size = (manifest.sampler or {}).get("patch_size")
    name = staging_dir.resolve().name
    front = [
        "---",
        "license: other",
        f"pretty_name: {json.dumps(name)}",
        "task_categories:",
        *(f"- {category}" for category in _TASK_CATEGORIES.get(task, ["other"])),
        "tags:",
        "- remote-sensing",
        "- earth-observation",
        "- geospatial",
        "- mapcv",
        "size_categories:",
        f"- {_size_category(patches)}",
        "---",
    ]
    body = [
        f"# {name}",
        "",
        f"A {_TASK_NAMES.get(task, 'image')} dataset of {patches:,} patches"
        + (f" of {patch_size} × {patch_size} pixels" if patch_size else "")
        + f", made with [mapcv](https://github.com/tahamukhtar20/mapcv) {manifest.mapcv_version or ''}.".rstrip(),
        "",
        "## Imagery",
        "",
        *_source_rows(manifest),
        "",
    ]
    classes = manifest.class_map
    if classes:
        body += ["## Classes", "", "| ID | Class |", "| --- | --- |"]
        if task in ("segmentation", "change") and 0 not in classes.values():
            body.append("| 0 | background |")
        body += [
            f"| {cid} | {label} |"
            for label, cid in sorted(classes.items(), key=lambda item: item[1])
        ]
        if manifest.ignore_index is not None:
            body.append(f"| {manifest.ignore_index} | ignore (no imagery or no label) |")
        body.append("")
    splits = _split_counts(staging_dir)
    if splits:
        body += ["## Splits", "", "| Split | Patches |", "| --- | --- |"]
        body += [f"| {split} | {count:,} |" for split, count in splits.items()]
        body.append("")
    if stats.get("sources"):
        body += [
            "## Normalisation",
            "",
            f"Per-band mean and standard deviation over the `{stats.get('split', 'all')}` "
            "patches, from `stats.json` (`mapcv stats`):",
            "",
            "| Source | Band | Mean | Std |",
            "| --- | --- | --- | --- |",
        ]
        for source, values in stats["sources"].items():
            for band, mean, std in zip(values["bands"], values["mean"], values["std"]):
                if mean is not None:
                    body.append(f"| `{source}` | {band} | {mean:.6g} | {std:.6g} |")
        body.append("")
    labels = (manifest.target.labels if manifest.target is not None else None) or {}
    body += [
        "## Files",
        "",
        "`manifest.json` lists every patch with its files, position and annotation summary"
        + ("; `splits/` holds the split lists" if splits else "")
        + ". The layout is described in the "
        "[mapcv dataset format](https://tahamukhtar20.github.io/mapcv/reference/dataset-format/).",
        "",
        "## Provenance",
        "",
        f"- Made by mapcv {manifest.mapcv_version or '(unknown version)'}, manifest version "
        f"{manifest.version}.",
    ]
    if labels.get("sha256"):
        body.append(f"- Label file SHA-256: `{labels['sha256']}`.")
    for record in manifest.sources:
        fingerprint = record.fingerprint or {}
        if fingerprint.get("sha256_head_tail") or fingerprint.get("sha256_head"):
            digest = fingerprint.get("sha256_head_tail") or fingerprint.get("sha256_head")
            body.append(f"- Imagery `{record.name}` fingerprint: `{digest}`.")
    credits = sorted({line for line in map(_attribution, manifest.sources) if line})
    if labels.get("osm"):
        # The masks are a database adapted from OpenStreetMap: ODbL share-alike applies.
        credits.append(
            "Labels © OpenStreetMap contributors, under the Open Database License (ODbL); "
            "datasets derived from them must be shared under the ODbL too"
        )
    if credits:
        body += ["", "## Attribution", ""] + [f"- {line}" for line in credits]
    body += [
        "",
        "## Licence",
        "",
        "The licence is `other` until you set it: the imagery provider's terms (and the "
        "label source's, such as ODbL for OpenStreetMap) decide what you may share and how "
        "to credit it. Check them before publishing; see "
        "[Providers & licensing](https://tahamukhtar20.github.io/mapcv/project/providers/).",
        "",
    ]
    return "\n".join(front + [""] + body)


def write_card(staging_dir: Path, overwrite: bool = False) -> Path:
    """Write the card to ``README.md``; an existing one is kept unless ``overwrite``.

    Raises:
        FileExistsError: ``README.md`` exists and ``overwrite`` is false.
    """
    path = staging_dir / CARD_FILENAME
    if path.exists() and not overwrite:
        raise FileExistsError(f"{path} exists; pass --force to replace it")
    path.write_text(card_text(staging_dir), encoding="utf-8", newline="\n")
    return path
