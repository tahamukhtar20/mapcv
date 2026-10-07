"""Rebuild the docs' figures from real mapcv datasets over openly licensed data.

Every figure is made by running mapcv on a tutorial's config, with the imagery swapped
for an open source so the images may be republished:

- Dutch national aerial photos, Beeldmateriaal Nederland via PDOK (CC BY 4.0), for the
  Amsterdam and Utrecht tutorials (the tutorials themselves use Esri imagery);
- Copernicus Sentinel-2 L2A (free and open), for the land-cover tutorial;
- OpenStreetMap labels (ODbL), the tutorials' own label files.

Usage (needs network; takes a few minutes):

    python website/scripts/figures.py WORKDIR

WORKDIR holds the generated datasets (inside your checkout, deleted when you like); the
figures are written to website/src/assets/figures/ as WebP. They share one clean style:
the Lato font (downloaded once, SIL Open Font License; DejaVu Sans if offline), a muted
palette, white background, a short left-aligned title stating what the figure shows, and
the data credit in grey. Needs matplotlib (``pip install "mapcv[docs]"``).
"""

from __future__ import annotations

import json
import os
import shutil
import sys
import urllib.request
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch, Rectangle
from PIL import Image

import mapcv

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "website" / "src" / "assets" / "figures"
PDOK = "https://service.pdok.nl/hwh/luchtfotorgb/wmts/v1_0/{layer}/EPSG:3857/{{z}}/{{x}}/{{y}}.jpeg"
AMSTERDAM = {"west": 4.9375, "south": 52.3725, "east": 4.9515, "north": 52.3780}
UTRECHT = {"west": 5.400, "south": 51.975, "east": 5.500, "north": 52.035}
EOPF_PRODUCT = (
    "https://objectstore.eodc.eu:2222/e05ab01a9d56408d82ac32d69a5aae2a:202505-s02msil2a/13/"
    "products/cpm_v256/S2A_MSIL2A_20250513T104041_N0511_R008_T31UFT_20250513T143716.zarr"
)
CREDIT_NL = "Imagery: Beeldmateriaal Nederland (CC BY 4.0) · Labels: © OpenStreetMap contributors"
CREDIT_S2 = "Contains modified Copernicus Sentinel data 2025 · Labels: © OpenStreetMap contributors"

# The muted seaborn "deep" palette, written out so the script needs no seaborn.
DEEP = {
    "blue": "#4c72b0",
    "orange": "#dd8452",
    "green": "#55a868",
    "red": "#c44e52",
    "purple": "#8172b3",
    "brown": "#937860",
    "pink": "#da8bc3",
    "grey": "#8c8c8c",
    "olive": "#ccb974",
    "cyan": "#64b5cd",
}
INK = "#1a1a1a"
MUTED = "#6b7280"
LATO = "https://raw.githubusercontent.com/google/fonts/main/ofl/lato/Lato-{}.ttf"


def rgb_of(color: str) -> tuple[int, int, int]:
    value = color.lstrip("#")
    return (int(value[0:2], 16), int(value[2:4], 16), int(value[4:6], 16))


def apply_clean_style() -> str:
    """Lato (fetched once into the user cache), white background, muted palette."""
    from matplotlib import font_manager

    cache = Path(os.environ.get("XDG_CACHE_HOME") or Path.home() / ".cache") / "mapcv-figures"
    cache.mkdir(parents=True, exist_ok=True)
    family = "DejaVu Sans"
    for weight in ("Regular", "Bold", "Italic"):
        path = cache / f"Lato-{weight}.ttf"
        try:
            if not path.exists():
                urllib.request.urlretrieve(LATO.format(weight), path)
            font_manager.fontManager.addfont(str(path))
            family = "Lato"
        except OSError:
            pass  # offline: keep DejaVu Sans
    matplotlib.rcParams.update(
        {
            "font.family": family,
            "font.size": 9,
            "figure.titlesize": 11,
            "figure.titleweight": "bold",
            "axes.titlesize": 8.5,
            "axes.titleweight": "regular",
            "text.color": INK,
            "legend.fontsize": 8,
            "legend.frameon": False,
            "figure.facecolor": "white",
            "savefig.facecolor": "white",
        }
    )
    return family


def panels(
    tiles: Sequence[Image.Image],
    columns: int,
    title: str,
    credit: str,
    labels: Sequence[str] | None = None,
    row_labels: Sequence[str] | None = None,
    legend: Sequence[tuple[str, str]] | None = None,
    boxes: Sequence[Sequence[tuple[float, float, float, float, str]]] | None = None,
    width: float = 7.2,
) -> plt.Figure:
    """``tiles`` in ``columns``: a left-aligned title, optional per-tile captions, row
    labels, (x, y, w, h, colour) boxes per tile, a legend and the credit line."""
    rows = -(-len(tiles) // columns)
    tile_w = (width - (0.32 if row_labels else 0.05)) / columns
    aspect = tiles[0].height / tiles[0].width
    tile_h = tile_w * aspect + (0.2 if labels else 0.0)
    top, bottom = 0.42 + (0.1 if labels else 0.0), 0.24 + (0.28 if legend else 0.0)
    height = rows * tile_h + top + bottom
    fig, axes = plt.subplots(rows, columns, figsize=(width, height), squeeze=False)
    fig.subplots_adjust(
        left=(0.32 if row_labels else 0.02) / width,
        right=1 - 0.02 / width,
        top=1 - top / height,
        bottom=bottom / height,
        wspace=0.04,
        hspace=0.18 if labels else 0.04,
    )
    for index, ax in enumerate(axes.flat):
        ax.set_axis_off()
        if index >= len(tiles):
            continue
        ax.imshow(np.asarray(tiles[index]), interpolation="nearest")
        if labels:
            ax.set_title(labels[index], loc="left", pad=3)
        for x, y, w, h, color in boxes[index] if boxes else ():
            ax.add_patch(
                Rectangle((x - 0.5, y - 0.5), w, h, fill=False, edgecolor=color, linewidth=0.9)
            )
    for r, name in enumerate(row_labels or ()):
        axes[r, 0].text(
            -0.05,
            0.5,
            name,
            transform=axes[r, 0].transAxes,
            rotation=90,
            ha="right",
            va="center",
            color=MUTED,
            fontsize=8.5,
        )
    fig.suptitle(title, x=0.02 / width, y=1 - 0.12 / height, ha="left", va="top")
    if legend:
        fig.legend(
            handles=[
                Patch(facecolor=color, edgecolor="none", label=name) for name, color in legend
            ],
            loc="lower left",
            bbox_to_anchor=(0.0, 0.2 / height),
            ncol=len(legend),
            handlelength=1.2,
            columnspacing=1.6,
        )
    fig.text(0.02 / width, 0.06 / height, credit, color=MUTED, fontsize=7, ha="left", va="bottom")
    return fig


def tint(
    image: Image.Image, mask: np.ndarray, colors: dict[int, str], alpha: float = 0.42
) -> Image.Image:
    """Colour each class (or instance) of ``mask`` over ``image``, with a 1 px outline."""
    pixels = np.asarray(image, dtype=np.float32).copy()
    for value, color in colors.items():
        inside = mask == value
        if not inside.any():
            continue
        rgb = np.array(rgb_of(color), dtype=np.float32)
        pixels[inside] = (1 - alpha) * pixels[inside] + alpha * rgb
        edge = inside & ~(
            np.roll(inside, 1, 0)
            & np.roll(inside, -1, 0)
            & np.roll(inside, 1, 1)
            & np.roll(inside, -1, 1)
        )
        pixels[edge] = rgb
    return Image.fromarray(pixels.astype(np.uint8))


def save(fig: plt.Figure, name: str) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    path = OUT / f"{name}.webp"
    fig.savefig(path, dpi=200, pil_kwargs={"quality": 85, "method": 6})
    plt.close(fig)
    print(f"wrote {path.relative_to(ROOT)} ({path.stat().st_size // 1024} KB)")


def pdok(layer: str = "Actueel_orthoHR") -> dict[str, Any]:
    return {"type": "xyz", "url_template": PDOK.format(layer=layer), "max_connections": 4}


def build(workdir: Path, name: str, config: dict[str, Any]) -> Path:
    """Generate one dataset (reused when it already exists)."""
    staging = workdir / name
    config = {
        **config,
        "writer": {**config.get("writer", {}), "staging_dir": str(staging)},
        # The tutorials' split: whole blocks per split, no leakage between them.
        "split": {"strategy": "spatial", "test_ratio": 0.2, "val_ratio": 0.1, "seed": 42},
    }
    if not (staging / "manifest.json").exists():
        shutil.rmtree(staging, ignore_errors=True)
        print(f"generating {name} ...", flush=True)
        mapcv.generate(mapcv.MapcvConfig.model_validate(config))
    return staging


def manifest(dataset: Path) -> dict[str, Any]:
    data: dict[str, Any] = json.loads((dataset / "manifest.json").read_text(encoding="utf-8"))
    return data


def rgb(dataset: Path, entry: dict[str, Any]) -> Image.Image:
    path = dataset / entry["files"]["image"]
    if path.suffix == ".npy":
        bands = np.load(path)[:3]  # b04, b03, b02 reflectance
        stretched = np.clip(np.nan_to_num(bands) / 0.3, 0, 1) ** (1 / 1.8)
        return Image.fromarray((np.moveaxis(stretched, 0, -1) * 255).astype(np.uint8))
    return Image.open(path).convert("RGB")


def busiest(entries: list[dict[str, Any]], count: int, score: Any) -> list[dict[str, Any]]:
    """``count`` patches with the highest ``score``, spread over the dataset (stable)."""
    ranked = sorted(entries, key=lambda e: (-score(e), e["row"], e["col"]))
    picked: list[dict[str, Any]] = []
    for entry in ranked:
        if all(abs(entry["row"] - p["row"]) + abs(entry["col"] - p["col"]) > 256 for p in picked):
            picked.append(entry)
        if len(picked) == count:
            break
    return sorted(picked, key=lambda e: (e["row"], e["col"]))


# ── figures ──────────────────────────────────────────────────────────────────


def segmentation(workdir: Path, labels: Path) -> None:
    dataset = build(
        workdir,
        "segmentation",
        {
            "region": AMSTERDAM,
            "imagery": {**pdok(), "zoom": 18},
            "labels": {"path": str(labels), "label_field": "class"},
            "sampler": {"patch_size": 256, "edge_strategy": "drop"},
        },
    )
    meta = manifest(dataset)
    picked = busiest(meta["patches"], 4, lambda e: e["summary"].get("class_pixels", {}).get("1", 0))
    tiles: list[Image.Image] = [rgb(dataset, entry) for entry in picked]
    for entry in picked:
        mask = np.asarray(Image.open(dataset / entry["files"]["mask"]))
        tiles.append(tint(rgb(dataset, entry), mask, {1: DEEP["orange"]}))
    save(
        panels(
            tiles,
            4,
            "Building masks on exactly the imagery's pixel grid",
            CREDIT_NL,
            row_labels=["image", "mask"],
            legend=[("building (mask value 1)", DEEP["orange"])],
        ),
        "segmentation-patches",
    )


def detection(workdir: Path, labels: Path) -> None:
    dataset = build(
        workdir,
        "detection",
        {
            "task": "detection",
            "region": AMSTERDAM,
            "imagery": {**pdok(), "zoom": 18},
            "labels": {"path": str(labels), "label_field": "class"},
            "detection": {"min_visible": 0.3, "min_box_pixels": 2, "formats": ["coco"]},
            "sampler": {"patch_size": 256, "edge_strategy": "drop"},
        },
    )
    boxes: dict[str, list[dict[str, Any]]] = {}
    for coco in sorted((dataset / "annotations").glob("instances_*.json")):
        data = json.loads(coco.read_text(encoding="utf-8"))
        names = {image["id"]: image["file_name"] for image in data["images"]}
        for ann in data["annotations"]:
            boxes.setdefault(names[ann["image_id"]], []).append(ann)
    meta = manifest(dataset)
    picked = busiest(
        meta["patches"], 4, lambda e: len(boxes.get(Path(e["files"]["image"]).name, []))
    )
    tiles = [rgb(dataset, entry) for entry in picked]
    drawn = [
        [
            (*ann["bbox"], DEEP["cyan"] if ann.get("truncated") else DEEP["orange"])
            for ann in boxes.get(Path(entry["files"]["image"]).name, [])
        ]
        for entry in picked
    ]
    save(
        panels(
            tiles,
            4,
            "One box per building, from the polygons clipped to each patch",
            CREDIT_NL,
            legend=[
                ("building", DEEP["orange"]),
                ("building cut by the patch edge (truncated)", DEEP["cyan"]),
            ],
            boxes=drawn,
        ),
        "detection-boxes",
    )


def instance(workdir: Path, labels: Path) -> None:
    dataset = build(
        workdir,
        "instance",
        {
            "task": "instance",
            "region": AMSTERDAM,
            "imagery": {**pdok(), "zoom": 18},
            "labels": {"path": str(labels), "label_field": "class"},
            "instance": {"min_visible": 0.3, "min_area": 4, "id_mask": True},
            "sampler": {"patch_size": 256, "edge_strategy": "drop"},
        },
    )
    meta = manifest(dataset)

    def count(entry: dict[str, Any]) -> int:
        ids = np.asarray(Image.open(dataset / entry["files"]["mask"]))
        return len(np.unique(ids)) - 1

    picked = busiest(meta["patches"], 4, count)
    cycle = list(DEEP.values())
    tiles = []
    for entry in picked:
        ids = np.asarray(Image.open(dataset / entry["files"]["mask"])).astype(np.int64)
        colors = {int(v): cycle[int(v) % len(cycle)] for v in np.unique(ids) if v}
        tiles.append(tint(rgb(dataset, entry), ids, colors, alpha=0.5))
    save(
        panels(
            tiles,
            4,
            "Every building is its own instance, even where buildings touch",
            CREDIT_NL + " · one colour per instance",
        ),
        "instance-masks",
    )


def classification(workdir: Path, labels: Path) -> None:
    dataset = build(
        workdir,
        "classification",
        {
            "task": "classification",
            "region": UTRECHT,
            "imagery": {**pdok("Actueel_ortho25"), "zoom": 16},
            "labels": {"path": str(labels), "label_field": "class"},
            "classification": {"mode": "single", "min_fraction": 0.5, "empty": "skip"},
            "sampler": {"patch_size": 128, "edge_strategy": "drop"},
        },
    )
    meta = manifest(dataset)
    names = {int(v): k for k, v in meta["target"]["class_map"].items()}
    by_class: dict[int, list[dict[str, Any]]] = {}
    for entry in meta["patches"]:
        for label in entry["summary"].get("labels", [])[:1]:
            by_class.setdefault(int(label), []).append(entry)
    tiles, captions = [], []
    for cid in sorted(by_class):
        # The clearest examples: patches the class covers most, apart from each other.
        def coverage(entry: dict[str, Any], cid: int = cid) -> float:
            return float(entry["summary"].get("class_coverage", {}).get(str(cid), 0.0))

        for entry in busiest(by_class[cid], 3, coverage):
            tiles.append(rgb(dataset, entry))
            captions.append(names.get(cid, str(cid)).replace("_", " "))
    save(
        panels(
            tiles,
            6,
            "Each patch gets the class that covers at least half of it",
            CREDIT_NL,
            labels=captions,
        ),
        "classification-patches",
    )


def land_cover(workdir: Path, labels: Path) -> None:
    classes = {"built_up": 1, "farmland": 2, "forest": 3, "water": 4}
    dataset = build(
        workdir,
        "land-cover",
        {
            "region": UTRECHT,
            "imagery": {
                "type": "eopf_zarr",
                "path": EOPF_PRODUCT,
                "resolution": 10,
                "bands": ["b04", "b03", "b02", "b08"],
            },
            "labels": {"path": str(labels), "label_field": "class", "classes": classes},
            "sampler": {"patch_size": 128, "edge_strategy": "drop", "max_empty_ratio": 0.2},
            "writer": {"image_format": "npy"},
        },
    )
    meta = manifest(dataset)

    def variety(entry: dict[str, Any]) -> int:
        return sum(1 for v in entry["summary"].get("class_pixels", {}).values() if v > 400)

    picked = busiest(meta["patches"], 3, variety)
    colors = {1: DEEP["red"], 2: DEEP["olive"], 3: DEEP["green"], 4: DEEP["blue"]}
    tiles = [rgb(dataset, entry) for entry in picked]
    for entry in picked:
        mask = np.asarray(Image.open(dataset / entry["files"]["mask"]))
        tiles.append(tint(rgb(dataset, entry), mask, colors))
    save(
        panels(
            tiles,
            3,
            "Sentinel-2 patches at 10 m with four land-cover classes",
            CREDIT_S2,
            row_labels=["true colour", "mask"],
            legend=[(name.replace("_", " "), colors[cid]) for name, cid in classes.items()],
        ),
        "land-cover-patches",
    )


def spatial_split(workdir: Path) -> None:
    """The segmentation dataset's patches on the map, outlined by split."""
    dataset = workdir / "segmentation"
    meta = manifest(dataset)
    split_of: dict[str, str] = {}
    for name in ("train", "val", "test"):
        for line in (dataset / "splits" / f"{name}.txt").read_text(encoding="utf-8").split():
            split_of[line] = name
    rows = max(e["row"] for e in meta["patches"]) + 256
    cols = max(e["col"] for e in meta["patches"]) + 256
    mosaic = Image.new("RGB", (cols, rows), "white")
    for entry in meta["patches"]:
        mosaic.paste(rgb(dataset, entry), (entry["col"], entry["row"]))
    colors = {"train": DEEP["blue"], "val": DEEP["orange"], "test": DEEP["red"]}
    fig = panels(
        [mosaic],
        1,
        "Spatial split: val and test come in whole blocks, away from train",
        CREDIT_NL + " · each square is one 256 px patch",
        legend=list(colors.items()),
    )
    ax = fig.axes[0]
    for entry in meta["patches"]:
        color = colors[split_of.get(Path(entry["files"]["image"]).name, "train")]
        x, y = entry["col"] + 6, entry["row"] + 6
        fill = 0.06 if color == colors["train"] else 0.28  # the held-out blocks stand out
        ax.add_patch(Rectangle((x, y), 244, 244, facecolor=color, alpha=fill, edgecolor="none"))
        ax.add_patch(Rectangle((x, y), 244, 244, fill=False, edgecolor=color, linewidth=1.4))
    save(fig, "spatial-split")


def change(workdir: Path) -> None:
    """The same patches in 2016 and 2026: what a before/after pair looks like."""
    # Strandeiland, IJburg: an island raised from open water and built on since 2019.
    region = {"west": 4.9990, "south": 52.3440, "east": 5.0230, "north": 52.3580}
    before = build(
        workdir,
        "before",
        {
            "region": region,
            "imagery": {**pdok("2016_ortho25"), "zoom": 16},
            "sampler": {"patch_size": 256, "edge_strategy": "drop"},
        },
    )
    after = build(
        workdir,
        "after",
        {
            "region": region,
            "imagery": {**pdok("2026_orthoHR"), "zoom": 16},
            "sampler": {"patch_size": 256, "edge_strategy": "drop"},
        },
    )
    a = {(e["row"], e["col"]): e for e in manifest(before)["patches"]}
    b = {(e["row"], e["col"]): e for e in manifest(after)["patches"]}
    common = sorted(set(a) & set(b))

    def difference(place: tuple[int, int]) -> float:
        x = np.asarray(rgb(before, a[place]), dtype=np.float32)
        y = np.asarray(rgb(after, b[place]), dtype=np.float32)
        return float(np.abs(x - y).mean())

    picked = sorted(common, key=lambda p: (-difference(p), p))[:3]
    tiles, captions = [], []
    for place in sorted(picked):
        tiles += [rgb(before, a[place]), rgb(after, b[place])]
        captions += ["2016", "2026"]
    save(
        panels(
            tiles,
            6,
            "Before/after pairs on one grid: open water in 2016, a new island in 2026",
            "Imagery: Beeldmateriaal Nederland (CC BY 4.0) · Strandeiland, IJburg, Amsterdam",
            labels=captions,
        ),
        "change-pairs",
    )


def main(argv: Sequence[str]) -> int:
    if len(argv) != 1:
        print(__doc__)
        return 2
    apply_clean_style()
    workdir = Path(argv[0]).resolve()
    workdir.mkdir(parents=True, exist_ok=True)
    buildings = ROOT / "examples" / "quickstart" / "buildings.geojson"
    landuse = ROOT / "examples" / "sentinel2-landcover" / "landuse.geojson"
    segmentation(workdir, buildings)
    spatial_split(workdir)
    detection(workdir, buildings)
    instance(workdir, buildings)
    classification(workdir, landuse)
    change(workdir)
    land_cover(workdir, landuse)
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
