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
figures are written to website/src/assets/figures/ as WebP.
"""

from __future__ import annotations

import json
import shutil
import sys
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageDraw, ImageFont

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

BACKGROUND = (24, 27, 33)
TEXT = (230, 232, 236)
GAP = 8
# One colour per class or split, readable on imagery.
PALETTE = [
    (255, 196, 0),
    (0, 200, 255),
    (255, 64, 129),
    (118, 255, 3),
    (179, 136, 255),
    (255, 145, 0),
]


def font(size: int) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    return ImageFont.load_default(size=size)


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


def tint(
    image: Image.Image, mask: np.ndarray, colors: dict[int, tuple[int, int, int]]
) -> Image.Image:
    """Colour each class (or instance) of ``mask`` over ``image``, with an outline."""
    pixels = np.asarray(image, dtype=np.float32).copy()
    for value, color in colors.items():
        inside = mask == value
        if not inside.any():
            continue
        pixels[inside] = 0.55 * pixels[inside] + 0.45 * np.array(color, dtype=np.float32)
        edge = inside & ~(
            np.roll(inside, 1, 0)
            & np.roll(inside, -1, 0)
            & np.roll(inside, 1, 1)
            & np.roll(inside, -1, 1)
        )
        pixels[edge] = color
    return Image.fromarray(pixels.astype(np.uint8))


def grid(
    tiles: Sequence[Image.Image],
    columns: int,
    caption: str,
    labels: Sequence[str] | None = None,
    legend: Sequence[tuple[str, tuple[int, int, int]]] | None = None,
    size: int = 256,
    height: int | None = None,
) -> Image.Image:
    """Tiles ``size`` wide (and ``height`` high, default square) in ``columns``."""
    tile_h = height or size
    rows = -(-len(tiles) // columns)
    label_h = 26 if labels else 0
    legend_h = 34 if legend else 0
    width = columns * size + (columns + 1) * GAP
    total_h = rows * (tile_h + label_h) + (rows + 1) * GAP + legend_h + 30
    canvas = Image.new("RGB", (width, total_h), BACKGROUND)
    draw = ImageDraw.Draw(canvas)
    for index, tile in enumerate(tiles):
        r, c = divmod(index, columns)
        x, y = GAP + c * (size + GAP), GAP + r * (tile_h + label_h + GAP)
        canvas.paste(tile.resize((size, tile_h), Image.Resampling.NEAREST), (x, y))
        if labels:
            draw.text((x + 4, y + tile_h + 4), labels[index], fill=TEXT, font=font(16))
    y = GAP + rows * (tile_h + label_h + GAP)
    if legend:
        x = GAP
        for name, color in legend:
            draw.rectangle((x, y + 6, x + 18, y + 24), fill=color)
            draw.text((x + 26, y + 6), name, fill=TEXT, font=font(16))
            x += 40 + int(draw.textlength(name, font=font(16)))
        y += legend_h
    draw.text((GAP, y + 4), caption, fill=(160, 165, 175), font=font(13))
    return canvas


def save(image: Image.Image, name: str) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    path = OUT / f"{name}.webp"
    image.save(path, "WEBP", quality=82, method=6)
    print(f"wrote {path.relative_to(ROOT)} ({path.stat().st_size // 1024} KB)")


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
    tiles: list[Image.Image] = []
    for entry in picked:
        tiles.append(rgb(dataset, entry))
    for entry in picked:
        mask = np.asarray(Image.open(dataset / entry["files"]["mask"]))
        tiles.append(tint(rgb(dataset, entry), mask, {1: PALETTE[0]}))
    save(
        grid(tiles, 4, CREDIT_NL, legend=[("building (mask value 1)", PALETTE[0])]),
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
    tiles = []
    for entry in picked:
        image = rgb(dataset, entry)
        draw = ImageDraw.Draw(image)
        for ann in boxes.get(Path(entry["files"]["image"]).name, []):
            x, y, w, h = ann["bbox"]
            color = PALETTE[1] if ann.get("truncated") else PALETTE[0]
            draw.rectangle((x, y, x + w, y + h), outline=color, width=2)
        tiles.append(image)
    save(
        grid(
            tiles,
            4,
            CREDIT_NL,
            legend=[("building", PALETTE[0]), ("building cut by the patch edge", PALETTE[1])],
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
    rng = np.random.default_rng(3)
    tiles = []
    for entry in picked:
        ids = np.asarray(Image.open(dataset / entry["files"]["mask"])).astype(np.int64)
        colors = {
            int(v): tuple(int(c) for c in rng.integers(60, 256, 3)) for v in np.unique(ids) if v
        }
        tiles.append(tint(rgb(dataset, entry), ids, colors))  # type: ignore[arg-type]
    save(grid(tiles, 4, CREDIT_NL + " · one colour per building"), "instance-masks")


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
    save(grid(tiles, 6, CREDIT_NL, labels=captions, size=160), "classification-patches")


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
    colors = {cid: PALETTE[i] for i, cid in enumerate(classes.values())}
    tiles = []
    for entry in picked:
        tiles.append(rgb(dataset, entry))
    for entry in picked:
        mask = np.asarray(Image.open(dataset / entry["files"]["mask"]))
        tiles.append(tint(rgb(dataset, entry), mask, colors))
    save(
        grid(
            tiles,
            3,
            CREDIT_S2,
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
    mosaic = Image.new("RGB", (cols, rows), BACKGROUND)
    for entry in meta["patches"]:
        mosaic.paste(rgb(dataset, entry), (entry["col"], entry["row"]))
    colors = {"train": PALETTE[3], "val": PALETTE[0], "test": PALETTE[2]}
    scale = 1024 / cols
    mosaic = mosaic.resize((1024, round(rows * scale)), Image.Resampling.LANCZOS)
    overlay = Image.new("RGBA", mosaic.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)
    for entry in meta["patches"]:
        split = split_of.get(Path(entry["files"]["image"]).name, "train")
        x, y = entry["col"] * scale, entry["row"] * scale
        box = (x + 2, y + 2, x + 256 * scale - 3, y + 256 * scale - 3)
        draw.rectangle(box, fill=(*colors[split], 60), outline=(*colors[split], 255), width=3)
    mosaic = Image.alpha_composite(mosaic.convert("RGBA"), overlay).convert("RGB")
    save(
        grid(
            [mosaic],
            1,
            CREDIT_NL + " · each square is one 256 px patch",
            legend=list(colors.items()),
            size=1024,
            height=mosaic.height,
        ),
        "spatial-split",
    )


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
        grid(tiles, 6, "Imagery: Beeldmateriaal Nederland (CC BY 4.0)", labels=captions, size=180),
        "change-pairs",
    )


def main(argv: Sequence[str]) -> int:
    if len(argv) != 1:
        print(__doc__)
        return 2
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
