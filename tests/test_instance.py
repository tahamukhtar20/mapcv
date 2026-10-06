"""Instance segmentation datasets (``task: instance``): COCO RLE masks and instance-ID PNGs.

Checked against references, not by eye:

* the RLE encoder equals ``pycocotools.mask.encode`` (and decodes back through it);
* the COCO files load in ``pycocotools.coco.COCO`` and every annotation decodes to the
  mask ``rasterio.features.rasterize`` (GDAL) burns for the same feature on the patch's
  pixel grid, limited to the pixels that are inside the patch, inside the raster and
  over imagery;
* ``area`` and ``bbox`` equal ``pycocotools.mask.area`` and ``toBbox`` of the mask;
* the instances of a patch, painted in feature order, equal the semantic segmentation
  mask mapcv writes for the same labels.
"""

from __future__ import annotations

import json
import math
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional, Tuple

import numpy as np
import numpy.typing as npt
import pytest
import shapely
import yaml
from PIL import Image
from pydantic import ValidationError
from shapely.geometry import MultiPolygon, Point, Polygon, box, mapping
from shapely.geometry.base import BaseGeometry

from mapcv._rle import counts_to_string, encode_mask, encode_part, mask_bbox, rle_counts
from mapcv.config import InstanceOptions, MapcvConfig
from mapcv.imagery import RasterMetadata
from mapcv.manifest import Manifest, ManifestMismatchError
from mapcv.pipeline import run_generate, run_split
from mapcv.splitter import SplitterConfig, _stratum
from mapcv.targets import InstanceTarget, create_target
from mapcv.targets.detection import to_pixels
from mapcv.targets.instance import MAX_INSTANCE_ID, InstanceWindow
from mapcv.writers import FilesWriter, InstanceWriter, check_compatible, create_writer

pycocotools_mask = pytest.importorskip("pycocotools.mask")
pycocotools_coco = pytest.importorskip("pycocotools.coco")
rasterio_features = pytest.importorskip("rasterio.features")
from rasterio.transform import Affine  # noqa: E402

R = 6_378_137.0
# A zoom-18 Web Mercator grid near Amsterdam.
RES = 0.5971642834779232
X0, Y0 = 549_582.2333704159, 6_868_784.235762509
HEIGHT, WIDTH = 300, 330
PATCH = 64
CLASSES = ("building", "car", "tree")
CLASS_IDS = {name: index + 1 for index, name in enumerate(CLASSES)}


# ── the RLE encoder against pycocotools ─────────────────────────────────────


def reference_counts(mask: npt.NDArray[np.bool_]) -> str:
    rle = pycocotools_mask.encode(np.asfortranarray(mask.astype(np.uint8)))
    counts = rle["counts"]
    return counts.decode("ascii") if isinstance(counts, bytes) else str(counts)


def decode(counts: str, height: int, width: int) -> npt.NDArray[np.bool_]:
    decoded = pycocotools_mask.decode({"size": [height, width], "counts": counts})
    return np.asarray(decoded, dtype=np.bool_)


def random_masks() -> Iterator[npt.NDArray[np.bool_]]:
    rng = np.random.default_rng(5)
    for height, width in [
        (1, 1),
        (1, 17),
        (17, 1),
        (2, 3),
        (7, 13),
        (64, 64),
        (65, 31),
        (255, 256),
    ]:
        for density in (0.02, 0.5, 0.98):
            yield rng.random((height, width)) < density
    # Blobby masks: long runs, and runs that differ in size by more than 15 (multi-group counts).
    yield np.pad(np.ones((40, 50), dtype=bool), ((3, 90), (200, 5)))
    for index in range(20):
        small = rng.random((9, 11)) < 0.5
        yield np.kron(small, np.ones((rng.integers(1, 40), rng.integers(1, 40)), dtype=bool)) > 0
    # Runs of 2**20 and more: counts that need four 5-bit groups.
    yield np.zeros((1100, 1000), dtype=bool)
    yield np.ones((1100, 1000), dtype=bool)
    half = np.zeros((1100, 1000), dtype=bool)
    half[:, 500:] = True
    yield half


def test_rle_equals_pycocotools_encode_and_round_trips() -> None:
    checked = 0
    for mask in random_masks():
        counts = encode_mask(mask)
        assert counts == reference_counts(mask)
        assert np.array_equal(decode(counts, *mask.shape), mask)
        assert sum(rle_counts(mask)) == mask.size
        checked += 1
    assert checked == 48


def test_rle_edge_cases() -> None:
    for shape in [(1, 1), (5, 7), (64, 64), (3, 1)]:
        empty = np.zeros(shape, dtype=bool)
        full = np.ones(shape, dtype=bool)
        assert encode_mask(empty) == reference_counts(empty)
        assert encode_mask(full) == reference_counts(full)
        assert rle_counts(empty) == [empty.size]
        assert rle_counts(full) == [0, full.size]  # the first run is always background
        for row, col in [(0, 0), (shape[0] - 1, shape[1] - 1), (shape[0] // 2, shape[1] // 2)]:
            single = np.zeros(shape, dtype=bool)
            single[row, col] = True
            assert encode_mask(single) == reference_counts(single)
            assert np.array_equal(decode(encode_mask(single), *shape), single)
    # Column-major order: a left-to-right stripe is one run per column.
    stripe = np.zeros((4, 3), dtype=bool)
    stripe[1, :] = True
    assert rle_counts(stripe) == [1, 1, 3, 1, 3, 1, 2]


def test_rle_of_a_part_placed_in_a_larger_mask_equals_pycocotools() -> None:
    rng = np.random.default_rng(3)
    for _ in range(400):
        height, width = int(rng.integers(1, 40)), int(rng.integers(1, 40))
        rows, columns = int(rng.integers(1, height + 1)), int(rng.integers(1, width + 1))
        y, x = int(rng.integers(0, height - rows + 1)), int(rng.integers(0, width - columns + 1))
        # Full columns, scattered pixels and solid blocks: runs that join across columns.
        kind = rng.integers(0, 3)
        part = (
            np.ones((rows, columns), dtype=bool)
            if kind == 0
            else rng.random((rows, columns)) < (0.5 if kind == 1 else 0.9)
        )
        mask = np.zeros((height, width), dtype=bool)
        mask[y : y + rows, x : x + columns] = part
        assert encode_part(part, x, y, height, width) == reference_counts(mask)
        assert encode_mask(mask) == reference_counts(mask)
    # Whole columns next to each other are one run, also at the right and bottom edges.
    mask = np.zeros((6, 5), dtype=bool)
    mask[:, 1:4] = True
    mask[:, 4] = True
    assert rle_counts(mask) == [6, 24] and encode_mask(mask) == reference_counts(mask)
    mask[0, 0] = True
    assert rle_counts(mask) == [0, 1, 5, 24] and encode_mask(mask) == reference_counts(mask)


def test_rle_counts_string_encoding_of_signed_differences() -> None:
    # Counts whose differences to the count two places earlier are negative and large.
    for counts in ([0, 5, 9000, 3, 2, 70000, 1], [100, 1, 1, 1, 1, 50, 4], [0, 1]):
        total = sum(counts)
        flat = np.zeros(total, dtype=bool)
        position = 0
        for index, count in enumerate(counts):
            flat[position : position + count] = index % 2 == 1
            position += count
        mask = flat.reshape((total, 1))  # one column: column-major order is the flat order
        assert counts_to_string(counts) == reference_counts(mask)


def test_bbox_area_and_rle_agree_with_pycocotools() -> None:
    rng = np.random.default_rng(1)
    for _ in range(100):
        mask = np.zeros((rng.integers(1, 80), rng.integers(1, 90)), dtype=bool)
        y, x = rng.integers(0, mask.shape[0]), rng.integers(0, mask.shape[1])
        mask[y : y + rng.integers(1, 30), x : x + rng.integers(1, 30)] = rng.random() < 0.9
        mask |= rng.random(mask.shape) < 0.01
        if not mask.any():
            continue
        rle = pycocotools_mask.encode(np.asfortranarray(mask.astype(np.uint8)))
        assert list(mask_bbox(mask)) == [int(v) for v in pycocotools_mask.toBbox(rle)]
        assert int(pycocotools_mask.area(rle)) == int(mask.sum())
    with pytest.raises(ValueError, match="empty mask"):
        mask_bbox(np.zeros((3, 3), dtype=bool))


# ── fixtures: a Web Mercator raster with holes in its validity mask ─────────


def to_lonlat(x: float, y: float) -> Tuple[float, float]:
    return math.degrees(x / R), math.degrees(2 * math.atan(math.exp(y / R)) - math.pi / 2)


def lonlat_geometry(geometry: BaseGeometry) -> BaseGeometry:
    return shapely.transform(
        geometry, lambda xy: np.array([to_lonlat(x, y) for x, y in xy], dtype=np.float64)
    )


def make_valid_mask(kind: str = "edges") -> npt.NDArray[np.bool_]:
    """Pixels with imagery: ``edges`` has a failed-tile-like block and a diagonal NoData edge."""
    rows, cols = np.mgrid[0:HEIGHT, 0:WIDTH]
    valid = np.ones((HEIGHT, WIDTH), dtype=np.bool_)
    if kind != "full":
        valid[100:140, 200:260] = False
        valid[rows - cols > 200] = False
    if kind == "noisy":
        valid[np.random.default_rng(2).random((HEIGHT, WIDTH)) < 0.03] = False
    return valid


class FakeSource:
    """A Web Mercator RGB raster with holes in its validity mask."""

    def __init__(self, valid: Optional[npt.NDArray[np.bool_]] = None) -> None:
        rng = np.random.default_rng(0)
        self.valid = make_valid_mask() if valid is None else valid
        image = rng.integers(1, 256, size=(HEIGHT, WIDTH, 3), dtype=np.uint8)
        image[~self.valid] = 0
        self.image = image
        self.metadata = RasterMetadata(
            source_type="xyz",
            product_id="fake",
            width=WIDTH,
            height=HEIGHT,
            bands=["red", "green", "blue"],
            dtype="uint8",
            crs="EPSG:3857",
            transform=(RES, 0.0, X0, 0.0, -RES, Y0),
            chunk_rows=128,
        )

    def read_window(
        self, row_start: int, row_stop: int, col_start: int, col_stop: int
    ) -> Tuple[npt.NDArray[np.uint8], npt.NDArray[np.bool_]]:
        return (
            self.image[row_start:row_stop, col_start:col_stop],
            self.valid[row_start:row_stop, col_start:col_stop],
        )

    def close(self) -> None:
        pass


def star(rng: random.Random, cx: float, cy: float, radius: float) -> Polygon:
    """An irregular star-shaped (so simple) polygon around (cx, cy), in metres."""
    count = rng.randint(3, 12)
    angles = sorted(rng.uniform(0, 2 * math.pi) for _ in range(count))
    coords = [
        (
            cx + math.cos(t) * radius * rng.uniform(0.4, 1.0),
            cy + math.sin(t) * radius * rng.uniform(0.4, 1.0),
        )
        for t in angles
    ]
    polygon = Polygon(coords)
    return (
        polygon if polygon.is_valid and polygon.area > 0 else box(cx, cy, cx + radius, cy + radius)
    )


@dataclass
class Feature:
    world: BaseGeometry  # Web Mercator metres
    name: str


def random_features(seed: int, count: int = 160) -> List[Feature]:
    """Ordinary, tiny, multi-part, holed and large objects, many crossing patch edges.

    They overlap each other, and some lie partly outside the raster.
    """
    rng = random.Random(seed)
    left, top = X0 - 20 * RES, Y0 + 20 * RES
    span_x, span_y = (WIDTH + 40) * RES, (HEIGHT + 40) * RES
    features: List[Feature] = []
    for index in range(count):
        cx = left + rng.uniform(0, span_x)
        cy = top - rng.uniform(0, span_y)
        kind = index % 5
        name = rng.choice(CLASSES)
        if kind == 0:
            geometry: BaseGeometry = star(rng, cx, cy, rng.uniform(3, 40) * RES)
        elif kind == 1:
            geometry = star(rng, cx, cy, rng.uniform(0.3, 2.5) * RES)
        elif kind == 2:
            first = star(rng, cx, cy, rng.uniform(2, 15) * RES)
            dx, dy = rng.uniform(-40, 40) * RES, rng.uniform(-40, 40) * RES
            second = star(rng, cx + dx, cy + dy, rng.uniform(2, 15) * RES)
            geometry = first if first.intersects(second) else MultiPolygon([first, second])
        elif kind == 3:
            radius = rng.uniform(8, 50) * RES
            outer = Point(cx, cy).buffer(radius, quad_segs=rng.randint(2, 8))
            hole = Point(cx, cy).buffer(radius * 0.5, quad_segs=3)
            geometry = Polygon(outer.exterior.coords, [hole.exterior.coords])
        else:
            geometry = star(rng, cx, cy, rng.uniform(40, 120) * RES)
        features.append(Feature(geometry, name))
    return features


def write_labels(path: Path, features: List[Feature]) -> None:
    collection = {
        "type": "FeatureCollection",
        "features": [
            {
                "type": "Feature",
                "properties": {"class": feature.name},
                "geometry": mapping(lonlat_geometry(feature.world)),
            }
            for feature in features
        ],
    }
    path.write_text(json.dumps(collection), encoding="utf-8")


def instance_config(
    root: Path,
    *,
    instance: Optional[Dict[str, Any]] = None,
    split: Optional[Dict[str, Any]] = None,
    sampler: Optional[Dict[str, Any]] = None,
    labels: Optional[Dict[str, Any]] = None,
    staging: str = "dataset",
    task: str = "instance",
) -> MapcvConfig:
    data: Dict[str, Any] = {
        "task": task,
        "region": {"west": 4.93, "south": 52.37, "east": 4.95, "north": 52.38},
        "imagery": {"type": "xyz", "zoom": 18, "url_template": "http://127.0.0.1/{z}/{x}/{y}.png"},
        "labels": {"path": str(root / "labels.geojson"), "label_field": "class", **(labels or {})},
        "sampler": {"patch_size": PATCH, "edge_strategy": "pad", **(sampler or {})},
        "writer": {"staging_dir": str(root / staging), "image_format": "png"},
    }
    if instance is not None:
        data["instance"] = instance
    if split is not None:
        data["split"] = split
    return MapcvConfig.model_validate(data)


def generate(config: MapcvConfig, source: Optional[FakeSource] = None) -> Manifest:
    with pytest.MonkeyPatch.context() as patch:
        patch.setattr("mapcv.pipeline.open_raster_source", lambda *a, **k: source or FakeSource())
        run_generate(config)
    return Manifest.load(config.writer.staging_dir / "manifest.json")


def load_coco(path: Path) -> Dict[str, Any]:
    data: Dict[str, Any] = json.loads(path.read_text(encoding="utf-8"))
    return data


def annotations_by_image(coco: Dict[str, Any]) -> Dict[int, List[Dict[str, Any]]]:
    grouped: Dict[int, List[Dict[str, Any]]] = {}
    for ann in coco["annotations"]:
        grouped.setdefault(ann["image_id"], []).append(ann)
    return grouped


def stored_mask(ann: Dict[str, Any]) -> npt.NDArray[np.bool_]:
    height, width = ann["segmentation"]["size"]
    return decode(ann["segmentation"]["counts"], height, width)


# ── the independent mask computation (rasterio / GDAL, shapely) ─────────────


@dataclass
class Expected:
    category: int
    mask: npt.NDArray[np.bool_]
    truncated: bool
    borderline: bool  # within rounding of a threshold: either outcome is right


def reference_mask(
    world: BaseGeometry,
    valid: npt.NDArray[np.bool_],
    row: int,
    col: int,
    all_touched: bool,
    patch: int = PATCH,
) -> npt.NDArray[np.bool_]:
    """GDAL's mask of ``world`` on the patch grid, over the raster's pixels that have imagery."""
    transform = Affine(RES, 0.0, X0 + col * RES, 0.0, -RES, Y0 - row * RES)
    burned = rasterio_features.rasterize(
        [(world, 1)], out_shape=(patch, patch), transform=transform, all_touched=all_touched
    )
    padded = np.zeros((HEIGHT + 2 * patch, WIDTH + 2 * patch), dtype=bool)
    padded[patch : patch + HEIGHT, patch : patch + WIDTH] = valid
    over_imagery = padded[patch + row : patch + row + patch, patch + col : patch + col + patch]
    return np.asarray(burned, dtype=bool) & over_imagery


def invalid_world_region(
    valid: npt.NDArray[np.bool_], row: int, col: int
) -> Optional[BaseGeometry]:
    """Pixels without imagery inside the patch, as a union of one box per pixel (world)."""
    r0, c0 = max(row, 0), max(col, 0)
    r1, c1 = min(row + PATCH, HEIGHT), min(col + PATCH, WIDTH)
    rows, cols = np.nonzero(~valid[r0:r1, c0:c1])
    if not len(rows):
        return None
    xs = X0 + (cols + c0) * RES
    ys = Y0 - (rows + r0) * RES
    return shapely.union_all(shapely.box(xs, ys - RES, xs + RES, ys))


def expected_instances(
    features: List[Feature],
    valid: npt.NDArray[np.bool_],
    row: int,
    col: int,
    options: InstanceOptions,
    all_touched: bool = False,
) -> List[Expected]:
    left, top = X0 + col * RES, Y0 - row * RES
    patch = box(left, top - PATCH * RES, left + PATCH * RES, top)
    raster = box(X0, Y0 - HEIGHT * RES, X0 + WIDTH * RES, Y0)
    invalid = invalid_world_region(valid, row, col)
    result: List[Expected] = []
    for feature in features:
        world = feature.world
        visible = world.intersection(patch).intersection(raster)
        if invalid is not None:
            visible = visible.difference(invalid)
        visible = shapely.union_all(
            [part for part in shapely.get_parts(visible) if part.geom_type == "Polygon"]
        )
        if visible.is_empty or visible.area <= 0:
            continue
        fraction = visible.area / world.area
        mask = reference_mask(world, valid, row, col, all_touched)
        pixels = int(mask.sum())
        whole = fraction > 1.0 - 1e-9
        shortfall = options.min_visible - fraction
        borderline = 1e-12 < abs(shortfall) < 1e-6
        keep = (whole or shortfall <= 1e-12) and pixels >= options.min_area
        if keep or borderline:
            result.append(Expected(CLASS_IDS[feature.name], mask, not whole, borderline))
    return result


def assert_matches(found: List[Dict[str, Any]], expected: List[Expected], where: str) -> int:
    """Every kept annotation is an expected mask (same order); returns how many matched."""
    unmatched = list(found)
    last = -1
    matched = 0
    for item in expected:
        match = next(
            (
                (position, ann)
                for position, ann in enumerate(unmatched)
                if ann["category_id"] == item.category
                and np.array_equal(stored_mask(ann), item.mask)
            ),
            None,
        )
        if match is None:
            assert item.borderline, f"{where}: missing {item.category} with {item.mask.sum()} px"
            continue
        position, ann = match
        index = found.index(ann)
        assert index > last, f"{where}: annotations are not in feature order"
        last = index
        unmatched.remove(ann)
        assert ann["truncated"] == item.truncated, (where, ann["id"])
        matched += 1
    assert not unmatched, f"{where}: unexpected annotations {[a['id'] for a in unmatched]}"
    return matched


@pytest.fixture(scope="module")
def split_dataset(
    tmp_path_factory: pytest.TempPathFactory,
) -> Iterator[Tuple[MapcvConfig, List[Feature]]]:
    root = tmp_path_factory.mktemp("split")
    features = random_features(seed=11)
    write_labels(root / "labels.geojson", features)
    config = instance_config(
        root,
        instance={"id_mask": True},
        split={"strategy": "random", "test_ratio": 0.25, "val_ratio": 0.2},
    )
    generate(config)
    yield config, features


# ── masks against rasterio ──────────────────────────────────────────────────


@pytest.mark.parametrize(
    ("options", "mask_kind", "all_touched"),
    [
        ({}, "edges", False),
        ({}, "noisy", True),
        ({"min_visible": 0.0, "min_area": 1}, "edges", False),
        ({"min_visible": 0.0, "min_area": 1}, "full", True),
        ({"min_visible": 0.0, "min_area": 1}, "noisy", False),
        ({"min_visible": 0.75, "min_area": 12}, "noisy", False),
        ({"min_visible": 1.0}, "edges", True),
    ],
    ids=[
        "defaults",
        "defaults-noisy-all-touched",
        "keep-everything",
        "keep-everything-all-touched",
        "keep-everything-noisy",
        "strict-noisy",
        "only-whole-all-touched",
    ],
)
def test_masks_equal_gdal_rasterization_of_each_feature(
    tmp_path: Path, options: Dict[str, Any], mask_kind: str, all_touched: bool
) -> None:
    features = random_features(seed=3, count=100)
    write_labels(tmp_path / "labels.geojson", features)
    valid = make_valid_mask(mask_kind)
    config = instance_config(
        tmp_path, instance=options, labels={"all_touched": all_touched}, staging="ds"
    )
    manifest = generate(config, FakeSource(valid))
    coco = load_coco(config.writer.staging_dir / "annotations" / "instances_all.json")
    grouped = annotations_by_image(coco)
    class_map = manifest.class_map
    assert class_map == CLASS_IDS
    matched = 0
    for index, entry in enumerate(manifest.patches):
        expected = expected_instances(
            features, valid, entry["row"], entry["col"], config.instance_options, all_touched
        )
        matched += assert_matches(
            grouped.get(index + 1, []), expected, f"patch {index} ({entry['row']}, {entry['col']})"
        )
    assert matched >= (100 if options.get("min_visible") == 0.0 else 3)


def test_coco_files_load_in_pycocotools_and_every_annotation_decodes_to_its_mask(
    split_dataset: Tuple[MapcvConfig, List[Feature]],
) -> None:
    config, features = split_dataset
    staging = config.writer.staging_dir
    manifest = Manifest.load(staging / "manifest.json")
    valid = make_valid_mask()
    seen = 0
    for split in ("train", "val", "test"):
        path = staging / "annotations" / f"instances_{split}.json"
        api = pycocotools_coco.COCO(str(path))
        names = (staging / "splits" / f"{split}.txt").read_text().split()
        assert sorted(api.imgs[i]["file_name"] for i in api.getImgIds()) == sorted(names)
        assert api.getCatIds() == [1, 2, 3]
        for image_id in api.getImgIds():
            entry = manifest.patches[image_id - 1]
            expected = expected_instances(
                features, valid, entry["row"], entry["col"], config.instance_options
            )
            anns = api.loadAnns(api.getAnnIds(imgIds=[image_id]))
            assert_matches(anns, expected, f"image {image_id}")
            for ann in anns:
                mask = api.annToMask(ann)
                rle = pycocotools_mask.encode(np.asfortranarray(mask))
                assert ann["segmentation"]["size"] == [PATCH, PATCH]
                assert ann["area"] == int(pycocotools_mask.area(rle)) == int(mask.sum())
                assert ann["bbox"] == [int(v) for v in pycocotools_mask.toBbox(rle)]
                assert ann["iscrowd"] == 0 and isinstance(ann["truncated"], bool)
                assert ann["segmentation"]["counts"] == reference_counts(mask.astype(bool))
                seen += 1
    assert seen > 100


def strict_coco_check(coco: Dict[str, Any], patch_size: int) -> None:
    assert set(coco) == {"info", "licenses", "categories", "images", "annotations"}
    category_ids = [cat["id"] for cat in coco["categories"]]
    assert category_ids == sorted(set(category_ids))
    image_ids = set()
    for image in coco["images"]:
        assert set(image) == {"id", "file_name", "width", "height"}
        assert image["id"] not in image_ids
        image_ids.add(image["id"])
        assert image["width"] == image["height"] == patch_size
        assert image["file_name"] == f"patch_{image['id'] - 1:07d}.png"
    ann_ids = set()
    for ann in coco["annotations"]:
        assert set(ann) == {
            "id",
            "image_id",
            "category_id",
            "segmentation",
            "bbox",
            "area",
            "iscrowd",
            "truncated",
        }
        assert isinstance(ann["id"], int) and ann["id"] >= 1 and ann["id"] not in ann_ids
        ann_ids.add(ann["id"])
        assert ann["image_id"] in image_ids and ann["category_id"] in category_ids
        assert set(ann["segmentation"]) == {"size", "counts"}
        assert ann["segmentation"]["size"] == [patch_size, patch_size]
        assert isinstance(ann["segmentation"]["counts"], str)
        x, y, w, h = ann["bbox"]
        assert all(type(v) is int for v in ann["bbox"]) and type(ann["area"]) is int
        assert w > 0 and h > 0 and x >= 0 and y >= 0
        assert x + w <= patch_size and y + h <= patch_size
        assert 0 < ann["area"] <= w * h


def test_coco_documents_pass_a_strict_schema_check_and_ids_are_stable(
    split_dataset: Tuple[MapcvConfig, List[Feature]], tmp_path: Path
) -> None:
    config, _ = split_dataset
    staging = tmp_path / "copy"
    import shutil

    shutil.copytree(config.writer.staging_dir, staging)
    seen: Dict[int, Dict[str, Any]] = {}
    for name in ("train", "val", "test"):
        coco = load_coco(staging / "annotations" / f"instances_{name}.json")
        strict_coco_check(coco, PATCH)
        for ann in coco["annotations"]:
            assert ann["id"] not in seen
            seen[ann["id"]] = ann
    # A re-split moves annotations between files, keeping their IDs.
    run_split(staging, SplitterConfig(strategy="random", seed=5, test_ratio=0.4))
    for name in ("train", "val", "test"):
        listed = (staging / "splits" / f"{name}.txt").read_text().split()
        coco = load_coco(staging / "annotations" / f"instances_{name}.json")
        assert sorted(image["file_name"] for image in coco["images"]) == sorted(listed)
        for ann in coco["annotations"]:
            assert seen[ann["id"]] == ann


def test_instance_summary_counts_objects_per_class(
    split_dataset: Tuple[MapcvConfig, List[Feature]],
) -> None:
    config, _ = split_dataset
    staging = config.writer.staging_dir
    manifest = Manifest.load(staging / "manifest.json")
    by_image: Dict[int, List[Dict[str, Any]]] = {}
    for name in ("train", "val", "test"):
        for image_id, anns in annotations_by_image(
            load_coco(staging / "annotations" / f"instances_{name}.json")
        ).items():
            by_image[image_id] = anns
    assert any(not entry["summary"]["class_objects"] for entry in manifest.patches)
    for index, entry in enumerate(manifest.patches):
        counts: Dict[str, int] = {}
        for ann in by_image.get(index + 1, []):
            counts[str(ann["category_id"])] = counts.get(str(ann["category_id"]), 0) + 1
        assert entry["summary"]["class_objects"] == dict(
            sorted(counts.items(), key=lambda i: int(i[0]))
        )
        assert list(entry["summary"]) == ["class_objects", "empty_ratio"]


# ── overlaps, holes and the patch edge ──────────────────────────────────────


def world_box(col0: float, row0: float, col1: float, row1: float) -> Polygon:
    """A rectangle given in raster pixels (col, row), in Web Mercator metres."""
    return box(X0 + col0 * RES, Y0 - row1 * RES, X0 + col1 * RES, Y0 - row0 * RES)


def test_overlapping_instances_are_each_exact_and_the_id_mask_resolves_to_the_later_one(
    tmp_path: Path,
) -> None:
    first = world_box(10.2, 10.3, 40.4, 38.1)
    second = world_box(25.6, 20.2, 55.1, 50.7)  # overlaps the first
    third = world_box(28.3, 24.4, 36.2, 31.4)  # inside both
    inside_hole = Polygon(
        world_box(2, 2, 60, 60).exterior.coords, [world_box(20, 20, 45, 45).exterior.coords]
    )
    features = [
        Feature(first, "building"),
        Feature(second, "car"),
        Feature(third, "tree"),
        Feature(inside_hole, "building"),
    ]
    write_labels(tmp_path / "labels.geojson", features)
    config = instance_config(
        tmp_path,
        instance={"id_mask": True, "min_visible": 0.0, "min_area": 1},
        sampler={"edge_strategy": "drop"},
    )
    valid = make_valid_mask("full")
    manifest = generate(config, FakeSource(valid))
    staging = config.writer.staging_dir
    coco = load_coco(staging / "annotations" / "instances_all.json")
    entry = manifest.patches[0]
    assert (entry["row"], entry["col"]) == (0, 0)
    anns = annotations_by_image(coco)[1]
    assert [ann["category_id"] for ann in anns] == [1, 2, 3, 1]
    masks = [stored_mask(ann) for ann in anns]
    for feature, mask in zip(features, masks):
        assert np.array_equal(mask, reference_mask(feature.world, valid, 0, 0, False))
        assert not mask.all() and mask.any()
    assert (masks[0] & masks[1]).any() and (masks[1] & masks[2]).any()  # overlapping, all exact
    assert not masks[3][30, 30] and masks[3][5, 5]  # the hole is a hole
    assert (
        anns[3]["area"]
        == int(masks[3].sum())
        < int(reference_mask(box(*inside_hole.bounds), valid, 0, 0, False).sum())
    )

    # The ID mask paints the instances in annotation order: the later one wins.
    painted = np.zeros((PATCH, PATCH), dtype=np.uint16)
    for number, mask in enumerate(masks, start=1):
        painted[mask] = number
    ids = np.array(Image.open(staging / entry["files"]["mask"]))
    assert ids.dtype == np.uint16 and ids.shape == (PATCH, PATCH)
    assert np.array_equal(ids, painted)
    assert set(np.unique(ids).tolist()) == {0, 1, 2, 3, 4}
    assert entry["files"]["mask"] == "masks/patch_0000000.png"
    assert Image.open(staging / entry["files"]["mask"]).mode == "I;16"


def test_a_feature_cut_by_patch_edges_is_clipped_flagged_and_adds_up(tmp_path: Path) -> None:
    # One polygon over the corner shared by four patches, one wholly inside a patch.
    crossing = Polygon(
        [
            (X0 + 50.3 * RES, Y0 - 52.2 * RES),
            (X0 + 81.7 * RES, Y0 - 48.9 * RES),
            (X0 + 77.1 * RES, Y0 - 80.4 * RES),
            (X0 + 55.5 * RES, Y0 - 83.3 * RES),
        ]
    )
    small = world_box(5.2, 5.2, 12.8, 14.1)
    features = [Feature(crossing, "building"), Feature(small, "car")]
    write_labels(tmp_path / "labels.geojson", features)
    config = instance_config(
        tmp_path, instance={"min_visible": 0.0, "min_area": 1}, sampler={"edge_strategy": "drop"}
    )
    valid = make_valid_mask("full")
    manifest = generate(config, FakeSource(valid))
    coco = load_coco(config.writer.staging_dir / "annotations" / "instances_all.json")
    canvas = np.zeros((HEIGHT + PATCH, WIDTH + PATCH), dtype=bool)
    pieces = 0
    flags = {}
    for ann in coco["annotations"]:
        entry = manifest.patches[ann["image_id"] - 1]
        if ann["category_id"] == 1:
            r, c = entry["row"], entry["col"]
            canvas[r : r + PATCH, c : c + PATCH] |= stored_mask(ann)
            pieces += 1
            flags[(r, c)] = ann["truncated"]
        else:
            assert ann["truncated"] is False and ann["area"] > 0
    assert pieces == 4 and set(flags.values()) == {True}
    whole = np.asarray(
        rasterio_features.rasterize(
            [(crossing, 1)],
            out_shape=(HEIGHT + PATCH, WIDTH + PATCH),
            transform=Affine(RES, 0.0, X0, 0.0, -RES, Y0),
        ),
        dtype=bool,
    )
    assert np.array_equal(canvas, whole)  # the clipped pieces tile the exact feature mask


def test_instances_painted_in_feature_order_equal_the_segmentation_mask(tmp_path: Path) -> None:
    features = random_features(seed=17, count=120)
    write_labels(tmp_path / "labels.geojson", features)
    options = {"min_visible": 0.0, "min_area": 1}
    for all_touched in (False, True):
        labels = {"all_touched": all_touched}
        inst = instance_config(tmp_path, instance=options, labels=labels, staging=f"i{all_touched}")
        seg = instance_config(
            tmp_path,
            labels={**labels, "ignore_index": None},
            staging=f"s{all_touched}",
            task="segmentation",
        )
        valid = make_valid_mask("noisy")
        inst_manifest = generate(inst, FakeSource(valid))
        seg_manifest = generate(seg, FakeSource(valid))
        coco = load_coco(inst.writer.staging_dir / "annotations" / "instances_all.json")
        grouped = annotations_by_image(coco)
        assert [(e["row"], e["col"]) for e in inst_manifest.patches] == [
            (e["row"], e["col"]) for e in seg_manifest.patches
        ]
        compared = 0
        for index, entry in enumerate(seg_manifest.patches):
            semantic = np.array(Image.open(seg.writer.staging_dir / entry["files"]["mask"]))
            painted = np.zeros((PATCH, PATCH), dtype=np.uint8)
            for ann in grouped.get(index + 1, []):
                painted[stored_mask(ann)] = ann["category_id"]
            r, c = entry["row"], entry["col"]
            padded = np.zeros((HEIGHT + 2 * PATCH, WIDTH + 2 * PATCH), dtype=bool)
            padded[PATCH : PATCH + HEIGHT, PATCH : PATCH + WIDTH] = valid
            over_imagery = padded[PATCH + r : 2 * PATCH + r, PATCH + c : 2 * PATCH + c]
            # Segmentation without an ignore value keeps labels over missing imagery; instances
            # keep only pixels over imagery.
            assert np.array_equal(semantic[over_imagery], painted[over_imagery])
            assert not painted[~over_imagery].any()
            compared += int(painted.any())
        assert compared > 10


def test_min_visible_and_min_area_filter_instances(tmp_path: Path) -> None:
    # A 20 x 20 px square whose left 6 px (30%) are in the patch at col 0, 14 px in the next.
    square = world_box(58.0, 10.0, 78.0, 30.0)
    sliver = world_box(10.1, 40.1, 12.3, 41.3)  # 2 x 1 px by their centres
    write_labels(tmp_path / "labels.geojson", [Feature(square, "building"), Feature(sliver, "car")])
    valid = make_valid_mask("full")

    def kept(options: Dict[str, Any], name: str) -> List[Tuple[int, int, int]]:
        """(patch column, class, area) of every instance, sorted."""
        config = instance_config(
            tmp_path, instance=options, sampler={"edge_strategy": "drop"}, staging=name
        )
        manifest = generate(config, FakeSource(valid))
        coco = load_coco(config.writer.staging_dir / "annotations" / "instances_all.json")
        return sorted(
            (manifest.patches[a["image_id"] - 1]["col"], a["category_id"], a["area"])
            for a in coco["annotations"]
        )

    everything = kept({"min_visible": 0.0, "min_area": 1}, "all")
    assert everything == [(0, 1, 6 * 20), (0, 2, 2), (64, 1, 14 * 20)]
    assert kept({"min_visible": 0.3, "min_area": 1}, "vis30") == everything  # 30% is kept
    assert kept({"min_visible": 0.31, "min_area": 1}, "vis31") == [(0, 2, 2), (64, 1, 14 * 20)]
    assert kept({"min_visible": 0.0, "min_area": 3}, "area3") == [(0, 1, 6 * 20), (64, 1, 14 * 20)]
    assert kept({"min_visible": 0.0, "min_area": 121}, "area121") == [(64, 1, 14 * 20)]
    assert kept({"min_visible": 1.0, "min_area": 1}, "whole") == [(0, 2, 2)]


def test_truncation_by_missing_imagery(tmp_path: Path) -> None:
    square = world_box(20.0, 20.0, 40.0, 40.0)
    write_labels(tmp_path / "labels.geojson", [Feature(square, "building")])
    valid = make_valid_mask("full")
    valid[20:40, 30:] = False  # the right half of the square has no imagery
    config = instance_config(
        tmp_path, instance={"min_visible": 0.0}, sampler={"edge_strategy": "drop"}
    )
    manifest = generate(config, FakeSource(valid))
    coco = load_coco(config.writer.staging_dir / "annotations" / "instances_all.json")
    (ann,) = coco["annotations"]
    assert ann["truncated"] is True and ann["area"] == 10 * 20 and ann["bbox"] == [20, 20, 10, 20]
    assert manifest.patches[ann["image_id"] - 1]["summary"]["class_objects"] == {"1": 1}


def test_invalid_polygons_are_repaired(tmp_path: Path) -> None:
    left, top = X0 + 10 * RES, Y0 - 10 * RES
    bowtie = [(0, 0), (20, 0), (0, 20), (20, 20), (0, 0)]  # two triangles meeting at (10, 10)
    ring = [[*to_lonlat(left + x * RES, top - y * RES)] for x, y in bowtie]
    (tmp_path / "labels.geojson").write_text(
        json.dumps(
            {
                "type": "FeatureCollection",
                "features": [
                    {
                        "type": "Feature",
                        "properties": {"class": "tree"},
                        "geometry": {"type": "Polygon", "coordinates": [ring]},
                    }
                ],
            }
        )
    )
    config = instance_config(tmp_path, sampler={"edge_strategy": "drop"})
    generate(config, FakeSource(make_valid_mask("full")))
    coco = load_coco(config.writer.staging_dir / "annotations" / "instances_all.json")
    (ann,) = coco["annotations"]
    mask = stored_mask(ann)
    # A bowtie of two triangles: the top one and the bottom one, nothing at its sides.
    assert mask[12, 20] and mask[28, 20] and not mask[20, 13] and not mask[20, 27]
    assert ann["area"] == int(mask.sum()) and 150 < ann["area"] < 250


# ── a rotated grid, straight against rasterio ───────────────────────────────


@pytest.mark.parametrize("all_touched", [False, True])
@pytest.mark.parametrize("rotation", [0.0, 17.0, -40.0])
def test_window_masks_match_rasterio_on_rotated_grids(rotation: float, all_touched: bool) -> None:
    angle = math.radians(rotation)
    cos, sin = math.cos(angle), math.sin(angle)
    transform = (0.5 * cos, 0.5 * sin, 1000.0, 0.5 * sin, -0.5 * cos, 2000.0)
    if rotation == 0.0:
        transform = (0.5, 0.0, 1000.0, 0.0, -0.5, 2000.0)
    affine = Affine(*transform[:3], *transform[3:])
    height, width, size = 120, 150, 40
    rng = random.Random(4)
    pixel_polygons = []
    for _ in range(40):
        cx, cy = rng.uniform(-10, width + 10), rng.uniform(-10, height + 10)
        pixel_polygons.append(star(rng, cx, cy, rng.uniform(2, 30)))
    world = np.empty(len(pixel_polygons), dtype=object)
    world[:] = [
        shapely.transform(polygon, lambda xy: np.array([affine * tuple(p) for p in xy]))
        for polygon in pixel_polygons
    ]
    window = InstanceWindow(
        to_pixels(world, transform),
        world,
        np.ones(len(world), dtype=np.int64),
        transform,
        height,
        width,
        InstanceOptions(min_visible=0.0, min_area=1),
        all_touched,
    )
    compared = 0
    for row in (-15, 0, 33, 80):
        for col in (-20, 0, 41, 110):
            found = window.annotate(row, col, size, "zero", None)
            patch_transform = affine * Affine.translation(col, row)
            frame = np.zeros((size, size), dtype=bool)
            frame[max(-row, 0) : min(height - row, size), max(-col, 0) : min(width - col, size)] = (
                True
            )
            expected = []
            for geometry in world:
                burned = rasterio_features.rasterize(
                    [(geometry, 1)],
                    out_shape=(size, size),
                    transform=patch_transform,
                    all_touched=all_touched,
                )
                mask = np.asarray(burned, dtype=bool) & frame
                if mask.any():
                    expected.append(mask)
            # Features that only touch a patch (zero visible area) are not instances.
            assert len(found.instances) <= len(expected)
            decoded = [decode(item.counts, size, size) for item in found.instances]
            remaining = list(expected)
            for mask in decoded:
                position = next(
                    i for i, other in enumerate(remaining) if np.array_equal(other, mask)
                )
                remaining.pop(position)
                compared += 1
            for mask in remaining:  # what is left is only touched by the polygon, not covered
                assert mask.sum() < 40 or all_touched
    assert compared > 50


@pytest.mark.parametrize("all_touched", [False, True])
def test_pixel_centres_on_an_edge_follow_gdal_in_shifted_patches(all_touched: bool) -> None:
    identity = (1.0, 0.0, 0.0, 0.0, 1.0, 0.0)
    height, width, size = 48, 56, 20
    # Vertices and edges on whole and half pixels: pixel centres sit exactly on edges.
    rng = random.Random(6)
    polygons = []
    for _ in range(60):
        x0, y0 = rng.randrange(-8, 2 * width), rng.randrange(-8, 2 * height)
        w, h = rng.randrange(1, 30), rng.randrange(1, 30)
        polygon = box(x0 / 2, y0 / 2, x0 / 2 + w / 2, y0 / 2 + h / 2)
        if rng.random() < 0.4:  # a triangle half of the box, diagonal through pixel centres
            polygon = Polygon(
                [polygon.bounds[:2], polygon.bounds[2:3] + polygon.bounds[1:2], polygon.bounds[2:]]
            )
        polygons.append(polygon)
    geometries = np.empty(len(polygons), dtype=object)
    geometries[:] = polygons
    window = InstanceWindow(
        geometries,
        geometries,
        np.ones(len(polygons), dtype=np.int64),
        identity,
        height,
        width,
        InstanceOptions(min_visible=0.0, min_area=1),
        all_touched,
    )
    compared = 0
    for row in (-7, 0, 5, 13, 28):
        for col in (-5, 0, 7, 19, 36):
            found = window.annotate(row, col, size, "zero", None)
            frame = np.zeros((size, size), dtype=bool)
            frame[max(-row, 0) : min(height - row, size), max(-col, 0) : min(width - col, size)] = (
                True
            )
            transform = Affine(1.0, 0.0, col, 0.0, 1.0, row)
            decoded = [decode(item.counts, size, size) for item in found.instances]
            for polygon in polygons:
                burned = rasterio_features.rasterize(
                    [(polygon, 1)],
                    out_shape=(size, size),
                    transform=transform,
                    all_touched=all_touched,
                )
                mask = np.asarray(burned, dtype=bool) & frame
                visible = polygon.intersection(box(col, row, col + size, row + size)).intersection(
                    box(0, 0, width, height)
                )
                if mask.any() and visible.area > 0:
                    assert any(np.array_equal(mask, other) for other in decoded), (row, col)
                    compared += 1
    assert compared > 100


def write_geotiff(directory: Path, rotation: float) -> Tuple[Any, Dict[str, float]]:
    """A 400 x 300 px RGB UTM GeoTIFF near Paris with a NoData (0) block, and a region in it."""
    import rasterio
    from pyproj import Transformer

    width, height = 400, 300
    to_utm = Transformer.from_crs("EPSG:4326", "EPSG:32631", always_xy=True)
    center_x, center_y = to_utm.transform(2.35, 48.85)
    transform = Affine(1.0, 0.0, center_x - width / 2, 0.0, -1.0, center_y + height / 2)
    if rotation:
        transform = transform * Affine.rotation(rotation, (width / 2, height / 2))
    data = np.random.default_rng(3).integers(1, 250, size=(3, height, width)).astype("uint8")
    data[:, 40:90, 200:260] = 0
    path = directory / "scene.tif"
    with rasterio.open(
        path,
        "w",
        driver="GTiff",
        height=height,
        width=width,
        count=3,
        dtype="uint8",
        crs="EPSG:32631",
        transform=transform,
        nodata=0,
    ) as dst:
        dst.write(data)
    to_lonlat_utm = Transformer.from_crs("EPSG:32631", "EPSG:4326", always_xy=True)
    corners = [transform * (col, row) for col in (0, width) for row in (0, height)]
    lons, lats = to_lonlat_utm.transform([x for x, _ in corners], [y for _, y in corners])
    west, east, south, north = min(lons), max(lons), min(lats), max(lats)
    shrink = 0.25 if rotation else 0.05
    dx, dy = (east - west) * shrink, (north - south) * shrink
    region = {"west": west + dx, "south": south + dy, "east": east - dx, "north": north - dy}
    return (path, transform, data), region


@pytest.mark.parametrize("rotation", [0.0, 12.0])
def test_geotiff_masks_equal_rasterio_in_the_file_crs(tmp_path: Path, rotation: float) -> None:
    """Labels reprojected into the file's UTM grid, a rotated one too, with a NoData block."""
    from pyproj import Transformer

    (path, file_transform, data), region = write_geotiff(tmp_path, rotation)
    rng = random.Random(1)
    west, south = region["west"], region["south"]
    dx, dy = region["east"] - west, region["north"] - south
    features: List[Dict[str, Any]] = []
    for index in range(14):
        fx, fy = rng.uniform(0.0, 0.9), rng.uniform(0.0, 0.9)
        size = rng.uniform(0.05, 0.25)
        ring = [
            [west + (fx + size * px) * dx, south + (fy + size * py) * dy]
            for px, py in ((0, 0), (1, 0.1), (0.8, 1), (0.1, 0.7), (0, 0))
        ]
        features.append(
            {
                "type": "Feature",
                "properties": {"kind": "a" if index % 2 else "b"},
                "geometry": {"type": "Polygon", "coordinates": [ring]},
            }
        )
    labels = tmp_path / "labels.geojson"
    labels.write_text(json.dumps({"type": "FeatureCollection", "features": features}))
    config = MapcvConfig.model_validate(
        {
            "task": "instance",
            "region": region,
            "imagery": {"type": "geotiff", "path": str(path)},
            "labels": {"path": str(labels), "label_field": "kind"},
            "instance": {"min_visible": 0.0, "min_area": 1},
            "sampler": {"patch_size": 48, "edge_strategy": "drop"},
            "writer": {"staging_dir": str(tmp_path / "dataset")},
        }
    )
    run_generate(config)
    staging = config.writer.staging_dir
    manifest = Manifest.load(staging / "manifest.json")
    coco = load_coco(staging / "annotations" / "instances_all.json")
    grouped = annotations_by_image(coco)
    to_utm = Transformer.from_crs("EPSG:4326", "EPSG:32631", always_xy=True)
    worlds = [
        shapely.transform(
            shapely.geometry.shape(feature["geometry"]),
            lambda xy: np.column_stack(to_utm.transform(xy[:, 0], xy[:, 1])),
        )
        for feature in features
    ]
    # Where the dataset raster starts in the file (a whole number of pixels), for NoData.
    a, b, c, d, e, f = manifest.source.transform or (0.0,) * 6
    t = file_transform
    det = t.a * t.e - t.b * t.d
    offset_col = round(((c - t.c) * t.e - (f - t.f) * t.b) / det)
    offset_row = round(((f - t.f) * t.a - (c - t.c) * t.d) / det)
    valid_file = (data != 0).any(axis=0)
    checked = 0
    for index, entry in enumerate(manifest.patches):
        row, col = entry["row"], entry["col"]
        valid = valid_file[
            row + offset_row : row + offset_row + 48, col + offset_col : col + offset_col + 48
        ]
        transform = Affine(*manifest.patch_transform(entry))
        expected = []
        for world in worlds:
            burned = np.asarray(
                rasterio_features.rasterize([(world, 1)], out_shape=(48, 48), transform=transform),
                dtype=bool,
            )
            burned &= valid
            if burned.any():
                expected.append(burned)
        found = [stored_mask(ann) for ann in grouped.get(index + 1, [])]
        assert len(found) == len(expected), (index, len(found), len(expected))
        for mask, want in zip(found, expected):
            assert np.array_equal(mask, want)
            checked += 1
    assert checked >= 10


# ── patch bookkeeping: summaries, ratios, the 16-bit limit ──────────────────


def test_min_label_ratio_drops_patches_without_enough_instances(tmp_path: Path) -> None:
    write_labels(tmp_path / "labels.geojson", random_features(seed=6, count=40))
    config = instance_config(tmp_path, sampler={"min_label_ratio": 0.02})
    manifest = generate(config)
    assert manifest.patches and len(manifest.patches) < 30
    staging = config.writer.staging_dir
    coco = load_coco(staging / "annotations" / "instances_all.json")
    grouped = annotations_by_image(coco)
    for index, entry in enumerate(manifest.patches):
        assert entry["summary"]["class_objects"]
        union = np.zeros((PATCH, PATCH), dtype=bool)
        for ann in grouped[index + 1]:
            union |= stored_mask(ann)
        assert union.sum() / PATCH**2 >= 0.02  # overlaps are counted once


def test_the_instance_id_limit_is_the_16_bit_maximum_and_overflow_is_an_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    assert MAX_INSTANCE_ID == int(np.iinfo(np.uint16).max)
    monkeypatch.setattr("mapcv.targets.instance.MAX_INSTANCE_ID", 3)
    size = 16
    squares = np.empty(5, dtype=object)
    squares[:] = [box(2 * i, 2 * i, 2 * i + 1, 2 * i + 1) for i in range(5)]
    identity = (1.0, 0.0, 0.0, 0.0, 1.0, 0.0)

    def window(id_mask: bool) -> InstanceWindow:
        return InstanceWindow(
            squares,
            squares,
            np.ones(5, dtype=np.int64),
            identity,
            size,
            size,
            InstanceOptions(min_area=1, id_mask=id_mask),
            False,
        )

    with pytest.raises(ValueError, match="more than 3 instances.*instance.id_mask: false"):
        window(True).annotate(0, 0, size, "zero", None)
    assert len(window(False).annotate(0, 0, size, "zero", None).instances) == 5


def test_a_patch_without_labels_has_no_instances_and_a_zero_id_mask(tmp_path: Path) -> None:
    far = world_box(10, 10, 20, 20)
    write_labels(tmp_path / "labels.geojson", [Feature(far, "tree")])
    config = instance_config(
        tmp_path, instance={"id_mask": True}, sampler={"edge_strategy": "drop"}
    )
    manifest = generate(config, FakeSource(make_valid_mask("full")))
    staging = config.writer.staging_dir
    empty = [e for e in manifest.patches if not e["summary"]["class_objects"]]
    assert len(empty) == len(manifest.patches) - 1
    ids = np.array(Image.open(staging / empty[0]["files"]["mask"]))
    assert ids.shape == (PATCH, PATCH) and not ids.any()
    coco = load_coco(staging / "annotations" / "instances_all.json")
    assert len(coco["images"]) == len(manifest.patches) and len(coco["annotations"]) == 1


# ── determinism, resuming, stratification ───────────────────────────────────


def snapshot(staging: Path) -> Dict[str, bytes]:
    return {
        path.relative_to(staging).as_posix(): path.read_bytes()
        for path in sorted(staging.rglob("*"))
        if path.is_file()
    }


def test_two_runs_write_identical_bytes(tmp_path: Path) -> None:
    write_labels(tmp_path / "labels.geojson", random_features(seed=2, count=100))
    snapshots = []
    for name in ("one", "two"):
        config = instance_config(
            tmp_path, instance={"id_mask": True}, split={"strategy": "spatial"}, staging=name
        )
        generate(config)
        snapshots.append(snapshot(config.writer.staging_dir))
    assert snapshots[0] == snapshots[1]
    assert any(name.startswith("masks/") for name in snapshots[0])


def test_resume_reproduces_the_uninterrupted_dataset(tmp_path: Path) -> None:
    write_labels(tmp_path / "labels.geojson", random_features(seed=9))
    config = instance_config(tmp_path, instance={"id_mask": True}, split={"strategy": "spatial"})
    manifest = generate(config)
    staging = config.writer.staging_dir
    before = snapshot(staging)
    generate(config)  # a finished dataset: nothing to do, nothing changes
    assert snapshot(staging) == before

    # Lose the second half: its chunks' instances and the manifest rows.
    partial = Manifest.load(staging / "manifest.json")
    kept = len(partial.patches) // 2
    partial.patches = partial.patches[:kept]
    partial.save(staging / "manifest.json")
    kept_chunks = {entry["chunk"] for entry in partial.patches}
    for entry in manifest.patches[kept:]:
        if entry["chunk"] not in kept_chunks:
            (staging / "annotations" / "objects" / f"chunk_{entry['chunk']:06d}.json").unlink(
                missing_ok=True
            )
    (staging / "annotations" / "instances_train.json").unlink()
    generate(config)
    assert snapshot(staging) == before


def test_resume_refuses_changed_instance_options(tmp_path: Path) -> None:
    write_labels(tmp_path / "labels.geojson", random_features(seed=1, count=20))
    generate(instance_config(tmp_path))
    for options in ({"min_visible": 0.5}, {"min_area": 9}, {"id_mask": True}):
        with pytest.raises(ManifestMismatchError, match="task options"):
            generate(instance_config(tmp_path, instance=options))
    with pytest.raises(ManifestMismatchError, match="labels"):
        generate(instance_config(tmp_path, labels={"all_touched": True}))
    with pytest.raises(ManifestMismatchError, match="task"):
        generate(instance_config(tmp_path, task="detection"))


def test_split_after_generation_rebuilds_the_coco_files(tmp_path: Path) -> None:
    write_labels(tmp_path / "labels.geojson", random_features(seed=4, count=60))
    config = instance_config(tmp_path, split={"strategy": "random", "seed": 1})
    generate(config)
    staging = config.writer.staging_dir
    run_split(staging, SplitterConfig(strategy="random", seed=2, test_ratio=0.5))
    test_names = (staging / "splits" / "test.txt").read_text().split()
    coco = load_coco(staging / "annotations" / "instances_test.json")
    assert sorted(image["file_name"] for image in coco["images"]) == sorted(test_names)
    assert not (staging / "dataset.yaml").exists() and not (staging / "labels").exists()


def test_stratification_uses_instance_classes_and_balances_them(tmp_path: Path) -> None:
    def entry(objects: Dict[str, int]) -> Any:
        return {"summary": {"class_objects": objects, "empty_ratio": 0.0}}

    assert _stratum(entry({})) == (0, "")
    assert _stratum(entry({"1": 2, "3": 5})) == (1, "3")
    write_labels(tmp_path / "labels.geojson", random_features(seed=5, count=200))
    config = instance_config(
        tmp_path, split={"strategy": "stratified", "test_ratio": 0.3, "val_ratio": 0.0}
    )
    manifest = generate(config)
    test = set((config.writer.staging_dir / "splits" / "test.txt").read_text().split())
    strata: Dict[Any, List[bool]] = {}
    for entry_ in manifest.patches:
        strata.setdefault(_stratum(entry_), []).append(manifest.patch_name(entry_) in test)
    assert len(strata) > 1
    for members in strata.values():
        assert sum(members) == math.ceil(len(members) * 0.3)


def test_manifest_records_the_instance_target(
    split_dataset: Tuple[MapcvConfig, List[Feature]],
) -> None:
    config, _ = split_dataset
    manifest = Manifest.load(config.writer.staging_dir / "manifest.json")
    assert manifest.task == "instance" and manifest.version == 3
    target = manifest.target
    assert target is not None and target.type == "instance"
    assert target.class_map == CLASS_IDS
    assert target.ignore_index is None and target.dtype == "uint16"
    assert target.options == {"min_visible": 0.3, "min_area": 4, "id_mask": True}
    assert set(target.labels or {}) == {"label_field", "classes", "all_touched", "sha256"}
    assert manifest.writer is not None and manifest.writer["mask_format"] == "png"
    for entry in manifest.patches:
        assert set(entry["files"]) == {"image", "mask"}
        assert entry["files"]["image"].startswith("images/")
        assert entry["files"]["mask"].startswith("masks/")


def test_without_id_masks_only_images_and_annotations_are_written(tmp_path: Path) -> None:
    write_labels(tmp_path / "labels.geojson", random_features(seed=8, count=30))
    config = instance_config(tmp_path)
    manifest = generate(config)
    staging = config.writer.staging_dir
    assert not (staging / "masks").exists()
    assert manifest.writer is not None and "mask_format" not in manifest.writer
    assert all(list(entry["files"]) == ["image"] for entry in manifest.patches)
    assert manifest.target is not None and manifest.target.dtype is None


@pytest.mark.parametrize("mask_format", ["npy", "tif"])
def test_id_masks_in_other_formats(tmp_path: Path, mask_format: str) -> None:
    write_labels(tmp_path / "labels.geojson", random_features(seed=8, count=30))
    config = instance_config(tmp_path, instance={"id_mask": True, "min_area": 1})
    data = config.model_dump(mode="json", exclude_unset=True)
    data["writer"]["mask_format"] = mask_format
    config = MapcvConfig.model_validate(data)
    manifest = generate(config)
    staging = config.writer.staging_dir
    coco = load_coco(staging / "annotations" / "instances_all.json")
    grouped = annotations_by_image(coco)
    checked = 0
    for index, entry in enumerate(manifest.patches):
        path = staging / entry["files"]["mask"]
        assert path.suffix == f".{mask_format}"
        if mask_format == "npy":
            ids = np.load(path)
        else:
            rasterio = pytest.importorskip("rasterio")
            with rasterio.open(path) as src:
                ids = src.read(1)
        painted = np.zeros((PATCH, PATCH), dtype=np.uint16)
        for number, ann in enumerate(grouped.get(index + 1, []), start=1):
            painted[stored_mask(ann)] = number
        assert ids.dtype == np.uint16 and np.array_equal(ids, painted)
        checked += int(painted.any())
    assert checked > 3


def test_world_files_cover_images_and_id_masks(tmp_path: Path) -> None:
    write_labels(tmp_path / "labels.geojson", random_features(seed=3, count=20))
    config = instance_config(tmp_path, instance={"id_mask": True})
    data = config.model_dump(mode="json", exclude_unset=True)
    data["writer"]["world_files"] = True
    manifest = generate(MapcvConfig.model_validate(data))
    assert manifest.writer is not None and manifest.writer["world_files"] is True
    entry = manifest.patches[0]
    assert entry["files"] == {
        "image": "images/patch_0000000.png",
        "mask": "masks/patch_0000000.png",
        "image_world": "images/patch_0000000.pgw",
        "mask_world": "masks/patch_0000000.pgw",
    }
    staging = tmp_path / "dataset"
    assert all((staging / path).exists() for path in entry["files"].values())
    assert (staging / entry["files"]["image_world"]).read_text() == (
        staging / entry["files"]["mask_world"]
    ).read_text()


def test_npy_and_tif_images_work_too(tmp_path: Path) -> None:
    write_labels(tmp_path / "labels.geojson", random_features(seed=12, count=40))
    config = instance_config(tmp_path, split={"strategy": "random"})
    data = config.model_dump(mode="json", exclude_unset=True)
    data["writer"].update(image_format="tif", footprints=False)
    manifest = generate(MapcvConfig.model_validate(data))
    assert manifest.patches[0]["files"]["image"] == "images/patch_0000000.tif"
    coco = load_coco(tmp_path / "dataset" / "annotations" / "instances_train.json")
    assert all(image["file_name"].endswith(".tif") for image in coco["images"])
    assert not (tmp_path / "dataset" / "patches.geojson").exists()


def test_kml_labels_work_for_instances(tmp_path: Path) -> None:
    left, top = X0 + 10 * RES, Y0 - 10 * RES
    ring = [to_lonlat(left + x * RES, top - y * RES) for x, y in [(0, 0), (12, 0), (12, 8), (0, 8)]]
    coordinates = " ".join(f"{lon},{lat},0" for lon, lat in [*ring, ring[0]])
    (tmp_path / "labels.kml").write_text(
        '<?xml version="1.0" encoding="UTF-8"?><kml xmlns="http://www.opengis.net/kml/2.2">'
        "<Document><Placemark><name>a</name><Polygon><outerBoundaryIs><LinearRing>"
        f"<coordinates>{coordinates}</coordinates></LinearRing></outerBoundaryIs></Polygon>"
        "</Placemark></Document></kml>",
        encoding="utf-8",
    )
    config = instance_config(tmp_path, labels={"path": str(tmp_path / "labels.kml")})
    data = config.model_dump(mode="json", exclude_unset=True)
    data["labels"].pop("label_field")
    generate(MapcvConfig.model_validate(data))
    coco = load_coco(tmp_path / "dataset" / "annotations" / "instances_all.json")
    (ann,) = coco["annotations"]
    assert ann["area"] == 12 * 8 and ann["bbox"] == [10, 10, 12, 8] and not ann["truncated"]


# ── config validation, factories, the large-feature warning ─────────────────


def _raw(tmp_path: Path, **changes: Any) -> Dict[str, Any]:
    data: Dict[str, Any] = {
        "task": "instance",
        "region": {"west": 4.93, "south": 52.37, "east": 4.95, "north": 52.38},
        "imagery": {"type": "xyz", "zoom": 18, "source": "esri_satellite"},
        "labels": {"path": str(tmp_path / "labels.geojson")},
        "sampler": {"patch_size": 256},
        "writer": {"staging_dir": str(tmp_path / "dataset")},
    }
    data.update(changes)
    return data


def test_instance_defaults(tmp_path: Path) -> None:
    config = MapcvConfig.model_validate(_raw(tmp_path))
    assert config.task == "instance" and config.instance is None
    assert config.instance_options == InstanceOptions(min_visible=0.3, min_area=4, id_mask=False)


@pytest.mark.parametrize(
    ("changes", "message"),
    [
        ({"labels": None}, "task: instance needs labels"),
        ({"labels": {"type": "raster", "path": "x.tif", "classes": {"1": 1}}}, "vector labels"),
        ({"instance": {"min_visible": 1.5}}, "min_visible"),
        ({"instance": {"min_area": 0}}, "min_area"),
        ({"instance": {"id_mask": "sometimes"}}, "id_mask"),
        ({"instance": {"min_visibility": 0.3}}, "min_visibility"),
        ({"task": "segmentation", "instance": {}}, "only applies to task: instance"),
        ({"task": "detection", "instance": {}}, "only applies to task: instance"),
        ({"detection": {}}, "only applies to task: detection"),
        (
            {"labels": {"path": "x.geojson", "ignore_index": 255}},
            "labels.ignore_index marks mask pixels",
        ),
        ({"sampler": {"patch_size": 256, "pad_mode": "reflect"}}, "mirrors instances"),
        ({"writer": {"staging_dir": "out", "mask_format": "png"}}, "instance.id_mask: true"),
        (
            {"writer": {"staging_dir": "out", "image_format": "tif", "world_files": True}},
            "world_files",
        ),
    ],
)
def test_invalid_instance_configs_fail_clearly(
    tmp_path: Path, changes: Dict[str, Any], message: str
) -> None:
    with pytest.raises(ValidationError, match=message):
        MapcvConfig.model_validate(_raw(tmp_path, **changes))


def test_valid_instance_combinations(tmp_path: Path) -> None:
    config = MapcvConfig.model_validate(
        _raw(
            tmp_path,
            labels={"path": "x.geojson", "all_touched": True},
            instance={"id_mask": True},
            writer={"staging_dir": "out", "mask_format": "npy"},
        )
    )
    assert config.instance_options.id_mask
    sampler = {"patch_size": 256, "pad_mode": "reflect", "edge_strategy": "shift"}
    assert MapcvConfig.model_validate(_raw(tmp_path, sampler=sampler)).task == "instance"


def test_factories_pick_the_instance_target_and_writer(tmp_path: Path) -> None:
    config = MapcvConfig.model_validate(_raw(tmp_path))
    target = create_target(config)
    assert isinstance(target, InstanceTarget)
    writer = create_writer(config.writer, target)
    assert isinstance(writer, InstanceWriter)
    check_compatible(target, writer)
    with pytest.raises(ValueError, match="cannot write instance targets"):
        check_compatible(target, FilesWriter(config.writer))


def test_large_features_that_can_never_be_kept_are_reported(tmp_path: Path) -> None:
    big = world_box(-20, -20, 180, 180)  # 40,000 px²; a patch shows 4,096
    write_labels(tmp_path / "labels.geojson", [Feature(big, "tree")])
    config = instance_config(tmp_path)
    valid = np.ones((HEIGHT, WIDTH), dtype=np.bool_)
    with pytest.warns(UserWarning, match="instance.min_visible"):
        manifest = generate(config, FakeSource(valid))
    assert not any(entry["summary"]["class_objects"] for entry in manifest.patches)


def test_features_outside_the_raster_are_reported(tmp_path: Path) -> None:
    write_labels(tmp_path / "labels.geojson", [Feature(world_box(2000, 2000, 2010, 2010), "tree")])
    with pytest.warns(UserWarning, match="no label feature intersects the imagery extent"):
        generate(instance_config(tmp_path), FakeSource())


# ── the CLI, end to end against a local tile server ─────────────────────────

TILE_ZOOM, TILE_X0, TILE_Y0, TILES_X, TILES_Y = 18, 134_700, 86_100, 4, 3


@pytest.fixture()
def tile_server() -> Iterator[str]:
    import io
    import threading
    from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

    class Tiles(BaseHTTPRequestHandler):
        def log_message(self, *args: object) -> None:
            pass

        def do_GET(self) -> None:  # noqa: N802 - http.server API
            z, x, y = (int(part) for part in self.path.strip("/").split(".")[0].split("/"))
            buffer = io.BytesIO()
            Image.new("RGB", (256, 256), (40 + x % 7 * 20, 60 + y % 5 * 30, 90)).save(buffer, "PNG")
            body = buffer.getvalue()
            self.send_response(200)
            self.send_header("Content-Type", "image/png")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

    server = ThreadingHTTPServer(("127.0.0.1", 0), Tiles)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    yield f"http://127.0.0.1:{server.server_address[1]}/{{z}}/{{x}}/{{y}}.png"
    server.shutdown()


def _tile_lon(x: float) -> float:
    return float(x / 2**TILE_ZOOM * 360 - 180)


def _tile_lat(y: float) -> float:
    return math.degrees(math.atan(math.sinh(math.pi * (1 - 2 * y / 2**TILE_ZOOM))))


def test_cli_builds_coco_masks_from_a_tile_server(tmp_path: Path, tile_server: str) -> None:
    from typer.testing import CliRunner

    from mapcv.cli import app

    rng = random.Random(8)
    features = []
    for index in range(60):
        x = TILE_X0 + rng.uniform(-0.1, TILES_X + 0.1)
        y = TILE_Y0 + rng.uniform(-0.1, TILES_Y + 0.1)
        size = rng.uniform(0.02, 0.3)
        ring = [(x, y), (x + size, y), (x + size * 0.7, y + size), (x, y + size * 0.8), (x, y)]
        features.append(
            {
                "type": "Feature",
                "properties": {"kind": "shed" if index % 3 else "house"},
                "geometry": {
                    "type": "Polygon",
                    "coordinates": [[[_tile_lon(px), _tile_lat(py)] for px, py in ring]],
                },
            }
        )
    (tmp_path / "labels.geojson").write_text(
        json.dumps({"type": "FeatureCollection", "features": features})
    )
    eps = 1e-7
    (tmp_path / "mapcv.yaml").write_text(
        f"""\
task: instance
region:
  west: {_tile_lon(TILE_X0) + eps}
  south: {_tile_lat(TILE_Y0 + TILES_Y) + eps}
  east: {_tile_lon(TILE_X0 + TILES_X) - eps}
  north: {_tile_lat(TILE_Y0) - eps}
imagery:
  type: xyz
  zoom: {TILE_ZOOM}
  url_template: "{tile_server}"
labels:
  path: labels.geojson
  label_field: kind
instance:
  min_visible: 0.25
  id_mask: true
sampler:
  patch_size: 192
writer:
  staging_dir: dataset
split:
  strategy: spatial
  test_ratio: 0.2
  val_ratio: 0.2
""",
        encoding="utf-8",
    )
    runner = CliRunner()
    config = str(tmp_path / "mapcv.yaml")
    result = runner.invoke(app, ["validate", config])
    assert result.exit_code == 0, result.output
    assert "instance · COCO RLE masks · min_visible 0.25" in result.output
    result = runner.invoke(app, ["plan", config])
    assert result.exit_code == 0, result.output
    assert "Objects" in result.output and "one mask each" in result.output
    assert "instance-ID PNGs" in result.output

    result = runner.invoke(app, ["generate", config, "--yes"])
    assert result.exit_code == 0, result.output
    assert "annotations/" in result.output and "masks/" in result.output
    assert "tutorials/instance-segmentation" in result.output
    staging = tmp_path / "dataset"
    manifest = Manifest.load(staging / "manifest.json")
    assert manifest.task == "instance" and manifest.class_map == {"house": 1, "shed": 2}
    assert len(manifest.patches) == 6 * 4  # 1,024 x 768 px in 192 px patches with padding
    total = 0
    for split in ("train", "val", "test"):
        path = staging / "annotations" / f"instances_{split}.json"
        strict_coco_check(load_coco(path), 192)
        api = pycocotools_coco.COCO(str(path))
        names = (staging / "splits" / f"{split}.txt").read_text().split()
        assert sorted(api.imgs[i]["file_name"] for i in api.getImgIds()) == sorted(names)
        for image_id in api.getImgIds():
            anns = api.loadAnns(api.getAnnIds(imgIds=[image_id]))
            total += len(anns)
            ids = np.array(Image.open(staging / manifest.patches[image_id - 1]["files"]["mask"]))
            painted = np.zeros(ids.shape, dtype=np.uint16)
            for number, ann in enumerate(anns, start=1):
                painted[api.annToMask(ann) > 0] = number
            assert np.array_equal(ids, painted)
    assert total > 30

    result = runner.invoke(app, ["info", str(staging)])
    assert result.exit_code == 0, result.output
    assert "instance" in result.output and "objects" in result.output and "house" in result.output
    result = runner.invoke(app, ["split", str(staging), "--strategy", "random", "--seed", "3"])
    assert result.exit_code == 0, result.output
    names = (staging / "splits" / "test.txt").read_text().split()
    coco = load_coco(staging / "annotations" / "instances_test.json")
    assert sorted(image["file_name"] for image in coco["images"]) == sorted(names)
    result = runner.invoke(app, ["generate", config, "--yes"])
    assert result.exit_code == 0 and "Nothing left to do" in result.output


def test_init_template_and_wizard_offer_instance_segmentation(tmp_path: Path) -> None:
    from typer.testing import CliRunner

    from mapcv.cli import app

    runner = CliRunner()
    written = tmp_path / "template.yaml"
    result = runner.invoke(app, ["init", str(written), "--template", "instance"])
    assert result.exit_code == 0, result.output
    config = MapcvConfig.from_yaml(written)
    assert config.task == "instance" and config.instance_options == InstanceOptions()

    labels = tmp_path / "aoi.geojson"
    labels.write_text(
        '{"type":"FeatureCollection","features":[{"type":"Feature","properties":{"kind":"roof"},'
        '"geometry":{"type":"Polygon","coordinates":[[[74.3,31.5],[74.31,31.5],'
        "[74.31,31.51],[74.3,31.5]]]}}]}"
    )
    out = tmp_path / "mapcv.yaml"
    answers = ["esri", str(labels), "17", "y", "kind", "instance", "y", "256", "./ds", "y"]
    result = runner.invoke(
        app, ["init", str(out), "--interactive"], input="\n".join(answers) + "\n"
    )
    assert result.exit_code == 0, result.output
    config = MapcvConfig.from_yaml(out)
    assert config.task == "instance"
    assert config.instance_options == InstanceOptions(id_mask=True)
    assert yaml.safe_load(out.read_text())["instance"]["min_visible"] == 0.3
