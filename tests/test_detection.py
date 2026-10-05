"""Object detection datasets (``task: detection``): boxes, COCO and YOLO outputs.

The box tests compare mapcv's output with an independent computation: each
feature is clipped in world coordinates (shapely) against the patch footprint,
the raster extent and the pixels without imagery, then transformed to patch
pixels by hand.
"""

from __future__ import annotations

import json
import math
import os
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional, Tuple

import numpy as np
import numpy.typing as npt
import pytest
import shapely
import yaml
from pydantic import ValidationError
from shapely.geometry import MultiPolygon, Point, Polygon, box, mapping
from shapely.geometry.base import BaseGeometry

from mapcv.config import DetectionOptions, MapcvConfig
from mapcv.imagery import RasterMetadata
from mapcv.manifest import Manifest, ManifestMismatchError
from mapcv.pipeline import run_generate, run_split
from mapcv.splitter import SplitterConfig, _stratum
from mapcv.targets import DetectionTarget, create_target
from mapcv.targets.detection import mask_region, visible_parts
from mapcv.writers import DetectionWriter, FilesWriter, check_compatible, create_writer

R = 6_378_137.0
# A zoom-18 Web Mercator grid near Amsterdam.
RES = 0.5971642834779232
X0, Y0 = 549_582.2333704159, 6_868_784.235762509
HEIGHT, WIDTH = 300, 330
PATCH = 64
CLASSES = ("building", "car", "tree")


def to_lonlat(x: float, y: float) -> Tuple[float, float]:
    return math.degrees(x / R), math.degrees(2 * math.atan(math.exp(y / R)) - math.pi / 2)


def lonlat_geometry(geometry: BaseGeometry) -> BaseGeometry:
    return shapely.transform(
        geometry, lambda xy: np.array([to_lonlat(x, y) for x, y in xy], dtype=np.float64)
    )


def make_valid_mask(kind: str = "edges") -> npt.NDArray[np.bool_]:
    """Pixels with imagery.

    ``edges``: no imagery in a failed-tile-like block and below a diagonal NoData
    edge. ``block``: only the block. ``noisy``: scattered black pixels as well.
    """
    rows, cols = np.mgrid[0:HEIGHT, 0:WIDTH]
    valid = np.ones((HEIGHT, WIDTH), dtype=np.bool_)
    valid[100:140, 200:260] = False
    if kind in ("edges", "noisy"):
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
    world: BaseGeometry  # Web Mercator metres; points stay points
    name: str


def random_features(seed: int, count: int = 160, points: bool = False) -> List[Feature]:
    rng = random.Random(seed)
    left, top = X0 - 20 * RES, Y0 + 20 * RES
    span_x, span_y = (WIDTH + 40) * RES, (HEIGHT + 40) * RES
    features: List[Feature] = []
    for index in range(count):
        cx = left + rng.uniform(0, span_x)
        cy = top - rng.uniform(0, span_y)
        kind = index % (6 if points else 5)
        name = rng.choice(CLASSES)
        if kind == 0:  # ordinary object, often crossing a patch edge
            geometry: BaseGeometry = star(rng, cx, cy, rng.uniform(3, 40) * RES)
        elif kind == 1:  # tiny object
            geometry = star(rng, cx, cy, rng.uniform(0.3, 2.5) * RES)
        elif kind == 2:  # multipolygon: one object of two parts
            first = star(rng, cx, cy, rng.uniform(2, 15) * RES)
            dx, dy = rng.uniform(-40, 40) * RES, rng.uniform(-40, 40) * RES
            second = star(rng, cx + dx, cy + dy, rng.uniform(2, 15) * RES)
            if first.intersects(second):
                geometry = first
            else:
                geometry = MultiPolygon([first, second])
        elif kind == 3:  # polygon with a hole
            radius = rng.uniform(8, 50) * RES
            outer = Point(cx, cy).buffer(radius, quad_segs=rng.randint(2, 8))
            hole = Point(cx, cy).buffer(radius * 0.5, quad_segs=3)
            geometry = Polygon(outer.exterior.coords, [hole.exterior.coords])
        elif kind == 4:  # large irregular object
            geometry = star(rng, cx, cy, rng.uniform(40, 120) * RES)
        else:
            geometry = Point(cx, cy)
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


def detection_config(
    root: Path,
    *,
    detection: Optional[Dict[str, Any]] = None,
    split: Optional[Dict[str, Any]] = None,
    sampler: Optional[Dict[str, Any]] = None,
    staging: str = "dataset",
) -> MapcvConfig:
    data: Dict[str, Any] = {
        "task": "detection",
        "region": {"west": 4.93, "south": 52.37, "east": 4.95, "north": 52.38},
        "imagery": {"type": "xyz", "zoom": 18, "url_template": "http://127.0.0.1/{z}/{x}/{y}.png"},
        "labels": {"path": str(root / "labels.geojson"), "label_field": "class"},
        "sampler": {"patch_size": PATCH, "edge_strategy": "pad", **(sampler or {})},
        "writer": {"staging_dir": str(root / staging), "image_format": "png"},
    }
    if detection is not None:
        data["detection"] = detection
    if split is not None:
        data["split"] = split
    return MapcvConfig.model_validate(data)


def generate(config: MapcvConfig, source: Optional[FakeSource] = None) -> Manifest:
    with pytest.MonkeyPatch.context() as patch:
        patch.setattr("mapcv.pipeline.open_raster_source", lambda *a, **k: source or FakeSource())
        run_generate(config)
    return Manifest.load(config.writer.staging_dir / "manifest.json")


# ── the independent box computation ─────────────────────────────────────────


@dataclass
class Expected:
    category: int
    bbox: Tuple[float, float, float, float]
    area: float
    truncated: bool
    borderline: bool  # within rounding of a threshold: either outcome is right


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


def expected_objects(
    features: List[Feature],
    class_map: Dict[str, int],
    valid: npt.NDArray[np.bool_],
    row: int,
    col: int,
    options: DetectionOptions,
) -> List[Expected]:
    left, top = X0 + col * RES, Y0 - row * RES
    patch = box(left, top - PATCH * RES, left + PATCH * RES, top)
    raster = box(X0, Y0 - HEIGHT * RES, X0 + WIDTH * RES, Y0)
    invalid = invalid_world_region(valid, row, col)
    result: List[Expected] = []
    for feature in features:
        world = feature.world
        if world.geom_type == "Point":
            if options.point_box_size is None:
                continue
            half = options.point_box_size * RES / 2
            world = box(world.x - half, world.y - half, world.x + half, world.y + half)
        visible = world.intersection(patch).intersection(raster)
        if invalid is not None:
            visible = visible.difference(invalid)
        # The visible area: drop lines and points where the feature only touches.
        visible = shapely.union_all(
            [part for part in shapely.get_parts(visible) if part.geom_type == "Polygon"]
        )
        if visible.is_empty or visible.area <= 0:
            continue
        fraction = visible.area / world.area
        minx, miny, maxx, maxy = visible.bounds
        x = (minx - X0) / RES - col
        y = (Y0 - maxy) / RES - row
        w = (maxx - minx) / RES
        h = (maxy - miny) / RES
        # Areas from different geometry operations agree to ~1e-12; a fully visible
        # object counts as whole, and thresholds are met within that noise.
        whole = fraction > 1.0 - 1e-9
        shortfall = options.min_visible - fraction
        borderline = (
            1e-12 < abs(shortfall) < 1e-6
            or abs(w - options.min_box_pixels) < 1e-3
            or abs(h - options.min_box_pixels) < 1e-3
        )
        keep = (whole or shortfall <= 1e-12) and min(w, h) >= options.min_box_pixels
        if keep or borderline:
            result.append(
                Expected(
                    category=class_map[feature.name],
                    bbox=(x, y, w, h),
                    area=visible.area / RES**2,
                    truncated=not whole,
                    borderline=borderline,
                )
            )
    return result


def assert_matches(found: List[Dict[str, Any]], expected: List[Expected], where: str) -> None:
    unmatched = list(found)
    for item in expected:
        match = next(
            (
                ann
                for ann in unmatched
                if ann["category_id"] == item.category
                and all(abs(a - b) < 2e-3 for a, b in zip(ann["bbox"], item.bbox))
            ),
            None,
        )
        if match is None:
            assert item.borderline, f"{where}: missing {item}"
            continue
        unmatched.remove(match)
        assert match["area"] == pytest.approx(item.area, rel=1e-6, abs=2e-3), where
        assert match["truncated"] == item.truncated, (where, item)
    assert not unmatched, f"{where}: unexpected {unmatched}"


# ── fixtures: one dataset without a split, one with ─────────────────────────


@pytest.fixture(scope="module")
def unsplit(
    tmp_path_factory: pytest.TempPathFactory,
) -> Iterator[Tuple[MapcvConfig, List[Feature]]]:
    root = tmp_path_factory.mktemp("unsplit")
    features = random_features(seed=7, points=True)
    write_labels(root / "labels.geojson", features)
    config = detection_config(root, detection={"point_box_size": 6})
    with pytest.warns(UserWarning, match="no split"):
        generate(config)
    yield config, features


@pytest.fixture(scope="module")
def split_dataset(
    tmp_path_factory: pytest.TempPathFactory,
) -> Iterator[Tuple[MapcvConfig, List[Feature]]]:
    root = tmp_path_factory.mktemp("split")
    features = random_features(seed=11)
    write_labels(root / "labels.geojson", features)
    config = detection_config(
        root, split={"strategy": "random", "test_ratio": 0.25, "val_ratio": 0.2}
    )
    generate(config)
    yield config, features


def load_coco(path: Path) -> Dict[str, Any]:
    data: Dict[str, Any] = json.loads(path.read_text(encoding="utf-8"))
    return data


# ── boxes against the independent computation ───────────────────────────────


@pytest.mark.parametrize(
    "options",
    [
        {"point_box_size": 6},
        {"min_visible": 0.0, "min_box_pixels": 0, "point_box_size": 3},
        {"min_visible": 0.75, "min_box_pixels": 5},
        {"min_visible": 1.0},
    ],
    ids=["defaults", "keep-everything", "strict", "only-whole"],
)
@pytest.mark.parametrize("mask", ["edges", "block", "noisy"])
def test_boxes_match_an_independent_world_space_clip(
    tmp_path: Path, options: Dict[str, Any], mask: str
) -> None:
    features = random_features(seed=3, count=220, points=True)
    write_labels(tmp_path / "labels.geojson", features)
    config = detection_config(tmp_path, detection=options)
    source = FakeSource(valid=make_valid_mask(mask))
    with pytest.warns(UserWarning):
        manifest = generate(config, source)
    coco = load_coco(config.writer.staging_dir / "annotations" / "instances_all.json")
    by_image: Dict[int, List[Dict[str, Any]]] = {}
    for ann in coco["annotations"]:
        by_image.setdefault(ann["image_id"], []).append(ann)
    assert len(coco["images"]) == len(manifest.patches) == 30  # 6 x 5 patches, padded edges
    checked = truncated = 0
    for index, entry in enumerate(manifest.patches):
        found = by_image.get(index + 1, [])
        expected = expected_objects(
            features,
            manifest.class_map,
            source.valid,
            entry["row"],
            entry["col"],
            config.detection_options,
        )
        assert_matches(found, expected, f"patch {index} at {entry['row']},{entry['col']}")
        counts: Dict[str, int] = {}
        for ann in found:
            counts[str(ann["category_id"])] = counts.get(str(ann["category_id"]), 0) + 1
        assert entry["summary"]["class_objects"] == dict(
            sorted(counts.items(), key=lambda kv: int(kv[0]))
        )
        checked += len(found)
        truncated += sum(1 for ann in found if ann["truncated"])
    assert checked > 5
    if config.detection_options.min_visible < 1.0:
        assert 0 < truncated < checked
    else:
        assert truncated == 0


def test_holes_do_not_change_the_box_and_multipolygons_are_one_object(tmp_path: Path) -> None:
    left, top = X0 + 10 * RES, Y0 - 10 * RES
    outer = box(left, top - 30 * RES, left + 30 * RES, top)
    hole = box(left + 5 * RES, top - 25 * RES, left + 25 * RES, top - 5 * RES)
    second = box(left + 40 * RES, top - 45 * RES, left + 50 * RES, top - 35 * RES)
    features = [
        Feature(Polygon(outer.exterior.coords, [hole.exterior.coords]), "building"),
        Feature(outer, "car"),
        Feature(MultiPolygon([outer, second]), "tree"),
    ]
    write_labels(tmp_path / "labels.geojson", features)
    config = detection_config(tmp_path, sampler={"patch_size": 128, "edge_strategy": "drop"})
    with pytest.warns(UserWarning, match="no split"):
        generate(config, FakeSource(valid=np.ones((HEIGHT, WIDTH), dtype=np.bool_)))
    coco = load_coco(config.writer.staging_dir / "annotations" / "instances_all.json")
    first = [ann for ann in coco["annotations"] if ann["image_id"] == 1]
    assert [ann["category_id"] for ann in first] == [1, 2, 3]  # feature order
    holed, plain, multi = first
    assert holed["bbox"] == plain["bbox"] == [10.0, 10.0, 30.0, 30.0]
    assert holed["area"] == pytest.approx(900 - 400) and plain["area"] == pytest.approx(900)
    assert multi["bbox"] == [10.0, 10.0, 50.0, 45.0]
    assert multi["area"] == pytest.approx(1000) and not multi["truncated"]


def test_no_imagery_keeps_only_the_visible_part(tmp_path: Path) -> None:
    # A square half over a failed tile: its box shrinks to the half with imagery.
    valid = np.ones((HEIGHT, WIDTH), dtype=np.bool_)
    valid[:, 20:] = False
    left, top = X0 + 10 * RES, Y0 - 10 * RES
    write_labels(
        tmp_path / "labels.geojson",
        [Feature(box(left, top - 20 * RES, left + 20 * RES, top), "building")],
    )
    config = detection_config(tmp_path, detection={"min_visible": 0.5})
    with pytest.warns(UserWarning, match="no split"):
        generate(config, FakeSource(valid=valid))
    coco = load_coco(config.writer.staging_dir / "annotations" / "instances_all.json")
    (ann,) = coco["annotations"]
    assert ann["bbox"] == [10.0, 10.0, 10.0, 20.0]
    assert ann["area"] == 200.0 and ann["truncated"] is True


def test_padding_beyond_the_raster_is_not_visible(tmp_path: Path) -> None:
    # The last patch column starts at 320 and pads 54 px past the 330 px raster.
    left, top = X0 + 325 * RES, Y0 - 5 * RES
    write_labels(
        tmp_path / "labels.geojson",
        [Feature(box(left, top - 10 * RES, left + 20 * RES, top), "car")],
    )
    config = detection_config(tmp_path, detection={"min_visible": 0.2})
    with pytest.warns(UserWarning):
        manifest = generate(config, FakeSource(valid=np.ones((HEIGHT, WIDTH), dtype=np.bool_)))
    coco = load_coco(config.writer.staging_dir / "annotations" / "instances_all.json")
    (ann,) = coco["annotations"]
    entry = manifest.patches[ann["image_id"] - 1]
    assert (entry["row"], entry["col"], entry["padded"]) == (0, 320, True)
    assert ann["bbox"] == [5.0, 5.0, 5.0, 10.0] and ann["truncated"] is True


@pytest.mark.parametrize("max_rectangles", [0, 10_000])
def test_both_ways_of_removing_no_imagery_pixels_agree(
    monkeypatch: pytest.MonkeyPatch, max_rectangles: int
) -> None:
    # 0: always cut by valid runs; 10,000: always subtract the union of invalid pixels.
    monkeypatch.setattr("mapcv.targets.detection._MAX_UNION_RECTANGLES", max_rectangles)
    rng = np.random.default_rng(4)
    shapes = [
        Point(rng.uniform(0, 40), rng.uniform(0, 30)).buffer(rng.uniform(1, 15)) for _ in range(25)
    ]
    geometries = np.empty(len(shapes), dtype=object)
    # visible_parts needs geometries inside the mask (annotate clips them to it first).
    geometries[:] = [shape.intersection(box(3, 5, 43, 35)) for shape in shapes]
    for share in (0.02, 0.3, 0.9):
        valid = rng.random((30, 40)) >= share
        result = visible_parts(geometries, valid, 3, 5)
        rows, cols = np.nonzero(~valid)
        invalid = shapely.union_all(shapely.box(cols + 3, rows + 5, cols + 4, rows + 6))
        for geometry, got in zip(geometries, result):
            want = geometry.difference(invalid)
            assert got.area == pytest.approx(want.area, abs=1e-9)
            if want.area > 0:
                assert got.bounds == pytest.approx(want.bounds, abs=1e-9)
            assert got.geom_type in ("Polygon", "MultiPolygon")


def test_mask_region_matches_one_box_per_pixel() -> None:
    rng = np.random.default_rng(1)
    for _ in range(20):
        mask = rng.random((23, 31)) < rng.uniform(0.05, 0.9)
        region = mask_region(mask, 5, 7)
        rows, cols = np.nonzero(mask)
        reference = shapely.union_all(shapely.box(cols + 5, rows + 7, cols + 6, rows + 8))
        assert region is not None
        assert region.symmetric_difference(reference).area == pytest.approx(0.0, abs=1e-9)
    assert mask_region(np.zeros((4, 4), dtype=bool), 0, 0) is None


# ── COCO output ─────────────────────────────────────────────────────────────


def strict_coco_check(coco: Dict[str, Any], patch_size: int) -> None:
    assert set(coco) == {"info", "licenses", "categories", "images", "annotations"}
    category_ids = [cat["id"] for cat in coco["categories"]]
    assert category_ids == sorted(set(category_ids))
    for cat in coco["categories"]:
        assert set(cat) == {"id", "name", "supercategory"}
        assert isinstance(cat["id"], int) and isinstance(cat["name"], str) and cat["name"]
    image_ids = set()
    for image in coco["images"]:
        assert set(image) == {"id", "file_name", "width", "height"}
        assert isinstance(image["id"], int) and image["id"] >= 1
        assert image["id"] not in image_ids
        image_ids.add(image["id"])
        assert image["width"] == image["height"] == patch_size
        assert image["file_name"] == f"patch_{image['id'] - 1:07d}.png"
    ann_ids = set()
    for ann in coco["annotations"]:
        assert set(ann) == {"id", "image_id", "category_id", "bbox", "area", "iscrowd", "truncated"}
        assert isinstance(ann["id"], int) and ann["id"] >= 1 and ann["id"] not in ann_ids
        ann_ids.add(ann["id"])
        assert ann["image_id"] in image_ids
        assert ann["category_id"] in category_ids
        assert ann["iscrowd"] == 0 and isinstance(ann["truncated"], bool)
        x, y, w, h = ann["bbox"]
        assert all(isinstance(v, (int, float)) for v in ann["bbox"])
        assert w > 0 and h > 0 and 0 <= x and 0 <= y
        assert x + w <= patch_size + 1e-9 and y + h <= patch_size + 1e-9
        # Box sides and the area are each rounded to 1e-4 px.
        assert 0 < ann["area"] <= (w + 2e-4) * (h + 2e-4)


def test_coco_files_load_in_pycocotools_and_pass_a_strict_schema_check(
    split_dataset: Tuple[MapcvConfig, List[Feature]],
) -> None:
    coco_api = pytest.importorskip("pycocotools.coco")
    config, _ = split_dataset
    staging = config.writer.staging_dir
    manifest = Manifest.load(staging / "manifest.json")
    splits = {
        name: (staging / "splits" / f"{name}.txt").read_text().split()
        for name in ("train", "val", "test")
    }
    all_ids: Dict[int, Dict[str, Any]] = {}
    for name, names in splits.items():
        path = staging / "annotations" / f"instances_{name}.json"
        coco = load_coco(path)
        strict_coco_check(coco, PATCH)
        assert sorted(image["file_name"] for image in coco["images"]) == sorted(names)
        api = coco_api.COCO(str(path))
        assert sorted(api.getImgIds()) == sorted(image["id"] for image in coco["images"])
        assert api.getCatIds() == [1, 2, 3]
        assert [cat["name"] for cat in api.loadCats(api.getCatIds())] == list(CLASSES)
        for image in coco["images"]:
            ids = api.getAnnIds(imgIds=[image["id"]])
            loaded = api.loadAnns(ids)
            own = [ann for ann in coco["annotations"] if ann["image_id"] == image["id"]]
            assert loaded == own
            entry = manifest.patches[image["id"] - 1]
            assert entry["files"]["image"] == f"images/{image['file_name']}"
            counts: Dict[str, int] = {}
            for ann in own:
                counts[str(ann["category_id"])] = counts.get(str(ann["category_id"]), 0) + 1
            assert entry["summary"]["class_objects"] == dict(
                sorted(counts.items(), key=lambda kv: int(kv[0]))
            )
        for ann in coco["annotations"]:
            assert ann["id"] not in all_ids  # unique across the split files
            all_ids[ann["id"]] = ann
    # IDs are stable: a re-split moves annotations between files with the same IDs.
    run_split(staging, SplitterConfig(strategy="random", seed=5, test_ratio=0.4))
    for name in ("train", "val", "test"):
        for ann in load_coco(staging / "annotations" / f"instances_{name}.json")["annotations"]:
            assert all_ids[ann["id"]] == ann


def test_images_without_objects_are_listed_without_annotations(
    split_dataset: Tuple[MapcvConfig, List[Feature]],
) -> None:
    config, _ = split_dataset
    staging = config.writer.staging_dir
    manifest = Manifest.load(staging / "manifest.json")
    empty = [i + 1 for i, e in enumerate(manifest.patches) if not e["summary"]["class_objects"]]
    assert empty, "the fixture should have patches without objects"
    listed = set()
    for name in ("train", "val", "test"):
        coco = load_coco(staging / "annotations" / f"instances_{name}.json")
        annotated = {ann["image_id"] for ann in coco["annotations"]}
        listed |= {image["id"] for image in coco["images"] if image["id"] not in annotated}
    assert set(empty) == listed  # a random split puts every patch in a list


# ── YOLO output ─────────────────────────────────────────────────────────────


def img2label_path(image: str) -> str:
    """Ultralytics' ``img2label_paths`` for one image path."""
    sa, sb = f"{os.sep}images{os.sep}", f"{os.sep}labels{os.sep}"
    return sb.join(image.rsplit(sa, 1)).rsplit(".", 1)[0] + ".txt"


def test_yolo_labels_equal_the_coco_boxes(split_dataset: Tuple[MapcvConfig, List[Feature]]) -> None:
    config, _ = split_dataset
    staging = config.writer.staging_dir
    names = yaml.safe_load((staging / "dataset.yaml").read_text())["names"]
    for split in ("train", "val", "test"):
        coco = load_coco(staging / "annotations" / f"instances_{split}.json")
        category_names = {cat["id"]: cat["name"] for cat in coco["categories"]}
        for image in coco["images"]:
            label = staging / "labels" / (Path(image["file_name"]).stem + ".txt")
            anns = [ann for ann in coco["annotations"] if ann["image_id"] == image["id"]]
            if not anns:
                assert not label.exists()  # Ultralytics: no label file for a background image
                continue
            lines = label.read_text().splitlines()
            assert len(lines) == len(anns)
            for line, ann in zip(lines, anns):
                fields = line.split()
                assert len(fields) == 5
                index = int(fields[0])
                cx, cy, w, h = (float(v) for v in fields[1:])
                assert all(0.0 <= v <= 1.0 for v in (cx, cy, w, h))
                assert names[index] == category_names[ann["category_id"]]
                width, height = image["width"], image["height"]
                denormalized = [(cx - w / 2) * width, (cy - h / 2) * height, w * width, h * height]
                assert denormalized == pytest.approx(ann["bbox"], abs=1e-6)


def test_dataset_yaml_follows_ultralytics_conventions(
    split_dataset: Tuple[MapcvConfig, List[Feature]],
) -> None:
    config, _ = split_dataset
    staging = config.writer.staging_dir
    data = yaml.safe_load((staging / "dataset.yaml").read_text(encoding="utf-8"))
    assert list(data) == ["path", "train", "val", "test", "names"]
    root = Path(data["path"])
    assert root.is_absolute() and root == staging.resolve()
    assert data["names"] == {0: "building", 1: "car", 2: "tree"}
    for split in ("train", "val", "test"):
        list_file = root / data[split]  # check_det_dataset: (path / data[k]).resolve()
        assert list_file.is_file()
        lines = list_file.read_text(encoding="utf-8").strip().splitlines()
        expected = (staging / "splits" / f"{split}.txt").read_text().split()
        assert [Path(line).name for line in lines] == expected
        for line in lines:
            assert line.startswith("./images/")
            # get_img_files: "./" is relative to the list file's folder.
            image = line.replace("./", str(list_file.parent) + os.sep, 1).replace("/", os.sep)
            assert Path(image).is_file()
            label = Path(img2label_path(image))
            assert label.parent == staging.resolve() / "labels"
            entry_objects = label.exists()
            stem = Path(image).name
            manifest = Manifest.load(staging / "manifest.json")
            entry = next(e for e in manifest.patches if e["files"]["image"].endswith(stem))
            assert entry_objects == bool(entry["summary"]["class_objects"])
            if entry_objects:
                for row in label.read_text().splitlines():
                    assert 0 <= int(row.split()[0]) < len(data["names"])


# ── splits, resume, formats ─────────────────────────────────────────────────


def test_stratification_uses_object_classes() -> None:
    def entry(objects: Dict[str, int]) -> Any:
        return {"summary": {"class_objects": objects, "empty_ratio": 0.0}}

    assert _stratum(entry({})) == (0, "")
    assert _stratum(entry({"1": 2, "3": 5})) == (1, "3")
    assert _stratum(entry({"1": 2, "2": 2})) == (1, "1")


def test_stratified_split_balances_object_classes(tmp_path: Path) -> None:
    features = random_features(seed=5, count=200)
    write_labels(tmp_path / "labels.geojson", features)
    config = detection_config(
        tmp_path, split={"strategy": "stratified", "test_ratio": 0.3, "val_ratio": 0.0}
    )
    manifest = generate(config)
    test = set((config.writer.staging_dir / "splits" / "test.txt").read_text().split())
    strata: Dict[Any, List[bool]] = {}
    for entry in manifest.patches:
        strata.setdefault(_stratum(entry), []).append(manifest.patch_name(entry) in test)
    for members in strata.values():
        assert sum(members) == math.ceil(len(members) * 0.3)


def test_resume_reproduces_the_uninterrupted_dataset(tmp_path: Path) -> None:
    features = random_features(seed=9)
    write_labels(tmp_path / "labels.geojson", features)
    config = detection_config(tmp_path, split={"strategy": "spatial"})
    manifest = generate(config)
    staging = config.writer.staging_dir

    def snapshot() -> Dict[str, bytes]:
        return {
            str(path.relative_to(staging)): path.read_bytes()
            for path in sorted(staging.rglob("*"))
            if path.is_file()
        }

    before = snapshot()
    # Lose the second half: its chunks' boxes, labels and the manifest rows.
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
    for name in ("instances_train.json", "dataset.yaml"):
        target = staging / "annotations" / name if name.endswith(".json") else staging / name
        target.unlink()
    generate(config)
    assert snapshot() == before


def test_resume_refuses_changed_detection_options(tmp_path: Path) -> None:
    write_labels(tmp_path / "labels.geojson", random_features(seed=1, count=20))
    with pytest.warns(UserWarning, match="no split"):
        generate(detection_config(tmp_path))
    with pytest.raises(ManifestMismatchError, match="task options"):
        generate(detection_config(tmp_path, detection={"min_visible": 0.5}))
    with pytest.raises(ManifestMismatchError, match="task options"):
        generate(detection_config(tmp_path, detection={"formats": ["coco"]}))


def test_split_after_generation_writes_the_split_outputs(tmp_path: Path) -> None:
    write_labels(tmp_path / "labels.geojson", random_features(seed=2, count=60))
    config = detection_config(tmp_path)
    with pytest.warns(UserWarning, match="no split"):
        generate(config)
    staging = config.writer.staging_dir
    assert (staging / "annotations" / "instances_all.json").exists()
    assert not (staging / "dataset.yaml").exists()
    all_ids = {
        ann["id"]
        for ann in load_coco(staging / "annotations" / "instances_all.json")["annotations"]
    }
    run_split(staging, SplitterConfig(strategy="random", test_ratio=0.2, val_ratio=0.2))
    assert not (staging / "annotations" / "instances_all.json").exists()
    assert (staging / "dataset.yaml").exists()
    ids = set()
    for name in ("train", "val", "test"):
        ids |= {
            ann["id"]
            for ann in load_coco(staging / "annotations" / f"instances_{name}.json")["annotations"]
        }
    assert ids == all_ids  # random split: every patch is in a list


@pytest.mark.parametrize("formats", [["coco"], ["yolo"]])
def test_only_the_chosen_formats_are_written(tmp_path: Path, formats: List[str]) -> None:
    write_labels(tmp_path / "labels.geojson", random_features(seed=4, count=60))
    config = detection_config(
        tmp_path, detection={"formats": formats}, split={"strategy": "random"}
    )
    generate(config)
    staging = config.writer.staging_dir
    assert (staging / "annotations" / "instances_train.json").exists() == ("coco" in formats)
    assert (staging / "dataset.yaml").exists() == ("yolo" in formats)
    assert (staging / "labels").exists() == ("yolo" in formats)
    assert (staging / "images").is_dir() and not (staging / "Masks").exists()


def test_min_label_ratio_drops_patches_without_enough_objects(tmp_path: Path) -> None:
    write_labels(tmp_path / "labels.geojson", random_features(seed=6, count=40))
    config = detection_config(tmp_path, sampler={"min_label_ratio": 0.02})
    with pytest.warns(UserWarning):
        manifest = generate(config)
    assert manifest.patches and len(manifest.patches) < 30
    assert all(entry["summary"]["class_objects"] for entry in manifest.patches)


def test_manifest_records_the_detection_target(
    unsplit: Tuple[MapcvConfig, List[Feature]],
) -> None:
    config, _ = unsplit
    manifest = Manifest.load(config.writer.staging_dir / "manifest.json")
    assert manifest.task == "detection"
    target = manifest.target
    assert target is not None and target.type == "detection"
    assert target.class_map == {"building": 1, "car": 2, "tree": 3}
    assert target.ignore_index is None and target.dtype is None
    assert target.options == {
        "min_visible": 0.3,
        "min_box_pixels": 2.0,
        "formats": ["coco", "yolo"],
        "point_box_size": 6.0,
    }
    assert set(target.labels or {}) == {"label_field", "classes", "sha256"}
    assert manifest.writer is not None and "mask_format" not in manifest.writer
    for entry in manifest.patches:
        assert list(entry["files"]) == ["image"]
        assert entry["files"]["image"].startswith("images/")
        assert list(entry["summary"]) == ["class_objects", "empty_ratio"]


# ── config validation ───────────────────────────────────────────────────────


def _raw(tmp_path: Path, **changes: Any) -> Dict[str, Any]:
    data: Dict[str, Any] = {
        "task": "detection",
        "region": {"west": 4.93, "south": 52.37, "east": 4.95, "north": 52.38},
        "imagery": {"type": "xyz", "zoom": 18, "source": "esri_satellite"},
        "labels": {"path": str(tmp_path / "labels.geojson")},
        "sampler": {"patch_size": 256},
        "writer": {"staging_dir": str(tmp_path / "dataset")},
    }
    data.update(changes)
    return data


def test_detection_defaults(tmp_path: Path) -> None:
    config = MapcvConfig.model_validate(_raw(tmp_path))
    assert config.task == "detection" and config.detection is None
    assert config.detection_options == DetectionOptions(
        min_visible=0.3, min_box_pixels=2.0, formats=["coco", "yolo"], point_box_size=None
    )
    reordered = MapcvConfig.model_validate(_raw(tmp_path, detection={"formats": ["yolo", "coco"]}))
    assert reordered.detection_options.formats == ["coco", "yolo"]


@pytest.mark.parametrize(
    ("changes", "message"),
    [
        ({"labels": None}, "task: detection needs labels"),
        ({"detection": {"min_visible": 1.5}}, "min_visible"),
        ({"detection": {"min_box_pixels": -1}}, "min_box_pixels"),
        ({"detection": {"formats": []}}, "at least one"),
        ({"detection": {"formats": ["coco", "coco"]}}, "duplicates"),
        ({"detection": {"formats": ["voc"]}}, "formats"),
        ({"detection": {"point_box_size": 0}}, "point_box_size"),
        ({"detection": {"min_visibility": 0.3}}, "min_visibility"),
        ({"task": "segmentation", "detection": {}}, "only applies to task: detection"),
        (
            {"labels": {"path": "x.geojson", "ignore_index": 255}},
            "labels.ignore_index marks mask pixels",
        ),
        ({"labels": {"path": "x.geojson", "all_touched": True}}, "all_touched"),
        ({"labels": {"path": "x.kml"}, "detection": {"point_box_size": 8}}, "KML points"),
        ({"sampler": {"patch_size": 256, "pad_mode": "reflect"}}, "mirrors objects"),
    ],
)
def test_invalid_detection_configs_fail_clearly(
    tmp_path: Path, changes: Dict[str, Any], message: str
) -> None:
    with pytest.raises(ValidationError, match=message):
        MapcvConfig.model_validate(_raw(tmp_path, **changes))


def test_reflect_padding_is_fine_without_edge_padding(tmp_path: Path) -> None:
    sampler = {"patch_size": 256, "pad_mode": "reflect", "edge_strategy": "shift"}
    assert MapcvConfig.model_validate(_raw(tmp_path, sampler=sampler)).task == "detection"


def test_factories_pick_the_detection_target_and_writer(tmp_path: Path) -> None:
    config = MapcvConfig.model_validate(_raw(tmp_path))
    target = create_target(config)
    assert isinstance(target, DetectionTarget)
    writer = create_writer(config.writer, target)
    assert isinstance(writer, DetectionWriter)
    check_compatible(target, writer)
    with pytest.raises(ValueError, match="cannot write detection targets"):
        check_compatible(target, FilesWriter(config.writer))


def test_large_objects_that_can_never_be_kept_are_reported(tmp_path: Path) -> None:
    big = box(X0, Y0 - 200 * RES, X0 + 200 * RES, Y0)  # 40,000 px²; a patch shows 4,096
    write_labels(tmp_path / "labels.geojson", [Feature(big, "tree")])
    config = detection_config(tmp_path)
    with pytest.warns(UserWarning, match="so large"):
        manifest = generate(config, FakeSource(valid=np.ones((HEIGHT, WIDTH), dtype=np.bool_)))
    assert not any(entry["summary"]["class_objects"] for entry in manifest.patches)
    lenient = detection_config(tmp_path, detection={"min_visible": 0.0}, staging="lenient")
    with pytest.warns(UserWarning, match="no split"):
        manifest = generate(lenient, FakeSource(valid=np.ones((HEIGHT, WIDTH), dtype=np.bool_)))
    covered = [entry for entry in manifest.patches if entry["summary"]["class_objects"]]
    assert len(covered) == 16  # 200 px spans 4 x 4 patches of 64 px (3.125 rounded up)


# ── the CLI, end to end against a local tile server ─────────────────────────

TILE_ZOOM, TILE_X0, TILE_Y0, TILES_X, TILES_Y = 18, 134_700, 86_100, 5, 4
MISSING_TILE = (TILE_X0 + 3, TILE_Y0 + 1)


@pytest.fixture()
def tile_server() -> Iterator[str]:
    import io
    import threading
    from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

    from PIL import Image

    class Tiles(BaseHTTPRequestHandler):
        def log_message(self, *args: object) -> None:
            pass

        def do_GET(self) -> None:  # noqa: N802 - http.server API
            z, x, y = (int(part) for part in self.path.strip("/").split(".")[0].split("/"))
            if (x, y) == MISSING_TILE:
                self.send_error(404)
                return
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


def test_cli_builds_coco_and_yolo_from_a_tile_server(tmp_path: Path, tile_server: str) -> None:
    from typer.testing import CliRunner

    from mapcv.cli import app

    pycoco = pytest.importorskip("pycocotools.coco")
    rng = random.Random(8)
    features = []
    for index in range(80):
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
task: detection
region:
  west: {_tile_lon(TILE_X0) + eps}
  south: {_tile_lat(TILE_Y0 + TILES_Y) + eps}
  east: {_tile_lon(TILE_X0 + TILES_X) - eps}
  north: {_tile_lat(TILE_Y0) - eps}
imagery:
  type: xyz
  zoom: {TILE_ZOOM}
  url_template: "{tile_server}"
  max_failed_ratio: 0.1
labels:
  path: labels.geojson
  label_field: kind
detection:
  min_visible: 0.25
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
    result = runner.invoke(app, ["plan", config])
    assert result.exit_code == 0, result.output
    # 75 of the 80 features reach into the region.
    assert "Objects  ≈ 75" in result.output
    assert "detection · coco, yolo · min_visible 0.25" in result.output

    result = runner.invoke(app, ["generate", config, "--yes"])
    assert result.exit_code == 0, result.output
    assert "objects" in result.output and "dataset.yaml" in result.output
    staging = tmp_path / "dataset"
    manifest = Manifest.load(staging / "manifest.json")
    assert manifest.task == "detection" and manifest.class_map == {"house": 1, "shed": 2}
    # 1,280 x 1,024 px in 192 px patches with padding: 7 x 6.
    assert len(manifest.patches) == 42
    assert {path.name for path in (staging / "images").iterdir()} == {
        Path(entry["files"]["image"]).name for entry in manifest.patches
    }
    data = yaml.safe_load((staging / "dataset.yaml").read_text(encoding="utf-8"))
    assert data["names"] == {0: "house", 1: "shed"}
    total = 0
    for split in ("train", "val", "test"):
        path = staging / "annotations" / f"instances_{split}.json"
        coco = load_coco(path)
        strict_coco_check(coco, 192)
        api = pycoco.COCO(str(path))
        names = (staging / "splits" / f"{split}.txt").read_text().split()
        assert sorted(api.imgs[i]["file_name"] for i in api.getImgIds()) == sorted(names)
        for image_id in api.getImgIds():
            anns = api.loadAnns(api.getAnnIds(imgIds=[image_id]))
            total += len(anns)
            label = staging / "labels" / (Path(api.imgs[image_id]["file_name"]).stem + ".txt")
            assert label.exists() == bool(anns)
            if anns:
                rows = [line.split() for line in label.read_text().splitlines()]
                for row, ann in zip(rows, anns):
                    bx, by, bw, bh = (float(v) * 192 for v in row[1:])
                    assert [bx - bw / 2, by - bh / 2, bw, bh] == pytest.approx(
                        ann["bbox"], abs=1e-6
                    )
                    assert data["names"][int(row[0])] == api.cats[ann["category_id"]]["name"]
    assert total > 40

    # The missing tile has no imagery: no box reaches into it.
    for entry, stored in zip(manifest.patches, load_objects_for(staging, manifest)):
        for _, ox, oy, ow, oh, *_ in stored:
            gx0 = entry["col"] + ox
            gy0 = entry["row"] + oy
            tx0 = (MISSING_TILE[0] - TILE_X0) * 256
            ty0 = (MISSING_TILE[1] - TILE_Y0) * 256
            inside = gx0 >= tx0 and gx0 + ow <= tx0 + 256 and gy0 >= ty0 and gy0 + oh <= ty0 + 256
            assert not inside

    result = runner.invoke(app, ["info", str(staging)])
    assert result.exit_code == 0, result.output
    assert "detection" in result.output and "objects" in result.output and "house" in result.output

    result = runner.invoke(app, ["split", str(staging), "--strategy", "random", "--seed", "3"])
    assert result.exit_code == 0, result.output
    test_names = (staging / "splits" / "test.txt").read_text().split()
    coco = load_coco(staging / "annotations" / "instances_test.json")
    assert sorted(image["file_name"] for image in coco["images"]) == sorted(test_names)
    assert (staging / "test.txt").read_text().split() == [f"./images/{n}" for n in test_names]


def load_objects_for(staging: Path, manifest: Manifest) -> List[List[List[Any]]]:
    from mapcv.writers.detection import load_objects

    return load_objects(manifest, staging)


def test_init_wizard_asks_for_the_task_and_box_formats(tmp_path: Path) -> None:
    from typer.testing import CliRunner

    from mapcv.cli import app

    labels = tmp_path / "aoi.geojson"
    labels.write_text(
        '{"type":"FeatureCollection","features":[{"type":"Feature","properties":{"kind":"roof"},'
        '"geometry":{"type":"Polygon","coordinates":[[[74.3,31.5],[74.31,31.5],'
        "[74.31,31.51],[74.3,31.5]]]}}]}"
    )
    out = tmp_path / "mapcv.yaml"
    answers = ["esri", str(labels), "17", "y", "kind", "detection", "yolo", "256", "./ds", "y"]
    result = CliRunner().invoke(
        app, ["init", str(out), "--interactive"], input="\n".join(answers) + "\n"
    )
    assert result.exit_code == 0, result.output
    config = MapcvConfig.from_yaml(out)
    assert config.task == "detection"
    assert config.detection_options.formats == ["yolo"]
    assert config.detection_options.min_visible == 0.3


def test_self_intersecting_polygons_are_repaired(tmp_path: Path) -> None:
    left, top = X0 + 10 * RES, Y0 - 10 * RES
    # A bowtie: two triangles meeting at (20, 20) px.
    bowtie = Polygon(
        [
            (left, top),
            (left + 20 * RES, top - 20 * RES),
            (left + 20 * RES, top),
            (left, top - 20 * RES),
            (left, top),
        ]
    )
    assert not bowtie.is_valid
    write_labels(tmp_path / "labels.geojson", [Feature(bowtie, "building")])
    config = detection_config(tmp_path)
    with pytest.warns(UserWarning, match="no split"):
        generate(config, FakeSource(valid=np.ones((HEIGHT, WIDTH), dtype=np.bool_)))
    (ann,) = load_coco(config.writer.staging_dir / "annotations" / "instances_all.json")[
        "annotations"
    ]
    assert ann["bbox"] == [10.0, 10.0, 20.0, 20.0]
    assert ann["area"] == pytest.approx(200.0) and not ann["truncated"]


def test_npy_patches_get_boxes_too(tmp_path: Path) -> None:
    from mapcv.manifest import SourceRecord, TargetRecord
    from mapcv.sampler import PatchMeta
    from mapcv.splitter import SplitLists
    from mapcv.targets.detection import DetectedObject, PatchObjects
    from mapcv.writer import WriterConfig

    config = WriterConfig(staging_dir=tmp_path, image_format="npy")
    writer = DetectionWriter(config, DetectionOptions())
    manifest = Manifest(
        task="detection",
        sources=[SourceRecord(bands=["b04", "b08"], dtype="float32")],
        target=TargetRecord(type="detection", class_map={"field": 3}, options={}),
        sampler={"patch_size": 8},
    )
    images = np.ones((2, 8, 8, 2), dtype=np.float32)
    objects = [
        PatchObjects((DetectedObject(3, (1.0, 2.0, 4.0, 3.5), 12.5, False),), (), 8),
        PatchObjects((), (), 8),
    ]
    meta = [PatchMeta(row=0, col=0, padded=False), PatchMeta(row=0, col=8, padded=False)]
    writer.write(images, objects, meta, manifest, 0)
    assert np.load(tmp_path / "images" / "patch_0000000.npy").shape == (2, 8, 8)
    assert [entry["summary"]["class_objects"] for entry in manifest.patches] == [{"3": 1}, {}]
    assert (tmp_path / "labels" / "patch_0000000.txt").read_text() == "0 0.375 0.46875 0.5 0.4375\n"
    assert not (tmp_path / "labels" / "patch_0000001.txt").exists()
    lists = SplitLists(train=["patch_0000000.npy"], val=["patch_0000001.npy"], test=[])
    writer.finalize(manifest, lists)
    train = load_coco(tmp_path / "annotations" / "instances_train.json")
    assert train["categories"] == [{"id": 3, "name": "field", "supercategory": "object"}]
    assert train["annotations"][0]["bbox"] == [1.0, 2.0, 4.0, 3.5]
    data = yaml.safe_load((tmp_path / "dataset.yaml").read_text())
    assert "test" not in data and data["names"] == {0: "field"}  # an empty test split


def test_kml_labels_work_for_detection(tmp_path: Path) -> None:
    left, top = X0 + 10 * RES, Y0 - 10 * RES
    corners = [
        to_lonlat(x, y)
        for x, y in [
            (left, top),
            (left + 12 * RES, top),
            (left + 12 * RES, top - 8 * RES),
            (left, top - 8 * RES),
            (left, top),
        ]
    ]
    ring = " ".join(f"{lon},{lat}" for lon, lat in corners)
    (tmp_path / "labels.kml").write_text(
        '<?xml version="1.0"?><kml xmlns="http://www.opengis.net/kml/2.2"><Document>'
        '<Placemark><ExtendedData><Data name="class"><value>car</value></Data></ExtendedData>'
        f"<Polygon><outerBoundaryIs><LinearRing><coordinates>{ring}</coordinates>"
        "</LinearRing></outerBoundaryIs></Polygon></Placemark></Document></kml>"
    )
    config = MapcvConfig.model_validate(
        {
            **detection_config(tmp_path).model_dump(mode="json", exclude_none=True),
            "labels": {"path": str(tmp_path / "labels.kml"), "label_field": "class"},
        }
    )
    with pytest.warns(UserWarning, match="no split"):
        generate(config, FakeSource(valid=np.ones((HEIGHT, WIDTH), dtype=np.bool_)))
    (ann,) = load_coco(config.writer.staging_dir / "annotations" / "instances_all.json")[
        "annotations"
    ]
    assert ann["bbox"] == pytest.approx([10.0, 10.0, 12.0, 8.0], abs=1e-3)


@dataclass
class GeoTiff:
    path: Path
    transform: Any  # rasterio Affine of the file


def write_geotiff(directory: Path, rotation: float) -> Tuple[GeoTiff, Dict[str, float]]:
    """A 400 x 300 px RGB UTM GeoTIFF near Paris with a NoData (0) block, and a region in it."""
    import rasterio
    from pyproj import Transformer
    from rasterio.transform import Affine

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
    return GeoTiff(path, transform), region


@pytest.mark.parametrize("rotation", [0.0, 12.0])
def test_geotiff_boxes_match_an_independent_clip_in_the_file_crs(
    tmp_path: Path, rotation: float
) -> None:
    pytest.importorskip("rasterio")
    from pyproj import Transformer
    from shapely.affinity import affine_transform

    raster, region = write_geotiff(tmp_path, rotation)
    # Irregular polygons over the region, one crossing the NoData block.
    rng = random.Random(1)
    west, south = region["west"], region["south"]
    dx, dy = region["east"] - west, region["north"] - south
    features: List[Dict[str, Any]] = []
    for index in range(12):
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
            "task": "detection",
            "region": region,
            "imagery": {"type": "geotiff", "path": str(raster.path)},
            "labels": {"path": str(labels), "label_field": "kind"},
            "detection": {"min_visible": 0.0, "min_box_pixels": 0},
            "sampler": {"patch_size": 48, "edge_strategy": "drop"},
            "writer": {"staging_dir": str(tmp_path / "dataset")},
        }
    )
    with pytest.warns(UserWarning, match="no split"):
        run_generate(config)
    staging = config.writer.staging_dir
    manifest = Manifest.load(staging / "manifest.json")
    coco = load_coco(staging / "annotations" / "instances_all.json")
    to_utm = Transformer.from_crs("EPSG:4326", "EPSG:32631", always_xy=True)
    worlds = [
        (
            shapely.transform(
                shapely.geometry.shape(feature["geometry"]),
                lambda xy: np.column_stack(to_utm.transform(xy[:, 0], xy[:, 1])),
            ),
            manifest.class_map[feature["properties"]["kind"]],
        )
        for feature in features
    ]
    a, b, c, d, e, f = manifest.source.transform or (0.0,) * 6
    det = a * e - b * d
    # World -> dataset raster pixels (the inverse affine) for shapely's affine_transform.
    inverse = [e / det, -b / det, -d / det, a / det, (b * f - e * c) / det, (d * c - a * f) / det]
    # Where the dataset window starts in the file, to place the NoData block.
    t = raster.transform
    t_det = t.a * t.e - t.b * t.d
    offset_col = round(((c - t.c) * t.e - (f - t.f) * t.b) / t_det)
    offset_row = round(((f - t.f) * t.a - (c - t.c) * t.d) / t_det)
    hole = box(200 - offset_col, 40 - offset_row, 260 - offset_col, 90 - offset_row)
    by_image: Dict[int, List[Dict[str, Any]]] = {}
    for ann in coco["annotations"]:
        by_image.setdefault(ann["image_id"], []).append(ann)
    checked = 0
    for index, entry in enumerate(manifest.patches):
        row, col = entry["row"], entry["col"]
        expected = []
        for world, cid in worlds:
            visible = affine_transform(world, inverse).intersection(
                box(col, row, col + 48, row + 48)
            )
            visible = visible.difference(hole)
            if visible.area <= 0:
                continue
            x0, y0, x1, y1 = visible.bounds
            expected.append((cid, [x0 - col, y0 - row, x1 - x0, y1 - y0]))
        found = [(ann["category_id"], ann["bbox"]) for ann in by_image.get(index + 1, [])]
        assert len(found) == len(expected), (index, found, expected)
        for (cid, bbox), (want_cid, want) in zip(found, expected):
            assert cid == want_cid
            assert bbox == pytest.approx(want, abs=2e-3)
            checked += 1
    assert checked >= 10
