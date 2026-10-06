"""Patch classification datasets (``task: classification``): one label or a label set per patch.

Checked against references, not by eye:

* a patch's coverage per class equals an independent computation: the mask GDAL burns
  (``rasterio.features.rasterize`` for vector labels, ``rasterio.warp.reproject`` of the label
  raster for ``labels.type: raster``) on the patch's own pixel grid, counted over the pixels that
  have imagery and are not ignored; rotated grids, NoData blocks and ignore pixels included;
* the kept patches and their labels equal the rule applied to that reference (single and
  multi, ties, ``min_fraction``, ``empty``);
* ``labels.csv``, ``labels_<split>.csv``, ``classes.txt`` and ``labels.json`` parse and agree
  with the manifest and the splits; the same config gives the same bytes; a resumed run and a
  re-split leave consistent files.
"""

from __future__ import annotations

import csv
import json
import math
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional, Sequence, Tuple

import numpy as np
import numpy.typing as npt
import pytest
import shapely
import yaml
from PIL import Image
from pydantic import ValidationError
from shapely.geometry import MultiPolygon, Point, Polygon, box, mapping
from shapely.geometry.base import BaseGeometry
from typer.testing import CliRunner

pytest.importorskip("rasterio", reason="coverage is checked against rasterio (GDAL)")
import rasterio  # noqa: E402
import rasterio.features  # noqa: E402
import rasterio.warp  # noqa: E402
from pyproj import Transformer  # noqa: E402
from rasterio.crs import CRS  # noqa: E402
from rasterio.enums import Resampling  # noqa: E402
from rasterio.transform import Affine  # noqa: E402

from mapcv.cli import app  # noqa: E402
from mapcv.config import ClassificationOptions, MapcvConfig  # noqa: E402
from mapcv.imagery import RasterMetadata  # noqa: E402
from mapcv.manifest import (  # noqa: E402
    Manifest,
    ManifestEntry,
    ManifestMismatchError,
    PatchSummary,
)
from mapcv.pipeline import run_generate, run_split  # noqa: E402
from mapcv.planning import _CLASSIFICATION_BYTES, plan  # noqa: E402
from mapcv.sampler import PatchMeta  # noqa: E402
from mapcv.splitter import SplitterConfig, _stratum  # noqa: E402
from mapcv.targets import ClassificationTarget, PatchLabels, create_target  # noqa: E402
from mapcv.targets.classification import ClassificationWindow  # noqa: E402
from mapcv.writers import (  # noqa: E402
    ClassificationWriter,
    FilesWriter,
    check_compatible,
    create_writer,
)

runner = CliRunner()
# Random splits of overlapping patches warn about leakage; edge patches warn about the region.
pytestmark = pytest.mark.filterwarnings(
    "ignore:Patches overlap or repeat:UserWarning", "ignore:the region extends:UserWarning"
)

R = 6_378_137.0
# A zoom-18 Web Mercator grid near Amsterdam.
RES = 0.5971642834779232
X0, Y0 = 549_582.2333704159, 6_868_784.235762509
HEIGHT, WIDTH = 300, 330
PATCH = 64
CLASSES = ("building", "car", "tree")
CLASS_IDS = {name: index + 1 for index, name in enumerate(CLASSES)}
IGNORE = 255


# ── the rule on hand-made masks ─────────────────────────────────────────────


def labels_of(
    mask: Sequence[Sequence[int]],
    *,
    valid: Optional[Sequence[Sequence[bool]]] = None,
    ignore: Optional[int] = IGNORE,
    **options: Any,
) -> PatchLabels:
    array = np.asarray(mask, dtype=np.uint8)
    window = ClassificationWindow(array, ignore, ClassificationOptions(**options))
    valid_patch = None if valid is None else np.asarray(valid, dtype=np.bool_)
    return window.annotate(0, 0, array.shape[0], "zero", valid_patch)


def block(*parts: Tuple[int, int]) -> List[List[int]]:
    """A 4 x 4 mask holding ``count`` pixels of each ``value``, in row-major order."""
    flat = [value for value, count in parts for _ in range(count)]
    flat += [0] * (16 - len(flat))
    return [flat[row * 4 : row * 4 + 4] for row in range(4)]


def test_coverage_is_the_share_of_valid_pixels() -> None:
    labelled = labels_of(block((1, 6), (2, 2)))
    assert labelled.coverage == {1: 0.375, 2: 0.125}
    assert labelled.labeled_pixels == 8 and labelled.patch_size == 4
    # Background has no coverage; a patch of it has no label.
    nothing = labels_of(block())
    assert nothing.coverage == {} and nothing.labels == () and nothing.labeled_pixels == 0


def test_single_picks_the_largest_class_and_ties_go_to_the_lowest_id() -> None:
    assert labels_of(block((3, 5), (1, 4), (2, 3))).labels == (3,)
    assert labels_of(block((7, 4), (2, 4), (5, 4))).labels == (2,)
    assert labels_of(block((9, 2), (4, 2))).labels == (4,)
    # Equal coverage, listed in the opposite order of the ids.
    assert labels_of([[9, 9, 3, 3]] * 4).labels == (3,)


def test_multi_gives_every_qualifying_class_in_id_order() -> None:
    mask = block((9, 3), (2, 1), (4, 6))
    assert labels_of(mask, mode="multi").labels == (2, 4, 9)
    assert labels_of(mask, mode="multi", min_fraction=0.15).labels == (4, 9)
    assert labels_of(mask, mode="multi", min_fraction=0.35).labels == (4,)
    assert labels_of(mask, mode="multi", min_fraction=0.35).coverage == {
        2: 1 / 16,
        4: 6 / 16,
        9: 3 / 16,
    }


def test_min_fraction_is_inclusive_and_zero_means_any_labeled_pixel() -> None:
    mask = block((5, 4), (6, 3), (7, 1))  # 0.25, 0.1875, 0.0625 of 16
    assert labels_of(mask, mode="multi", min_fraction=0.0).labels == (5, 6, 7)
    assert labels_of(mask, mode="multi", min_fraction=0.0625).labels == (5, 6, 7)
    assert labels_of(mask, mode="multi", min_fraction=0.0626).labels == (5, 6)
    assert labels_of(mask, mode="multi", min_fraction=0.1875).labels == (5, 6)
    assert labels_of(mask, mode="multi", min_fraction=0.1876).labels == (5,)
    assert labels_of(mask, mode="multi", min_fraction=0.25).labels == (5,)
    assert labels_of(mask, mode="multi", min_fraction=0.2501).labels == ()
    # Single: the largest class must itself reach the fraction.
    assert labels_of(mask, min_fraction=0.25).labels == (5,)
    assert labels_of(mask, min_fraction=0.26).labels == ()
    assert labels_of(block((1, 16)), min_fraction=1.0).labels == (1,)


def test_a_patch_without_a_qualifying_class_is_skipped_or_labeled_background() -> None:
    mask = block((5, 2))
    skipped = labels_of(mask, min_fraction=0.5)
    assert skipped.labels == () and skipped.coverage == {5: 0.125}
    kept = labels_of(mask, min_fraction=0.5, empty="background")
    assert kept.labels == (0,) and kept.coverage == {5: 0.125}
    # A class that qualifies replaces the background label.
    assert labels_of(mask, empty="background").labels == (5,)
    assert labels_of(block(), empty="background", mode="multi").labels == (0,)
    blind = labels_of(block(), valid=[[False] * 4] * 4, empty="background")
    assert blind.labels == () and blind.coverage == {}


def test_ignored_and_invalid_pixels_are_not_counted() -> None:
    # 8 ignored pixels leave 8 valid ones, 4 of them class 1.
    mask = [[1, 1, 255, 255], [1, 1, 255, 255], [0, 0, 255, 255], [0, 0, 255, 255]]
    assert labels_of(mask).coverage == {1: 0.5}
    # Imagery-less pixels are not valid either: class 2 sits on 2 invalid pixels.
    valid = [[True] * 4, [True] * 4, [True, True, False, False], [True] * 4]
    mask2 = [[1, 1, 0, 0], [1, 0, 0, 0], [0, 0, 2, 2], [0, 0, 0, 0]]
    shown = labels_of(mask2, valid=valid)
    assert shown.coverage == {1: 3 / 14} and shown.labels == (1,)
    assert shown.labeled_pixels == 3
    # Without an ignore value, only imagery decides.
    assert labels_of(mask2, valid=valid, ignore=None).coverage == {1: 3 / 14}
    # No valid pixel at all: nothing to label.
    assert labels_of(mask2, valid=[[False] * 4] * 4).labels == ()
    assert labels_of(mask2, valid=[[False] * 4] * 4, empty="background").labels == ()


def test_padding_is_not_valid() -> None:
    window = ClassificationWindow(
        np.ones((4, 4), dtype=np.uint8), IGNORE, ClassificationOptions(empty="background")
    )
    # The patch sticks out of the window by 2 rows and 2 columns: 4 of its 16 pixels exist.
    edge = window.annotate(2, 2, 4, "zero", None)
    assert edge.coverage == {1: 1.0} and edge.labeled_pixels == 4
    edge = window.annotate(-3, -1, 4, "zero", np.ones((4, 4), dtype=np.bool_))
    assert edge.coverage == {1: 1.0} and edge.labeled_pixels == 3
    # Reflect padding never mirrors labels in: only the 4 real pixels count.
    assert window.annotate(3, 0, 4, "reflect", None).labeled_pixels == 4
    # A patch without any valid pixel cannot be judged: dropped even with empty: background.
    blind = window.annotate(0, 0, 4, "zero", np.zeros((4, 4), dtype=np.bool_))
    assert blind.coverage == {} and blind.labels == () and not window.accepts(blind, 0.0)


def test_accepts_applies_min_label_ratio_to_the_whole_patch_and_requires_a_label() -> None:
    window = ClassificationWindow(np.zeros((4, 4), dtype=np.uint8), None, ClassificationOptions())
    quarter = labels_of(block((1, 4)))
    assert window.accepts(quarter, 0.0) and window.accepts(quarter, 0.25)
    assert not window.accepts(quarter, 0.2501)
    # Ignored pixels count in the whole patch, not as labeled.
    ignored = labels_of([[1, 255, 255, 255]] * 4)
    assert ignored.coverage == {1: 1.0} and window.accepts(ignored, 0.25)
    assert not window.accepts(ignored, 0.26)
    assert not window.accepts(labels_of(block()), 0.0)
    assert window.accepts(labels_of(block(), empty="background"), 0.0)


# ── a Web Mercator raster with holes in its validity mask ───────────────────


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
    """A Web Mercator RGB raster with holes in its validity mask, optionally rotated."""

    def __init__(
        self, valid: Optional[npt.NDArray[np.bool_]] = None, rotation: float = 0.0
    ) -> None:
        rng = np.random.default_rng(0)
        self.valid = make_valid_mask() if valid is None else valid
        image = rng.integers(1, 256, size=(HEIGHT, WIDTH, 3), dtype=np.uint8)
        image[~self.valid] = 0
        self.image = image
        affine = Affine(RES, 0.0, X0, 0.0, -RES, Y0)
        if rotation:
            affine = affine * Affine.rotation(rotation, (WIDTH / 2, HEIGHT / 2))
        self.affine = affine
        a, b, c, d, e, f = tuple(affine)[:6]
        self.metadata = RasterMetadata(
            source_type="xyz",
            product_id="fake",
            width=WIDTH,
            height=HEIGHT,
            bands=["red", "green", "blue"],
            dtype="uint8",
            crs="EPSG:3857",
            transform=(a, b, c, d, e, f),
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


def random_features(seed: int, count: int = 90) -> List[Feature]:
    """Small and large, multi-part and holed polygons; they overlap and cross patch edges."""
    rng = random.Random(seed)
    left, top = X0 - 20 * RES, Y0 + 20 * RES
    span_x, span_y = (WIDTH + 40) * RES, (HEIGHT + 40) * RES
    features: List[Feature] = []
    for index in range(count):
        cx = left + rng.uniform(0, span_x)
        cy = top - rng.uniform(0, span_y)
        kind = index % 4
        name = rng.choice(CLASSES)
        if kind == 0:
            geometry: BaseGeometry = star(rng, cx, cy, rng.uniform(3, 40) * RES)
        elif kind == 1:
            first = star(rng, cx, cy, rng.uniform(2, 15) * RES)
            second = star(
                rng,
                cx + rng.uniform(-40, 40) * RES,
                cy + rng.uniform(-40, 40) * RES,
                rng.uniform(2, 15) * RES,
            )
            geometry = first if first.intersects(second) else MultiPolygon([first, second])
        elif kind == 2:
            radius = rng.uniform(8, 50) * RES
            outer = Point(cx, cy).buffer(radius, quad_segs=rng.randint(2, 8))
            hole = Point(cx, cy).buffer(radius * 0.5, quad_segs=3)
            geometry = Polygon(outer.exterior.coords, [hole.exterior.coords])
        else:
            geometry = star(rng, cx, cy, rng.uniform(40, 120) * RES)
        features.append(Feature(geometry, name))
    return features


def write_labels(path: Path, features: List[Feature], field: str = "class") -> None:
    collection = {
        "type": "FeatureCollection",
        "features": [
            {
                "type": "Feature",
                "properties": {field: feature.name},
                "geometry": mapping(lonlat_geometry(feature.world)),
            }
            for feature in features
        ],
    }
    path.write_text(json.dumps(collection), encoding="utf-8")


def config_for(
    root: Path,
    *,
    classification: Optional[Dict[str, Any]] = None,
    split: Optional[Dict[str, Any]] = None,
    sampler: Optional[Dict[str, Any]] = None,
    labels: Optional[Dict[str, Any]] = None,
    writer: Optional[Dict[str, Any]] = None,
    staging: str = "dataset",
) -> MapcvConfig:
    data: Dict[str, Any] = {
        "task": "classification",
        "region": {"west": 4.93, "south": 52.37, "east": 4.95, "north": 52.38},
        "imagery": {"type": "xyz", "zoom": 18, "url_template": "http://127.0.0.1/{z}/{x}/{y}.png"},
        "labels": {"path": str(root / "labels.geojson"), "label_field": "class", **(labels or {})},
        "sampler": {"patch_size": PATCH, "edge_strategy": "drop", **(sampler or {})},
        "writer": {"staging_dir": str(root / staging), "image_format": "png", **(writer or {})},
    }
    if classification is not None:
        data["classification"] = classification
    if split is not None:
        data["split"] = split
    return MapcvConfig.model_validate(data)


def generate(config: MapcvConfig, source: Optional[FakeSource] = None) -> Manifest:
    with pytest.MonkeyPatch.context() as patch:
        patch.setattr("mapcv.pipeline.open_raster_source", lambda *a, **k: source or FakeSource())
        run_generate(config)
    return Manifest.load(config.writer.staging_dir / "manifest.json")


# ── the independent computation (rasterio / GDAL) ───────────────────────────


@dataclass
class Counts:
    by_class: Dict[int, int]  # pixels per class ID (background excluded)
    valid: int  # valid pixels: with imagery, not ignored


def rule(counts: Counts, mode: str, min_fraction: float, empty: str) -> Optional[Tuple[int, ...]]:
    """The labels the rule gives (``(0,)`` is background); ``None`` for a skipped patch.

    Written apart from mapcv's: classes sorted by pixel count rather than a running maximum.
    """
    shares = {cid: n / counts.valid for cid, n in counts.by_class.items() if n > 0}
    qualifying = sorted(cid for cid, share in shares.items() if share >= min_fraction)
    if mode == "single":
        ranked = sorted(qualifying, key=lambda cid: (-counts.by_class[cid], cid))
        labels = tuple(ranked[:1])
    else:
        labels = tuple(qualifying)
    if labels:
        return labels
    return (0,) if empty == "background" else None


def vector_counts(
    features: List[Feature],
    valid: npt.NDArray[np.bool_],
    affine: Affine,
    row: int,
    col: int,
    *,
    all_touched: bool,
    ignore: Optional[int] = IGNORE,
) -> Counts:
    """GDAL's mask of the labels on the patch's grid, counted over the valid pixels."""
    patch = affine * Affine.translation(col, row)
    shapes = [(feature.world, CLASS_IDS[feature.name]) for feature in features]
    burned = rasterio.features.rasterize(
        shapes, out_shape=(PATCH, PATCH), transform=patch, all_touched=all_touched, dtype="uint8"
    )
    padded = np.zeros((HEIGHT + 2 * PATCH, WIDTH + 2 * PATCH), dtype=bool)
    padded[PATCH : PATCH + HEIGHT, PATCH : PATCH + WIDTH] = valid
    inside = padded[PATCH + row : PATCH + row + PATCH, PATCH + col : PATCH + col + PATCH]
    if ignore is not None:
        inside = inside & (burned != ignore)
    values, numbers = np.unique(burned[inside], return_counts=True)
    by_class = {int(v): int(n) for v, n in zip(values, numbers) if v != 0}
    return Counts(by_class, int(inside.sum()))


def grid_anchors(edge: str, stride: int) -> List[Tuple[int, int]]:
    last_row = HEIGHT - PATCH if edge == "drop" else HEIGHT - 1
    last_col = WIDTH - PATCH if edge == "drop" else WIDTH - 1
    return [
        (row, col)
        for row in range(0, last_row + 1, stride)
        for col in range(0, last_col + 1, stride)
    ]


def assert_dataset_equals_reference(
    manifest: Manifest,
    features: List[Feature],
    source: FakeSource,
    *,
    mode: str = "single",
    min_fraction: float = 0.0,
    empty: str = "skip",
    all_touched: bool = False,
    ignore: Optional[int] = IGNORE,
    edge: str = "drop",
    stride: int = PATCH,
) -> Tuple[int, int]:
    """Every anchor is kept exactly when the reference rule says so, with the same coverage
    and labels. Returns the numbers of kept and skipped patches."""
    kept = {(entry["row"], entry["col"]): entry for entry in manifest.patches}
    skipped = 0
    for row, col in grid_anchors(edge, stride):
        counts = vector_counts(
            features, source.valid, source.affine, row, col, all_touched=all_touched, ignore=ignore
        )
        expected = rule(counts, mode, min_fraction, empty) if counts.valid else None
        if counts.valid == 0 and empty == "background":
            expected = None  # nothing to look at: no valid pixel, no label
        entry = kept.get((row, col))
        if expected is None:
            assert entry is None, f"patch ({row}, {col}) should have been skipped"
            skipped += 1
            continue
        assert entry is not None, f"patch ({row}, {col}) is missing, expected {expected}"
        summary = entry["summary"]
        assert summary["labels"] == list(expected), (row, col)
        reference = {str(cid): n / counts.valid for cid, n in sorted(counts.by_class.items())}
        assert list(summary["class_coverage"]) == list(reference), (row, col)
        for cid, share in reference.items():
            assert summary["class_coverage"][cid] == pytest.approx(share, abs=1e-15, rel=0)
    assert len(kept) == len(manifest.patches)
    assert {(e["row"], e["col"]) for e in manifest.patches} <= set(grid_anchors(edge, stride))
    return len(kept), skipped


VECTOR_CASES = [
    # (id, classification options, valid mask kind, rotation, all_touched, sampler)
    ("defaults", {}, "edges", 0.0, False, {}),
    ("defaults-rotated", {}, "edges", 17.0, False, {}),
    ("multi", {"mode": "multi"}, "edges", 0.0, False, {}),
    ("multi-rotated-noisy", {"mode": "multi"}, "noisy", -31.0, False, {}),
    ("multi-min-fraction", {"mode": "multi", "min_fraction": 0.1}, "edges", 0.0, True, {}),
    ("single-min-fraction", {"min_fraction": 0.35}, "noisy", 8.0, False, {}),
    ("background", {"empty": "background", "min_fraction": 0.3}, "edges", 0.0, False, {}),
    (
        "multi-background-rotated",
        {"mode": "multi", "empty": "background", "min_fraction": 0.05},
        "noisy",
        45.0,
        True,
        {},
    ),
    ("overlapping", {"mode": "multi"}, "edges", 0.0, False, {"stride": 40}),
    ("padded-edges", {"mode": "multi", "min_fraction": 0.02}, "edges", 0.0, False, {"pad": True}),
    ("padded-rotated", {"empty": "background"}, "noisy", 12.0, False, {"pad": True}),
    ("full-imagery", {"mode": "multi"}, "full", 0.0, False, {}),
]


@pytest.mark.parametrize(
    ("options", "mask_kind", "rotation", "all_touched", "extra"),
    [case[1:] for case in VECTOR_CASES],
    ids=[case[0] for case in VECTOR_CASES],
)
def test_vector_coverage_and_labels_equal_the_gdal_reference(
    tmp_path: Path,
    options: Dict[str, Any],
    mask_kind: str,
    rotation: float,
    all_touched: bool,
    extra: Dict[str, Any],
) -> None:
    features = random_features(seed=7)
    write_labels(tmp_path / "labels.geojson", features)
    source = FakeSource(make_valid_mask(mask_kind), rotation)
    stride = int(extra.get("stride", PATCH))
    edge = "pad" if extra.get("pad") else "drop"
    config = config_for(
        tmp_path,
        classification=options,
        labels={"all_touched": all_touched},
        sampler={"edge_strategy": edge, "stride": stride},
    )
    manifest = generate(config, source)
    kept, skipped = assert_dataset_equals_reference(
        manifest,
        features,
        source,
        mode=options.get("mode", "single"),
        min_fraction=options.get("min_fraction", 0.0),
        empty=options.get("empty", "skip"),
        all_touched=all_touched,
        edge=edge,
        stride=stride,
    )
    assert kept >= 8
    if options.get("min_fraction", 0.0) >= 0.3 and options.get("empty", "skip") == "skip":
        assert skipped >= 1


def test_the_reference_comparison_has_teeth(tmp_path: Path) -> None:
    """The reference detects a changed rule, so a passing comparison means something."""
    features = random_features(seed=7)
    write_labels(tmp_path / "labels.geojson", features)
    source = FakeSource(make_valid_mask("noisy"), 17.0)
    manifest = generate(config_for(tmp_path, classification={"mode": "multi"}), source)
    with pytest.raises(AssertionError):
        assert_dataset_equals_reference(manifest, features, source, mode="single")
    with pytest.raises(AssertionError):
        assert_dataset_equals_reference(manifest, features, source, mode="multi", min_fraction=0.2)
    with pytest.raises(AssertionError):
        assert_dataset_equals_reference(manifest, features, source, mode="multi", all_touched=True)
    shifted = FakeSource(make_valid_mask("noisy"), 18.0)
    with pytest.raises(AssertionError):
        assert_dataset_equals_reference(manifest, features, shifted, mode="multi")


def test_without_an_ignore_value_only_imagery_decides_what_is_valid(tmp_path: Path) -> None:
    features = random_features(seed=9)
    write_labels(tmp_path / "labels.geojson", features)
    source = FakeSource(make_valid_mask("edges"), 9.0)
    config = config_for(
        tmp_path,
        classification={"mode": "multi", "min_fraction": 0.05},
        labels={"ignore_index": None},
        sampler={"edge_strategy": "pad"},
    )
    manifest = generate(config, source)
    assert manifest.ignore_index is None
    assert_dataset_equals_reference(
        manifest,
        features,
        source,
        mode="multi",
        min_fraction=0.05,
        ignore=None,
        edge="pad",
    )


def test_overlapping_polygons_follow_the_later_wins_rule(tmp_path: Path) -> None:
    """The later feature takes the pixels both cover, as in the segmentation mask."""

    def square(first: float, last: float) -> BaseGeometry:
        # Edges a quarter pixel off the pixel grid: no pixel centre lies on an edge.
        return box(X0 + first * RES, Y0 - last * RES, X0 + last * RES, Y0 - first * RES)

    big = square(5.25, 45.25)  # 40 x 40 px
    small = square(25.25, 60.25)  # 35 x 35 px, 20 x 20 of them inside the first
    for order, areas in (
        (["building", "car"], {1: 1200, 2: 1225}),  # car, the later one, keeps its 1,225 px
        (["car", "building"], {2: 1200, 1: 1225}),
    ):
        write_labels(
            tmp_path / "labels.geojson", [Feature(big, order[0]), Feature(small, order[1])]
        )
        config = config_for(tmp_path, classification={"mode": "multi"}, staging="-".join(order))
        manifest = generate(config, FakeSource(make_valid_mask("full")))
        first = manifest.patches[0]
        assert (first["row"], first["col"]) == (0, 0) and len(manifest.patches) == 1
        coverage = first["summary"]["class_coverage"]
        assert {int(cid): round(share * PATCH**2) for cid, share in coverage.items()} == areas
        assert first["summary"]["labels"] == [1, 2]


def test_a_patch_over_a_label_free_area_is_skipped_or_background(tmp_path: Path) -> None:
    write_labels(
        tmp_path / "labels.geojson", [Feature(box(X0, Y0 - 30 * RES, X0 + 30 * RES, Y0), "tree")]
    )
    skipping = generate(config_for(tmp_path), FakeSource(make_valid_mask("full")))
    assert [(e["row"], e["col"]) for e in skipping.patches] == [(0, 0)]
    assert skipping.class_map == {"tree": 1}
    keeping = generate(
        config_for(tmp_path, classification={"empty": "background"}, staging="bg"),
        FakeSource(make_valid_mask("full")),
    )
    assert len(keeping.patches) == 20  # 4 x 5 patches
    labels = [tuple(e["summary"]["labels"]) for e in keeping.patches]
    assert labels[0] == (1,) and set(labels[1:]) == {(0,)}
    assert all(e["summary"]["class_coverage"] == {} for e in keeping.patches[1:])


def test_no_label_features_near_the_raster_give_background_or_nothing(tmp_path: Path) -> None:
    write_labels(
        tmp_path / "labels.geojson",
        [Feature(box(X0 + 5000, Y0 - 5100, X0 + 5100, Y0 - 5000), "tree")],
    )
    with pytest.warns(UserWarning, match="no label polygon intersects the imagery extent"):
        nothing = generate(config_for(tmp_path), FakeSource())
    assert nothing.patches == []
    with pytest.warns(UserWarning, match="no label polygon intersects"):
        everything = generate(
            config_for(tmp_path, classification={"empty": "background"}, staging="bg"),
            FakeSource(make_valid_mask("full")),
        )
    assert {tuple(e["summary"]["labels"]) for e in everything.patches} == {(0,)}
    assert len(everything.patches) == 20
    assert (tmp_path / "bg" / "classes.txt").read_text() == "background\ntree\n"


def test_min_label_ratio_keeps_its_segmentation_meaning(tmp_path: Path) -> None:
    """The labeled share of the whole patch must reach it, before any class rule applies."""
    features = random_features(seed=5)
    write_labels(tmp_path / "labels.geojson", features)
    source = FakeSource(make_valid_mask("edges"))
    options = {"mode": "multi", "min_fraction": 0.1}
    everything = generate(config_for(tmp_path, classification=options), source)
    strict = generate(
        config_for(
            tmp_path, classification=options, sampler={"min_label_ratio": 0.4}, staging="strict"
        ),
        source,
    )

    def labeled_ratio(entry: Any) -> float:
        counts = vector_counts(
            features, source.valid, source.affine, entry["row"], entry["col"], all_touched=False
        )
        return sum(counts.by_class.values()) / PATCH**2

    wanted = [e for e in everything.patches if labeled_ratio(e) >= 0.4]
    assert 0 < len(wanted) < len(everything.patches)
    assert [(e["row"], e["col"]) for e in strict.patches] == [(e["row"], e["col"]) for e in wanted]
    for kept, reference in zip(strict.patches, wanted):
        assert kept["summary"] == reference["summary"]


# ── classes without a label_field, and class names ──────────────────────────


def test_without_a_label_field_every_feature_is_the_class_object(tmp_path: Path) -> None:
    write_labels(
        tmp_path / "labels.geojson",
        [Feature(box(X0, Y0 - 30 * RES, X0 + 30 * RES, Y0), "tree")],
    )
    config = config_for(
        tmp_path, classification={"empty": "background"}, labels={"label_field": None}
    )
    manifest = generate(config, FakeSource(make_valid_mask("full")))
    assert manifest.class_map == {}
    assert (config.writer.staging_dir / "classes.txt").read_text() == "background\nobject\n"
    rows = read_csv(config.writer.staging_dir / "labels.csv")
    assert rows[0] == {"image": "patch_0000000.png", "labels": "object"}
    assert {row["labels"] for row in rows[1:]} == {"background"}


def named_features(names: Sequence[str]) -> List[Feature]:
    """One 20 x 20 px square per name, one patch (64 x 64 px) each along the top row."""
    return [
        Feature(
            box(
                X0 + (i * PATCH + 10) * RES,
                Y0 - 30 * RES,
                X0 + (i * PATCH + 30) * RES,
                Y0 - 10 * RES,
            ),
            name,
        )
        for i, name in enumerate(names)
    ]


def test_single_label_names_may_hold_spaces_commas_and_unicode(tmp_path: Path) -> None:
    names = ["open water", "wet,land", 'say "hi"', "forêt"]
    write_labels(tmp_path / "labels.geojson", named_features(names))
    config = config_for(tmp_path, sampler={"edge_strategy": "drop"})
    generate(config, FakeSource(make_valid_mask("full")))
    staging = config.writer.staging_dir
    rows = read_csv(staging / "labels.csv")
    classes = (staging / "classes.txt").read_text(encoding="utf-8").splitlines()
    assert sorted(classes) == sorted(names)
    parsed = {row["image"]: row["labels"] for row in rows}
    document = json.loads((staging / "labels.json").read_text(encoding="utf-8"))
    assert document["classes"] == classes
    assert {image: " ".join(labels) for image, labels in document["images"].items()} == parsed
    assert set(parsed.values()) == set(names)


@pytest.mark.parametrize(
    ("name", "options", "message"),
    [
        ("open water", {"mode": "multi"}, "contains whitespace"),
        ("tab\tbed", {"mode": "single"}, "control character"),
        ("line\nbreak", {"mode": "multi"}, "control character"),
        ("background", {"empty": "background"}, "label of patches without a class"),
    ],
)
def test_unusable_class_names_are_refused_before_any_patch_is_written(
    tmp_path: Path, name: str, options: Dict[str, Any], message: str
) -> None:
    write_labels(tmp_path / "labels.geojson", named_features([name]))
    config = config_for(tmp_path, classification=options)
    with pytest.raises(ValueError, match=message):
        generate(config)
    assert not (config.writer.staging_dir / "images").exists()
    # With the settings that do not need the rule, the same labels are fine.
    ok = {"background": {"empty": "skip"}, "open water": {"mode": "single"}}
    if name in ok:
        generate(config_for(tmp_path, classification=ok[name], staging="ok"))


# ── the files ───────────────────────────────────────────────────────────────


def read_csv(path: Path) -> List[Dict[str, str]]:
    raw = path.read_bytes()
    assert b"\r" not in raw and raw.endswith(b"\n")
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def tree_bytes(root: Path) -> Dict[str, bytes]:
    return {
        str(path.relative_to(root)): path.read_bytes()
        for path in sorted(root.rglob("*"))
        if path.is_file()
    }


@pytest.fixture(scope="module")
def split_dataset(
    tmp_path_factory: pytest.TempPathFactory,
) -> Iterator[Tuple[MapcvConfig, Manifest, List[Feature]]]:
    root = tmp_path_factory.mktemp("split")
    features = random_features(seed=11, count=45)
    write_labels(root / "labels.geojson", features)
    config = config_for(
        root,
        classification={"mode": "multi", "min_fraction": 0.12, "empty": "background"},
        split={"strategy": "random", "test_ratio": 0.25, "val_ratio": 0.2},
        sampler={"edge_strategy": "pad", "stride": 48},
    )
    manifest = generate(config, FakeSource(make_valid_mask("noisy"), 14.0))
    yield config, manifest, features


def split_names(staging: Path) -> Dict[str, List[str]]:
    return {
        name: (staging / "splits" / f"{name}.txt").read_text(encoding="utf-8").split()
        for name in ("train", "val", "test")
    }


def test_the_label_files_parse_and_match_the_manifest(
    split_dataset: Tuple[MapcvConfig, Manifest, List[Feature]],
) -> None:
    config, manifest, _ = split_dataset
    staging = config.writer.staging_dir
    names = {0: "background", **{cid: name for name, cid in manifest.class_map.items()}}
    assert (staging / "classes.txt").read_text(encoding="utf-8") == (
        "background\nbuilding\ncar\ntree\n"
    )
    document = json.loads((staging / "labels.json").read_text(encoding="utf-8"))
    assert list(document) == ["classes", "images"]
    assert document["classes"] == ["background", "building", "car", "tree"]
    assert list(document["images"]) == [Path(e["files"]["image"]).name for e in manifest.patches]
    rows = read_csv(staging / "labels.csv")
    assert list(rows[0]) == ["image", "labels", "split"]
    splits = split_names(staging)
    where = {name: split for split, listed in splits.items() for name in listed}
    assert len(rows) == len(manifest.patches) > 40
    for entry, row in zip(manifest.patches, rows):
        image = Path(entry["files"]["image"]).name
        expected = [names[cid] for cid in entry["summary"]["labels"]]
        assert row == {"image": image, "labels": " ".join(expected), "split": where[image]}
        assert document["images"][image] == expected
        assert (staging / "images" / image).is_file()
    for split in ("train", "val", "test"):
        per_split = read_csv(staging / f"labels_{split}.csv")
        assert [row["image"] for row in per_split] == [
            row["image"] for row in rows if row["split"] == split
        ]
        assert {tuple(row) for row in per_split} == {("image", "labels")}
        assert sorted(row["image"] for row in per_split) == sorted(splits[split])
    assert (staging / "labels.csv").read_text().startswith("image,labels,split\n")
    # Multi-label sets in the data, background among them.
    assert any(" " in row["labels"] for row in rows)
    assert any(row["labels"] == "background" for row in rows)


def test_the_manifest_records_the_classification_target(
    split_dataset: Tuple[MapcvConfig, Manifest, List[Feature]],
) -> None:
    config, manifest, _ = split_dataset
    raw = json.loads((config.writer.staging_dir / "manifest.json").read_text(encoding="utf-8"))
    assert raw["version"] == 3 and raw["task"] == "classification"
    target = raw["target"]
    assert target["type"] == "classification"
    assert target["class_map"] == {"building": 1, "car": 2, "tree": 3}
    assert target["ignore_index"] == 255 and target["dtype"] is None
    assert target["options"] == {"mode": "multi", "min_fraction": 0.12, "empty": "background"}
    assert target["labels"]["label_field"] == "class" and len(target["labels"]["sha256"]) == 64
    assert raw["writer"]["layout"] == "files" and "mask_format" not in raw["writer"]
    for entry in raw["patches"]:
        assert set(entry["files"]) == {"image"} and entry["files"]["image"].startswith("images/")
        assert list(entry["summary"]) == ["class_coverage", "labels", "empty_ratio"]
        coverage = entry["summary"]["class_coverage"]
        assert list(coverage) == sorted(coverage, key=int)
        assert all(0 < share <= 1 for share in coverage.values())
        assert entry["summary"]["labels"] == sorted(entry["summary"]["labels"])
        assert sum(coverage.values()) <= 1 + 1e-12
    assert manifest.task == "classification"


def test_the_same_config_gives_the_same_bytes(tmp_path: Path) -> None:
    write_labels(tmp_path / "labels.geojson", random_features(seed=2))
    trees = []
    for staging in ("one", "two"):
        config = config_for(
            tmp_path,
            classification={"mode": "multi", "empty": "background", "min_fraction": 0.02},
            split={"strategy": "spatial"},
            staging=staging,
        )
        generate(config, FakeSource(make_valid_mask("noisy"), 21.0))
        trees.append(tree_bytes(config.writer.staging_dir))
    assert trees[0].keys() == trees[1].keys()
    assert "labels.csv" in trees[0] and "labels.json" in trees[0] and "classes.txt" in trees[0]
    for name in trees[0]:
        assert trees[0][name] == trees[1][name], name


def test_a_finished_run_resumes_as_a_no_op_and_an_interrupted_one_completes(tmp_path: Path) -> None:
    write_labels(tmp_path / "labels.geojson", random_features(seed=4))
    config = config_for(
        tmp_path,
        classification={"mode": "multi"},
        split={"strategy": "random", "seed": 3},
        sampler={"stride": 48},
    )
    source = FakeSource(make_valid_mask("edges"))
    full = generate(config, source)
    staging = config.writer.staging_dir
    before = tree_bytes(staging)
    again = generate(config, source)
    assert again.patches == full.patches
    assert tree_bytes(staging) == before

    # Lose the second half of the patches (and their label rows): the run catches up exactly.
    half = len(full.patches) // 2
    partial = full.model_copy(update={"patches": full.patches[:half]})
    partial.save(staging / "manifest.json")
    for name in ("labels.csv", "labels.json", "labels_train.csv"):
        (staging / name).unlink()
    for entry in full.patches[half:]:
        (staging / entry["files"]["image"]).unlink()
    generate(config, source)
    assert tree_bytes(staging) == before


def test_resume_refuses_changed_classification_options(tmp_path: Path) -> None:
    write_labels(tmp_path / "labels.geojson", random_features(seed=1, count=20))
    generate(config_for(tmp_path))
    for options in ({"mode": "multi"}, {"min_fraction": 0.1}, {"empty": "background"}):
        with pytest.raises(ManifestMismatchError, match="task options"):
            generate(config_for(tmp_path, classification=options))
    with pytest.raises(ManifestMismatchError, match="labels"):
        generate(config_for(tmp_path, labels={"all_touched": True}))
    segmentation = config_for(tmp_path).model_dump(mode="json", exclude_unset=True)
    segmentation["task"] = "segmentation"
    with pytest.raises(ManifestMismatchError, match="task"):
        generate(MapcvConfig.model_validate(segmentation))


def test_resplitting_rewrites_the_split_dependent_files_consistently(tmp_path: Path) -> None:
    write_labels(tmp_path / "labels.geojson", random_features(seed=6))
    config = config_for(
        tmp_path,
        classification={"mode": "multi", "empty": "background"},
        sampler={"stride": 40},
    )
    manifest = generate(config, FakeSource(make_valid_mask("edges")))
    staging = config.writer.staging_dir
    # Without a split: no split column and no per-split files.
    assert list(read_csv(staging / "labels.csv")[0]) == ["image", "labels"]
    assert not list(staging.glob("labels_*.csv"))
    images_before = tree_bytes(staging / "images")
    labels_json = (staging / "labels.json").read_bytes()

    # The default spatial split leaves overlapping patches out of train and val.
    counts = run_split(staging, SplitterConfig(strategy="spatial", seed=1))
    assert counts["dropped"] > 0
    assert_split_files_consistent(staging, manifest)
    rows = read_csv(staging / "labels.csv")
    assert sum(1 for row in rows if row["split"] == "") == counts["dropped"]
    first = {row["image"]: row["split"] for row in rows}

    counts = run_split(staging, SplitterConfig(strategy="random", seed=9, test_ratio=0.4))
    assert counts["test"] > 0
    assert_split_files_consistent(staging, manifest)
    assert {row["image"]: row["split"] for row in read_csv(staging / "labels.csv")} != first
    # Nothing else moves: not the images, the manifest or labels.json.
    assert tree_bytes(staging / "images") == images_before
    assert (staging / "labels.json").read_bytes() == labels_json
    assert Manifest.load(staging / "manifest.json").patches == manifest.patches

    # A writer told there is no split removes the per-split files again.
    ClassificationWriter.from_manifest(manifest, staging).write_annotations(manifest, None)
    assert not list(staging.glob("labels_*.csv"))
    assert list(read_csv(staging / "labels.csv")[0]) == ["image", "labels"]


def assert_split_files_consistent(staging: Path, manifest: Manifest) -> None:
    names = split_names(staging)
    rows = read_csv(staging / "labels.csv")
    assert [row["image"] for row in rows] == [
        Path(e["files"]["image"]).name for e in manifest.patches
    ]
    for split, listed in names.items():
        assert sorted(row["image"] for row in rows if row["split"] == split) == sorted(listed)
        per_split = read_csv(staging / f"labels_{split}.csv")
        assert sorted(row["image"] for row in per_split) == sorted(listed)
        by_image = {row["image"]: row["labels"] for row in rows}
        assert all(by_image[row["image"]] == row["labels"] for row in per_split)


def test_the_cli_split_command_updates_the_label_tables(tmp_path: Path) -> None:
    write_labels(tmp_path / "labels.geojson", random_features(seed=6))
    config = config_for(tmp_path, classification={"mode": "multi"}, sampler={"stride": 64})
    manifest = generate(config, FakeSource(make_valid_mask("full")))
    staging = config.writer.staging_dir
    result = runner.invoke(app, ["split", str(staging), "--strategy", "random", "--seed", "4"])
    assert result.exit_code == 0, result.output
    assert_split_files_consistent(staging, manifest)


def test_class_coverage_in_the_summary_is_the_reference_for_every_kept_patch(
    split_dataset: Tuple[MapcvConfig, Manifest, List[Feature]],
) -> None:
    _, manifest, features = split_dataset
    source = FakeSource(make_valid_mask("noisy"), 14.0)
    kept, skipped = assert_dataset_equals_reference(
        manifest,
        features,
        source,
        mode="multi",
        min_fraction=0.12,
        empty="background",
        edge="pad",
        stride=48,
    )
    assert kept == len(manifest.patches)
    assert skipped >= 0


# ── stratified splits and `mapcv info` ──────────────────────────────────────


def test_strata_come_from_the_assigned_labels() -> None:
    def entry(labels: List[int], coverage: Dict[str, float]) -> Any:
        return {
            "summary": {"labels": labels, "class_coverage": coverage, "empty_ratio": 0.0},
        }

    assert _stratum(entry([], {})) == (0, "")
    assert _stratum(entry([0], {})) == (0, "")
    assert _stratum(entry([0], {"2": 0.01})) == (0, "")
    assert _stratum(entry([2], {"2": 0.4, "3": 0.5})) == (1, "2")
    # Several labels: the one with the largest coverage; ties go to the lowest ID.
    assert _stratum(entry([1, 3, 12], {"1": 0.1, "3": 0.3, "12": 0.5})) == (1, "12")
    assert _stratum(entry([2, 10], {"2": 0.25, "10": 0.25})) == (1, "2")


def test_stratified_splits_balance_the_classes(tmp_path: Path) -> None:
    write_labels(tmp_path / "labels.geojson", random_features(seed=5, count=120))
    config = config_for(
        tmp_path,
        classification={"empty": "background", "min_fraction": 0.05},
        split={"strategy": "stratified", "test_ratio": 0.3, "val_ratio": 0.0},
        sampler={"stride": 32},
    )
    manifest = generate(config, FakeSource(make_valid_mask("edges")))
    test = set(split_names(config.writer.staging_dir)["test"])
    strata: Dict[Any, List[bool]] = {}
    for entry in manifest.patches:
        strata.setdefault(_stratum(entry), []).append(manifest.patch_name(entry) in test)
    assert {leading for _, leading in strata} == {"1", "2", "3"}
    for members in strata.values():
        assert sum(members) == math.ceil(len(members) * 0.3)


def test_info_shows_the_patches_per_label(
    split_dataset: Tuple[MapcvConfig, Manifest, List[Feature]],
) -> None:
    config, manifest, _ = split_dataset
    result = runner.invoke(app, ["info", str(config.writer.staging_dir)])
    assert result.exit_code == 0, result.output
    assert "classification" in result.output and "version 3" in result.output
    assert "do not count towards coverage" in result.output
    assert "label" in result.output and "patches" in result.output
    flat = {}
    for line in result.output.splitlines():
        parts = line.split()
        if len(parts) == 4 and parts[0] in {"background", "building", "car", "tree"}:
            flat[parts[0]] = (parts[1], parts[2], parts[3])
    assert set(flat) == {"background", "building", "car", "tree"}
    total = len(manifest.patches)
    ids = {"background": 0, "building": 1, "car": 2, "tree": 3}
    for name, (cid, count, share) in flat.items():
        wanted = sum(1 for e in manifest.patches if ids[name] in e["summary"]["labels"])
        assert int(cid) == ids[name] and int(count.replace(",", "")) == wanted
        assert share == f"{wanted / total:.1%}"


# ── label rasters (labels.type: raster) ─────────────────────────────────────

IMAGERY_EPSG = 32631
SENTINEL = -999_999  # reference value of pixels outside the label raster
NODATA = 250
RASTER_CLASSES = {0: 0, 10: 1, 20: 2, 30: 3, 40: 3}
RASTER_NAMES = {1: "tree_cover", 2: "built_up", 3: "water"}
VALUES = (0, 10, 20, 30, 40, 99, NODATA)
# rasterio 1.5 honours ``reproject(..., tolerance=0)``; older versions always approximate
# the transformation between two CRSs (up to 0.125 pixel), which is no exact reference.
EXACT_WARP = tuple(int(part) for part in rasterio.__version__.split(".")[:2]) >= (1, 5)


def utm(lon: float, lat: float) -> Tuple[float, float]:
    x, y = Transformer.from_crs("EPSG:4326", f"EPSG:{IMAGERY_EPSG}", always_xy=True).transform(
        lon, lat
    )
    return float(x), float(y)


@dataclass
class Scene:
    path: Path
    transform: Affine
    width: int
    height: int
    data: npt.NDArray[np.uint8]

    def region(self, margin: float = 0.04) -> Dict[str, float]:
        corners = [self.transform * (c, r) for c in (0, self.width) for r in (0, self.height)]
        xs, ys = zip(*corners)
        west, south, east, north = rasterio.warp.transform_bounds(
            CRS.from_epsg(IMAGERY_EPSG), CRS.from_epsg(4326), min(xs), min(ys), max(xs), max(ys)
        )
        dx, dy = (east - west) * margin, (north - south) * margin
        return {"west": west + dx, "south": south + dy, "east": east - dx, "north": north - dy}

    def has_imagery(
        self, x: npt.NDArray[np.float64], y: npt.NDArray[np.float64]
    ) -> npt.NDArray[np.bool_]:
        """Whether the file has data (not NoData) at world coordinates."""
        cols, rows = ~self.transform * (x, y)
        col, row = np.floor(cols).astype(int), np.floor(rows).astype(int)
        inside = (col >= 0) & (col < self.width) & (row >= 0) & (row < self.height)
        found = np.zeros(x.shape, dtype=bool)
        found[inside] = np.any(self.data[:, row[inside], col[inside]] != 0, axis=0)
        return found


def make_scene(
    directory: Path,
    *,
    width: int = 320,
    height: int = 288,
    rotation: float = 0.0,
    nodata_block: Optional[Tuple[int, int, int, int]] = (60, 120, 150, 230),
) -> Scene:
    """A 3-band uint8 GeoTIFF in UTM 31N (NoData 0, in a block and along an edge)."""
    cx, cy = utm(3.0, 48.85)
    transform = Affine(1.0, 0.0, cx - width / 2, 0.0, -1.0, cy + height / 2)
    if rotation:
        transform = transform * Affine.rotation(rotation, (width / 2, height / 2))
    data = np.random.default_rng(3).integers(1, 250, size=(3, height, width), dtype=np.uint8)
    if nodata_block is not None:
        r0, r1, c0, c1 = nodata_block
        data[:, r0:r1, c0:c1] = 0
        data[:, :, :7] = 0
    path = directory / "scene.tif"
    with rasterio.open(
        path,
        "w",
        driver="GTiff",
        height=height,
        width=width,
        count=3,
        dtype="uint8",
        crs=CRS.from_epsg(IMAGERY_EPSG),
        transform=transform,
        nodata=0,
        tiled=True,
        blockxsize=128,
        blockysize=128,
    ) as dst:
        dst.write(data)
    return Scene(path, transform, width, height, data)


def make_label_raster(
    directory: Path,
    scene: Scene,
    epsg: int,
    pixel: float,
    *,
    cover: float = 0.8,
    offset: Tuple[float, float] = (0.0, 0.0),
    blobs: int = 6,
) -> Path:
    """A north-up label raster of blobs (so patches get several classes) over the middle
    ``cover`` of the scene, with NoData and unmapped values mixed in."""
    corners = [scene.transform * (c, r) for c in (0, scene.width) for r in (0, scene.height)]
    to_label = Transformer.from_crs(f"EPSG:{IMAGERY_EPSG}", f"EPSG:{epsg}", always_xy=True)
    xs, ys = to_label.transform([x for x, _ in corners], [y for _, y in corners])
    west, east, south, north = min(xs), max(xs), min(ys), max(ys)
    span_x, span_y = (east - west) * cover, (north - south) * cover
    width, height = int(round(span_x / pixel)), int(round(span_y / pixel))
    left = (west + east) / 2 - span_x / 2 + offset[0] * pixel
    top = (south + north) / 2 + span_y / 2 - offset[1] * pixel
    rng = np.random.default_rng(11)
    data = np.zeros((height, width), dtype=np.uint8)
    rows, cols = np.mgrid[0:height, 0:width]
    for _ in range(blobs * 4):
        cy, cx = rng.integers(0, height), rng.integers(0, width)
        radius = rng.integers(max(3, min(height, width) // 25), min(height, width) // 5)
        data[(rows - cy) ** 2 + (cols - cx) ** 2 <= radius**2] = rng.choice(VALUES[1:])
    path = directory / f"labels_{epsg}.tif"
    with rasterio.open(
        path,
        "w",
        driver="GTiff",
        height=height,
        width=width,
        count=1,
        dtype="uint8",
        crs=CRS.from_epsg(epsg),
        transform=Affine(pixel, 0.0, left, 0.0, -pixel, top),
        nodata=NODATA,
        tiled=True,
        blockxsize=64,
        blockysize=64,
        compress="deflate",
    ) as dst:
        dst.write(data, 1)
    return path


def raster_config(
    root: Path,
    scene: Scene,
    label_path: Path,
    *,
    classification: Optional[Dict[str, Any]] = None,
    labels: Optional[Dict[str, Any]] = None,
    sampler: Optional[Dict[str, Any]] = None,
    split: Optional[Dict[str, Any]] = None,
    staging: str = "dataset",
) -> MapcvConfig:
    classes = {
        value: ({"id": cid, "name": RASTER_NAMES[cid]} if cid else {"id": 0})
        for value, cid in RASTER_CLASSES.items()
    }
    data: Dict[str, Any] = {
        "task": "classification",
        "region": scene.region(),
        "imagery": {"type": "geotiff", "path": str(scene.path)},
        "labels": {"type": "raster", "path": str(label_path), "classes": classes, **(labels or {})},
        "sampler": {"patch_size": 48, "edge_strategy": "drop", **(sampler or {})},
        "writer": {"staging_dir": str(root / staging), "image_format": "png"},
    }
    if classification is not None:
        data["classification"] = classification
    if split is not None:
        data["split"] = split
    return MapcvConfig.model_validate(data)


def raster_counts(
    scene: Scene,
    label_path: Path,
    patch: Affine,
    size: int,
    *,
    ignore: Optional[int] = IGNORE,
    unmapped: str = "background",
) -> Counts:
    """Coverage by GDAL's nearest-neighbour warp of the label raster onto the patch grid."""
    with rasterio.open(label_path) as src:
        exact: Dict[str, Any] = {"tolerance": 0} if EXACT_WARP else {}
        if not EXACT_WARP and src.crs != CRS.from_epsg(IMAGERY_EPSG):
            pytest.skip("rasterio < 1.5 cannot reproject without approximating (tolerance)")
        raw = src.read(1).astype(np.int32)
        warped = np.full((size, size), SENTINEL, dtype=np.int32)
        rasterio.warp.reproject(
            raw,
            warped,
            src_transform=src.transform,
            src_crs=src.crs,
            src_nodata=None,
            dst_transform=patch,
            dst_crs=CRS.from_epsg(IMAGERY_EPSG),
            dst_nodata=SENTINEL,
            resampling=Resampling.nearest,
            **exact,
        )
    cols, rows = np.meshgrid(np.arange(size) + 0.5, np.arange(size) + 0.5)
    x, y = patch * (cols, rows)
    has_imagery = scene.has_imagery(np.asarray(x), np.asarray(y))
    outside = (warped == SENTINEL) | (warped == NODATA)
    unknown = ~np.isin(warped, list(RASTER_CLASSES)) & ~outside
    ignored = outside | (unknown & (unmapped == "ignore"))
    mapped = np.zeros(warped.shape, dtype=np.int64)
    for value, cid in RASTER_CLASSES.items():
        mapped[warped == value] = cid
    valid = has_imagery if ignore is None else has_imagery & ~ignored
    if ignore is None:
        ignored = np.zeros_like(ignored)
    classes = mapped[valid & ~ignored]
    values, numbers = np.unique(classes, return_counts=True)
    return Counts(
        {int(v): int(n) for v, n in zip(values, numbers) if v != 0}, int(np.count_nonzero(valid))
    )


# Label raster grids: the imagery's CRS and a finer grid, and a coarser one in another CRS.
RASTER_CASES = [
    ("same-crs-finer", IMAGERY_EPSG, 0.7, 0.0, {}, {}),
    ("same-crs-rotated-multi", IMAGERY_EPSG, 1.3, 15.0, {"mode": "multi"}, {}),
    (
        "lonlat-rotated-background",
        4326,
        1.2e-5,
        -25.0,
        {"mode": "multi", "min_fraction": 0.1, "empty": "background"},
        {},
    ),
    ("lonlat-min-fraction", 4326, 2.5e-5, 0.0, {"min_fraction": 0.3}, {}),
    ("unmapped-ignored", IMAGERY_EPSG, 2.1, 7.0, {"mode": "multi"}, {"unmapped": "ignore"}),
    (
        "no-ignore-index",
        IMAGERY_EPSG,
        0.9,
        0.0,
        {"mode": "multi", "min_fraction": 0.05},
        {"ignore_index": None},
    ),
]


@pytest.mark.parametrize(
    ("epsg", "pixel", "rotation", "options", "label_settings"),
    [case[1:] for case in RASTER_CASES],
    ids=[case[0] for case in RASTER_CASES],
)
def test_raster_label_coverage_equals_the_gdal_warp_reference(
    tmp_path: Path,
    epsg: int,
    pixel: float,
    rotation: float,
    options: Dict[str, Any],
    label_settings: Dict[str, Any],
) -> None:
    scene = make_scene(tmp_path, rotation=rotation)
    label_path = make_label_raster(tmp_path, scene, epsg, pixel)
    config = raster_config(
        tmp_path, scene, label_path, classification=options, labels=label_settings
    )
    run_generate(config)
    manifest = Manifest.load(config.writer.staging_dir / "manifest.json")
    ignore = label_settings.get("ignore_index", IGNORE)
    mode = options.get("mode", "single")
    min_fraction = options.get("min_fraction", 0.0)
    empty = options.get("empty", "skip")
    source = manifest.source
    assert source.transform is not None
    base = Affine(*source.transform)
    size = 48
    kept = {(e["row"], e["col"]): e for e in manifest.patches}
    height, width = _source_size(config)
    skipped = checked = 0
    for row in range(0, height - size + 1, size):
        for col in range(0, width - size + 1, size):
            patch = base * Affine.translation(col, row)
            counts = raster_counts(
                scene,
                label_path,
                patch,
                size,
                ignore=ignore,
                unmapped=label_settings.get("unmapped", "background"),
            )
            expected = rule(counts, mode, min_fraction, empty) if counts.valid else None
            entry = kept.get((row, col))
            if expected is None:
                assert entry is None, f"({row}, {col}) should be skipped"
                skipped += 1
                continue
            assert entry is not None, f"({row}, {col}) missing, expected {expected}"
            summary = entry["summary"]
            assert summary["labels"] == list(expected)
            reference = {str(c): n / counts.valid for c, n in sorted(counts.by_class.items())}
            assert list(summary["class_coverage"]) == list(reference), (row, col)
            for cid, share in reference.items():
                assert summary["class_coverage"][cid] == pytest.approx(share, abs=1e-15, rel=0)
            checked += 1
    assert checked >= 6 and len(kept) == checked
    assert manifest.class_map == {name: cid for cid, name in RASTER_NAMES.items()}


def _source_size(config: MapcvConfig) -> Tuple[int, int]:
    """Height and width of the raster mapcv reads (the region's window of the file)."""
    from mapcv.imagery import open_raster_source

    source = open_raster_source(config.region, config.primary_imagery)
    try:
        return source.metadata.height, source.metadata.width
    finally:
        source.close()


def test_raster_labels_with_the_sample_classes_are_named_and_sorted(tmp_path: Path) -> None:
    scene = make_scene(tmp_path)
    label_path = make_label_raster(tmp_path, scene, IMAGERY_EPSG, 1.0)
    config = raster_config(
        tmp_path,
        scene,
        label_path,
        classification={"mode": "multi", "empty": "background"},
        split={"strategy": "random"},
    )
    run_generate(config)
    staging = config.writer.staging_dir
    assert (staging / "classes.txt").read_text() == "background\ntree_cover\nbuilt_up\nwater\n"
    manifest = Manifest.load(staging / "manifest.json")
    assert manifest.target is not None and manifest.target.labels is not None
    assert manifest.target.labels["type"] == "raster"
    assert "fingerprint" in manifest.target.labels
    rows = read_csv(staging / "labels.csv")
    assert len(rows) == len(manifest.patches)
    assert {label for row in rows for label in row["labels"].split()} <= {
        "background",
        "tree_cover",
        "built_up",
        "water",
    }
    result = runner.invoke(app, ["info", str(staging)])
    assert result.exit_code == 0 and "tree_cover" in result.output


# ── configuration ───────────────────────────────────────────────────────────


def _raw(tmp_path: Path, **changes: Any) -> Dict[str, Any]:
    data: Dict[str, Any] = {
        "task": "classification",
        "region": {"west": 4.93, "south": 52.37, "east": 4.95, "north": 52.38},
        "imagery": {"type": "xyz", "zoom": 18, "source": "esri_satellite"},
        "labels": {"path": str(tmp_path / "labels.geojson"), "label_field": "class"},
        "sampler": {"patch_size": 64},
        "writer": {"staging_dir": str(tmp_path / "dataset")},
    }
    data.update(changes)
    return data


def test_classification_defaults(tmp_path: Path) -> None:
    config = MapcvConfig.model_validate(_raw(tmp_path))
    assert config.task == "classification" and config.classification is None
    assert config.classification_options == ClassificationOptions(
        mode="single", min_fraction=0.0, empty="skip"
    )


def test_classification_is_a_supported_task(tmp_path: Path) -> None:
    for task in ("regression",):
        with pytest.raises(ValidationError) as caught:
            MapcvConfig.model_validate(_raw(tmp_path, task=task))
        assert f"task '{task}' is not supported yet" in str(caught.value)
        supported = "supported: segmentation, detection, instance, classification, change"
        assert supported in str(caught.value)
        assert "planned: regression" in str(caught.value)


@pytest.mark.parametrize(
    ("changes", "message"),
    [
        ({"labels": None}, "task: classification needs labels"),
        ({"classification": {"mode": "several"}}, "mode"),
        ({"classification": {"min_fraction": 1.5}}, "min_fraction"),
        ({"classification": {"min_fraction": -0.1}}, "min_fraction"),
        ({"classification": {"empty": "drop"}}, "empty"),
        ({"classification": {"rule": "majority"}}, "rule"),
        ({"task": "segmentation", "classification": {}}, "only applies to task: classification"),
        ({"task": "detection", "classification": {}}, "only applies to task: classification"),
        ({"instance": {}}, "only applies to task: instance"),
        ({"detection": {}}, "only applies to task: detection"),
        ({"sampler": {"patch_size": 64, "pad_mode": "reflect"}}, "mirrors imagery"),
        ({"writer": {"staging_dir": "out", "mask_format": "png"}}, "writes no masks"),
        (
            {"writer": {"staging_dir": "out", "image_format": "tif", "world_files": True}},
            "world_files",
        ),
        (
            {
                "classification": {"empty": "background"},
                "sampler": {"patch_size": 64, "min_label_ratio": 0.1},
            },
            "min_label_ratio",
        ),
        (
            {
                "classification": {"mode": "multi"},
                "labels": {
                    "type": "raster",
                    "path": "x.tif",
                    "classes": {"1": {"id": 1, "name": "open water"}},
                },
            },
            "contains whitespace",
        ),
        (
            {
                "classification": {"empty": "background"},
                "labels": {
                    "type": "raster",
                    "path": "x.tif",
                    "classes": {"1": {"id": 1, "name": "background"}},
                },
            },
            "label of patches without a class",
        ),
    ],
)
def test_invalid_classification_configs_fail_clearly(
    tmp_path: Path, changes: Dict[str, Any], message: str
) -> None:
    with pytest.raises(ValidationError, match=message):
        MapcvConfig.model_validate(_raw(tmp_path, **changes))


def test_valid_classification_combinations(tmp_path: Path) -> None:
    sampler = {"patch_size": 64, "pad_mode": "reflect", "edge_strategy": "shift"}
    config = MapcvConfig.model_validate(
        _raw(
            tmp_path,
            classification={"mode": "multi", "min_fraction": 0.2, "empty": "background"},
            sampler=sampler,
        )
    )
    assert config.classification_options.mode == "multi"
    raster = MapcvConfig.model_validate(
        _raw(
            tmp_path,
            labels={
                "type": "raster",
                "path": "x.tif",
                "classes": {"1": {"id": 1, "name": "water"}},
                "ignore_index": None,
            },
            writer={"staging_dir": "out", "image_format": "jpg"},
            sampler={"patch_size": 64, "min_label_ratio": 0.2},
        )
    )
    assert raster.task == "classification"
    # Settings that keep the segmentation-only options out of the way.
    MapcvConfig.model_validate(
        _raw(tmp_path, labels={"path": "x.geojson", "all_touched": True, "ignore_index": 7})
    )


def test_factories_pick_the_classification_target_and_writer(tmp_path: Path) -> None:
    config = MapcvConfig.model_validate(_raw(tmp_path))
    target = create_target(config)
    assert isinstance(target, ClassificationTarget) and target.type == "classification"
    writer = create_writer(config.writer, target)
    assert isinstance(writer, ClassificationWriter)
    check_compatible(target, writer)
    with pytest.raises(ValueError, match="cannot write classification targets"):
        check_compatible(target, FilesWriter(config.writer))
    assert writer.fingerprint() == {
        "layout": "files",
        "image_format": "png",
        "jpg_quality": 95,
        "jpg_subsampling": "4:2:0",
    }


# ── planning, templates, the wizard and the CLI end to end ──────────────────


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


def test_the_cli_builds_a_classification_dataset_from_a_tile_server(
    tmp_path: Path, tile_server: str
) -> None:
    rng = random.Random(8)
    features = []
    for index in range(40):
        x = TILE_X0 + rng.uniform(-0.1, TILES_X + 0.1)
        y = TILE_Y0 + rng.uniform(-0.1, TILES_Y + 0.1)
        size = rng.uniform(0.05, 0.5)
        ring = [(x, y), (x + size, y), (x + size * 0.7, y + size), (x, y + size * 0.8), (x, y)]
        features.append(
            {
                "type": "Feature",
                "properties": {"kind": ("shed", "house", "pond")[index % 3]},
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
task: classification
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
classification:
  mode: multi
  min_fraction: 0.01
  empty: background
sampler:
  patch_size: 128
writer:
  staging_dir: dataset
split:
  strategy: spatial
  test_ratio: 0.2
  val_ratio: 0.2
""",
        encoding="utf-8",
    )
    config = str(tmp_path / "mapcv.yaml")
    result = runner.invoke(app, ["validate", config])
    assert result.exit_code == 0, result.output
    assert "classification · multi-label · min_fraction 0.01 · empty: background" in result.output
    result = runner.invoke(app, ["plan", config])
    assert result.exit_code == 0, result.output
    assert "classification" in result.output and "polygon(s)" in result.output

    result = runner.invoke(app, ["generate", config, "--yes"])
    assert result.exit_code == 0, result.output
    for needle in ("labels.csv", "labels.json", "classes.txt", "tutorials/classification"):
        assert needle in result.output
    staging = tmp_path / "dataset"
    manifest = Manifest.load(staging / "manifest.json")
    assert manifest.task == "classification" and manifest.class_map == {
        "house": 1,
        "pond": 2,
        "shed": 3,
    }
    assert len(manifest.patches) == 8 * 6  # 1,024 x 768 px in 128 px patches
    rows = read_csv(staging / "labels.csv")
    assert len(rows) == 48 and {row["split"] for row in rows} <= {"train", "val", "test", ""}
    assert (staging / "classes.txt").read_text() == "background\nhouse\npond\nshed\n"
    before = tree_bytes(staging)

    result = runner.invoke(app, ["info", str(staging)])
    assert result.exit_code == 0, result.output
    assert "classification" in result.output and "house" in result.output
    result = runner.invoke(app, ["generate", config, "--yes"])
    assert result.exit_code == 0 and "Nothing left to do" in result.output
    assert tree_bytes(staging) == before


def test_planning_counts_classification_label_files(tmp_path: Path) -> None:
    write_labels(tmp_path / "labels.geojson", random_features(seed=3, count=10))
    estimate = plan(config_for(tmp_path))
    assert estimate.task == "classification" and estimate.objects is None
    segmentation = config_for(tmp_path).model_dump(mode="json", exclude_unset=True)
    segmentation.pop("classification", None)
    segmentation["task"] = "segmentation"
    reference = plan(MapcvConfig.model_validate(segmentation))
    # Image files only (no masks), plus a few bytes per patch for the label tables.
    assert estimate.patches == reference.patches > 0
    assert estimate.output_bytes >= estimate.patches * _CLASSIFICATION_BYTES
    assert abs(estimate.output_bytes - reference.output_bytes) < estimate.patches * 1000
    # A label file that misses the region is reported in the words of this task.
    write_labels(
        tmp_path / "labels.geojson", [Feature(box(X0 + 1e6, Y0, X0 + 1e6 + 100, Y0 + 100), "tree")]
    )
    assert any(
        "no patch would get a label" in message for message in plan(config_for(tmp_path)).warnings
    )


def test_init_template_and_wizard_offer_classification(tmp_path: Path) -> None:
    written = tmp_path / "template.yaml"
    result = runner.invoke(app, ["init", str(written), "--template", "classification"])
    assert result.exit_code == 0, result.output
    config = MapcvConfig.from_yaml(written)
    assert (
        config.task == "classification" and config.classification_options == ClassificationOptions()
    )
    assert yaml.safe_load(written.read_text())["classification"]["mode"] == "single"

    labels = tmp_path / "aoi.geojson"
    labels.write_text(
        '{"type":"FeatureCollection","features":[{"type":"Feature","properties":{"kind":"roof"},'
        '"geometry":{"type":"Polygon","coordinates":[[[74.3,31.5],[74.31,31.5],'
        "[74.31,31.51],[74.3,31.5]]]}}]}"
    )
    out = tmp_path / "mapcv.yaml"
    answers = ["esri", str(labels), "17", "y", "kind", "classification", "multi", "64", "./ds", "y"]
    result = runner.invoke(
        app, ["init", str(out), "--interactive"], input="\n".join(answers) + "\n"
    )
    assert result.exit_code == 0, result.output
    config = MapcvConfig.from_yaml(out)
    assert config.task == "classification"
    assert config.classification_options == ClassificationOptions(mode="multi")
    assert config.sampler.patch_size == 64


def test_a_label_file_without_polygons_labels_every_patch_background_or_nothing(
    tmp_path: Path,
) -> None:
    point = {
        "type": "Feature",
        "properties": {"class": "tree"},
        "geometry": {"type": "Point", "coordinates": [4.94, 52.375]},
    }
    (tmp_path / "labels.geojson").write_text(
        json.dumps({"type": "FeatureCollection", "features": [point]}), encoding="utf-8"
    )
    source = FakeSource(make_valid_mask("edges"))
    with pytest.warns(UserWarning):
        nothing = generate(config_for(tmp_path), source)
    assert nothing.patches == []
    assert (tmp_path / "dataset" / "labels.csv").read_text() == "image,labels\n"
    assert (tmp_path / "dataset" / "labels.json").read_text() == (
        '{\n  "classes": ["object"],\n  "images": {}\n}\n'
    )
    with pytest.warns(UserWarning):
        kept = generate(
            config_for(tmp_path, classification={"empty": "background"}, staging="bg"), source
        )
    assert {tuple(e["summary"]["labels"]) for e in kept.patches} == {(0,)}
    assert kept.patches and all(e["summary"]["class_coverage"] == {} for e in kept.patches)
    # Nothing labeled anywhere: `info` still works, without a table of labels.
    result = runner.invoke(app, ["info", str(tmp_path / "dataset")])
    assert result.exit_code == 0, result.output
    assert "classification" in result.output and "background" not in result.output


# ── the writer and the factories on their own ───────────────────────────────


def test_the_writer_refuses_annotations_that_are_not_patch_labels(tmp_path: Path) -> None:
    config = config_for(tmp_path)
    writer = ClassificationWriter(config.writer, ClassificationOptions())
    manifest = Manifest(task="classification")
    images = np.zeros((1, 8, 8, 3), dtype=np.uint8)
    meta = [PatchMeta(row=0, col=0, padded=False, empty_ratio=0.0)]
    with pytest.raises(TypeError, match="one PatchLabels per patch"):
        writer.write(images, [], meta, manifest, 0)
    with pytest.raises(TypeError, match="PatchLabels expected"):
        writer.write(images, [None], meta, manifest, 0)  # type: ignore[list-item]
    writer.write(images[:0], [], [], manifest, 0)  # nothing to write is fine
    assert manifest.patches == [] and not (tmp_path / "dataset" / "images").exists()


def test_the_label_tables_need_the_labels_in_the_manifest(tmp_path: Path) -> None:
    config = config_for(tmp_path)
    writer = ClassificationWriter(config.writer, ClassificationOptions())
    entry = ManifestEntry(
        row=0,
        col=0,
        padded=False,
        chunk=0,
        files={"image": "images/patch_0000000.png"},
        summary=PatchSummary(empty_ratio=0.0),
    )
    manifest = Manifest(task="classification", patches=[entry])
    with pytest.raises(ValueError, match="patch_0000000.png has no labels in the manifest"):
        writer.write_annotations(manifest, None)


def test_the_target_factory_checks_the_labels_it_gets(tmp_path: Path) -> None:
    config = MapcvConfig.model_validate(_raw(tmp_path))
    with pytest.raises(ValueError, match="task: classification needs labels"):
        create_target(config.model_copy(update={"labels": None}))
    target = create_target(config)
    with pytest.raises(RuntimeError, match="prepare"):
        target.record()


def test_writer_options_footprints_and_world_files(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    write_labels(tmp_path / "labels.geojson", random_features(seed=8, count=30))
    source = FakeSource(make_valid_mask("full"))
    plain = config_for(tmp_path, writer={"footprints": False}, staging="plain")
    generate(plain, source)
    assert not (plain.writer.staging_dir / "patches.geojson").exists()
    assert (plain.writer.staging_dir / "labels.csv").exists()

    worlds = config_for(tmp_path, writer={"world_files": True}, staging="worlds")
    manifest = generate(worlds, source)
    assert manifest.writer is not None and manifest.writer["world_files"] is True
    assert "mask_format" not in manifest.writer
    assert (worlds.writer.staging_dir / "images" / "patch_0000000.pgw").is_file()
    assert (worlds.writer.staging_dir / "patches.geojson").is_file()

    def refuse(*args: object, **kwargs: object) -> None:
        raise RuntimeError("no pyproj")

    monkeypatch.setattr("mapcv.writers.classification.write_footprints", refuse)
    with pytest.warns(UserWarning, match="patches.geojson was not written: no pyproj"):
        generate(config_for(tmp_path, staging="nofootprints"), source)
    # The label tables do not depend on it.
    assert (tmp_path / "nofootprints" / "labels.json").is_file()
