"""Correctness checks on a generated dataset, independent of mapcv's own code paths.

* images are compared pixel by pixel with what the synthetic tile server served
  (exact for PNG, a small tolerance for JPEG output);
* masks are compared with ``rasterio.features.rasterize`` on labels reprojected
  with ``pyproj`` (both optional, the mask check is skipped without them);
* the manifest is cross-checked against the files on disk;
* train/val/test lists are disjoint, complete, and no train or validation patch
  shares a pixel with a held-out one.

Every violated expectation becomes a string in ``problems``; the harness fails
the run if there is any.
"""

from __future__ import annotations

import hashlib
import json
import math
import random
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple

import numpy as np
import numpy.typing as npt
from PIL import Image
from shapely import STRtree
from shapely.geometry import Polygon, box

from benchmarks.scenarios import CLASSES, PATCH, X0, Y0, ZOOM, Scenario
from benchmarks.tileserver import TILE, decoded_tile, is_failing

try:
    import rasterio.features
    from affine import Affine
    from pyproj import Transformer

    HAVE_MASK_REFERENCE = True
except ImportError:
    HAVE_MASK_REFERENCE = False

MASK_REFERENCE_HINT = (
    "mask check skipped: it needs rasterio and pyproj (uv sync --group bench, "
    "or pip install rasterio pyproj psutil)"
)
ORIGIN = 20037508.342789244  # Web Mercator half-extent in metres
IGNORE_INDEX = 255  # labels.ignore_index default: written where there is no imagery
# JPEG output is lossy: allowed mean absolute difference per patch / over all patches.
JPEG_PATCH_TOLERANCE = 6.0
JPEG_MEAN_TOLERANCE = 3.0
# Rasterization may differ from rasterio on pixels that straddle a polygon edge.
MASK_DISAGREE_LIMIT_PCT = 0.01
SPLIT_NAMES = ("train", "val", "test")


@dataclass
class CheckReport:
    """Outcome of :func:`check_dataset`."""

    stats: Dict[str, Any] = field(default_factory=dict)
    problems: List[str] = field(default_factory=list)
    skipped: List[str] = field(default_factory=list)


def tree_hash(dataset: Path) -> str:
    """One digest over every image, mask, the manifest and the split lists.

    The manifest is hashed as parsed JSON with sorted keys, so only its content
    counts, not its formatting or key order.
    """
    digest = hashlib.sha256()
    manifest = json.loads((dataset / "manifest.json").read_text(encoding="utf-8"))
    digest.update(json.dumps(manifest, sort_keys=True).encode())
    files: List[Path] = []
    for sub in ("Images", "Masks", "splits"):
        files.extend(sorted(path for path in (dataset / sub).rglob("*") if path.is_file()))
    for path in files:
        digest.update(path.relative_to(dataset).as_posix().encode())
        digest.update(hashlib.sha256(path.read_bytes()).digest())
    return digest.hexdigest()


def expected_image(row: int, col: int, scenario: Scenario) -> npt.NDArray[np.uint8]:
    """The patch at raster (row, col) as the server's tiles define it.

    Failed tiles are black, which is what mapcv fills them with.
    """
    pixels = np.zeros((PATCH, PATCH, 3), dtype=np.uint8)
    for tile_y in range(row // TILE, (row + PATCH - 1) // TILE + 1):
        for tile_x in range(col // TILE, (col + PATCH - 1) // TILE + 1):
            if is_failing(X0 + tile_x, Y0 + tile_y, scenario.fail_every):
                continue
            tile = decoded_tile(X0 + tile_x, Y0 + tile_y, scenario.tile_format)
            top, left = max(row - tile_y * TILE, 0), max(col - tile_x * TILE, 0)
            bottom = min(row + PATCH - tile_y * TILE, TILE)
            right = min(col + PATCH - tile_x * TILE, TILE)
            pixels[
                tile_y * TILE + top - row : tile_y * TILE + bottom - row,
                tile_x * TILE + left - col : tile_x * TILE + right - col,
            ] = tile[top:bottom, left:right]
    return pixels


def _load_array(path: Path) -> npt.NDArray[Any]:
    if path.suffix == ".npy":
        return np.asarray(np.load(path))
    with Image.open(path) as image:
        return np.asarray(image)


def _expected_transform() -> Tuple[float, float, float]:
    """(pixel size, west, north) in EPSG:3857 of the raster whose first tile is (X0, Y0)."""
    pixel = 2 * ORIGIN / (TILE * 2**ZOOM)
    return pixel, -ORIGIN + X0 * TILE * pixel, ORIGIN - Y0 * TILE * pixel


class _MaskReference:
    """Reference masks: labels reprojected with pyproj, burned with rasterio."""

    def __init__(self, geometries: Sequence[Tuple[Polygon, int]]) -> None:
        import shapely

        to_mercator = Transformer.from_crs("EPSG:4326", "EPSG:3857", always_xy=True)

        def project(coords: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
            x, y = to_mercator.transform(coords[:, 0], coords[:, 1])
            return np.column_stack([x, y])

        polygons = shapely.transform(np.array([g for g, _ in geometries], dtype=object), project)
        self._classes = [cls for _, cls in geometries]
        self._polygons = list(polygons)
        self._index = STRtree(self._polygons)

    def mask(self, row: int, col: int) -> npt.NDArray[np.uint8]:
        pixel, west, north = _expected_transform()
        transform = Affine(pixel, 0, west + col * pixel, 0, -pixel, north - row * pixel)
        window = box(
            transform.c,
            transform.f - PATCH * pixel,
            transform.c + PATCH * pixel,
            transform.f,
        )
        hits = self._index.query(window, predicate="intersects")
        if len(hits) == 0:
            return np.zeros((PATCH, PATCH), dtype=np.uint8)
        shapes = [(self._polygons[i], self._classes[i]) for i in sorted(hits)]
        return np.asarray(
            rasterio.features.rasterize(
                shapes, out_shape=(PATCH, PATCH), transform=transform, fill=0, dtype="uint8"
            )
        )


def check_dataset(
    scenario: Scenario,
    dataset: Path,
    geometries: Sequence[Tuple[Polygon, int]],
) -> CheckReport:
    """Validate ``dataset`` against everything the scenario lets us derive independently."""
    report = CheckReport()
    problems = report.problems
    manifest = json.loads((dataset / "manifest.json").read_text(encoding="utf-8"))
    entries: List[Dict[str, Any]] = manifest["patches"]

    if scenario.fail_every:
        failing = sum(
            is_failing(X0 + x, Y0 + y, scenario.fail_every)
            for x in range(scenario.nx)
            for y in range(scenario.ny)
        )
        report.stats["tiles_failing"] = failing
        if failing == 0:
            problems.append("the scenario is meant to inject failures but none fall in its region")
    _check_manifest(scenario, dataset, manifest, entries, report)
    if problems:
        return report  # later checks index into the files the manifest promised

    reference: Optional[_MaskReference] = None
    if HAVE_MASK_REFERENCE:
        reference = _MaskReference(geometries)
    else:
        report.skipped.append(MASK_REFERENCE_HINT)

    rng = random.Random(1)
    sample = scenario.check_sample
    chosen = entries if sample <= 0 or sample >= len(entries) else rng.sample(entries, sample)
    worst_image, mean_image_sum, mask_wrong, mask_total, count_bad = 0, 0.0, 0, 0, 0
    lossy_output = scenario.image_format == "jpg"
    for entry in chosen:
        row, col = entry["row"], entry["col"]
        name = entry["files"]["image"]
        image = _load_array(dataset / name)
        mask = _load_array(dataset / entry["files"]["mask"])
        wanted = expected_image(row, col, scenario)
        difference = np.abs(image.astype(np.int16) - wanted.astype(np.int16))
        worst_image = max(worst_image, int(difference.max()))
        patch_mean = float(difference.mean())
        mean_image_sum += patch_mean
        if lossy_output:
            if patch_mean > JPEG_PATCH_TOLERANCE:
                problems.append(f"{name}: mean JPEG error {patch_mean:.1f}")
        elif difference.max() != 0 and len(problems) < 20:
            problems.append(f"{name}: image differs from the served tiles")
        if reference is not None:
            wanted_mask = reference.mask(row, col)
            # Pixels without imagery (failed tiles, black) carry the ignore value.
            wanted_mask[~wanted.any(axis=-1)] = IGNORE_INDEX
            mask_wrong += int((wanted_mask != mask).sum())
            mask_total += mask.size
        values, counts = np.unique(mask, return_counts=True)
        counted = {str(int(v)): int(n) for v, n in zip(values, counts)}
        if counted != entry["summary"].get("class_pixels"):
            count_bad += 1

    report.stats.update(
        patches=len(entries),
        patches_checked=len(chosen),
        image_max_abs_diff=worst_image,
        image_mean_abs_diff=round(mean_image_sum / max(1, len(chosen)), 3),
        class_count_mismatches=count_bad,
    )
    if lossy_output and report.stats["image_mean_abs_diff"] > JPEG_MEAN_TOLERANCE:
        problems.append(f"mean JPEG error {report.stats['image_mean_abs_diff']} over all patches")
    if count_bad:
        problems.append(f"{count_bad} manifest class counts differ from the mask files")
    if reference is not None:
        disagree = 100 * mask_wrong / max(1, mask_total)
        report.stats["mask_disagree_pct"] = round(disagree, 5)
        if disagree > MASK_DISAGREE_LIMIT_PCT:
            problems.append(f"masks disagree with rasterio on {disagree:.4f} % of pixels")
    _check_splits(scenario, dataset, entries, report)
    return report


def _check_manifest(
    scenario: Scenario,
    dataset: Path,
    manifest: Dict[str, Any],
    entries: List[Dict[str, Any]],
    report: CheckReport,
) -> None:
    problems = report.problems
    if len(entries) != scenario.expected_patches():
        problems.append(f"{len(entries)} patches, expected {scenario.expected_patches()}")
    if manifest.get("version") != 3 or manifest.get("task") != "segmentation":
        problems.append(
            f"manifest version {manifest.get('version')}, task {manifest.get('task')}; "
            "expected version 3, segmentation"
        )
        return
    (source,) = manifest["sources"]
    target = manifest["target"] or {}
    pixel, west, north = _expected_transform()
    a, _, c, _, e, f = source["transform"]
    if not (
        math.isclose(a, pixel, rel_tol=1e-9)
        and math.isclose(e, -pixel, rel_tol=1e-9)
        and math.isclose(c, west, abs_tol=1e-6)
        and math.isclose(f, north, abs_tol=1e-6)
    ):
        problems.append(f"manifest transform {source['transform']} is not the tile grid's")
    if source.get("crs") != "EPSG:3857":
        problems.append(f"manifest crs is {source.get('crs')}")
    if target.get("class_map") != {name: i + 1 for i, name in enumerate(sorted(CLASSES))}:
        problems.append(f"unexpected class map {target.get('class_map')}")
    if target.get("ignore_index") != IGNORE_INDEX:
        problems.append(f"manifest ignore_index is {target.get('ignore_index')}")
    if any(set(entry["files"]) != {"image", "mask"} for entry in entries):
        problems.append("a manifest entry does not list exactly an image and a mask file")
        return

    keys = {(entry["row"], entry["col"]) for entry in entries}
    if len(keys) != len(entries):
        problems.append("duplicate (row, col) entries in the manifest")
    off_grid = [k for k in keys if k[0] % scenario.stride or k[1] % scenario.stride]
    if off_grid:
        problems.append(f"{len(off_grid)} patches are off the stride-{scenario.stride} grid")

    images = sorted(path.name for path in (dataset / "Images").iterdir())
    masks = sorted(path.name for path in (dataset / "Masks").iterdir())
    listed_images = sorted(entry["files"]["image"] for entry in entries)
    listed_masks = sorted(entry["files"]["mask"] for entry in entries)
    if ["Images/" + name for name in images] != listed_images:
        problems.append(f"manifest lists {len(entries)} images, Images/ holds {len(images)}")
    if ["Masks/" + name for name in masks] != listed_masks:
        problems.append(f"manifest lists {len(entries)} masks, Masks/ holds {len(masks)}")
    leftovers = [p.name for p in dataset.rglob("*.tmp")]
    if leftovers:
        problems.append(f"temporary files left behind: {leftovers[:3]}")


def _overlapping(
    patch: Tuple[int, int], held: Dict[Tuple[int, int], List[Tuple[int, int]]]
) -> bool:
    """Whether ``patch`` shares a pixel with any patch in the bucketed ``held`` set."""
    row, col = patch
    for bucket_row in (row // PATCH - 1, row // PATCH, row // PATCH + 1):
        for bucket_col in (col // PATCH - 1, col // PATCH, col // PATCH + 1):
            for other_row, other_col in held.get((bucket_row, bucket_col), ()):
                if abs(row - other_row) < PATCH and abs(col - other_col) < PATCH:
                    return True
    return False


def _bucket(positions: Sequence[Tuple[int, int]]) -> Dict[Tuple[int, int], List[Tuple[int, int]]]:
    buckets: Dict[Tuple[int, int], List[Tuple[int, int]]] = {}
    for row, col in positions:
        buckets.setdefault((row // PATCH, col // PATCH), []).append((row, col))
    return buckets


def _check_splits(
    scenario: Scenario, dataset: Path, entries: List[Dict[str, Any]], report: CheckReport
) -> None:
    problems = report.problems
    # Split lists name a patch by its image's file name.
    where = {
        entry["files"]["image"].rsplit("/", 1)[-1]: (entry["row"], entry["col"])
        for entry in entries
    }
    lists: Dict[str, List[str]] = {}
    for name in SPLIT_NAMES:
        text = (dataset / "splits" / f"{name}.txt").read_text(encoding="utf-8")
        if text and not text.endswith("\n"):
            problems.append(f"{name}.txt lacks a trailing newline")
        lists[name] = [line for line in text.splitlines() if line]
        unknown = [line for line in lists[name] if line not in where]
        if unknown:
            problems.append(f"{name}.txt names {len(unknown)} files that are not in the manifest")
        if len(set(lists[name])) != len(lists[name]):
            problems.append(f"{name}.txt lists a patch twice")
    sets: Dict[str, Set[str]] = {name: set(lines) for name, lines in lists.items()}
    for first, second in (("train", "val"), ("train", "test"), ("val", "test")):
        shared = sets[first] & sets[second]
        if shared:
            problems.append(f"{first} and {second} share {len(shared)} patches")

    positions = {name: [where[n] for n in sets[name] if n in where] for name in SPLIT_NAMES}
    leaks = 0
    held_out = _bucket(positions["test"])
    for name in ("train", "val"):
        leaks += sum(_overlapping(position, held_out) for position in positions[name])
    val_buckets = _bucket(positions["val"])
    leaks += sum(_overlapping(position, val_buckets) for position in positions["train"])
    if leaks:
        problems.append(f"{leaks} patches share pixels with a patch in another split")

    used = sum(len(names) for names in sets.values())
    report.stats["split_counts"] = {name: len(names) for name, names in sets.items()}
    report.stats["dropped_for_leakage"] = len(entries) - used
    if scenario.stride == PATCH and used != len(entries):
        problems.append(f"{len(entries) - used} patches dropped without any overlap to avoid")
    if used > len(entries):
        problems.append("splits list more patches than the manifest has")
