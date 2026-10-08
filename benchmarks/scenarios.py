"""Benchmark scenarios, deterministic synthetic labels and the matching mapcv config."""

from __future__ import annotations

import json
import math
import random
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Literal

from shapely.geometry import Polygon

ZOOM = 18
# Tile (X0, Y0) at zoom 18 is in the Amsterdam docklands; any tile would do, the
# synthetic tile server does not care, but real coordinates keep the numbers honest.
X0, Y0 = 134_700, 86_100
PATCH = 256
CLASSES = ["building", "road", "water"]  # mapcv numbers classes in sorted order: 1, 2, 3
LABEL_SEED = 0

Kind = Literal["generate", "strict", "resume"]
Group = Literal["quick", "standard", "large"]


@dataclass(frozen=True)
class Scenario:
    """One benchmark workload: a block of ``nx`` x ``ny`` tiles with ``polygons`` labels."""

    name: str
    description: str
    group: Group
    nx: int
    ny: int
    polygons: int
    tile_format: str = "png"  # what the tile server sends
    image_format: str = "png"  # what mapcv writes (writer.image_format)
    stride: int = PATCH
    strip_rows: int = 4  # imagery.strip_rows: tile rows held in memory at once
    fail_every: int = 0  # the server answers an error for about 1 tile in this many
    fail_status: int = 404  # 404 is not retried; 500 is, with a multi-second backoff
    policy: str = "lenient"
    max_failed_ratio: float = 0.05
    latency_ms: int = 0  # extra server latency per tile request
    check_sample: int = 0  # patches to compare pixel by pixel, 0 for all of them
    kind: Kind = "generate"

    @property
    def tiles(self) -> int:
        return self.nx * self.ny

    def expected_patches(self) -> int:
        """Patches on the drop-edge grid (the region is a whole number of tiles), less those
        whose tiles all fail: mapcv leaves out a patch without any imagery."""
        from benchmarks.tileserver import is_failing

        anchors = [
            [step * self.stride for step in range((n * PATCH - PATCH) // self.stride + 1)]
            for n in (self.nx, self.ny)
        ]
        kept = 0
        for x in anchors[0]:
            for y in anchors[1]:
                tiles = [
                    (X0 + tx, Y0 + ty)
                    for tx in range(x // PATCH, (x + PATCH - 1) // PATCH + 1)
                    for ty in range(y // PATCH, (y + PATCH - 1) // PATCH + 1)
                ]
                if not all(is_failing(tx, ty, self.fail_every) for tx, ty in tiles):
                    kept += 1
        return kept

    def tile_prefix(self) -> str:
        """Server option segment: ``n50/`` 404s every ~50th tile, ``l20/`` adds 20 ms."""
        options = []
        if self.fail_every:
            options.append(f"{'n' if self.fail_status == 404 else 'f'}{self.fail_every}")
        if self.latency_ms:
            options.append(f"l{self.latency_ms}")
        return "".join(f"{option}/" for option in options)

    def to_json(self) -> dict[str, Any]:
        data = asdict(self)
        data["tiles"] = self.tiles
        return data


_SCENARIOS = [
    # quick: seconds in total, what the pytest smoke test and `--quick` run
    Scenario("Q", "6x6 tiles, 60 polygons: end-to-end smoke test", "quick", 6, 6, 60),
    Scenario(
        "Q-failures",
        "Q with every ~7th tile failing under the lenient policy",
        "quick",
        6,
        6,
        60,
        fail_every=7,
        max_failed_ratio=0.5,
    ),
    Scenario(
        "Q-strict",
        "Q with failing tiles under the strict policy: must exit non-zero, cleanly",
        "quick",
        6,
        6,
        60,
        fail_every=7,
        policy="strict",
        kind="strict",
    ),
    Scenario(
        "Q-resume",
        "Q interrupted with Ctrl-C and resumed: identical to an uninterrupted run",
        "quick",
        6,
        6,
        60,
        strip_rows=1,
        latency_ms=30,
        kind="resume",
    ),
    # standard: roughly a minute per repeat
    Scenario("S", "10x10 tiles (100 patches), 300 polygons", "standard", 10, 10, 300),
    Scenario("M", "32x32 tiles (1,024 patches), 3,000 polygons", "standard", 32, 32, 3_000),
    Scenario(
        "M-jpg",
        "M written as JPEG (writer.image_format: jpg)",
        "standard",
        32,
        32,
        3_000,
        image_format="jpg",
    ),
    Scenario(
        "M-jpgtiles",
        "M with a server that sends JPEG tiles (decode cost, lossy source)",
        "standard",
        32,
        32,
        3_000,
        tile_format="jpg",
    ),
    Scenario(
        "M-overlap",
        "M with 50 % overlapping patches (stride 128): 3,969 patches, split must not leak",
        "standard",
        32,
        32,
        3_000,
        stride=128,
    ),
    Scenario(
        "M-failures",
        "M with about 2 % of tiles failing under the lenient policy",
        "standard",
        32,
        32,
        3_000,
        fail_every=50,
    ),
    Scenario(
        "M-strict",
        "M with failing tiles under the strict policy: must exit non-zero, cleanly",
        "standard",
        32,
        32,
        3_000,
        fail_every=50,
        policy="strict",
        kind="strict",
    ),
    Scenario(
        "M-resume",
        "M interrupted with Ctrl-C at about 40 % and resumed: identical to an uninterrupted run",
        "standard",
        32,
        32,
        3_000,
        strip_rows=1,
        latency_ms=20,
        kind="resume",
    ),
    # large: minutes per repeat, opt-in by name
    Scenario(
        "L",
        "100x100 tiles (10,000 patches), 30,000 polygons",
        "large",
        100,
        100,
        30_000,
        check_sample=400,
    ),
    Scenario(
        "XL",
        "200x200 tiles (40,000 patches), 30,000 polygons: memory and disk at the largest size",
        "large",
        200,
        200,
        30_000,
        check_sample=400,
    ),
    Scenario(
        "M-polys100k",
        "M's raster with 100,000 polygons: label indexing and rasterization cost",
        "large",
        32,
        32,
        100_000,
        check_sample=300,
    ),
]

SCENARIOS: dict[str, Scenario] = {scenario.name: scenario for scenario in _SCENARIOS}


def names_in_group(group: Group) -> list[str]:
    """Scenario names of one group, in definition order."""
    return [scenario.name for scenario in _SCENARIOS if scenario.group == group]


def _lon(x: float) -> float:
    return float(x / 2**ZOOM * 360 - 180)


def _lat(y: float) -> float:
    return math.degrees(math.atan(math.sinh(math.pi * (1 - 2 * y / 2**ZOOM))))


def region_bounds(scenario: Scenario) -> tuple[float, float, float, float]:
    """(west, south, east, north) of the scenario's tile block, shrunk by 1e-7 degrees.

    The shrink keeps the edges strictly inside the outer tiles so the region
    snaps to exactly ``nx`` x ``ny`` tiles whatever the rounding.
    """
    eps = 1e-7
    return (
        _lon(X0) + eps,
        _lat(Y0 + scenario.ny) + eps,
        _lon(X0 + scenario.nx) - eps,
        _lat(Y0) - eps,
    )


def make_labels(scenario: Scenario, path: Path) -> list[tuple[Polygon, int]]:
    """Write jittered, non-overlapping rectangles in three classes; return them with class ids.

    Fully deterministic (seeded), so every run and every machine sees the same labels.
    """
    rng = random.Random(LABEL_SEED)
    west, south, east, north = region_bounds(scenario)
    columns = max(1, int(math.sqrt(scenario.polygons * (east - west) / (north - south))))
    rows = max(1, math.ceil(scenario.polygons / columns))
    cell_w, cell_h = (east - west) / columns, (north - south) / rows
    features: list[dict[str, Any]] = []
    geometries: list[tuple[Polygon, int]] = []
    for row in range(rows):
        for column in range(columns):
            if len(features) >= scenario.polygons:
                break
            width, height = rng.uniform(0.3, 0.8) * cell_w, rng.uniform(0.3, 0.8) * cell_h
            x = west + column * cell_w + rng.uniform(0, cell_w - width)
            y = south + row * cell_h + rng.uniform(0, cell_h - height)
            name = CLASSES[rng.randrange(len(CLASSES))]
            ring = [(x, y), (x + width, y), (x + width, y + height), (x, y + height), (x, y)]
            features.append(
                {
                    "type": "Feature",
                    "properties": {"class": name},
                    "geometry": {"type": "Polygon", "coordinates": [ring]},
                }
            )
            geometries.append((Polygon(ring), CLASSES.index(name) + 1))
    path.write_text(json.dumps({"type": "FeatureCollection", "features": features}))
    return geometries


def config_yaml(scenario: Scenario, port: int) -> str:
    """The mapcv YAML for a scenario against the local tile server on ``port``."""
    west, south, east, north = region_bounds(scenario)
    url = (
        f"http://127.0.0.1:{port}/{scenario.tile_prefix()}{{z}}/{{x}}/{{y}}.{scenario.tile_format}"
    )
    return f"""region: {{west: {west!r}, south: {south!r}, east: {east!r}, north: {north!r}}}
imagery:
  type: xyz
  zoom: {ZOOM}
  url_template: '{url}'
  max_connections: 16
  strip_rows: {scenario.strip_rows}
  policy: {scenario.policy}
  max_failed_ratio: {scenario.max_failed_ratio}
  cache: false
labels:
  path: labels.geojson
  label_field: class
sampler:
  patch_size: {PATCH}
  stride: {scenario.stride}
  edge_strategy: drop
writer:
  staging_dir: ./dataset
  image_format: {scenario.image_format}
split:
  strategy: spatial
  test_ratio: 0.2
  val_ratio: 0.1
  seed: 42
"""
