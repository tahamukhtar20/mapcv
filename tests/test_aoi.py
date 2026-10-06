"""Areas of interest (``region.path``): patches only over polygons, named regions, and
``split.strategy: region``.

Which patches are kept is compared with an independent computation: every grid patch
as a box in the imagery CRS (from rasterio's transform), intersected with the polygons
projected by pyproj; a patch is kept when the overlap has a positive area.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Set, Tuple

import numpy as np
import pytest
from pydantic import ValidationError
from pyproj import Transformer
from shapely.geometry import box
from shapely.ops import transform as shapely_transform

pytest.importorskip("rasterio", reason="AOI tests write their imagery with rasterio")
from rasterio.transform import Affine  # noqa: E402

from mapcv._mapcv_rs import grid_sample_anchors  # noqa: E402
from mapcv.config import MapcvConfig  # noqa: E402
from mapcv.manifest import Manifest  # noqa: E402
from mapcv.pipeline import run_generate, run_split  # noqa: E402
from mapcv.planning import plan  # noqa: E402
from test_multi_source import EPSG, reference_transform, write_raster  # noqa: E402

PATCH = 32
WIDTH, HEIGHT = 640, 384
TO_LONLAT = Transformer.from_crs(f"EPSG:{EPSG}", "EPSG:4326", always_xy=True)
TO_UTM = Transformer.from_crs("EPSG:4326", f"EPSG:{EPSG}", always_xy=True)


def _lonlat_box(ref: Affine, col0: float, row0: float, col1: float, row1: float) -> Any:
    """A lon/lat polygon over raster pixels ``[col0, col1) x [row0, row1)``."""
    x0, y0 = ref * (col0, row1)
    x1, y1 = ref * (col1, row0)
    return shapely_transform(TO_LONLAT.transform, box(x0, y0, x1, y1).segmentize(5.0))


def _write_aoi(path: Path, polygons: List[Tuple[Any, Dict[str, Any]]]) -> Path:
    features = [
        {"type": "Feature", "properties": props, "geometry": geometry.__geo_interface__}
        for geometry, props in polygons
    ]
    path.write_text(json.dumps({"type": "FeatureCollection", "features": features}))
    return path


@pytest.fixture
def scene(tmp_path: Path) -> Dict[str, Any]:
    ref = reference_transform()
    write_raster(tmp_path / "image.tif", ref, WIDTH, HEIGHT, seed=1)
    # Three regions: two far apart on the left and right, one small in the middle.
    polygons = [
        (_lonlat_box(ref, 20, 30, 150, 300), {"name": "west"}),
        (_lonlat_box(ref, 470, 60, 620, 350), {"name": "east"}),
        (_lonlat_box(ref, 300, 150, 345, 200), {"name": "middle"}),
    ]
    aoi = _write_aoi(tmp_path / "aoi.geojson", polygons)
    return {"ref": ref, "aoi": aoi, "polygons": polygons}


def _config(tmp_path: Path, aoi: Path, staging: str = "dataset", **extra: Any) -> MapcvConfig:
    return MapcvConfig.model_validate(
        {
            "region": {"path": str(aoi), "name_field": "name", **extra.pop("region", {})},
            "imagery": {"type": "geotiff", "path": str(tmp_path / "image.tif"), "chunk_rows": 64},
            "sampler": {"patch_size": PATCH, "edge_strategy": "drop"},
            "writer": {"staging_dir": str(tmp_path / staging), "image_format": "png"},
            **extra,
        }
    )


def _expected(
    config: MapcvConfig, manifest: Manifest, polygons: List[Tuple[Any, Dict[str, Any]]]
) -> Dict[Tuple[int, int], str]:
    """Every grid anchor whose patch overlaps a polygon, with its region (largest overlap)."""
    from mapcv.imagery import open_raster_source

    source = open_raster_source(config.region, config.primary_imagery)
    height, width = source.metadata.height, source.metadata.width
    transform = Affine(*manifest.source.transform)  # type: ignore[misc]
    world = [
        (shapely_transform(TO_UTM.transform, geometry), props["name"])
        for geometry, props in polygons
    ]
    expected: Dict[Tuple[int, int], str] = {}
    for row, col in grid_sample_anchors(height, width, PATCH, PATCH, "drop"):
        x0, y0 = transform * (col, row + PATCH)
        x1, y1 = transform * (col + PATCH, row)
        patch = box(x0, y0, x1, y1)
        areas: Dict[str, float] = {}
        for geometry, name in world:
            area = patch.intersection(geometry).area
            if area > 0:
                areas[name] = areas.get(name, 0.0) + area
        if areas:
            expected[(row, col)] = max(areas, key=lambda name: areas[name])
    return expected


def test_patches_cover_only_the_area_of_interest(tmp_path: Path, scene: Dict[str, Any]) -> None:
    config = _config(tmp_path, scene["aoi"])
    region = config.region
    west = min(g.bounds[0] for g, _ in scene["polygons"])
    assert region.west == pytest.approx(west) and region.path == scene["aoi"]
    manifest = run_generate(config).manifest
    got = {(entry["row"], entry["col"]): entry["summary"]["region"] for entry in manifest.patches}
    assert got == _expected(config, manifest, scene["polygons"])
    assert set(got.values()) == {"west", "east", "middle"}
    record = manifest.model_extra or {}
    assert record["region"]["name_field"] == "name" and len(record["region"]["aoi_sha256"]) == 64


def test_far_apart_polygons_are_read_in_small_windows(
    tmp_path: Path, scene: Dict[str, Any], monkeypatch: pytest.MonkeyPatch
) -> None:
    from mapcv.imagery import GeoTiffRasterSource

    widths: List[int] = []
    real_read = GeoTiffRasterSource.read_window

    def read_window(self: Any, r0: int, r1: int, c0: int, c1: int) -> Any:
        widths.append(c1 - c0)
        return real_read(self, r0, r1, c0, c1)

    monkeypatch.setattr(GeoTiffRasterSource, "read_window", read_window)
    manifest = run_generate(_config(tmp_path, scene["aoi"])).manifest
    assert manifest.patches
    # No window spans from the west polygon to the east one (about 600 px apart).
    assert max(widths) < 300


def test_resume_and_an_edited_area(tmp_path: Path, scene: Dict[str, Any]) -> None:
    config = _config(tmp_path, scene["aoi"])
    full = run_generate(config).manifest
    staging = config.writer.staging_dir
    cut = Manifest.load(staging / "manifest.json")
    cut.patches = cut.patches[: len(cut.patches) // 2]
    cut.save(staging / "manifest.json")
    assert run_generate(config).manifest.patches == full.patches
    _write_aoi(scene["aoi"], scene["polygons"][:2])
    with pytest.raises(ValueError, match="region"):
        run_generate(_config(tmp_path, scene["aoi"]))


def test_region_split_keeps_each_region_in_one_split(tmp_path: Path, scene: Dict[str, Any]) -> None:
    config = _config(tmp_path, scene["aoi"])
    manifest = run_generate(config).manifest
    from mapcv.splitter import SplitterConfig

    run_split(
        config.writer.staging_dir, SplitterConfig(strategy="region", test_ratio=0.3, val_ratio=0.2)
    )
    splits: Dict[str, Set[str]] = {}
    region_of = {manifest.patch_name(e): e["summary"]["region"] for e in manifest.patches}
    for name in ("train", "val", "test"):
        names = (config.writer.staging_dir / "splits" / f"{name}.txt").read_text().split()
        splits[name] = {region_of[n] for n in names}
    # Three regions of very different sizes: val may stay empty; no region is shared.
    assert splits["train"] and splits["test"]
    assert not (splits["train"] & splits["test"]) and not (splits["val"] & splits["test"])
    assert not (splits["train"] & splits["val"])

    two = _write_aoi(tmp_path / "two.geojson", scene["polygons"][:2])
    small = _config(tmp_path, two, staging="two")
    run_generate(small)
    with pytest.warns(UserWarning, match="Only 2 region"):
        run_split(small.writer.staging_dir, SplitterConfig(strategy="region"))


def test_region_split_needs_regions(tmp_path: Path, scene: Dict[str, Any]) -> None:
    from mapcv.splitter import SplitterConfig

    ref = scene["ref"]
    west, south = TO_LONLAT.transform(*(ref * (40, 340)))
    east, north = TO_LONLAT.transform(*(ref * (600, 40)))
    config = MapcvConfig.model_validate(
        {
            "region": {"west": west, "south": south, "east": east, "north": north},
            "imagery": {"type": "geotiff", "path": str(tmp_path / "image.tif")},
            "sampler": {"patch_size": PATCH},
            "writer": {"staging_dir": str(tmp_path / "plain")},
        }
    )
    run_generate(config)
    with pytest.raises(ValueError, match="records no regions"):
        run_split(config.writer.staging_dir, SplitterConfig(strategy="region"))


def test_numbered_regions_and_plan_estimate(tmp_path: Path, scene: Dict[str, Any]) -> None:
    aoi = _write_aoi(tmp_path / "unnamed.geojson", [(g, {}) for g, _ in scene["polygons"]])
    config = MapcvConfig.model_validate(
        {
            "region": {"path": str(aoi)},
            "imagery": {"type": "geotiff", "path": str(tmp_path / "image.tif")},
            "sampler": {"patch_size": PATCH, "edge_strategy": "drop"},
            "writer": {"staging_dir": str(tmp_path / "numbered")},
        }
    )
    estimate = plan(config)
    manifest = run_generate(config).manifest
    assert {entry["summary"]["region"] for entry in manifest.patches} == {"1", "2", "3"}
    # The estimate scales the box's patches by the area covered: close to the real count.
    assert 0.6 * len(manifest.patches) <= estimate.patches <= 1.6 * len(manifest.patches)


def test_mcp_refuses_an_area_outside_its_root(tmp_path: Path, scene: Dict[str, Any]) -> None:
    from mapcv.agent_tools import Sandbox, ToolFailure, ToolState, config_paths, parse_config_text

    root = tmp_path / "root"
    root.mkdir()
    state = ToolState(Sandbox(root))
    text = (
        f"region: {{path: {scene['aoi']}}}\n"
        "imagery: {type: xyz, zoom: 15, source: esri_satellite}\n"
        "sampler: {patch_size: 256}\nwriter: {staging_dir: out}\n"
    )
    with pytest.raises(ToolFailure, match="region.path"):
        parse_config_text(state, text, root)
    config = _config(tmp_path, scene["aoi"])
    assert dict(config_paths(config))["region.path"] == scene["aoi"]


@pytest.mark.parametrize(
    ("region", "message"),
    [
        ({"path": "aoi.geojson", "west": 1}, "not both"),
        ({"path": "missing.geojson"}, "region.path not found"),
        ({"path": "aoi.tif"}, "must be a polygon file"),
        (
            {"west": 4.9, "south": 52.3, "east": 4.91, "north": 52.31, "name_field": "n"},
            "need region.path",
        ),
    ],
)
def test_config_refuses_bad_areas(tmp_path: Path, region: Dict[str, Any], message: str) -> None:
    with pytest.raises(ValidationError) as raised:
        MapcvConfig.model_validate(
            {
                "region": region,
                "imagery": {"type": "xyz", "zoom": 15, "source": "esri_satellite"},
                "sampler": {"patch_size": 256},
                "writer": {"staging_dir": "out"},
            }
        )
    assert message in str(raised.value)


def test_area_path_resolves_against_the_config(tmp_path: Path, scene: Dict[str, Any]) -> None:
    folder = tmp_path / "project"
    (folder / "areas").mkdir(parents=True)
    (folder / "areas" / "aoi.geojson").write_text(scene["aoi"].read_text())
    (folder / "mapcv.yaml").write_text(
        "region: {path: areas/aoi.geojson, name_field: name}\n"
        "imagery: {type: xyz, zoom: 15, source: esri_satellite}\n"
        "sampler: {patch_size: 256}\nwriter: {staging_dir: out}\n"
    )
    config = MapcvConfig.from_yaml(folder / "mapcv.yaml")
    assert config.region.path == folder / "areas" / "aoi.geojson"
    assert np.isfinite([config.region.west, config.region.north]).all()


# ── Exact geometry (a lon/lat grid of half-degree pixels: no rounding anywhere) ───


def _exact_aoi(tmp_path: Path, polygons: List[Tuple[Any, Dict[str, Any]]]) -> Any:
    from mapcv.aoi import AreaOfInterest
    from mapcv.config import RegionConfig

    path = _write_aoi(tmp_path / "exact.geojson", polygons)
    region = RegionConfig.model_validate({"path": str(path), "name_field": "name"})
    # Half-degree pixels: pixel (row, col) spans lon col/2.. and lat -row/2..; exact in floats.
    return AreaOfInterest(region, "EPSG:4326", (0.5, 0.0, 0.0, 0.0, -0.5, 0.0))


def _lonlat(col0: float, row0: float, col1: float, row1: float) -> Any:
    return box(col0 / 2, -row1 / 2, col1 / 2, -row0 / 2)


def test_a_patch_that_only_touches_a_polygon_is_not_kept(tmp_path: Path) -> None:
    aoi = _exact_aoi(tmp_path, [(_lonlat(64, 32, 128, 96), {"name": "a"})])
    anchors = [(row, col) for row in range(0, 128, 32) for col in range(0, 160, 32)]
    kept = aoi.keep(anchors, 32)
    # Columns 64..128 and rows 32..96 only; neighbours share an edge but no area.
    assert kept == [(32, 64), (32, 96), (64, 64), (64, 96)]


def test_the_region_covering_most_of_a_patch_names_it(tmp_path: Path) -> None:
    aoi = _exact_aoi(
        tmp_path,
        [
            (_lonlat(0, 0, 10, 32), {"name": "small"}),
            (_lonlat(10, 0, 32, 32), {"name": "large"}),
            (_lonlat(32, 0, 48, 32), {"name": "left"}),
            (_lonlat(48, 0, 64, 32), {"name": "right"}),
        ],
    )
    assert aoi.region_of(0, 0, 32) == "large"
    # An exact tie goes to the region listed first in the file.
    assert aoi.region_of(0, 32, 32) == "left"
    assert aoi.region_of(0, 200, 32) == ""
