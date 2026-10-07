"""Buffered line and point labels, and the annotated area of partly labeled scenes.

Masks are compared with ``rasterio.features.rasterize`` of the expected geometries on
each patch's own transform: the lines and points buffered in metres in the imagery's
UTM zone, and everything outside the annotated area set to the ignore value.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import numpy.typing as npt
import pytest
from PIL import Image
from pydantic import ValidationError
from pyproj import Transformer
from shapely.geometry import LineString, Point
from shapely.ops import transform as shapely_transform

pytest.importorskip("rasterio", reason="these tests compare masks with rasterio")
import rasterio.features
from rasterio.transform import Affine
from test_multi_source import (
    EPSG,
    PATCH,
    reference_transform,
    region_inside,
    write_raster,
)

from mapcv.config import MapcvConfig
from mapcv.labels import buffer_metres, load_vector_labels
from mapcv.manifest import Manifest
from mapcv.pipeline import run_generate

WIDTH, HEIGHT = 448, 384
TO_UTM = Transformer.from_crs("EPSG:4326", f"EPSG:{EPSG}", always_xy=True)


def _feature(geometry: Any, kind: str | None = "road") -> dict[str, Any]:
    return {
        "type": "Feature",
        "properties": {} if kind is None else {"kind": kind},
        "geometry": geometry.__geo_interface__,
    }


def _write(path: Path, features: list[dict[str, Any]]) -> Path:
    path.write_text(json.dumps({"type": "FeatureCollection", "features": features}))
    return path


def _at(region: dict[str, float], fx: float, fy: float) -> tuple[float, float]:
    return (
        region["west"] + fx * (region["east"] - region["west"]),
        region["south"] + fy * (region["north"] - region["south"]),
    )


# ── Buffering in metres ──────────────────────────────────────────────────────


@pytest.mark.parametrize(
    ("lon", "lat", "epsg"), [(3.0, 48.85, 32631), (-70.6, -33.4, 32719), (24.9, 60.2, 32635)]
)
def test_buffers_are_metres_on_the_ground(lon: float, lat: float, epsg: int) -> None:
    to_utm = Transformer.from_crs("EPSG:4326", f"EPSG:{epsg}", always_xy=True)
    line = LineString([(lon, lat), (lon + 0.002, lat + 0.001)])
    road = shapely_transform(to_utm.transform, buffer_metres(line, 3.0))
    axis = shapely_transform(to_utm.transform, line)
    # Every vertex of the polygon is 3 m from the line, to within 1 cm.
    distances = [axis.distance(Point(xy)) for xy in road.exterior.coords]
    assert min(distances) == pytest.approx(3.0, abs=0.01)
    assert max(distances) == pytest.approx(3.0, abs=0.01)
    disc = shapely_transform(to_utm.transform, buffer_metres(Point(lon, lat), 5.0))
    assert disc.area == pytest.approx(np.pi * 25.0, rel=0.01)


def test_lines_and_points_become_polygons_only_when_buffered(tmp_path: Path) -> None:
    path = _write(
        tmp_path / "mixed.geojson",
        [
            _feature(LineString([(3.0, 48.85), (3.001, 48.851)])),
            _feature(Point(3.002, 48.852), "tree"),
            _feature(Point(3.0, 48.85).buffer(0.0005), "field"),
        ],
    )
    with pytest.warns(UserWarning, match="skipped 2 without polygon geometry"):
        plain, _ = load_vector_labels(path, "kind")
    assert len(plain) == 1
    buffered, classes = load_vector_labels(path, "kind", buffer=(6.0, 10.0))
    assert [geometry.geom_type for geometry, _ in buffered] == ["Polygon"] * 3
    assert classes == {"field": 1, "road": 2, "tree": 3}
    only_lines, _ = load_vector_labels(path, "kind", buffer=(6.0, None))
    assert len(only_lines) == 2  # the point is still skipped


# ── Datasets ─────────────────────────────────────────────────────────────────


@pytest.fixture
def scene(tmp_path: Path) -> dict[str, Any]:
    ref = reference_transform()
    write_raster(tmp_path / "image.tif", ref, WIDTH, HEIGHT, seed=1)
    return {"region": region_inside(ref, WIDTH, HEIGHT)}


def _config(
    tmp_path: Path, region: dict[str, float], labels: dict[str, Any], **extra: Any
) -> MapcvConfig:
    return MapcvConfig.model_validate(
        {
            "region": region,
            "imagery": {"type": "geotiff", "path": str(tmp_path / "image.tif")},
            "labels": labels,
            "sampler": {"patch_size": PATCH, "edge_strategy": "drop"},
            "writer": {"staging_dir": str(tmp_path / extra.pop("staging", "dataset"))},
            **extra,
        }
    )


def _rasterize(
    shapes: list[tuple[Any, int]], entry: Any, manifest: Manifest
) -> npt.NDArray[np.uint8]:
    burned: npt.NDArray[np.uint8] = rasterio.features.rasterize(
        [(geometry.__geo_interface__, value) for geometry, value in shapes]
        or [(Point(0, 0).__geo_interface__, 0)],
        out_shape=(PATCH, PATCH),
        transform=Affine(*manifest.patch_transform(entry)),
        fill=0,
        dtype="uint8",
    )
    return burned


def test_buffered_roads_and_trees_in_masks(tmp_path: Path, scene: dict[str, Any]) -> None:
    region = scene["region"]
    road = LineString([_at(region, 0.1, 0.2), _at(region, 0.5, 0.6), _at(region, 0.9, 0.55)])
    trees = [Point(_at(region, fx, fy)) for fx, fy in ((0.3, 0.8), (0.7, 0.25), (0.15, 0.6))]
    path = _write(
        tmp_path / "osm.geojson", [_feature(road, "road")] + [_feature(t, "tree") for t in trees]
    )
    labels = {
        "path": str(path),
        "label_field": "kind",
        "classes": {"road": 1, "tree": 2},
        "buffer": {"line": 6, "point": 8},
    }
    config = _config(tmp_path, region, labels)
    manifest = run_generate(config).manifest
    assert manifest.target is not None and manifest.target.labels is not None
    assert manifest.target.labels["buffer"] == {"line": 6.0, "point": 8.0}
    # Expected: the same shapes buffered in metres in the imagery's UTM zone.
    expected = [(shapely_transform(TO_UTM.transform, road).buffer(3.0, quad_segs=8), 1)] + [
        (shapely_transform(TO_UTM.transform, tree).buffer(4.0, quad_segs=8), 2) for tree in trees
    ]
    labeled = 0
    for entry in manifest.patches:
        mask = np.asarray(Image.open(config.writer.staging_dir / entry["files"]["mask"]))
        want = _rasterize(expected, entry, manifest)
        np.testing.assert_array_equal(mask, want, err_msg=f"{entry['row']},{entry['col']}")
        labeled += int((want > 0).any())
    assert labeled > 0


def test_pixels_outside_the_annotated_area_are_ignored(
    tmp_path: Path, scene: dict[str, Any]
) -> None:
    region = scene["region"]
    house = Point(_at(region, 0.4, 0.4)).buffer(0.0002)
    labels_path = _write(tmp_path / "houses.geojson", [_feature(house, "house")])
    area = Point(_at(region, 0.4, 0.45)).buffer(0.0006)
    area_path = _write(tmp_path / "area.geojson", [_feature(area, None)])
    config = _config(tmp_path, region, {"path": str(labels_path), "annotated_area": str(area_path)})
    manifest = run_generate(config).manifest
    record = manifest.target.labels if manifest.target is not None else None
    assert record is not None and "annotated_area" not in record
    assert len(record["annotated_area_sha256"]) == 64
    inside_and_outside = 0
    for entry in manifest.patches:
        mask = np.asarray(Image.open(config.writer.staging_dir / entry["files"]["mask"]))
        house_px = _rasterize([(shapely_transform(TO_UTM.transform, house), 1)], entry, manifest)
        area_px = _rasterize([(shapely_transform(TO_UTM.transform, area), 1)], entry, manifest)
        want = np.where(area_px == 1, house_px, 255).astype(np.uint8)
        np.testing.assert_array_equal(mask, want, err_msg=f"{entry['row']},{entry['col']}")
        inside_and_outside += int(area_px.any() and not area_px.all())
    assert inside_and_outside > 0

    # Editing the area is a different dataset: the resume refuses it.
    _write(area_path, [_feature(Point(_at(region, 0.5, 0.5)).buffer(0.0006), None)])
    with pytest.raises(ValueError, match="labels"):
        run_generate(config)


def test_classification_coverage_counts_only_the_annotated_area(
    tmp_path: Path, scene: dict[str, Any]
) -> None:
    region = scene["region"]
    field = Point(_at(region, 0.5, 0.5)).buffer(0.0015)
    labels_path = _write(tmp_path / "fields.geojson", [_feature(field, "crop")])
    area_path = _write(
        tmp_path / "area.geojson", [_feature(Point(_at(region, 0.5, 0.5)).buffer(0.0008), None)]
    )
    config = _config(
        tmp_path,
        region,
        {"path": str(labels_path), "annotated_area": str(area_path)},
        task="classification",
        classification={"empty": "skip"},
    )
    manifest = run_generate(config).manifest
    for entry in manifest.patches:
        assert all(0 < share <= 1 for share in entry["summary"]["class_coverage"].values())


def test_existing_label_settings_record_no_new_keys(tmp_path: Path, scene: dict[str, Any]) -> None:
    path = _write(
        tmp_path / "a.geojson", [_feature(Point(_at(scene["region"], 0.5, 0.5)).buffer(0.0003))]
    )
    manifest = run_generate(_config(tmp_path, scene["region"], {"path": str(path)})).manifest
    assert manifest.target is not None and manifest.target.labels is not None
    assert {"buffer", "annotated_area", "annotated_area_sha256"}.isdisjoint(manifest.target.labels)


# ── Config ───────────────────────────────────────────────────────────────────


BASE: dict[str, Any] = {
    "region": {"west": 4.9, "south": 52.3, "east": 4.91, "north": 52.31},
    "imagery": {"type": "xyz", "zoom": 18, "source": "esri_satellite"},
    "sampler": {"patch_size": 256},
    "writer": {"staging_dir": "out"},
}


@pytest.mark.parametrize(
    ("changes", "message"),
    [
        ({"labels": {"path": "a.geojson", "buffer": {}}}, "needs a line or point distance"),
        ({"labels": {"path": "a.geojson", "buffer": {"line": -1}}}, "greater than 0"),
        ({"labels": {"path": "a.kml", "buffer": {"line": 3}}}, "does not read from KML"),
        (
            {
                "task": "detection",
                "labels": {"path": "a.geojson", "buffer": {"point": 3}},
                "detection": {"point_box_size": 5},
            },
            "keep one",
        ),
        ({"labels": {"path": "a.geojson", "annotated_area": "area.tif"}}, "must be a polygon file"),
        (
            {"labels": {"path": "a.geojson", "annotated_area": "a.geojson", "ignore_index": None}},
            "needs labels.ignore_index",
        ),
        (
            {"task": "detection", "labels": {"path": "a.geojson", "annotated_area": "a.geojson"}},
            "not detection",
        ),
    ],
)
def test_config_refuses_bad_buffers_and_areas(changes: dict[str, Any], message: str) -> None:
    with pytest.raises(ValidationError) as raised:
        MapcvConfig.model_validate({**BASE, **changes})
    assert message in str(raised.value)


def test_area_path_resolves_and_is_sandboxed(tmp_path: Path) -> None:
    from mapcv.agent_tools import config_paths

    folder = tmp_path / "project"
    folder.mkdir()
    (folder / "mapcv.yaml").write_text(
        "region: {west: 4.9, south: 52.3, east: 4.91, north: 52.31}\n"
        "imagery: {type: xyz, zoom: 18, source: esri_satellite}\n"
        "labels: {path: a.geojson, annotated_area: areas/done.geojson}\n"
        "sampler: {patch_size: 256}\n"
        "writer: {staging_dir: out}\n"
    )
    config = MapcvConfig.from_yaml(folder / "mapcv.yaml")
    assert dict(config_paths(config))["labels.annotated_area"] == folder / "areas" / "done.geojson"
