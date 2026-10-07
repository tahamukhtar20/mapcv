"""Several label files (``labels.files``): one class map, later files win overlaps.

Masks are compared with ``rasterio.features.rasterize`` of every file's features, in
file order, with the class IDs the class map gives their names.
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
from shapely.geometry import LineString, box
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
from mapcv.manifest import Manifest
from mapcv.pipeline import run_generate
from mapcv.planning import plan

WIDTH, HEIGHT = 448, 384
TO_UTM = Transformer.from_crs("EPSG:4326", f"EPSG:{EPSG}", always_xy=True)


def _write(path: Path, features: list[tuple[Any, dict[str, Any]]]) -> Path:
    path.write_text(
        json.dumps(
            {
                "type": "FeatureCollection",
                "features": [
                    {"type": "Feature", "properties": props, "geometry": geom.__geo_interface__}
                    for geom, props in features
                ],
            }
        )
    )
    return path


@pytest.fixture
def scene(tmp_path: Path) -> dict[str, Any]:
    ref = reference_transform()
    write_raster(tmp_path / "image.tif", ref, WIDTH, HEIGHT, seed=1)
    region = region_inside(ref, WIDTH, HEIGHT)
    west, south = region["west"], region["south"]
    dx, dy = region["east"] - west, region["north"] - south

    def rect(x0: float, y0: float, x1: float, y1: float) -> Any:
        return box(west + x0 * dx, south + y0 * dy, west + x1 * dx, south + y1 * dy)

    forest, crop = rect(0.05, 0.05, 0.55, 0.6), rect(0.5, 0.3, 0.95, 0.95)
    houses = [rect(0.2, 0.2, 0.3, 0.3), rect(0.6, 0.6, 0.7, 0.72)]  # on the forest, on the crop
    road = LineString([(west + 0.1 * dx, south + 0.45 * dy), (west + 0.9 * dx, south + 0.5 * dy)])
    files = {
        "fields": _write(
            tmp_path / "fields.geojson",
            [(forest, {"landuse": "forest"}), (crop, {"landuse": "crop"})],
        ),
        "houses": _write(tmp_path / "houses.geojson", [(house, {}) for house in houses]),
        "roads": _write(tmp_path / "roads.geojson", [(road, {})]),
    }
    shapes = {"forest": forest, "crop": crop, "houses": houses, "road": road}
    return {"region": region, "files": files, "shapes": shapes}


def _files(scene: dict[str, Any], road_buffer: bool = True) -> list[dict[str, Any]]:
    files = scene["files"]
    road: dict[str, Any] = {"path": str(files["roads"]), "class": "road"}
    if road_buffer:
        road["buffer"] = {"line": 8}
    return [
        {"path": str(files["fields"]), "label_field": "landuse"},
        {"path": str(files["houses"]), "class": "building"},
        road,
    ]


def _config(
    tmp_path: Path, scene: dict[str, Any], labels: dict[str, Any], staging: str = "out"
) -> MapcvConfig:
    return MapcvConfig.model_validate(
        {
            "region": scene["region"],
            "imagery": {"type": "geotiff", "path": str(tmp_path / "image.tif")},
            "labels": labels,
            "sampler": {"patch_size": PATCH, "edge_strategy": "drop"},
            "writer": {"staging_dir": str(tmp_path / staging)},
        }
    )


def _expected(
    scene: dict[str, Any], ids: dict[str, int], manifest: Manifest, entry: Any
) -> npt.NDArray[np.uint8]:
    shapes = scene["shapes"]

    def utm(geometry: Any) -> Any:
        return shapely_transform(TO_UTM.transform, geometry)

    ordered: list[tuple[Any, int]] = []
    # File order: fields (forest, crop), then houses, then the road; later ones win.
    for name in ("forest", "crop"):
        if name in ids:
            ordered.append((utm(shapes[name]), ids[name]))
    ordered += [(utm(house), ids["building"]) for house in shapes["houses"]]
    ordered.append((utm(shapes["road"]).buffer(4.0, quad_segs=8), ids["road"]))
    burned: npt.NDArray[np.uint8] = rasterio.features.rasterize(
        [(geometry.__geo_interface__, value) for geometry, value in ordered],
        out_shape=(PATCH, PATCH),
        transform=Affine(*manifest.patch_transform(entry)),
        fill=0,
        dtype="uint8",
    )
    return burned


def _check(config: MapcvConfig, scene: dict[str, Any], ids: dict[str, int]) -> Manifest:
    manifest = run_generate(config).manifest
    seen = set()
    for entry in manifest.patches:
        mask = np.asarray(Image.open(config.writer.staging_dir / entry["files"]["mask"]))
        want = _expected(scene, ids, manifest, entry)
        np.testing.assert_array_equal(mask, want, err_msg=f"{entry['row']},{entry['col']}")
        seen |= set(np.unique(want).tolist())
    assert set(ids.values()) <= seen, "not every class reached a patch"
    return manifest


def test_files_share_one_class_map_and_later_files_win(
    tmp_path: Path, scene: dict[str, Any]
) -> None:
    classes = {"building": 1, "road": 2, "forest": 3, "crop": 4}
    config = _config(tmp_path, scene, {"files": _files(scene), "classes": classes})
    manifest = _check(config, scene, classes)
    assert manifest.class_map == classes
    record = manifest.target.labels if manifest.target is not None else None
    assert record is not None and "path" not in record
    assert [file.get("class") for file in record["files"]] == [None, "building", "road"]
    assert all("path" not in file for file in record["files"])
    assert len(record["sha256"]) == 64


def test_ids_follow_sorted_names_across_files_without_classes(
    tmp_path: Path, scene: dict[str, Any]
) -> None:
    config = _config(tmp_path, scene, {"files": _files(scene)})
    ids = {"building": 1, "crop": 2, "forest": 3, "road": 4}
    manifest = _check(config, scene, ids)
    assert manifest.class_map == ids


def test_names_missing_from_classes_are_skipped(tmp_path: Path, scene: dict[str, Any]) -> None:
    classes = {"building": 1, "road": 2, "forest": 3}
    config = _config(tmp_path, scene, {"files": _files(scene), "classes": classes})
    with pytest.warns(UserWarning, match=r"labels.files: skipped 1 feature\(s\)"):
        _check(config, scene, classes)


def test_editing_any_file_is_a_resume_mismatch(tmp_path: Path, scene: dict[str, Any]) -> None:
    config = _config(tmp_path, scene, {"files": _files(scene)})
    run_generate(config)
    houses = scene["files"]["houses"]
    houses.write_text(houses.read_text().replace("]]]", "]]] ", 1))
    with pytest.raises(ValueError, match="labels"):
        run_generate(config)


def test_plan_counts_every_file_and_mcp_sandboxes_them(
    tmp_path: Path, scene: dict[str, Any]
) -> None:
    from mapcv.agent_tools import config_paths

    config = _config(tmp_path, scene, {"files": _files(scene)})
    estimate = plan(config)
    assert estimate.labels is not None and estimate.labels.polygons == 5
    assert estimate.labels.classes == {"building": 1, "crop": 2, "forest": 3, "road": 4}
    keys = dict(config_paths(config))
    assert keys["labels.files[2].path"] == scene["files"]["roads"]


# ── Config ───────────────────────────────────────────────────────────────────


BASE: dict[str, Any] = {
    "region": {"west": 4.9, "south": 52.3, "east": 4.91, "north": 52.31},
    "imagery": {"type": "xyz", "zoom": 18, "source": "esri_satellite"},
    "sampler": {"patch_size": 256},
    "writer": {"staging_dir": "out"},
}
ONE = {"path": "a.geojson", "class": "a"}


@pytest.mark.parametrize(
    ("labels", "message"),
    [
        ({"path": "a.geojson", "files": [ONE]}, "exactly one of path (one label file), files"),
        ({"files": []}, "labels.files is empty"),
        ({"files": [{"path": "a.geojson"}]}, "needs exactly one of label_field"),
        ({"files": [{**ONE, "label_field": "k"}]}, "needs exactly one of label_field"),
        ({"files": [ONE], "label_field": "k"}, "set label_field on each file"),
        ({"files": [{**ONE, "layer": "x"}]}, "layer picks a table of a GeoPackage"),
        ({"files": [{"path": "a.kml", "class": "a", "buffer": {"line": 2}}]}, "from KML"),
        ({"files": [{"path": "a.tif", "class": "a"}]}, "is a raster"),
        ({"files": [{**ONE, "colour": "red"}]}, "colour"),
    ],
)
def test_config_refuses_bad_label_files(labels: dict[str, Any], message: str) -> None:
    with pytest.raises(ValidationError) as raised:
        MapcvConfig.model_validate({**BASE, "labels": labels})
    assert message in str(raised.value)


def test_relative_label_file_paths_resolve(tmp_path: Path) -> None:
    folder = tmp_path / "project"
    folder.mkdir()
    (folder / "mapcv.yaml").write_text(
        "region: {west: 4.9, south: 52.3, east: 4.91, north: 52.31}\n"
        "imagery: {type: xyz, zoom: 18, source: esri_satellite}\n"
        "labels:\n"
        "  files:\n"
        "    - {path: a.geojson, class: a}\n"
        "    - {path: sub/b.gpkg, layer: roads, label_field: kind}\n"
        "sampler: {patch_size: 256}\n"
        "writer: {staging_dir: out}\n"
    )
    config = MapcvConfig.from_yaml(folder / "mapcv.yaml")
    assert config.labels is not None
    paths = [path for _, path in config.labels.keyed_files()]  # type: ignore[union-attr]
    assert paths == [folder / "a.geojson", folder / "sub" / "b.gpkg"]


def test_a_single_path_is_unchanged(tmp_path: Path) -> None:
    config = MapcvConfig.model_validate({**BASE, "labels": {"path": "a.geojson"}})
    assert config.labels is not None
    dumped = config.labels.model_dump(mode="json")
    assert "files" not in dumped and dumped["path"] == "a.geojson"
