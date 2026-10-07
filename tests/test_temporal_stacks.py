"""Several dates in one file per patch (``writer.stack_sources``).

Each stacked patch is compared with what rasterio reads from every date's file at the
patch's position (nearest neighbour for a coarser date), and the masks with the same
dataset written as separate files.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pytest
from pydantic import ValidationError

pytest.importorskip("rasterio", reason="stack tests write their rasters with rasterio")
import rasterio
from rasterio.transform import Affine
from test_multi_source import (
    PATCH,
    expected_on_patch_grid,
    reference_transform,
    region_inside,
    write_labels,
    write_raster,
)

from mapcv.config import MapcvConfig
from mapcv.manifest import Manifest
from mapcv.pipeline import run_generate

WIDTH, HEIGHT = 448, 384
DATES = ["d1", "d2", "d3"]


@pytest.fixture
def dates(tmp_path: Path) -> dict[str, Any]:
    ref = reference_transform()
    for seed, name in enumerate(DATES[:2], start=1):
        write_raster(
            tmp_path / f"{name}.tif", ref, WIDTH, HEIGHT, count=4, dtype="uint16", seed=seed
        )
    # The third date on a 2x coarser grid, shifted by whole pixels.
    coarse = Affine(2.0, 0.0, ref.c - 4, 0.0, -2.0, ref.f + 6)
    write_raster(
        tmp_path / "d3.tif",
        coarse,
        WIDTH // 2 + 8,
        HEIGHT // 2 + 8,
        count=4,
        dtype="uint16",
        seed=3,
    )
    region = region_inside(ref, WIDTH, HEIGHT)
    return {"region": region, "labels": write_labels(tmp_path, region)}


def stack_config(
    tmp_path: Path,
    region: dict[str, float],
    labels: Path,
    image_format: str,
    *,
    stack: bool = True,
    staging: str = "stack",
    names: list[str] = DATES,
) -> MapcvConfig:
    return MapcvConfig.model_validate(
        {
            "region": region,
            "imagery": [
                {"type": "geotiff", "name": name, "path": str(tmp_path / f"{name}.tif")}
                for name in names
            ],
            "labels": {"path": str(labels), "label_field": "kind", "classes": {"a": 1}},
            "sampler": {"patch_size": PATCH, "edge_strategy": "drop"},
            "writer": {
                "staging_dir": str(tmp_path / staging),
                "image_format": image_format,
                "mask_format": "png",
                "stack_sources": stack,
            },
        }
    )


def _expected_stack(tmp_path: Path, manifest: Manifest, entry: Any) -> np.ndarray:
    return np.stack(
        [expected_on_patch_grid(tmp_path / f"{name}.tif", manifest, entry)[0] for name in DATES]
    )


def test_npy_stacks_are_time_channel_height_width(tmp_path: Path, dates: dict[str, Any]) -> None:
    config = stack_config(tmp_path, dates["region"], dates["labels"], "npy")
    manifest = run_generate(config).manifest
    assert manifest.writer is not None and manifest.writer["stack_sources"] is True
    assert [record.name for record in manifest.sources] == DATES
    staging = config.writer.staging_dir
    assert sorted(path.name for path in staging.iterdir() if path.is_dir()) == ["Images", "Masks"]
    for entry in manifest.patches:
        assert set(entry["files"]) == {"image", "mask"}
        stack = np.load(staging / entry["files"]["image"])
        assert stack.shape == (3, 4, PATCH, PATCH) and stack.dtype == np.uint16
        np.testing.assert_array_equal(stack, _expected_stack(tmp_path, manifest, entry))
    assert (
        manifest.patch_name(manifest.patches[0]) == Path(manifest.patches[0]["files"]["image"]).name
    )


def test_geotiff_stacks_name_every_band(tmp_path: Path, dates: dict[str, Any]) -> None:
    config = stack_config(tmp_path, dates["region"], dates["labels"], "tif")
    manifest = run_generate(config).manifest
    staging = config.writer.staging_dir
    for entry in manifest.patches[:5]:
        with rasterio.open(staging / entry["files"]["image"]) as patch:
            data = patch.read()
            names = list(patch.descriptions)
            assert patch.transform.almost_equals(Affine(*manifest.patch_transform(entry)))
        want = _expected_stack(tmp_path, manifest, entry)
        np.testing.assert_array_equal(data, want.reshape(12, PATCH, PATCH))
    bands = manifest.sources[0].bands
    assert names == [f"{date}_{band}" for date in DATES for band in bands]


def test_stacked_and_separate_datasets_have_the_same_masks(
    tmp_path: Path, dates: dict[str, Any]
) -> None:
    stacked = run_generate(stack_config(tmp_path, dates["region"], dates["labels"], "npy")).manifest
    separate_config = stack_config(
        tmp_path, dates["region"], dates["labels"], "npy", stack=False, staging="separate"
    )
    separate = run_generate(separate_config).manifest
    assert len(stacked.patches) == len(separate.patches)
    for a, b in zip(stacked.patches, separate.patches):
        assert a["summary"] == b["summary"]
        assert (tmp_path / "stack" / a["files"]["mask"]).read_bytes() == (
            tmp_path / "separate" / b["files"]["mask"]
        ).read_bytes()
        stack = np.load(tmp_path / "stack" / a["files"]["image"])
        for index, name in enumerate(DATES):
            np.testing.assert_array_equal(
                stack[index], np.load(tmp_path / "separate" / b["files"][name])
            )


def test_sources_that_cannot_share_an_array_are_refused(
    tmp_path: Path, dates: dict[str, Any]
) -> None:
    write_raster(
        tmp_path / "rgb.tif", reference_transform(), WIDTH, HEIGHT, count=3, dtype="uint8", seed=9
    )
    config = stack_config(
        tmp_path, dates["region"], dates["labels"], "npy", names=["d1", "rgb"], staging="bad"
    )
    with pytest.raises(ValueError, match="'d1' has 4 band\\(s\\) of uint16, 'rgb' 3 of uint8"):
        run_generate(config)
    assert not list((tmp_path / "bad").rglob("*.npy"))


def test_resume_reproduces_the_stacks(tmp_path: Path, dates: dict[str, Any]) -> None:
    config = stack_config(tmp_path, dates["region"], dates["labels"], "npy")
    full = run_generate(config).manifest
    staging = config.writer.staging_dir
    files = {path: path.read_bytes() for path in staging.rglob("*.npy")}
    cut = Manifest.load(staging / "manifest.json")
    cut.patches = cut.patches[: len(cut.patches) // 2]
    cut.save(staging / "manifest.json")
    assert run_generate(config).manifest.patches == full.patches
    assert {path: path.read_bytes() for path in staging.rglob("*.npy")} == files
    with pytest.raises(ValueError, match="writer"):
        run_generate(stack_config(tmp_path, dates["region"], dates["labels"], "npy", stack=False))


XYZ = {"type": "xyz", "zoom": 18, "source": "esri_satellite"}
BASE: dict[str, Any] = {
    "region": {"west": 4.9, "south": 52.3, "east": 4.91, "north": 52.31},
    "sampler": {"patch_size": 256},
}


@pytest.mark.parametrize(
    ("changes", "message"),
    [
        (
            {"imagery": XYZ, "writer": {"staging_dir": "o", "image_format": "tif"}},
            "write imagery as a list",
        ),
        (
            {
                "imagery": [{**XYZ, "name": "a"}],
                "writer": {"staging_dir": "o", "image_format": "tif"},
            },
            "write imagery as a list of two or more",
        ),
        (
            {
                "imagery": [{**XYZ, "name": "a"}, {**XYZ, "name": "b"}],
                "writer": {"staging_dir": "o", "image_format": "png"},
            },
            "needs writer.image_format npy",
        ),
        (
            {
                "task": "change",
                "labels": {"path": "c.geojson"},
                "imagery": [{**XYZ, "name": "a"}, {**XYZ, "name": "b"}],
                "writer": {"staging_dir": "o", "image_format": "tif"},
            },
            "applies to segmentation and regression datasets",
        ),
    ],
)
def test_config_refuses_stacking_it_cannot_do(changes: dict[str, Any], message: str) -> None:
    data = {**BASE, **changes}
    data["writer"] = {**data["writer"], "stack_sources": True}
    with pytest.raises(ValidationError) as raised:
        MapcvConfig.model_validate(data)
    assert message in str(raised.value)
