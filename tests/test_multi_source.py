"""Several imagery sources on one grid (``imagery`` as a list of named sources).

The first source's grid is the dataset's grid. Every other source must share its CRS
and have the same pixels or a whole number of them across, on a grid whose origin
falls on a pixel corner of the first; coarser pixels are repeated (nearest neighbour,
exact). Patches of every source are compared with what rasterio reads (or, for a
coarser source, with ``rasterio.warp.reproject(..., Resampling.nearest)`` onto the
patch's grid), and the masks with a single-source run of the first source.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import numpy.typing as npt
import pytest
from PIL import Image
from pydantic import ValidationError
from typer.testing import CliRunner

pytest.importorskip("rasterio", reason="multi-source tests write their rasters with rasterio")
import rasterio  # noqa: E402
import rasterio.warp  # noqa: E402
from pyproj import Transformer  # noqa: E402
from rasterio.crs import CRS  # noqa: E402
from rasterio.enums import Resampling  # noqa: E402
from rasterio.transform import Affine  # noqa: E402

from mapcv.cli import app  # noqa: E402
from mapcv.config import MapcvConfig  # noqa: E402
from mapcv.imagery import AlignedSource, GridAlignment, RasterMetadata, grid_alignment  # noqa: E402
from mapcv.manifest import Manifest, ManifestEntry, ManifestMismatchError  # noqa: E402
from mapcv.pipeline import run_generate  # noqa: E402
from mapcv.planning import plan  # noqa: E402

runner = CliRunner()

EPSG = 32631
CENTER_LON, CENTER_LAT = 3.0, 48.85
PATCH = 64


# ── Rasters on related grids ─────────────────────────────────────────────────


def _origin() -> Tuple[float, float]:
    """Top-left corner of the reference grid: a whole metre near (3E, 48.85N)."""
    x, y = Transformer.from_crs("EPSG:4326", f"EPSG:{EPSG}", always_xy=True).transform(
        CENTER_LON, CENTER_LAT
    )
    return float(round(x)) - 320.0, float(round(y)) + 256.0


def write_raster(
    path: Path,
    transform: Affine,
    width: int,
    height: int,
    *,
    count: int = 3,
    dtype: str = "uint8",
    seed: int = 1,
    epsg: int = EPSG,
) -> npt.NDArray[Any]:
    rng = np.random.default_rng(seed)
    if np.dtype(dtype).kind == "f":
        data = rng.uniform(1.0, 1000.0, size=(count, height, width)).astype(dtype)
    else:
        data = rng.integers(1, 250, size=(count, height, width)).astype(dtype)
    with rasterio.open(
        path,
        "w",
        driver="GTiff",
        height=height,
        width=width,
        count=count,
        dtype=dtype,
        crs=CRS.from_epsg(epsg),
        transform=transform,
        tiled=True,
        blockxsize=128,
        blockysize=128,
        compress="deflate",
    ) as dst:
        dst.write(data)
    return data


def reference_transform() -> Affine:
    x0, y0 = _origin()
    return Affine(1.0, 0.0, x0, 0.0, -1.0, y0)


def region_inside(
    transform: Affine, width: int, height: int, margin: float = 0.1
) -> Dict[str, float]:
    """A lon/lat box well inside the raster (the grid snaps outward from it)."""
    to_lonlat = Transformer.from_crs(f"EPSG:{EPSG}", "EPSG:4326", always_xy=True)
    x0, y0 = transform * (width * margin, height * (1 - margin))
    x1, y1 = transform * (width * (1 - margin), height * margin)
    west, south = to_lonlat.transform(x0, y0)
    east, north = to_lonlat.transform(x1, y1)
    return {"west": west, "south": south, "east": east, "north": north}


def write_labels(directory: Path, region: Dict[str, float]) -> Path:
    west, south = region["west"], region["south"]
    dx, dy = region["east"] - west, region["north"] - south

    def at(fx: float, fy: float) -> List[float]:
        return [west + fx * dx, south + fy * dy]

    ring = [at(0.2, 0.2), at(0.7, 0.25), at(0.6, 0.8), at(0.2, 0.2)]
    feature = {
        "type": "Feature",
        "properties": {"kind": "a"},
        "geometry": {"type": "Polygon", "coordinates": [ring]},
    }
    path = directory / "labels.geojson"
    path.write_text(json.dumps({"type": "FeatureCollection", "features": [feature]}))
    return path


def config_for(
    tmp_path: Path,
    imagery: Any,
    region: Dict[str, float],
    *,
    labels: Optional[Path] = None,
    staging: str = "dataset",
    image_format: str = "png",
    **sampler: Any,
) -> MapcvConfig:
    data: Dict[str, Any] = {
        "region": region,
        "imagery": imagery,
        "sampler": {"patch_size": PATCH, "mode": "grid", "edge_strategy": "drop", **sampler},
        "writer": {"staging_dir": str(tmp_path / staging), "image_format": image_format},
    }
    if labels is not None:
        data["labels"] = {"path": str(labels), "label_field": "kind", "classes": {"a": 1}}
    return MapcvConfig.model_validate(data)


def read_png(path: Path) -> npt.NDArray[Any]:
    """A written RGB patch as ``(bands, h, w)``."""
    return np.moveaxis(np.asarray(Image.open(path)), -1, 0)


def expected_on_patch_grid(
    path: Path, manifest: Manifest, entry: ManifestEntry
) -> Tuple[npt.NDArray[Any], npt.NDArray[np.bool_]]:
    """The file's pixels on the patch's grid, by rasterio nearest-neighbour reprojection.

    Same CRS and pixel corners on the patch grid: nearest neighbour picks, for every
    patch pixel, the file pixel containing its centre, which is what the pipeline
    must write. Pixels outside the file come back as zero and invalid.
    """
    patch = Affine(*manifest.patch_transform(entry))
    with rasterio.open(path) as src:
        data = src.read()
        out = np.zeros((src.count, PATCH, PATCH), dtype=data.dtype)
        rasterio.warp.reproject(
            data,
            out,
            src_transform=src.transform,
            src_crs=src.crs,
            dst_transform=patch,
            dst_crs=src.crs,
            resampling=Resampling.nearest,
        )
        # Which patch pixels have their centre inside the file.
        rows, cols = np.mgrid[0:PATCH, 0:PATCH]
        xs, ys = patch * (cols + 0.5, rows + 0.5)
        fc, fr = ~src.transform * (xs, ys)
        inside = (fc >= 0) & (fc < src.width) & (fr >= 0) & (fr < src.height)
    out[:, ~inside] = 0
    return out, inside


# ── Grid alignment ───────────────────────────────────────────────────────────


def _meta(transform: Tuple[float, ...], crs: str = "EPSG:32631") -> RasterMetadata:
    return RasterMetadata(
        source_type="geotiff",
        product_id="x",
        width=100,
        height=100,
        bands=["1"],
        dtype="uint8",
        crs=crs,
        transform=transform,  # type: ignore[arg-type]
        chunk_rows=64,
    )


REF = (10.0, 0.0, 500_000.0, 0.0, -10.0, 5_400_000.0)


def test_identical_grids_align_as_identity() -> None:
    alignment = grid_alignment(_meta(REF), _meta(REF), "b")
    assert alignment == GridAlignment(1, 0, 0)
    assert alignment.identity


def test_whole_pixel_offsets_and_coarser_pixels_align() -> None:
    # Origin 3 columns right and 2 rows down; pixels twice as large.
    other = (20.0, 0.0, 500_030.0, 0.0, -20.0, 5_399_980.0)
    assert grid_alignment(_meta(REF), _meta(other), "b") == GridAlignment(2, 2, 3)
    # Negative offsets: the other grid starts above and left of the first.
    other = (30.0, 0.0, 499_910.0, 0.0, -30.0, 5_400_050.0)
    assert grid_alignment(_meta(REF), _meta(other), "b") == GridAlignment(3, -5, -9)


def test_rotated_grids_align_when_they_share_the_rotation() -> None:
    rotated = Affine(*REF[:6]) * Affine.rotation(17)
    ref = tuple(rotated)[:6]
    coarse = tuple(rotated * Affine.translation(4, -6) * Affine.scale(2))[:6]
    assert grid_alignment(_meta(ref), _meta(coarse), "b") == GridAlignment(2, -6, 4)


def test_crs_spelled_differently_is_the_same_crs() -> None:
    wkt = CRS.from_epsg(EPSG).to_wkt()
    assert grid_alignment(_meta(REF), _meta(REF, crs=wkt), "b").identity


@pytest.mark.parametrize(
    ("other", "crs", "message"),
    [
        (REF, "EPSG:32632", "in EPSG:32632, but the first source is in EPSG:32631"),
        ((10.0, 0.0, 500_004.0, 0.0, -10.0, 5_400_000.0), None, "shifted by (0.0000, 0.4000)"),
        ((5.0, 0.0, 500_000.0, 0.0, -5.0, 5_400_000.0), None, "list the source with the finest"),
        ((15.0, 0.0, 500_000.0, 0.0, -15.0, 5_400_000.0), None, "not the first source's"),
        ((10.0, 0.0, 500_000.0, 0.0, 10.0, 5_400_000.0), None, "not the first source's"),
        ((20.0, 0.0, 500_000.0, 0.0, -10.0, 5_400_000.0), None, "not the first source's"),
    ],
)
def test_grids_that_do_not_line_up_are_refused(
    other: Tuple[float, ...], crs: Optional[str], message: str
) -> None:
    with pytest.raises(ValueError, match="imagery 'b'") as raised:
        grid_alignment(_meta(REF), _meta(other, crs=crs or "EPSG:32631"), "b")
    assert message in str(raised.value)


class ArraySource:
    """A source over an in-memory ``(h, w, c)`` array, valid where ``valid`` says."""

    def __init__(self, data: npt.NDArray[Any], valid: npt.NDArray[np.bool_]) -> None:
        self.data, self.valid = data, valid
        self.metadata = RasterMetadata(
            "geotiff",
            "x",
            data.shape[1],
            data.shape[0],
            ["1"],
            "uint8",
            "EPSG:32631",
            REF,
            64,
        )
        self.reads: List[Tuple[int, int, int, int]] = []

    def read_window(self, r0: int, r1: int, c0: int, c1: int) -> Any:
        assert 0 <= r0 < r1 <= self.data.shape[0] and 0 <= c0 < c1 <= self.data.shape[1]
        self.reads.append((r0, r1, c0, c1))
        return self.data[r0:r1, c0:c1].copy(), self.valid[r0:r1, c0:c1].copy()

    def close(self) -> None:
        pass


@pytest.mark.parametrize(
    "alignment",
    [
        GridAlignment(1, 0, 0),
        GridAlignment(1, 5, -3),
        GridAlignment(3, -4, 7),
        GridAlignment(2, 9, 0),
    ],
)
@pytest.mark.parametrize(
    "window", [(0, 40, 0, 40), (10, 61, 3, 50), (-20, 5, 30, 90), (55, 90, -9, 2)]
)
def test_aligned_reads_repeat_coarse_pixels_exactly(
    alignment: GridAlignment, window: Tuple[int, int, int, int]
) -> None:
    rng = np.random.default_rng(5)
    data = rng.integers(1, 250, size=(17, 23, 2)).astype(np.uint8)
    valid = rng.random((17, 23)) > 0.2
    source = ArraySource(data, valid)
    image, mask = AlignedSource(source, alignment).read_window(*window)

    r0, r1, c0, c1 = window
    k, ro, co = alignment.factor, alignment.row_offset, alignment.col_offset
    want = np.zeros((r1 - r0, c1 - c0, 2), dtype=np.uint8)
    want_valid = np.zeros((r1 - r0, c1 - c0), dtype=bool)
    for i, r in enumerate(range(r0, r1)):
        for j, c in enumerate(range(c0, c1)):
            sr, sc = (r - ro) // k, (c - co) // k
            if 0 <= sr < data.shape[0] and 0 <= sc < data.shape[1]:
                want[i, j] = data[sr, sc]
                want_valid[i, j] = valid[sr, sc]
    np.testing.assert_array_equal(image, want)
    np.testing.assert_array_equal(mask, want_valid)


def test_aligned_read_outside_the_source_is_empty() -> None:
    data = np.full((10, 10, 3), 7, dtype=np.uint8)
    source = ArraySource(data, np.ones((10, 10), dtype=bool))
    image, mask = AlignedSource(source, GridAlignment(1, 0, 0)).read_window(20, 30, 0, 5)
    assert image.shape == (10, 5, 3) and image.dtype == np.uint8
    assert not image.any() and not mask.any()


# ── Config ───────────────────────────────────────────────────────────────────


XYZ = {"type": "xyz", "zoom": 18, "source": "esri_satellite"}
BASE: Dict[str, Any] = {
    "region": {"west": 4.9, "south": 52.3, "east": 4.91, "north": 52.31},
    "sampler": {"patch_size": 256},
    "writer": {"staging_dir": "out"},
}


def test_config_accepts_a_list_of_named_sources() -> None:
    config = MapcvConfig.model_validate(
        {**BASE, "imagery": [{**XYZ, "name": "before"}, {**XYZ, "name": "after", "zoom": 17}]}
    )
    assert config.multi_source
    assert config.source_names == ["before", "after"]
    assert config.primary_imagery.zoom == 18  # type: ignore[union-attr]
    single = MapcvConfig.model_validate({**BASE, "imagery": XYZ})
    assert not single.multi_source
    assert single.source_names == ["image"]
    assert single.sources == [single.imagery]


@pytest.mark.parametrize(
    ("imagery", "message"),
    [
        ([XYZ], "imagery source 1 has no name"),
        ([{**XYZ, "name": "a"}, {**XYZ, "name": "a"}], "imagery name 'a' is used twice"),
        ([{**XYZ, "name": "After"}], "lowercase letters"),
        ([{**XYZ, "name": "a/b"}], "lowercase letters"),
        ([{**XYZ, "name": "mask"}], "imagery name 'mask' is reserved"),
        ([{**XYZ, "name": "rgb_world"}], "imagery name 'rgb_world' is reserved"),
        ({**XYZ, "name": "a"}, "imagery.name only applies when imagery is a list"),
        ([], "imagery is an empty list"),
        ([{**XYZ, "name": "a"}, {"name": "b", "zoom": 18}], "imagery.type is required"),
        (
            [{**XYZ, "name": "a"}, {"type": "eopf_zarr", "name": "s2", "path": "s3://b/x.zarr"}],
            "imagery 's2' (EOPF Zarr) requires writer.image_format='npy'",
        ),
    ],
)
def test_config_refuses_bad_source_lists(imagery: Any, message: str) -> None:
    with pytest.raises(ValidationError, match="imagery") as raised:
        MapcvConfig.model_validate({**BASE, "imagery": imagery})
    assert message in str(raised.value)


@pytest.mark.parametrize("task", ["detection", "instance", "classification"])
def test_tasks_with_one_image_per_patch_refuse_several_sources(task: str) -> None:
    with pytest.raises(ValidationError, match=f"task: {task} reads one imagery source"):
        MapcvConfig.model_validate(
            {
                **BASE,
                "task": task,
                "labels": {"path": "labels.geojson"},
                "imagery": [{**XYZ, "name": "a"}, {**XYZ, "name": "b"}],
            }
        )


def test_relative_paths_of_every_source_resolve_against_the_config(tmp_path: Path) -> None:
    folder = tmp_path / "project"
    folder.mkdir()
    (folder / "mapcv.yaml").write_text(
        "region: {west: 2.99, south: 48.84, east: 3.01, north: 48.86}\n"
        "imagery:\n"
        "  - {type: geotiff, name: a, path: a.tif}\n"
        "  - {type: geotiff, name: b, path: sub/b.tif}\n"
        "sampler: {patch_size: 64}\n"
        "writer: {staging_dir: out, image_format: tif}\n"
    )
    config = MapcvConfig.from_yaml(folder / "mapcv.yaml")
    paths = [source.path for source in config.sources]  # type: ignore[union-attr]
    assert paths == [str(folder / "a.tif"), str(folder / "sub" / "b.tif")]


# ── End to end ───────────────────────────────────────────────────────────────


@pytest.fixture
def scene(tmp_path: Path) -> Dict[str, Any]:
    """A reference raster, a co-registered one, and a 2x coarser one on a shifted grid."""
    ref = reference_transform()
    width, height = 448, 384
    write_raster(tmp_path / "a.tif", ref, width, height, seed=1)
    write_raster(tmp_path / "b.tif", ref, width, height, seed=2)
    # 2 m pixels starting 7 reference pixels left and 3 above the reference origin.
    coarse = Affine(2.0, 0.0, ref.c - 7, 0.0, -2.0, ref.f + 3)
    write_raster(tmp_path / "c.tif", coarse, width // 2 + 8, height // 2 + 8, seed=3)
    region = region_inside(ref, width, height)
    return {"region": region, "labels": write_labels(tmp_path, region)}


def _sources(tmp_path: Path, *names: str) -> List[Dict[str, Any]]:
    return [
        {"type": "geotiff", "name": name, "path": str(tmp_path / f"{name}.tif")} for name in names
    ]


def test_every_source_is_written_on_the_first_sources_grid(
    tmp_path: Path, scene: Dict[str, Any]
) -> None:
    config = config_for(
        tmp_path, _sources(tmp_path, "a", "b", "c"), scene["region"], labels=scene["labels"]
    )
    result = run_generate(config)
    manifest = result.manifest
    staging = config.writer.staging_dir
    assert len(manifest.patches) >= 12
    assert [record.name for record in manifest.sources] == ["a", "b", "c"]
    grid = manifest.sources[2].model_extra or {}
    assert grid["factor"] == 2
    # The offset is where c's window starts on a's window: from the two recorded transforms.
    a_x, a_y = manifest.sources[0].transform[2], manifest.sources[0].transform[5]  # type: ignore[index]
    c_x, c_y = manifest.sources[2].transform[2], manifest.sources[2].transform[5]  # type: ignore[index]
    assert grid["offset"] == [round(a_y - c_y), round(c_x - a_x)]
    assert not (manifest.sources[1].model_extra or {})
    for entry in manifest.patches:
        assert list(entry["files"]) == ["a", "b", "c", "mask"]
        stem = Path(entry["files"]["a"]).name
        assert entry["files"] == {
            "a": f"Images/a/{stem}",
            "b": f"Images/b/{stem}",
            "c": f"Images/c/{stem}",
            "mask": f"Masks/{stem}",
        }
        for name in ("a", "b", "c"):
            want, inside = expected_on_patch_grid(tmp_path / f"{name}.tif", manifest, entry)
            assert inside.all()
            np.testing.assert_array_equal(
                read_png(staging / entry["files"][name]),
                want,
                err_msg=f"{name} at {entry['row']},{entry['col']}",
            )
    # The footprint index has one feature per patch, georeferenced like the first source.
    footprints = json.loads((staging / "patches.geojson").read_text())
    assert len(footprints["features"]) == len(manifest.patches)


def test_masks_and_first_source_match_a_single_source_run(
    tmp_path: Path, scene: Dict[str, Any]
) -> None:
    multi = config_for(
        tmp_path, _sources(tmp_path, "a", "b", "c"), scene["region"], labels=scene["labels"]
    )
    single = config_for(
        tmp_path,
        {"type": "geotiff", "path": str(tmp_path / "a.tif")},
        scene["region"],
        labels=scene["labels"],
        staging="single",
    )
    many = run_generate(multi).manifest
    one = run_generate(single).manifest
    assert [(e["row"], e["col"]) for e in many.patches] == [
        (e["row"], e["col"]) for e in one.patches
    ]
    labeled = 0
    for a, b in zip(many.patches, one.patches):
        assert a["summary"] == b["summary"]
        assert (multi.writer.staging_dir / a["files"]["a"]).read_bytes() == (
            single.writer.staging_dir / b["files"]["image"]
        ).read_bytes()
        assert (multi.writer.staging_dir / a["files"]["mask"]).read_bytes() == (
            single.writer.staging_dir / b["files"]["mask"]
        ).read_bytes()
        labeled += int(a["summary"]["class_pixels"].get("1", 0) > 0)
    assert labeled > 0, "no patch holds the label: the comparison would prove nothing"
    assert many.sources[0].transform == one.sources[0].transform


def test_pixels_missing_in_any_source_have_no_imagery(tmp_path: Path) -> None:
    ref = reference_transform()
    width, height = 448, 384
    write_raster(tmp_path / "a.tif", ref, width, height, seed=1)
    # The second source covers only the left half of the first.
    write_raster(tmp_path / "half.tif", ref, width // 2, height, seed=2)
    region = region_inside(ref, width, height)
    labels = write_labels(tmp_path, region)
    config = config_for(
        tmp_path,
        _sources(tmp_path, "a", "half"),
        region,
        labels=labels,
        image_format="tif",
        max_empty_ratio=1.0,
    )
    with pytest.warns(UserWarning, match="region extends beyond the GeoTIFF 'half.tif'"):
        manifest = run_generate(config).manifest
    staging = config.writer.staging_dir
    partial = 0
    for entry in manifest.patches:
        want, inside = expected_on_patch_grid(tmp_path / "half.tif", manifest, entry)
        with rasterio.open(staging / entry["files"]["half"]) as patch:
            np.testing.assert_array_equal(patch.read(), want)
        mask = np.asarray(Image.open(staging / entry["files"]["mask"]))
        # Where the second source has no imagery the mask says so (ignore_index 255).
        assert (mask[~inside] == 255).all()
        assert entry["summary"]["empty_ratio"] == pytest.approx(1 - inside.mean())
        partial += int(0 < inside.mean() < 1)
    assert partial > 0, "no patch straddles the second source's edge"


def test_resume_reproduces_an_uninterrupted_run_and_refuses_changed_sources(
    tmp_path: Path, scene: Dict[str, Any]
) -> None:
    config = config_for(tmp_path, _sources(tmp_path, "a", "c"), scene["region"])
    full = run_generate(config).manifest
    staging = config.writer.staging_dir
    files = {path: path.read_bytes() for path in staging.rglob("*.png")}
    manifest_path = staging / "manifest.json"
    cut = Manifest.load(manifest_path)
    cut.patches = cut.patches[: len(cut.patches) // 2]
    cut.save(manifest_path)
    resumed = run_generate(config).manifest
    assert resumed.patches == full.patches
    assert {path: path.read_bytes() for path in staging.rglob("*.png")} == files

    renamed = config_for(
        tmp_path,
        [_sources(tmp_path, "a")[0], {**_sources(tmp_path, "c")[0], "name": "d"}],
        scene["region"],
    )
    with pytest.raises(ManifestMismatchError, match="sources"):
        run_generate(renamed)
    swapped = config_for(tmp_path, _sources(tmp_path, "a", "b"), scene["region"])
    with pytest.raises(ManifestMismatchError, match="sources|b: "):
        run_generate(swapped)


def test_a_misaligned_source_fails_before_anything_is_written(tmp_path: Path) -> None:
    ref = reference_transform()
    write_raster(tmp_path / "a.tif", ref, 448, 384, seed=1)
    shifted = Affine(1.0, 0.0, ref.c + 0.5, 0.0, -1.0, ref.f)
    write_raster(tmp_path / "b.tif", shifted, 448, 384, seed=2)
    config = config_for(tmp_path, _sources(tmp_path, "a", "b"), region_inside(ref, 448, 384))
    with pytest.raises(ValueError, match="imagery 'b' is on a grid shifted by"):
        run_generate(config)
    assert not list(config.writer.staging_dir.rglob("*.png"))


def test_sources_with_other_bands_and_dtypes_write_their_own_patches(tmp_path: Path) -> None:
    ref = reference_transform()
    write_raster(tmp_path / "a.tif", ref, 448, 384, seed=1)
    write_raster(tmp_path / "dem.tif", ref, 448, 384, count=1, dtype="float32", seed=4)
    config = config_for(
        tmp_path, _sources(tmp_path, "a", "dem"), region_inside(ref, 448, 384), image_format="npy"
    )
    manifest = run_generate(config).manifest
    assert manifest.sources[0].patch_shape == [3, PATCH, PATCH]
    assert manifest.sources[1].patch_shape == [1, PATCH, PATCH]
    assert manifest.sources[1].dtype == "float32"
    for entry in manifest.patches[:4]:
        dem = np.load(config.writer.staging_dir / entry["files"]["dem"])
        want, _ = expected_on_patch_grid(tmp_path / "dem.tif", manifest, entry)
        assert dem.dtype == np.float32
        np.testing.assert_array_equal(dem, want)


def test_plan_and_info_name_every_source(tmp_path: Path, scene: Dict[str, Any]) -> None:
    config = config_for(tmp_path, _sources(tmp_path, "a", "c"), scene["region"])
    single = config_for(
        tmp_path, {"type": "geotiff", "path": str(tmp_path / "a.tif")}, scene["region"]
    )
    estimate, one = plan(config), plan(single)
    assert estimate.patches == one.patches
    assert estimate.raster_px == one.raster_px
    assert estimate.output_bytes > one.output_bytes
    assert estimate.imagery.startswith("a: GeoTIFF") and "; c: GeoTIFF" in estimate.imagery

    run_generate(config)
    result = runner.invoke(app, ["info", str(config.writer.staging_dir)], env={"COLUMNS": "200"})
    assert result.exit_code == 0, result.output
    assert "Source a" in result.output and "Source c" in result.output
    assert "2× coarser" in result.output


# ── Smaller paths ────────────────────────────────────────────────────────────


def test_an_unreadable_crs_is_not_the_same_crs() -> None:
    with pytest.raises(ValueError, match="in not-a-crs, but the first source is in EPSG:32631"):
        grid_alignment(_meta(REF), _meta(REF, crs="not-a-crs"), "b")


def test_empty_reads_learn_the_shape_once() -> None:
    source = ArraySource(np.ones((4, 4, 2), dtype=np.uint16), np.ones((4, 4), dtype=bool))
    aligned = AlignedSource(source, GridAlignment(1, 0, 0))
    for _ in range(2):
        image, mask = aligned.read_window(10, 12, 0, 3)
        assert image.shape == (2, 3, 2) and image.dtype == np.uint16 and not mask.any()
    assert source.reads == [(0, 1, 0, 1)]
    aligned.close()


def test_a_source_that_fails_to_open_closes_the_ones_before_it(
    tmp_path: Path, scene: Dict[str, Any], monkeypatch: pytest.MonkeyPatch
) -> None:
    from mapcv import pipeline

    config = config_for(tmp_path, _sources(tmp_path, "a", "b"), scene["region"])
    closed: List[str] = []
    real_open = pipeline.open_raster_source  # type: ignore[attr-defined]

    def open_source(region: Any, imagery: Any, **kwargs: Any) -> Any:
        if imagery.name == "b":
            raise OSError("cannot read b")
        source = real_open(region, imagery, **kwargs)
        source.close = lambda: closed.append(imagery.name)  # type: ignore[method-assign]
        return source

    monkeypatch.setattr(pipeline, "open_raster_source", open_source)
    with pytest.raises(OSError, match="cannot read b"):
        run_generate(config)
    assert closed == ["a"]


def test_jpg_patches_and_world_files_of_every_source(tmp_path: Path, scene: Dict[str, Any]) -> None:
    data: Dict[str, Any] = {
        "region": scene["region"],
        "imagery": _sources(tmp_path, "a", "c"),
        "sampler": {"patch_size": PATCH, "mode": "grid", "edge_strategy": "drop"},
        "writer": {
            "staging_dir": str(tmp_path / "jpg"),
            "image_format": "jpg",
            "world_files": True,
        },
    }
    config = MapcvConfig.model_validate(data)
    manifest = run_generate(config).manifest
    entry = manifest.patches[0]
    stem = Path(entry["files"]["a"]).stem
    assert entry["files"] == {
        "a": f"Images/a/{stem}.jpg",
        "c": f"Images/c/{stem}.jpg",
        "a_world": f"Images/a/{stem}.jgw",
        "c_world": f"Images/c/{stem}.jgw",
    }
    for key in entry["files"]:
        assert (config.writer.staging_dir / entry["files"][key]).exists()
    # Both world files place the patch on the first source's grid.
    assert (config.writer.staging_dir / entry["files"]["a_world"]).read_text() == (
        config.writer.staging_dir / entry["files"]["c_world"]
    ).read_text()
    assert plan(config).output_bytes > 0


def test_writers_refuse_patches_they_cannot_store(tmp_path: Path) -> None:
    from mapcv.manifest import SourceRecord
    from mapcv.writer import WriterConfig, write_source_images
    from mapcv.writers import FilesWriter, create_writer
    from mapcv.targets import DetectionTarget
    from mapcv.config import DetectionOptions, LabelsConfig

    config = WriterConfig(staging_dir=tmp_path / "out")
    from mapcv.sampler import PatchMeta

    meta: List[PatchMeta] = [{"row": 0, "col": 0, "padded": False, "empty_ratio": 0.0}]
    manifest = Manifest()
    floats = np.zeros((1, 4, 4, 3), dtype=np.float32)
    assert (
        write_source_images(
            floats[:0], [], config, manifest, 0, 0, source=SourceRecord(name="b"), images_dir="x"
        )
        == []
    )
    with pytest.raises(ValueError, match="imagery 'b' gives float32"):
        write_source_images(
            floats, meta, config, manifest, 0, 0, source=SourceRecord(name="b"), images_dir="x"
        )
    images = np.zeros((1, 4, 4, 3), dtype=np.uint8)
    with pytest.raises(ValueError, match="made for one imagery source"):
        FilesWriter(config).write(images, None, meta, manifest, 0, others={"b": images})
    with pytest.raises(ValueError, match=r"expected patches of imagery \['b'\]"):
        FilesWriter(config, ["a", "b"]).write(images, None, meta, manifest, 0, others={"c": images})
    target = DetectionTarget(LabelsConfig(path=Path("x.geojson")), DetectionOptions(), 64)
    with pytest.raises(ValueError, match="task: detection writes one image per patch"):
        create_writer(config, target, ["a", "b"])


def test_cli_plans_validates_and_summarizes_several_sources(
    tmp_path: Path, scene: Dict[str, Any]
) -> None:
    import yaml

    data = {
        "region": scene["region"],
        "imagery": _sources(tmp_path, "a", "c")
        + [{"type": "geotiff", "name": "gone", "path": str(tmp_path / "missing.tif")}],
        "sampler": {"patch_size": PATCH, "edge_strategy": "drop"},
        "writer": {"staging_dir": str(tmp_path / "cli"), "image_format": "png"},
    }
    path = tmp_path / "multi.yaml"
    path.write_text(yaml.safe_dump(data))
    env = {"COLUMNS": "200"}
    result = runner.invoke(app, ["validate", str(path)], env=env)
    assert result.exit_code == 0, result.output
    assert "a: GeoTIFF" in result.output and "c: GeoTIFF" in result.output
    assert "imagery 'gone' path not found" in result.output

    data["imagery"] = data["imagery"][:2]
    path.write_text(yaml.safe_dump(data))
    result = runner.invoke(app, ["plan", str(path)], env=env)
    assert result.exit_code == 0, result.output
    assert "Several sources" in result.output
    result = runner.invoke(app, ["generate", str(path), "--yes"], env=env)
    assert result.exit_code == 0, result.output
    assert "Source a" in result.output and "Source c" in result.output
