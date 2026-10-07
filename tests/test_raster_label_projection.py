"""Cross-CRS raster-label lookup: the projection of the pixel centres is bit for bit the
one a call per point gives.

PROJ transforms the points of an array one after another, so mapcv cuts the pixel
centres of a window into pieces and projects them in parallel threads. Labels must not
depend on that: a centre a hair from a label-pixel boundary falls on one side or the
other with the last bit of its coordinate. These tests project the same centres point by
point (``Transformer.transform(x, y)`` on scalars, which is what the first version of
the lookup did) and require ``np.array_equal`` on the projected coordinates, on the
fractional label pixels and on the label pixel indices, for several CRS pairs, for a
rotated grid, for operations PROJ picks per point, and for centres that lie exactly on
label-pixel boundaries.
"""

from __future__ import annotations

import os
import signal
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import numpy.typing as npt
import pytest

pytest.importorskip("rasterio", reason="the label rasters are written with rasterio")
pytest.importorskip("pyproj", reason="cross-CRS lookup projects with pyproj")
import rasterio
from pyproj import Transformer
from rasterio.crs import CRS
from rasterio.transform import Affine

from mapcv.config import RasterLabelsConfig
from mapcv.targets import raster_labels
from mapcv.targets.raster_labels import LabelRasterSampler

Window = tuple[float, float, float, float, float, float]
Array = npt.NDArray[np.float64]

HEIGHT, WIDTH = 288, 320  # 92,160 pixel centres: several pieces once the minimum is lowered


def utm(lon: float, lat: float, epsg: int = 32631) -> tuple[float, float]:
    x, y = Transformer.from_crs("EPSG:4326", f"EPSG:{epsg}", always_xy=True).transform(lon, lat)
    return float(x), float(y)


def rotated(window: Window, degrees: float) -> Window:
    a, b, c, d, e, f = window
    r = Affine(a, b, c, d, e, f) * Affine.rotation(degrees, (WIDTH / 2, HEIGHT / 2))
    return (r.a, r.b, r.c, r.d, r.e, r.f)


def window_at(x: float, y: float, pixel: float) -> Window:
    """A north-up window whose centre is (x, y)."""
    return (pixel, 0.0, x - WIDTH * pixel / 2, 0.0, -pixel, y + HEIGHT * pixel / 2)


# (imagery CRS, label CRS, window in the imagery CRS, label pixel size)
# Label pixel sizes are powers of two, so label-pixel boundaries are exact in floats.
_PARIS = utm(3.0, 48.85)
_MERCATOR = Transformer.from_crs("EPSG:4326", "EPSG:3857", always_xy=True).transform(3.0, 48.85)
_CASES: dict[str, tuple[str, int, Window, float]] = {
    "utm-to-wgs84": ("EPSG:32631", 4326, window_at(*_PARIS, 10.0), 2.0**-14),
    "utm-to-wgs84-rotated": (
        "EPSG:32631",
        4326,
        rotated(window_at(*_PARIS, 10.0), 17.0),
        2.0**-14,
    ),
    "mercator-to-utm": ("EPSG:3857", 32631, window_at(*_MERCATOR, 8.0), 8.0),
    "utm-to-mercator": ("EPSG:32631", 3857, window_at(*_PARIS, 8.0), 8.0),
    "utm-to-mercator-rotated": (
        "EPSG:32631",
        3857,
        rotated(window_at(*_PARIS, 8.0), -33.0),
        8.0,
    ),
    # Polar stereographic: the axes are far from north-up in most of the window.
    "antarctic-to-wgs84": (
        "EPSG:3031",
        4326,
        window_at(-1_500_000.0, 1_000_000.0, 100.0),
        2.0**-10,
    ),
    "arctic-to-utm": ("EPSG:3413", 32617, window_at(-200_000.0, -1_500_000.0, 100.0), 64.0),
    # A datum shift (Helmert) on top of the projections.
    "utm-to-lambert93": ("EPSG:32631", 2154, window_at(*_PARIS, 10.0), 8.0),
    # PROJ picks an operation per point from several that cover different areas.
    "etrs89-laea-to-utm": ("EPSG:3035", 32632, window_at(4_300_000.0, 3_100_000.0, 500.0), 128.0),
    "nad27-to-wgs84": ("EPSG:4267", 4326, (0.5, 0.0, -170.0, 0.0, -0.5, 75.0), 2.0**-4),
    "wgs84-to-nad27": ("EPSG:4326", 4267, (0.5, 0.0, -170.0, 0.0, -0.5, 75.0), 2.0**-4),
    "ed50-to-wgs84": ("EPSG:4230", 4326, (0.25, 0.0, -10.0, 0.0, -0.25, 70.0), 2.0**-4),
}


def label_raster(path: Path, epsg: int, transform: Affine) -> Path:
    """A tiny label raster: only its CRS and transform matter to the projection."""
    with rasterio.open(
        path,
        "w",
        driver="GTiff",
        height=8,
        width=8,
        count=1,
        dtype="uint8",
        crs=CRS.from_epsg(epsg),
        transform=transform,
    ) as dst:
        dst.write(np.ones((8, 8), dtype=np.uint8), 1)
    return path


def sampler_for(path: Path, imagery_crs: str) -> LabelRasterSampler:
    config = RasterLabelsConfig.model_validate(
        {"type": "raster", "path": str(path), "classes": {1: 1}}
    )
    return LabelRasterSampler(config, imagery_crs)


def centres(window: Window) -> tuple[Array, Array]:
    """Imagery CRS coordinates of every pixel centre, computed as the sampler does."""
    a, b, c, d, e, f = window
    u = np.arange(WIDTH, dtype=np.float64)[None, :] + 0.5
    v = np.arange(HEIGHT, dtype=np.float64)[:, None] + 0.5
    x, y = np.broadcast_arrays(a * u + b * v + c, d * u + e * v + f)
    return np.ascontiguousarray(x), np.ascontiguousarray(y)


def project_point_by_point(transformer: Transformer, x: Array, y: Array) -> tuple[Array, Array]:
    """The reference: one ``Transformer.transform`` call per point."""
    lx = np.empty(x.shape, dtype=np.float64)
    ly = np.empty(y.shape, dtype=np.float64)
    for index, (px, py) in enumerate(zip(x.ravel().tolist(), y.ravel().tolist())):
        qx, qy = transformer.transform(px, py)
        lx.flat[index], ly.flat[index] = qx, qy
    return lx, ly


def fractional_pixels(sampler: LabelRasterSampler, lx: Array, ly: Array) -> tuple[Array, Array]:
    """Fractional label (row, col) of projected coordinates, as the sampler computes them."""
    la, lb, lc, ld, le, lf = sampler.transform
    det = la * le - lb * ld
    return (la * (ly - lf) - ld * (lx - lc)) / det, (le * (lx - lc) - lb * (ly - lf)) / det


def label_grid_with_tie(
    lx: Array, ly: Array, pixel: float, pick: tuple[int, int, int, int]
) -> Affine:
    """A north-up label grid with ``pixel`` size (a power of two, so the arithmetic is
    exact) on which the projected centre of imagery pixel ``(row, col)`` lies exactly on
    the corner of label pixel ``(label_row, label_col)``; ``pick`` is the four numbers."""
    row, col, label_row, label_col = pick
    return Affine(
        pixel,
        0.0,
        float(lx[row, col]) - label_col * pixel,
        0.0,
        -pixel,
        float(ly[row, col]) + label_row * pixel,
    )


@pytest.fixture
def small_pieces(monkeypatch: pytest.MonkeyPatch) -> None:
    """Five uneven pieces of a window, whatever the machine's core count."""
    monkeypatch.setattr(raster_labels, "_PROJ_THREADS", 5)
    monkeypatch.setattr(raster_labels, "_MIN_PIECE", 1_000)


@pytest.mark.parametrize("name", list(_CASES))
@pytest.mark.parametrize("pieces", ["default", "uneven"])
def test_projected_centres_equal_the_per_point_projection(
    name: str, pieces: str, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    if pieces == "uneven":
        monkeypatch.setattr(raster_labels, "_PROJ_THREADS", 5)
        monkeypatch.setattr(raster_labels, "_MIN_PIECE", 1_000)
    imagery_crs, label_epsg, window, pixel = _CASES[name]
    x, y = centres(window)
    reference_transformer = Transformer.from_crs(imagery_crs, f"EPSG:{label_epsg}", always_xy=True)
    ref_x, ref_y = project_point_by_point(reference_transformer, x, y)

    transform = Affine(pixel, 0.0, 0.0, 0.0, -pixel, 0.0)
    sampler = sampler_for(label_raster(tmp_path / "labels.tif", label_epsg, transform), imagery_crs)
    got_x, got_y = sampler._project(x, y)
    assert got_x.shape == x.shape
    # np.array_equal on floats compares values; NaN and inf must match too.
    assert np.array_equal(got_x, ref_x, equal_nan=True)
    assert np.array_equal(got_y, ref_y, equal_nan=True)
    assert np.isfinite(ref_x).mean() > 0.9, "the case should project most of its points"


@pytest.mark.parametrize("name", list(_CASES))
def test_fractional_label_pixels_and_indices_equal_the_per_point_ones(
    name: str, tmp_path: Path, small_pieces: None
) -> None:
    imagery_crs, label_epsg, window, pixel = _CASES[name]
    x, y = centres(window)
    transformer = Transformer.from_crs(imagery_crs, f"EPSG:{label_epsg}", always_xy=True)
    ref_x, ref_y = project_point_by_point(transformer, x, y)

    # Label grids whose boundaries pass exactly through projected centres.
    label_transform = label_grid_with_tie(ref_x, ref_y, pixel, (100, 150, 37, 5))
    sampler = sampler_for(
        label_raster(tmp_path / "labels.tif", label_epsg, label_transform), imagery_crs
    )
    ref_rows, ref_cols = fractional_pixels(sampler, ref_x, ref_y)
    # The construction does put the first pick on a label-pixel boundary.
    assert (ref_rows[100, 150], ref_cols[100, 150]) == (37.0, 5.0)

    u = np.arange(WIDTH, dtype=np.float64)[None, :] + 0.5
    v = np.arange(HEIGHT, dtype=np.float64)[:, None] + 0.5
    rows, cols = sampler._fractional(window, u, v)
    assert np.array_equal(rows, ref_rows, equal_nan=True)
    assert np.array_equal(cols, ref_cols, equal_nan=True)

    size = 1 << 20  # a raster big enough that no finite index is out of range
    assert np.array_equal(sampler._indices(rows, size), sampler._indices(ref_rows, size))
    assert np.array_equal(sampler._indices(cols, size), sampler._indices(ref_cols, size))
    # The tie rule: a centre exactly on a boundary belongs to the right / lower pixel.
    assert sampler._indices(rows, size)[100, 150] == 37
    assert sampler._indices(cols, size)[100, 150] == 5


@pytest.mark.parametrize("name", ["utm-to-wgs84-rotated", "utm-to-mercator", "utm-to-lambert93"])
def test_sampled_masks_equal_the_per_point_lookup(
    name: str, tmp_path: Path, small_pieces: None
) -> None:
    """The whole lookup, with a centre on a label-pixel boundary, against the label
    pixels the per-point projection selects."""
    imagery_crs, label_epsg, window, pixel = _CASES[name]
    x, y = centres(window)
    transformer = Transformer.from_crs(imagery_crs, f"EPSG:{label_epsg}", always_xy=True)
    ref_x, ref_y = project_point_by_point(transformer, x, y)
    size = 1500
    transform = label_grid_with_tie(ref_x, ref_y, pixel, (100, 150, 700, 700))
    values = np.random.default_rng(4).integers(1, 4, size=(size, size), dtype=np.uint8)
    path = tmp_path / "labels.tif"
    with rasterio.open(
        path,
        "w",
        driver="GTiff",
        height=size,
        width=size,
        count=1,
        dtype="uint8",
        crs=CRS.from_epsg(label_epsg),
        transform=transform,
        tiled=True,
    ) as dst:
        dst.write(values, 1)
    config = RasterLabelsConfig.model_validate(
        {"type": "raster", "path": str(path), "classes": {1: 1, 2: 2, 3: 3}, "ignore_index": 255}
    )
    sampler = LabelRasterSampler(config, imagery_crs)

    ref_rows, ref_cols = fractional_pixels(sampler, ref_x, ref_y)
    rows, cols = sampler._indices(ref_rows, size), sampler._indices(ref_cols, size)
    inside = (rows >= 0) & (cols >= 0)
    assert inside.mean() > 0.99, "the label raster should cover the window"
    expected = np.full(rows.shape, 255, dtype=np.uint8)
    expected[inside] = values[rows[inside], cols[inside]]
    assert np.array_equal(sampler.sample(window, HEIGHT, WIDTH), expected)


# ── interpolated lookup ──────────────────────────────────────────────────────

# Windows where the projection is far from smooth, or not finite everywhere.
_HARD_CASES: dict[str, tuple[str, int, Window, float]] = {
    # Longitude jumps from +180 to -180 inside the window.
    "antimeridian": ("EPSG:32660", 4326, window_at(*utm(180.0, 60.0, 32660), 30.0), 2.0**-12),
    # Longitude turns around the pole, which is inside the window.
    "south-pole": ("EPSG:3031", 4326, window_at(0.0, 0.0, 50.0), 2.0**-8),
    # Far outside UTM zone 31: points PROJ cannot project come back as inf.
    "outside-utm": ("EPSG:4326", 32631, (0.25, 0.0, 60.0, 0.0, -0.25, 40.0), 4096.0),
}


def exact_indices(
    sampler: LabelRasterSampler, window: Window, start: int, stop: int, width: int
) -> tuple[npt.NDArray[np.int64], npt.NDArray[np.int64]]:
    u = np.arange(width, dtype=np.float64)[None, :] + 0.5
    v = np.arange(start, stop, dtype=np.float64)[:, None] + 0.5
    rows, cols = sampler._fractional(window, u, v)
    return sampler._indices(rows, sampler.info.height), sampler._indices(cols, sampler.info.width)


def big_label_raster(path: Path, epsg: int, transform: Affine, size: int) -> Path:
    """A label raster of ``size`` x ``size`` pixels, written as a single sparse tile."""
    with rasterio.open(
        path,
        "w",
        driver="GTiff",
        height=size,
        width=size,
        count=1,
        dtype="uint8",
        crs=CRS.from_epsg(epsg),
        transform=transform,
        tiled=True,
        blockxsize=1024,
        blockysize=1024,
        sparse_ok=True,
    ):
        pass
    return path


@pytest.mark.parametrize("name", list(_CASES) + list(_HARD_CASES))
@pytest.mark.parametrize("finer", [1, 64])
def test_interpolated_label_pixels_equal_the_exact_ones(
    name: str, finer: int, tmp_path: Path
) -> None:
    """Every centre gets the label pixel the exact projection gives, on grids as coarse
    as the imagery and 64 times finer, with centres exactly on label-pixel corners."""
    imagery_crs, label_epsg, window, pixel = {**_CASES, **_HARD_CASES}[name]
    x, y = centres(window)
    transformer = Transformer.from_crs(imagery_crs, f"EPSG:{label_epsg}", always_xy=True)
    lx, ly = transformer.transform(x, y)
    pick = (144, 160, 1 << 14, 1 << 14)
    if not (np.isfinite(lx[144, 160]) and np.isfinite(ly[144, 160])):
        pick = (0, 0, 1 << 14, 1 << 14)
    transform = label_grid_with_tie(lx, ly, pixel / finer, pick)
    size = 1 << 15
    sampler = sampler_for(
        big_label_raster(tmp_path / "labels.tif", label_epsg, transform, size), imagery_crs
    )
    for start, stop, width in ((0, HEIGHT, WIDTH), (5, 38, 33), (17, 19, WIDTH), (0, HEIGHT, 2)):
        rows, cols = sampler._interpolated_pixels(window, start, stop, width)
        ref_rows, ref_cols = exact_indices(sampler, window, start, stop, width)
        assert np.array_equal(rows, ref_rows), (start, stop, width)
        assert np.array_equal(cols, ref_cols), (start, stop, width)


def test_centres_on_label_pixel_boundaries_are_projected_exactly(tmp_path: Path) -> None:
    """A label grid on which a whole row of centres lies on boundaries: the tie rule
    holds for each, as the exact projection gives it."""
    imagery_crs, label_epsg, window, pixel = _CASES["utm-to-mercator"]
    x, y = centres(window)
    lx, ly = Transformer.from_crs(imagery_crs, f"EPSG:{label_epsg}", always_xy=True).transform(x, y)
    size = 1 << 15
    for pick in [(r, c, 1 << 14, 1 << 14) for r in (0, 16, 31, HEIGHT - 1) for c in (0, 16, 77)]:
        transform = label_grid_with_tie(lx, ly, pixel / 64, pick)
        path = big_label_raster(
            tmp_path / f"labels-{pick[0]}-{pick[1]}.tif", label_epsg, transform, size
        )
        sampler = sampler_for(path, imagery_crs)
        rows, cols = sampler._interpolated_pixels(window, 0, HEIGHT, WIDTH)
        assert (rows[pick[0], pick[1]], cols[pick[0], pick[1]]) == (1 << 14, 1 << 14)
        ref_rows, ref_cols = exact_indices(sampler, window, 0, HEIGHT, WIDTH)
        assert np.array_equal(rows, ref_rows) and np.array_equal(cols, ref_cols)


@pytest.mark.parametrize(
    ("name", "most"), [("utm-to-wgs84", 0.05), ("utm-to-lambert93", 0.05), ("antimeridian", 1.0)]
)
def test_smooth_projections_project_few_centres(
    name: str, most: float, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    imagery_crs, label_epsg, window, pixel = {**_CASES, **_HARD_CASES}[name]
    transform = Affine(pixel, 0.0, 0.0, 0.0, -pixel, 0.0)
    sampler = sampler_for(label_raster(tmp_path / "labels.tif", label_epsg, transform), imagery_crs)
    projected = []
    fractional = sampler._fractional

    def count(*args: Any) -> Any:
        projected.append(np.broadcast(*args[1:]).size)
        return fractional(*args)

    monkeypatch.setattr(sampler, "_fractional", count)
    sampler._interpolated_pixels(window, 0, HEIGHT, WIDTH)
    assert sum(projected) <= most * HEIGHT * WIDTH
    # The antimeridian cells are projected whole, the others interpolated.
    assert sum(projected) > 0.01 * HEIGHT * WIDTH or name != "antimeridian"


def test_operations_picked_per_point_are_never_interpolated(tmp_path: Path) -> None:
    imagery_crs, label_epsg, _, pixel = _CASES["nad27-to-wgs84"]
    transform = Affine(pixel, 0.0, 0.0, 0.0, -pixel, 0.0)
    path = label_raster(tmp_path / "labels.tif", label_epsg, transform)
    assert not sampler_for(path, imagery_crs)._interpolate
    assert sampler_for(path, "EPSG:32631")._interpolate


def test_small_arrays_are_projected_on_the_calling_thread(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    def no_pool() -> Any:
        raise AssertionError("a thread pool for a few points")

    monkeypatch.setattr(raster_labels, "_proj_executor", no_pool)
    imagery_crs, label_epsg, window, pixel = _CASES["utm-to-wgs84"]
    sampler = sampler_for(
        label_raster(
            tmp_path / "labels.tif", label_epsg, Affine(pixel, 0.0, 0.0, 0.0, -pixel, 0.0)
        ),
        imagery_crs,
    )
    x, y = centres(window)
    reference = Transformer.from_crs(imagery_crs, f"EPSG:{label_epsg}", always_xy=True)
    got_x, got_y = sampler._project(x[:3, :3], y[:3, :3])
    ref_x, ref_y = project_point_by_point(reference, x[:3, :3], y[:3, :3])
    assert np.array_equal(got_x, ref_x) and np.array_equal(got_y, ref_y)


def test_forgetting_the_pool_lets_the_next_projection_create_one(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(raster_labels, "_proj_pool", object())
    raster_labels._forget_proj_pool()
    assert raster_labels._proj_pool is None


@pytest.mark.skipif(not hasattr(os, "fork"), reason="needs os.fork (POSIX)")
@pytest.mark.skipif(
    sys.platform == "darwin",
    reason="a forked child of a multi-threaded process crashed (SIGSEGV) on macOS CI; "
    "fork is not supported there after threads exist and the default start method is spawn",
)
@pytest.mark.filterwarnings("ignore:This process .* is multi-threaded:DeprecationWarning")
def test_a_forked_child_projects_with_a_fresh_pool(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A child forked after the thread pool was used must not wait for the parent's
    (vanished) worker threads: it gets a pool of its own and the same coordinates."""
    monkeypatch.setattr(raster_labels, "_PROJ_THREADS", 4)
    monkeypatch.setattr(raster_labels, "_MIN_PIECE", 1_000)
    imagery_crs, label_epsg, window, pixel = _CASES["utm-to-wgs84"]
    sampler = sampler_for(
        label_raster(
            tmp_path / "labels.tif", label_epsg, Affine(pixel, 0.0, 0.0, 0.0, -pixel, 0.0)
        ),
        imagery_crs,
    )
    x, y = centres(window)
    parent_x, parent_y = sampler._project(x, y)
    assert raster_labels._proj_pool is not None, "the parent should have used the pool"

    result = tmp_path / "child.npy"
    pid = os.fork()
    if pid == 0:  # the child: never returns into pytest
        status = 1
        try:
            child_x, child_y = sampler._project(x, y)
            np.save(result, np.stack([child_x, child_y]))
            status = 0
        finally:
            os._exit(status)
    deadline = time.monotonic() + 30
    finished = 0
    while finished == 0 and time.monotonic() < deadline:
        finished, status = os.waitpid(pid, os.WNOHANG)
        time.sleep(0.05)
    if finished == 0:
        os.kill(pid, signal.SIGKILL)
        os.waitpid(pid, 0)
        pytest.fail("the forked child did not finish in 30 s: it waits for a dead thread pool")
    assert os.waitstatus_to_exitcode(status) == 0
    child_x, child_y = np.load(result)
    assert np.array_equal(child_x, parent_x) and np.array_equal(child_y, parent_y)
