"""Property-based tests (Hypothesis): tile math against mercantile, the rasterizer against
rasterio (GDAL), and the invariants of the sampler and the splitter on generated inputs.

Examples are derandomized (see conftest.py), so CI runs are reproducible; run with
``--hypothesis-profile=explore`` to search with fresh random examples.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

pytest.importorskip("hypothesis", reason="property tests need hypothesis (test group)")
mercantile = pytest.importorskip("mercantile")

from hypothesis import assume, given, settings
from hypothesis import strategies as st

from mapcv import _mapcv_rs
from mapcv.manifest import Manifest, ManifestEntry, PatchSummary, TargetRecord
from mapcv.splitter import SplitterConfig, split_manifest

LAT_LIMIT = 85.0511287798066
lngs = st.floats(-180.0, 180.0, allow_nan=False, exclude_max=True)
lats = st.floats(-LAT_LIMIT + 1e-9, LAT_LIMIT - 1e-9, allow_nan=False)
zooms = st.integers(0, 22)


def _off_boundary(value: float, scale: float) -> bool:
    """Whether ``value * scale`` is clear of an integer: points on a tile edge may
    round either way in two correct implementations."""
    scaled = value * scale
    return abs(scaled - round(scaled)) > 1e-6


# ── Tile math against mercantile ─────────────────────────────────────────────


@given(lngs, lats, zooms)
def test_tile_matches_mercantile(lng: float, lat: float, zoom: int) -> None:
    expected = mercantile.tile(lng, lat, zoom)
    x_frac = (lng + 180.0) / 360.0
    sin = math.sin(math.radians(lat))
    y_frac = 0.5 - math.log((1 + sin) / (1 - sin)) / (4 * math.pi)
    assume(_off_boundary(x_frac, 2**zoom) and _off_boundary(y_frac, 2**zoom))
    found = _mapcv_rs.tile(lng, lat, zoom)
    assert (found.x, found.y, found.z) == (expected.x, expected.y, expected.z)


@given(lngs, lats)
def test_xy_matches_mercantile(lng: float, lat: float) -> None:
    x, y = _mapcv_rs.xy(lng, lat)
    ex, ey = mercantile.xy(lng, lat)
    assert x == pytest.approx(ex, rel=1e-12, abs=1e-6)
    assert y == pytest.approx(ey, rel=1e-12, abs=1e-6)


@given(st.data(), zooms)
def test_bounds_match_mercantile(data: st.DataObject, zoom: int) -> None:
    x = data.draw(st.integers(0, 2**zoom - 1))
    y = data.draw(st.integers(0, 2**zoom - 1))
    found = _mapcv_rs.bounds(x, y, zoom)
    expected = mercantile.bounds(x, y, zoom)
    assert (found.west, found.south, found.east, found.north) == pytest.approx(
        tuple(expected), abs=1e-9
    )
    xy_found = _mapcv_rs.xy_bounds(x, y, zoom)
    xy_expected = mercantile.xy_bounds(x, y, zoom)
    assert (xy_found.west, xy_found.south, xy_found.east, xy_found.north) == pytest.approx(
        tuple(xy_expected), abs=1e-6
    )


@given(
    lngs,
    lats,
    st.floats(1e-6, 0.5),
    st.floats(1e-6, 0.5),
    st.integers(0, 14),
)
def test_tiles_match_mercantile(
    west: float, south: float, width: float, height: float, zoom: int
) -> None:
    east = min(west + width, 179.999999)
    north = min(south + height, LAT_LIMIT - 1e-9)
    assume(east > west and north > south)
    expected = {(t.x, t.y) for t in mercantile.tiles(west, south, east, north, [zoom])}
    corners = [(west, south), (east, north)]
    for lng, lat in corners:
        x_frac = (lng + 180.0) / 360.0
        sin = math.sin(math.radians(lat))
        y_frac = 0.5 - math.log((1 + sin) / (1 - sin)) / (4 * math.pi)
        assume(_off_boundary(x_frac, 2**zoom) and _off_boundary(y_frac, 2**zoom))
    found = {(t.x, t.y) for t in _mapcv_rs.tiles(west, south, east, north, [zoom])}
    assert found == expected


# ── Rasterizer against rasterio ──────────────────────────────────────────────


@st.composite
def star_polygons(draw: st.DrawFn, height: int, width: int) -> list[tuple[float, float]]:
    """A simple (star-shaped) polygon in pixel coordinates, possibly past the edges."""
    cx = draw(st.floats(-0.25 * width, 1.25 * width))
    cy = draw(st.floats(-0.25 * height, 1.25 * height))
    count = draw(st.integers(3, 12))
    angles = sorted(
        draw(
            st.lists(
                st.floats(0, 2 * math.pi, exclude_max=True),
                min_size=count,
                max_size=count,
                unique=True,
            )
        )
    )
    radii = draw(st.lists(st.floats(0.3, 0.6 * max(height, width)), min_size=count, max_size=count))
    return [(cx + r * math.cos(a), cy + r * math.sin(a)) for a, r in zip(angles, radii)]


@settings(max_examples=150)
@given(st.data(), st.booleans())
def test_rasterize_matches_rasterio(data: st.DataObject, all_touched: bool) -> None:
    pytest.importorskip("rasterio")
    from rasterio.features import rasterize as rio_rasterize
    from rasterio.transform import Affine
    from shapely.geometry import Polygon

    height = data.draw(st.integers(1, 40))
    width = data.draw(st.integers(1, 40))
    size = data.draw(st.sampled_from([0.5, 1.0, 2.5, 10.0]))
    x0 = data.draw(st.floats(-1e5, 1e5))
    y0 = data.draw(st.floats(-1e5, 1e5))
    transform = Affine(size, 0.0, x0, 0.0, -size, y0)
    count = data.draw(st.integers(1, 3))
    polygons = []
    for index in range(count):
        ring_px = data.draw(star_polygons(height, width))
        ring = [transform * (col, row) for col, row in ring_px]
        assume(Polygon(ring).is_valid and Polygon(ring).area > 0)
        polygons.append((ring, index + 1))

    found = _mapcv_rs.rasterize(
        [([ring], class_id) for ring, class_id in polygons],
        height,
        width,
        tuple(transform)[:6],
        all_touched,
    )
    expected = rio_rasterize(
        [(Polygon(ring), class_id) for ring, class_id in polygons],
        out_shape=(height, width),
        transform=transform,
        fill=0,
        all_touched=all_touched,
        dtype="uint8",
    )
    if all_touched:
        # A polygon that meets a pixel only along its edge (or overlaps it by a sliver far
        # below a pixel) is a tie: whether it "touches" depends on the last bit of rounding,
        # and GDAL's own answer differs between platforms (x86-64 and arm64 macOS disagree).
        # Such pixels may differ; every other pixel must match.
        from shapely.geometry import box

        for row, col in zip(*np.nonzero(found != expected)):
            left, top = x0 + col * size, y0 - row * size
            pixel = box(left, top - size, left + size, top)
            assert any(
                Polygon(ring).intersects(pixel)
                and Polygon(ring).intersection(pixel).area < 1e-9 * size**2
                for ring, _ in polygons
            ), f"pixel ({row}, {col}) differs and is not an edge tie"
        found = np.where(found != expected, expected, found)
    np.testing.assert_array_equal(found, expected)


def test_rasterize_counts_an_edge_contact_as_touching() -> None:
    """A triangle that meets the raster only along the top edge of its pixel: with
    all_touched mapcv burns the pixel, as GDAL does on x86-64 (GDAL on arm64 macOS does not;
    see the tie rule above). Found by hypothesis on macOS CI."""
    transform = (2.5, 0.0, 0.0, 0.0, -2.5, -33.0)
    ring = [(1.25, -33.0), (0.9375, -33.0), (0.3545777318290328, -31.801344656671077)]
    assert _mapcv_rs.rasterize([([ring], 1)], 1, 1, transform, True).tolist() == [[1]]
    assert _mapcv_rs.rasterize([([ring], 1)], 1, 1, transform, False).tolist() == [[0]]


# ── Sampler invariants ───────────────────────────────────────────────────────

dims = st.integers(1, 300)
edges = st.sampled_from(["pad", "drop", "shift"])


@given(dims, dims, st.integers(1, 128), st.integers(1, 160), edges)
def test_grid_anchors(height: int, width: int, patch: int, stride: int, edge: str) -> None:
    anchors = _mapcv_rs.grid_sample_anchors(height, width, patch, stride, edge)
    assert len(set(anchors)) == len(anchors)
    assert anchors == sorted(anchors)
    for row, col in anchors:
        assert 0 <= row < height and 0 <= col < width
        if edge != "pad" and height >= patch and width >= patch:
            assert row + patch <= height and col + patch <= width
    if edge == "drop" and (height < patch or width < patch):
        assert anchors == []
    if edge != "drop" and stride <= patch:
        # Every pixel is inside some patch.
        rows = {r for r, _ in anchors}
        cols = {c for _, c in anchors}
        assert _covered(rows, patch, height) and _covered(cols, patch, width)
        assert len(anchors) == len(rows) * len(cols)


def _covered(starts: set[int], patch: int, size: int) -> bool:
    covered = np.zeros(size, dtype=bool)
    for start in starts:
        covered[start : start + patch] = True
    return bool(covered.all())


@given(dims, dims, st.integers(1, 64), st.integers(0, 500), st.integers(0, 2**64 - 1), edges)
def test_random_anchors(
    height: int, width: int, patch: int, count: int, seed: int, edge: str
) -> None:
    anchors = _mapcv_rs.random_sample_anchors(height, width, patch, count, seed, edge)
    capacity = _mapcv_rs.random_anchor_capacity(height, width, patch, edge)
    assert len(anchors) == min(count, capacity)
    assert len(set(anchors)) == len(anchors)
    assert anchors == _mapcv_rs.random_sample_anchors(height, width, patch, count, seed, edge)

    # Per axis: every start whose patch fits, or one padded start at 0 on an axis
    # shorter than the patch (none with "drop").
    def axis(size: int) -> int:
        return size - patch + 1 if size >= patch else (0 if edge == "drop" else 1)

    assert capacity == axis(height) * axis(width)
    for row, col in anchors:
        assert 0 <= row <= max(0, height - patch) and 0 <= col <= max(0, width - patch)


# ── Splitter invariants ──────────────────────────────────────────────────────


def _manifest(positions: list[tuple[int, int]], patch: int, stride: int) -> Manifest:
    manifest = Manifest(
        target=TargetRecord(type="segmentation", class_map={"a": 1}),
        sampler={"patch_size": patch, "stride": stride},
    )
    manifest.patches = [
        ManifestEntry(
            row=row,
            col=col,
            padded=False,
            chunk=0,
            files={"image": f"Images/patch_{index:07d}.png"},
            summary=PatchSummary(class_pixels={"0": 5, "1": index % 3}, empty_ratio=0.0),
        )
        for index, (row, col) in enumerate(positions)
    ]
    return manifest


def _overlap(a: ManifestEntry, b: ManifestEntry, patch: int) -> bool:
    return abs(a["row"] - b["row"]) < patch and abs(a["col"] - b["col"]) < patch


@settings(max_examples=60)
@given(
    st.integers(1, 12),
    st.integers(1, 12),
    st.sampled_from([4, 8]),
    st.sampled_from([0.5, 1.0]),
    st.sampled_from(["spatial", "random", "stratified"]),
    st.floats(0.0, 0.5),
    st.floats(0.0, 0.5),
    st.integers(0, 10**6),
    st.sampled_from([None, 4, 16]),
)
def test_split_invariants(
    tmp_path_factory: pytest.TempPathFactory,
    rows: int,
    cols: int,
    patch: int,
    stride_fraction: float,
    strategy: str,
    test_ratio: float,
    val_ratio: float,
    seed: int,
    block: int | None,
) -> None:
    stride = int(patch * stride_fraction)
    positions = [(r * stride, c * stride) for r in range(rows) for c in range(cols)]
    manifest = _manifest(positions, patch, stride)
    config = SplitterConfig(
        test_ratio=test_ratio,
        val_ratio=val_ratio,
        seed=seed,
        strategy=strategy,  # type: ignore[arg-type]
        block_size=block if strategy == "spatial" else None,
        labeled_ratios=[0.3],
    )
    out = tmp_path_factory.mktemp("splits")
    import warnings

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        counts, lists = split_manifest(manifest, config, out)
        again, lists_again = split_manifest(manifest, config, tmp_path_factory.mktemp("again"))
    assert lists == lists_again and counts == again  # deterministic

    train, val, test = set(lists.train), set(lists.val), set(lists.test)
    names = {manifest.patch_name(entry) for entry in manifest.patches}
    assert not (train & val) and not (train & test) and not (val & test)
    assert train | val | test <= names
    assert counts["train"] + counts["val"] + counts["test"] + counts["dropped"] == len(names)

    labeled = (out / "30" / "labeled.txt").read_text().split()
    unlabeled = (out / "30" / "unlabeled.txt").read_text().split()
    assert sorted(labeled + unlabeled) == sorted(lists.train)
    assert len(labeled) == math.ceil(len(lists.train) * 0.3)

    if strategy == "spatial":
        by_name = {manifest.patch_name(entry): entry for entry in manifest.patches}
        for name in train | val:
            for held in test:
                assert not _overlap(by_name[name], by_name[held], patch)
        for name in train:
            for held in val:
                assert not _overlap(by_name[name], by_name[held], patch)
    else:
        assert counts["dropped"] == 0
    if strategy == "random":
        assert len(test) == math.ceil(len(names) * test_ratio)
        assert len(val) == math.ceil((len(names) - len(test)) * val_ratio)
