"""Region boxes, area-of-interest polygons and vector labels given in 0..360° longitudes.

They are moved into -180..180 (a feature that crosses 180° is cut there), so a dataset
made from 0..360 input is byte-identical to one made from the same data in -180..180.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pytest
from pydantic import ValidationError
from shapely import make_valid
from shapely.affinity import translate
from shapely.geometry import LineString, MultiPolygon, Point, Polygon, box, mapping

pytest.importorskip("rasterio", reason="these tests write their rasters with rasterio")
import rasterio
from pyproj import Transformer
from rasterio.transform import from_origin

from mapcv.config import MapcvConfig, RegionConfig
from mapcv.labels import _wrap_longitudes, load_vector_labels
from mapcv.pipeline import run_generate

PATCH = 64


def _raster(path: Path, crs: str, lon: float, lat: float, res: float) -> Path:
    """A 192 x 192 three-band raster centred on (lon, lat) in ``crs``."""
    x, y = Transformer.from_crs("EPSG:4326", crs, always_xy=True).transform(lon, lat)
    data = np.random.default_rng(1).integers(1, 250, size=(3, 192, 192)).astype("uint8")
    with rasterio.open(
        path,
        "w",
        driver="GTiff",
        width=192,
        height=192,
        count=3,
        dtype="uint8",
        crs=crs,
        transform=from_origin(x - 96 * res, y + 96 * res, res, res),
    ) as dst:
        dst.write(data)
    return path


def _geojson(path: Path, geometries: list[Any]) -> Path:
    features = [
        {"type": "Feature", "properties": {"name": str(index)}, "geometry": mapping(geometry)}
        for index, geometry in enumerate(geometries, start=1)
    ]
    path.write_text(json.dumps({"type": "FeatureCollection", "features": features}))
    return path


def _generate(tmp_path: Path, name: str, region: dict[str, Any], image: Path, labels: Path) -> Path:
    out = tmp_path / name
    config = MapcvConfig.model_validate(
        {
            "region": region,
            "imagery": {"type": "geotiff", "path": str(image)},
            "labels": {"path": str(labels)},
            "sampler": {"patch_size": PATCH},
            "writer": {"staging_dir": str(out), "image_format": "png"},
        }
    )
    run_generate(config)
    return out


def _files(root: Path) -> dict[str, bytes]:
    return {
        str(path.relative_to(root)): path.read_bytes()
        for folder in ("Images", "Masks")
        for path in sorted((root / folder).iterdir())
    }


def _foreground(root: Path) -> int:
    from PIL import Image

    return sum(int((np.asarray(Image.open(p)) > 0).sum()) for p in (root / "Masks").iterdir())


# ── The region box ───────────────────────────────────────────────────────────


@pytest.mark.parametrize(
    ("west", "east", "expected"),
    [
        (199.5, 200.5, (-160.5, -159.5)),
        (180.0, 200.0, (-180.0, -160.0)),
        (300.0, 360.0, (-60.0, 0.0)),
        (10.0, 20.0, (10.0, 20.0)),
        (-170.0, 180.0, (-170.0, 180.0)),
    ],
)
def test_a_region_box_in_0_360_is_moved_into_minus_180_180(
    west: float, east: float, expected: tuple[float, float]
) -> None:
    region = RegionConfig(west=west, south=10.0, east=east, north=11.0)
    assert (region.west, region.east) == expected


@pytest.mark.parametrize(
    ("west", "east", "message"),
    [
        (
            170.0,
            190.0,
            (
                "crosses the antimeridian (180°), which is not supported; split it at 180° "
                "into two runs, one with west=170, east=180 and one with west=-180, east=-170"
            ),
        ),
        (-10.0, 200.0, "crosses the antimeridian"),
        (10.0, 370.0, "region.east must be a longitude in -180..180 (or 0..360 for both"),
        (-190.0, 10.0, "region.west must be a longitude in -180..180 (or 0..360 for both"),
        (200.0, 170.0, "region.west must be a longitude in -180..180"),
    ],
)
def test_a_region_box_that_cannot_be_moved_is_refused(
    west: float, east: float, message: str
) -> None:
    with pytest.raises(ValidationError) as caught:
        RegionConfig(west=west, south=10.0, east=east, north=11.0)
    assert message in str(caught.value)


# ── Labels and areas of interest ─────────────────────────────────────────────


@pytest.mark.parametrize(("crs", "res"), [("EPSG:4326", 0.0001), ("EPSG:3857", 10.0)])
def test_labels_and_region_in_0_360_give_the_same_dataset(
    tmp_path: Path, crs: str, res: float
) -> None:
    # On main the 0..360 labels left every mask empty on these grids.
    lon, lat = -160.0, 20.0
    image = _raster(tmp_path / "image.tif", crs, lon, lat, res)
    span = 0.004
    shapes = [
        Polygon([(lon - span, lat - span), (lon + span / 2, lat - span), (lon, lat + span)]),
        box(lon + span / 4, lat + span / 4, lon + span, lat + span),
    ]
    standard = _geojson(tmp_path / "standard.geojson", shapes)
    shifted = _geojson(tmp_path / "shifted.geojson", [translate(s, 360.0) for s in shapes])
    region = {"west": lon - 0.008, "south": lat - 0.008, "east": lon + 0.008, "north": lat + 0.008}
    region_360 = {**region, "west": region["west"] + 360.0, "east": region["east"] + 360.0}

    reference = _generate(tmp_path, "reference", region, image, standard)
    assert _foreground(reference) > 1000
    for name, box_, labels in (
        ("labels_360", region, shifted),
        ("region_360", region_360, standard),
        ("both_360", region_360, shifted),
    ):
        assert _files(_generate(tmp_path, name, box_, image, labels)) == _files(reference), name


def test_an_area_of_interest_file_in_0_360(tmp_path: Path) -> None:
    lon, lat = -160.0, 20.0
    image = _raster(tmp_path / "image.tif", "EPSG:4326", lon, lat, 0.0001)
    labels = _geojson(tmp_path / "labels.geojson", [box(lon - 0.003, lat - 0.003, lon, lat)])
    area = box(lon - 0.006, lat - 0.004, lon + 0.002, lat + 0.005)
    standard = _geojson(tmp_path / "aoi.geojson", [area])
    shifted = _geojson(tmp_path / "aoi_360.geojson", [translate(area, 360.0)])
    reference = _generate(tmp_path, "reference", {"path": str(standard)}, image, labels)
    other = _generate(tmp_path, "shifted", {"path": str(shifted)}, image, labels)
    assert _foreground(reference) > 0
    assert _files(other) == _files(reference)
    assert (other / "patches.geojson").read_bytes() == (reference / "patches.geojson").read_bytes()


@pytest.mark.parametrize("side", [1.0, -1.0])
def test_a_label_that_crosses_180_is_cut_there(tmp_path: Path, side: float) -> None:
    # One polygon over 179.99..180.01 in 0..360, against the same polygon as the
    # two-part MultiPolygon RFC 7946 asks for in -180..180, on a raster that ends at
    # 180° (side 1) and one that starts at -180° (side -1).
    lat = 20.0
    image = _raster(tmp_path / "image.tif", "EPSG:4326", side * 179.9904, lat, 0.0001)
    edges = sorted((side * 179.981, side * 180.0))
    region = {"west": edges[0], "south": lat - 0.009, "east": edges[1], "north": lat + 0.009}
    crossing = Polygon([(179.99, 19.995), (180.01, 19.998), (180.005, 20.006), (179.993, 20.004)])
    west = crossing.intersection(box(179.0, 19.0, 180.0, 21.0))
    east = translate(crossing.intersection(box(180.0, 19.0, 181.0, 21.0)), -360.0)
    split = MultiPolygon([west, east])
    reference = _generate(
        tmp_path, "split", region, image, _geojson(tmp_path / "split.geojson", [split])
    )
    assert _foreground(reference) > 1000
    crossing_file = _geojson(tmp_path / "crossing.geojson", [crossing])
    assert _files(_generate(tmp_path, "crossing", region, image, crossing_file)) == _files(
        reference
    )


def test_every_vector_format_reads_0_360(tmp_path: Path) -> None:
    square = box(200.0, 20.0, 200.01, 20.01)
    kml = tmp_path / "labels.kml"
    ring = " ".join(f"{x},{y}" for x, y in square.exterior.coords)
    kml.write_text(
        '<?xml version="1.0" encoding="UTF-8"?><kml xmlns="http://www.opengis.net/kml/2.2">'
        "<Document><Placemark><Polygon><outerBoundaryIs><LinearRing><coordinates>"
        f"{ring}</coordinates></LinearRing></outerBoundaryIs></Polygon></Placemark>"
        "</Document></kml>"
    )
    geojson = _geojson(tmp_path / "labels.geojson", [square])
    expected = translate(square, -360.0)
    for path in (kml, geojson):
        (geometry, class_id), *rest = load_vector_labels(path)[0]
        assert not rest and class_id == 1
        assert geometry.equals_exact(expected, 1e-12), path.name


def test_lines_and_points_are_moved_before_buffering(tmp_path: Path) -> None:
    line = LineString([(200.0, 20.0), (200.001, 20.0)])
    point = Point(200.0, 20.001)
    path = _geojson(tmp_path / "shifted.geojson", [line, point])
    standard = _geojson(
        tmp_path / "standard.geojson", [translate(line, -360.0), translate(point, -360.0)]
    )
    got = load_vector_labels(path, buffer=(4.0, 6.0))[0]
    want = load_vector_labels(standard, buffer=(4.0, 6.0))[0]
    assert len(got) == len(want) == 2
    for (a, _), (b, _) in zip(got, want):
        assert a.equals_exact(b, 1e-9)


def test_geometries_in_minus_180_180_are_left_as_they_are() -> None:
    geometries = [
        box(-180.0, 0.0, 180.0, 1.0),
        box(170.0, 0.0, 180.0, 1.0),
        Point(-170.0, 5.0),
        None,
        Polygon(),
        # Not 0..360 either (a vertex west of 0): left for the existing checks.
        LineString([(-10.0, 0.0), (190.0, 0.0)]),
    ]
    result = _wrap_longitudes(geometries)
    assert all(a is b for a, b in zip(result, geometries))


@pytest.mark.parametrize(
    ("coordinates", "note"),
    [([200.0, 20.0], "0..360 rather than -180..180"), ([500000.0, 20.0], "probably projected")],
)
def test_inspect_labels_tells_0_360_from_projected(
    tmp_path: Path, coordinates: list[float], note: str
) -> None:
    from mapcv.agent_tools import Sandbox, ToolState, inspect_labels

    _geojson(tmp_path / "l.geojson", [Point(*coordinates)])
    notes = inspect_labels(ToolState(Sandbox(tmp_path, False)), "l.geojson").data["notes"]
    assert any(note in text for text in notes), notes


@pytest.mark.parametrize(
    "polygon",
    [
        box(179.99, 20.0, 180.01, 20.01),
        MultiPolygon([box(179.99, 20.0, 180.0, 20.01), box(-180.0, 20.0, -179.99, 20.01)]),
    ],
    ids=["0_360", "rfc7946_split"],
)
def test_an_area_of_interest_across_180_is_refused(tmp_path: Path, polygon: Any) -> None:
    # Its box would span every longitude (a raster as wide as the world).
    path = _geojson(tmp_path / "aoi.geojson", [polygon])
    with pytest.raises(ValidationError, match=r"crosses the antimeridian \(180°\)"):
        RegionConfig.model_validate({"path": str(path)})


def test_an_area_of_interest_around_the_world_is_still_accepted(tmp_path: Path) -> None:
    path = _geojson(tmp_path / "aoi.geojson", [box(-180.0, 20.0, 180.0, 20.01)])
    region = RegionConfig.model_validate({"path": str(path)})
    assert (region.west, region.east) == (-180.0, 180.0)


def test_the_wizard_takes_a_box_in_0_360(tmp_path: Path) -> None:
    from test_cli_polish import _wizard

    # The first box crosses 180°; the second is the box east of it, in 0..360.
    answers = ["esri", "170,19.5,190,20.5", "199.5,19.5,200.5,20.5", "17", "", "256", "./ds", "n"]
    output, out = _wizard(tmp_path, answers)
    assert "crosses the antimeridian" in " ".join(output.split())
    region = MapcvConfig.from_yaml(out).region
    assert (region.west, region.east) == (-160.5, -159.5)


def test_a_self_intersecting_label_across_180_is_cut_without_an_error() -> None:
    bowtie = Polygon([(179.9, 0.0), (180.1, 1.0), (180.1, 0.0), (179.9, 1.0)])
    assert not bowtie.is_valid
    (moved,) = _wrap_longitudes([bowtie])
    assert moved is not None and moved.is_valid
    west, _, east, _ = moved.bounds
    assert (west, east) == (-180.0, 180.0)
    # Cut and moved, not changed: the area of the repaired bowtie (its two triangles).
    assert moved.area == pytest.approx(make_valid(bowtie).area, rel=1e-12)
