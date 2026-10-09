"""``labels.osm``: labels from OpenStreetMap through a (local) Overpass API.

The server answers with buildings, roads, a roundabout and a water multipolygon whose
outer ring is split over two ways and has an island. The features must get the right
classes and shapes, the answer must be cached (one request), and a generated mask must
equal rasterio's rasterization of the expected polygons.
"""

from __future__ import annotations

import json
import threading
import time
import urllib.parse
from collections.abc import Iterator
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any

import numpy as np
import pytest
from pydantic import ValidationError
from shapely import make_valid
from shapely.geometry import Polygon, shape

from mapcv.config import LabelsConfig, MapcvConfig, OsmClass, OsmLabelsSource
from mapcv.osm import ATTRIBUTION, features_from_overpass, osm_labels_file, overpass_query

pytest.importorskip("rasterio", reason="masks are compared with rasterio")
from test_multi_source import PATCH, reference_transform, region_inside, write_raster

WIDTH, HEIGHT = 320, 256


def _ll(points: list[tuple[float, float]]) -> list[dict[str, float]]:
    return [{"lon": lon, "lat": lat} for lon, lat in points]


def _rect(west: float, south: float, east: float, north: float) -> list[tuple[float, float]]:
    return [(west, south), (east, south), (east, north), (west, north), (west, south)]


class Overpass:
    def __init__(self, region: dict[str, float]) -> None:
        w, s = region["west"], region["south"]
        dx, dy = region["east"] - w, region["north"] - s

        def at(fx: float, fy: float) -> tuple[float, float]:
            return (w + fx * dx, s + fy * dy)

        self.house = _rect(*at(0.1, 0.1), *at(0.3, 0.4))
        outer = _rect(*at(0.5, 0.2), *at(0.9, 0.8))
        self.island = _rect(*at(0.6, 0.4), *at(0.7, 0.5))
        self.lake = Polygon(outer, [self.island])
        self.road = [at(0.05, 0.6), at(0.45, 0.65), at(0.45, 0.95)]
        roundabout = _rect(*at(0.12, 0.7), *at(0.18, 0.8))
        self.elements = [
            {"type": "way", "id": 10, "tags": {"building": "yes"}, "geometry": _ll(self.house)},
            {
                "type": "way",
                "id": 11,
                "tags": {"highway": "residential"},
                "geometry": _ll(self.road),
            },
            {
                "type": "way",
                "id": 12,
                "tags": {"highway": "primary", "junction": "roundabout"},
                "geometry": _ll(roundabout),
            },
            {"type": "way", "id": 13, "tags": {"highway": "footway"}, "geometry": _ll(self.road)},
            # Tagged for two classes, outside the imagery: the first class in config order wins.
            {
                "type": "way",
                "id": 14,
                "tags": {"building": "boathouse", "natural": "water"},
                "geometry": _ll(_rect(w - 1.0, s - 1.0, w - 0.9, s - 0.9)),
            },
            {
                "type": "relation",
                "id": 20,
                "tags": {"type": "multipolygon", "natural": "water"},
                "members": [
                    {"type": "way", "ref": 1, "role": "outer", "geometry": _ll(outer[:3])},
                    {"type": "way", "ref": 2, "role": "outer", "geometry": _ll(outer[2:])},
                    {"type": "way", "ref": 3, "role": "inner", "geometry": _ll(self.island)},
                ],
            },
            {
                "type": "node",
                "id": 30,
                "lat": at(0.5, 0.5)[1],
                "lon": at(0.5, 0.5)[0],
                "tags": {"building": "kiosk"},
            },
        ]
        self.queries: list[str] = []
        self.status = 200
        # An answer to send as it is instead of the elements, and a slow server that sends
        # one byte a second.
        self.raw: bytes | None = None
        self.drip = False
        # A chunked answer sent as it is (chunk headers included), then the connection closes.
        self.chunked: bytes | None = None
        owner = self

        class Handler(BaseHTTPRequestHandler):
            def do_POST(self) -> None:
                body = self.rfile.read(int(self.headers["Content-Length"])).decode()
                owner.queries.append(urllib.parse.parse_qs(body)["data"][0])
                if owner.status != 200:
                    self.send_response(owner.status)
                    self.end_headers()
                    return
                if owner.chunked is not None:
                    self.send_response(200)
                    self.send_header("Transfer-Encoding", "chunked")
                    self.end_headers()
                    self.wfile.write(owner.chunked)
                    self.wfile.flush()
                    self.close_connection = True
                    return
                if owner.drip:
                    self.send_response(200)
                    self.send_header("Content-Length", "1000")
                    self.end_headers()
                    try:
                        for _ in range(1000):
                            self.wfile.write(b" ")
                            self.wfile.flush()
                            time.sleep(1)
                    except OSError:
                        pass  # the client gave up
                    return
                answer = (
                    owner.raw
                    if owner.raw is not None
                    else json.dumps(
                        {
                            "osm3s": {"timestamp_osm_base": "2026-10-01T12:00:00Z"},
                            "elements": owner.elements,
                        }
                    ).encode()
                )
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(answer)))
                self.end_headers()
                self.wfile.write(answer)

            def log_message(self, *args: Any) -> None:
                pass

        self.server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self.url = f"http://127.0.0.1:{self.server.server_address[1]}/api/interpreter"
        threading.Thread(target=self.server.serve_forever, daemon=True).start()


@pytest.fixture
def region() -> dict[str, float]:
    return region_inside(reference_transform(), WIDTH, HEIGHT, margin=0.0)


@pytest.fixture
def overpass(region: dict[str, float]) -> Iterator[Overpass]:
    served = Overpass(region)
    yield served
    served.server.shutdown()
    served.server.server_close()


CLASSES = [
    {"name": "building", "tags": {"building": "*"}},
    {"name": "water", "tags": {"natural": "water"}},
    {"name": "road", "tags": {"highway": ["residential", "primary"]}},
]


def _config(
    tmp_path: Path, region: dict[str, float], overpass: Overpass, **labels: Any
) -> MapcvConfig:
    return MapcvConfig.model_validate(
        {
            "region": region,
            "imagery": {"type": "geotiff", "path": str(tmp_path / "image.tif")},
            "labels": {"osm": {"classes": CLASSES, "overpass_url": overpass.url}, **labels},
            "sampler": {"patch_size": PATCH, "edge_strategy": "drop"},
            "writer": {"staging_dir": str(tmp_path / "dataset")},
        }
    )


def test_query_classes_and_shapes(
    tmp_path: Path, region: dict[str, float], overpass: Overpass
) -> None:
    labels = _config(tmp_path, region, overpass).labels
    assert isinstance(labels, LabelsConfig) and labels.osm is not None
    source = labels.osm
    assert source.bbox == (region["west"], region["south"], region["east"], region["north"])
    box = f"({region['south']!r},{region['west']!r},{region['north']!r},{region['east']!r})"
    assert overpass_query(source) == (
        "[out:json][timeout:180];\n(\n"
        f'  nwr["building"]{box};\n'
        f'  nwr["natural"="water"]{box};\n'
        f'  nwr["highway"~"^(residential|primary)$"]{box};\n'
        ");\nout geom;\n"
    )
    features = features_from_overpass({"elements": overpass.elements}, source.classes)
    got = [
        (f["properties"]["class"], f["properties"]["osm_id"], f["geometry"]["type"])
        for f in features
    ]
    assert got == [
        ("building", "node/30", "Point"),
        ("building", "way/10", "Polygon"),
        ("building", "way/14", "Polygon"),
        ("water", "relation/20", "Polygon"),
        ("road", "way/11", "LineString"),
        ("road", "way/12", "LineString"),  # a closed highway is a line, not an area
    ]
    lake = shape(features[3]["geometry"])
    assert lake.equals(overpass.lake) and len(lake.interiors) == 1  # rings joined, island cut out


def test_the_answer_is_cached_with_its_provenance(
    tmp_path: Path, region: dict[str, float], overpass: Overpass
) -> None:
    labels = _config(tmp_path, region, overpass).labels
    assert isinstance(labels, LabelsConfig) and labels.osm is not None
    first = osm_labels_file(labels.osm)
    again = osm_labels_file(labels.osm)
    assert first == again and len(overpass.queries) == 1
    data = json.loads(first.read_text(encoding="utf-8"))
    assert data["osm_base"] == "2026-10-01T12:00:00Z" and data["license"] == ATTRIBUTION
    assert data["query"] == overpass.queries[0]
    other = labels.osm.model_copy(update={"classes": labels.osm.classes[:1]})
    assert osm_labels_file(other) != first and len(overpass.queries) == 2


def test_failures(tmp_path: Path, region: dict[str, float], overpass: Overpass) -> None:
    labels = _config(tmp_path, region, overpass).labels
    assert isinstance(labels, LabelsConfig) and labels.osm is not None
    overpass.status = 429
    with pytest.raises(RuntimeError, match="Overpass request to .* failed"):
        osm_labels_file(labels.osm, tmp_path / "osm")
    overpass.status = 200
    overpass.elements = []
    from mapcv import osm

    original = osm._fetch
    try:
        osm._fetch = lambda source, query: {"remark": "runtime error: Query timed out"}
        with pytest.raises(RuntimeError, match="Overpass reported an error"):
            osm_labels_file(labels.osm, tmp_path / "other")
    finally:
        osm._fetch = original


_FOREST = [OsmClass(name="forest", tags={"landuse": "*"})]


def _relation(*rings: tuple[str, list[tuple[float, float]]]) -> dict[str, Any]:
    members = [{"type": "way", "role": role, "geometry": _ll(ring)} for role, ring in rings]
    tags = {"type": "multipolygon", "landuse": "forest"}
    return {"type": "relation", "id": 7, "tags": tags, "members": members}


def _only_geometry(element: dict[str, Any]) -> Any:
    (feature,) = features_from_overpass({"elements": [element]}, _FOREST)
    return shape(feature["geometry"])


def test_an_island_in_a_lake_of_a_multipolygon_is_kept() -> None:
    # An outer ring inside an inner ring (an island in a clearing): unioning the outers
    # and cutting out the inners lost the island.
    outer, hole, island = _rect(0, 0, 10, 10), _rect(2, 2, 8, 8), _rect(4, 4, 6, 6)
    got = _only_geometry(_relation(("outer", outer), ("inner", hole), ("outer", island)))
    expected = Polygon(outer, [hole]).union(Polygon(island))  # shapely, by nesting
    assert got.equals(expected) and got.area == 68.0
    # Nesting decides, not the roles: the same rings with the island tagged inner too.
    swapped = _only_geometry(_relation(("outer", outer), ("inner", hole), ("inner", island)))
    assert swapped.equals(expected)


def test_a_self_intersecting_way_keeps_both_lobes() -> None:
    # A figure-eight ring: buffer(0) kept one triangle; make_valid keeps both.
    bowtie = [(74.301, 31.481), (74.309, 31.489), (74.309, 31.481), (74.301, 31.489)]
    ring = [*bowtie, bowtie[0]]
    way = {"type": "way", "id": 1, "tags": {"landuse": "x"}, "geometry": _ll(ring)}
    got = _only_geometry(way)
    expected = make_valid(Polygon(ring))
    assert got.is_valid and got.equals(expected)
    lobe = Polygon([(74.301, 31.481), (74.305, 31.485), (74.301, 31.489)])
    assert got.area == pytest.approx(2 * lobe.area)  # it was one lobe


@pytest.mark.parametrize(
    "element",
    [
        5,
        {"type": "node", "lat": 1.0, "lon": 1.0, "tags": {"landuse": "a"}},  # no id
        {"type": "node", "id": 2, "lon": 1.0, "tags": {"landuse": "a"}},  # no lat
        {"type": "way", "id": 3, "tags": {"landuse": "a"}, "geometry": [{"lat": 1.0}]},
        {
            "type": "relation",
            "id": 4,
            "tags": {"landuse": "a", "type": "multipolygon"},
            "members": 5,
        },
    ],
)
def test_malformed_overpass_elements_are_skipped_with_a_warning(element: Any) -> None:
    good = {"type": "node", "id": 1, "lat": 1.0, "lon": 1.0, "tags": {"landuse": "a"}}
    with pytest.warns(UserWarning, match="skipped 1 element"):
        features = features_from_overpass({"elements": [good, element]}, _FOREST)
    assert [feature["properties"]["osm_id"] for feature in features] == ["node/1"]
    # Tags that are null are no tags (the element matches no class).
    untagged = {"type": "node", "id": 9, "lat": 1.0, "lon": 1.0, "tags": None}
    assert features_from_overpass({"elements": [untagged]}, _FOREST) == []


@pytest.mark.parametrize(
    ("raw", "message"),
    [
        (b"<html><body>rate limited</body></html>", r"not Overpass JSON \(it starts with '<html>"),
        (b"[1, 2]", r"JSON that is not an Overpass answer \(not an object\)"),
        (b'{"elements": 5}', r"JSON that is not an Overpass answer \(its 'elements' is not a list"),
    ],
)
def test_an_answer_that_is_not_overpass_json_is_a_message(
    tmp_path: Path, region: dict[str, float], overpass: Overpass, raw: bytes, message: str
) -> None:
    labels = _config(tmp_path, region, overpass).labels
    assert isinstance(labels, LabelsConfig) and labels.osm is not None
    overpass.raw = raw
    with pytest.raises(
        RuntimeError, match=r"^Overpass at http://127\.0\.0\.1:\d+/api/interpreter "
    ):
        osm_labels_file(labels.osm, tmp_path / "osm")
    with pytest.raises(RuntimeError, match=message):
        osm_labels_file(labels.osm, tmp_path / "osm")


@pytest.mark.parametrize(
    ("raw", "reason"),
    [
        # The connection closes right after a chunk header announcing 255 bytes.
        (b"ff\r\n", "closed the connection before the whole answer arrived"),
        # A chunk that stops half way.
        (b"20\r\n" + b'{"elements"', "closed the connection before the whole answer arrived"),
        # A chunk size that is not a number (http.client reports the lost framing as a
        # short read, after the ValueError this used to raise).
        (b"zz\r\n{}\r\n0\r\n\r\n", "closed the connection before the whole answer arrived"),
    ],
    ids=["after-header", "mid-chunk", "garbled-size"],
)
def test_an_answer_cut_or_garbled_in_transit_is_a_message(
    tmp_path: Path, region: dict[str, float], overpass: Overpass, raw: bytes, reason: str
) -> None:
    labels = _config(tmp_path, region, overpass).labels
    assert isinstance(labels, LabelsConfig) and labels.osm is not None
    overpass.chunked = raw
    with pytest.raises(RuntimeError) as raised:
        osm_labels_file(labels.osm, tmp_path / "osm")
    message = str(raised.value)
    assert message.startswith("Overpass request to http://127.0.0.1:")
    assert "failed" in message and reason in message
    assert "try again later" in message
    assert not (tmp_path / "osm").exists() or not list((tmp_path / "osm").glob("*.geojson"))


def test_a_read_that_raises_a_value_error_is_reported_as_a_bad_answer() -> None:
    import http.client

    from mapcv import osm

    class Garbled:
        def read1(self, size: int) -> bytes:
            raise ValueError("invalid literal for int() with base 16: b'zz\\r\\n'")

    with pytest.raises(http.client.HTTPException, match="garbled chunk in the answer"):
        osm._read_answer(Garbled(), time.monotonic() + 5, "Overpass")
    assert "did not answer in valid HTTP" in osm._describe(http.client.HTTPException("bad"))
    assert osm._describe(http.client.BadStatusLine("x")).startswith("the server did not answer")


def test_a_slow_overpass_answer_stops_at_the_deadline(
    tmp_path: Path, region: dict[str, float], overpass: Overpass, monkeypatch: pytest.MonkeyPatch
) -> None:
    # A server sending one byte a second: each read is quick, so a per-read timeout never
    # fires; the deadline is for the whole answer.
    from mapcv import osm

    monkeypatch.setattr(osm, "ANSWER_MARGIN_SECONDS", 2)
    labels = _config(tmp_path, region, overpass).labels
    assert isinstance(labels, LabelsConfig) and labels.osm is not None
    source = labels.osm.model_copy(update={"timeout": 1})
    overpass.drip = True
    started = time.monotonic()
    with pytest.raises(RuntimeError, match=r"did not answer within 3 s \(labels\.osm\.timeout 1 s"):
        osm_labels_file(source, tmp_path / "osm")
    assert time.monotonic() - started < 6


def test_an_answer_larger_than_the_limit_is_an_error(
    tmp_path: Path, region: dict[str, float], overpass: Overpass, monkeypatch: pytest.MonkeyPatch
) -> None:
    from mapcv import osm

    monkeypatch.setattr(osm, "MAX_ANSWER_BYTES", 100)
    labels = _config(tmp_path, region, overpass).labels
    assert isinstance(labels, LabelsConfig) and labels.osm is not None
    overpass.raw = b'{"elements": []' + b" " * 200 + b"}"
    with pytest.raises(RuntimeError, match="sent more than"):
        osm_labels_file(labels.osm, tmp_path / "osm")


def test_a_mask_from_osm_labels(
    tmp_path: Path, region: dict[str, float], overpass: Overpass
) -> None:
    from pyproj import Transformer
    from rasterio.features import rasterize
    from rasterio.transform import Affine
    from shapely.ops import transform as reproject

    from mapcv.manifest import Manifest
    from mapcv.pipeline import run_generate

    write_raster(tmp_path / "image.tif", reference_transform(), WIDTH, HEIGHT, seed=4)
    config = _config(
        tmp_path,
        region,
        overpass,
        classes={"water": 7, "building": 3, "road": 5},
        buffer={"line": 6.0},
    )
    run_generate(config)
    manifest = Manifest.load(tmp_path / "dataset" / "manifest.json")
    assert manifest.class_map == {"building": 3, "water": 7, "road": 5}
    assert manifest.target is not None and manifest.target.labels is not None
    assert manifest.target.labels["osm"]["classes"][0] == CLASSES[0]
    to_utm = Transformer.from_crs("EPSG:4326", "EPSG:32631", always_xy=True).transform
    house = reproject(to_utm, Polygon(overpass.house))
    lake = reproject(to_utm, overpass.lake)
    seen = set()
    for entry in manifest.patches:
        mask = np.asarray(
            __import__("PIL.Image", fromlist=["Image"]).open(
                tmp_path / "dataset" / entry["files"]["mask"]
            )
        )
        transform = Affine(*manifest.patch_transform(entry))
        expected = rasterize(
            [(house, 3), (lake, 7)],
            out_shape=(PATCH, PATCH),
            transform=transform,
            fill=0,
            dtype="uint8",
        )
        # Roads (5) are buffered lines drawn after the polygons; compare everywhere else.
        roads = mask == 5
        np.testing.assert_array_equal(mask[~roads], expected[~roads])
        seen |= set(np.unique(mask).tolist())
    assert {0, 3, 5, 7} <= seen
    # Without a buffer the lines and the point are left out, with a warning.
    plain = _config(tmp_path, region, overpass, classes={"water": 7, "building": 3, "road": 5})
    from mapcv.targets.segmentation import load_labels

    with pytest.warns(UserWarning):
        geometries, _ = load_labels(plain.labels)  # type: ignore[arg-type]
    assert sorted(cid for _, cid in geometries) == [3, 3, 7]  # two buildings, one lake


@pytest.mark.parametrize(
    ("labels", "message"),
    [
        ({"path": "a.geojson", "osm": {"classes": CLASSES}}, "exactly one of path"),
        ({"osm": {"classes": CLASSES}, "label_field": "kind"}, "label_field does not apply"),
        ({"osm": {"classes": CLASSES}, "classes": {"forest": 2}}, "names forest"),
        ({"osm": {"classes": []}}, "at least 1 item"),
        ({"osm": {"classes": [CLASSES[0], CLASSES[0]]}}, "must be unique"),
        ({"osm": {"classes": [{"name": "x", "tags": {}}]}}, "needs tags"),
        ({"osm": {"classes": [{"name": "x", "tags": {'bad"key': "*"}}]}}, "keys are letters"),
        ({"osm": {"classes": [{"name": "x", "tags": {"k": 'a"b'}}]}}, "no quotes"),
        ({"osm": {"classes": CLASSES, "overpass_url": "http://overpass.example.org"}}, "https://"),
    ],
)
def test_config_refusals(labels: dict[str, Any], message: str) -> None:
    with pytest.raises(ValidationError, match=message):
        LabelsConfig.model_validate(labels)


def test_osm_labels_as_the_after_set_of_change_detection(
    tmp_path: Path, region: dict[str, float], overpass: Overpass
) -> None:
    # Hand labels from before (the house, class building) against today's OpenStreetMap
    # (the house and the lake). OpenStreetMap alone numbers its classes in config order
    # (water 1, building 2) and the file set alone has building 1; compared on one class
    # map by name, the house is unchanged and the lake is the change.
    from pyproj import Transformer
    from rasterio.features import rasterize
    from rasterio.transform import Affine
    from shapely.ops import transform as reproject

    from mapcv.agent_tools import Sandbox, ToolFailure, ToolState, make_plan_for
    from mapcv.card import card_text
    from mapcv.manifest import Manifest
    from mapcv.pipeline import run_generate

    write_raster(tmp_path / "image.tif", reference_transform(), WIDTH, HEIGHT, seed=4)
    before = tmp_path / "before.geojson"
    before.write_text(
        json.dumps(
            {
                "type": "FeatureCollection",
                "features": [
                    {
                        "type": "Feature",
                        "properties": {},
                        "geometry": {"type": "Polygon", "coordinates": [overpass.house]},
                    }
                ],
            }
        )
    )
    osm_classes = [CLASSES[1], CLASSES[0]]
    config = MapcvConfig.model_validate(
        {
            "task": "change",
            "region": region,
            "imagery": [
                {"type": "geotiff", "name": "before", "path": str(tmp_path / "image.tif")},
                {"type": "geotiff", "name": "after", "path": str(tmp_path / "image.tif")},
            ],
            "change": {
                "before": {"files": [{"path": str(before), "class": "building"}]},
                "after": {"osm": {"classes": osm_classes, "overpass_url": overpass.url}},
            },
            "sampler": {"patch_size": PATCH, "edge_strategy": "drop"},
            "writer": {"staging_dir": str(tmp_path / "dataset")},
        }
    )
    after = config.change_options.after
    assert after is not None and after.osm is not None
    assert after.osm.bbox == (region["west"], region["south"], region["east"], region["north"])
    with pytest.raises(ToolFailure, match="labels.osm downloads OpenStreetMap labels"):
        make_plan_for(ToolState(Sandbox(tmp_path)), config)

    with pytest.warns(UserWarning):  # the kiosk node needs labels.buffer
        run_generate(config)
    manifest = Manifest.load(tmp_path / "dataset" / "manifest.json")
    to_utm = Transformer.from_crs("EPSG:4326", "EPSG:32631", always_xy=True).transform
    lake = reproject(to_utm, overpass.lake)
    changed = 0
    for entry in manifest.patches:
        mask = np.asarray(
            __import__("PIL.Image", fromlist=["Image"]).open(
                tmp_path / "dataset" / entry["files"]["mask"]
            )
        )
        expected = rasterize(
            [(lake, 1)],
            out_shape=(PATCH, PATCH),
            transform=Affine(*manifest.patch_transform(entry)),
            fill=0,
            dtype="uint8",
        )
        np.testing.assert_array_equal(mask, expected)
        changed += int(expected.any())
    assert changed > 0
    assert "Open Database License (ODbL)" in card_text(tmp_path / "dataset")


def test_an_osm_query_needs_a_box() -> None:
    source = OsmLabelsSource.model_validate({"classes": CLASSES})
    with pytest.raises(ValueError, match="labels.osm.bbox"):
        overpass_query(source)


def test_cli_card_and_mcp(tmp_path: Path, region: dict[str, float], overpass: Overpass) -> None:
    import yaml
    from typer.testing import CliRunner

    from mapcv.agent_tools import Sandbox, ToolFailure, ToolState, make_plan_for
    from mapcv.card import card_text
    from mapcv.cli import app
    from mapcv.pipeline import run_generate

    write_raster(tmp_path / "image.tif", reference_transform(), WIDTH, HEIGHT, seed=4)
    config = _config(tmp_path, region, overpass, buffer={"line": 6.0})
    path = tmp_path / "osm.yaml"
    path.write_text(yaml.safe_dump(config.model_dump(mode="json", exclude_none=True)))
    result = CliRunner().invoke(app, ["validate", str(path)], env={"COLUMNS": "200"})
    assert (
        result.exit_code == 0
        and "OpenStreetMap (Overpass) · building, water, road" in result.output
    )
    with pytest.raises(ToolFailure, match="labels.osm downloads OpenStreetMap labels"):
        make_plan_for(ToolState(Sandbox(tmp_path)), config)
    run_generate(config)
    assert "Open Database License (ODbL)" in card_text(tmp_path / "dataset")
    assert isinstance(OsmLabelsSource.model_validate({"classes": CLASSES}).overpass_url, str)
