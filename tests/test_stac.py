"""``imagery.search`` (find the product in a STAC API) and ``imagery.scl_mask``.

A local STAC API serves items over several pages (a POST ``next`` link with a token,
then a GET one); the choice must be the least cloudy item covering the whole region,
earliest first on ties. End to end, the found product is a real EOPF Zarr product (see
test_eopf_zarr_fixture) with a scene classification band, and masked SCL classes must
become pixels without imagery.
"""

from __future__ import annotations

import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional

import numpy as np
import pytest
from pydantic import ValidationError

from mapcv.config import EOPFZarrImageryConfig, MapcvConfig, StacSearchConfig
from mapcv.stac import _interval, find_product

REGION = (4.40, 48.70, 4.45, 48.75)


def _square(west: float, south: float, east: float, north: float) -> Dict[str, Any]:
    ring = [[west, south], [east, south], [east, north], [west, north], [west, south]]
    return {"type": "Polygon", "coordinates": [ring]}


def _item(
    item_id: str, cloud: Optional[float], when: str, covers: bool = True, href: str = "x.zarr"
) -> Dict[str, Any]:
    geometry = _square(4.0, 48.0, 5.0, 49.0) if covers else _square(4.42, 48.0, 5.0, 49.0)
    properties: Dict[str, Any] = {"datetime": when}
    if cloud is not None:
        properties["eo:cloud_cover"] = cloud
    return {
        "type": "Feature",
        "id": item_id,
        "geometry": geometry,
        "properties": properties,
        "assets": {"product": {"href": href}},
    }


class Catalog:
    """A STAC API: ``/search`` answers in pages of two items."""

    def __init__(self, items: List[Dict[str, Any]], status: int = 200) -> None:
        self.items = items
        self.status = status
        self.requests: List[Dict[str, Any]] = []
        owner = self

        class Handler(BaseHTTPRequestHandler):
            def _answer(self, start: int) -> None:
                if owner.status != 200:
                    self.send_response(owner.status)
                    self.end_headers()
                    return
                page = owner.items[start : start + 2]
                links = []
                if start + 2 < len(owner.items):
                    following = start + 2
                    if start == 0:  # the first next link is a POST with a token
                        links.append(
                            {
                                "rel": "next",
                                "href": f"{owner.url}/search",
                                "method": "POST",
                                "body": {"token": following},
                                "merge": True,
                            }
                        )
                    else:
                        links.append(
                            {"rel": "next", "href": f"{owner.url}/search?start={following}"}
                        )
                body = json.dumps(
                    {"type": "FeatureCollection", "features": page, "links": links}
                ).encode()
                self.send_response(200)
                self.send_header("Content-Type", "application/geo+json")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)

            def do_POST(self) -> None:  # noqa: N802
                request = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
                owner.requests.append(request)
                self._answer(int(request.get("token", 0)))

            def do_GET(self) -> None:  # noqa: N802
                owner.requests.append({"get": self.path})
                self._answer(int(self.path.rsplit("=", 1)[1]))

            def log_message(self, *args: Any) -> None:
                pass

        self.server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self.url = f"http://127.0.0.1:{self.server.server_address[1]}"
        threading.Thread(target=self.server.serve_forever, daemon=True).start()

    def close(self) -> None:
        self.server.shutdown()
        self.server.server_close()


@pytest.fixture
def catalog() -> Iterator[Catalog]:
    served = Catalog([])
    yield served
    served.close()


def _search(catalog: Catalog, **extra: Any) -> StacSearchConfig:
    return StacSearchConfig(catalog=catalog.url, datetime="2025-05-01/2025-05-31", **extra)


def test_the_least_cloudy_covering_item_wins(catalog: Catalog) -> None:
    catalog.items = [
        _item("partial", 0.5, "2025-05-02T10:00:00Z", covers=False),
        _item("cloudy", 60.0, "2025-05-03T10:00:00Z"),
        _item("a-later", 5.0, "2025-05-20T10:00:00Z"),
        _item("unknown-cloud", None, "2025-05-04T10:00:00Z"),
        _item("z-earlier", 5.0, "2025-05-10T10:00:00Z", href="https://example.org/earlier.zarr"),
        _item("also-cloudy", 30.0, "2025-05-11T10:00:00Z"),
    ]
    match = find_product(_search(catalog), REGION)
    assert (match.item_id, match.href, match.cloud_cover) == (
        "z-earlier",
        "https://example.org/earlier.zarr",
        5.0,
    )
    assert match.candidates == 2
    first, token, last = catalog.requests
    assert first == {
        "collections": ["sentinel-2-l2a"],
        "bbox": list(REGION),
        "datetime": "2025-05-01T00:00:00Z/2025-05-31T23:59:59Z",
        "limit": 100,
    }
    assert token == {**first, "token": 2} and last == {"get": "/search?start=4"}
    # A higher limit lets the cloudy items in; the choice stays the least cloudy.
    assert find_product(_search(catalog, max_cloud=100), REGION).item_id == "z-earlier"


def test_search_failures(catalog: Catalog) -> None:
    catalog.items = [
        _item("cloudy", 60.0, "2025-05-03T10:00:00Z"),
        _item("partial", 1.0, "2025-05-02T10:00:00Z", covers=False),
    ]
    with pytest.raises(ValueError, match=r"\(2 found, 1 covering the whole region\)"):
        find_product(_search(catalog), REGION)
    catalog.items = [_item("ok", 1.0, "2025-05-03T10:00:00Z")]
    with pytest.raises(ValueError, match="no 'visual' asset"):
        find_product(_search(catalog, asset="visual"), REGION)
    catalog.status = 503
    with pytest.raises(RuntimeError, match="STAC search at .* failed"):
        find_product(_search(catalog), REGION)


@pytest.mark.parametrize(
    ("value", "interval"),
    [
        ("2025-05-13", "2025-05-13T00:00:00Z/2025-05-13T23:59:59Z"),
        ("2025-05-01/2025-05-31", "2025-05-01T00:00:00Z/2025-05-31T23:59:59Z"),
        ("2025-05-01/..", "2025-05-01T00:00:00Z/.."),
        ("../2025-05-31T12:00:00Z", "../2025-05-31T12:00:00Z"),
        ("2025-05-13T10:40:00+02:00", "2025-05-13T10:40:00+02:00"),
    ],
)
def test_datetime_intervals(value: str, interval: str) -> None:
    assert _interval(StacSearchConfig(datetime=value).datetime) == interval


@pytest.mark.parametrize(
    ("data", "message"),
    [
        ({"type": "eopf_zarr"}, "exactly one of 'path'"),
        (
            {"type": "eopf_zarr", "path": "a.zarr", "search": {"datetime": "2025-05-01"}},
            "exactly one of 'path'",
        ),
        ({"type": "eopf_zarr", "search": {"datetime": "May 2025"}}, "imagery.search.datetime"),
        ({"type": "eopf_zarr", "search": {"datetime": ".."}}, "imagery.search.datetime"),
        (
            {
                "type": "eopf_zarr",
                "search": {"datetime": "2025-05-01", "catalog": "http://stac.example.org"},
            },
            "https:// STAC API",
        ),
        (
            {
                "type": "eopf_zarr",
                "search": {"datetime": "2025-05-01", "catalog": "https://u:p@stac.example.org"},
            },
            "credentials",
        ),
        (
            {"type": "eopf_zarr", "search": {"datetime": "2025-05-01", "max_cloud": 101}},
            "less than or equal to 100",
        ),
        ({"type": "eopf_zarr", "path": "a.zarr", "scl_mask": [12]}, "scene classes 0-11"),
        ({"type": "eopf_zarr", "path": "a.zarr", "scl_mask": []}, "scene classes 0-11"),
    ],
)
def test_config_refusals(data: Dict[str, Any], message: str) -> None:
    with pytest.raises(ValidationError, match=message):
        EOPFZarrImageryConfig.model_validate(data)


def test_config_normalisation() -> None:
    config = EOPFZarrImageryConfig.model_validate(
        {"type": "eopf_zarr", "path": "a.zarr", "scl_mask": [9, 3, 8, 9]}
    )
    assert config.scl_mask == [3, 8, 9]
    search = StacSearchConfig(datetime=" 2025-05-01 ", catalog="https://stac.example.org/")
    assert (search.catalog, search.datetime) == ("https://stac.example.org", "2025-05-01")


# ── End to end with a real product ───────────────────────────────────────────


def test_a_searched_product_with_masked_clouds(catalog: Catalog, tmp_path: Path) -> None:
    xr = pytest.importorskip("xarray")
    pytest.importorskip("xarray_eopf")
    from pyproj import Transformer

    from mapcv.manifest import Manifest
    from mapcv.pipeline import run_generate
    from test_eopf_zarr_fixture import (
        EPSG,
        HEIGHT,
        NAME,
        WIDTH,
        X0,
        Y0,
        _decoded,
        _digital_numbers,
        _group,
    )

    r10 = {name: _digital_numbers(WIDTH, HEIGHT, seed) for seed, name in enumerate(["b02", "b04"])}
    scl = np.random.default_rng(7).integers(0, 12, size=(HEIGHT // 2, WIDTH // 2)).astype(np.uint8)
    scl_group = xr.Dataset(
        {"scl": xr.DataArray(scl, dims=("y", "x"), attrs={"proj:epsg": EPSG})},
        coords={
            "x": X0 + 20 * (np.arange(WIDTH // 2) + 0.5),
            "y": Y0 - 20 * (np.arange(HEIGHT // 2) + 0.5),
        },
    )
    tree = xr.DataTree.from_dict(
        {
            "/": xr.Dataset(
                attrs={"stac_discovery": {"properties": {"proj:code": f"EPSG:{EPSG}"}}}
            ),
            "/measurements/reflectance/r10m": _group(10, r10),
            "/conditions/mask/l2a_classification/r20m": scl_group,
        }
    )
    product = tmp_path / NAME
    tree.to_zarr(product, mode="w")

    to_lonlat = Transformer.from_crs(f"EPSG:{EPSG}", "EPSG:4326", always_xy=True)
    west, north = to_lonlat.transform(X0 + 10 * 12 + 1, Y0 - 10 * 12 - 1)
    east, south = to_lonlat.transform(X0 + 10 * 76 - 1, Y0 - 10 * 76 + 1)
    region = {"west": west, "south": south, "east": east, "north": north}
    catalog.items = [
        _item("best", 2.0, "2025-05-09T10:00:00Z", href=str(product)),
        _item("worse", 9.0, "2025-05-08T10:00:00Z", href="https://example.org/never-read.zarr"),
    ]
    catalog.items = [
        {**item, "geometry": _square(west - 0.1, south - 0.1, east + 0.1, north + 0.1)}
        for item in catalog.items
    ]
    config = MapcvConfig.model_validate(
        {
            "region": region,
            "imagery": {
                "type": "eopf_zarr",
                "search": {"catalog": catalog.url, "datetime": "2025-05-01/2025-05-31"},
                "bands": ["b02", "b04"],
                "scl_mask": [3, 8, 9, 10],
            },
            "sampler": {"patch_size": 32, "edge_strategy": "drop", "max_empty_ratio": 1.0},
            "writer": {"staging_dir": str(tmp_path / "dataset"), "image_format": "npy"},
        }
    )
    run_generate(config)
    manifest = Manifest.load(tmp_path / "dataset" / "manifest.json")
    source = manifest.source
    assert source.product_id == NAME
    assert source.fingerprint == {
        "stac": {"catalog": catalog.url, "collection": "sentinel-2-l2a", "item": "best"},
        "scl_mask": [3, 8, 9, 10],
    }
    masked_somewhere = False
    for entry in manifest.patches:
        _, _, c, _, _, f = manifest.patch_transform(entry)
        col, row = round((c - X0) / 10), round((Y0 - f) / 10)
        patch = np.load(tmp_path / "dataset" / entry["files"]["image"])
        classes = np.repeat(np.repeat(scl, 2, 0), 2, 1)[row : row + 32, col : col + 32]
        masked = np.isin(classes, [3, 8, 9, 10])
        expected = _decoded(r10["b04"])[row : row + 32, col : col + 32].astype(np.float32)
        expected[masked] = np.nan
        np.testing.assert_allclose(patch[1], expected, equal_nan=True)
        assert np.isnan(patch[0][masked]).all()
        masked_somewhere |= bool(masked.any())
        assert entry["summary"]["empty_ratio"] == pytest.approx(
            float((masked | np.isnan(_decoded(r10["b02"])[row : row + 32, col : col + 32])).mean()),
            abs=1e-9,
        )
    assert masked_somewhere and len(manifest.patches) >= 4


def test_a_remote_catalog_cannot_point_to_local_files(monkeypatch: pytest.MonkeyPatch) -> None:
    from mapcv.config import RegionConfig
    from mapcv.imagery import EOPFZarrRasterSource
    from mapcv.stac import StacMatch

    monkeypatch.setattr(
        "mapcv.stac.find_product",
        lambda search, bbox: StacMatch("evil", "/etc/secret.zarr", 0.0, "2025-05-01", 1),
    )
    config = EOPFZarrImageryConfig.model_validate(
        {
            "type": "eopf_zarr",
            "search": {"datetime": "2025-05-01", "catalog": "https://stac.example.org"},
        }
    )
    region = RegionConfig(west=REGION[0], south=REGION[1], east=REGION[2], north=REGION[3])
    with pytest.raises(ValueError, match="may only point to https:// or s3:// products"):
        EOPFZarrRasterSource(region, config)
    monkeypatch.setattr(
        "mapcv.stac.find_product",
        lambda search, bbox: StacMatch(
            "creds", "https://u:p@example.org/a.zarr", 0.0, "2025-05-01", 1
        ),
    )
    with pytest.raises(ValueError, match="STAC item creds: .*credentials"):
        EOPFZarrRasterSource(region, config)
