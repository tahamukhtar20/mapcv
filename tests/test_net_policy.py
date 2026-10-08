"""Requests never reach this machine or a private network from a public URL.

A host name is judged by the addresses it resolves to when connecting, redirects are
checked hop by hop, and the MCP server connects to public addresses only unless started
with --allow-local-urls. The Rust side has its own tests in ``src/http_policy.rs``.
"""

from __future__ import annotations

import asyncio
import ipaddress
import re
import socket
import threading
from collections.abc import Callable, Iterator
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any

import pytest

from mapcv import _net, osm, stac
from mapcv._net import AddressRefused, public_addresses_only

_real_getaddrinfo = socket.getaddrinfo


class Recorder:
    """An HTTP server on ``host`` that records each request and answers with ``answer``."""

    def __init__(self, host: str = "127.0.0.1") -> None:
        self.requests: list[str] = []
        self.answer: Callable[[BaseHTTPRequestHandler], None] = _json_answer
        recorder = self

        class Handler(BaseHTTPRequestHandler):
            def _handle(self) -> None:
                recorder.requests.append(f"{self.command} {self.path}")
                length = int(self.headers.get("Content-Length") or 0)
                if length:
                    self.rfile.read(length)
                recorder.answer(self)

            do_GET = do_POST = _handle

            def log_message(self, *args: Any) -> None:
                pass

        self.server = ThreadingHTTPServer((host, 0), Handler)
        self.port = self.server.server_address[1]
        threading.Thread(target=self.server.serve_forever, daemon=True).start()

    def close(self) -> None:
        self.server.shutdown()
        self.server.server_close()


def _json_answer(handler: BaseHTTPRequestHandler) -> None:
    body = b'{"type": "FeatureCollection", "features": [], "links": [], "elements": []}'
    handler.send_response(200)
    handler.send_header("Content-Type", "application/json")
    handler.send_header("Content-Length", str(len(body)))
    handler.end_headers()
    handler.wfile.write(body)


def _redirect_to(location: str) -> Callable[[BaseHTTPRequestHandler], None]:
    def answer(handler: BaseHTTPRequestHandler) -> None:
        handler.send_response(302)
        handler.send_header("Location", location)
        handler.send_header("Content-Length", "0")
        handler.end_headers()

    return answer


@pytest.fixture()
def service() -> Iterator[Recorder]:
    """An "internal service" on 127.0.0.1."""
    server = Recorder()
    yield server
    server.close()


def _fake_dns(monkeypatch: pytest.MonkeyPatch, names: dict[str, str]) -> None:
    """Resolve each of ``names`` to its address; other names as usual."""

    def getaddrinfo(host: Any, port: Any, *args: Any, **kwargs: Any) -> Any:
        if isinstance(host, str) and host in names:
            return _real_getaddrinfo(names[host], port, *args, **kwargs)
        return _real_getaddrinfo(host, port, *args, **kwargs)

    monkeypatch.setattr(socket, "getaddrinfo", getaddrinfo)


@pytest.mark.parametrize(
    "text",
    [
        "127.0.0.1",
        "10.2.3.4",
        "172.16.0.1",
        "192.168.1.1",
        "169.254.169.254",
        "100.100.100.200",
        "0.0.0.0",
        "255.255.255.255",
        "224.0.0.251",
        "::1",
        "::",
        "fd00:ec2::254",
        "fe80::1",
        "ff02::1",
        "::ffff:127.0.0.1",
        "::ffff:169.254.169.254",
    ],
)
def test_internal_addresses(text: str) -> None:
    assert _net.is_internal_ip(ipaddress.ip_address(text))


@pytest.mark.parametrize("text", ["8.8.8.8", "100.128.0.1", "2001:4860::8888", "::ffff:8.8.8.8"])
def test_public_addresses(text: str) -> None:
    assert not _net.is_internal_ip(ipaddress.ip_address(text))


def test_a_stac_catalog_name_resolving_to_this_machine_is_not_connected(
    service: Recorder, monkeypatch: pytest.MonkeyPatch
) -> None:
    # Plain http, as a public catalog cannot be: the config check is skipped, the rules not.
    _fake_dns(monkeypatch, {"catalog.example": "127.0.0.1"})
    with pytest.raises(AddressRefused, match="catalog.example resolves to an address on this"):
        stac._post(f"http://catalog.example:{service.port}/search", {})
    assert service.requests == []


def test_an_overpass_name_resolving_to_this_machine_is_not_connected(
    service: Recorder, monkeypatch: pytest.MonkeyPatch
) -> None:
    from mapcv.config import OsmLabelsSource

    _fake_dns(monkeypatch, {"overpass.example": "127.0.0.1"})
    source = OsmLabelsSource.model_validate(
        {"classes": [{"name": "b", "tags": {"building": "*"}}], "overpass_url": "https://x"}
    ).model_copy(update={"overpass_url": f"http://overpass.example:{service.port}/api"})
    with pytest.raises(RuntimeError, match="refused: overpass.example resolves to"):
        osm._fetch(source, "[out:json];")
    assert service.requests == []


def test_a_url_naming_this_machine_still_reaches_it(service: Recorder) -> None:
    """A local test server, named by its address, keeps working (outside the MCP server)."""
    with _net.urlopen(f"http://127.0.0.1:{service.port}/search", timeout=10) as response:
        assert response.status == 200
    with _net.urlopen(f"http://localhost:{service.port}/search", timeout=10) as response:
        assert response.status == 200
    assert len(service.requests) == 2


def test_the_allow_local_variable_lets_names_resolve_internally(
    service: Recorder, monkeypatch: pytest.MonkeyPatch
) -> None:
    _fake_dns(monkeypatch, {"tiles.corp.example": "127.0.0.1"})
    monkeypatch.setenv("MAPCV_ALLOW_LOCAL_URLS", "1")
    with _net.urlopen(f"http://tiles.corp.example:{service.port}/x", timeout=10) as response:
        assert response.status == 200
    assert service.requests == ["GET /x"]


@pytest.fixture()
def second_loopback() -> Iterator[Recorder]:
    """A server on 127.0.0.2, standing in for a public host (Linux has the whole /8)."""
    try:
        server = Recorder("127.0.0.2")
    except OSError:
        pytest.skip("127.0.0.2 is not configured on this machine")
    yield server
    server.close()


def _only_127_0_0_1_is_internal(monkeypatch: pytest.MonkeyPatch) -> None:
    real = _net.is_internal_ip

    def internal(ip: ipaddress.IPv4Address | ipaddress.IPv6Address) -> bool:
        return ip == ipaddress.ip_address("127.0.0.1") or (
            real(ip) and not str(ip).startswith("127.")
        )

    monkeypatch.setattr(_net, "is_internal_ip", internal)


@pytest.mark.parametrize("target", ["127.0.0.1", "internal.example"])
def test_a_public_server_cannot_redirect_into_this_machine(
    target: str,
    service: Recorder,
    second_loopback: Recorder,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """By a literal address (the redirect check) or by a name (the connect-time check)."""
    _only_127_0_0_1_is_internal(monkeypatch)
    _fake_dns(monkeypatch, {"public.example": "127.0.0.2", "internal.example": "127.0.0.1"})
    second_loopback.answer = _redirect_to(f"http://{target}:{service.port}/latest/meta-data")
    with pytest.raises(AddressRefused, match="private network"):
        stac._get(f"http://public.example:{second_loopback.port}/search")
    assert second_loopback.requests == ["GET /search"]
    assert service.requests == []


def test_redirect_rules() -> None:
    start = "https://catalog.example/search"
    assert _net.redirect_refusal(start, start, "https://cdn.example/x") is None
    assert "plain http" in (_net.redirect_refusal(start, start, "http://cdn.example/x") or "")
    assert "credentials" in (_net.redirect_refusal(start, start, "https://u:p@c.example/") or "")
    assert "only http(s)" in (_net.redirect_refusal(start, start, "file:///etc/passwd") or "")
    assert _net.redirect_refusal(start, start, "https://[::ffff:10.0.0.1]/") is not None
    local = "http://127.0.0.1:8000/search"
    assert _net.redirect_refusal(local, local, "http://localhost:8001/x") is None


# ── The MCP server: public addresses only ────────────────────────────────────


def test_public_only_refuses_this_machine_before_connecting(service: Recorder) -> None:
    from mapcv._mapcv_rs import TileIndex, fetch_tiles
    from mapcv.geotiff import GeoTiff

    url = f"http://127.0.0.1:{service.port}"
    with public_addresses_only():
        assert _net.public_only()
        with pytest.raises(AddressRefused, match="--allow-local-urls"):
            _net.urlopen(f"{url}/search", timeout=10)
        with pytest.raises(RuntimeError, match="--allow-local-urls"):
            GeoTiff(f"{url}/remote.tif")
        with pytest.raises(ValueError, match="--allow-local-urls"):
            fetch_tiles([TileIndex(1, 1, 3)], url + "/{z}/{x}/{y}.png", None, 1, "lenient", 1.0)
    assert service.requests == []
    assert not _net.public_only()
    # Outside the block, the same local server is reachable again.
    with _net.urlopen(f"{url}/search", timeout=10) as response:
        assert response.status == 200


def _plan_message(state: Any, url: str) -> str:
    from mapcv.agent_tools import ToolFailure, plan

    config = (
        "region: {west: 10.0, south: 50.0, east: 10.01, north: 50.01}\n"
        f'imagery: {{type: geotiff, path: "{url}"}}\n'
        "sampler: {patch_size: 256}\nwriter: {staging_dir: out}\n"
    )
    with pytest.raises(ToolFailure) as excinfo:
        plan(state, yaml_text=config)
    return excinfo.value.message


def test_a_read_only_mcp_server_does_not_probe_this_machine(
    tmp_path: Path, service: Recorder
) -> None:
    from mapcv.agent_tools import Sandbox, ToolState

    closed = socket.socket()
    closed.bind(("127.0.0.1", 0))
    closed_port = closed.getsockname()[1]
    closed.close()
    state = ToolState(Sandbox(tmp_path))
    messages = [
        _plan_message(state, f"http://127.0.0.1:{service.port}/admin/x.tif"),
        _plan_message(state, f"http://127.0.0.1:{closed_port}/admin/x.tif"),
        _plan_message(state, f"http://localhost:{closed_port}/x.tif"),
    ]
    assert service.requests == []
    # Open port or closed, the answer is the same: it tells nothing about the target.
    shown = {re.sub(r"http://[^ ]+", "URL", message) for message in messages}
    assert len(shown) == 1, shown
    assert "--allow-local-urls" in messages[0]


def test_allow_local_urls_lets_the_mcp_server_read_a_local_server(
    tmp_path: Path, service: Recorder
) -> None:
    from mapcv.agent_tools import Sandbox, ToolState

    state = ToolState(Sandbox(tmp_path, allow_local_urls=True))
    message = _plan_message(state, f"http://127.0.0.1:{service.port}/x.tif")
    assert "range requests" in message  # it answered: the request was made
    assert service.requests == ["GET /x.tif"]


# ── fsspec (EOPF products over https) ────────────────────────────────────────


def test_the_fsspec_session_does_not_connect_to_names_resolving_internally(
    service: Recorder, monkeypatch: pytest.MonkeyPatch
) -> None:
    pytest.importorskip("aiohttp")
    _fake_dns(monkeypatch, {"products.example": "127.0.0.1"})
    options = _net.fsspec_options("https://products.example/S2.zarr")
    assert set(options) == {"get_client"}

    async def fetch(url: str) -> int:
        session = await options["get_client"]()
        async with session, session.get(url) as response:
            return int(response.status)

    with pytest.raises(OSError, match="products.example resolves to an address"):
        asyncio.run(fetch(f"http://products.example:{service.port}/S2.zarr/.zmetadata"))
    service.answer = _redirect_to("http://169.254.169.254/latest/meta-data")
    with pytest.raises(AddressRefused, match="169.254.169.254"):
        asyncio.run(fetch(f"http://127.0.0.1:{service.port}/S2.zarr/.zmetadata"))
    assert service.requests == ["GET /S2.zarr/.zmetadata"]
    # A product named by a local address keeps fsspec's own client.
    assert _net.fsspec_options(f"https://127.0.0.1:{service.port}/S2.zarr") == {}


def test_an_eopf_product_is_opened_with_the_rules(monkeypatch: pytest.MonkeyPatch) -> None:
    pytest.importorskip("fsspec")
    pytest.importorskip("numcodecs")
    import fsspec.config
    from numcodecs import get_codec

    from mapcv.imagery import _eopf_guards

    before = fsspec.config.conf.get("https")
    with _eopf_guards("https://products.example/S2.zarr"):
        assert fsspec.config.conf["https"]["get_client"] is _net._public_client
        with pytest.raises(ValueError, match="pickle"):
            get_codec({"id": "pickle"})
    assert fsspec.config.conf.get("https") == before
    assert get_codec({"id": "pickle"}) is not None  # restored for everyone else
    with _eopf_guards("/data/S2.zarr"), pytest.raises(ValueError, match="pickle"):
        get_codec({"id": "pickle"})
