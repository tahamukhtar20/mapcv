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


def test_the_host_a_config_names_is_trusted_wherever_it_resolves(
    service: Recorder, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A docker-compose service, an on-prem server or an /etc/hosts alias: the user wrote it."""
    from mapcv.config import OsmLabelsSource

    _fake_dns(monkeypatch, {"catalog.lab": "127.0.0.1", "overpass.lab": "127.0.0.1"})
    stac._post(f"http://catalog.lab:{service.port}/search", {})
    source = OsmLabelsSource.model_validate(
        {"classes": [{"name": "b", "tags": {"building": "*"}}], "overpass_url": "https://x"}
    ).model_copy(update={"overpass_url": f"http://overpass.lab:{service.port}/api"})
    osm._fetch(source, "[out:json];")
    assert service.requests == ["POST /search", "POST /api"]


class _FakeSocket:
    """Records where a connection goes, and fails it (nothing is sent anywhere)."""

    attempts: list[Any] = []

    def __init__(self, *args: Any) -> None:
        pass

    def settimeout(self, value: Any) -> None:
        pass

    def bind(self, address: Any) -> None:
        pass

    def connect(self, address: Any) -> None:
        type(self).attempts.append(address[0])
        raise ConnectionRefusedError("refused (test)")

    def close(self) -> None:
        pass


def test_a_start_host_on_a_private_network_is_connected(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _fake_dns(monkeypatch, {"tiles.corp.example": "10.0.0.7"})
    _FakeSocket.attempts = []
    monkeypatch.setattr(socket, "socket", _FakeSocket)
    with pytest.raises(OSError) as excinfo:
        _net.urlopen("http://tiles.corp.example:8080/1/2/3.png", timeout=1)
    assert not isinstance(excinfo.value, AddressRefused)
    assert _FakeSocket.attempts == ["10.0.0.7"]


@pytest.mark.parametrize(
    "target", ["127.0.0.1:{port}", "internal.lab:8080", "public.example:{port}"]
)
def test_a_redirect_cannot_lead_to_another_host_on_a_private_network(
    target: str, service: Recorder, monkeypatch: pytest.MonkeyPatch
) -> None:
    """By a literal address (the redirect check), by a name resolving to 10.x, or to another
    port of the configured host (the connect-time check)."""
    _fake_dns(monkeypatch, {"public.example": "127.0.0.1", "internal.lab": "10.0.0.7"})
    first = Recorder()
    try:
        location = "http://" + target.format(port=service.port) + "/latest/meta-data"
        first.answer = _redirect_to(location)
        with pytest.raises(AddressRefused, match="private network"):
            stac._get(f"http://public.example:{first.port}/search")
        assert first.requests == ["GET /search"]
    finally:
        first.close()
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


def test_public_only_refuses_a_name_resolving_to_this_machine(
    service: Recorder, monkeypatch: pytest.MonkeyPatch
) -> None:
    _fake_dns(monkeypatch, {"catalog.lab": "127.0.0.1"})
    with public_addresses_only(), pytest.raises(AddressRefused, match="--allow-local-urls"):
        stac._post(f"http://catalog.lab:{service.port}/search", {})
    assert service.requests == []


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
    product = f"http://products.example:{service.port}/S2.zarr"

    async def fetch(options: dict[str, Any], url: str) -> int:
        session = await options["get_client"]()
        async with session, session.get(url) as response:
            return int(response.status)

    # The product the config names is the user's host: connected wherever it resolves.
    trusted = _net.fsspec_options(product)
    assert set(trusted) == {"get_client"}
    assert asyncio.run(fetch(trusted, f"{product}/.zmetadata")) == 200
    # One a catalog named is judged by its addresses.
    linked = _net.fsspec_options(product, trust_host=False)
    with pytest.raises(OSError, match="products.example resolves to an address"):
        asyncio.run(fetch(linked, f"{product}/.zmetadata"))
    service.answer = _redirect_to("http://169.254.169.254/latest/meta-data")
    with pytest.raises(AddressRefused, match="169.254.169.254"):
        asyncio.run(fetch(trusted, f"{product}/.zmetadata"))
    assert service.requests == ["GET /S2.zarr/.zmetadata"] * 2
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
        assert fsspec.config.conf["https"]["get_client"].func is _net._public_client
        with pytest.raises(ValueError, match="pickle"):
            get_codec({"id": "pickle"})
    assert fsspec.config.conf.get("https") == before
    assert get_codec({"id": "pickle"}) is not None  # restored for everyone else
    with _eopf_guards("/data/S2.zarr"), pytest.raises(ValueError, match="pickle"):
        get_codec({"id": "pickle"})
