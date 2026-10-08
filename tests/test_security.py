"""Security and robustness tests."""

from __future__ import annotations

import io
import re
import time
from typing import Any

import numpy as np
import pytest
from PIL import Image

from mapcv._mapcv_rs import TileIndex, fetch_tiles, stitch_tiles


def _make_tile(r: int, g: int, b: int) -> bytes:
    """Return PNG bytes for a solid-colour 256×256 RGB tile."""
    img = Image.fromarray(np.full((256, 256, 3), [r, g, b], dtype=np.uint8), mode="RGB")
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


RED = _make_tile(255, 0, 0)


def test_url_sanitization_on_failure(httpserver: Any) -> None:
    """Query parameters should be redacted from error messages."""
    httpserver.expect_request("/tile/0/0/0.png").respond_with_data("Not Found", status=404)

    # URL with sensitive API key in query param
    base_url = httpserver.url_for("/tile/{z}/{x}/{y}.png")
    url_template = f"{base_url}?api_key=SECRET_123#fragment-secret"

    t = TileIndex(0, 0, 0)

    with pytest.raises(RuntimeError) as excinfo:
        fetch_tiles(
            tiles=[t],
            url_template=url_template,
            policy="strict",
        )

    error_msg = str(excinfo.value)
    assert "SECRET_123" not in error_msg
    assert "api_key" not in error_msg
    assert "fragment-secret" not in error_msg
    # Host and the tile's z/x/y identify the request; the rest of the path is elided.
    host = httpserver.url_for("/").rstrip("/")
    assert f"{host}/…/0/0/0.png" in error_msg


def test_path_secrets_are_not_shown_on_failure(httpserver: Any) -> None:
    """Keys and short-lived map IDs in the path (``/v1/<key>/...``) are elided too."""
    httpserver.expect_request("/v1/PATHSECRET99/0/0/0.png").respond_with_data("No", status=403)
    url_template = httpserver.url_for("/v1/PATHSECRET99/{z}/{x}/{y}.png")
    with pytest.raises(RuntimeError) as excinfo:
        fetch_tiles(tiles=[TileIndex(0, 0, 0)], url_template=url_template, policy="strict")
    assert "PATHSECRET99" not in str(excinfo.value) and "HTTP 403" in str(excinfo.value)


def test_stitch_rejects_canvas_above_byte_budget() -> None:
    """A sparse tile extent must be rejected before a huge canvas is allocated."""
    t1 = TileIndex(0, 0, 16)
    t2 = TileIndex(2731, 0, 16)

    with pytest.raises(RuntimeError, match="exceeding"):
        stitch_tiles([(t1, RED), (t2, RED)])


def test_stitch_usize_overflow_prevention() -> None:
    """stitch_tiles should handle coordinate differences that would overflow u32."""
    t1 = TileIndex(0, 0, 31)
    t2 = TileIndex(4294967295, 0, 31)

    with pytest.raises(RuntimeError, match="canvas"):
        stitch_tiles([(t1, RED), (t2, RED)])


# ── Redirects (a tile server or a GeoTIFF host decides where a redirect goes) ──


def _redirecting(httpserver: Any, seen: list[dict[str, str]]) -> None:
    """``/redir/...`` answers 302 to the same server under another host name (``127.0.0.1``
    instead of ``localhost``), whose ``/final/...`` records the request headers."""
    from werkzeug import Request, Response

    port = httpserver.port

    def redirect(request: Request) -> Response:
        target = f"http://127.0.0.1:{port}/final/" + request.path.removeprefix("/redir/")
        return Response(status=302, headers={"Location": target})

    def final(request: Request) -> Response:
        seen.append(dict(request.headers))
        return Response(RED, content_type="image/png")

    httpserver.expect_request(re.compile("^/redir/")).respond_with_handler(redirect)
    httpserver.expect_request(re.compile("^/final/")).respond_with_handler(final)


def test_a_keyed_tile_url_is_not_redirected_to_another_host(httpserver: Any) -> None:
    """The key of a url_template never reaches another host: not in a Referer, not at all."""
    seen: list[dict[str, str]] = []
    _redirecting(httpserver, seen)
    template = httpserver.url_for("/redir/{z}/{x}/{y}.png") + "?key=SECRETQ"
    with pytest.raises(RuntimeError) as excinfo:
        fetch_tiles(tiles=[TileIndex(0, 0, 0)], url_template=template, policy="strict")
    message = str(excinfo.value)
    assert "redirected to another host (http://127.0.0.1:" in message
    assert "SECRETQ" not in message
    assert seen == []  # the other host got no request


def test_a_tile_redirect_sends_no_referer(httpserver: Any) -> None:
    seen: list[dict[str, str]] = []
    _redirecting(httpserver, seen)
    template = httpserver.url_for("/redir/{z}/{x}/{y}.png")
    results, failed, _ = fetch_tiles(
        tiles=[TileIndex(0, 0, 0)], url_template=template, policy="strict"
    )
    assert failed == 0 and len(results) == 1
    assert len(seen) == 1
    assert not any(name.lower() == "referer" for name in seen[0])


def test_a_remote_geotiff_is_not_redirected_to_plain_http_elsewhere(httpserver: Any) -> None:
    """Plain http is only for this machine: a loopback server may not hand the read to a
    plain-http host elsewhere (192.0.2.1 is a documentation address, never reached)."""
    from werkzeug import Response

    from mapcv._mapcv_rs import GeoTiff

    httpserver.expect_request("/remote.tif").respond_with_response(
        Response(status=302, headers={"Location": "http://192.0.2.1:9/remote.tif"})
    )
    started = time.monotonic()
    with pytest.raises(RuntimeError, match=r"redirected to plain http \(http://192\.0\.2\.1:9\)"):
        GeoTiff(httpserver.url_for("/remote.tif"))
    assert time.monotonic() - started < 10  # refused, not tried and timed out
