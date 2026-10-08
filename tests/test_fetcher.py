"""Tests for the async tile fetcher (requires pytest-httpserver)."""

from __future__ import annotations

import re
import socket
import threading
import time
from collections.abc import Callable, Iterator
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any

import pytest

from mapcv._mapcv_rs import TileIndex, fetch_tiles

# Minimal valid PNG signatures followed by distinct payloads; the fetcher only
# checks the signature, decoding happens later.
PNG_1 = b"\x89PNG\r\n\x1a\n" + b"tile-1"
PNG_2 = b"\x89PNG\r\n\x1a\n" + b"tile-2"


def test_fetch_tiles_mock(httpserver: Any) -> None:
    httpserver.expect_request("/tile/14/2621/6331.png").respond_with_data(PNG_1)
    httpserver.expect_request("/tile/14/2622/6331.png").respond_with_data(PNG_2)

    url_template = httpserver.url_for("/tile/{z}/{x}/{y}.png")

    tile_list = [
        TileIndex(2621, 6331, 14),
        TileIndex(2622, 6331, 14),
    ]

    progress_updates: list[int] = []

    def callback(c: int) -> None:
        progress_updates.append(c)

    results, failed, _ = fetch_tiles(
        tile_list, url_template, callback=callback, max_connections=2, policy="strict"
    )

    assert len(results) == 2
    assert failed == 0
    # `completed` is a monotonic counter incremented after each outcome regardless
    # of which tile finishes first, so [1, 2] is always the order even with buffer_unordered.
    assert progress_updates == [1, 2]

    results_dict = {(t.x, t.y, t.z): b for t, b in results}
    assert results_dict[(2621, 6331, 14)] == PNG_1
    assert results_dict[(2622, 6331, 14)] == PNG_2


def test_fetch_tiles_lenient(httpserver: Any) -> None:
    httpserver.expect_request("/tile/14/1/1.png").respond_with_data(PNG_1)
    httpserver.expect_request("/tile/14/2/1.png").respond_with_data(b"NOT FOUND", status=404)

    url_template = httpserver.url_for("/tile/{z}/{x}/{y}.png")
    tile_list = [
        TileIndex(1, 1, 14),
        TileIndex(2, 1, 14),
    ]

    results, failed, _ = fetch_tiles(
        tile_list,
        url_template,
        callback=None,
        max_connections=2,
        policy="lenient",
        max_failed_ratio=1.0,  # permissive - this test is about omit-on-failure, not the ratio gate
    )

    assert len(results) == 1
    assert failed == 1
    assert results[0][0].x == 1


def test_fetch_tiles_strict(httpserver: Any) -> None:
    httpserver.expect_request("/tile/14/1/1.png").respond_with_data(b"NOT FOUND", status=404)

    url_template = httpserver.url_for("/tile/{z}/{x}/{y}.png")
    tile_list = [TileIndex(1, 1, 14)]

    with pytest.raises(RuntimeError, match="HTTP 404"):
        fetch_tiles(tile_list, url_template, callback=None, max_connections=2, policy="strict")


def test_fetch_tiles_ignore_returns_black_pixels(httpserver: Any) -> None:
    """Ignore policy: 404 tiles still appear in results as non-empty black PNG bytes."""
    httpserver.expect_request("/tile/14/1/1.png").respond_with_data(b"NOT FOUND", status=404)

    url_template = httpserver.url_for("/tile/{z}/{x}/{y}.png")
    tile_list = [TileIndex(1, 1, 14)]

    results, failed, _ = fetch_tiles(
        tile_list,
        url_template,
        callback=None,
        max_connections=1,
        policy="ignore",
        max_failed_ratio=1.0,  # allow all failures so the ratio check doesn't fire
    )

    assert len(results) == 1, "Ignore policy must keep the tile in results"
    assert failed == 1
    tile, data = results[0]
    assert tile.x == 1
    assert len(data) > 0, "Black fill must produce non-empty bytes"


def test_fetch_tiles_ignore_ignores_ratio(httpserver: Any) -> None:
    """Ignore policy: black-fill tiles ignore max_failed_ratio."""
    httpserver.expect_request("/tile/14/1/1.png").respond_with_data(b"NOT FOUND", status=404)
    httpserver.expect_request("/tile/14/2/1.png").respond_with_data(b"NOT FOUND", status=404)

    url_template = httpserver.url_for("/tile/{z}/{x}/{y}.png")
    tile_list = [TileIndex(1, 1, 14), TileIndex(2, 1, 14)]

    # 2/2 failed = 100 % > 0 % threshold, but policy is "ignore" so it should pass
    results, failed, _ = fetch_tiles(
        tile_list,
        url_template,
        callback=None,
        max_connections=2,
        policy="ignore",
        max_failed_ratio=0.0,
    )
    assert len(results) == 2
    assert failed == 2


def test_fetch_tiles_max_failed_ratio_exceeded(httpserver: Any) -> None:
    """Exceeding max_failed_ratio with lenient policy raises RuntimeError."""
    httpserver.expect_request("/tile/14/1/1.png").respond_with_data(PNG_1, status=200)
    httpserver.expect_request("/tile/14/2/1.png").respond_with_data(b"NOT FOUND", status=404)
    httpserver.expect_request("/tile/14/3/1.png").respond_with_data(b"NOT FOUND", status=404)

    url_template = httpserver.url_for("/tile/{z}/{x}/{y}.png")
    tile_list = [
        TileIndex(1, 1, 14),
        TileIndex(2, 1, 14),
        TileIndex(3, 1, 14),
    ]

    # 2/3 ~= 66.7% > 50% threshold
    with pytest.raises(RuntimeError, match="Too many failed tiles") as excinfo:
        fetch_tiles(
            tile_list,
            url_template,
            callback=None,
            max_connections=3,
            policy="lenient",
            max_failed_ratio=0.5,
        )
    message = str(excinfo.value)
    assert "2 x HTTP 404 Not Found" in message
    assert "/…/14/" in message  # one example URL: host and z/x/y, the rest elided
    assert "max_connections" in message


def test_fetch_tiles_reports_why_tiles_failed(httpserver: Any) -> None:
    httpserver.expect_request("/tile/14/1/1.png").respond_with_data(b"<html>", status=200)
    httpserver.expect_request("/tile/14/2/1.png").respond_with_data(b"gone", status=410)
    httpserver.expect_request("/tile/14/3/1.png").respond_with_data(b"gone", status=410)
    url_template = httpserver.url_for("/tile/{z}/{x}/{y}.png")
    tiles = [TileIndex(x, 1, 14) for x in (1, 2, 3)]

    _, failed, (causes, example) = fetch_tiles(tiles, url_template, policy="ignore")

    assert failed == 3
    assert dict(causes) == {"HTTP 410 Gone": 2, "response is not an image": 1}
    assert example is not None and "/…/14/" in example


def test_fetch_tiles_reasons_empty_without_failures(httpserver: Any) -> None:
    httpserver.expect_request("/tile/14/1/1.png").respond_with_data(PNG_1, status=200)
    url_template = httpserver.url_for("/tile/{z}/{x}/{y}.png")
    assert fetch_tiles([TileIndex(1, 1, 14)], url_template)[2] == ([], None)


def test_fetch_tiles_max_failed_ratio_not_exceeded(httpserver: Any) -> None:
    """Under max_failed_ratio threshold: results returned normally."""
    httpserver.expect_request("/tile/14/1/1.png").respond_with_data(PNG_1, status=200)
    httpserver.expect_request("/tile/14/2/1.png").respond_with_data(b"NOT FOUND", status=404)

    url_template = httpserver.url_for("/tile/{z}/{x}/{y}.png")
    tile_list = [TileIndex(1, 1, 14), TileIndex(2, 1, 14)]

    # 1/2 = 50 % which is not > 60 % threshold
    results, failed, _ = fetch_tiles(
        tile_list,
        url_template,
        callback=None,
        max_connections=2,
        policy="lenient",
        max_failed_ratio=0.6,
    )
    assert len(results) == 1
    assert failed == 1
    assert results[0][0].x == 1


def test_fetch_tiles_non_image_response_is_a_failure(httpserver: Any) -> None:
    """An HTML error page served with 200 OK is counted as failed, not returned."""
    httpserver.expect_request("/tile/14/1/1.png").respond_with_data(
        b"<html>quota exceeded</html>", content_type="text/html"
    )
    results, failed, _ = fetch_tiles(
        [TileIndex(1, 1, 14)],
        httpserver.url_for("/tile/{z}/{x}/{y}.png"),
        policy="lenient",
        max_failed_ratio=1.0,
    )
    assert results == []
    assert failed == 1

    with pytest.raises(RuntimeError, match="not an image"):
        fetch_tiles(
            [TileIndex(1, 1, 14)], httpserver.url_for("/tile/{z}/{x}/{y}.png"), policy="strict"
        )


def test_fetch_tiles_lenient_ratio_is_case_insensitive(httpserver: Any) -> None:
    httpserver.expect_request("/tile/14/1/1.png").respond_with_data(b"NOT FOUND", status=404)
    with pytest.raises(RuntimeError, match="Too many failed tiles"):
        fetch_tiles(
            [TileIndex(1, 1, 14)],
            httpserver.url_for("/tile/{z}/{x}/{y}.png"),
            policy="LENIENT",
            max_failed_ratio=0.0,
        )


def test_fetch_tiles_invalid_arguments_raise_value_error() -> None:
    with pytest.raises(ValueError, match="Unknown policy"):
        fetch_tiles([TileIndex(1, 1, 14)], "http://localhost/{z}/{x}/{y}.png", policy="nope")
    with pytest.raises(ValueError, match="max_connections"):
        fetch_tiles([TileIndex(1, 1, 14)], "http://localhost/{z}/{x}/{y}.png", max_connections=0)


def test_fetch_tiles_truncated_body_follows_policy() -> None:
    """A body cut short by the server is retried, then handled by the policy."""
    import socket
    import threading

    server = socket.socket()
    server.bind(("127.0.0.1", 0))
    server.listen()
    port = server.getsockname()[1]
    stop = threading.Event()

    def serve() -> None:
        server.settimeout(0.2)
        while not stop.is_set():
            try:
                conn, _ = server.accept()
            except OSError:
                continue
            with conn:
                conn.recv(4096)
                conn.sendall(b"HTTP/1.1 200 OK\r\nContent-Length: 100000\r\n\r\n\x89PNG\r\n")

    thread = threading.Thread(target=serve, daemon=True)
    thread.start()
    try:
        results, failed, _ = fetch_tiles(
            [TileIndex(1, 1, 14)],
            f"http://127.0.0.1:{port}/{{z}}/{{x}}/{{y}}.png",
            policy="ignore",
        )
    finally:
        stop.set()
        thread.join()
        server.close()
    assert failed == 1
    assert len(results) == 1  # black fill under the ignore policy


# ── Misbehaving servers ─────────────────────────────────────────────────────


@pytest.fixture
def raw_server() -> Iterator[Callable[[Callable[[Any], None]], str]]:
    """Start a local HTTP server whose GET handler is the given function; returns its
    URL template."""
    servers: list[ThreadingHTTPServer] = []

    def start(respond: Callable[[Any], None]) -> str:
        class Handler(BaseHTTPRequestHandler):
            protocol_version = "HTTP/1.0"

            def do_GET(self) -> None:
                respond(self)

            def log_message(self, *args: Any) -> None:
                pass

        server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        server.daemon_threads = True
        servers.append(server)
        threading.Thread(target=server.serve_forever, daemon=True).start()
        return f"http://127.0.0.1:{server.server_address[1]}/{{z}}/{{x}}/{{y}}.png"

    yield start
    for server in servers:
        server.shutdown()
        server.server_close()


def _declares_ten_gigabytes(handler: Any) -> None:
    handler.send_response(200)
    handler.send_header("Content-Length", "10000000000")
    handler.end_headers()
    try:
        while True:
            handler.wfile.write(b"\x89PNG\r\n\x1a\n" + bytes(1 << 20))
    except OSError:
        pass


def _streams_without_a_length(handler: Any) -> None:
    handler.send_response(200)
    handler.end_headers()
    try:
        handler.wfile.write(b"\x89PNG\r\n\x1a\n")
        while True:
            handler.wfile.write(bytes(1 << 20))
    except OSError:
        pass


@pytest.mark.parametrize("respond", [_declares_ten_gigabytes, _streams_without_a_length])
def test_a_huge_tile_body_fails_that_tile_instead_of_filling_memory(
    raw_server: Callable[[Callable[[Any], None]], str], respond: Callable[[Any], None]
) -> None:
    template = raw_server(respond)
    started = time.monotonic()
    results, failed, (causes, example) = fetch_tiles(
        [TileIndex(1, 1, 14)], template, policy="ignore", cache_headers=True
    )
    assert time.monotonic() - started < 10
    assert failed == 1
    assert causes == [("response too large", 1)]
    assert example is not None and "larger than 4 MiB" in example
    assert len(results) == 1 and results[0][2] is None  # a black fill, not the body


def test_a_normal_tile_below_the_size_cap_is_read(httpserver: Any) -> None:
    body = b"\x89PNG\r\n\x1a\n" + bytes(3 * 1024 * 1024)
    httpserver.expect_request("/t/14/1/1.png").respond_with_data(body)
    results, failed, _ = fetch_tiles(
        [TileIndex(1, 1, 14)], httpserver.url_for("/t/{z}/{x}/{y}.png"), policy="strict"
    )
    assert failed == 0 and results[0][1] == body


def test_a_refused_connection_stops_the_fetch_after_the_first_few_tiles() -> None:
    sock = socket.socket()
    sock.bind(("127.0.0.1", 0))
    port = sock.getsockname()[1]
    sock.close()  # nothing listens here now
    tiles = [TileIndex(x, 1, 14) for x in range(200)]
    callbacks: list[int] = []
    started = time.monotonic()
    with pytest.raises(RuntimeError, match="not answering: the first 8 requests all failed"):
        fetch_tiles(
            tiles,
            f"http://127.0.0.1:{port}/{{z}}/{{x}}/{{y}}.png",
            callback=callbacks.append,
            max_connections=4,
            policy="ignore",
        )
    assert time.monotonic() - started < 30
    assert len(callbacks) < 20  # not all 200 tiles were tried


def test_ignore_still_finishes_when_few_tiles_fail(httpserver: Any) -> None:
    httpserver.expect_request("/t/14/1/1.png").respond_with_data(PNG_1)
    httpserver.expect_request("/t/14/2/1.png").respond_with_data(b"gone", status=404)
    results, failed, _ = fetch_tiles(
        [TileIndex(1, 1, 14), TileIndex(2, 1, 14)],
        httpserver.url_for("/t/{z}/{x}/{y}.png"),
        policy="ignore",
    )
    assert failed == 1 and len(results) == 2


def test_lenient_stops_as_soon_as_too_many_tiles_failed(httpserver: Any) -> None:
    httpserver.expect_request(re.compile(r"/t/14/\d+/1\.png")).respond_with_data(
        b"gone", status=404
    )
    callbacks: list[int] = []
    tiles = [TileIndex(x, 1, 14) for x in range(400)]
    with pytest.raises(RuntimeError, match="Too many failed tiles: "):
        fetch_tiles(
            tiles,
            httpserver.url_for("/t/{z}/{x}/{y}.png"),
            callback=callbacks.append,
            max_connections=1,
            policy="lenient",
            max_failed_ratio=0.05,
        )
    assert len(callbacks) <= 21  # more than 5% of 400 would be 21 failures; it stops at 21


def test_a_server_that_never_answers_is_given_up_on_within_a_minute(
    raw_server: Callable[[Callable[[Any], None]], str],
) -> None:
    def never_answers(handler: Any) -> None:
        time.sleep(120)

    template = raw_server(never_answers)
    started = time.monotonic()
    with pytest.raises(RuntimeError, match="not answering"):
        fetch_tiles(
            [TileIndex(x, 1, 14) for x in range(100)],
            template,
            max_connections=8,
            policy="ignore",
        )
    assert time.monotonic() - started < 60
