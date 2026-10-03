"""Tests for the async tile fetcher (requires pytest-httpserver)."""

from __future__ import annotations

import pytest
from typing import Any
from mapcv._mapcv_rs import PyTileIndex, fetch_tiles

# Minimal valid PNG signatures followed by distinct payloads; the fetcher only
# checks the signature, decoding happens later.
PNG_1 = b"\x89PNG\r\n\x1a\n" + b"tile-1"
PNG_2 = b"\x89PNG\r\n\x1a\n" + b"tile-2"


def test_fetch_tiles_mock(httpserver: Any) -> None:
    httpserver.expect_request("/tile/14/2621/6331.png").respond_with_data(PNG_1)
    httpserver.expect_request("/tile/14/2622/6331.png").respond_with_data(PNG_2)

    url_template = httpserver.url_for("/tile/{z}/{x}/{y}.png")

    tile_list = [
        PyTileIndex(2621, 6331, 14),
        PyTileIndex(2622, 6331, 14),
    ]

    progress_updates: list[int] = []

    def callback(c: int) -> None:
        progress_updates.append(c)

    results, failed = fetch_tiles(
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
        PyTileIndex(1, 1, 14),
        PyTileIndex(2, 1, 14),
    ]

    results, failed = fetch_tiles(
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
    tile_list = [PyTileIndex(1, 1, 14)]

    with pytest.raises(RuntimeError, match="HTTP 404"):
        fetch_tiles(tile_list, url_template, callback=None, max_connections=2, policy="strict")


def test_fetch_tiles_ignore_returns_black_pixels(httpserver: Any) -> None:
    """Ignore policy: 404 tiles still appear in results as non-empty black PNG bytes."""
    httpserver.expect_request("/tile/14/1/1.png").respond_with_data(b"NOT FOUND", status=404)

    url_template = httpserver.url_for("/tile/{z}/{x}/{y}.png")
    tile_list = [PyTileIndex(1, 1, 14)]

    results, failed = fetch_tiles(
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
    tile_list = [PyTileIndex(1, 1, 14), PyTileIndex(2, 1, 14)]

    # 2/2 failed = 100 % > 0 % threshold, but policy is "ignore" so it should pass
    results, failed = fetch_tiles(
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
        PyTileIndex(1, 1, 14),
        PyTileIndex(2, 1, 14),
        PyTileIndex(3, 1, 14),
    ]

    # 2/3 ~= 66.7% > 50% threshold
    with pytest.raises(RuntimeError, match="Too many failed tiles"):
        fetch_tiles(
            tile_list,
            url_template,
            callback=None,
            max_connections=3,
            policy="lenient",
            max_failed_ratio=0.5,
        )


def test_fetch_tiles_max_failed_ratio_not_exceeded(httpserver: Any) -> None:
    """Under max_failed_ratio threshold: results returned normally."""
    httpserver.expect_request("/tile/14/1/1.png").respond_with_data(PNG_1, status=200)
    httpserver.expect_request("/tile/14/2/1.png").respond_with_data(b"NOT FOUND", status=404)

    url_template = httpserver.url_for("/tile/{z}/{x}/{y}.png")
    tile_list = [PyTileIndex(1, 1, 14), PyTileIndex(2, 1, 14)]

    # 1/2 = 50 % which is not > 60 % threshold
    results, failed = fetch_tiles(
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
    results, failed = fetch_tiles(
        [PyTileIndex(1, 1, 14)],
        httpserver.url_for("/tile/{z}/{x}/{y}.png"),
        policy="lenient",
        max_failed_ratio=1.0,
    )
    assert results == []
    assert failed == 1

    with pytest.raises(RuntimeError, match="not an image"):
        fetch_tiles(
            [PyTileIndex(1, 1, 14)], httpserver.url_for("/tile/{z}/{x}/{y}.png"), policy="strict"
        )


def test_fetch_tiles_lenient_ratio_is_case_insensitive(httpserver: Any) -> None:
    httpserver.expect_request("/tile/14/1/1.png").respond_with_data(b"NOT FOUND", status=404)
    with pytest.raises(RuntimeError, match="Too many failed tiles"):
        fetch_tiles(
            [PyTileIndex(1, 1, 14)],
            httpserver.url_for("/tile/{z}/{x}/{y}.png"),
            policy="LENIENT",
            max_failed_ratio=0.0,
        )


def test_fetch_tiles_invalid_arguments_raise_value_error() -> None:
    with pytest.raises(ValueError, match="Unknown policy"):
        fetch_tiles([PyTileIndex(1, 1, 14)], "http://localhost/{z}/{x}/{y}.png", policy="nope")
    with pytest.raises(ValueError, match="max_connections"):
        fetch_tiles([PyTileIndex(1, 1, 14)], "http://localhost/{z}/{x}/{y}.png", max_connections=0)


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
        results, failed = fetch_tiles(
            [PyTileIndex(1, 1, 14)],
            f"http://127.0.0.1:{port}/{{z}}/{{x}}/{{y}}.png",
            policy="ignore",
        )
    finally:
        stop.set()
        thread.join()
        server.close()
    assert failed == 1
    assert len(results) == 1  # black fill under the ignore policy
