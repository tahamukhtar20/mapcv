"""Deterministic synthetic XYZ tile server for offline benchmarks.

Every pixel (X, Y) in global zoom-level pixel coordinates has a colour computed
by :func:`expected_rgb`, so any output patch can be checked exactly against the
tiles that were served, with no imagery licence or network involved.

URL layout: ``/{z}/{x}/{y}.png`` (or ``.jpg``). An optional first path segment
injects faults: ``/n50/{z}/{x}/{y}.png`` answers HTTP 404 for the tiles where
:func:`is_failing` holds with ``every=50`` (about 2 % of tiles, always the same
ones), ``/f50/...`` does the same with HTTP 500 (which mapcv retries with a
backoff, so it mostly measures sleeping), ``/l5/...`` adds 5 ms of latency to
every response.
"""

from __future__ import annotations

import argparse
import io
import sys
import threading
import time
from functools import lru_cache
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import numpy as np
import numpy.typing as npt
from PIL import Image

TILE = 256


def expected_rgb(xs: npt.NDArray[np.int64], ys: npt.NDArray[np.int64]) -> npt.NDArray[np.uint8]:
    """Colour of global pixel coordinates (broadcastable int64 arrays).

    The blue channel is always odd, so no pixel is pure black: mapcv treats
    all-zero pixels as "no imagery", which the checks model explicitly for
    failed tiles only.
    """
    red = (xs * 3 + ys) & 255
    green = (xs ^ ys) & 255
    blue = (((xs // 7) * 13 + (ys // 5) * 29) & 255) | 1
    stacked = np.stack(np.broadcast_arrays(red, green, blue), axis=-1)
    return stacked.astype(np.uint8)


def is_failing(x: int, y: int, every: int) -> bool:
    """Whether the fault-injecting server fails tile (x, y); ``every`` of 0 means never."""
    return every > 0 and (x * 31 + y * 17) % every == 0


@lru_cache(maxsize=4096)
def tile_pixels(x: int, y: int) -> npt.NDArray[np.uint8]:
    """The lossless 256 x 256 x 3 content of tile (x, y)."""
    xs = np.arange(x * TILE, (x + 1) * TILE, dtype=np.int64)[None, :]
    ys = np.arange(y * TILE, (y + 1) * TILE, dtype=np.int64)[:, None]
    return expected_rgb(xs, ys)


def encode_tile(x: int, y: int, fmt: str) -> bytes:
    """The bytes the server sends for tile (x, y): PNG, or JPEG quality 90."""
    buffer = io.BytesIO()
    image = Image.fromarray(tile_pixels(x, y))
    if fmt == "png":
        image.save(buffer, format="PNG", compress_level=1)
    else:
        image.save(buffer, format="JPEG", quality=90)
    return buffer.getvalue()


@lru_cache(maxsize=4096)
def decoded_tile(x: int, y: int, fmt: str) -> npt.NDArray[np.uint8]:
    """What a client sees after decoding the served bytes (exact for PNG, lossy for JPEG)."""
    with Image.open(io.BytesIO(encode_tile(x, y, fmt))) as image:
        return np.asarray(image.convert("RGB"), dtype=np.uint8)


_cache_lock = threading.Lock()
_encoded: dict[tuple[int, int, str], bytes] = {}


def _cached_tile(x: int, y: int, fmt: str) -> bytes:
    key = (x, y, fmt)
    body = _encoded.get(key)
    if body is None:
        body = encode_tile(x, y, fmt)
        with _cache_lock:
            _encoded[key] = body
    return body


def _parse(path: str) -> tuple[int, int, int, str, int, int, float] | None:
    """``(z, x, y, ext, fail_every, fail_status, latency_s)`` for a request path, or None."""
    parts = path.split("?")[0].strip("/").split("/")
    fail_every, fail_status, latency = 0, 500, 0.0
    try:
        while parts and parts[0][:1] in ("f", "n", "l"):
            option = parts.pop(0)
            if option[0] in ("f", "n"):
                fail_every = int(option[1:])
                fail_status = 500 if option[0] == "f" else 404
            else:
                latency = int(option[1:]) / 1000
        z, x = int(parts[0]), int(parts[1])
        y_text, ext = parts[2].split(".")
        if ext not in ("png", "jpg"):
            return None
        return z, x, int(y_text), ext, fail_every, fail_status, latency
    except (ValueError, IndexError):
        return None


class Handler(BaseHTTPRequestHandler):
    """Serves the synthetic tiles; quiet and keep-alive free."""

    def log_message(self, format: str, *args: object) -> None:
        pass

    def do_GET(self) -> None:
        parsed = _parse(self.path)
        if parsed is None:
            self.send_error(404)
            return
        _, x, y, ext, fail_every, fail_status, latency = parsed
        if latency:
            time.sleep(latency)
        if is_failing(x, y, fail_every):
            body = b"<html>rate limited</html>"
            self.send_response(fail_status)
            self.send_header("Content-Type", "text/html")
        else:
            body = _cached_tile(x, y, ext)
            self.send_response(200)
            self.send_header("Content-Type", "image/png" if ext == "png" else "image/jpeg")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)


class Server(ThreadingHTTPServer):
    """Threading server that stays quiet when a client hangs up (mapcv's strict policy does)."""

    def handle_error(self, request: object, client_address: object) -> None:
        if not isinstance(sys.exc_info()[1], ConnectionError):
            super().handle_error(request, client_address)  # type: ignore[arg-type]


def make_server(port: int = 0) -> ThreadingHTTPServer:
    """A tile server on 127.0.0.1; ``port`` 0 picks a free one."""
    # The default listen backlog of 5 drops bursts of connections, and the
    # client then waits out a one second SYN retry per dropped connection.
    Server.request_queue_size = 1024
    server = Server(("127.0.0.1", port), Handler)
    server.daemon_threads = True
    return server


def main() -> None:
    """Run a standalone server: ``python -m benchmarks.tileserver --port 8765``."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--port", type=int, default=0, help="0 (default) picks a free port")
    args = parser.parse_args()
    server = make_server(args.port)
    print(server.server_address[1], flush=True)  # the harness reads the port from stdout
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        sys.exit(0)


if __name__ == "__main__":
    main()
