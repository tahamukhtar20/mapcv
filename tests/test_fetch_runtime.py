"""The tile fetcher's shared runtime: connections carry over between ``fetch_tiles``
calls (each chunk of a run is one call), and a forked child process fetches with its
own runtime instead of hanging on the parent's."""

from __future__ import annotations

import io
import multiprocessing
import sys
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any, Iterator, Set

import numpy as np
import pytest
from PIL import Image

from mapcv._mapcv_rs import TileIndex, fetch_tiles

_buffer = io.BytesIO()
Image.fromarray(np.zeros((256, 256, 3), np.uint8)).save(_buffer, "PNG")
PNG = _buffer.getvalue()


class Server:
    def __init__(self) -> None:
        self.ports: Set[int] = set()
        owner = self

        class Handler(BaseHTTPRequestHandler):
            protocol_version = "HTTP/1.1"  # keep-alive
            wbufsize = 1 << 16
            disable_nagle_algorithm = True

            def do_GET(self) -> None:  # noqa: N802
                owner.ports.add(self.client_address[1])
                self.send_response(200)
                self.send_header("Content-Type", "image/png")
                self.send_header("Content-Length", str(len(PNG)))
                self.end_headers()
                self.wfile.write(PNG)

            def log_message(self, *args: Any) -> None:
                pass

        self.server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self.template = f"http://127.0.0.1:{self.server.server_address[1]}/{{z}}/{{x}}/{{y}}.png"
        threading.Thread(target=self.server.serve_forever, daemon=True).start()


@pytest.fixture
def server() -> Iterator[Server]:
    served = Server()
    yield served
    served.server.shutdown()
    served.server.server_close()


def test_connections_carry_over_between_calls(server: Server) -> None:
    for call in range(10):
        results, failed, _ = fetch_tiles(
            [TileIndex(x, call, 12) for x in range(8)], server.template, max_connections=2
        )
        assert failed == 0 and len(results) == 8
    # 80 requests over at most a few connections; a client per call would open 2 each.
    assert len(server.ports) <= 4


def _child_fetch(template: str, queue: Any) -> None:
    results, failed, _ = fetch_tiles([TileIndex(x, 0, 12) for x in range(4)], template)
    queue.put((len(results), failed))


@pytest.mark.skipif(
    sys.platform != "linux", reason="fork is the default start method only on Linux"
)
def test_a_forked_child_fetches_with_its_own_runtime(server: Server) -> None:
    # The parent's runtime exists before the fork; the child must not use it.
    assert fetch_tiles([TileIndex(0, 0, 12)], server.template)[1] == 0
    context = multiprocessing.get_context("fork")
    queue = context.Queue()
    child = context.Process(target=_child_fetch, args=(server.template, queue))
    child.start()
    child.join(timeout=60)
    alive = child.is_alive()
    if alive:
        child.kill()
    assert not alive, "the forked child hung on the parent's runtime"
    assert child.exitcode == 0 and queue.get(timeout=5) == (4, 0)
