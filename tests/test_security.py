"""Security and robustness tests."""

from __future__ import annotations

import io
from typing import Any

import numpy as np
import pytest
from PIL import Image

from mapcv._mapcv_rs import PyTileIndex, fetch_tiles, stitch_tiles


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

    t = PyTileIndex(0, 0, 0)

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
    assert httpserver.url_for("/tile/0/0/0.png") in error_msg


def test_stitch_rejects_canvas_above_byte_budget() -> None:
    """A sparse tile extent must be rejected before a huge canvas is allocated."""
    t1 = PyTileIndex(0, 0, 16)
    t2 = PyTileIndex(2731, 0, 16)

    with pytest.raises(RuntimeError, match="exceeding"):
        stitch_tiles([(t1, RED), (t2, RED)])


def test_stitch_usize_overflow_prevention() -> None:
    """stitch_tiles should handle coordinate differences that would overflow u32."""
    t1 = PyTileIndex(0, 0, 31)
    t2 = PyTileIndex(4294967295, 0, 31)

    with pytest.raises(RuntimeError, match="canvas"):
        stitch_tiles([(t1, RED), (t2, RED)])
