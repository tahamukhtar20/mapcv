"""Security and robustness tests."""

from __future__ import annotations
import pytest
import numpy as np
from PIL import Image
import io
from mapcv._mapcv_rs import PyTileIndex, stitch_tiles, fetch_tiles

def _make_tile(r: int, g: int, b: int) -> bytes:
    """Return PNG bytes for a solid-colour 256×256 RGB tile."""
    img = Image.fromarray(np.full((256, 256, 3), [r, g, b], dtype=np.uint8), mode="RGB")
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()

RED = _make_tile(255, 0, 0)

def test_url_sanitization_on_failure(httpserver):
    """Query parameters should be redacted from error messages."""
    httpserver.expect_request("/tile/0/0/0.png").respond_with_data("Not Found", status=404)

    # URL with sensitive API key in query param
    base_url = httpserver.url_for("/tile/{z}/{x}/{y}.png")
    url_template = f"{base_url}?api_key=SECRET_123"

    t = PyTileIndex(0, 0, 0)

    with pytest.raises(RuntimeError) as excinfo:
        fetch_tiles(
            tiles=[t],
            url_template=url_template,
            policy="strict"
        )

    error_msg = str(excinfo.value)
    assert "SECRET_123" not in error_msg
    assert "api_key" not in error_msg
    assert httpserver.url_for("/tile/0/0/0.png") in error_msg

def test_stitch_oom_prevention():
    """stitch_tiles should reject excessively large ranges to prevent OOM."""
    # 16385 tiles in one dimension is just over our 16384 limit
    t1 = PyTileIndex(0, 0, 16)
    t2 = PyTileIndex(16384, 0, 16) # range is max-min+1 = 16384-0+1 = 16385

    with pytest.raises(RuntimeError) as excinfo:
        stitch_tiles([(t1, RED), (t2, RED)])

    assert "tile range too large" in str(excinfo.value)

def test_stitch_usize_overflow_prevention():
    """stitch_tiles should handle coordinate differences that would overflow u32."""
    t1 = PyTileIndex(0, 0, 31)
    t2 = PyTileIndex(4294967295, 0, 31)

    with pytest.raises(RuntimeError) as excinfo:
        stitch_tiles([(t1, RED), (t2, RED)])

    assert "tile range too large" in str(excinfo.value)
