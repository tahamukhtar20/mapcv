"""Tests for the region download and stitch helpers."""

from __future__ import annotations

from typing import Any, List, Tuple

import pytest

from mapcv import downloader
from mapcv._mapcv_rs import PyTileIndex
from mapcv.downloader import download_region, resolve_url_template, stitch_region


def test_resolve_url_template_explicit() -> None:
    template = "https://example.com/{z}/{x}/{y}"
    assert resolve_url_template(template, None) == template


def test_resolve_url_template_source() -> None:
    assert (
        resolve_url_template(None, "esri_satellite") == downloader.URL_TEMPLATES["esri_satellite"]
    )


def test_resolve_url_template_precedence() -> None:
    template = "https://example.com/{z}/{x}/{y}"
    assert resolve_url_template(template, "esri_satellite") == template


def test_resolve_url_template_invalid_source() -> None:
    with pytest.raises(ValueError, match="Unknown tile source: invalid_source"):
        resolve_url_template(None, "invalid_source")


def test_resolve_url_template_missing_both() -> None:
    with pytest.raises(ValueError, match="Provide url_template or source"):
        resolve_url_template(None, None)


def _png(value: int) -> bytes:
    from io import BytesIO

    import numpy as np
    from PIL import Image

    buffer = BytesIO()
    Image.fromarray(np.full((256, 256, 3), value, dtype=np.uint8)).save(buffer, format="PNG")
    return buffer.getvalue()


def _fake_fetch(calls: List[Tuple[int, str]]) -> Any:
    def fetch(
        requested: List[PyTileIndex], template: str, **kwargs: Any
    ) -> Tuple[List[Tuple[PyTileIndex, bytes]], int, str]:
        calls.append((len(requested), template))
        return [(tile, _png(tile.x % 200)) for tile in requested], 0, ""

    return fetch


def test_download_region_fetches_every_tile_in_the_snapped_bbox(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: List[Tuple[int, str]] = []
    monkeypatch.setattr(downloader, "fetch_tiles_rs", _fake_fetch(calls))
    results = download_region(4.88, 52.37, 4.89, 52.375, 15, source="esri_satellite")
    assert calls == [(len(results), downloader.URL_TEMPLATES["esri_satellite"])]
    assert len({(tile.x, tile.y) for tile, _ in results}) == len(results) > 0


def test_stitch_region_returns_image_and_transform(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(downloader, "fetch_tiles_rs", _fake_fetch([]))
    image, transform = stitch_region(4.88, 52.37, 4.89, 52.375, 15, source="esri_satellite")
    assert image.ndim == 3 and image.shape[2] == 3
    assert image.shape[0] % 256 == 0 and image.shape[1] % 256 == 0
    assert transform[0] > 0 and transform[4] < 0
