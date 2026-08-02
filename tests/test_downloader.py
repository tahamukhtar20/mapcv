"""Tests for tile-strip iteration and download helpers."""

from __future__ import annotations

from typing import Any, List, Tuple

import pytest

from mapcv import downloader
from mapcv._mapcv_rs import PyTileIndex
from mapcv.downloader import iter_tile_strips, resolve_url_template


def test_resolve_url_template_explicit() -> None:
    template = "https://example.com/{z}/{x}/{y}"
    assert resolve_url_template(template, None) == template


def test_resolve_url_template_source() -> None:
    assert (
        resolve_url_template(None, "osm")
        == "https://tile.openstreetmap.org/{z}/{x}/{y}.png"
    )


def test_resolve_url_template_precedence() -> None:
    template = "https://example.com/{z}/{x}/{y}"
    assert resolve_url_template(template, "osm") == template


def test_resolve_url_template_invalid_source() -> None:
    with pytest.raises(ValueError, match="Unknown tile source: invalid_source"):
        resolve_url_template(None, "invalid_source")


def test_resolve_url_template_missing_both() -> None:
    with pytest.raises(ValueError, match="Provide url_template or source"):
        resolve_url_template(None, None)


def test_iter_tile_strips_groups_rows() -> None:
    strips = iter_tile_strips(-0.1, -0.1, 0.1, 0.1, 6, strip_rows=2)
    assert strips
    flat_tiles = [tile for strip in strips for tile in strip]
    rows: List[int] = sorted({tile.y for tile in flat_tiles})
    strip_rows = [sorted({tile.y for tile in strip}) for strip in strips]

    assert sum(len(rows) for rows in strip_rows) == len(rows)
    assert all(len(rows) <= 2 for rows in strip_rows)

    first_rows = strip_rows[0]
    assert first_rows == rows[: len(first_rows)]


def test_iter_tile_strips_invalid_size() -> None:
    with pytest.raises(ValueError, match="strip_rows must be positive"):
        iter_tile_strips(-0.1, -0.1, 0.1, 0.1, 6, strip_rows=0)


def _stub_strip_downloads(
    monkeypatch: pytest.MonkeyPatch,
    outcomes: List[Tuple[List[Tuple[PyTileIndex, bytes]], int]],
) -> None:
    strips = [[PyTileIndex(index, 0, 1)] for index in range(len(outcomes))]
    pending = iter(outcomes)

    def fake_iter_strips(*args: Any, **kwargs: Any) -> List[List[PyTileIndex]]:
        return strips

    def fake_fetch(*args: Any, **kwargs: Any) -> Tuple[List[Tuple[PyTileIndex, bytes]], int]:
        return next(pending)

    monkeypatch.setattr(downloader, "_in_jupyter", lambda: True)
    monkeypatch.setattr(downloader, "iter_tile_strips", fake_iter_strips)
    monkeypatch.setattr(downloader, "fetch_tiles_rs", fake_fetch)


def test_download_strips_enforces_global_lenient_ratio(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    tile = PyTileIndex(0, 0, 1)
    _stub_strip_downloads(monkeypatch, [([(tile, b"ok")], 0), ([], 1), ([], 1)])

    with pytest.raises(RuntimeError, match="2/3"):
        downloader.download_region_strips(
            -1,
            -1,
            1,
            1,
            1,
            1,
            url_template="https://example.com/{z}/{x}/{y}.png",
            policy="lenient",
            snap_to_tiles=False,
            max_failed_ratio=0.5,
        )


def test_download_strips_ignore_bypasses_global_ratio(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    tiles = [PyTileIndex(index, 0, 1) for index in range(2)]
    _stub_strip_downloads(
        monkeypatch,
        [([(tiles[0], b"black")], 1), ([(tiles[1], b"black")], 1)],
    )

    results = downloader.download_region_strips(
        -1,
        -1,
        1,
        1,
        1,
        1,
        url_template="https://example.com/{z}/{x}/{y}.png",
        policy="ignore",
        snap_to_tiles=False,
        max_failed_ratio=0.0,
    )

    assert sum(len(strip) for strip in results) == 2
