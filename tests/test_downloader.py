"""Tests for tile-strip iteration and download helpers."""

from __future__ import annotations

from typing import List

import pytest
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
