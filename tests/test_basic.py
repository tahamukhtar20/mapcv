"""Smoke tests for the package and its Rust extension."""

from __future__ import annotations

import mapcv
from mapcv import _mapcv_rs


def test_extension_loads() -> None:
    assert callable(_mapcv_rs.rasterize)


def test_public_api_names_resolve() -> None:
    for name in mapcv.__all__:
        assert getattr(mapcv, name) is not None
