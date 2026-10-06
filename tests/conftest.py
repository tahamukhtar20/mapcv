"""Shared test setup."""

from __future__ import annotations

import pytest

from mapcv.tile_cache import CACHE_ENV


@pytest.fixture(autouse=True)
def _private_tile_cache(
    monkeypatch: pytest.MonkeyPatch, tmp_path_factory: pytest.TempPathFactory
) -> None:
    """Every test gets an empty tile cache of its own, never the user's."""
    monkeypatch.setenv(CACHE_ENV, str(tmp_path_factory.mktemp("tile-cache")))
