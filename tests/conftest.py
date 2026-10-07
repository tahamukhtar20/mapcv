"""Shared test setup."""

from __future__ import annotations

import pytest

from mapcv.tile_cache import CACHE_ENV

try:
    from hypothesis import settings
except ImportError:  # the release tests run without the test group
    pass
else:
    # Reproducible by default: the same examples on every run and machine. Use
    # `pytest --hypothesis-profile=explore` to search with fresh random examples.
    settings.register_profile("default", derandomize=True, deadline=None, print_blob=True)
    settings.register_profile("explore", deadline=None, max_examples=2000, print_blob=True)
    settings.load_profile("default")


@pytest.fixture(autouse=True)
def _private_tile_cache(
    monkeypatch: pytest.MonkeyPatch, tmp_path_factory: pytest.TempPathFactory
) -> None:
    """Every test gets an empty tile cache of its own, never the user's."""
    monkeypatch.setenv(CACHE_ENV, str(tmp_path_factory.mktemp("tile-cache")))


@pytest.fixture(autouse=True)
def _plain_usage_errors(monkeypatch: pytest.MonkeyPatch) -> None:
    """Typer forces colour into its usage errors on GitHub Actions (GITHUB_ACTIONS), which
    splits the text tests look for with escape codes; render them as in a pipe."""
    try:
        from typer import rich_utils
    except ImportError:  # pragma: no cover - Typer without Rich
        return
    monkeypatch.setattr(rich_utils, "FORCE_TERMINAL", False, raising=False)
