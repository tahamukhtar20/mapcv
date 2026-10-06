"""Tests that the licence policy files agree with each other."""

from __future__ import annotations

from pathlib import Path

import pytest

_ROOT = Path(__file__).resolve().parents[1]


def test_about_accepts_exactly_what_deny_allows() -> None:
    """The cargo-about list and the cargo-deny allowlist name the same licences."""
    tomllib = pytest.importorskip("tomllib")  # Python 3.11+
    deny = _ROOT / "deny.toml"
    about = _ROOT / "about.toml"
    if not deny.exists() or not about.exists():
        pytest.skip("not a source checkout (the sdist leaves out the licence tooling)")
    allowed = tomllib.loads(deny.read_text(encoding="utf-8"))["licenses"]["allow"]
    accepted = tomllib.loads(about.read_text(encoding="utf-8"))["accepted"]
    assert sorted(accepted) == sorted(allowed)
