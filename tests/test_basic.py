"""Smoke test: verify the Rust extension loads and responds correctly."""

from __future__ import annotations

from mapcv import hello


def test_hello() -> None:
    assert hello() == "Hello from mapcv Rust core!"
