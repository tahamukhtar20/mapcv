"""Tests for the repository maintenance scripts."""

from __future__ import annotations

import importlib.util
import zipfile
from pathlib import Path
from types import ModuleType

import pytest

_SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"


def _load(name: str) -> ModuleType:
    spec = importlib.util.spec_from_file_location(name, _SCRIPTS / f"{name}.py")
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


_CHANGELOG = """# Changelog

## [0.2.0] - 2026-10-20

### Features

- New thing.

## [0.1.0] - 2026-05-04

- First release.
"""


def test_changelog_section_extracts_one_version() -> None:
    section = _load("changelog_section").changelog_section(_CHANGELOG, "0.2.0")
    assert section == "### Features\n\n- New thing.\n"


def test_changelog_section_requires_the_version() -> None:
    with pytest.raises(ValueError, match="no section for version 0.3.0"):
        _load("changelog_section").changelog_section(_CHANGELOG, "0.3.0")


@pytest.mark.parametrize(
    ("title", "ok"),
    [
        ("feat(imagery): add EOPF input", True),
        ("chore(deps): bump the python-dependencies group", True),
        ("ci(deps): bump actions/checkout from 6 to 7", True),
        ("Bump numpy from 2.3 to 2.4", False),
        ("build: support Python 3.14", False),
    ],
)
def test_pr_title_policy(title: str, ok: bool) -> None:
    pattern = _load("check_pr_title").TITLE_PATTERN
    assert bool(pattern.fullmatch(title)) is ok


def _wheel(path: Path, files: dict[str, bytes]) -> Path:
    """Write a minimal wheel holding ``files`` and a RECORD with their hashes."""
    record_hash = _load("check_installed_wheel").record_hash
    record = "".join(f"{name},{record_hash(data)},{len(data)}\n" for name, data in files.items())
    with zipfile.ZipFile(path, "w") as zf:
        for name, data in files.items():
            zf.writestr(name, data)
        zf.writestr("mapcv-0.2.0.dist-info/RECORD", record + "mapcv-0.2.0.dist-info/RECORD,,\n")
    return path


def test_check_installed_wheel_finds_the_wheel_with_the_same_module(tmp_path: Path) -> None:
    script = _load("check_installed_wheel")
    _wheel(tmp_path / "mapcv-0.2.0-cp310-abi3-win_amd64.whl", {"mapcv/_mapcv_rs.pyd": b"win"})
    linux = _wheel(
        tmp_path / "mapcv-0.2.0-cp310-abi3-manylinux_2_17_x86_64.whl",
        {"mapcv/_mapcv_rs.abi3.so": b"local build"},
    )
    digest = script.record_hash(b"local build")
    assert script.find_wheel(tmp_path, "mapcv/_mapcv_rs.abi3.so", digest) == linux


def test_check_installed_wheel_rejects_a_different_build(tmp_path: Path) -> None:
    script = _load("check_installed_wheel")
    _wheel(
        tmp_path / "mapcv-0.2.0-cp310-abi3-manylinux_2_17_x86_64.whl",
        {"mapcv/_mapcv_rs.abi3.so": b"local build"},
    )
    digest = script.record_hash(b"published build")
    assert script.find_wheel(tmp_path, "mapcv/_mapcv_rs.abi3.so", digest) is None
