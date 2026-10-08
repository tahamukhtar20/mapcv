"""Mosaic file patterns (``imagery.path``): a path that exists is that file; ``**`` does not
loop through symbolic links; ordinary patterns match what ``glob`` matches."""

from __future__ import annotations

import glob
import os
import time
from pathlib import Path

import pytest

from mapcv.config import GeoTiffImageryConfig
from mapcv.filepattern import is_pattern, matching_files

needs_symlinks = pytest.mark.skipif(os.name == "nt", reason="symbolic links need privileges")


def _touch(*paths: Path) -> None:
    for path in paths:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"x")


def _config(path: Path) -> GeoTiffImageryConfig:
    return GeoTiffImageryConfig(path=str(path))


def test_a_file_with_brackets_in_its_name_is_that_file(tmp_path: Path) -> None:
    _touch(tmp_path / "img[12].tif", tmp_path / "img1.tif", tmp_path / "img2.tif")
    config = _config(tmp_path / "img[12].tif")
    assert not config.is_pattern
    assert config.files() == [str(tmp_path / "img[12].tif")]


def test_a_bracket_pattern_still_matches_when_no_such_file_exists(tmp_path: Path) -> None:
    _touch(tmp_path / "img1.tif", tmp_path / "img2.tif", tmp_path / "img3.tif")
    config = _config(tmp_path / "img[12].tif")
    assert config.is_pattern
    assert [Path(found).name for found in config.files()] == ["img1.tif", "img2.tif"]


def test_a_folder_with_brackets_in_its_name_is_that_folder(tmp_path: Path) -> None:
    _touch(tmp_path / "survey [2024]" / "a.tif", tmp_path / "survey [2024]" / "b.tif")
    _touch(tmp_path / "survey 2" / "other.tif")
    literal = _config(tmp_path / "survey [2024]" / "ortho.tif")
    assert not literal.is_pattern
    mosaic = _config(tmp_path / "survey [2024]" / "*.tif")
    assert [Path(found).name for found in mosaic.files()] == ["a.tif", "b.tif"]
    assert {Path(found).parent.name for found in mosaic.files()} == {"survey [2024]"}


def test_a_folder_with_brackets_is_a_folder_not_a_pattern(tmp_path: Path) -> None:
    (tmp_path / "scenes [a]").mkdir()
    with pytest.raises(ValueError, match="is a folder"):
        _config(tmp_path / "scenes [a]").files()


def test_a_missing_file_says_no_files_match(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match="no files match"):
        _config(tmp_path / "img[12].tif").files()


@needs_symlinks
def test_a_recursive_pattern_does_not_follow_folder_links(tmp_path: Path) -> None:
    root = tmp_path / "t"
    _touch(root / "img.tif", root / "sub" / "deep.tif")
    (root / "loop").symlink_to("..", target_is_directory=True)
    (root / "loop2").symlink_to(".", target_is_directory=True)
    started = time.monotonic()
    found = _config(root / "**" / "*.tif").files()
    assert time.monotonic() - started < 5
    assert [Path(path).relative_to(root).as_posix() for path in found] == [
        "img.tif",
        "sub/deep.tif",
    ]


@needs_symlinks
def test_a_file_reached_through_several_links_is_listed_once(tmp_path: Path) -> None:
    _touch(tmp_path / "a.tif", tmp_path / "other.tif")
    (tmp_path / "b.tif").symlink_to("a.tif")
    (tmp_path / "c.tif").symlink_to(tmp_path / "other.tif")
    found = [Path(path).name for path in _config(tmp_path / "*.tif").files()]
    assert found == ["a.tif", "c.tif"] or found == ["a.tif", "other.tif"]
    assert len(found) == 2


@pytest.mark.parametrize(
    "pattern",
    [
        "*.tif",
        "img?.tif",
        "img[12].tif",
        "img[!1].tif",
        "*/*.tif",
        "**/*.tif",
        "a/**/*.tif",
        "**/deep/*.tif",
        "**",
        "a/**",
        "?/b/*.tif",
        "nothing*.tif",
        "[ab]/**/*.tif",
    ],
)
def test_ordinary_patterns_match_what_glob_matches(tmp_path: Path, pattern: str) -> None:
    _touch(
        tmp_path / "img1.tif",
        tmp_path / "img2.tif",
        tmp_path / "img10.tif",
        tmp_path / ".hidden.tif",
        tmp_path / "a" / "x.tif",
        tmp_path / "a" / "b" / "y.tif",
        tmp_path / "a" / "b" / "deep" / "z.tif",
        tmp_path / "b" / "b" / "w.tif",
        tmp_path / ".cache" / "v.tif",
    )
    expected = sorted(
        found
        for found in glob.glob(str(tmp_path / pattern), recursive=True)
        if Path(found).is_file()
    )
    assert matching_files(tmp_path / pattern) == expected
    assert is_pattern(tmp_path / pattern)


def test_a_plain_path_is_not_a_pattern(tmp_path: Path) -> None:
    _touch(tmp_path / "scene.tif")
    assert not is_pattern(tmp_path / "scene.tif")
    assert not is_pattern(tmp_path / "missing.tif")
