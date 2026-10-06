"""Malformed GeoTIFFs found by fuzzing must raise ``ValueError``, never panic.

Each file below crashed the Rust reader (an arithmetic overflow, which panics in debug
builds and silently wraps in release builds). They are built byte by byte, so no
imaging library is needed. See ``fuzz/`` for the fuzz targets.
"""

from __future__ import annotations

import struct
from pathlib import Path

import pytest

from mapcv.geotiff import GeoTiff

SHORT, LONG, LONG8 = 3, 4, 16
HUGE = 1 << 31

Tag = tuple[int, int, int]


def inline_tiff(tags: list[Tag], *, bigtiff: bool = False) -> bytes:
    """A little-endian TIFF with one IFD of single-value tags ``(tag, type, value)``."""
    if bigtiff:
        out = b"II" + struct.pack("<HHHQQ", 43, 8, 0, 16, len(tags))
        for tag, field_type, value in tags:
            out += struct.pack("<HHQQ", tag, field_type, 1, value)
        out += struct.pack("<Q", 0)
    else:
        out = b"II" + struct.pack("<HIH", 42, 8, len(tags))
        for tag, field_type, value in tags:
            out += struct.pack("<HHII", tag, field_type, 1, value)
        out += struct.pack("<I", 0)
    return out + bytes(8)


def test_chunk_count_overflow(tmp_path: Path) -> None:
    # 2**31 x 2**31 pixels in 1x1 chunks with 65535 bands in separate planes.
    path = tmp_path / "chunk_count.tif"
    path.write_bytes(
        inline_tiff(
            [
                (256, LONG, HUGE),
                (257, LONG, HUGE),
                (258, SHORT, 8),
                (277, SHORT, 65535),
                (284, SHORT, 2),
                (322, LONG, 1),
                (323, LONG, 1),
                (324, LONG, 0),
                (325, LONG, 0),
            ]
        )
    )
    with pytest.raises(ValueError, match="impossible number of chunks"):
        GeoTiff(path)


def test_huge_tile_size_overflow(tmp_path: Path) -> None:
    path = tmp_path / "huge_tile.tif"
    path.write_bytes(
        inline_tiff(
            [
                (256, LONG, HUGE),
                (257, LONG, HUGE),
                (258, SHORT, 8),
                (277, SHORT, 300),
                (322, LONG, HUGE),
                (323, LONG, HUGE),
                (324, LONG, 8),
                (325, LONG, 8),
            ]
        )
    )
    tif = GeoTiff(path)
    with pytest.raises(ValueError, match="decodes to more than"):
        tif.read_window(0, 1, 0, 1, bands=[0])


def test_huge_strip_row_skip_overflow(tmp_path: Path) -> None:
    path = tmp_path / "huge_strip.tif"
    path.write_bytes(
        inline_tiff(
            [
                (256, LONG, HUGE),
                (257, LONG, HUGE),
                (258, SHORT, 8),
                (277, SHORT, 65535),
                (278, LONG, HUGE),
                (273, LONG, 8),
                (279, LONG, 8),
            ]
        )
    )
    tif = GeoTiff(path)
    with pytest.raises(ValueError, match="decodes to more than"):
        tif.read_window(1 << 20, (1 << 20) + 1, 0, 1, bands=[0])


def test_strip_offset_near_2_pow_64(tmp_path: Path) -> None:
    path = tmp_path / "strip_offset.tif"
    path.write_bytes(
        inline_tiff(
            [
                (256, LONG, 4),
                (257, LONG, 100),
                (258, SHORT, 8),
                (259, SHORT, 1),
                (277, SHORT, 1),
                (278, LONG, 100),
                (273, LONG8, 2**64 - 8),
                (279, LONG8, 400),
            ],
            bigtiff=True,
        )
    )
    tif = GeoTiff(path)
    with pytest.raises(ValueError, match="out of range"):
        tif.read_window(50, 51, 0, 1)
