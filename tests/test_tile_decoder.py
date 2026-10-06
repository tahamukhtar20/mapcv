"""The Rust tile decoder must give exactly the pixels Pillow's ``convert("RGB")`` gives.

``XYZRasterSource.read_window`` decodes the tiles Rust can match to Pillow bit for bit
(8-bit PNG, WebP, GIF) and leaves every other tile (JPEG, 16-bit PNG, ...) to Pillow.
These tests check both halves: identical pixels where Rust decodes, and that Rust
declines the rest.
"""

from __future__ import annotations

import io
import struct
import threading
import time
import zlib
from types import SimpleNamespace
from typing import Any, Dict, Iterator, List, Optional, Tuple

import numpy as np
import numpy.typing as npt
import pytest
from PIL import Image, ImageFile, ImageFilter

from mapcv._mapcv_rs import PyTileIndex, decode_tile_window
from mapcv.config import RegionConfig, XYZImageryConfig
from mapcv.imagery import XYZRasterSource

ImageFile.MAXBLOCK = 2**25  # Pillow's default buffer is too small for noisy progressive JPEGs

pytestmark = pytest.mark.filterwarnings("ignore:Palette images with Transparency")

TILE = 256
U8 = npt.NDArray[np.uint8]


def _content(kind: str) -> U8:
    rng = np.random.default_rng({"noise": 1, "photo": 2, "extreme": 3}[kind])
    if kind == "noise":
        return rng.integers(0, 256, (TILE, TILE, 3), dtype=np.uint8)
    if kind == "extreme":
        return np.where(rng.random((TILE, TILE, 3)) < 0.5, 0, 255).astype(np.uint8)
    blobs = np.kron(rng.normal(size=(16, 16, 3)), np.ones((16, 16, 1)))
    smooth = Image.fromarray(np.clip(blobs * 40 + 128, 0, 255).astype(np.uint8))
    smooth = smooth.filter(ImageFilter.GaussianBlur(6))
    photo = np.asarray(smooth).astype(np.int64) + rng.integers(-6, 7, (TILE, TILE, 3))
    photo[100:140, 50:200] = [250, 10, 10]
    clipped: U8 = np.clip(photo, 0, 255).astype(np.uint8)
    return clipped


KINDS = ["noise", "photo", "extreme"]


def _chunk(kind: bytes, data: bytes) -> bytes:
    return struct.pack(">I", len(data)) + kind + data + struct.pack(">I", zlib.crc32(kind + data))


def _png(
    samples: npt.NDArray[Any],
    color_type: int,
    depth: int,
    palette: Optional[npt.NDArray[Any]] = None,
    trns: Optional[bytes] = None,
    interlace: bool = False,
) -> bytes:
    """Write a PNG of any colour type and bit depth (Pillow cannot write most of them)."""
    channels = {0: 1, 2: 3, 3: 1, 4: 2, 6: 4}[color_type]
    samples = samples.reshape(TILE, TILE, channels)

    def pack(rows: npt.NDArray[Any]) -> bytes:
        out = []
        for row in rows:
            if depth == 16:
                out.append(b"\0" + row.astype(">u2").tobytes())
            elif depth == 8:
                out.append(b"\0" + row.astype(np.uint8).tobytes())
            else:
                per_byte = 8 // depth
                flat = row.reshape(-1).astype(np.uint8)
                flat = np.concatenate([flat, np.zeros((-len(flat)) % per_byte, np.uint8)])
                flat = flat.reshape(-1, per_byte)
                packed = np.zeros(len(flat), np.uint8)
                for i in range(per_byte):
                    packed |= flat[:, i] << (8 - depth * (i + 1))
                out.append(b"\0" + packed.tobytes())
        return b"".join(out)

    if interlace:  # Adam7 passes as (row start, col start, row step, col step)
        passes = [(0, 0, 8, 8), (0, 4, 8, 8), (4, 0, 8, 4), (0, 2, 4, 4)]
        passes += [(2, 0, 4, 2), (0, 1, 2, 2), (1, 0, 2, 1)]
        raw = b"".join(pack(samples[r0::dr, c0::dc]) for r0, c0, dr, dc in passes)
    else:
        raw = pack(samples)
    out = b"\x89PNG\r\n\x1a\n"
    out += _chunk(b"IHDR", struct.pack(">IIBBBBB", TILE, TILE, depth, color_type, 0, 0, interlace))
    if palette is not None:
        out += _chunk(b"PLTE", palette.astype(np.uint8).tobytes())
    if trns is not None:
        out += _chunk(b"tRNS", trns)
    return out + _chunk(b"IDAT", zlib.compress(raw)) + _chunk(b"IEND", b"")


def _encode(image: Image.Image, fmt: str, **options: Any) -> bytes:
    buffer = io.BytesIO()
    image.save(buffer, format=fmt, **options)
    return buffer.getvalue()


def _png_cases(kind: str) -> Iterator[Tuple[str, bytes]]:
    rgb = _content(kind)
    gray, alpha = rgb[..., 0], rgb[..., 1]
    rng = np.random.default_rng(7)
    yield "png-rgb", _png(rgb, 2, 8)
    yield "png-rgb-interlaced", _png(rgb, 2, 8, interlace=True)
    yield "png-rgba", _png(np.dstack([rgb, alpha]), 6, 8)
    yield "png-l", _png(gray, 0, 8)
    yield "png-la", _png(np.dstack([gray, alpha]), 4, 8)
    yield "png-rgb-colorkey", _png(rgb, 2, 8, trns=struct.pack(">HHH", *map(int, rgb[0, 0])))
    yield "png-l-colorkey", _png(gray, 0, 8, trns=struct.pack(">H", int(gray[0, 0])))
    for depth in (1, 2, 4):
        yield f"png-l{depth}bit", _png(gray >> (8 - depth), 0, depth)
        palette = rng.integers(0, 256, (2**depth, 3))
        indices = rgb[..., 0] % (2**depth)
        yield f"png-p{depth}bit", _png(indices, 3, depth, palette)
        alphas = bytes(rng.integers(0, 256, max(1, 2**depth // 2)).astype(np.uint8))
        yield f"png-p{depth}bit-trns", _png(indices, 3, depth, palette, trns=alphas)
    palette = rng.integers(0, 256, (256, 3))
    yield "png-p8", _png(gray, 3, 8, palette)
    yield "png-p8-interlaced", _png(gray, 3, 8, palette, interlace=True)
    yield (
        "png-p8-trns",
        _png(gray, 3, 8, palette, trns=bytes(rng.integers(0, 256, 256).astype(np.uint8))),
    )
    yield "png-p8-trns-partial", _png(gray, 3, 8, palette, trns=bytes([0, 128]))
    yield "png-pillow-rgb", _encode(Image.fromarray(rgb), "PNG", compress_level=1)
    yield "png-pillow-p", _encode(Image.fromarray(rgb).quantize(64), "PNG")


def _other_cases(kind: str) -> Iterator[Tuple[str, bytes]]:
    rgb = _content(kind)
    image = Image.fromarray(rgb)
    rgba = Image.fromarray(np.dstack([rgb, rgb[..., 1]]))
    yield "webp-lossless", _encode(image, "WEBP", lossless=True)
    yield "webp-lossless-alpha", _encode(rgba, "WEBP", lossless=True)
    for quality in (5, 50, 80, 100):
        yield f"webp-lossy-q{quality}", _encode(image, "WEBP", quality=quality, method=4)
    yield "webp-lossy-m0", _encode(image, "WEBP", quality=70, method=0)
    yield "webp-lossy-alpha", _encode(rgba, "WEBP", quality=70)
    yield (
        "webp-animated",
        _encode(image, "WEBP", save_all=True, append_images=[Image.fromarray(_content("noise"))]),
    )
    paletted = image.quantize(256)
    yield "gif", _encode(paletted, "GIF")
    yield "gif-interlaced", _encode(paletted, "GIF", interlace=True)
    yield "gif-transparent", _encode(paletted, "GIF", transparency=3)
    yield "gif-16-colours", _encode(image.quantize(16), "GIF")
    yield "gif-grey", _encode(Image.fromarray(rgb[..., 0]), "GIF")


# Everything Rust must decode, and decode exactly like Pillow.
RUST_CASES = [
    pytest.param(data, id=f"{name}-{kind}")
    for kind in KINDS
    for name, data in (*_png_cases(kind), *_other_cases(kind))
]


def _jpeg_cases(kind: str) -> Iterator[Tuple[str, bytes]]:
    image = Image.fromarray(_content(kind))
    for quality in (50, 75, 95):
        for subsampling in (0, 1, 2):
            yield (
                f"jpeg-q{quality}-s{subsampling}",
                _encode(image, "JPEG", quality=quality, subsampling=subsampling),
            )
    yield "jpeg-progressive", _encode(image, "JPEG", quality=85, progressive=True)
    yield "jpeg-optimized", _encode(image, "JPEG", quality=85, optimize=True)
    yield "jpeg-grey", _encode(image.convert("L"), "JPEG", quality=85)
    yield "jpeg-cmyk", _encode(image.convert("CMYK"), "JPEG", quality=85)


def _pillow_only_cases(kind: str) -> Iterator[Tuple[str, bytes]]:
    rgb = _content(kind)
    for name, color_type, channels in (
        ("l16", 0, 1),
        ("la16", 4, 2),
        ("rgb16", 2, 3),
        ("rgba16", 6, 4),
    ):
        samples = np.dstack([rgb.astype(np.int64)[..., i % 3] * 257 + 5 for i in range(channels)])
        yield f"png-{name}", _png(samples, color_type, 16)
    yield "bmp", _encode(Image.fromarray(rgb), "BMP")


# Everything Rust must leave to Pillow, because its pixels are not guaranteed to match.
PILLOW_CASES = [
    pytest.param(data, id=f"{name}-{kind}")
    for kind in KINDS
    for name, data in (*_jpeg_cases(kind), *_pillow_only_cases(kind))
]


def _pillow(data: bytes) -> U8:
    with Image.open(io.BytesIO(data)) as image:
        return np.asarray(image.convert("RGB"), dtype=np.uint8)


@pytest.mark.parametrize("data", RUST_CASES)
def test_rust_decode_is_identical_to_pillow(data: bytes) -> None:
    window, valid, undecoded = decode_tile_window([(0, 0, data)], 0, 0, 0, TILE, 0, TILE)

    assert undecoded == []
    expected = _pillow(data)
    assert window.shape == (TILE, TILE, 3)
    assert window.dtype == np.uint8
    assert np.array_equal(window, expected)
    assert valid.dtype == np.bool_
    assert np.array_equal(valid, np.any(expected != 0, axis=-1))


@pytest.mark.parametrize("data", PILLOW_CASES)
def test_rust_leaves_unmatched_formats_to_pillow(data: bytes) -> None:
    window, valid, undecoded = decode_tile_window([(4, 7, data)], 4, 7, 0, TILE, 0, TILE)

    assert undecoded == [(4, 7)]
    assert not window.any()
    assert not valid.any()


def _gif_with_frame(left: int, top: int, width: int, height: int) -> bytes:
    """A 256x256 GIF whose only frame is ``width x height`` at ``(left, top)``."""
    indices = np.random.default_rng(3).integers(0, 256, (height, width), dtype=np.uint8)
    palette = np.random.default_rng(4).integers(0, 256, (256, 3), dtype=np.uint8)
    out = b"GIF89a" + struct.pack("<HHBBB", TILE, TILE, 0xF7, 0, 0) + palette.tobytes()
    out += b"\x2c" + struct.pack("<HHHHB", left, top, width, height, 0)
    # LZW with 9-bit codes and a clear code every 250 pixels: valid, if uncompressed.
    stream, bits, count = bytearray(), 0, 0

    def put(code: int) -> None:
        nonlocal bits, count
        bits |= code << count
        count += 9
        while count >= 8:
            stream.append(bits & 0xFF)
            bits >>= 8
            count -= 8

    put(256)
    for position, value in enumerate(indices.reshape(-1)):
        put(int(value))
        if position % 250 == 249:
            put(256)
    put(257)
    if count:
        stream.append(bits & 0xFF)
    out += b"\x08"
    for start in range(0, len(stream), 255):
        block = bytes(stream[start : start + 255])
        out += bytes([len(block)]) + block
    return out + b"\x00;"


def test_gif_frames_that_do_not_fill_the_screen_are_left_to_pillow() -> None:
    full = _gif_with_frame(0, 0, TILE, TILE)
    partial = _gif_with_frame(10, 20, 100, 80)

    assert decode_tile_window([(0, 0, full)], 0, 0, 0, TILE, 0, TILE)[2] == []
    assert decode_tile_window([(0, 0, partial)], 0, 0, 0, TILE, 0, TILE)[2] == [(0, 0)]


# Reading windows through XYZRasterSource ------------------------------------------------


def _reference_window(
    tiles: Dict[Tuple[int, int], bytes],
    origin: Tuple[int, int],
    rows: Tuple[int, int],
    cols: Tuple[int, int],
) -> Tuple[U8, npt.NDArray[np.bool_]]:
    """``read_window`` as it was before the Rust decoder: Pillow, one tile at a time."""
    height, width = rows[1] - rows[0], cols[1] - cols[0]
    window = np.zeros((height, width, 3), dtype=np.uint8)
    valid = np.zeros((height, width), dtype=np.bool_)
    for (tile_x, tile_y), payload in sorted(tiles.items(), key=lambda kv: (kv[0][1], kv[0][0])):
        tile = _pillow(payload)
        global_row, global_col = (tile_y - origin[1]) * TILE, (tile_x - origin[0]) * TILE
        r0, c0 = max(0, rows[0] - global_row), max(0, cols[0] - global_col)
        r1, c1 = min(TILE, rows[1] - global_row), min(TILE, cols[1] - global_col)
        if r1 <= r0 or c1 <= c0:
            continue
        t_r0, t_c0 = global_row + r0 - rows[0], global_col + c0 - cols[0]
        window[t_r0 : t_r0 + r1 - r0, t_c0 : t_c0 + c1 - c0] = tile[r0:r1, c0:c1]
        valid[t_r0 : t_r0 + r1 - r0, t_c0 : t_c0 + c1 - c0] = True
    valid &= np.any(window != 0, axis=-1)
    return window, valid


def _source(
    monkeypatch: pytest.MonkeyPatch, grid: List[PyTileIndex], payloads: Dict[Tuple[int, int], bytes]
) -> XYZRasterSource:
    monkeypatch.setattr(
        "mapcv.imagery.snap_bbox",
        lambda *args, **kwargs: SimpleNamespace(west=0, south=0, east=1, north=1),
    )
    monkeypatch.setattr("mapcv.imagery.tiles", lambda *args, **kwargs: grid)
    monkeypatch.setattr(
        "mapcv.imagery.fetch_tiles",
        lambda requested, *args, **kwargs: (
            [(t, payloads[(t.x, t.y)], None) for t in requested if (t.x, t.y) in payloads],
            0,
            ([], None),
        ),
    )
    region = RegionConfig(west=10.0, south=45.0, east=10.15, north=45.15)
    return XYZRasterSource(region, XYZImageryConfig(zoom=12, source="esri_satellite"))


def _mixed_grid() -> Tuple[List[PyTileIndex], Dict[Tuple[int, int], bytes]]:
    """4x3 tiles at (20, 30): PNG, WebP, JPEG and 16-bit PNG, one black, one missing."""
    rng = np.random.default_rng(11)
    grid = [PyTileIndex(x, y, 12) for y in range(30, 33) for x in range(20, 24)]
    makers = [
        lambda a: _encode(Image.fromarray(a), "PNG"),
        lambda a: _encode(Image.fromarray(a), "WEBP", quality=80),
        lambda a: _encode(Image.fromarray(a), "JPEG", quality=80),
        lambda a: _png(a.astype(np.int64) * 257, 2, 16),
        lambda a: _encode(Image.fromarray(a).quantize(32), "GIF"),
    ]
    payloads: Dict[Tuple[int, int], bytes] = {}
    for index, tile in enumerate(grid):
        pixels = _content("photo") if index % 2 else _content("noise")
        payloads[(tile.x, tile.y)] = makers[index % len(makers)](np.roll(pixels, index, axis=0))
    payloads[(21, 31)] = _encode(Image.fromarray(np.zeros((TILE, TILE, 3), np.uint8)), "PNG")
    half = np.zeros((TILE, TILE, 3), np.uint8)
    half[:, 128:] = rng.integers(0, 256, (TILE, 128, 3), dtype=np.uint8)
    payloads[(22, 31)] = _encode(Image.fromarray(half), "PNG")
    del payloads[(23, 32)]  # a tile the fetcher did not return
    return grid, payloads


@pytest.mark.parametrize(
    ("rows", "cols"),
    [
        ((0, 768), (0, 1024)),
        ((0, 256), (0, 256)),
        ((100, 600), (200, 900)),
        ((255, 257), (255, 257)),
        ((300, 301), (0, 1024)),
        ((0, 768), (1023, 1024)),
        ((500, 768), (770, 1024)),
    ],
)
def test_read_window_matches_the_pillow_implementation(
    monkeypatch: pytest.MonkeyPatch, rows: Tuple[int, int], cols: Tuple[int, int]
) -> None:
    grid, payloads = _mixed_grid()
    source = _source(monkeypatch, grid, payloads)

    window, valid = source.read_window(rows[0], rows[1], cols[0], cols[1])

    expected_window, expected_valid = _reference_window(payloads, (20, 30), rows, cols)
    assert window.dtype == np.uint8
    assert valid.dtype == np.bool_
    assert np.array_equal(window, expected_window)
    assert np.array_equal(valid, expected_valid)


def test_undecodable_tiles_raise_the_same_errors_as_before(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    grid = [PyTileIndex(3, 4, 12), PyTileIndex(4, 4, 12)]
    good = _encode(Image.fromarray(_content("photo")), "PNG")
    broken = good[: len(good) // 2]
    small = _encode(Image.new("RGB", (128, 128)), "PNG")

    source = _source(monkeypatch, grid, {(3, 4): good, (4, 4): broken})
    with pytest.raises(RuntimeError, match=r"Unable to decode XYZ tile 4/4"):
        source.read_window(0, 256, 0, 512)

    source = _source(monkeypatch, grid, {(3, 4): good, (4, 4): small})
    with pytest.raises(ValueError, match=r"256x256 RGB-compatible"):
        source.read_window(0, 256, 0, 512)

    source = _source(monkeypatch, grid, {(3, 4): good, (4, 4): b"<html>rate limited</html>"})
    with pytest.raises(RuntimeError, match=r"Unable to decode XYZ tile 4/4"):
        source.read_window(0, 256, 0, 512)


def test_first_bad_pillow_tile_raises_when_several_are_decoded_in_threads(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    grid = [PyTileIndex(x, 4, 12) for x in range(3, 7)]
    jpeg = _encode(Image.fromarray(_content("photo")), "JPEG")
    payloads = {(3, 4): jpeg, (4, 4): jpeg[:100], (5, 4): jpeg, (6, 4): jpeg[:50]}
    source = _source(monkeypatch, grid, payloads)

    with pytest.raises(RuntimeError, match=r"Unable to decode XYZ tile 4/4"):
        source.read_window(0, 256, 0, 1024)


def test_empty_window_and_missing_tiles(monkeypatch: pytest.MonkeyPatch) -> None:
    source = _source(monkeypatch, [PyTileIndex(3, 4, 12)], {})

    window, valid = source.read_window(0, 256, 0, 256)  # the tile was not fetched
    assert window.shape == (256, 256, 3)
    assert not window.any()
    assert not valid.any()
    empty, empty_valid = source.read_window(5, 5, 0, 256)
    assert empty.shape == (0, 256, 3)
    assert empty_valid.shape == (0, 256)


def test_decoding_releases_the_gil() -> None:
    """Other Python threads keep running while tiles decode."""
    tile = _png(_content("noise"), 2, 8)
    side = 16
    tiles = [(x, y, tile) for y in range(side) for x in range(side)]
    ticks = 0
    stop = threading.Event()

    def spin() -> None:
        nonlocal ticks
        while not stop.is_set():
            sum(range(2000))
            ticks += 1

    thread = threading.Thread(target=spin)
    thread.start()
    try:
        while ticks == 0:
            time.sleep(0.001)
        before = ticks
        decode_tile_window(tiles, 0, 0, 0, side * TILE, 0, side * TILE)
        gained = ticks - before
    finally:
        stop.set()
        thread.join()
    # With the GIL held for the whole call the spinner would get a handful of turns.
    assert gained > 20


def test_window_size_is_bounded() -> None:
    with pytest.raises(ValueError, match="exceeds"):
        decode_tile_window([], 0, 0, 0, 1 << 20, 0, 1 << 20)
