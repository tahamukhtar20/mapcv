"""Regenerate the tiny TIFF, chunk and tile seeds in fuzz/seeds (needs the test dependencies).

    .venv/bin/python fuzz/gen_seeds.py

The KML seeds are written by hand, and the three TIFFs under tests/data/geotiff are
copied as they are. The files generated here cover each TIFF codec, both byte orders,
BigTIFF, strips and tiles, JPEG and WebP chunks, and every PNG, WebP and GIF colour
type the tile decoder takes (Pillow cannot write interlaced PNGs). Files named
regression_* are minimized crash inputs found by the fuzzers; they are kept by hand.
"""

from __future__ import annotations

import io
import shutil
import struct
import tempfile
import zlib
from pathlib import Path

import numpy as np
import rasterio
from PIL import Image
from rasterio.crs import CRS
from rasterio.transform import Affine

ROOT = Path(__file__).resolve().parent
TIFF = ROOT / "seeds" / "geotiff"
TILE = ROOT / "seeds" / "tile"
CHUNK = ROOT / "seeds" / "chunk"
TILE_PX = 256


def tiff(
    name: str, dtype: str = "uint8", count: int = 1, size: int = 32, **options: object
) -> None:
    rows, cols, bands = np.meshgrid(
        np.arange(size), np.arange(size), np.arange(count), indexing="ij"
    )
    data = ((rows * 3 + cols * 5 + bands * 40) % 200).astype(dtype).transpose(2, 0, 1)
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "seed.tif"
        with rasterio.open(
            path,
            "w",
            driver="GTiff",
            height=size,
            width=size,
            count=count,
            dtype=dtype,
            crs=CRS.from_epsg(32633),
            transform=Affine(10, 0, 500000, 0, -10, 4500000),
            nodata=0,
            **options,
        ) as dst:
            dst.write(data)
        shutil.copy(path, TIFF / f"{name}.tif")


def tile_image(mode: str) -> Image.Image:
    rows, cols = np.meshgrid(np.arange(TILE_PX), np.arange(TILE_PX), indexing="ij")
    base = ((rows // 16 + cols // 16) * 9 % 250 + 1).astype(np.uint8)
    channels = {
        "L": [base],
        "LA": [base, np.full_like(base, 255)],
        "RGB": [base, base[::-1], base.T],
        "RGBA": [base, base[::-1], base.T, np.full_like(base, 200)],
    }[mode]
    return Image.fromarray(np.dstack(channels).squeeze(), mode)


def tile(name: str, image: Image.Image, fmt: str, **options: object) -> None:
    buffer = io.BytesIO()
    image.save(buffer, fmt, **options)
    (TILE / name).write_bytes(buffer.getvalue())


def chunk(
    name: str,
    codec: int,
    data: bytes,
    *,
    width: int,
    rows: int,
    samples: int = 1,
    sample_bytes: int = 1,
    predictor: int = 0,
    photometric: int = 0,
    flags: int = 0,
    tables: bytes = b"",
) -> None:
    """One chunk_decode seed: the 11-byte parameter header (see the target), tables, data.

    `codec` indexes mapcv_fuzz::COMPRESSIONS: 0 none, 1 LZW, 2 JPEG, 3 Deflate,
    4 PackBits, 5 old Deflate, 6 ZSTD, 7 WebP.
    """
    header = struct.pack(
        "<BBBBBHHBB",
        codec,
        predictor,
        flags,
        {1: 0, 2: 1, 4: 2, 8: 3}[sample_bytes],
        samples - 1,
        width - 1,
        rows - 1,
        photometric,
        len(tables),
    )
    (CHUNK / f"{name}.bin").write_bytes(header + tables + data)


def encode(image: Image.Image, fmt: str, **options: object) -> bytes:
    buffer = io.BytesIO()
    image.save(buffer, fmt, **options)
    return buffer.getvalue()


def chunk_seeds() -> None:
    row = bytes(range(16))
    chunk("none_u16", 0, row, width=4, rows=2, sample_bytes=2)
    chunk("deflate_pred2_u16", 3, zlib.compress(row), width=4, rows=2, sample_bytes=2, predictor=1)
    chunk(
        "old_deflate_pred3_f32", 5, zlib.compress(row), width=2, rows=2, sample_bytes=4, predictor=2
    )
    packbits = bytes([15]) + row
    chunk("packbits_u8", 4, packbits, width=4, rows=4)
    gray = Image.fromarray((np.arange(256, dtype=np.uint8).reshape(16, 16)), "L")
    rgb = Image.merge("RGB", [gray, gray.transpose(Image.Transpose.FLIP_LEFT_RIGHT), gray])
    chunk("jpeg_gray", 2, encode(gray, "JPEG", quality=80), width=16, rows=16, photometric=1)
    chunk(
        "jpeg_ycbcr",
        2,
        encode(rgb, "JPEG", quality=80),
        width=16,
        rows=16,
        samples=3,
        photometric=1,
    )
    chunk("webp_rgb", 7, encode(rgb, "WEBP", lossless=True), width=16, rows=16, samples=3)
    rgba = rgb.convert("RGBA")
    chunk("webp_rgba", 7, encode(rgba, "WEBP", lossless=True), width=16, rows=16, samples=4)
    chunk("webp_rgb_lossy_as_rgba", 7, encode(rgb, "WEBP", quality=50), width=16, rows=8, samples=4)
    # Re-encoded by the target: any bytes become a valid stream of each codec.
    for codec, name in ((1, "lzw"), (3, "deflate"), (6, "zstd")):
        chunk(f"reencode_{name}", codec, bytes(range(64)), width=8, rows=8, flags=2, predictor=1)


def main() -> None:
    TIFF.mkdir(parents=True, exist_ok=True)
    TILE.mkdir(parents=True, exist_ok=True)
    CHUNK.mkdir(parents=True, exist_ok=True)
    chunk_seeds()
    for path in sorted((ROOT.parent / "tests" / "data" / "geotiff").glob("*.tif")):
        shutil.copy(path, TIFF / path.name)

    tiff("u8_none_strips")
    tiff("u8_lzw_tiled", compress="lzw", tiled=True, blockxsize=16, blockysize=16)
    tiff("u16_packbits", dtype="uint16", compress="packbits")
    tiff("u16_deflate_pred2_rgb", dtype="uint16", count=3, compress="deflate", predictor=2)
    tiff("f32_deflate_pred3", dtype="float32", compress="deflate", predictor=3)
    tiff("f64_zstd_pred2", dtype="float64", compress="zstd", predictor=2)
    tiff(
        "u8_jpeg_ycbcr",
        count=3,
        compress="jpeg",
        photometric="ycbcr",
        tiled=True,
        blockxsize=16,
        blockysize=16,
    )
    tiff("u8_webp_rgba", count=4, compress="webp", tiled=True, blockxsize=16, blockysize=16)
    tiff("i32_bigtiff_be", dtype="int32", BIGTIFF="YES", ENDIANNESS="BIG", compress="lzw")
    tiff(
        "u8_planar_deflate",
        count=3,
        interleave="band",
        compress="deflate",
        tiled=True,
        blockxsize=16,
        blockysize=16,
    )

    for mode in ("L", "LA", "RGB", "RGBA"):
        image = tile_image(mode)
        tile(f"png_{mode.lower()}.png", image, "PNG")
    rgb = tile_image("RGB")
    rgba = tile_image("RGBA")
    tile("png_palette.png", rgb.quantize(16), "PNG")
    tile("jpeg_rgb.jpg", rgb, "JPEG", quality=70)
    tile("webp_lossy.webp", rgb, "WEBP", quality=60)
    tile("webp_lossless_alpha.webp", rgba, "WEBP", lossless=True)
    tile("gif_palette.gif", rgb.quantize(32), "GIF")


if __name__ == "__main__":
    main()
