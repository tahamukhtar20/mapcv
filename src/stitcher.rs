//! Tile stitching: parallel PNG decode → contiguous (H, W, 3) buffer.

use crate::tile_math::{xy_bounds, TileIndex};
use image::io::Reader as ImageReader;
use rayon::prelude::*;
use std::io::Cursor;

/// Decode each tile's PNG bytes in parallel and assemble into a single
/// `(H, W, 3)` row-major u8 buffer.
///
/// Returns `(canvas, min_x, min_y, height, width)`.  The caller uses
/// `min_x`/`min_y` to compute the affine transform via [`tile_transform`].
///
/// # Panics
/// Panics if any tile's bytes cannot be decoded as an image.
#[must_use]
pub fn stitch_tiles(tiles: &[(u32, u32, u8, Vec<u8>)]) -> (Vec<u8>, u32, u32, usize, usize) {
    if tiles.is_empty() {
        return (Vec::new(), 0, 0, 0, 0);
    }

    let min_x = tiles.iter().map(|(x, _, _, _)| *x).min().unwrap();
    let min_y = tiles.iter().map(|(_, y, _, _)| *y).min().unwrap();
    let max_x = tiles.iter().map(|(x, _, _, _)| *x).max().unwrap();
    let max_y = tiles.iter().map(|(_, y, _, _)| *y).max().unwrap();
    let n_x = (max_x - min_x + 1) as usize;
    let n_y = (max_y - min_y + 1) as usize;
    let h = n_y * 256;
    let w = n_x * 256;

    // Decode all tiles in parallel; each thread produces (tx, ty, rgb_bytes).
    let decoded: Vec<(u32, u32, Vec<u8>)> = tiles
        .par_iter()
        .map(|(tx, ty, _, data)| {
            let reader = ImageReader::new(Cursor::new(data))
                .with_guessed_format()
                .expect("cursor I/O never fails");
            let pixels = reader
                .decode()
                .expect("tile PNG decode failed")
                .to_rgb8()
                .into_raw();
            (*tx, *ty, pixels)
        })
        .collect();

    let mut canvas = vec![0u8; h * w * 3];
    for (tx, ty, pixels) in decoded {
        let row_offset = (ty - min_y) as usize * 256;
        let col_offset = (tx - min_x) as usize * 256;
        for r in 0..256_usize {
            let src = r * 256 * 3;
            let dst = (row_offset + r) * w * 3 + col_offset * 3;
            canvas[dst..dst + 256 * 3].copy_from_slice(&pixels[src..src + 256 * 3]);
        }
    }

    (canvas, min_x, min_y, h, w)
}

/// Affine transform `(a, b, c, d, e, f)` mapping pixel `(col, row)` to
/// Web Mercator `(x, y)` in metres, for a tile grid whose top-left tile
/// is `(min_x, min_y)` at the given zoom level.
///
/// This matches the rasterio `Affine` convention used by `mapcv.rasterize`.
#[must_use]
pub fn tile_transform(min_x: u32, min_y: u32, zoom: u8) -> (f64, f64, f64, f64, f64, f64) {
    let b = xy_bounds(TileIndex {
        x: min_x,
        y: min_y,
        z: zoom,
    });
    let tile_w = b.east - b.west;
    let tile_h = b.north - b.south;
    let px = tile_w / 256.0;
    let py = tile_h / 256.0;
    (px, 0.0, b.west, 0.0, -py, b.north)
}
