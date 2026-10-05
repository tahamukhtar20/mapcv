//! Tile stitching: parallel PNG decode -> contiguous (H, W, 3) buffer.

use crate::fetcher::{TILE_PX, TILE_PX_F};
use crate::tile_math::{xy_bounds, TileIndex};
use image::ImageReader;
use rayon::prelude::*;
use std::io::Cursor;

const RGB_CHANNELS: usize = 3;
const MAX_CANVAS_BYTES: usize = 512 * 1024 * 1024;

/// Why [`stitch_tiles`] could not build a canvas.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum StitchError {
    /// The tiles cannot be stitched as given: they mix zoom levels or a tile
    /// does not decode to `TILE_PX x TILE_PX` pixels.
    InvalidInput(String),
    /// The tiles are well formed but stitching failed: a tile could not be
    /// decoded or the canvas would exceed the memory budget.
    Failed(String),
}

impl std::fmt::Display for StitchError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            StitchError::InvalidInput(msg) | StitchError::Failed(msg) => f.write_str(msg),
        }
    }
}

/// Decode each tile's PNG bytes in parallel and assemble into a single
/// `(H, W, 3)` row-major u8 buffer.
///
/// Returns `(canvas, min_x, min_y, height, width)`.  The caller uses
/// `min_x`/`min_y` to compute the affine transform via [`tile_transform`].
///
/// # Errors
/// Returns [`StitchError::InvalidInput`] if tiles span more than one zoom level
/// or a tile does not decode to `TILE_PX x TILE_PX` pixels, and
/// [`StitchError::Failed`] if a tile's bytes cannot be decoded or the canvas
/// would be too large.
pub fn stitch_tiles(
    tiles: &[(u32, u32, u8, Vec<u8>)],
) -> Result<(Vec<u8>, u32, u32, usize, usize), StitchError> {
    let Some(&(first_x, first_y, _, _)) = tiles.first() else {
        return Ok((Vec::new(), 0, 0, 0, 0));
    };

    let zooms: std::collections::HashSet<u8> = tiles.iter().map(|(_, _, z, _)| *z).collect();
    if zooms.len() > 1 {
        return Err(StitchError::InvalidInput(format!(
            "stitch_tiles: all tiles must be at the same zoom level, got {zooms:?}"
        )));
    }

    let (mut min_x, mut min_y, mut max_x, mut max_y) = (first_x, first_y, first_x, first_y);
    for &(x, y, _, _) in tiles {
        min_x = min_x.min(x);
        min_y = min_y.min(y);
        max_x = max_x.max(x);
        max_y = max_y.max(y);
    }
    let tile_count_x = u64::from(max_x) - u64::from(min_x) + 1;
    let tile_count_y = u64::from(max_y) - u64::from(min_y) + 1;
    let failed = |msg: &str| StitchError::Failed(format!("stitch_tiles: {msg}"));
    let n_x = usize::try_from(tile_count_x)
        .map_err(|_| failed("canvas width exceeds platform capacity"))?;
    let n_y = usize::try_from(tile_count_y)
        .map_err(|_| failed("canvas height exceeds platform capacity"))?;
    let h = n_y
        .checked_mul(TILE_PX)
        .ok_or_else(|| failed("canvas height overflows usize"))?;
    let w = n_x
        .checked_mul(TILE_PX)
        .ok_or_else(|| failed("canvas width overflows usize"))?;
    let canvas_bytes = h
        .checked_mul(w)
        .and_then(|area| area.checked_mul(RGB_CHANNELS))
        .ok_or_else(|| failed("canvas size overflows usize"))?;

    if canvas_bytes > MAX_CANVAS_BYTES {
        return Err(failed(&format!(
            "canvas requires {canvas_bytes} bytes, exceeding the {MAX_CANVAS_BYTES}-byte limit"
        )));
    }

    // Decode all tiles in parallel; each thread produces (tx, ty, rgb_bytes).
    let decoded: Vec<(u32, u32, Vec<u8>)> = tiles
        .par_iter()
        .map(|(tx, ty, tz, data)| {
            let reader = ImageReader::new(Cursor::new(data))
                .with_guessed_format()
                .map_err(|e| StitchError::Failed(format!("tile format detection failed: {e}")))?;
            let image = reader
                .decode()
                .map_err(|e| StitchError::Failed(format!("tile PNG decode failed: {e}")))?;
            // The row copy below assumes exactly TILE_PX x TILE_PX pixels: a smaller
            // tile would index out of bounds and a larger one would be scrambled.
            if usize::try_from(image.width()) != Ok(TILE_PX)
                || usize::try_from(image.height()) != Ok(TILE_PX)
            {
                return Err(StitchError::InvalidInput(format!(
                    "stitch_tiles: tile {tz}/{tx}/{ty} is {}x{} pixels; every tile must be \
                     {TILE_PX}x{TILE_PX}",
                    image.width(),
                    image.height()
                )));
            }
            Ok((*tx, *ty, image.to_rgb8().into_raw()))
        })
        .collect::<Result<Vec<_>, StitchError>>()?;

    let mut canvas = vec![0u8; canvas_bytes];
    for (tx, ty, pixels) in decoded {
        let row_offset = (ty - min_y) as usize * TILE_PX;
        let col_offset = (tx - min_x) as usize * TILE_PX;
        for r in 0..TILE_PX {
            let src = r * TILE_PX * RGB_CHANNELS;
            let dst = (row_offset + r) * w * RGB_CHANNELS + col_offset * RGB_CHANNELS;
            canvas[dst..dst + TILE_PX * RGB_CHANNELS]
                .copy_from_slice(&pixels[src..src + TILE_PX * RGB_CHANNELS]);
        }
    }

    Ok((canvas, min_x, min_y, h, w))
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
    let px = tile_w / TILE_PX_F;
    let py = tile_h / TILE_PX_F;
    (px, 0.0, b.west, 0.0, -py, b.north)
}

#[cfg(test)]
mod tests {
    use super::*;
    use image::{ImageFormat, RgbImage};

    fn png(width: u32, height: u32) -> Vec<u8> {
        let mut buf = Cursor::new(Vec::new());
        RgbImage::new(width, height)
            .write_to(&mut buf, ImageFormat::Png)
            .unwrap();
        buf.into_inner()
    }

    #[test]
    fn wrong_size_tiles_are_invalid_input() {
        for (w, h) in [(1, 1), (512, 512), (256, 255)] {
            let err = stitch_tiles(&[(0, 0, 1, png(w, h))]).unwrap_err();
            assert!(
                matches!(err, StitchError::InvalidInput(_)),
                "{w}x{h}: {err}"
            );
        }
    }

    #[test]
    fn mixed_zooms_are_invalid_input() {
        let tile = png(256, 256);
        let err = stitch_tiles(&[(0, 0, 1, tile.clone()), (0, 0, 2, tile)]).unwrap_err();
        assert!(matches!(err, StitchError::InvalidInput(_)));
    }

    #[test]
    fn undecodable_tile_fails() {
        let err = stitch_tiles(&[(0, 0, 1, b"not a png".to_vec())]).unwrap_err();
        assert!(matches!(err, StitchError::Failed(_)));
    }
}
