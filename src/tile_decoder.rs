//! Parallel tile decoding straight into a window buffer.
//!
//! `XYZRasterSource.read_window` used to decode each fetched tile with Pillow, one
//! at a time, holding the GIL. [`decode_window`] decodes the tiles on all cores and
//! writes them into the `(H, W, 3)` window and its validity mask in one pass.
//!
//! The output must be byte-identical to Pillow's `Image.convert("RGB")`, so a tile
//! is only decoded here when its format and colour type are known to decode to the
//! same pixels. Every other tile is reported back as *undecoded* and the caller
//! decodes it with Pillow.

use crate::tile_math::TILE_PX;
use image::{DynamicImage, ExtendedColorType, ImageDecoder, ImageFormat, ImageReader};
#[cfg(feature = "python")]
use numpy::{IntoPyArray, PyArray2, PyArray3};
#[cfg(feature = "python")]
use pyo3::prelude::*;
#[cfg(feature = "python")]
use pyo3::types::PyBytes;
use rayon::prelude::*;
use std::io::Cursor;

const RGB_CHANNELS: usize = 3;
/// Refuse windows that cannot sensibly fit in memory (a strip is a few hundred MB).
const MAX_WINDOW_BYTES: usize = 4 * 1024 * 1024 * 1024;
/// Tiles decoded per rayon thread before their rows are copied into the window.
/// Bounds the transient memory to a few MB while keeping every core busy.
const TILES_PER_THREAD_BATCH: usize = 4;

/// What [`decode_tile_window`] returns: window, validity mask, undecoded tiles.
#[cfg(feature = "python")]
type DecodedWindow<'py> = (
    Bound<'py, PyArray3<u8>>,
    Bound<'py, PyArray2<bool>>,
    Vec<(u32, u32)>,
);

/// One fetched tile: its grid position and encoded bytes.
#[derive(Clone, Copy)]
pub struct Tile<'a> {
    /// Tile column in the XYZ grid.
    pub x: u32,
    /// Tile row in the XYZ grid.
    pub y: u32,
    /// Encoded image bytes (PNG, JPEG, WebP, GIF).
    pub data: &'a [u8],
}

/// A decoded window.
pub struct Window {
    /// `(height, width, 3)` row-major RGB, zero where no tile was decoded.
    pub rgb: Vec<u8>,
    /// `(height, width)`: true where a decoded tile has a non-zero pixel.
    pub valid: Vec<bool>,
    /// Tiles that were left for the caller to decode, in row-major tile order.
    pub undecoded: Vec<(u32, u32)>,
}

/// Whether the first GIF frame covers the whole logical screen.
///
/// The `image` crate composites a smaller or offset first frame onto the screen
/// differently from Pillow, so such files are left to Pillow.
fn gif_first_frame_fills_screen(data: &[u8]) -> bool {
    const HEADER_LEN: usize = 13;
    let le16 = |at: usize| {
        data.get(at..at + 2)
            .map(|b| u16::from_le_bytes([b[0], b[1]]))
    };
    let (Some(screen_w), Some(screen_h), Some(&flags)) = (le16(6), le16(8), data.get(10)) else {
        return false;
    };
    let mut pos = HEADER_LEN;
    if flags & 0x80 != 0 {
        pos += 3 * (1usize << ((flags & 0x07) + 1));
    }
    loop {
        match data.get(pos) {
            // Extension: skip its data sub-blocks.
            Some(0x21) => {
                pos += 2;
                while let Some(&len) = data.get(pos) {
                    pos += 1 + usize::from(len);
                    if len == 0 {
                        break;
                    }
                }
            }
            // Image descriptor: left, top, width, height.
            Some(0x2C) => {
                return matches!(
                    (le16(pos + 1), le16(pos + 3), le16(pos + 5), le16(pos + 7)),
                    (Some(0), Some(0), Some(w), Some(h)) if w == screen_w && h == screen_h
                );
            }
            _ => return false,
        }
    }
}

/// Whether decoding `data` as `format` with colour type `color` is known to give
/// the same RGB pixels as Pillow's `convert("RGB")`.
///
/// Verified bit-identical on a large synthetic matrix (see
/// `tests/test_tile_decoder.py`): 8-bit PNG of every colour type, palette and
/// bit depth (interlaced or not), lossy and lossless WebP with and without
/// alpha, and GIF. Left to Pillow on purpose:
/// - JPEG: the `image` crate's IDCT and chroma upsampling differ from
///   libjpeg-turbo's by up to 4 levels per channel;
/// - 16-bit PNG and every other format.
fn matches_pillow(format: ImageFormat, color: ExtendedColorType, data: &[u8]) -> bool {
    use ExtendedColorType::{La8, Rgb8, Rgba8, L8};
    match format {
        ImageFormat::Png => matches!(color, Rgb8 | Rgba8 | L8 | La8),
        ImageFormat::WebP => matches!(color, Rgb8 | Rgba8),
        ImageFormat::Gif => gif_first_frame_fills_screen(data),
        _ => false,
    }
}

/// Decode `data` to `TILE_PX x TILE_PX` RGB, or `None` when the caller must use Pillow.
/// Fully transparent pixels come out black (see [`opaque_rgb`]).
fn decode_tile(data: &[u8]) -> Option<Vec<u8>> {
    let reader = ImageReader::new(Cursor::new(data))
        .with_guessed_format()
        .ok()?;
    let format = reader.format()?;
    let decoder = reader.into_decoder().ok()?;
    if !matches_pillow(format, decoder.original_color_type(), data) {
        return None;
    }
    let (width, height) = decoder.dimensions();
    if usize::try_from(width) != Ok(TILE_PX) || usize::try_from(height) != Ok(TILE_PX) {
        return None;
    }
    let image = DynamicImage::from_decoder(decoder).ok()?;
    let rgb = if image.color().has_alpha() {
        opaque_rgb(&image.into_rgba8().into_raw())
    } else {
        image.into_rgb8().into_raw()
    };
    (rgb.len() == TILE_PX * TILE_PX * RGB_CHANNELS).then_some(rgb)
}

/// RGB of RGBA pixels, with fully transparent pixels zeroed: a transparent pixel has no
/// imagery (its hidden colour is arbitrary), and all-zero pixels count as empty.
fn opaque_rgb(rgba: &[u8]) -> Vec<u8> {
    let mut rgb = Vec::with_capacity(rgba.len() / 4 * RGB_CHANNELS);
    for px in rgba.as_chunks::<4>().0 {
        if px[3] == 0 {
            rgb.extend_from_slice(&[0; RGB_CHANNELS]);
        } else {
            rgb.extend_from_slice(&px[..RGB_CHANNELS]);
        }
    }
    rgb
}

/// A tile that intersects the window, with its pixel origin in window coordinates.
struct Placed<'a> {
    tile: Tile<'a>,
    /// Row of the tile's top edge relative to the window's top edge (may be negative).
    top: i64,
    /// Column of the tile's left edge relative to the window's left edge (may be negative).
    left: i64,
}

/// Copy the part of `pixels` (one tile row, `TILE_PX` wide) that falls inside the
/// window into `row`, and set the matching `valid` entries.
fn blit_row(pixels: &[u8], left: i64, row: &mut [u8], valid: &mut [bool]) {
    let window_width = valid.len();
    let tile_px = i64::try_from(TILE_PX).unwrap_or(i64::MAX);
    let first = left.max(0);
    let last = (left + tile_px).min(i64::try_from(window_width).unwrap_or(i64::MAX));
    let (Ok(first), Ok(last), Ok(skip)) = (
        usize::try_from(first),
        usize::try_from(last),
        usize::try_from(first - left),
    ) else {
        return;
    };
    let count = last.saturating_sub(first);
    let src = &pixels[skip * RGB_CHANNELS..(skip + count) * RGB_CHANNELS];
    row[first * RGB_CHANNELS..last * RGB_CHANNELS].copy_from_slice(src);
    for (flag, px) in valid[first..last]
        .iter_mut()
        .zip(src.as_chunks::<RGB_CHANNELS>().0)
    {
        // Failed tiles are black-filled by the fetcher and some providers serve
        // black NoData, so all-zero pixels count as empty.
        *flag = px[0] | px[1] | px[2] != 0;
    }
}

/// Decode `tiles` in parallel into the window `rows x cols` (pixel ranges of the
/// raster whose top-left tile is `origin`).
///
/// # Errors
/// Returns a message when the window would be larger than the memory budget.
pub fn decode_window(
    tiles: &[Tile<'_>],
    origin: (u32, u32),
    rows: (usize, usize),
    cols: (usize, usize),
) -> Result<Window, String> {
    let height = rows.1.saturating_sub(rows.0);
    let width = cols.1.saturating_sub(cols.0);
    let window_bytes = height
        .checked_mul(width)
        .and_then(|area| area.checked_mul(RGB_CHANNELS))
        .filter(|&bytes| bytes <= MAX_WINDOW_BYTES)
        .ok_or_else(|| {
            format!(
                "decode_tile_window: a {height}x{width} window exceeds the \
                 {MAX_WINDOW_BYTES}-byte limit"
            )
        })?;
    let mut rgb = vec![0u8; window_bytes];
    let mut valid = vec![false; height * width];
    let mut undecoded = Vec::new();
    if window_bytes == 0 {
        return Ok(Window {
            rgb,
            valid,
            undecoded,
        });
    }

    let tile_px = i64::try_from(TILE_PX).map_err(|e| e.to_string())?;
    let to_i64 = |value: usize| i64::try_from(value).map_err(|e| e.to_string());
    let (row_start, col_start) = (to_i64(rows.0)?, to_i64(cols.0)?);
    let (window_h, window_w) = (to_i64(height)?, to_i64(width)?);
    let mut placed: Vec<Placed<'_>> = tiles
        .iter()
        .filter_map(|&tile| {
            let top = (i64::from(tile.y) - i64::from(origin.1)) * tile_px - row_start;
            let left = (i64::from(tile.x) - i64::from(origin.0)) * tile_px - col_start;
            let overlaps =
                top < window_h && top + tile_px > 0 && left < window_w && left + tile_px > 0;
            overlaps.then_some(Placed { tile, top, left })
        })
        .collect();
    placed.sort_by_key(|p| (p.tile.y, p.tile.x));

    // Group consecutive tile rows until a group holds enough tiles to keep every
    // thread busy, then decode a group in parallel and copy its rows in parallel.
    let target = rayon::current_num_threads() * TILES_PER_THREAD_BATCH;
    let mut start = 0;
    while start < placed.len() {
        let mut end = start;
        while end < placed.len() && (end == start || end - start < target) {
            let row_y = placed[end].tile.y;
            while end < placed.len() && placed[end].tile.y == row_y {
                end += 1;
            }
        }
        let group = &placed[start..end];
        let decoded: Vec<Option<Vec<u8>>> =
            group.par_iter().map(|p| decode_tile(p.tile.data)).collect();
        undecoded.extend(
            group
                .iter()
                .zip(&decoded)
                .filter(|(_, pixels)| pixels.is_none())
                .map(|(p, _)| (p.tile.x, p.tile.y)),
        );

        // Window rows of this group: from the first tile row's top to the last's bottom.
        let first_row = usize::try_from(group[0].top.max(0)).map_err(|e| e.to_string())?;
        let last_row = usize::try_from(
            group
                .iter()
                .map(|p| p.top + tile_px)
                .max()
                .unwrap_or(0)
                .min(window_h),
        )
        .map_err(|e| e.to_string())?;
        rgb[first_row * width * RGB_CHANNELS..last_row * width * RGB_CHANNELS]
            .par_chunks_mut(width * RGB_CHANNELS)
            .zip(valid[first_row * width..last_row * width].par_chunks_mut(width))
            .enumerate()
            .for_each(|(offset, (rgb_row, valid_row))| {
                let row = to_i64(first_row + offset).unwrap_or(i64::MAX);
                for (p, pixels) in group.iter().zip(&decoded) {
                    let Some(pixels) = pixels else { continue };
                    let local = row - p.top;
                    if !(0..tile_px).contains(&local) {
                        continue;
                    }
                    let Ok(local) = usize::try_from(local) else {
                        continue;
                    };
                    let tile_row = &pixels
                        [local * TILE_PX * RGB_CHANNELS..(local + 1) * TILE_PX * RGB_CHANNELS];
                    blit_row(tile_row, p.left, rgb_row, valid_row);
                }
            });
        start = end;
    }
    Ok(Window {
        rgb,
        valid,
        undecoded,
    })
}

/// Decode fetched tiles into a window of the tile raster, without holding the GIL.
///
/// `tiles` is a list of `(x, y, encoded_bytes)`. `origin_x`/`origin_y` is the top-left
/// tile of the raster; `row_start..row_stop` and `col_start..col_stop` are the
/// window's pixel range within it. Returns `(window, valid, undecoded)`: an
/// `(H, W, 3)` uint8 array, an `(H, W)` bool array that is true where a decoded
/// tile has a non-zero pixel, and the `(x, y)` of the tiles that were not decoded
/// here (formats whose pixels are not guaranteed to match Pillow's, and tiles that
/// do not decode), for the caller to decode with Pillow.
///
/// # Errors
/// Raises `ValueError` when the window exceeds the memory budget.
#[cfg(feature = "python")]
#[pyfunction]
#[pyo3(signature = (tiles, origin_x, origin_y, row_start, row_stop, col_start, col_stop))]
#[allow(clippy::needless_pass_by_value, clippy::too_many_arguments)]
pub fn decode_tile_window<'py>(
    py: Python<'py>,
    tiles: Vec<(u32, u32, Bound<'py, PyBytes>)>,
    origin_x: u32,
    origin_y: u32,
    row_start: usize,
    row_stop: usize,
    col_start: usize,
    col_stop: usize,
) -> PyResult<DecodedWindow<'py>> {
    let borrowed: Vec<Tile<'_>> = tiles
        .iter()
        .map(|(x, y, data)| Tile {
            x: *x,
            y: *y,
            data: data.as_bytes(),
        })
        .collect();
    let window = py
        .detach(|| {
            decode_window(
                &borrowed,
                (origin_x, origin_y),
                (row_start, row_stop),
                (col_start, col_stop),
            )
        })
        .map_err(pyo3::exceptions::PyValueError::new_err)?;
    let height = row_stop.saturating_sub(row_start);
    let width = col_stop.saturating_sub(col_start);
    let rgb = numpy::ndarray::Array3::from_shape_vec((height, width, RGB_CHANNELS), window.rgb)
        .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))?;
    let valid = numpy::ndarray::Array2::from_shape_vec((height, width), window.valid)
        .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))?;
    Ok((
        rgb.into_pyarray(py),
        valid.into_pyarray(py),
        window.undecoded,
    ))
}

#[cfg(test)]
mod tests {
    use super::*;
    use image::{ImageBuffer, ImageFormat, Rgb, RgbImage};

    /// Pixel `(row, col)` of tile `(tx, ty)`: never black (blue >= 1) and distinct per tile.
    fn pixel(tx: u32, ty: u32, row: usize, col: usize) -> [u8; 3] {
        [
            u8::try_from((col + 7 * tx as usize + 1) % 256).unwrap(),
            u8::try_from((row + 13 * ty as usize + 1) % 256).unwrap(),
            u8::try_from(1 + (tx + ty) % 200).unwrap(),
        ]
    }

    fn tile_image(tx: u32, ty: u32) -> RgbImage {
        let size = u32::try_from(TILE_PX).unwrap();
        RgbImage::from_fn(size, size, |c, r| {
            Rgb(pixel(tx, ty, r as usize, c as usize))
        })
    }

    fn encode(image: &RgbImage, format: ImageFormat) -> Vec<u8> {
        let mut buf = Cursor::new(Vec::new());
        image.write_to(&mut buf, format).unwrap();
        buf.into_inner()
    }

    fn grid(origin: (u32, u32), nx: u32, ny: u32) -> Vec<(u32, u32, Vec<u8>)> {
        let mut tiles = Vec::new();
        for ty in origin.1..origin.1 + ny {
            for tx in origin.0..origin.0 + nx {
                tiles.push((tx, ty, encode(&tile_image(tx, ty), ImageFormat::Png)));
            }
        }
        tiles
    }

    fn borrow(tiles: &[(u32, u32, Vec<u8>)]) -> Vec<Tile<'_>> {
        tiles
            .iter()
            .map(|(x, y, data)| Tile { x: *x, y: *y, data })
            .collect()
    }

    /// The window pixel `(r, c)` that a decoded grid with top-left tile `origin`
    /// must hold, given the window starts at pixel `(row0, col0)`.
    fn expected(origin: (u32, u32), row0: usize, col0: usize, r: usize, c: usize) -> [u8; 3] {
        let (gr, gc) = (row0 + r, col0 + c);
        pixel(
            origin.0 + u32::try_from(gc / TILE_PX).unwrap(),
            origin.1 + u32::try_from(gr / TILE_PX).unwrap(),
            gr % TILE_PX,
            gc % TILE_PX,
        )
    }

    fn assert_window(w: &Window, origin: (u32, u32), rows: (usize, usize), cols: (usize, usize)) {
        let width = cols.1 - cols.0;
        for r in 0..rows.1 - rows.0 {
            for c in 0..width {
                let at = (r * width + c) * RGB_CHANNELS;
                assert_eq!(
                    w.rgb[at..at + 3],
                    expected(origin, rows.0, cols.0, r, c),
                    "pixel {r},{c}"
                );
                assert!(w.valid[r * width + c], "valid {r},{c}");
            }
        }
    }

    #[test]
    fn full_window_places_every_tile() {
        let origin = (10, 20);
        let tiles = grid(origin, 3, 2);
        let w = decode_window(&borrow(&tiles), origin, (0, 512), (0, 768)).unwrap();
        assert_eq!(w.undecoded, Vec::new());
        assert_eq!(w.rgb.len(), 512 * 768 * 3);
        assert_window(&w, origin, (0, 512), (0, 768));
    }

    #[test]
    fn partial_window_crops_tiles_at_every_edge() {
        let origin = (3, 4);
        let tiles = grid(origin, 3, 3);
        for (rows, cols) in [
            ((100, 400), (300, 600)),
            ((0, 256), (0, 256)),
            ((255, 257), (255, 257)),
            ((200, 700), (10, 20)),
            ((511, 512), (0, 768)),
        ] {
            let w = decode_window(&borrow(&tiles), origin, rows, cols).unwrap();
            assert_eq!(w.undecoded, Vec::new());
            assert_window(&w, origin, rows, cols);
        }
    }

    #[test]
    fn black_pixels_are_not_valid_and_missing_tiles_stay_empty() {
        let origin = (0, 0);
        let mut tiles = grid(origin, 3, 1);
        // Tile 0: all black. Tile 1: black except one pixel. Tile 2: absent.
        tiles[0].2 = encode(&RgbImage::new(256, 256), ImageFormat::Png);
        let mut one = RgbImage::new(256, 256);
        one.put_pixel(5, 9, Rgb([0, 0, 1]));
        tiles[1].2 = encode(&one, ImageFormat::Png);
        tiles.truncate(2);
        let w = decode_window(&borrow(&tiles), origin, (0, 256), (0, 768)).unwrap();
        assert_eq!(w.undecoded, Vec::new());
        let valid_at: Vec<usize> = (0..w.valid.len()).filter(|&i| w.valid[i]).collect();
        assert_eq!(valid_at, vec![9 * 768 + 256 + 5]);
        assert!(w.rgb[..256 * 3].iter().all(|&b| b == 0));
    }

    #[test]
    fn tiles_left_to_pillow_are_reported_and_leave_zeros() {
        let origin = (0, 0);
        let mut tiles = grid(origin, 3, 2);
        tiles[1].2 = encode(&tile_image(1, 0), ImageFormat::Jpeg);
        tiles[2].2 = b"not an image".to_vec();
        tiles[3].2 = encode(&RgbImage::new(128, 128), ImageFormat::Png);
        let rgb16: ImageBuffer<Rgb<u16>, Vec<u16>> = ImageBuffer::new(256, 256);
        let mut buf = Cursor::new(Vec::new());
        rgb16.write_to(&mut buf, ImageFormat::Png).unwrap();
        tiles[4].2 = buf.into_inner();
        let w = decode_window(&borrow(&tiles), origin, (0, 512), (0, 768)).unwrap();
        assert_eq!(w.undecoded, vec![(1, 0), (2, 0), (0, 1), (1, 1)]);
        // Tiles (0, 0) and (2, 1) decoded and are in place; the rest is still zero.
        assert_eq!(w.rgb[..3], pixel(0, 0, 0, 0));
        let at = (300 * 768 + 600) * 3;
        assert_eq!(w.rgb[at..at + 3], pixel(2, 1, 44, 88));
        assert!(w.rgb[256 * 3..512 * 3].iter().all(|&b| b == 0));
        assert!(!w.valid[300]);
    }

    #[test]
    fn many_tile_rows_exercise_the_batching() {
        let origin = (50, 70);
        let tiles = grid(origin, 5, 40);
        let (rows, cols) = ((37, 256 * 40 - 91), (11, 256 * 5 - 3));
        let w = decode_window(&borrow(&tiles), origin, rows, cols).unwrap();
        assert_eq!(w.undecoded, Vec::new());
        assert_window(&w, origin, rows, cols);
    }

    #[test]
    fn tiles_outside_the_window_are_ignored() {
        let origin = (10, 10);
        let mut tiles = grid(origin, 2, 2);
        tiles.push((9, 10, b"left of the raster".to_vec()));
        tiles.push((12, 10, encode(&tile_image(12, 10), ImageFormat::Png)));
        let w = decode_window(&borrow(&tiles), origin, (0, 256), (0, 256)).unwrap();
        assert_eq!(w.undecoded, Vec::new());
        assert_window(&w, origin, (0, 256), (0, 256));
    }

    #[test]
    fn empty_and_oversized_windows() {
        let tiles = grid((0, 0), 1, 1);
        let w = decode_window(&borrow(&tiles), (0, 0), (5, 5), (0, 256)).unwrap();
        assert!(w.rgb.is_empty() && w.valid.is_empty() && w.undecoded.is_empty());
        let err = decode_window(&borrow(&tiles), (0, 0), (0, 1 << 20), (0, 1 << 20));
        assert!(err.is_err());
    }

    #[test]
    fn webp_tiles_decode() {
        let origin = (0, 0);
        let mut tiles = grid(origin, 2, 1);
        tiles[0].2 = encode(&tile_image(0, 0), ImageFormat::WebP);
        let w = decode_window(&borrow(&tiles), origin, (0, 256), (0, 512)).unwrap();
        assert_eq!(w.undecoded, Vec::new());
        assert_window(&w, origin, (0, 256), (0, 512));
    }

    #[test]
    fn gif_must_start_with_a_full_screen_frame() {
        fn gif(screen: (u16, u16), frame: (u16, u16, u16, u16), extension: bool) -> Vec<u8> {
            let mut out = b"GIF89a".to_vec();
            out.extend(screen.0.to_le_bytes());
            out.extend(screen.1.to_le_bytes());
            out.extend([0x80, 0, 0]);
            out.extend([0u8; 6]); // 2-colour global palette
            if extension {
                out.extend([0x21, 0xF9, 4, 0, 0, 0, 0, 0]);
            }
            out.push(0x2C);
            for v in [frame.0, frame.1, frame.2, frame.3] {
                out.extend(v.to_le_bytes());
            }
            out
        }
        assert!(gif_first_frame_fills_screen(&gif(
            (256, 256),
            (0, 0, 256, 256),
            false
        )));
        assert!(gif_first_frame_fills_screen(&gif(
            (256, 256),
            (0, 0, 256, 256),
            true
        )));
        assert!(!gif_first_frame_fills_screen(&gif(
            (256, 256),
            (10, 0, 246, 256),
            false
        )));
        assert!(!gif_first_frame_fills_screen(&gif(
            (256, 256),
            (0, 0, 100, 80),
            true
        )));
        assert!(!gif_first_frame_fills_screen(
            &gif((256, 256), (0, 0, 256, 256), true)[..20]
        ));
        assert!(!gif_first_frame_fills_screen(b"GIF89a"));
    }
}
