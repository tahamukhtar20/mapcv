//! Scanline polygon rasterizer.
//!
//! Burns `(polygon, class_id)` pairs into a u8 raster using either the
//! pixel-center scanline rule (default, matches rasterio's `all_touched=False`)
//! or supercover line drawing on edges (`all_touched=True`).
//!
//! Coordinate conventions match rasterio:
//! - The affine transform maps `(col, row)` -> `(x, y)` in world coords.
//! - Pixel `(col, row)` has its center at `(col + 0.5, row + 0.5)`.
//! - A 6-tuple `(a, b, c, d, e, f)` represents:
//!   `x = a*col + b*row + c`, `y = d*col + e*row + f`.
//! - The inverse transform maps world `(x, y)` -> pixel `(col, row)`.

// f64 cast precision loss is acceptable: pixel coords are bounded well below 2^52.
// Single-char names mirror the rasterio Affine convention (a..f) on purpose.
#![allow(
    clippy::cast_precision_loss,
    clippy::cast_possible_truncation,
    clippy::cast_sign_loss,
    clippy::many_single_char_names
)]

const EPS: f64 = 1e-12;

/// A 2D affine transform mapping (col, row) -> (x, y).
#[derive(Clone, Copy, Debug)]
pub struct Affine {
    /// Coefficient `a`: x scale per column.
    pub a: f64,
    /// Coefficient `b`: x shear per row.
    pub b: f64,
    /// Coefficient `c`: x offset (world x at col=0, row=0).
    pub c: f64,
    /// Coefficient `d`: y shear per column.
    pub d: f64,
    /// Coefficient `e`: y scale per row.
    pub e: f64,
    /// Coefficient `f`: y offset (world y at col=0, row=0).
    pub f: f64,
}

impl Affine {
    /// Invert the affine transform.
    ///
    /// # Errors
    /// Returns an error if the transform is singular (zero determinant).
    pub fn inverse(self) -> Result<Self, String> {
        let det = self.a * self.e - self.b * self.d;
        if det.abs() < EPS {
            return Err("Affine transform is singular (determinant ~ 0)".to_string());
        }
        let inv_det = 1.0 / det;
        let ia = self.e * inv_det;
        let ib = -self.b * inv_det;
        let id = -self.d * inv_det;
        let ie = self.a * inv_det;
        let ic = -(ia * self.c + ib * self.f);
        let if_ = -(id * self.c + ie * self.f);
        Ok(Affine {
            a: ia,
            b: ib,
            c: ic,
            d: id,
            e: ie,
            f: if_,
        })
    }

    /// Apply the transform to a point.
    #[must_use]
    pub fn apply(&self, x: f64, y: f64) -> (f64, f64) {
        (
            self.a * x + self.b * y + self.c,
            self.d * x + self.e * y + self.f,
        )
    }
}

/// A polygon: a list of rings in pixel coordinates. Ring 0 is the exterior;
/// any subsequent rings are holes. The even-odd fill rule handles holes
/// implicitly when all rings are merged into one edge list.
pub type RingSet = Vec<Vec<(f64, f64)>>;

/// Iterate the directed edges of a ring, auto-closing if the ring's last
/// vertex does not coincide with the first.
fn ring_edges(ring: &[(f64, f64)]) -> impl Iterator<Item = ((f64, f64), (f64, f64))> + '_ {
    let closing = if ring.len() >= 2 {
        let first = ring[0];
        let last = ring[ring.len() - 1];
        if (first.0 - last.0).abs() > EPS || (first.1 - last.1).abs() > EPS {
            Some((last, first))
        } else {
            None
        }
    } else {
        None
    };
    ring.windows(2)
        .map(|w| (w[0], w[1]))
        .chain(closing.into_iter())
}

/// Burn one polygon (in pixel coords) into the output raster using the
/// scanline rule: a pixel is filled iff its center is inside the polygon.
fn fill_scanline(rings: &RingSet, out: &mut [u8], width: usize, height: usize, class_id: u8) {
    if rings.is_empty() {
        return;
    }

    // Each edge: (y_top, y_bottom, x_at_y_top, slope dx/dy).
    let mut edges: Vec<(f64, f64, f64, f64)> = Vec::new();
    let mut y_min = f64::INFINITY;
    let mut y_max = f64::NEG_INFINITY;

    for ring in rings {
        if ring.len() < 2 {
            continue;
        }
        for ((x0, y0), (x1, y1)) in ring_edges(ring) {
            // Skip horizontal edges entirely. They contribute nothing to the
            // scanline parity count and only complicate vertex handling.
            if (y0 - y1).abs() < EPS {
                continue;
            }
            let (yt, yb, xt, xb) = if y0 < y1 {
                (y0, y1, x0, x1)
            } else {
                (y1, y0, x1, x0)
            };
            let slope = (xb - xt) / (yb - yt);
            edges.push((yt, yb, xt, slope));
            y_min = y_min.min(yt);
            y_max = y_max.max(yb);
        }
    }

    if edges.is_empty() {
        return;
    }

    let height_f = height as f64;
    let width_f = width as f64;

    // Pixel rows whose centers (row + 0.5) might fall inside [y_min, y_max).
    // row + 0.5 >= y_min => row >= y_min - 0.5 => row_start = ceil(y_min - 0.5)
    // row + 0.5 <  y_max => row <  y_max - 0.5 => row_end_exclusive = ceil(y_max - 0.5)
    let row_start = (y_min - 0.5).ceil().max(0.0);
    let row_end = (y_max - 0.5).ceil().min(height_f);
    if row_start >= row_end {
        return;
    }
    let row_start_i = row_start as usize;
    let row_end_i = row_end as usize;

    let mut xs: Vec<f64> = Vec::new();

    for row in row_start_i..row_end_i {
        let y = row as f64 + 0.5;
        xs.clear();
        for &(yt, yb, xt, slope) in &edges {
            // Half-open rule [yt, yb): edges that start at the scanline count,
            // edges that end at it do not. Prevents double-counting at vertices.
            if y >= yt && y < yb {
                xs.push(xt + slope * (y - yt));
            }
        }
        if xs.len() < 2 {
            continue;
        }
        xs.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
        debug_assert!(
            xs.len().is_multiple_of(2),
            "odd intersection count on scanline {row}"
        );

        let row_offset = row * width;
        let mut i = 0;
        while i + 1 < xs.len() {
            let x_left = xs[i];
            let x_right = xs[i + 1];
            // Pixel col is filled iff col + 0.5 in [x_left, x_right).
            // col >= x_left - 0.5 => col_start = ceil(x_left - 0.5)
            // col <  x_right - 0.5 => col_end_exclusive = ceil(x_right - 0.5)
            let col_start = (x_left - 0.5).ceil().max(0.0);
            let col_end = (x_right - 0.5).ceil().min(width_f);
            if col_end > col_start {
                let cs = col_start as usize;
                let ce = col_end as usize;
                for col in cs..ce {
                    out[row_offset + col] = class_id;
                }
            }
            i += 2;
        }
    }
}

/// Mark every pixel that an edge segment crosses ("supercover" rasterization).
/// Used for `all_touched=True`. Returns early if the segment's bounding box
/// is fully outside the image: a polygon with vertices far outside the
/// raster could otherwise burn billions of useless iterations here.
fn mark_edge_supercover(
    p0: (f64, f64),
    p1: (f64, f64),
    out: &mut [u8],
    width: usize,
    height: usize,
    class_id: u8,
) {
    let (x0, y0) = p0;
    let (x1, y1) = p1;
    let width_f = width as f64;
    let height_f = height as f64;
    if x0.max(x1) < 0.0 || x0.min(x1) >= width_f {
        return;
    }
    if y0.max(y1) < 0.0 || y0.min(y1) >= height_f {
        return;
    }
    let dx = x1 - x0;
    let dy = y1 - y0;
    let steps = dx.abs().max(dy.abs()).ceil() as usize + 1;
    let inv_steps = 1.0 / steps as f64;
    for i in 0..=steps {
        let t = i as f64 * inv_steps;
        let x = x0 + t * dx;
        let y = y0 + t * dy;
        let col = x.floor();
        let row = y.floor();
        if col >= 0.0 && col < width_f && row >= 0.0 && row < height_f {
            let c = col as usize;
            let r = row as usize;
            out[r * width + c] = class_id;
        }
    }
}

fn fill_all_touched(rings: &RingSet, out: &mut [u8], width: usize, height: usize, class_id: u8) {
    fill_scanline(rings, out, width, height, class_id);
    for ring in rings {
        if ring.len() < 2 {
            continue;
        }
        for (p0, p1) in ring_edges(ring) {
            mark_edge_supercover(p0, p1, out, width, height, class_id);
        }
    }
}

fn world_rings_to_pixel(world: &RingSet, inv: &Affine) -> RingSet {
    world
        .iter()
        .map(|ring| {
            ring.iter()
                .map(|&(x, y)| inv.apply(x, y))
                .collect::<Vec<_>>()
        })
        .collect()
}

/// Burn a sequence of `(polygon, class_id)` pairs into an output raster.
///
/// Each polygon is a `RingSet` (exterior ring followed by zero or more
/// holes), with vertices in world coordinates. The `transform` maps
/// (col, row) -> (x, y) in world space; rasterization happens in pixel
/// space using its inverse.
///
/// Polygons are processed in order; later polygons overwrite earlier ones
/// (replace / last-writer-wins).
///
/// # Errors
/// Returns an error if `transform` is singular or if `width * height`
/// overflows `usize`.
///
/// # Panics
/// Does not panic on well-formed input.
pub fn rasterize(
    polygons: &[(RingSet, u8)],
    width: usize,
    height: usize,
    transform: Affine,
    all_touched: bool,
) -> Result<Vec<u8>, String> {
    let inv = transform.inverse()?;
    let buf_len = width
        .checked_mul(height)
        .ok_or_else(|| "raster dimensions overflow usize".to_string())?;
    let mut out = vec![0u8; buf_len];
    for (rings, class_id) in polygons {
        let pixel_rings = world_rings_to_pixel(rings, &inv);
        if all_touched {
            fill_all_touched(&pixel_rings, &mut out, width, height, *class_id);
        } else {
            fill_scanline(&pixel_rings, &mut out, width, height, *class_id);
        }
    }
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn identity() -> Affine {
        Affine {
            a: 1.0,
            b: 0.0,
            c: 0.0,
            d: 0.0,
            e: 1.0,
            f: 0.0,
        }
    }

    #[test]
    fn unit_square_fills_one_pixel_at_origin() {
        let rings = vec![vec![
            (0.0, 0.0),
            (1.0, 0.0),
            (1.0, 1.0),
            (0.0, 1.0),
            (0.0, 0.0),
        ]];
        let mask = rasterize(&[(rings, 1)], 4, 4, identity(), false).unwrap();
        assert_eq!(mask[0], 1);
        let ones: usize = mask.iter().map(|&v| usize::from(v == 1)).sum();
        assert_eq!(ones, 1);
    }

    #[test]
    fn full_image_square() {
        let rings = vec![vec![
            (0.0, 0.0),
            (4.0, 0.0),
            (4.0, 4.0),
            (0.0, 4.0),
            (0.0, 0.0),
        ]];
        let mask = rasterize(&[(rings, 7)], 4, 4, identity(), false).unwrap();
        assert!(mask.iter().all(|&v| v == 7));
    }

    #[test]
    fn polygon_with_hole() {
        // 10x10 outer ring, 4x4 hole in the middle (cols 3..=6, rows 3..=6).
        let outer = vec![
            (0.0, 0.0),
            (10.0, 0.0),
            (10.0, 10.0),
            (0.0, 10.0),
            (0.0, 0.0),
        ];
        let hole = vec![(3.0, 3.0), (7.0, 3.0), (7.0, 7.0), (3.0, 7.0), (3.0, 3.0)];
        let mask = rasterize(&[(vec![outer, hole], 1)], 10, 10, identity(), false).unwrap();
        // Hole pixel center (5.5, 5.5) should be 0.
        assert_eq!(mask[5 * 10 + 5], 0);
        // Edge pixel (col=1, row=1) should be 1.
        assert_eq!(mask[10 + 1], 1);
    }

    #[test]
    fn last_writer_wins() {
        let r1 = vec![vec![
            (0.0, 0.0),
            (4.0, 0.0),
            (4.0, 4.0),
            (0.0, 4.0),
            (0.0, 0.0),
        ]];
        let r2 = vec![vec![
            (2.0, 2.0),
            (4.0, 2.0),
            (4.0, 4.0),
            (2.0, 4.0),
            (2.0, 2.0),
        ]];
        let mask = rasterize(&[(r1, 1), (r2, 2)], 4, 4, identity(), false).unwrap();
        assert_eq!(mask[3 * 4 + 3], 2);
        assert_eq!(mask[0], 1);
    }

    #[test]
    fn singular_transform_errors() {
        let bad = Affine {
            a: 0.0,
            b: 0.0,
            c: 0.0,
            d: 0.0,
            e: 0.0,
            f: 0.0,
        };
        let rings = vec![vec![(0.0, 0.0), (1.0, 0.0), (1.0, 1.0), (0.0, 0.0)]];
        assert!(rasterize(&[(rings, 1)], 4, 4, bad, false).is_err());
    }
}
