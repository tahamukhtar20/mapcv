// This file is part of mapcv (MIT License, see LICENSE). Parts of it are
// derived from GDAL 3.12.1 (https://github.com/OSGeo/gdal/tree/v3.12.1) and
// remain under GDAL's MIT license; see also THIRD_PARTY_NOTICES.md:
//
// - `Scratch::fill`: `GDALdllImageFilledPolygon()`, alg/llrasterize.cpp
// - `walk_edge`: `GDALdllImageLineAllTouched()`, alg/llrasterize.cpp
// - `is_clockwise` and the ring reorientation in `Scratch::load`:
//   `OGRLineString::isClockwise()`, ogr/ogrlinestring.cpp, as used by
//   `GDALCollectRingsFromGeometry()`, alg/gdalrasterize.cpp
// - `Affine::inverse`, `Affine::apply`: `GDALInvGeoTransform()` and
//   `GDALGenImgProjTransform()`, alg/gdaltransformer.cpp
//
// alg/llrasterize.cpp:
//   Copyright (c) 2000, Frank Warmerdam <warmerdam@pobox.com>
//   Copyright (c) 2011, Even Rouault <even dot rouault at spatialys.com>
// alg/gdalrasterize.cpp:
//   Copyright (c) 2005, Frank Warmerdam <warmerdam@pobox.com>
//   Copyright (c) 2008-2013, Even Rouault <even dot rouault at spatialys.com>
// ogr/ogrlinestring.cpp:
//   Copyright (c) 1999, Frank Warmerdam
//   Copyright (c) 2008-2014, Even Rouault <even dot rouault at spatialys.com>
// alg/gdaltransformer.cpp:
//   Copyright (c) 2002, i3 - information integration and imaging
//                            Fort Collin, CO
//   Copyright (c) 2008-2013, Even Rouault <even dot rouault at spatialys.com>
//   Copyright (c) 2021, CLS
//
// Permission is hereby granted, free of charge, to any person obtaining a
// copy of this software and associated documentation files (the "Software"),
// to deal in the Software without restriction, including without limitation
// the rights to use, copy, modify, merge, publish, distribute, sublicense,
// and/or sell copies of the Software, and to permit persons to whom the
// Software is furnished to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included
// in all copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS
// OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL
// THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
// FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
// DEALINGS IN THE SOFTWARE.

//! Scanline polygon rasterizer.
//!
//! Burns `(polygon, class_id)` pairs into a u8 raster with the same pixel
//! rules as `rasterio.features.rasterize`, which calls GDAL's
//! `GDALRasterizeGeometries`. The fill and edge walking below follow GDAL's
//! `alg/llrasterize.cpp` (MIT licensed) operation for operation, so that
//! pixels whose centres sit exactly on an edge resolve the same way:
//!
//! - `all_touched=False`: a pixel is burned when its centre is inside the
//!   polygon (even-odd rule over every ring, so holes need no special care).
//!   On the scanline through row centre `row + 0.5`, each non-horizontal edge
//!   spanning `[y_top, y_bottom)` crosses at some `x`, which is rounded to the
//!   pixel boundary `floor(x + 0.5)`; columns between consecutive boundaries
//!   are burned. A centre exactly on a left edge is therefore outside and one
//!   exactly on a right edge inside. Horizontal edges never take part in the
//!   parity count; one lying exactly on a centre row is burned on its own
//!   when it runs right to left after its ring has been turned clockwise in
//!   world coordinates (GDAL's "bottom horizontal edge" rule).
//! - `all_touched=True`: the same fill, plus every pixel an edge passes
//!   through. Edges lying on pixel boundaries (within 1e-4 px) burn nothing,
//!   so a pixel-aligned square burns exactly the pixels inside it. Like GDAL,
//!   edges within 0.01 px of vertical or horizontal are drawn as straight runs
//!   along one column or row. This is the rule of GDAL 3.11 and later; older
//!   GDAL also drew any edge whose ends share a column or row that way, which
//!   could miss pixels a diagonal edge crosses.
//!
//! mapcv's own output is identical on every platform (Rust never contracts
//! floating-point operations into fused multiply-adds); rasterio may differ
//! from it on tie cases on arm64, where GDAL is built with FMA contraction.
//!
//! Coordinate conventions match rasterio:
//! - The affine transform maps `(col, row)` -> `(x, y)` in world coords.
//! - Pixel `(col, row)` has its center at `(col + 0.5, row + 0.5)`.
//! - A 6-tuple `(a, b, c, d, e, f)` represents:
//!   `x = a*col + b*row + c`, `y = d*col + e*row + f`.
//! - The inverse transform maps world `(x, y)` -> pixel `(col, row)`.

// f64 cast precision loss is acceptable: pixel coords are bounded well below 2^52.
// Single-char names mirror the rasterio Affine convention (a..f) on purpose.
// Exact float comparisons are deliberate: they reproduce GDAL's tie rules.
#![allow(
    clippy::cast_precision_loss,
    clippy::cast_possible_truncation,
    clippy::cast_possible_wrap,
    clippy::cast_sign_loss,
    clippy::float_cmp,
    clippy::many_single_char_names
)]

/// Edges whose ends both lie this close to a pixel boundary are treated as
/// lying on it by the `all_touched` edge walker (GDAL's
/// `EPSILON_INTERSECT_ONLY`).
const ALIGNED_EPS: f64 = 1e-4;

/// Edges closer than this (in pixels) to vertical or horizontal are walked
/// as a single column or row, as GDAL does.
const AXIS_EPS: f64 = 0.01;

/// Smallest row step the edge walker takes when it sits on a row boundary.
const MIN_ROW_STEP: f64 = 1e-9;

/// Vertex distance below which ring orientation falls back to the shoelace
/// sum (OGR's `isClockwise` epsilon, in world units).
const ORIENTATION_EPS: f64 = 1e-5;

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
    /// Invert the affine transform, with the same arithmetic as GDAL's
    /// `GDALInvGeoTransform` so pixel coordinates round identically.
    ///
    /// # Errors
    /// Returns an error if a coefficient is not finite, or if the transform
    /// is singular: its determinant is zero relative to the size of its
    /// coefficients (`|det| <= 1e-10 * max|coef|^2`), so transforms with
    /// tiny pixels (such as 5e-7 degrees) are still accepted.
    pub fn inverse(self) -> Result<Self, String> {
        let Affine { a, b, c, d, e, f } = self;
        if ![a, b, c, d, e, f].iter().all(|v| v.is_finite()) {
            return Err("Affine transform has a non-finite coefficient".to_string());
        }
        // No rotation: invert each axis on its own, as GDAL does.
        if b == 0.0 && d == 0.0 && a != 0.0 && e != 0.0 {
            return Ok(Affine {
                a: 1.0 / a,
                b: 0.0,
                c: -c / a,
                d: 0.0,
                e: 1.0 / e,
                f: -f / e,
            });
        }
        let det = a * e - b * d;
        let magnitude = a.abs().max(b.abs()).max(d.abs().max(e.abs()));
        if det.abs() <= 1e-10 * magnitude * magnitude {
            return Err("Affine transform is singular (determinant ~ 0)".to_string());
        }
        let inv_det = 1.0 / det;
        Ok(Affine {
            a: e * inv_det,
            b: -b * inv_det,
            c: (b * f - c * e) * inv_det,
            d: -d * inv_det,
            e: a * inv_det,
            f: (-a * f + c * d) * inv_det,
        })
    }

    /// Apply the transform to a point (in GDAL's evaluation order).
    #[must_use]
    pub fn apply(&self, x: f64, y: f64) -> (f64, f64) {
        (
            self.c + x * self.a + y * self.b,
            self.f + x * self.d + y * self.e,
        )
    }
}

/// A polygon: a list of rings in pixel coordinates. Ring 0 is the exterior;
/// any subsequent rings are holes. The even-odd fill rule handles holes
/// implicitly when all rings are merged into one edge list.
pub type RingSet = Vec<Vec<(f64, f64)>>;

/// OGR's `OGRLineString::isClockwise` on a closed ring (first point repeated
/// last), in world coordinates with y pointing up.
fn is_clockwise(p: &[(f64, f64)]) -> bool {
    let n = p.len();
    if n < 2 {
        return true;
    }
    let mut use_fallback = false;
    // Lowest, then rightmost, vertex.
    let mut v = 0;
    for i in 1..n - 1 {
        if p[i].1 < p[v].1 || (p[i].1 == p[v].1 && p[i].0 > p[v].0) {
            v = i;
            use_fallback = false;
        } else if p[i].1 == p[v].1 && p[i].0 == p[v].0 {
            // Two vertices share the pivot position: the cross product
            // below would be meaningless.
            use_fallback = true;
        }
    }
    let near = |q: (f64, f64)| {
        (q.0 - p[v].0).abs() < ORIENTATION_EPS && (q.1 - p[v].1).abs() < ORIENTATION_EPS
    };
    let prev = if v == 0 { n - 2 } else { v - 1 };
    if near(p[prev]) {
        use_fallback = true;
    }
    let dx0 = p[prev].0 - p[v].0;
    let dy0 = p[prev].1 - p[v].1;
    let next = if v + 1 >= n - 1 { 0 } else { v + 1 };
    if near(p[next]) {
        use_fallback = true;
    }
    let dx1 = p[next].0 - p[v].0;
    let dy1 = p[next].1 - p[v].1;
    let cross = dx1 * dy0 - dx0 * dy1;
    if !use_fallback {
        if cross > 0.0 {
            return false;
        } else if cross < 0.0 {
            return true;
        }
    }
    // Degenerate pivot: shoelace sum.
    let mut sum = p[0].0 * (p[1].1 - p[n - 1].1);
    for i in 1..n - 1 {
        sum += p[i].0 * (p[i + 1].1 - p[i - 1].1);
    }
    sum += p[n - 1].0 * (p[0].1 - p[n - 2].1);
    sum < 0.0
}

/// The output buffer plus the value being burned.
struct Canvas<'a> {
    out: &'a mut [u8],
    width: usize,
    height: usize,
    class_id: u8,
}

impl Canvas<'_> {
    /// Burn columns `start..=end` of `row`, clipped to the raster.
    fn burn_span(&mut self, row: i64, start: i64, end: i64) {
        let start = start.max(0);
        let end = end.min(self.width as i64 - 1);
        if start > end || row < 0 || row >= self.height as i64 {
            return;
        }
        let offset = row as usize * self.width;
        self.out[offset + start as usize..=offset + end as usize].fill(self.class_id);
    }

    /// Burn one pixel, ignoring positions outside the raster.
    fn burn_pixel(&mut self, row: i64, col: i64) {
        if row >= 0 && col >= 0 && (row as usize) < self.height && (col as usize) < self.width {
            self.out[row as usize * self.width + col as usize] = self.class_id;
        }
    }
}

/// A non-horizontal edge, oriented so that `y_top < y_bottom`.
#[derive(Clone, Copy)]
struct Edge {
    y_top: f64,
    y_bottom: f64,
    x_top: f64,
    /// `x_bottom - x_top`.
    dx: f64,
    /// `y_bottom - y_top`.
    dy: f64,
}

/// One polygon in pixel coordinates, plus buffers reused across polygons.
#[derive(Default)]
struct Scratch {
    xs: Vec<f64>,
    ys: Vec<f64>,
    /// Number of points in each ring (each ring is closed).
    ring_sizes: Vec<usize>,
    edges: Vec<Edge>,
    active: Vec<Edge>,
    crossings: Vec<i64>,
}

impl Scratch {
    /// Load one polygon: close each ring, turn it clockwise in world
    /// coordinates (as GDAL does before burning) and map it to pixel space.
    /// Returns `false` if nothing can be burned, including when a vertex is
    /// not finite.
    fn load(&mut self, rings: &RingSet, inv: &Affine) -> bool {
        self.xs.clear();
        self.ys.clear();
        self.ring_sizes.clear();
        let mut closed: Vec<(f64, f64)> = Vec::new();
        for ring in rings {
            if ring.len() < 2 {
                continue;
            }
            closed.clear();
            closed.extend_from_slice(ring);
            if ring[0] != ring[ring.len() - 1] {
                closed.push(ring[0]);
            }
            if !is_clockwise(&closed) {
                closed.reverse();
            }
            for &(x, y) in &closed {
                let (px, py) = inv.apply(x, y);
                if !px.is_finite() || !py.is_finite() {
                    return false;
                }
                self.xs.push(px);
                self.ys.push(py);
            }
            self.ring_sizes.push(closed.len());
        }
        !self.xs.is_empty()
    }

    /// `true` if the polygon cannot touch the raster at all (both the fill
    /// and the edge walker would burn nothing).
    fn outside(&self, width: f64, height: f64) -> bool {
        let fold = |v: &[f64]| {
            v.iter()
                .fold((f64::INFINITY, f64::NEG_INFINITY), |(lo, hi), &x| {
                    (lo.min(x), hi.max(x))
                })
        };
        let (x_min, x_max) = fold(&self.xs);
        let (y_min, y_max) = fold(&self.ys);
        x_max < 0.0 || x_min > width || y_max < 0.0 || y_min > height
    }

    /// Iterate each ring's edges as `(from, to)` index pairs, starting with
    /// the edge from the last point back to the first.
    fn edge_indices(&self) -> impl Iterator<Item = (usize, usize)> + '_ {
        let mut start = 0;
        self.ring_sizes.iter().flat_map(move |&size| {
            let first = start;
            start += size;
            (0..size).map(move |k| {
                let from = if k == 0 {
                    first + size - 1
                } else {
                    first + k - 1
                };
                (from, first + k)
            })
        })
    }

    /// Burn the pixels whose centres are inside the polygon (GDAL's
    /// `GDALdllImageFilledPolygon`).
    fn fill(&mut self, canvas: &mut Canvas) {
        let y_min = self.ys.iter().copied().fold(f64::INFINITY, f64::min);
        let y_max = self.ys.iter().copied().fold(f64::NEG_INFINITY, f64::max);
        let first_row = y_min.max(0.0) as i64;
        let last_row = y_max.min(canvas.height as f64 - 1.0) as i64;
        let max_col = canvas.width as i64 - 1;
        if first_row > last_row {
            return;
        }

        let mut edges = std::mem::take(&mut self.edges);
        edges.clear();
        for (from, to) in self.edge_indices() {
            let (y1, y2) = (self.ys[from], self.ys[to]);
            let (x1, x2) = (self.xs[from], self.xs[to]);
            if y1 < y2 {
                edges.push(Edge {
                    y_top: y1,
                    y_bottom: y2,
                    x_top: x1,
                    dx: x2 - x1,
                    dy: y2 - y1,
                });
            } else if y1 > y2 {
                edges.push(Edge {
                    y_top: y2,
                    y_bottom: y1,
                    x_top: x2,
                    dx: x1 - x2,
                    dy: y1 - y2,
                });
            } else if x1 > x2 {
                // Horizontal edge running right to left: burned on its own if
                // it lies exactly on a pixel-centre row.
                let row = (y1 - 0.5).floor();
                if row + 0.5 != y1 {
                    continue;
                }
                let row = row as i64;
                if row < first_row || row > last_row {
                    continue;
                }
                let start = (x2 + 0.5).floor();
                let end = (x1 + 0.5).floor();
                if start > max_col as f64 || end <= 0.0 {
                    continue;
                }
                let start = start.max(0.0) as i64;
                let end = end.min(canvas.width as f64) as i64;
                canvas.burn_span(row, start, end - 1);
            }
            // Horizontal edges running left to right are skipped: the rows
            // they sit on are already handled by the edges below them.
        }
        edges.sort_unstable_by(|p, q| p.y_top.total_cmp(&q.y_top));

        let mut active = std::mem::take(&mut self.active);
        let mut crossings = std::mem::take(&mut self.crossings);
        active.clear();
        let mut next = 0;
        let int_min = f64::from(i32::MIN);
        let int_max = f64::from(i32::MAX);
        for row in first_row..=last_row {
            let y = row as f64 + 0.5;
            while next < edges.len() && edges[next].y_top <= y {
                active.push(edges[next]);
                next += 1;
            }
            // Half-open [y_top, y_bottom): a vertex on the scanline is
            // counted once, by the edge that starts there.
            active.retain(|edge| y < edge.y_bottom);
            if active.is_empty() {
                if next == edges.len() {
                    break;
                }
                continue;
            }
            crossings.clear();
            crossings.extend(active.iter().map(|edge| {
                let x = ((y - edge.y_top) * edge.dx / edge.dy + edge.x_top).clamp(int_min, int_max);
                (x + 0.5).floor() as i64
            }));
            crossings.sort_unstable();
            // Even-odd: burn between crossings 0-1, 2-3, ...
            let (pairs, _) = crossings.as_chunks::<2>();
            for &[left, right] in pairs {
                if left <= max_col && right > 0 {
                    canvas.burn_span(row, left, right - 1);
                }
            }
        }
        self.edges = edges;
        self.active = active;
        self.crossings = crossings;
    }

    /// Burn every pixel an edge passes through (GDAL's
    /// `GDALdllImageLineAllTouched` with `bIntersectOnly`).
    fn walk_edges(&self, canvas: &mut Canvas) {
        let mut start = 0;
        for &size in &self.ring_sizes {
            for k in start + 1..start + size {
                walk_edge(
                    (self.xs[k - 1], self.ys[k - 1]),
                    (self.xs[k], self.ys[k]),
                    canvas,
                );
            }
            start += size;
        }
    }
}

/// Burn the pixels one edge passes through, stepping cell by cell from its
/// left end. A port of one segment of GDAL's `GDALdllImageLineAllTouched`.
#[allow(clippy::too_many_lines)]
fn walk_edge(p0: (f64, f64), p1: (f64, f64), canvas: &mut Canvas) {
    let width = canvas.width as f64;
    let height = canvas.height as f64;
    let last_col = canvas.width as i64 - 1;
    let last_row = canvas.height as i64 - 1;
    let (mut x, mut y) = p0;
    let (mut x_end, mut y_end) = p1;

    if (y < 0.0 && y_end < 0.0)
        || (y > height && y_end > height)
        || (x < 0.0 && x_end < 0.0)
        || (x > width && x_end > width)
    {
        return;
    }
    let int_range = f64::from(i32::MIN)..=f64::from(i32::MAX);
    if ![x, y, x_end, y_end].iter().all(|v| int_range.contains(v)) {
        return;
    }
    // Walk left to right.
    if x > x_end {
        std::mem::swap(&mut x, &mut x_end);
        std::mem::swap(&mut y, &mut y_end);
    }
    let on_boundary = |v: f64| (v - v.round()).abs() < ALIGNED_EPS;

    // (Nearly) vertical: one column.
    if (x - x_end).abs() < AXIS_EPS {
        if on_boundary(x) && on_boundary(x_end) {
            return;
        }
        if y_end < y {
            std::mem::swap(&mut y, &mut y_end);
        }
        let col = x_end.floor() as i64;
        if col < 0 || col > last_col {
            return;
        }
        let first = (y.floor() as i64).max(0);
        let last = ((y_end - ALIGNED_EPS).floor() as i64).min(last_row);
        for row in first..=last {
            canvas.burn_pixel(row, col);
        }
        return;
    }

    // (Nearly) horizontal: one row.
    if (y - y_end).abs() < AXIS_EPS {
        if on_boundary(y) && on_boundary(y_end) {
            return;
        }
        let row = y.floor() as i64;
        if row < 0 || row > last_row {
            return;
        }
        let first = (x.floor() as i64).max(0);
        let last = ((x_end - ALIGNED_EPS).floor() as i64).min(last_col);
        canvas.burn_span(row, first, last);
        return;
    }

    // General case: clip to the raster, then step from cell to cell.
    let slope = (y_end - y) / (x_end - x);
    if x_end > width {
        y_end -= (x_end - width) * slope;
        x_end = width;
    }
    if x < 0.0 {
        y += (0.0 - x) * slope;
        x = 0.0;
    }
    if y_end > y {
        if y < 0.0 {
            x += (0.0 - y) / slope;
            y = 0.0;
        }
        if y_end >= height {
            // GDAL moves the end the wrong way here; the extra cells are all
            // below the raster, so the burned pixels are unaffected.
            x_end += (y_end - height) / slope;
            if x_end > width {
                x_end = width;
            }
        }
    } else {
        if y >= height {
            x += (height - y) / slope;
            y = height;
        }
        if y_end < 0.0 {
            x_end -= (y_end - 0.0) / slope;
        }
    }
    // When the x clip above leaves `y_end == y` on a rising edge, GDAL takes
    // the branch just above and pushes `x_end` past the right border; it then
    // writes those out-of-raster cells into the start of the next row of its
    // buffer. Only cells left of the border are real, so stop there.
    x_end = x_end.min(width);

    while x >= 0.0 && x < x_end {
        let col = x.floor() as i64;
        let row = y.floor() as i64;
        if (0..=last_row).contains(&row) {
            canvas.burn_pixel(row, col);
        } else if (slope > 0.0 && row > last_row) || (slope < 0.0 && row < 0) {
            // y only moves away from the raster from here on.
            break;
        }
        let mut step_x = (x + 1.0).floor() - x;
        let mut step_y = step_x * slope;
        if (y + step_y).floor() as i64 == row {
            // Next column, same row.
            x += step_x;
            y += step_y;
        } else if slope < 0.0 {
            // Row above.
            step_y = row as f64 - y;
            if step_y > -MIN_ROW_STEP {
                step_y = -MIN_ROW_STEP;
            }
            step_x = step_y / slope;
            x += step_x;
            y += step_y;
        } else {
            // Row below.
            step_y = (row + 1) as f64 - y;
            if step_y < MIN_ROW_STEP {
                step_y = MIN_ROW_STEP;
            }
            step_x = step_y / slope;
            x += step_x;
            y += step_y;
        }
    }
}

/// Burn a sequence of `(polygon, class_id)` pairs into an output raster.
///
/// Each polygon is a `RingSet` (exterior ring followed by zero or more
/// holes), with vertices in world coordinates; rings are closed if their
/// last vertex differs from the first. The `transform` maps
/// (col, row) -> (x, y) in world space; rasterization happens in pixel
/// space using its inverse. Pixel rules match
/// `rasterio.features.rasterize` (see the module docs). Polygons with a
/// non-finite vertex are skipped.
///
/// Polygons are processed in order; later polygons overwrite earlier ones
/// (replace / last-writer-wins).
///
/// # Errors
/// Returns an error if `transform` is singular or not finite, or if
/// `width * height` overflows `usize` or the raster cannot be allocated.
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
    // `vec![0; n]` aborts the process when the allocation fails.
    let mut out = Vec::new();
    out.try_reserve_exact(buf_len)
        .map_err(|_| format!("a {width} x {height} raster is too large to allocate"))?;
    out.resize(buf_len, 0u8);
    if buf_len == 0 {
        return Ok(out);
    }
    let mut canvas = Canvas {
        out: &mut out,
        width,
        height,
        class_id: 0,
    };
    let mut scratch = Scratch::default();
    for (rings, class_id) in polygons {
        if !scratch.load(rings, &inv) || scratch.outside(width as f64, height as f64) {
            continue;
        }
        canvas.class_id = *class_id;
        if all_touched {
            scratch.walk_edges(&mut canvas);
        }
        scratch.fill(&mut canvas);
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

    fn square(x0: f64, y0: f64, side: f64) -> RingSet {
        vec![vec![
            (x0, y0),
            (x0 + side, y0),
            (x0 + side, y0 + side),
            (x0, y0 + side),
            (x0, y0),
        ]]
    }

    fn rows(mask: &[u8], width: usize) -> Vec<Vec<u8>> {
        mask.chunks(width).map(<[u8]>::to_vec).collect()
    }

    #[test]
    fn unit_square_fills_one_pixel_at_origin() {
        let mask = rasterize(&[(square(0.0, 0.0, 1.0), 1)], 4, 4, identity(), false).unwrap();
        assert_eq!(mask[0], 1);
        let ones: usize = mask.iter().map(|&v| usize::from(v == 1)).sum();
        assert_eq!(ones, 1);
    }

    #[test]
    fn full_image_square() {
        let mask = rasterize(&[(square(0.0, 0.0, 4.0), 7)], 4, 4, identity(), false).unwrap();
        assert!(mask.iter().all(|&v| v == 7));
    }

    #[test]
    fn polygon_with_hole() {
        // 10x10 outer ring, 4x4 hole in the middle (cols 3..=6, rows 3..=6).
        let outer = square(0.0, 0.0, 10.0).remove(0);
        let hole = square(3.0, 3.0, 4.0).remove(0);
        let mask = rasterize(&[(vec![outer, hole], 1)], 10, 10, identity(), false).unwrap();
        // Hole pixel center (5.5, 5.5) should be 0.
        assert_eq!(mask[5 * 10 + 5], 0);
        // Edge pixel (col=1, row=1) should be 1.
        assert_eq!(mask[10 + 1], 1);
    }

    #[test]
    fn last_writer_wins() {
        let r1 = square(0.0, 0.0, 4.0);
        let r2 = square(2.0, 2.0, 2.0);
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

    #[test]
    fn tiny_pixels_are_not_singular() {
        // #71: 5e-7 degree pixels have a determinant of 2.5e-13.
        let degrees = Affine {
            a: 5e-7,
            b: 0.0,
            c: 10.0,
            d: 0.0,
            e: -5e-7,
            f: 50.0,
        };
        assert!(degrees.inverse().is_ok());
        let rotated = Affine {
            a: 5e-7,
            b: 1e-8,
            c: 10.0,
            d: 1e-8,
            e: -5e-7,
            f: 50.0,
        };
        assert!(rotated.inverse().is_ok());
    }

    #[test]
    fn all_touched_burns_every_crossed_pixel() {
        // #70: the edge (0.1, 0.5) -> (3.1, 3.5) crosses pixel (row 1, col 1)
        // through a corner the old point sampling stepped over.
        let tri = vec![vec![(0.1, 0.5), (3.1, 3.5), (0.0, 4.0), (0.1, 0.5)]];
        let mask = rasterize(&[(tri, 1)], 4, 4, identity(), true).unwrap();
        assert_eq!(
            rows(&mask, 4),
            vec![
                vec![1, 0, 0, 0],
                vec![1, 1, 0, 0],
                vec![1, 1, 1, 0],
                vec![1, 1, 1, 1],
            ]
        );
    }

    #[test]
    fn near_horizontal_edges_keep_parity() {
        // #71: edges with |dy| < 1e-12 straddling the row-0 centre line used
        // to be dropped, leaving an odd crossing count. A notch from the top
        // reaches down to y = notch; the centre (1.5, 0.5) is inside the
        // polygon only when the notch stops above it.
        let ring = |notch: f64, eps: f64| {
            vec![vec![
                (0.0, 0.0),
                (4.0, 0.0),
                (4.0, 1.0),
                (2.0, notch + eps),
                (1.0, notch - eps),
                (0.0, 1.0),
            ]]
        };
        for eps in [0.0, 1e-13, 2e-11, 1e-6] {
            for (notch, expected) in [(0.6, [1, 1, 1, 1]), (0.4, [1, 0, 1, 1])] {
                let mask = rasterize(&[(ring(notch, eps), 1)], 4, 1, identity(), false).unwrap();
                assert_eq!(mask, expected, "notch={notch} eps={eps}");
            }
        }
        // The issue's own ring puts the centre (1.5, 0.5) on the notch edge,
        // so rounding decides; these are GDAL 3.12's answers.
        for (eps, expected) in [(1e-13, [1, 1, 1, 1]), (2e-11, [1, 0, 1, 1])] {
            let mask = rasterize(&[(ring(0.5, eps), 1)], 4, 1, identity(), false).unwrap();
            assert_eq!(mask, expected, "eps={eps}");
        }
    }

    #[test]
    fn duplicate_vertices_and_vertex_on_centre_row() {
        // Repeated vertices and a vertex exactly on a pixel-centre row must
        // not change the result.
        let plain = vec![vec![(0.0, 0.0), (4.0, 0.0), (4.0, 4.0), (0.0, 4.0)]];
        let noisy = vec![vec![
            (0.0, 0.0),
            (0.0, 0.0),
            (4.0, 0.0),
            (4.0, 1.5),
            (4.0, 1.5),
            (4.0, 4.0),
            (0.0, 4.0),
            (0.0, 2.5),
            (0.0, 0.0),
        ]];
        for all_touched in [false, true] {
            let a = rasterize(&[(plain.clone(), 1)], 6, 6, identity(), all_touched).unwrap();
            let b = rasterize(&[(noisy.clone(), 1)], 6, 6, identity(), all_touched).unwrap();
            assert_eq!(a, b, "all_touched={all_touched}");
        }
    }

    #[test]
    fn pixel_aligned_square_all_touched_burns_interior_only() {
        // #124: GDAL burns 2x2 for the (1,1)-(3,3) square, not 3x3.
        let mask = rasterize(&[(square(1.0, 1.0, 2.0), 1)], 5, 5, identity(), true).unwrap();
        assert_eq!(
            rows(&mask, 5),
            vec![
                vec![0, 0, 0, 0, 0],
                vec![0, 1, 1, 0, 0],
                vec![0, 1, 1, 0, 0],
                vec![0, 0, 0, 0, 0],
                vec![0, 0, 0, 0, 0],
            ]
        );
    }

    #[test]
    fn centres_on_edges_follow_gdal() {
        // #124: the square (0.5, 0.5)-(2.5, 2.5) has pixel centres on all four
        // edges. Left edge: outside; right edge: inside. Under the identity
        // transform the clockwise ring runs right to left along y=0.5, so that
        // row is burned and the y=2.5 row is not.
        let mask = rasterize(&[(square(0.5, 0.5, 2.0), 1)], 4, 4, identity(), false).unwrap();
        assert_eq!(
            rows(&mask, 4),
            vec![
                vec![0, 1, 1, 0],
                vec![0, 1, 1, 0],
                vec![0, 0, 0, 0],
                vec![0, 0, 0, 0],
            ]
        );
    }

    #[test]
    fn ring_orientation_does_not_matter() {
        let ccw = square(0.5, 0.5, 2.0);
        let cw = vec![ccw[0].iter().rev().copied().collect::<Vec<_>>()];
        for all_touched in [false, true] {
            let a = rasterize(&[(ccw.clone(), 1)], 4, 4, identity(), all_touched).unwrap();
            let b = rasterize(&[(cw.clone(), 1)], 4, 4, identity(), all_touched).unwrap();
            assert_eq!(a, b);
        }
    }

    #[test]
    fn far_outside_polygon_is_skipped() {
        let mask = rasterize(&[(square(1e9, 1e9, 10.0), 1)], 4, 4, identity(), true).unwrap();
        assert!(mask.iter().all(|&v| v == 0));
    }

    #[test]
    fn non_finite_vertices_are_skipped() {
        let rings = vec![vec![(0.0, 0.0), (f64::NAN, 0.0), (4.0, 4.0), (0.0, 4.0)]];
        for all_touched in [false, true] {
            let mask = rasterize(&[(rings.clone(), 1)], 4, 4, identity(), all_touched).unwrap();
            assert!(mask.iter().all(|&v| v == 0));
        }
    }

    #[test]
    fn polygon_right_of_the_raster_burns_nothing() {
        // GDAL 3.12 burns (row 1, col 1) and (row 1, col 3) here: cells at
        // x >= 58 that it writes past the end of row 0.
        let tri = vec![vec![(63.0, 3.75), (62.0, 8.0), (58.0, -6.5)]];
        let mask = rasterize(&[(tri, 1)], 58, 5, identity(), true).unwrap();
        assert!(mask.iter().all(|&v| v == 0));
    }

    #[test]
    fn steep_edge_leaving_the_raster_terminates() {
        // A nearly vertical (but not within 0.01 px) edge running far past
        // the bottom of the raster.
        let tri = vec![vec![(1.0, 0.5), (1.02, 1e9), (0.0, 1e9)]];
        let mask = rasterize(&[(tri, 1)], 4, 4, identity(), true).unwrap();
        assert_eq!(mask[4], 1);
    }

    #[test]
    fn huge_and_non_finite_edges_terminate() {
        // Edges that start far outside the raster must neither hang nor
        // overflow, whatever the magnitude.
        let square = |far: f64| vec![vec![(0.5, 0.5), (far, 0.5), (far, 2.5), (0.5, 2.5)]];
        for far in [1e6, 1e9, 1e15, 1e19, 1e300, f64::INFINITY, f64::NAN] {
            for all_touched in [false, true] {
                rasterize(&[(square(far), 1)], 4, 4, identity(), all_touched).unwrap();
            }
        }
    }

    #[test]
    fn unallocatable_raster_errors() {
        assert!(rasterize(&[], usize::MAX / 2, 2, identity(), false).is_err());
        assert!(rasterize(&[], 1 << 62, 1, identity(), false).is_err());
    }
}
