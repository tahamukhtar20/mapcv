//! Tile math implementation matching mercantile.
//! Web Mercator projection and XYZ tile logic.

use std::f64::consts::PI;

const RE: f64 = 6_378_137.0;
const CE: f64 = 2.0 * PI * RE;
const EPSILON: f64 = 1e-14;
const LL_EPSILON: f64 = 1e-11;
const MAX_LAT: f64 = 85.051_129;
const MIN_LAT: f64 = -85.051_129;
const MAX_LNG: f64 = 180.0;
const MIN_LNG: f64 = -180.0;
const MAX_ZOOM: u8 = 32;

/// Standard tile pixel dimension used by XYZ tile servers.
pub(crate) const TILE_PX: usize = 256;
/// `TILE_PX` as `f64`, derived from `TILE_PX` to stay in sync.
// 256 is exactly representable in f64 (2^8), so no precision is lost.
#[allow(clippy::cast_precision_loss)]
pub(crate) const TILE_PX_F: f64 = TILE_PX as f64;

/// An XYZ tile coordinate.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct TileIndex {
    /// X index.
    pub x: u32,
    /// Y index.
    pub y: u32,
    /// Zoom level.
    pub z: u8,
}

/// An axis-aligned bounding box.
///
/// Coordinate space depends on the API: some functions use geographic
/// longitude/latitude, others use Web Mercator metres.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Bounds {
    /// Western edge.
    pub west: f64,
    /// Southern edge.
    pub south: f64,
    /// Eastern edge.
    pub east: f64,
    /// Northern edge.
    pub north: f64,
}

/// Backward-compatible alias for [`Bounds`].
pub type BBox = Bounds;

/// Convert (lng, lat) to Web Mercator (x, y) in metres.
#[must_use]
pub fn xy(lng: f64, lat: f64) -> (f64, f64) {
    let x = RE * lng.to_radians();
    let y = if lat <= -90.0 {
        f64::NEG_INFINITY
    } else if lat >= 90.0 {
        f64::INFINITY
    } else {
        RE * ((PI * 0.25) + (0.5 * lat.to_radians())).tan().ln()
    };
    (x, y)
}

/// Map (lng, lat) to fractional tile coordinates in [0, 1) using the Web Mercator projection.
#[must_use]
pub(crate) fn xy_fractional(lng: f64, lat: f64) -> (f64, f64) {
    let x = lng / 360.0 + 0.5;
    let sinlat = lat.to_radians().sin();
    let y = 0.5 - 0.25 * ((1.0 + sinlat) / (1.0 - sinlat)).ln() / PI;
    (x, y)
}

/// Get the tile containing a coordinate at the given zoom.
#[must_use]
pub fn tile(lng: f64, lat: f64, zoom: u8) -> TileIndex {
    let clamped_lng = lng.clamp(MIN_LNG, MAX_LNG);
    let clamped_lat = lat.clamp(MIN_LAT, MAX_LAT);
    let clamped_zoom = zoom.min(MAX_ZOOM);
    let (x, y) = xy_fractional(clamped_lng, clamped_lat);

    let z2_f = 2f64.powi(i32::from(clamped_zoom));
    let max_index = if clamped_zoom == MAX_ZOOM {
        u32::MAX
    } else {
        (1u32 << clamped_zoom) - 1
    };

    #[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
    let xtile = if x >= 1.0 {
        max_index
    } else if x <= 0.0 {
        0
    } else {
        let clamped = (x * z2_f).floor().min(f64::from(max_index));
        clamped as u32
    };

    #[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
    let ytile = if y >= 1.0 {
        max_index
    } else if y <= 0.0 {
        0
    } else {
        let clamped = ((y + EPSILON) * z2_f).floor().min(f64::from(max_index));
        clamped as u32
    };

    TileIndex {
        x: xtile,
        y: ytile,
        z: clamped_zoom,
    }
}

/// Web Mercator bounds of a tile in metres.
#[must_use]
pub fn xy_bounds(tile: TileIndex) -> BBox {
    let clamped_zoom = tile.z.min(MAX_ZOOM);
    let max_index = if clamped_zoom == MAX_ZOOM {
        u32::MAX
    } else {
        (1u32 << clamped_zoom) - 1
    };
    let x = tile.x.min(max_index);
    let y = tile.y.min(max_index);

    let z2 = 2f64.powi(i32::from(clamped_zoom));
    let tile_size = CE / z2;

    let left = f64::from(x) * tile_size - CE / 2.0;
    let right = left + tile_size;
    let bottom = CE / 2.0 - (f64::from(y) + 1.0) * tile_size;
    let top = CE / 2.0 - f64::from(y) * tile_size;

    BBox {
        west: left,
        south: bottom,
        east: right,
        north: top,
    }
}

/// Geographic bounds of a tile in degrees.
#[must_use]
pub fn bounds(tile: TileIndex) -> BBox {
    let clamped_zoom = tile.z.min(MAX_ZOOM);
    let max_index = if clamped_zoom == MAX_ZOOM {
        u32::MAX
    } else {
        (1u32 << clamped_zoom) - 1
    };
    let x = tile.x.min(max_index);
    let y = tile.y.min(max_index);

    let z2 = 2f64.powi(i32::from(clamped_zoom));
    let west = f64::from(x) / z2 * 360.0 - 180.0;
    let east = (f64::from(x) + 1.0) / z2 * 360.0 - 180.0;

    let n = PI - 2.0 * PI * (f64::from(y) / z2);
    let s = PI - 2.0 * PI * ((f64::from(y) + 1.0) / z2);

    let north = n.sinh().atan().to_degrees();
    let south = s.sinh().atan().to_degrees();

    BBox {
        west,
        south,
        east,
        north,
    }
}

/// Most tiles [`tiles`] enumerates in one call: 2^24, about 16.8 million, the
/// whole world at zoom 12.
///
/// The whole world has 2^26 tiles at zoom 13 and 2^30 at zoom 15; a list that
/// long would exhaust memory long before anything could be downloaded.
pub const MAX_TILES: u64 = 1 << 24;

/// Reject boxes that have no sensible tile cover.
///
/// Edge cases, decided to match the Python `RegionConfig` checks
/// (`west < east`, `south < north`) without guessing what a caller meant:
/// - A NaN coordinate is an error.
/// - `south > north` is an error: the box is inverted, not empty.
/// - `west > east` is accepted here; [`tiles`] reads it as crossing the
///   antimeridian (as mercantile does) and [`snap_bbox`] rejects it.
/// - `west == east` or `south == north` (a point or a line) is accepted and
///   covers the tiles that contain it, the same tile [`tile`] returns for a
///   point. It never yields an empty cover.
/// - Out-of-range values (including infinities) are clamped to the Web
///   Mercator world, as before.
fn check_bbox(west: f64, south: f64, east: f64, north: f64) -> Result<(), String> {
    if west.is_nan() || south.is_nan() || east.is_nan() || north.is_nan() {
        return Err(format!(
            "bbox coordinates must be numbers, got ({west}, {south}, {east}, {north})"
        ));
    }
    if south > north {
        return Err(format!(
            "bbox south ({south}) must not be greater than north ({north})"
        ));
    }
    Ok(())
}

/// Inclusive tile range `(min_x, min_y, max_x, max_y)` covering a box that
/// does not cross the antimeridian (`west <= east`, `south <= north`).
///
/// The lower-right corner is nudged inward by `LL_EPSILON` so a box ending
/// exactly on a tile edge does not pull in the next tile (mercantile's rule).
/// For a box narrower than that nudge (a zero-width or zero-height box in
/// particular) on a tile edge, the nudge would cross back over the edge and
/// leave an empty range; the range is then the single column or row that
/// [`tile`] assigns the edge to, so a point snaps to the tile containing it.
fn tile_range(west: f64, south: f64, east: f64, north: f64, zoom: u8) -> (u32, u32, u32, u32) {
    let ul = tile(west.max(MIN_LNG), north.min(MAX_LAT), zoom);
    let lr = tile(
        east.min(MAX_LNG) - LL_EPSILON,
        south.max(MIN_LAT) + LL_EPSILON,
        zoom,
    );
    (ul.x, ul.y, lr.x.max(ul.x), lr.y.max(ul.y))
}

/// Expand a bbox outward to full tile boundaries at the given zoom.
///
/// A point or line snaps to the tiles that contain it (see [`check_bbox`]).
///
/// # Errors
/// Returns an error for a NaN coordinate, for `south > north`, and for
/// `west > east`: a box crossing the antimeridian cannot be represented by a
/// single snapped box, so it must be split into two at ±180°. (This function
/// used to return the whole world for these.)
pub fn snap_bbox(west: f64, south: f64, east: f64, north: f64, zoom: u8) -> Result<BBox, String> {
    check_bbox(west, south, east, north)?;
    if west > east {
        return Err(format!(
            "bbox west ({west}) is greater than east ({east}): boxes crossing the \
             antimeridian are not supported; split it into two boxes at 180°"
        ));
    }
    let zoom = zoom.min(MAX_ZOOM);
    let (min_x, min_y, max_x, max_y) = tile_range(west, south, east, north, zoom);
    let ul = bounds(TileIndex {
        x: min_x,
        y: min_y,
        z: zoom,
    });
    let lr = bounds(TileIndex {
        x: max_x,
        y: max_y,
        z: zoom,
    });

    Ok(BBox {
        west: ul.west,
        south: lr.south,
        east: lr.east,
        north: ul.north,
    })
}

/// Split a box with `west > east` at the antimeridian, as mercantile does.
fn split_bbox(west: f64, south: f64, east: f64, north: f64) -> Vec<(f64, f64, f64, f64)> {
    if west > east {
        vec![(MIN_LNG, south, east, north), (west, south, MAX_LNG, north)]
    } else {
        vec![(west, south, east, north)]
    }
}

/// All tiles overlapping a geographic bounding box, at each zoom in `zooms`.
///
/// Matches `mercantile.tiles`, including reading `west > east` as a box that
/// crosses the antimeridian, except that a point or line yields the tiles
/// containing it where mercantile can yield none (see [`check_bbox`]).
///
/// # Errors
/// Returns an error for a NaN coordinate, for `south > north`, and when the
/// cover would exceed [`MAX_TILES`] tiles; the count is checked before any
/// tile is allocated.
pub fn tiles(
    west: f64,
    south: f64,
    east: f64,
    north: f64,
    zooms: &[u8],
) -> Result<Vec<TileIndex>, String> {
    check_bbox(west, south, east, north)?;
    let mut ranges = Vec::new();
    for (w, s, e, n) in split_bbox(west, south, east, north) {
        for &z in zooms {
            let zoom = z.min(MAX_ZOOM);
            ranges.push((zoom, tile_range(w, s, e, n, zoom)));
        }
    }

    let count: u128 = ranges
        .iter()
        .map(|&(_, (x0, y0, x1, y1))| {
            (u128::from(x1) - u128::from(x0) + 1) * (u128::from(y1) - u128::from(y0) + 1)
        })
        .sum();
    if count > u128::from(MAX_TILES) {
        return Err(format!(
            "bbox ({west}, {south}, {east}, {north}) covers {count} tiles at zoom {zooms:?}, \
             more than the limit of {MAX_TILES}; use a lower zoom or a smaller region"
        ));
    }

    // `count` fits in usize: it is at most MAX_TILES.
    let mut result = Vec::with_capacity(usize::try_from(count).unwrap_or(0));
    for (zoom, (x0, y0, x1, y1)) in ranges {
        for x in x0..=x1 {
            for y in y0..=y1 {
                result.push(TileIndex { x, y, z: zoom });
            }
        }
    }
    Ok(result)
}

/// Reject a set of same-zoom tiles that wraps around the antimeridian.
///
/// Stitching places tiles by their offset from the smallest column, so tiles
/// from both sides of ±180° (column 0 and the last column) would become a
/// canvas spanning the whole world width, mostly empty. Such a set is
/// recognised by touching both edge columns with a run of at least half the
/// world's columns missing in between; a contiguous region, even one with a
/// few failed tiles, never has that gap. Mixed zoom levels are left to the
/// stitcher to report.
///
/// # Errors
/// Returns an error describing the gap when the tiles wrap.
pub fn check_no_antimeridian_wrap(tiles: &[TileIndex]) -> Result<(), String> {
    let Some(first) = tiles.first() else {
        return Ok(());
    };
    let zoom = first.z;
    if zoom > MAX_ZOOM || tiles.iter().any(|t| t.z != zoom) {
        return Ok(());
    }
    let n_columns = 1u64 << zoom;
    let columns: std::collections::BTreeSet<u32> = tiles.iter().map(|t| t.x).collect();
    let touches_both_edges =
        columns.first() == Some(&0) && columns.last().map(|&x| u64::from(x)) == Some(n_columns - 1);
    if !touches_both_edges {
        return Ok(());
    }
    let widest_gap = columns
        .iter()
        .zip(columns.iter().skip(1))
        .map(|(&a, &b)| u64::from(b - a - 1))
        .max()
        .unwrap_or(0);
    if widest_gap > 0 && widest_gap >= n_columns / 2 {
        return Err(format!(
            "tiles at zoom {zoom} sit on both sides of the antimeridian (columns 0 and \
             {}, with {widest_gap} empty columns between them); stitch each side separately",
            n_columns - 1
        ));
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn column_tiles(zoom: u8, columns: &[u32]) -> Vec<TileIndex> {
        columns
            .iter()
            .map(|&x| TileIndex { x, y: 0, z: zoom })
            .collect()
    }

    #[test]
    fn tiles_across_the_antimeridian_cannot_be_stitched_as_one() {
        let wrapped = tiles(179.9, 0.0, -179.9, 0.1, &[16]).unwrap();
        assert!(check_no_antimeridian_wrap(&wrapped).is_err());
        assert!(check_no_antimeridian_wrap(&column_tiles(3, &[0, 1, 7])).is_err());
    }

    #[test]
    fn contiguous_or_whole_world_tiles_can_be_stitched() {
        let world = tiles(-180.0, -85.0, 180.0, 85.0, &[3]).unwrap();
        assert!(check_no_antimeridian_wrap(&world).is_ok());
        // A whole-world row with a failed tile in it.
        assert!(check_no_antimeridian_wrap(&column_tiles(3, &[0, 1, 2, 4, 5, 6, 7])).is_ok());
        assert!(check_no_antimeridian_wrap(&column_tiles(1, &[0, 1])).is_ok());
        assert!(check_no_antimeridian_wrap(&column_tiles(0, &[0])).is_ok());
        assert!(check_no_antimeridian_wrap(&column_tiles(16, &[0, 2731])).is_ok());
        // Out-of-range columns are left for the stitcher's canvas checks.
        assert!(check_no_antimeridian_wrap(&column_tiles(31, &[0, u32::MAX])).is_ok());
        assert!(check_no_antimeridian_wrap(&[]).is_ok());
    }

    fn snap(west: f64, south: f64, east: f64, north: f64, zoom: u8) -> BBox {
        snap_bbox(west, south, east, north, zoom).unwrap()
    }

    #[test]
    fn point_snaps_to_the_tile_containing_it() {
        for &(lng, lat) in &[(0.0, 0.0), (13.4, 52.5), (-180.0, 85.0), (180.0, -85.0)] {
            for zoom in [0, 1, 7, 15, 32] {
                assert_eq!(
                    snap(lng, lat, lng, lat, zoom),
                    bounds(tile(lng, lat, zoom)),
                    "({lng}, {lat}) z{zoom}"
                );
                assert_eq!(
                    tiles(lng, lat, lng, lat, &[zoom]).unwrap(),
                    vec![tile(lng, lat, zoom)]
                );
            }
        }
    }

    #[test]
    fn zero_width_and_zero_height_boxes_cover_one_column_or_row() {
        // On the prime meridian and the equator, both tile edges at zoom 3.
        let column = tiles(0.0, -10.0, 0.0, 10.0, &[3]).unwrap();
        assert!(column.iter().all(|t| t.x == 4));
        assert_eq!(column.len(), 2);
        let row = tiles(-10.0, 0.0, 10.0, 0.0, &[3]).unwrap();
        assert!(row.iter().all(|t| t.y == 4));
        assert_eq!(row.len(), 2);
    }

    #[test]
    fn degenerate_boxes_never_snap_to_the_whole_world() {
        let world = BBox {
            west: MIN_LNG,
            south: MIN_LAT,
            east: MAX_LNG,
            north: MAX_LAT,
        };
        for zoom in [1, 10, 15] {
            assert_ne!(snap(0.0, 0.0, 0.0, 0.0, zoom), world);
            assert_ne!(snap(0.0, 0.0, 1e-13, 1e-13, zoom), world);
        }
    }

    #[test]
    fn inverted_and_nan_boxes_are_errors() {
        assert!(snap_bbox(0.0, 1.0, 1.0, 0.0, 10).is_err());
        assert!(tiles(0.0, 1.0, 1.0, 0.0, &[10]).is_err());
        assert!(snap_bbox(f64::NAN, 0.0, 1.0, 1.0, 10).is_err());
        assert!(tiles(0.0, 0.0, 1.0, f64::NAN, &[10]).is_err());
    }

    #[test]
    fn antimeridian_box_is_split_by_tiles_and_rejected_by_snap() {
        let err = snap_bbox(179.9, 0.0, -179.9, 0.1, 16).unwrap_err();
        assert!(err.contains("antimeridian"), "{err}");
        let covered = tiles(179.9, 0.0, -179.9, 0.1, &[16]).unwrap();
        // Two narrow strips at the two edges of the world, not its full width.
        assert!(covered.len() < 1000, "{} tiles", covered.len());
        assert!(covered.iter().all(|t| t.x < 20 || t.x > 65_515));
    }

    #[test]
    fn huge_covers_are_rejected_before_allocating() {
        let err = tiles(-180.0, -85.0, 180.0, 85.0, &[15]).unwrap_err();
        assert!(err.contains("more than the limit"), "{err}");
        assert!(tiles(-180.0, -85.0, 180.0, 85.0, &[32]).is_err());
        // Counts add up across zooms.
        assert!(tiles(-180.0, -85.0, 180.0, 85.0, &[12]).is_ok());
        assert!(tiles(-180.0, -85.0, 180.0, 85.0, &[12, 12]).is_err());
        // Snapping does not enumerate tiles, so it has no limit.
        assert!(snap_bbox(-180.0, -85.0, 180.0, 85.0, 32).is_ok());
    }

    #[test]
    fn ordinary_box_snaps_to_its_tile_range() {
        let b = snap(-122.42, 37.77, -122.41, 37.78, 14);
        let covered = tiles(-122.42, 37.77, -122.41, 37.78, &[14]).unwrap();
        let min_x = covered.iter().map(|t| t.x).min().unwrap();
        let max_y = covered.iter().map(|t| t.y).max().unwrap();
        assert!(
            (b.west
                - bounds(TileIndex {
                    x: min_x,
                    y: 0,
                    z: 14
                })
                .west)
                .abs()
                < 1e-12
        );
        assert!(
            (b.south
                - bounds(TileIndex {
                    x: 0,
                    y: max_y,
                    z: 14
                })
                .south)
                .abs()
                < 1e-12
        );
    }
}
