//! Tile math implementation matching mercantile.
//! Web Mercator projection and XYZ tile logic.

use std::f64::consts::PI;

const RE: f64 = 6_378_137.0;
const CE: f64 = 2.0 * PI * RE;
const EPSILON: f64 = 1e-14;
const LL_EPSILON: f64 = 1e-11;

/// Represents an XYZ tile coordinate.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct TileIndex {
    /// X coordinate
    pub x: u32,
    /// Y coordinate
    pub y: u32,
    /// Zoom level
    pub z: u8,
}

/// An axis-aligned bounding box.
///
/// The coordinate space depends on the API returning or consuming it. Some
/// functions use geographic longitude/latitude values, while others use
/// projected coordinates such as Web Mercator meters.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Bounds {
    /// West longitude
    pub west: f64,
    /// South latitude
    pub south: f64,
    /// East longitude
    pub east: f64,
    /// North latitude
    pub north: f64,
}

/// Backward-compatible alias for [`Bounds`].
pub type BBox = Bounds;

/// Convert longitude and latitude to web mercator x, y (in meters).
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

/// Note: This is an internal helper.
#[must_use]
pub fn xy_fractional(lng: f64, lat: f64) -> (f64, f64) {
    let x = lng / 360.0 + 0.5;
    let sinlat = lat.to_radians().sin();
    let y = 0.5 - 0.25 * ((1.0 + sinlat) / (1.0 - sinlat)).ln() / PI;
    (x, y)
}

/// Get the tile containing a longitude and latitude.
#[must_use]
pub fn tile(lng: f64, lat: f64, zoom: u8) -> TileIndex {
    let (x, y) = xy_fractional(lng, lat);

    let z2 = 1u32 << zoom;

    #[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
    let xtile = if x >= 1.0 {
        z2 - 1
    } else {
        (x * f64::from(z2)).floor() as u32
    };

    #[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
    let ytile = if y >= 1.0 {
        z2 - 1
    } else {
        ((y + EPSILON) * f64::from(z2)).floor() as u32
    };

    TileIndex {
        x: xtile,
        y: ytile,
        z: zoom,
    }
}

/// Get the web mercator bounding box of a tile in meters.
#[must_use]
pub fn xy_bounds(tile: TileIndex) -> BBox {
    let z2 = f64::powi(2.0, i32::from(tile.z));
    let tile_size = CE / z2;

    let left = f64::from(tile.x) * tile_size - CE / 2.0;
    let right = left + tile_size;
    let bottom = CE / 2.0 - f64::from(tile.y + 1) * tile_size;
    let top = CE / 2.0 - f64::from(tile.y) * tile_size;

    BBox {
        west: left,
        south: bottom,
        east: right,
        north: top,
    }
}

fn split_bbox(west: f64, south: f64, east: f64, north: f64) -> Vec<(f64, f64, f64, f64)> {
    let mut bboxes = Vec::new();
    if west > east {
        bboxes.push((-180.0, south, east, north));
        bboxes.push((west, south, 180.0, north));
    } else {
        bboxes.push((west, south, east, north));
    }
    bboxes
}

/// Get the tiles overlapped by a geographic bounding box.
#[must_use]
pub fn tiles(west: f64, south: f64, east: f64, north: f64, zooms: &[u8]) -> Vec<TileIndex> {
    let bboxes = split_bbox(west, south, east, north);

    let mut result = Vec::new();

    for (w, s, e, n) in bboxes {
        let w_clamped = w.max(-180.0);
        let s_clamped = s.max(-85.051_129);
        let e_clamped = e.min(180.0);
        let n_clamped = n.min(85.051_129);

        for &z in zooms {
            let ul_tile = tile(w_clamped, n_clamped, z);
            let lr_tile = tile(e_clamped - LL_EPSILON, s_clamped + LL_EPSILON, z);

            for i in ul_tile.x..=lr_tile.x {
                for j in ul_tile.y..=lr_tile.y {
                    result.push(TileIndex { x: i, y: j, z });
                }
            }
        }
    }

    result
}
