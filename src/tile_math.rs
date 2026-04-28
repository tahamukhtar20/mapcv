//! Tile math implementation matching mercantile.
//! Web Mercator projection and XYZ tile logic.

use std::f64::consts::PI;

const RE: f64 = 6378137.0;
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

/// A geographic bounding box.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct BBox {
    /// West longitude
    pub west: f64,
    /// South latitude
    pub south: f64,
    /// East longitude
    pub east: f64,
    /// North latitude
    pub north: f64,
}

/// Convert longitude and latitude to web mercator x, y (in meters).
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

/// Internal fractional tile coordinate math.
fn _xy_fractional(lng: f64, lat: f64) -> (f64, f64) {
    let x = lng / 360.0 + 0.5;
    let sinlat = lat.to_radians().sin();
    let y = 0.5 - 0.25 * ((1.0 + sinlat) / (1.0 - sinlat)).ln() / PI;
    (x, y)
}

/// Get the tile containing a longitude and latitude.
pub fn tile(lng: f64, lat: f64, zoom: u8) -> TileIndex {
    let (x, y) = _xy_fractional(lng, lat);
    let z2 = f64::powi(2.0, zoom as i32);

    let xtile = if x <= 0.0 {
        0
    } else if x >= 1.0 {
        (z2 - 1.0) as u32
    } else {
        ((x + EPSILON) * z2).floor() as u32
    };

    let ytile = if y <= 0.0 {
        0
    } else if y >= 1.0 {
        (z2 - 1.0) as u32
    } else {
        ((y + EPSILON) * z2).floor() as u32
    };

    TileIndex { x: xtile, y: ytile, z: zoom }
}

/// Get the web mercator bounding box of a tile in meters.
pub fn xy_bounds(tile: &TileIndex) -> BBox {
    let tile_size = CE / f64::powi(2.0, tile.z as i32);

    let left = (tile.x as f64) * tile_size - CE / 2.0;
    let right = left + tile_size;

    let top = CE / 2.0 - (tile.y as f64) * tile_size;
    let bottom = top - tile_size;

    BBox {
        west: left,
        south: bottom,
        east: right,
        north: top,
    }
}

/// Get the tiles overlapped by a geographic bounding box.
pub fn tiles(west: f64, south: f64, east: f64, north: f64, zooms: &[u8]) -> Vec<TileIndex> {
    let mut bboxes = Vec::new();

    if west > east {
        bboxes.push((-180.0, south, east, north));
        bboxes.push((west, south, 180.0, north));
    } else {
        bboxes.push((west, south, east, north));
    }

    let mut result = Vec::new();

    for (w, s, e, n) in bboxes {
        let w = w.max(-180.0);
        let s = s.max(-85.051129);
        let e = e.min(180.0);
        let n = n.min(85.051129);

        for &z in zooms {
            let ul_tile = tile(w, n, z);
            let lr_tile = tile(e - LL_EPSILON, s + LL_EPSILON, z);

            for i in ul_tile.x..=lr_tile.x {
                for j in ul_tile.y..=lr_tile.y {
                    result.push(TileIndex { x: i, y: j, z });
                }
            }
        }
    }

    result
}
