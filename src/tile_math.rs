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
    /// Western edge
    pub west: f64,
    /// Southern edge
    pub south: f64,
    /// Eastern edge
    pub east: f64,
    /// Northern edge
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
pub(crate) fn xy_fractional(lng: f64, lat: f64) -> (f64, f64) {
    let x = lng / 360.0 + 0.5;
    let sinlat = lat.to_radians().sin();
    let y = 0.5 - 0.25 * ((1.0 + sinlat) / (1.0 - sinlat)).ln() / PI;
    (x, y)
}

/// Get the tile containing a longitude and latitude.
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

/// Get the web mercator bounding box of a tile in meters.
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

/// Get the geographic bounding box of a tile in degrees.
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

/// Snap a geographic bounding box outward to tile boundaries at the given zoom.
#[must_use]
pub fn snap_bbox(west: f64, south: f64, east: f64, north: f64, zoom: u8) -> BBox {
    let tiles_vec = tiles(west, south, east, north, &[zoom]);
    let mut iter = tiles_vec.into_iter();
    let Some(first) = iter.next() else {
        return BBox {
            west: MIN_LNG,
            south: MIN_LAT,
            east: MAX_LNG,
            north: MAX_LAT,
        };
    };

    let mut min_x = first.x;
    let mut min_y = first.y;
    let mut max_x = first.x;
    let mut max_y = first.y;
    let clamped_zoom = first.z;

    for tile in iter {
        min_x = min_x.min(tile.x);
        min_y = min_y.min(tile.y);
        max_x = max_x.max(tile.x);
        max_y = max_y.max(tile.y);
    }

    let ul = bounds(TileIndex {
        x: min_x,
        y: min_y,
        z: clamped_zoom,
    });
    let lr = bounds(TileIndex {
        x: max_x,
        y: max_y,
        z: clamped_zoom,
    });

    BBox {
        west: ul.west,
        south: lr.south,
        east: lr.east,
        north: ul.north,
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
        let w_clamped = w.max(MIN_LNG);
        let s_clamped = s.max(MIN_LAT);
        let e_clamped = e.min(MAX_LNG);
        let n_clamped = n.min(MAX_LAT);

        for &z in zooms {
            let clamped_zoom = z.min(MAX_ZOOM);
            let ul_tile = tile(w_clamped, n_clamped, clamped_zoom);
            let lr_tile = tile(e_clamped - LL_EPSILON, s_clamped + LL_EPSILON, clamped_zoom);

            for i in ul_tile.x..=lr_tile.x {
                for j in ul_tile.y..=lr_tile.y {
                    result.push(TileIndex {
                        x: i,
                        y: j,
                        z: clamped_zoom,
                    });
                }
            }
        }
    }

    result
}
