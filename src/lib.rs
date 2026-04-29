//! `MapCV` Rust Core
//!
//! This module provides the Rust core for `mapcv`, including tile math
//! utilities and Python bindings for those operations.
// PyO3's #[pyfunction] macro generates `PyErr`-to-`PyErr` coercions that
// clippy::useless_conversion flags. This cannot be suppressed at a narrower
// scope because the lint fires inside the macro expansion.
#![allow(clippy::useless_conversion)]

pub mod fetcher;
pub mod tile_math;

use pyo3::prelude::*;
use tile_math::{BBox, TileIndex};

/// A simple test function to ensure Python bindings work.
#[pyfunction]
fn hello() -> String {
    String::from("Hello from mapcv Rust core!")
}

/// The `mapcv` Rust extension module.
#[pyclass]
#[derive(Clone)]
struct PyTileIndex {
    #[pyo3(get)]
    x: u32,
    #[pyo3(get)]
    y: u32,
    #[pyo3(get)]
    z: u8,
}

#[pymethods]
impl PyTileIndex {
    #[new]
    fn new(x: u32, y: u32, z: u8) -> Self {
        PyTileIndex { x, y, z }
    }
}
impl From<TileIndex> for PyTileIndex {
    fn from(t: TileIndex) -> Self {
        PyTileIndex {
            x: t.x,
            y: t.y,
            z: t.z,
        }
    }
}

#[pyclass]
#[derive(Clone)]
struct PyBBox {
    #[pyo3(get)]
    west: f64,
    #[pyo3(get)]
    south: f64,
    #[pyo3(get)]
    east: f64,
    #[pyo3(get)]
    north: f64,
}

#[pymethods]
impl PyBBox {
    #[new]
    fn new(west: f64, south: f64, east: f64, north: f64) -> Self {
        PyBBox {
            west,
            south,
            east,
            north,
        }
    }
}
impl From<BBox> for PyBBox {
    fn from(b: BBox) -> Self {
        PyBBox {
            west: b.west,
            south: b.south,
            east: b.east,
            north: b.north,
        }
    }
}

/// Convert longitude/latitude to Web Mercator meters.
#[pyfunction]
fn xy(lng: f64, lat: f64) -> (f64, f64) {
    tile_math::xy(lng, lat)
}

/// Get the tile containing a longitude and latitude.
#[pyfunction]
fn tile(lng: f64, lat: f64, zoom: u8) -> PyTileIndex {
    tile_math::tile(lng, lat, zoom).into()
}

/// Get the tiles overlapped by a geographic bounding box.
#[pyfunction]
#[allow(clippy::needless_pass_by_value)]
fn tiles(west: f64, south: f64, east: f64, north: f64, zooms: Vec<u8>) -> Vec<PyTileIndex> {
    let result = tile_math::tiles(west, south, east, north, &zooms);
    result.into_iter().map(Into::into).collect()
}

/// Get the web mercator bounding box of a tile in meters.
#[pyfunction]
fn xy_bounds(x: u32, y: u32, z: u8) -> PyBBox {
    let t = TileIndex { x, y, z };
    tile_math::xy_bounds(t).into()
}

/// Get the geographic bounding box of a tile in degrees.
#[pyfunction]
fn bounds(x: u32, y: u32, z: u8) -> PyBBox {
    let t = TileIndex { x, y, z };
    tile_math::bounds(t).into()
}

/// Snap a geographic bounding box outward to tile boundaries at a zoom.
#[pyfunction]
fn snap_bbox(west: f64, south: f64, east: f64, north: f64, zoom: u8) -> PyBBox {
    tile_math::snap_bbox(west, south, east, north, zoom).into()
}

/// Fetch satellite tiles concurrently from a URL template.
///
/// Returns a list of `(PyTileIndex, bytes)` pairs for all successfully fetched
/// tiles (and, with the `ignore` policy, tiles filled with black `NoData` pixels).
///
/// Raises `RuntimeError` if the fraction of failed tiles exceeds
/// `max_failed_ratio`.
#[allow(clippy::needless_pass_by_value, clippy::cast_precision_loss)]
#[pyfunction]
#[pyo3(signature = (tiles, url_template, callback=None, max_connections=16, policy="lenient", max_failed_ratio=0.05))]
fn fetch_tiles(
    py: Python,
    tiles: Vec<PyTileIndex>,
    url_template: String,
    callback: Option<PyObject>,
    max_connections: usize,
    policy: &str,
    max_failed_ratio: f64,
) -> PyResult<Vec<(PyTileIndex, PyObject)>> {
    let rust_tiles: Vec<TileIndex> = tiles
        .into_iter()
        .map(|t| TileIndex {
            x: t.x,
            y: t.y,
            z: t.z,
        })
        .collect();

    let total = rust_tiles.len();

    let (results, failed) = fetcher::fetch_tiles(
        py,
        rust_tiles,
        url_template,
        callback,
        max_connections,
        policy,
    )?;

    if total > 0 && failed as f64 / total as f64 > max_failed_ratio {
        return Err(pyo3::exceptions::PyRuntimeError::new_err(format!(
            "Too many failed tiles: {failed}/{total} ({:.1}% exceeds {:.1}% threshold)",
            100.0 * failed as f64 / total as f64,
            100.0 * max_failed_ratio,
        )));
    }

    Ok(results
        .into_iter()
        .map(|(t, bytes)| {
            (
                PyTileIndex::from(t),
                pyo3::types::PyBytes::new_bound(py, &bytes).into(),
            )
        })
        .collect())
}

#[pymodule]
fn _mapcv_rs(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(hello, m)?)?;
    m.add_function(wrap_pyfunction!(xy, m)?)?;
    m.add_function(wrap_pyfunction!(tile, m)?)?;
    m.add_function(wrap_pyfunction!(tiles, m)?)?;
    m.add_function(wrap_pyfunction!(xy_bounds, m)?)?;
    m.add_function(wrap_pyfunction!(bounds, m)?)?;
    m.add_function(wrap_pyfunction!(snap_bbox, m)?)?;
    m.add_function(wrap_pyfunction!(fetch_tiles, m)?)?;
    m.add_class::<PyTileIndex>()?;
    m.add_class::<PyBBox>()?;
    Ok(())
}
