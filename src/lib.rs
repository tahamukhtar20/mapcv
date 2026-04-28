//! MapCV Rust Core
//!
//! This module provides the performance-critical implementations for mapcv,
//! including tile fetching, stitching, and rasterization.

pub mod tile_math;

use pyo3::prelude::*;
use tile_math::{TileIndex, BBox};

/// A simple test function to ensure Python bindings work.
#[pyfunction]
fn hello() -> PyResult<String> {
    Ok("Hello from mapcv Rust core!".to_string())
}

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

impl From<TileIndex> for PyTileIndex {
    fn from(t: TileIndex) -> Self {
        PyTileIndex { x: t.x, y: t.y, z: t.z }
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

impl From<BBox> for PyBBox {
    fn from(b: BBox) -> Self {
        PyBBox { west: b.west, south: b.south, east: b.east, north: b.north }
    }
}

/// Convert longitude/latitude to Web Mercator meters.
#[pyfunction]
fn xy(lng: f64, lat: f64) -> PyResult<(f64, f64)> {
    Ok(tile_math::xy(lng, lat))
}

/// Get the tile containing a longitude and latitude.
#[pyfunction]
fn tile(lng: f64, lat: f64, zoom: u8) -> PyResult<PyTileIndex> {
    Ok(tile_math::tile(lng, lat, zoom).into())
}

/// Get the tiles overlapped by a geographic bounding box.
#[pyfunction]
fn tiles(west: f64, south: f64, east: f64, north: f64, zooms: Vec<u8>) -> PyResult<Vec<PyTileIndex>> {
    let result = tile_math::tiles(west, south, east, north, &zooms);
    Ok(result.into_iter().map(Into::into).collect())
}

/// Get the web mercator bounding box of a tile in meters.
#[pyfunction]
fn xy_bounds(x: u32, y: u32, z: u8) -> PyResult<PyBBox> {
    let t = TileIndex { x, y, z };
    Ok(tile_math::xy_bounds(&t).into())
}

/// The mapcv Rust extension module.
#[pymodule]
fn _mapcv_rs(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(hello, m)?)?;
    m.add_function(wrap_pyfunction!(xy, m)?)?;
    m.add_function(wrap_pyfunction!(tile, m)?)?;
    m.add_function(wrap_pyfunction!(tiles, m)?)?;
    m.add_function(wrap_pyfunction!(xy_bounds, m)?)?;
    m.add_class::<PyTileIndex>()?;
    m.add_class::<PyBBox>()?;
    Ok(())
}
