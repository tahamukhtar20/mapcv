//! `MapCV` Rust Core
//!
//! This module provides the performance-critical implementations for `mapcv`.

use pyo3::prelude::*;

/// A simple test function to ensure Python bindings work.
#[pyfunction]
fn hello() -> String {
    String::from("Hello from mapcv Rust core!")
}

/// The `mapcv` Rust extension module.
#[pymodule]
fn _mapcv_rs(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(hello, m)?)?;
    Ok(())
}
