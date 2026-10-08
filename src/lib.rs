//! `MapCV` Rust Core
//!
//! The Rust core of `mapcv`: tile math, tile fetching and decoding, rasterization,
//! GeoTIFF I/O, KML parsing and patch writing.
//!
//! The Python bindings (`_mapcv_rs`) are behind the default `python` feature. Build
//! with `--no-default-features` to get just the Rust modules, with no `PyO3` and no
//! numpy; the fuzz targets in `fuzz/` link the crate that way.

#[cfg(feature = "python")]
pub mod fetcher;
pub mod geotiff;
pub mod geotiff_writer;
pub mod http_policy;
pub mod kml_parser;
pub mod patch_writer;
#[cfg(feature = "python")]
mod python;
pub mod rasterizer;
pub mod sampler;
pub mod stitcher;
pub mod tile_decoder;
pub mod tile_math;
