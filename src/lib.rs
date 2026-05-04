//! `MapCV` Rust Core
//!
//! This module provides the Rust core for `mapcv`, including tile math
//! utilities and Python bindings for those operations.
// PyO3's #[pyfunction] macro generates `PyErr`-to-`PyErr` coercions that
// clippy::useless_conversion flags. This cannot be suppressed at a narrower
// scope because the lint fires inside the macro expansion.
#![allow(clippy::useless_conversion)]

pub mod fetcher;
pub mod kml_parser;
pub mod patch_writer;
pub mod rasterizer;
pub mod sampler;
pub mod stitcher;
pub mod tile_math;

use numpy::{
    IntoPyArray, PyArray2, PyArray3, PyReadonlyArray3, PyReadonlyArray4, PyUntypedArrayMethods,
    ToPyArray,
};
use pyo3::prelude::*;
use std::collections::HashMap;
use tile_math::{BBox, TileIndex};

/// Return a greeting string confirming the Rust extension loaded correctly.
#[must_use]
#[pyfunction]
fn hello() -> String {
    String::from("Hello from mapcv Rust core!")
}

/// Python-visible XYZ tile index.
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
    /// Create a new tile index from column *x*, row *y*, and zoom *z*.
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

/// Python-visible geographic bounding box (WGS-84 degrees).
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
    /// Create a bounding box from *west*, *south*, *east*, *north* in WGS-84 degrees.
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

/// Convert (lng, lat) in EPSG:4326 to Web Mercator (x, y) in EPSG:3857.
#[must_use]
#[pyfunction]
fn xy(lng: f64, lat: f64) -> (f64, f64) {
    tile_math::xy(lng, lat)
}

/// Return the XYZ tile index for a (lng, lat) point at the given zoom level.
#[must_use]
#[pyfunction]
fn tile(lng: f64, lat: f64, zoom: u8) -> PyTileIndex {
    tile_math::tile(lng, lat, zoom).into()
}

/// Return all XYZ tiles covering the given bounding box at the specified zoom levels.
#[must_use]
#[pyfunction]
#[allow(clippy::needless_pass_by_value)]
fn tiles(west: f64, south: f64, east: f64, north: f64, zooms: Vec<u8>) -> Vec<PyTileIndex> {
    let result = tile_math::tiles(west, south, east, north, &zooms);
    result.into_iter().map(Into::into).collect()
}

/// Return the Web Mercator bounding box (EPSG:3857, meters) for an XYZ tile.
#[must_use]
#[pyfunction]
fn xy_bounds(x: u32, y: u32, z: u8) -> PyBBox {
    let t = TileIndex { x, y, z };
    tile_math::xy_bounds(t).into()
}

/// Return the geographic bounding box (EPSG:4326, degrees) for an XYZ tile.
#[must_use]
#[pyfunction]
fn bounds(x: u32, y: u32, z: u8) -> PyBBox {
    let t = TileIndex { x, y, z };
    tile_math::bounds(t).into()
}

/// Expand a bbox outward to the nearest tile boundaries at the given zoom level.
#[must_use]
#[pyfunction]
fn snap_bbox(west: f64, south: f64, east: f64, north: f64, zoom: u8) -> PyBBox {
    tile_math::snap_bbox(west, south, east, north, zoom).into()
}

/// Fetch satellite tiles concurrently from a URL template.
///
/// Returns a list of `(PyTileIndex, bytes)` pairs for successfully fetched
/// tiles (and, with the `ignore` policy, black `NoData`-filled tiles).
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

/// Generate grid (or sliding-window) patch anchor positions.
///
/// Returns a list of `(row, col)` top-left corners for `patch_size x patch_size`
/// patches sampled with the given `stride` across a `height x width` image.
/// `edge_strategy` is `"pad"` (default), `"drop"`, or `"shift"`.
#[pyfunction]
#[pyo3(signature = (height, width, patch_size, stride, edge_strategy = "pad"))]
fn grid_sample_anchors(
    height: usize,
    width: usize,
    patch_size: usize,
    stride: usize,
    edge_strategy: &str,
) -> PyResult<Vec<(usize, usize)>> {
    sampler::grid_anchors(height, width, patch_size, stride, edge_strategy)
        .map_err(pyo3::exceptions::PyValueError::new_err)
}

/// Generate random patch anchor positions using a seeded PRNG.
///
/// Returns `count` `(row, col)` top-left corners. `edge_strategy` is `"pad"`
/// (default), `"drop"`, or `"shift"`.
#[pyfunction]
#[pyo3(signature = (height, width, patch_size, count, seed = 42, edge_strategy = "pad"))]
fn random_sample_anchors(
    height: usize,
    width: usize,
    patch_size: usize,
    count: usize,
    seed: u64,
    edge_strategy: &str,
) -> PyResult<Vec<(usize, usize)>> {
    sampler::random_anchors(height, width, patch_size, count, seed, edge_strategy)
        .map_err(pyo3::exceptions::PyValueError::new_err)
}

/// Burn `(polygon, class_id)` pairs into a uint8 mask of shape `(height, width)`.
///
/// `polygons` is a list of `(rings, class_id)` pairs, where `rings` is a list
/// of rings (exterior first, then holes). Each ring is a list of `(x, y)`
/// world-coordinate vertices; rings are auto-closed if not already.
///
/// `transform` is a 6-tuple `(a, b, c, d, e, f)` mapping pixel `(col, row)` to
/// world `(x, y)` (rasterio Affine convention).
///
/// Polygons are written in order; later polygons overwrite earlier ones
/// (replace / last-writer-wins). Background pixels are 0.
#[allow(
    clippy::needless_pass_by_value,
    clippy::type_complexity,
    clippy::many_single_char_names
)]
#[pyfunction]
#[pyo3(signature = (polygons, height, width, transform, all_touched=false))]
fn rasterize(
    py: Python,
    polygons: Vec<(Vec<Vec<(f64, f64)>>, u8)>,
    height: usize,
    width: usize,
    transform: (f64, f64, f64, f64, f64, f64),
    all_touched: bool,
) -> PyResult<Py<PyArray2<u8>>> {
    if width == 0 || height == 0 {
        return Err(pyo3::exceptions::PyValueError::new_err(
            "width and height must be > 0",
        ));
    }
    let (a, b, c, d, e, f) = transform;
    let aff = rasterizer::Affine { a, b, c, d, e, f };
    let buf = rasterizer::rasterize(&polygons, width, height, aff, all_touched)
        .map_err(pyo3::exceptions::PyValueError::new_err)?;
    let arr = numpy::ndarray::Array2::from_shape_vec((height, width), buf)
        .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))?;
    Ok(arr.to_pyarray_bound(py).unbind())
}

/// Write image and mask patches to disk in parallel using rayon.
///
/// `image_patches` is a `(N, ps, ps, 3)` uint8 array.
/// `mask_patches`  is an optional `(N, ps, ps)` uint8 array.
/// `meta`          is a list of `(row, col, padded)` tuples.
///
/// Returns a list of `(filename, mask_filename, row, col, padded, strip_index,
/// class_counts, empty_ratio)` in patch order.
#[allow(
    clippy::needless_pass_by_value,
    clippy::too_many_arguments,
    clippy::type_complexity
)]
#[pyfunction]
#[pyo3(signature = (image_patches, mask_patches, meta, start_idx, strip_index, images_dir, masks_dir, image_format="png", jpg_quality=95))]
fn write_patches_rs(
    py: Python,
    image_patches: PyReadonlyArray4<u8>,
    mask_patches: Option<PyReadonlyArray3<u8>>,
    meta: Vec<(usize, usize, bool)>,
    start_idx: usize,
    strip_index: usize,
    images_dir: String,
    masks_dir: String,
    image_format: &str,
    jpg_quality: u8,
) -> PyResult<
    Vec<(
        String,
        Option<String>,
        usize,
        usize,
        bool,
        usize,
        HashMap<String, u64>,
        f64,
    )>,
> {
    let img_data: Vec<u8> = image_patches
        .as_slice()
        .map_err(|e| pyo3::exceptions::PyValueError::new_err(e.to_string()))?
        .to_vec();
    let shape = image_patches.shape();
    let (n_patches, patch_size) = (shape[0], shape[1]);

    let (msk_data, has_mask): (Vec<u8>, bool) = match mask_patches {
        Some(ref m) => (
            m.as_slice()
                .map_err(|e| pyo3::exceptions::PyValueError::new_err(e.to_string()))?
                .to_vec(),
            true,
        ),
        None => (Vec::new(), false),
    };

    let images_path = std::path::PathBuf::from(images_dir);
    let masks_path = std::path::PathBuf::from(masks_dir);
    let fmt = image_format.to_owned();

    let results = py
        .allow_threads(|| {
            patch_writer::write_patches(
                &img_data,
                &msk_data,
                has_mask,
                n_patches,
                patch_size,
                &meta,
                start_idx,
                strip_index,
                &images_path,
                &masks_path,
                &fmt,
                jpg_quality,
            )
        })
        .map_err(pyo3::exceptions::PyRuntimeError::new_err)?;

    Ok(results
        .into_iter()
        .map(|r| {
            (
                r.filename,
                r.mask_filename,
                r.row,
                r.col,
                r.padded,
                r.strip_index,
                r.class_counts,
                r.empty_ratio,
            )
        })
        .collect())
}

/// Decode and stitch satellite tile bytes into a single `(H, W, 3)` RGB array.
///
/// Accepts a list of `(PyTileIndex, bytes)` pairs as returned by `fetch_tiles`.
/// Returns `(image_array, min_tile_x, min_tile_y)`.
#[allow(clippy::needless_pass_by_value)]
#[pyfunction]
fn stitch_tiles(
    py: Python,
    tile_data: Vec<(PyTileIndex, Vec<u8>)>,
) -> PyResult<(Py<PyArray3<u8>>, u32, u32)> {
    let raw: Vec<(u32, u32, u8, Vec<u8>)> = tile_data
        .into_iter()
        .map(|(t, bytes)| (t.x, t.y, t.z, bytes))
        .collect();
    let (canvas, min_x, min_y, h, w) =
        stitcher::stitch_tiles(&raw).map_err(pyo3::exceptions::PyRuntimeError::new_err)?;
    let arr = numpy::ndarray::Array3::from_shape_vec((h, w, 3), canvas)
        .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))?;
    Ok((arr.into_pyarray_bound(py).unbind(), min_x, min_y))
}

/// Compute the affine transform for a stitched tile grid.
///
/// Returns `(a, b, c, d, e, f)` mapping pixel `(col, row)` to Mercator `(x, y)` in metres.
#[must_use]
#[pyfunction]
fn tile_transform(min_x: u32, min_y: u32, zoom: u8) -> (f64, f64, f64, f64, f64, f64) {
    stitcher::tile_transform(min_x, min_y, zoom)
}

/// Parse KML bytes and return polygon geometries with class labels.
///
/// `label_field` is the `<Data name="...">` field to use for class IDs.
/// When `None` every polygon gets class 1.
///
/// Returns `(polygons, class_map)` where `polygons` is a list of
/// `(rings, class_id)` pairs (exterior ring first, then holes) and
/// `class_map` maps class names to integer IDs.
///
/// # Errors
/// Raises `RuntimeError` if the KML is malformed or coordinate parsing fails.
#[allow(clippy::type_complexity)]
#[pyfunction]
#[pyo3(signature = (data, label_field=None))]
fn parse_kml_rs(
    data: &[u8],
    label_field: Option<&str>,
) -> PyResult<(Vec<(Vec<Vec<Vec<(f64, f64)>>>, u8)>, HashMap<String, u8>)> {
    let result = kml_parser::parse_kml(data, label_field)
        .map_err(pyo3::exceptions::PyRuntimeError::new_err)?;
    Ok((result.polygons, result.class_map))
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
    m.add_function(wrap_pyfunction!(rasterize, m)?)?;
    m.add_function(wrap_pyfunction!(grid_sample_anchors, m)?)?;
    m.add_function(wrap_pyfunction!(random_sample_anchors, m)?)?;
    m.add_function(wrap_pyfunction!(stitch_tiles, m)?)?;
    m.add_function(wrap_pyfunction!(tile_transform, m)?)?;
    m.add_function(wrap_pyfunction!(write_patches_rs, m)?)?;
    m.add_function(wrap_pyfunction!(parse_kml_rs, m)?)?;
    m.add_class::<PyTileIndex>()?;
    m.add_class::<PyBBox>()?;
    Ok(())
}
