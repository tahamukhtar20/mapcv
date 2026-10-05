//! `MapCV` Rust Core
//!
//! This module provides the Rust core for `mapcv`, including tile math
//! utilities and Python bindings for those operations.
// PyO3's #[pyfunction] macro generates `PyErr`-to-`PyErr` coercions that
// clippy::useless_conversion flags. This cannot be suppressed at a narrower
// scope because the lint fires inside the macro expansion.
#![allow(clippy::useless_conversion)]

pub mod fetcher;
pub mod geotiff;
pub mod geotiff_writer;
pub mod kml_parser;
pub mod patch_writer;
pub mod rasterizer;
pub mod sampler;
pub mod stitcher;
pub mod tile_decoder;
pub mod tile_math;

use numpy::ndarray::{Dimension, Ix3, Ix4};
use numpy::{
    IntoPyArray, PyArray, PyArray2, PyArray3, PyArrayMethods, PyReadonlyArray, PyUntypedArray,
    PyUntypedArrayMethods, ToPyArray,
};
use pyo3::exceptions::{PyRuntimeError, PyTypeError, PyValueError};
use pyo3::prelude::*;
use tile_math::{BBox, TileIndex};

/// A fetched tile and its encoded image bytes, as returned to Python.
type FetchedTile = (PyTileIndex, Py<PyAny>);
/// Failed-tile counts per cause and one example message.
type FailureCauses = (Vec<(String, usize)>, Option<String>);

/// Format an array shape the way numpy prints it, e.g. `(2, 4, 4, 3)`.
fn fmt_shape(shape: &[usize]) -> String {
    if let [single] = shape {
        return format!("({single},)");
    }
    let dims: Vec<String> = shape.iter().map(ToString::to_string).collect();
    format!("({})", dims.join(", "))
}

/// Borrow *obj* as a C-contiguous `uint8` array with `D` dimensions.
///
/// Raises `TypeError` when *obj* is not a numpy array and `ValueError` when its
/// dtype, number of dimensions or memory layout is wrong, naming the argument
/// and the expected `layout` so the caller can fix it.
fn u8_array<'py, D: Dimension>(
    obj: &Bound<'py, PyAny>,
    name: &str,
    layout: &str,
) -> PyResult<PyReadonlyArray<'py, u8, D>> {
    let untyped = obj.cast::<PyUntypedArray>().map_err(|_| {
        let type_name = obj
            .get_type()
            .name()
            .map_or_else(|_| "an unknown type".to_owned(), |n| n.to_string());
        PyTypeError::new_err(format!(
            "{name} must be a numpy array shaped {layout}, got {type_name}"
        ))
    })?;
    let array = untyped.cast::<PyArray<u8, D>>().map_err(|_| {
        PyValueError::new_err(format!(
            "{name} must be a uint8 array shaped {layout}, got a {} array with shape {}",
            untyped.dtype(),
            fmt_shape(untyped.shape())
        ))
    })?;
    // A Fortran-ordered or strided view would be read in the wrong pixel order.
    if !array.is_c_contiguous() {
        return Err(PyValueError::new_err(format!(
            "{name} must be C-contiguous; pass numpy.ascontiguousarray({name})"
        )));
    }
    array
        .try_readonly()
        .map_err(|e| PyValueError::new_err(format!("{name} cannot be read: {e}")))
}

/// Python-visible XYZ tile index.
#[pyclass(from_py_object)]
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
#[pyclass(from_py_object)]
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
///
/// `west > east` is read as a box crossing the antimeridian (as in mercantile).
/// A point or line covers the tiles containing it.
///
/// # Errors
/// Raises `ValueError` for a NaN coordinate, for `south > north`, and when the
/// box would cover more than 2^24 tiles.
#[pyfunction]
#[allow(clippy::needless_pass_by_value)]
fn tiles(
    west: f64,
    south: f64,
    east: f64,
    north: f64,
    zooms: Vec<u8>,
) -> PyResult<Vec<PyTileIndex>> {
    let result =
        tile_math::tiles(west, south, east, north, &zooms).map_err(PyValueError::new_err)?;
    Ok(result.into_iter().map(Into::into).collect())
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
///
/// A point or line snaps to the tiles containing it.
///
/// # Errors
/// Raises `ValueError` for a NaN coordinate, for `south > north`, and for
/// `west > east` (a box crossing the antimeridian, which must be split).
#[pyfunction]
fn snap_bbox(west: f64, south: f64, east: f64, north: f64, zoom: u8) -> PyResult<PyBBox> {
    tile_math::snap_bbox(west, south, east, north, zoom)
        .map(Into::into)
        .map_err(PyValueError::new_err)
}

/// Fetch satellite tiles concurrently from a URL template.
///
/// Returns `(tiles, failed, (causes, example))`: `(PyTileIndex, bytes)` pairs for
/// successfully fetched tiles (and, with the `ignore` policy, black `NoData`-filled
/// tiles), the number of failed tiles, `(cause, count)` pairs and one example message.
/// Under the `lenient` policy, raises `RuntimeError` if the fraction of
/// failed tiles exceeds `max_failed_ratio`; `ignore` never enforces it.
#[allow(clippy::needless_pass_by_value, clippy::cast_precision_loss)]
#[pyfunction]
#[pyo3(signature = (tiles, url_template, callback=None, max_connections=16, policy="lenient", max_failed_ratio=0.05))]
fn fetch_tiles(
    py: Python,
    tiles: Vec<PyTileIndex>,
    url_template: String,
    callback: Option<Py<PyAny>>,
    max_connections: usize,
    policy: &str,
    max_failed_ratio: f64,
) -> PyResult<(Vec<FetchedTile>, usize, FailureCauses)> {
    // NaN would make the threshold comparison below always false.
    if !(0.0..=1.0).contains(&max_failed_ratio) {
        return Err(PyValueError::new_err(format!(
            "max_failed_ratio must be between 0 and 1, got {max_failed_ratio}"
        )));
    }
    let rust_tiles: Vec<TileIndex> = tiles
        .into_iter()
        .map(|t| TileIndex {
            x: t.x,
            y: t.y,
            z: t.z,
        })
        .collect();

    let total = rust_tiles.len();

    let (results, failed, failures) = fetcher::fetch_tiles(
        py,
        rust_tiles,
        url_template,
        callback,
        max_connections,
        policy,
    )?;

    let lenient = policy.eq_ignore_ascii_case("lenient");
    if lenient && total > 0 && failed as f64 / total as f64 > max_failed_ratio {
        return Err(PyRuntimeError::new_err(format!(
            "Too many failed tiles: {failed}/{total} ({:.1}% exceeds {:.1}% threshold): {}. \
             If the provider is busy or rate-limiting, try again later or lower \
             imagery.max_connections; raise imagery.max_failed_ratio to accept gaps.",
            100.0 * failed as f64 / total as f64,
            100.0 * max_failed_ratio,
            failures.describe(),
        )));
    }

    let results_py = results
        .into_iter()
        .map(|(t, bytes)| {
            (
                PyTileIndex::from(t),
                pyo3::types::PyBytes::new(py, &bytes).into_any().unbind(),
            )
        })
        .collect();

    Ok((results_py, failed, (failures.counts(), failures.example())))
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
        .map_err(PyValueError::new_err)
}

/// Generate random patch anchor positions using a seeded PRNG.
///
/// Returns up to `count` distinct `(row, col)` top-left corners, drawn
/// uniformly without replacement from the positions whose patch fits inside
/// the raster. When `count` exceeds the number of such positions (see
/// `random_anchor_capacity`) all of them are returned. `edge_strategy` is
/// `"pad"` (default), `"drop"`, or `"shift"`; it only matters for a raster
/// smaller than the patch, where `"drop"` returns nothing.
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
        .map_err(PyValueError::new_err)
}

/// Number of distinct anchors `random_sample_anchors` can return.
#[pyfunction]
#[pyo3(signature = (height, width, patch_size, edge_strategy = "pad"))]
fn random_anchor_capacity(
    height: usize,
    width: usize,
    patch_size: usize,
    edge_strategy: &str,
) -> PyResult<usize> {
    sampler::random_anchor_capacity(height, width, patch_size, edge_strategy)
        .map_err(PyValueError::new_err)
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
        return Err(PyValueError::new_err("width and height must be > 0"));
    }
    let (a, b, c, d, e, f) = transform;
    let aff = rasterizer::Affine { a, b, c, d, e, f };
    let buf = rasterizer::rasterize(&polygons, width, height, aff, all_touched)
        .map_err(PyValueError::new_err)?;
    let arr = numpy::ndarray::Array2::from_shape_vec((height, width), buf)
        .map_err(|e| PyRuntimeError::new_err(e.to_string()))?;
    Ok(arr.to_pyarray(py).unbind())
}

/// Write image and mask patches to disk in parallel using rayon.
///
/// `image_patches` is a C-contiguous `(N, ps, ps, 3)` uint8 array.
/// `mask_patches`  is an optional C-contiguous `(N, ps, ps)` uint8 array.
/// `meta`          is a list of `(row, col, padded)` tuples, one per patch.
/// `image_format`  is `"png"` or `"jpg"`; `jpg_quality` is in `1..=100`;
/// `jpg_subsampling` is `"4:2:0"` or `"4:4:4"` (JPEG only).
///
/// Returns a list of `(filename, mask_filename, row, col, padded, strip_index,
/// class_counts, empty_ratio)` in patch order. `class_counts` is a list of
/// `(class_id, pixel_count)` pairs in ascending class-id order.
///
/// # Errors
/// Raises `ValueError` when the arrays, `meta` or the format arguments do not
/// describe the same `N` square RGB patches, and `RuntimeError` when a patch
/// cannot be encoded or written.
#[allow(
    clippy::needless_pass_by_value,
    clippy::too_many_arguments,
    clippy::type_complexity
)]
#[pyfunction]
#[pyo3(signature = (image_patches, mask_patches, meta, start_idx, strip_index, images_dir, masks_dir, image_format="png", jpg_quality=95, jpg_subsampling="4:2:0"))]
fn write_patches_rs<'py>(
    py: Python<'py>,
    image_patches: &Bound<'py, PyAny>,
    mask_patches: Option<&Bound<'py, PyAny>>,
    meta: Vec<(usize, usize, bool)>,
    start_idx: usize,
    strip_index: usize,
    images_dir: String,
    masks_dir: String,
    image_format: &str,
    jpg_quality: u8,
    jpg_subsampling: &str,
) -> PyResult<
    Vec<(
        String,
        Option<String>,
        usize,
        usize,
        bool,
        usize,
        Vec<(u8, u64)>,
        f64,
    )>,
> {
    let images = u8_array::<Ix4>(image_patches, "image_patches", "(N, ps, ps, 3)")?;
    let shape = images.shape();
    let (n_patches, patch_size) = (shape[0], shape[1]);
    if shape[1] != shape[2] || shape[3] != 3 {
        return Err(PyValueError::new_err(format!(
            "image_patches must hold square RGB patches shaped (N, ps, ps, 3), got shape {}",
            fmt_shape(shape)
        )));
    }
    if patch_size == 0 {
        return Err(PyValueError::new_err(format!(
            "image_patches must have a patch size > 0, got shape {}",
            fmt_shape(shape)
        )));
    }
    if meta.len() != n_patches {
        return Err(PyValueError::new_err(format!(
            "meta has {} entries but image_patches holds {n_patches} patches",
            meta.len()
        )));
    }
    let masks = mask_patches
        .map(|m| u8_array::<Ix3>(m, "mask_patches", "(N, ps, ps)"))
        .transpose()?;
    if let Some(ref m) = masks {
        let expected = [n_patches, patch_size, patch_size];
        if m.shape() != expected {
            return Err(PyValueError::new_err(format!(
                "mask_patches must be shaped {} to match image_patches, got shape {}",
                fmt_shape(&expected),
                fmt_shape(m.shape())
            )));
        }
    }
    patch_writer::check_format(image_format, jpg_quality, jpg_subsampling)
        .map_err(PyValueError::new_err)?;
    if start_idx.checked_add(n_patches).is_none() {
        return Err(PyValueError::new_err(format!(
            "start_idx {start_idx} + {n_patches} patches overflows the patch index"
        )));
    }

    let img_data: Vec<u8> = images
        .as_slice()
        .map_err(|e| PyValueError::new_err(format!("image_patches cannot be read: {e}")))?
        .to_vec();
    let (msk_data, has_mask): (Vec<u8>, bool) = match masks {
        Some(ref m) => (
            m.as_slice()
                .map_err(|e| PyValueError::new_err(format!("mask_patches cannot be read: {e}")))?
                .to_vec(),
            true,
        ),
        None => (Vec::new(), false),
    };

    let images_path = std::path::PathBuf::from(images_dir);
    let masks_path = std::path::PathBuf::from(masks_dir);
    let fmt = image_format.to_owned();
    let subsampling = jpg_subsampling.to_owned();

    let results = py
        .detach(|| {
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
                &subsampling,
            )
        })
        .map_err(PyRuntimeError::new_err)?;

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

/// Write equally shaped rasters as GeoTIFF files, in parallel, with the GIL released.
///
/// `data` is the rasters' samples back to back as raw bytes (a C-contiguous
/// numpy array viewed as `uint8`), `n` rasters of `(height, width, bands)`
/// samples of `dtype` in native byte order; `transforms[i]` and `names[i]` are
/// the georeferencing and file name (inside `directory`) of raster `i`.
/// Existing files are overwritten.
#[allow(clippy::too_many_arguments, clippy::needless_pass_by_value)]
#[pyfunction]
#[pyo3(signature = (data, dtype, shape, transforms, names, directory, epsg, geographic, nodata=None, band_names=None, level=None))]
fn write_geotiffs_rs<'py>(
    py: Python<'py>,
    data: &Bound<'py, PyAny>,
    dtype: &str,
    shape: (usize, usize, usize, usize),
    transforms: Vec<[f64; 6]>,
    names: Vec<String>,
    directory: String,
    epsg: u32,
    geographic: bool,
    nodata: Option<f64>,
    band_names: Option<Vec<String>>,
    level: Option<u32>,
) -> PyResult<()> {
    let bytes = u8_array::<numpy::ndarray::Ix1>(data, "data", "(n_bytes,)")?;
    let (n, height, width, bands) = shape;
    let fmt = geotiff_writer::RasterFormat {
        width,
        height,
        bands,
        dtype: geotiff_writer::dtype_from_name(dtype).map_err(PyValueError::new_err)?,
    };
    if transforms.len() != n || names.len() != n {
        return Err(PyValueError::new_err(format!(
            "{n} rasters need {n} transforms and file names, got {} and {}",
            transforms.len(),
            names.len()
        )));
    }
    let band_names = band_names.unwrap_or_default();
    let options = geotiff_writer::Options {
        nodata,
        band_names: &band_names,
        level,
    };
    let slice = bytes
        .as_slice()
        .map_err(|e| PyValueError::new_err(format!("data cannot be read: {e}")))?;
    let dir = std::path::PathBuf::from(directory);
    let files: Vec<(String, geotiff_writer::Georef)> = names
        .into_iter()
        .zip(transforms)
        .map(|(name, transform)| {
            (
                name,
                geotiff_writer::Georef {
                    epsg,
                    geographic,
                    transform,
                },
            )
        })
        .collect();
    py.detach(|| geotiff_writer::write_all(slice, &fmt, &options, &dir, &files))
        .map_err(PyRuntimeError::new_err)
}

/// Decode and stitch satellite tile bytes into a single `(H, W, 3)` RGB array.
///
/// Tiles from both sides of the antimeridian raise `ValueError`; stitch each
/// side separately.
///
/// Accepts a list of `(PyTileIndex, bytes)` pairs as returned by `fetch_tiles`.
/// Returns `(image_array, min_tile_x, min_tile_y)`.
///
/// # Errors
/// Raises `ValueError` when tiles mix zoom levels or a tile does not decode to
/// 256x256 pixels, and `RuntimeError` when a tile cannot be decoded or the
/// canvas would be too large.
#[allow(clippy::needless_pass_by_value)]
#[pyfunction]
fn stitch_tiles(
    py: Python,
    tile_data: Vec<(PyTileIndex, Vec<u8>)>,
) -> PyResult<(Py<PyArray3<u8>>, u32, u32)> {
    let indices: Vec<TileIndex> = tile_data
        .iter()
        .map(|(t, _)| TileIndex {
            x: t.x,
            y: t.y,
            z: t.z,
        })
        .collect();
    tile_math::check_no_antimeridian_wrap(&indices)
        .map_err(|e| pyo3::exceptions::PyValueError::new_err(format!("stitch_tiles: {e}")))?;
    let raw: Vec<(u32, u32, u8, Vec<u8>)> = tile_data
        .into_iter()
        .map(|(t, bytes)| (t.x, t.y, t.z, bytes))
        .collect();
    let (canvas, min_x, min_y, h, w) = stitcher::stitch_tiles(&raw).map_err(|e| match e {
        stitcher::StitchError::InvalidInput(msg) => PyValueError::new_err(msg),
        stitcher::StitchError::Failed(msg) => PyRuntimeError::new_err(msg),
    })?;
    let arr = numpy::ndarray::Array3::from_shape_vec((h, w, 3), canvas)
        .map_err(|e| PyRuntimeError::new_err(e.to_string()))?;
    Ok((arr.into_pyarray(py).unbind(), min_x, min_y))
}

/// Compute the affine transform for a stitched tile grid.
///
/// Returns `(a, b, c, d, e, f)` mapping pixel `(col, row)` to Mercator `(x, y)` in metres.
#[must_use]
#[pyfunction]
fn tile_transform(min_x: u32, min_y: u32, zoom: u8) -> (f64, f64, f64, f64, f64, f64) {
    stitcher::tile_transform(min_x, min_y, zoom)
}

/// Parse KML bytes into polygon groups with their raw label values.
///
/// `label_field` names the `<Data>` or `<SimpleData>` field to read; when
/// `None`, every label is `None`. Class IDs are assigned by the Python caller.
///
/// Returns `(polygons, skipped_non_polygon)` where `polygons` is a list of
/// `(polygon_group, label)` pairs; each polygon is a list of rings, exterior
/// ring first.
///
/// # Errors
/// Raises `ValueError` if the KML is malformed or truncated, or a coordinate
/// is not a finite number.
#[allow(clippy::type_complexity)]
#[pyfunction]
#[pyo3(signature = (data, label_field=None))]
fn parse_kml_rs(
    data: &[u8],
    label_field: Option<&str>,
) -> PyResult<(Vec<(Vec<kml_parser::Polygon>, Option<String>)>, usize)> {
    let result = kml_parser::parse_kml(data, label_field)
        .map_err(|err| PyValueError::new_err(format!("invalid KML: {err}")))?;
    Ok((result.polygons, result.skipped_non_polygon))
}

/// Map a GeoTIFF reader error to `ValueError` (bad input, corrupt or
/// unsupported file) or `RuntimeError` (file system or network failure).
fn geotiff_error(error: geotiff::GeoTiffError) -> PyErr {
    match error {
        geotiff::GeoTiffError::Invalid(m) => PyValueError::new_err(m),
        geotiff::GeoTiffError::Io(m) => PyRuntimeError::new_err(m),
    }
}

/// Name of a TIFF photometric interpretation code.
fn photometric_name(code: u16) -> String {
    match code {
        0 => "miniswhite".to_owned(),
        1 => "minisblack".to_owned(),
        2 => "rgb".to_owned(),
        3 => "palette".to_owned(),
        4 => "mask".to_owned(),
        5 => "cmyk".to_owned(),
        6 => "ycbcr".to_owned(),
        8 => "cielab".to_owned(),
        other => format!("unknown ({other})"),
    }
}

/// A GeoTIFF or Cloud Optimized GeoTIFF opened without GDAL.
///
/// `GeoTiff(path, cache_bytes=64 MiB)` opens a local path, an `http(s)://` URL
/// or a public `s3://bucket/key` (read with HTTP range requests; anonymous
/// access only). `cache_bytes` bounds the block cache of a remote file.
#[pyclass(name = "GeoTiff", module = "mapcv._mapcv_rs", frozen)]
struct PyGeoTiff {
    inner: geotiff::GeoTiff,
}

#[pymethods]
impl PyGeoTiff {
    #[new]
    #[pyo3(signature = (path, cache_bytes = geotiff::DEFAULT_CACHE_BYTES))]
    fn new(py: Python<'_>, path: String, cache_bytes: usize) -> PyResult<Self> {
        py.detach(move || geotiff::GeoTiff::open(&path, cache_bytes))
            .map(|inner| PyGeoTiff { inner })
            .map_err(geotiff_error)
    }

    /// Structure and georeferencing as a dict.
    ///
    /// Keys: `width`, `height`, `count`, `dtype` (numpy name), `epsg` (int or
    /// None), `crs_error` (why `epsg` is None), `crs_citation`, `transform`
    /// (`(a, b, c, d, e, f)` or None), `raster_type` (`"area"`/`"point"`),
    /// `nodata`, `tiled`, `block_size` (`(rows, cols)`), `overviews` (list of
    /// `(height, width)`, largest first), `compression`, `predictor`,
    /// `planar`, `photometric`, `byte_order`, `bigtiff`.
    fn metadata<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, pyo3::types::PyDict>> {
        let g = &self.inner;
        let d = pyo3::types::PyDict::new(py);
        d.set_item("width", g.width())?;
        d.set_item("height", g.height())?;
        d.set_item("count", g.band_count())?;
        d.set_item("dtype", g.dtype().name())?;
        match g.epsg() {
            Ok(code) => {
                d.set_item("epsg", code)?;
                d.set_item("crs_error", py.None())?;
            }
            Err(reason) => {
                d.set_item("epsg", py.None())?;
                d.set_item("crs_error", reason)?;
            }
        }
        d.set_item("crs_citation", g.crs_citation())?;
        d.set_item(
            "transform",
            g.transform().map(|[a, b, c, dd, e, f]| (a, b, c, dd, e, f)),
        )?;
        d.set_item(
            "raster_type",
            match g.raster_type() {
                geotiff::georef::RasterType::Area => "area",
                geotiff::georef::RasterType::Point => "point",
            },
        )?;
        d.set_item("nodata", g.nodata())?;
        d.set_item("tiled", g.tiled())?;
        let levels = g.levels();
        d.set_item(
            "block_size",
            (levels[0].block_height, levels[0].block_width),
        )?;
        let overviews: Vec<(usize, usize)> =
            levels[1..].iter().map(|l| (l.height, l.width)).collect();
        d.set_item("overviews", overviews)?;
        d.set_item("compression", g.compression())?;
        d.set_item("predictor", g.predictor())?;
        d.set_item("planar", g.planar())?;
        d.set_item("photometric", photometric_name(g.photometric()))?;
        d.set_item(
            "byte_order",
            if g.little_endian() { "little" } else { "big" },
        )?;
        d.set_item("bigtiff", g.bigtiff())?;
        Ok(d)
    }

    /// Read rows `row0..row1` and columns `col0..col1` (half-open, in the
    /// pixel grid of `overview`; 0 = full resolution, 1 = largest overview).
    ///
    /// `bands` are 0-based band indices (all bands when None). Returns
    /// `(data, valid)`: `data` is `(rows, cols, bands)` in the file's dtype and
    /// `valid` a `(rows, cols)` bool array, False where the window extends past
    /// the raster (those pixels hold nodata, or 0 without one). Only the tiles
    /// or strips under the window are read; the GIL is released meanwhile.
    #[pyo3(signature = (row0, row1, col0, col1, bands = None, overview = 0))]
    #[allow(clippy::too_many_arguments, clippy::needless_pass_by_value)]
    fn read_window<'py>(
        &self,
        py: Python<'py>,
        row0: i64,
        row1: i64,
        col0: i64,
        col1: i64,
        bands: Option<Vec<usize>>,
        overview: usize,
    ) -> PyResult<(Bound<'py, PyAny>, Bound<'py, PyArray2<bool>>)> {
        let window = geotiff::Window {
            row0,
            row1,
            col0,
            col1,
        };
        let data = py
            .detach(|| self.inner.read_window(window, bands.as_deref(), overview))
            .map_err(geotiff_error)?;
        let shape = (data.height, data.width, data.bands);
        let shape_error = |e: numpy::ndarray::ShapeError| PyRuntimeError::new_err(e.to_string());
        macro_rules! to_numpy {
            ($v:expr) => {
                numpy::ndarray::Array3::from_shape_vec(shape, $v)
                    .map_err(shape_error)?
                    .into_pyarray(py)
                    .into_any()
            };
        }
        let array = match data.samples {
            geotiff::Samples::U8(v) => to_numpy!(v),
            geotiff::Samples::I8(v) => to_numpy!(v),
            geotiff::Samples::U16(v) => to_numpy!(v),
            geotiff::Samples::I16(v) => to_numpy!(v),
            geotiff::Samples::U32(v) => to_numpy!(v),
            geotiff::Samples::I32(v) => to_numpy!(v),
            geotiff::Samples::U64(v) => to_numpy!(v),
            geotiff::Samples::I64(v) => to_numpy!(v),
            geotiff::Samples::F32(v) => to_numpy!(v),
            geotiff::Samples::F64(v) => to_numpy!(v),
        };
        let valid = numpy::ndarray::Array2::from_shape_vec((data.height, data.width), data.valid)
            .map_err(shape_error)?
            .into_pyarray(py);
        Ok((array, valid))
    }
}

#[pymodule]
fn _mapcv_rs(m: &Bound<'_, PyModule>) -> PyResult<()> {
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
    m.add_function(wrap_pyfunction!(random_anchor_capacity, m)?)?;
    m.add_function(wrap_pyfunction!(stitch_tiles, m)?)?;
    m.add_function(wrap_pyfunction!(tile_transform, m)?)?;
    m.add_function(wrap_pyfunction!(tile_decoder::decode_tile_window, m)?)?;
    m.add_function(wrap_pyfunction!(write_patches_rs, m)?)?;
    m.add_function(wrap_pyfunction!(write_geotiffs_rs, m)?)?;
    m.add_function(wrap_pyfunction!(parse_kml_rs, m)?)?;
    m.add_class::<PyTileIndex>()?;
    m.add_class::<PyBBox>()?;
    m.add_class::<PyGeoTiff>()?;
    Ok(())
}
