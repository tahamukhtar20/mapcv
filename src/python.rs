//! Python bindings: the `_mapcv_rs` extension module.
//!
//! Compiled only with the default `python` feature. Everything here is a thin
//! argument-validating wrapper over the pure-Rust modules, which the fuzz targets
//! (`fuzz/`) use directly.
// PyO3's #[pyfunction] macro generates `PyErr`-to-`PyErr` coercions that
// clippy::useless_conversion flags. This cannot be suppressed at a narrower
// scope because the lint fires inside the macro expansion.
#![allow(clippy::useless_conversion)]

use crate::{
    fetcher, geotiff, geotiff_writer, kml_parser, patch_writer, rasterizer, sampler, stitcher,
    tile_decoder, tile_math,
};
use numpy::ndarray::{Dimension, Ix3, Ix4};
use numpy::{
    IntoPyArray, PyArray, PyArray2, PyArray3, PyArrayMethods, PyReadonlyArray, PyUntypedArray,
    PyUntypedArrayMethods,
};
use pyo3::exceptions::{PyRuntimeError, PyTypeError, PyValueError};
use pyo3::prelude::*;
use std::collections::BTreeMap;
use tile_math::{BBox, TileIndex};

/// A fetched tile and its encoded image bytes, as returned to Python; with
/// `cache_headers`, also its `(cache_control, expires, date, age)` header values.
type FetchedTile = Py<PyAny>;
/// `Cache-Control`, `Expires`, `Date` and `Age` of a tile response.
type CacheHeaderValues = (
    Option<String>,
    Option<String>,
    Option<String>,
    Option<String>,
);
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

/// Python-visible XYZ tile index: compared and hashed by value, and picklable.
#[pyclass(
    name = "TileIndex",
    module = "mapcv._mapcv_rs",
    frozen,
    eq,
    hash,
    from_py_object
)]
#[derive(Clone, PartialEq, Eq, Hash)]
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

    fn __repr__(&self) -> String {
        format!("TileIndex(x={}, y={}, z={})", self.x, self.y, self.z)
    }

    /// The constructor arguments, so `pickle` and `copy` rebuild the tile.
    fn __getnewargs__(&self) -> (u32, u32, u8) {
        (self.x, self.y, self.z)
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

/// Python-visible bounding box (WGS-84 degrees, or Web Mercator metres from
/// `xy_bounds`): compared and hashed by value, and picklable.
#[pyclass(
    name = "BBox",
    module = "mapcv._mapcv_rs",
    frozen,
    eq,
    hash,
    from_py_object
)]
#[derive(Clone, PartialEq)]
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

    fn __repr__(&self) -> String {
        format!(
            "BBox(west={:?}, south={:?}, east={:?}, north={:?})",
            self.west, self.south, self.east, self.north
        )
    }

    /// The constructor arguments, so `pickle` and `copy` rebuild the box.
    fn __getnewargs__(&self) -> (f64, f64, f64, f64) {
        (self.west, self.south, self.east, self.north)
    }
}

impl std::hash::Hash for PyBBox {
    /// Hashes the coordinates' bits, with `-0.0` as `0.0` so that equal boxes hash
    /// alike (`NaN` never equals itself, so its hash does not matter).
    fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
        for value in [self.west, self.south, self.east, self.north] {
            let normalised = if value == 0.0 { 0.0_f64 } else { value };
            normalised.to_bits().hash(state);
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
///
/// A latitude of exactly ±90 gives an infinite `y`.
///
/// # Errors
/// Raises `ValueError` for a longitude or latitude that is not a finite number.
#[pyfunction]
fn xy(lng: f64, lat: f64) -> PyResult<(f64, f64)> {
    tile_math::check_finite(lng, lat).map_err(PyValueError::new_err)?;
    Ok(tile_math::xy(lng, lat))
}

/// Return the XYZ tile index for a (lng, lat) point at the given zoom level.
///
/// A position beyond the world (a longitude past ±180, a latitude past ±90 or the Web
/// Mercator limit of about ±85.05) is in the edge column or row.
///
/// # Errors
/// Raises `ValueError` for a longitude or latitude that is not a finite number, and for
/// a zoom above 32.
#[pyfunction]
fn tile(lng: f64, lat: f64, zoom: u8) -> PyResult<PyTileIndex> {
    tile_math::check_finite(lng, lat).map_err(PyValueError::new_err)?;
    tile_math::check_zoom(zoom).map_err(PyValueError::new_err)?;
    Ok(tile_math::tile(lng, lat, zoom).into())
}

/// Return all XYZ tiles covering the given bounding box at the specified zoom levels.
///
/// `west > east` is read as a box crossing the antimeridian (as in mercantile).
/// A point or line covers the tiles containing it.
///
/// # Errors
/// Raises `ValueError` for a NaN coordinate, for `south > north`, for a zoom above
/// 32, and when the box would cover more than 2^24 tiles.
#[pyfunction]
#[allow(clippy::needless_pass_by_value)]
fn tiles(
    py: Python<'_>,
    west: f64,
    south: f64,
    east: f64,
    north: f64,
    zooms: Vec<u8>,
) -> PyResult<Vec<PyTileIndex>> {
    let result = py
        .detach(|| tile_math::tiles(west, south, east, north, &zooms))
        .map_err(PyValueError::new_err)?;
    Ok(result.into_iter().map(Into::into).collect())
}

/// Return the Web Mercator bounding box (EPSG:3857, meters) for an XYZ tile.
///
/// # Errors
/// Raises `ValueError` for a zoom above 32, or a column or row that does not exist
/// at that zoom (2^zoom or more).
#[pyfunction]
fn xy_bounds(x: u32, y: u32, z: u8) -> PyResult<PyBBox> {
    let t = TileIndex { x, y, z };
    tile_math::check_tile(t).map_err(PyValueError::new_err)?;
    Ok(tile_math::xy_bounds(t).into())
}

/// Return the geographic bounding box (EPSG:4326, degrees) for an XYZ tile.
///
/// # Errors
/// Raises `ValueError` for a zoom above 32, or a column or row that does not exist
/// at that zoom (2^zoom or more).
#[pyfunction]
fn bounds(x: u32, y: u32, z: u8) -> PyResult<PyBBox> {
    let t = TileIndex { x, y, z };
    tile_math::check_tile(t).map_err(PyValueError::new_err)?;
    Ok(tile_math::bounds(t).into())
}

/// Expand a bbox outward to the nearest tile boundaries at the given zoom level.
///
/// A point or line snaps to the tiles containing it.
///
/// # Errors
/// Raises `ValueError` for a NaN coordinate, for `south > north`, for a zoom above 32,
/// and for `west > east` (a box crossing the antimeridian, which must be split).
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
/// With `cache_headers`, each tile is `(PyTileIndex, bytes, headers)`, where `headers`
/// is `(cache_control, expires, date, age)` (each `None` when absent) for a tile the
/// server sent and `None` for a black fill.
/// Under the `lenient` policy, raises `RuntimeError` as soon as the fraction of
/// failed tiles exceeds `max_failed_ratio` (the rest are not fetched); `ignore` never
/// enforces it. Under any policy, `RuntimeError` is also raised when the first requests
/// all went unanswered (timeouts, refused connections, rate limiting, server errors).
#[allow(clippy::needless_pass_by_value, clippy::too_many_arguments)]
#[pyfunction]
#[pyo3(signature = (tiles, url_template, callback=None, max_connections=16, policy="lenient", max_failed_ratio=0.05, cache_headers=false))]
fn fetch_tiles(
    py: Python,
    tiles: Vec<PyTileIndex>,
    url_template: String,
    callback: Option<Py<PyAny>>,
    max_connections: usize,
    policy: &str,
    max_failed_ratio: f64,
    cache_headers: bool,
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

    // Only the lenient policy stops a fetch on a share of failed tiles.
    let lenient = policy.eq_ignore_ascii_case("lenient");
    let (results, failed, failures) = fetcher::fetch_tiles(
        py,
        rust_tiles,
        url_template,
        callback,
        max_connections,
        policy,
        lenient.then_some(max_failed_ratio),
    )?;

    let results_py = results
        .into_iter()
        .map(|(t, bytes, headers)| -> PyResult<FetchedTile> {
            let tile = PyTileIndex::from(t);
            let payload = pyo3::types::PyBytes::new(py, &bytes);
            Ok(if cache_headers {
                let values: Option<CacheHeaderValues> =
                    headers.map(|h| (h.cache_control, h.expires, h.date, h.age));
                (tile, payload, values)
                    .into_pyobject(py)?
                    .into_any()
                    .unbind()
            } else {
                (tile, payload).into_pyobject(py)?.into_any().unbind()
            })
        })
        .collect::<PyResult<Vec<_>>>()?;

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
    // The polygons are plain Rust data by now: burn them without holding the GIL.
    let buf = py
        .detach(|| rasterizer::rasterize(&polygons, width, height, aff, all_touched))
        .map_err(PyValueError::new_err)?;
    let arr = numpy::ndarray::Array2::from_shape_vec((height, width), buf)
        .map_err(|e| PyRuntimeError::new_err(e.to_string()))?;
    Ok(arr.into_pyarray(py).unbind())
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
fn write_patches<'py>(
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

    // Borrowed, not copied: the arrays stay alive (and, being read-only borrows,
    // unchanged) while the GIL is released below.
    let img_data: &[u8] = images
        .as_slice()
        .map_err(|e| PyValueError::new_err(format!("image_patches cannot be read: {e}")))?;
    let (msk_data, has_mask): (&[u8], bool) = match masks {
        Some(ref m) => (
            m.as_slice()
                .map_err(|e| PyValueError::new_err(format!("mask_patches cannot be read: {e}")))?,
            true,
        ),
        None => (&[], false),
    };

    let images_path = std::path::PathBuf::from(images_dir);
    let masks_path = std::path::PathBuf::from(masks_dir);
    let fmt = image_format.to_owned();
    let subsampling = jpg_subsampling.to_owned();

    let results = py
        .detach(|| {
            patch_writer::write_patches(
                img_data,
                msk_data,
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
fn write_geotiffs<'py>(
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
fn stitch_tiles<'py>(
    py: Python<'py>,
    tile_data: Vec<(PyTileIndex, Bound<'py, pyo3::types::PyBytes>)>,
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
    // The tiles' bytes are borrowed (``bytes`` is immutable and kept alive by
    // ``tile_data``), not copied element by element into new vectors.
    let raw: Vec<(u32, u32, u8, &[u8])> = tile_data
        .iter()
        .map(|(t, bytes)| (t.x, t.y, t.z, bytes.as_bytes()))
        .collect();
    let (canvas, min_x, min_y, h, w) =
        py.detach(|| stitcher::stitch_tiles(&raw))
            .map_err(|e| match e {
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
///
/// # Errors
/// Raises `ValueError` for a zoom above 32, or a column or row that does not exist
/// at that zoom (2^zoom or more).
#[pyfunction]
fn tile_transform(min_x: u32, min_y: u32, zoom: u8) -> PyResult<(f64, f64, f64, f64, f64, f64)> {
    tile_math::check_tile(TileIndex {
        x: min_x,
        y: min_y,
        z: zoom,
    })
    .map_err(PyValueError::new_err)?;
    Ok(stitcher::tile_transform(min_x, min_y, zoom))
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
fn parse_kml(
    py: Python<'_>,
    data: &[u8],
    label_field: Option<&str>,
) -> PyResult<(Vec<(Vec<kml_parser::Polygon>, Option<String>)>, usize)> {
    let result = py
        .detach(|| kml_parser::parse_kml(data, label_field))
        .map_err(|err| PyValueError::new_err(format!("invalid KML: {err}")))?;
    Ok((result.polygons, result.skipped_non_polygon))
}

/// Every `<Data>`/`<SimpleData>` field of each polygon placemark of a KML file.
///
/// Returns one `{name: value}` dict per entry of `parse_kml(data)[0]`, in the
/// same order, from a single pass over the file: `fields[i].get(name)` is the
/// label `parse_kml(data, name)` gives polygon `i`.
///
/// # Errors
/// As `parse_kml`.
#[pyfunction]
fn kml_fields(py: Python<'_>, data: &[u8]) -> PyResult<Vec<BTreeMap<String, String>>> {
    let result = py
        .detach(|| kml_parser::kml_fields(data))
        .map_err(|err| PyValueError::new_err(format!("invalid KML: {err}")))?;
    Ok(result.fields)
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
    path: String,
}

#[pymethods]
impl PyGeoTiff {
    #[new]
    #[pyo3(signature = (path, cache_bytes = geotiff::DEFAULT_CACHE_BYTES, trust_host = true))]
    fn new(py: Python<'_>, path: String, cache_bytes: usize, trust_host: bool) -> PyResult<Self> {
        py.detach(|| geotiff::GeoTiff::open_with(&path, cache_bytes, trust_host))
            .map(|inner| PyGeoTiff { inner, path })
            .map_err(geotiff_error)
    }

    fn __repr__(&self, py: Python<'_>) -> PyResult<String> {
        let path = pyo3::types::PyString::new(py, &self.path).repr()?;
        Ok(format!("GeoTiff({path})"))
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

/// Make every HTTP request of this process connect to public addresses only (`True`),
/// even one whose URL names this machine, or go back to the default (`False`).
///
/// The MCP server sets it unless it was started with `--allow-local-urls`.
#[pyfunction]
fn set_public_only(on: bool) {
    crate::http_policy::set_public_only(on);
}

#[pymodule]
fn _mapcv_rs(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(set_public_only, m)?)?;
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
    m.add_function(wrap_pyfunction!(write_patches, m)?)?;
    m.add_function(wrap_pyfunction!(write_geotiffs, m)?)?;
    m.add_function(wrap_pyfunction!(parse_kml, m)?)?;
    m.add_function(wrap_pyfunction!(kml_fields, m)?)?;
    m.add_class::<PyTileIndex>()?;
    m.add_class::<PyBBox>()?;
    // The names before mapcv 0.3, for code that imported them.
    m.add("PyTileIndex", m.getattr("TileIndex")?)?;
    m.add("PyBBox", m.getattr("BBox")?)?;
    m.add_class::<PyGeoTiff>()?;
    // The version this binary was built from; `mapcv doctor` compares it with the package's.
    m.add("__version__", env!("CARGO_PKG_VERSION"))?;
    Ok(())
}
