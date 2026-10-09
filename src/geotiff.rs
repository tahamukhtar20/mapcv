//! GDAL-free reader for GeoTIFF and Cloud Optimized GeoTIFF (COG) rasters.
//!
//! [`GeoTiff::open`] reads the TIFF structure (classic or BigTIFF, either byte
//! order), the georeferencing (EPSG code, affine transform, nodata) and the
//! overview levels of a local file or a remote one (`http(s)://`, or
//! `s3://bucket/key` for public buckets, read with HTTP range requests).
//! [`GeoTiff::read_window`] then decodes only the tiles or strips under a
//! pixel window, in parallel, into a `(rows, cols, bands)` array of the file's
//! own data type.
//!
//! The TIFF container and chunk codecs are implemented here rather than with
//! the `tiff` crate, whose decoder (0.11) keeps JPEG chunks in `YCbCr`
//! instead of converting them to RGB as GDAL does, inverts `WhiteIsZero`
//! data, rejects palette images (common for class rasters) and needs `&mut`
//! access per chunk, which rules out decoding the chunks of a window in
//! parallel.

pub mod codec;
pub mod georef;
pub mod ifd;
pub mod source;

use codec::MAX_CHUNK_BYTES;
use georef::{Georef, RasterType};
use ifd::{tag, ByteOrder, Ifd, UintArray};
use rayon::prelude::*;
use source::{ByteSource, HttpSource, LocalFile, MemorySource};
use std::fmt;

/// Error from opening or reading a GeoTIFF.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum GeoTiffError {
    /// Invalid arguments, a corrupt file, or a feature the reader does not support.
    Invalid(String),
    /// Reading the bytes failed (file system or network).
    Io(String),
}

impl fmt::Display for GeoTiffError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            GeoTiffError::Invalid(m) | GeoTiffError::Io(m) => f.write_str(m),
        }
    }
}

impl std::error::Error for GeoTiffError {}

type Result<T> = std::result::Result<T, GeoTiffError>;

/// Largest window returned by one read.
const MAX_WINDOW_BYTES: usize = 8 << 30;
/// Default bound on the block cache of a remote file.
pub const DEFAULT_CACHE_BYTES: usize = 64 << 20;

/// Sample data type of a raster.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum DType {
    /// 8-bit unsigned integer.
    U8,
    /// 8-bit signed integer.
    I8,
    /// 16-bit unsigned integer.
    U16,
    /// 16-bit signed integer.
    I16,
    /// 32-bit unsigned integer.
    U32,
    /// 32-bit signed integer.
    I32,
    /// 64-bit unsigned integer.
    U64,
    /// 64-bit signed integer.
    I64,
    /// 32-bit IEEE float.
    F32,
    /// 64-bit IEEE float.
    F64,
}

impl DType {
    /// Bytes per sample.
    #[must_use]
    pub fn size(self) -> usize {
        match self {
            DType::U8 | DType::I8 => 1,
            DType::U16 | DType::I16 => 2,
            DType::U32 | DType::I32 | DType::F32 => 4,
            DType::U64 | DType::I64 | DType::F64 => 8,
        }
    }

    /// The numpy name of the type, e.g. `"uint16"`.
    #[must_use]
    pub fn name(self) -> &'static str {
        match self {
            DType::U8 => "uint8",
            DType::I8 => "int8",
            DType::U16 => "uint16",
            DType::I16 => "int16",
            DType::U32 => "uint32",
            DType::I32 => "int32",
            DType::U64 => "uint64",
            DType::I64 => "int64",
            DType::F32 => "float32",
            DType::F64 => "float64",
        }
    }

    fn from_tags(format: u64, bits: u64) -> Result<DType> {
        // SampleFormat 4 ("void") is read as unsigned, as GDAL does.
        Ok(match (format, bits) {
            (1 | 4, 8) => DType::U8,
            (2, 8) => DType::I8,
            (1 | 4, 16) => DType::U16,
            (2, 16) => DType::I16,
            (1 | 4, 32) => DType::U32,
            (2, 32) => DType::I32,
            (1 | 4, 64) => DType::U64,
            (2, 64) => DType::I64,
            (3, 32) => DType::F32,
            (3, 64) => DType::F64,
            (3, 16) => {
                return Err(GeoTiffError::Invalid(
                    "16-bit float samples are not supported".to_owned(),
                ))
            }
            (5 | 6, _) => {
                return Err(GeoTiffError::Invalid(
                    "complex-valued samples are not supported".to_owned(),
                ))
            }
            (1..=4, b) => {
                return Err(GeoTiffError::Invalid(format!(
                    "{b}-bit samples are not supported (only 8, 16, 32 and 64 bits are)"
                )))
            }
            (f, _) => {
                return Err(GeoTiffError::Invalid(format!(
                    "unknown TIFF SampleFormat {f}"
                )))
            }
        })
    }
}

/// One resolution level: the full-resolution image or an overview.
#[derive(Debug)]
struct Level {
    width: usize,
    height: usize,
    /// Tile size, or (image width, rows per strip) for strips.
    chunk_width: usize,
    chunk_height: usize,
    tiled: bool,
    compression: u16,
    predictor: u16,
    photometric: u16,
    offsets: UintArray,
    byte_counts: UintArray,
    jpeg_tables: Option<Vec<u8>>,
    samples: usize,
    dtype: DType,
    planar: bool,
}

impl Level {
    // A flat sequence of tag reads and checks.
    #[allow(clippy::too_many_lines)]
    fn from_ifd(ifd: &Ifd) -> Result<Level> {
        let required = |t: u16, name: &str| -> Result<usize> {
            let value = ifd
                .uint(t, None)?
                .ok_or_else(|| GeoTiffError::Invalid(format!("TIFF tag {name} is missing")))?;
            usize::try_from(value)
                .ok()
                .filter(|&v| v > 0 && v <= 1 << 31)
                .ok_or_else(|| {
                    GeoTiffError::Invalid(format!("TIFF tag {name} = {value} is invalid"))
                })
        };
        let width = required(tag::IMAGE_WIDTH, "ImageWidth")?;
        let height = required(tag::IMAGE_LENGTH, "ImageLength")?;
        let samples = usize::try_from(ifd.uint(tag::SAMPLES_PER_PIXEL, Some(1))?.unwrap_or(1))
            .ok()
            .filter(|&s| (1..=65535).contains(&s))
            .ok_or_else(|| GeoTiffError::Invalid("invalid SamplesPerPixel".to_owned()))?;
        let uniform = |t: u16, default: u64, name: &str| -> Result<u64> {
            let Some(values) = ifd.uint_array(t)? else {
                return Ok(default);
            };
            match values.get(0) {
                Some(first) if values.iter().all(|v| v == first) => Ok(first),
                Some(_) => {
                    let shown: Vec<u64> = values.iter().take(8).collect();
                    Err(GeoTiffError::Invalid(format!(
                        "bands with different {name} values ({shown:?}{}) are not supported",
                        if values.len() > 8 { ", ..." } else { "" }
                    )))
                }
                None => Err(GeoTiffError::Invalid(format!("TIFF tag {name} is empty"))),
            }
        };
        let bits = uniform(tag::BITS_PER_SAMPLE, 1, "BitsPerSample")?;
        let format = uniform(tag::SAMPLE_FORMAT, 1, "SampleFormat")?;
        let dtype = DType::from_tags(format, bits)?;
        let compression = u16::try_from(ifd.uint(tag::COMPRESSION, Some(1))?.unwrap_or(1))
            .map_err(|_| GeoTiffError::Invalid("invalid Compression tag".to_owned()))?;
        codec::check_supported(compression, u16::try_from(bits).unwrap_or(u16::MAX))?;
        let predictor = match ifd.uint(tag::PREDICTOR, Some(1))?.unwrap_or(1) {
            1 => 1,
            2 => 2,
            3 if matches!(dtype, DType::F32 | DType::F64) => 3,
            3 => {
                return Err(GeoTiffError::Invalid(
                    "the floating-point predictor (3) is only valid for float samples".to_owned(),
                ))
            }
            p => return Err(GeoTiffError::Invalid(format!("unknown TIFF predictor {p}"))),
        };
        let planar = match ifd.uint(tag::PLANAR_CONFIGURATION, Some(1))?.unwrap_or(1) {
            1 => false,
            2 => samples > 1,
            p => {
                return Err(GeoTiffError::Invalid(format!(
                    "unknown TIFF PlanarConfiguration {p}"
                )))
            }
        };
        let photometric = u16::try_from(ifd.uint(tag::PHOTOMETRIC, Some(1))?.unwrap_or(1))
            .map_err(|_| GeoTiffError::Invalid("invalid PhotometricInterpretation".to_owned()))?;
        if photometric == 6 && compression != codec::compression::JPEG {
            return Err(GeoTiffError::Invalid(format!(
                "YCbCr photometric interpretation with {} compression is not supported (only \
                 with JPEG)",
                codec::compression_name(compression)
            )));
        }
        let tiled = ifd.has(tag::TILE_OFFSETS);
        let (chunk_width, chunk_height, offsets_tag, counts_tag) = if tiled {
            (
                required(tag::TILE_WIDTH, "TileWidth")?,
                required(tag::TILE_LENGTH, "TileLength")?,
                tag::TILE_OFFSETS,
                tag::TILE_BYTE_COUNTS,
            )
        } else {
            let rows = ifd.uint(tag::ROWS_PER_STRIP, None)?.unwrap_or(u64::MAX);
            let rows = usize::try_from(rows).unwrap_or(usize::MAX).clamp(1, height);
            (width, rows, tag::STRIP_OFFSETS, tag::STRIP_BYTE_COUNTS)
        };
        let offsets = ifd.uint_array(offsets_tag)?.ok_or_else(|| {
            GeoTiffError::Invalid("the TIFF has no StripOffsets or TileOffsets tag".to_owned())
        })?;
        let byte_counts = ifd.uint_array(counts_tag)?.ok_or_else(|| {
            GeoTiffError::Invalid(
                "the TIFF has no StripByteCounts or TileByteCounts tag".to_owned(),
            )
        })?;
        let planes = if planar { samples } else { 1 };
        // Dimensions up to 2^31 with 1-pixel chunks overflow a `usize` product.
        let expected = width
            .div_ceil(chunk_width)
            .checked_mul(height.div_ceil(chunk_height))
            .and_then(|n| n.checked_mul(planes))
            .ok_or_else(|| {
                GeoTiffError::Invalid(format!(
                    "the TIFF declares an impossible number of chunks: {width}x{height} pixels \
                     in {chunk_width}x{chunk_height} chunks"
                ))
            })?;
        if offsets.len() != expected || byte_counts.len() != expected {
            return Err(GeoTiffError::Invalid(format!(
                "the TIFF has {} chunk offsets and {} byte counts, expected {expected}",
                offsets.len(),
                byte_counts.len()
            )));
        }
        Ok(Level {
            width,
            height,
            chunk_width,
            chunk_height,
            tiled,
            compression,
            predictor,
            photometric,
            offsets,
            byte_counts,
            jpeg_tables: ifd.bytes(tag::JPEG_TABLES).map(<[u8]>::to_vec),
            samples,
            dtype,
            planar,
        })
    }

    fn chunks_across(&self) -> usize {
        self.width.div_ceil(self.chunk_width)
    }

    fn chunks_down(&self) -> usize {
        self.height.div_ceil(self.chunk_height)
    }

    /// Samples stored per pixel in one chunk.
    fn chunk_samples(&self) -> usize {
        if self.planar {
            1
        } else {
            self.samples
        }
    }

    /// Rows of image data in chunk row `cy` (strips at the bottom may be short).
    fn rows_in_chunk(&self, cy: usize) -> usize {
        if self.tiled {
            self.chunk_height
        } else {
            self.chunk_height.min(self.height - cy * self.chunk_height)
        }
    }
}

/// Size and block layout of one resolution level.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct LevelInfo {
    /// Width in pixels.
    pub width: usize,
    /// Height in pixels.
    pub height: usize,
    /// Block (tile or strip) width in pixels.
    pub block_width: usize,
    /// Block (tile or strip) height in pixels.
    pub block_height: usize,
}

/// A half-open pixel window `[row0, row1) x [col0, col1)`; it may extend past
/// the raster on any side.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Window {
    /// First row.
    pub row0: i64,
    /// One past the last row.
    pub row1: i64,
    /// First column.
    pub col0: i64,
    /// One past the last column.
    pub col1: i64,
}

/// Samples of a window, row-major `(rows, cols, bands)`, in the raster's type.
#[derive(Clone, Debug, PartialEq)]
#[allow(missing_docs)]
pub enum Samples {
    U8(Vec<u8>),
    I8(Vec<i8>),
    U16(Vec<u16>),
    I16(Vec<i16>),
    U32(Vec<u32>),
    I32(Vec<i32>),
    U64(Vec<u64>),
    I64(Vec<i64>),
    F32(Vec<f32>),
    F64(Vec<f64>),
}

/// Fill value for integer types: nodata when it is an integer the type can hold, else 0.
#[allow(clippy::cast_possible_truncation)] // the range check makes the cast exact
fn int_fill<T: TryFrom<i64> + Default>(nodata: Option<f64>) -> T {
    // -2^63 and 2^63, exactly.
    const I64_RANGE: std::ops::Range<f64> =
        -9_223_372_036_854_775_808.0..9_223_372_036_854_775_808.0;
    match nodata {
        Some(v) if v.fract() == 0.0 && I64_RANGE.contains(&v) => {
            T::try_from(v as i64).unwrap_or_default()
        }
        _ => T::default(),
    }
}

impl Samples {
    /// `n` samples of `dtype`, all set to `nodata` (or 0 when it has none or
    /// the type cannot hold it).
    #[allow(clippy::cast_possible_truncation)]
    fn filled(dtype: DType, n: usize, nodata: Option<f64>) -> Samples {
        match dtype {
            DType::U8 => Samples::U8(vec![int_fill(nodata); n]),
            DType::I8 => Samples::I8(vec![int_fill(nodata); n]),
            DType::U16 => Samples::U16(vec![int_fill(nodata); n]),
            DType::I16 => Samples::I16(vec![int_fill(nodata); n]),
            DType::U32 => Samples::U32(vec![int_fill(nodata); n]),
            DType::I32 => Samples::I32(vec![int_fill(nodata); n]),
            DType::U64 => Samples::U64(vec![int_fill(nodata); n]),
            DType::I64 => Samples::I64(vec![int_fill(nodata); n]),
            // f64 -> f32 rounds as GDAL does when it stores nodata in a float32 band.
            DType::F32 => Samples::F32(vec![nodata.map_or(0.0, |v| v as f32); n]),
            DType::F64 => Samples::F64(vec![nodata.unwrap_or(0.0); n]),
        }
    }

    fn as_bytes_mut(&mut self) -> &mut [u8] {
        match self {
            Samples::U8(v) => v.as_mut_slice(),
            Samples::I8(v) => bytemuck::cast_slice_mut(v),
            Samples::U16(v) => bytemuck::cast_slice_mut(v),
            Samples::I16(v) => bytemuck::cast_slice_mut(v),
            Samples::U32(v) => bytemuck::cast_slice_mut(v),
            Samples::I32(v) => bytemuck::cast_slice_mut(v),
            Samples::U64(v) => bytemuck::cast_slice_mut(v),
            Samples::I64(v) => bytemuck::cast_slice_mut(v),
            Samples::F32(v) => bytemuck::cast_slice_mut(v),
            Samples::F64(v) => bytemuck::cast_slice_mut(v),
        }
    }
}

/// The result of [`GeoTiff::read_window`].
#[derive(Clone, Debug, PartialEq)]
pub struct WindowData {
    /// Rows in the window.
    pub height: usize,
    /// Columns in the window.
    pub width: usize,
    /// Bands returned.
    pub bands: usize,
    /// `height * width * bands` samples, row-major `(rows, cols, bands)`.
    pub samples: Samples,
    /// `height * width` flags: `true` where the pixel lies inside the raster.
    /// Pixels outside it hold the nodata value (0 when there is none).
    pub valid: Vec<bool>,
}

/// An open GeoTIFF: structure, georeferencing and a byte source to read from.
pub struct GeoTiff {
    source: Box<dyn ByteSource>,
    order: ByteOrder,
    bigtiff: bool,
    /// Index 0 is full resolution; overviews follow, largest first.
    levels: Vec<Level>,
    georef: Georef,
    nodata: Option<f64>,
}

/// Whether `path` names a URL (`scheme://...`) rather than a local file.
fn is_url(path: &str) -> bool {
    path.split_once("://").is_some_and(|(scheme, _)| {
        scheme.len() > 1
            && scheme
                .chars()
                .all(|c| c.is_ascii_alphanumeric() || "+-.".contains(c))
    })
}

impl GeoTiff {
    /// Open a local path, an `http(s)://` URL or an `s3://bucket/key` URL.
    ///
    /// `cache_bytes` bounds the block cache of a remote file.
    ///
    /// # Errors
    /// Returns [`GeoTiffError::Io`] when the file cannot be read and
    /// [`GeoTiffError::Invalid`] when it is not a TIFF the reader supports.
    pub fn open(path_or_url: &str, cache_bytes: usize) -> Result<GeoTiff> {
        GeoTiff::open_with(path_or_url, cache_bytes, true)
    }

    /// [`GeoTiff::open`]; for a URL, `trust_host` as in [`HttpSource::open_with`].
    ///
    /// # Errors
    /// As [`GeoTiff::open`].
    pub fn open_with(path_or_url: &str, cache_bytes: usize, trust_host: bool) -> Result<GeoTiff> {
        let source: Box<dyn ByteSource> = if is_url(path_or_url) {
            Box::new(HttpSource::open_with(path_or_url, cache_bytes, trust_host)?)
        } else {
            Box::new(LocalFile::open(path_or_url)?)
        };
        GeoTiff::from_source(source)
    }

    /// Read the structure and georeferencing of a TIFF held in memory.
    ///
    /// # Errors
    /// As [`GeoTiff::open`], except that nothing is read from the file system.
    pub fn from_bytes(data: Vec<u8>) -> Result<GeoTiff> {
        GeoTiff::from_source(Box::new(MemorySource::new(data)))
    }

    /// Read the structure and georeferencing of the TIFF in `source`.
    ///
    /// # Errors
    /// As [`GeoTiff::open`].
    pub fn from_source(source: Box<dyn ByteSource>) -> Result<GeoTiff> {
        let name = source.describe();
        let with_name = |e: GeoTiffError| match e {
            GeoTiffError::Invalid(m) => GeoTiffError::Invalid(format!("{name}: {m}")),
            GeoTiffError::Io(m) => GeoTiffError::Io(m),
        };
        let tiff = ifd::read_tiff(source.as_ref())?;
        let main_ifd = &tiff.ifds[0];
        let main = Level::from_ifd(main_ifd).map_err(with_name)?;
        let mut levels = vec![main];
        for (i, ifd) in tiff.ifds.iter().enumerate().skip(1) {
            let subfile = ifd.uint(tag::NEW_SUBFILE_TYPE, Some(0))?.unwrap_or(0);
            // Bit 0: reduced-resolution image. Bit 2: transparency mask
            // (GDAL's internal masks), which is not an overview.
            if subfile & 1 == 0 || subfile & 4 != 0 {
                continue;
            }
            let level = Level::from_ifd(ifd)
                .map_err(|e| with_name(GeoTiffError::Invalid(format!("overview IFD {i}: {e}"))))?;
            let main = &levels[0];
            if level.samples != main.samples || level.dtype != main.dtype {
                return Err(GeoTiffError::Invalid(format!(
                    "{name}: overview IFD {i} has {} {} bands, the image {} {} bands",
                    level.samples,
                    level.dtype.name(),
                    main.samples,
                    main.dtype.name()
                )));
            }
            levels.push(level);
        }
        levels[1..].sort_by_key(|l| std::cmp::Reverse((l.width, l.height)));
        let georef = georef::read_georef(main_ifd).map_err(with_name)?;
        let nodata = georef::read_nodata(main_ifd).map_err(with_name)?;
        Ok(GeoTiff {
            source,
            order: tiff.order,
            bigtiff: tiff.bigtiff,
            levels,
            georef,
            nodata,
        })
    }

    /// Width of the full-resolution image.
    #[must_use]
    pub fn width(&self) -> usize {
        self.levels[0].width
    }

    /// Height of the full-resolution image.
    #[must_use]
    pub fn height(&self) -> usize {
        self.levels[0].height
    }

    /// Number of bands (samples per pixel).
    #[must_use]
    pub fn band_count(&self) -> usize {
        self.levels[0].samples
    }

    /// Sample data type.
    #[must_use]
    pub fn dtype(&self) -> DType {
        self.levels[0].dtype
    }

    /// EPSG code of the CRS, or why the file has none this reader supports.
    ///
    /// # Errors
    /// Returns the reason (no CRS, user-defined CRS, ...) as text.
    pub fn epsg(&self) -> std::result::Result<u32, String> {
        self.georef.epsg.clone()
    }

    /// Affine transform `[a, b, c, d, e, f]` of the full-resolution image
    /// (`x = a*col + b*row + c`, `y = d*col + e*row + f`, at pixel corners;
    /// `PixelIsPoint` files are shifted by half a pixel as GDAL does).
    #[must_use]
    pub fn transform(&self) -> Option<[f64; 6]> {
        self.georef.transform
    }

    /// `PixelIsArea` or `PixelIsPoint`.
    #[must_use]
    pub fn raster_type(&self) -> RasterType {
        self.georef.raster_type
    }

    /// The CRS citation GeoKeys, if any.
    #[must_use]
    pub fn crs_citation(&self) -> Option<&str> {
        self.georef.citation.as_deref()
    }

    /// The `GDAL_NODATA` value, if any.
    #[must_use]
    pub fn nodata(&self) -> Option<f64> {
        self.nodata
    }

    /// Whether the full-resolution image is tiled (rather than striped).
    #[must_use]
    pub fn tiled(&self) -> bool {
        self.levels[0].tiled
    }

    /// Whether bands are stored in separate planes (`PlanarConfiguration` 2).
    #[must_use]
    pub fn planar(&self) -> bool {
        self.levels[0].planar
    }

    /// Name of the compression of the full-resolution image.
    #[must_use]
    pub fn compression(&self) -> String {
        codec::compression_name(self.levels[0].compression)
    }

    /// The TIFF predictor of the full-resolution image (1 none, 2 horizontal, 3 float).
    #[must_use]
    pub fn predictor(&self) -> u16 {
        self.levels[0].predictor
    }

    /// The TIFF photometric interpretation code of the full-resolution image.
    #[must_use]
    pub fn photometric(&self) -> u16 {
        self.levels[0].photometric
    }

    /// Whether the file is little-endian.
    #[must_use]
    pub fn little_endian(&self) -> bool {
        self.order == ByteOrder::Little
    }

    /// Whether the file is a BigTIFF.
    #[must_use]
    pub fn bigtiff(&self) -> bool {
        self.bigtiff
    }

    /// Size and block layout of each level: full resolution first, then the
    /// overviews from largest to smallest.
    #[must_use]
    pub fn levels(&self) -> Vec<LevelInfo> {
        self.levels
            .iter()
            .map(|l| LevelInfo {
                width: l.width,
                height: l.height,
                block_width: l.chunk_width,
                block_height: l.chunk_height,
            })
            .collect()
    }

    /// Read the pixels of `window` at resolution level `overview` (0 = full
    /// resolution, 1 = largest overview, ...), in that level's pixel grid.
    ///
    /// `bands` selects bands by 0-based index (all bands when `None`); a band
    /// may be repeated. Only the tiles or strips intersecting the window are
    /// read and decoded. Parts of the window outside the raster, and sparse
    /// (absent) chunks inside it, hold the nodata value (0 without one);
    /// [`WindowData::valid`] marks the pixels inside the raster.
    ///
    /// # Errors
    /// Returns [`GeoTiffError::Invalid`] for an empty or too large window, an
    /// unknown band or overview, or corrupt data, and [`GeoTiffError::Io`]
    /// when reading fails.
    pub fn read_window(
        &self,
        window: Window,
        bands: Option<&[usize]>,
        overview: usize,
    ) -> Result<WindowData> {
        let level = self.levels.get(overview).ok_or_else(|| {
            GeoTiffError::Invalid(format!(
                "overview {overview} does not exist: the file has {} overview level(s)",
                self.levels.len() - 1
            ))
        })?;
        let all: Vec<usize> = (0..level.samples).collect();
        let bands = bands.unwrap_or(&all);
        if bands.is_empty() {
            return Err(GeoTiffError::Invalid("bands must not be empty".to_owned()));
        }
        if let Some(&bad) = bands.iter().find(|&&b| b >= level.samples) {
            return Err(GeoTiffError::Invalid(format!(
                "band index {bad} is out of range: the file has {} band(s) (indices are 0-based)",
                level.samples
            )));
        }
        let (height, width) = window_size(window, bands.len() * level.dtype.size())?;
        let mut data = WindowData {
            height,
            width,
            bands: bands.len(),
            samples: Samples::filled(level.dtype, height * width * bands.len(), self.nodata),
            valid: vec![false; height * width],
        };
        let Some(inside) = Inside::new(window, level) else {
            return Ok(data);
        };
        for r in inside.dr..inside.dr + (inside.rb - inside.ra) {
            let row = r * width + inside.dc;
            data.valid[row..row + (inside.cb - inside.ca)].fill(true);
        }
        let plan = Plan::new(level, &inside, bands)?;
        let decoded = self.decode(level, &plan)?;
        copy_chunks(
            level,
            &plan,
            &decoded,
            bands,
            &inside,
            width,
            data.samples.as_bytes_mut(),
        );
        Ok(data)
    }

    /// Fetch and decode (in parallel) the chunks of `plan`; sparse chunks are `None`.
    fn decode(&self, level: &Level, plan: &Plan) -> Result<Vec<Option<Vec<u8>>>> {
        let ranges: Vec<(u64, usize)> = plan.requests.iter().filter_map(|r| r.range).collect();
        let mut fetched = self.source.read_ranges(&ranges)?.into_iter();
        let raw: Vec<Option<Vec<u8>>> = plan
            .requests
            .iter()
            .map(|r| r.range.and_then(|_| fetched.next()))
            .collect();
        let format = codec::ChunkFormat {
            compression: level.compression,
            predictor: level.predictor,
            order: self.order,
            sample_bytes: level.dtype.size(),
            samples: level.chunk_samples(),
            width: level.chunk_width,
            height: level.chunk_height,
            photometric: level.photometric,
            jpeg_tables: level.jpeg_tables.as_deref(),
        };
        plan.requests
            .par_iter()
            .zip(raw.par_iter())
            .map(|(request, data)| {
                let Some(data) = data else { return Ok(None) };
                let mut format = format;
                if request.partial_strip {
                    format.compression = codec::compression::NONE;
                }
                codec::decode_chunk(&format, data, request.rows).map(Some)
            })
            .collect()
    }
}

/// Rows and columns of `window`, checking that it is non-empty and that its
/// samples (`pixel_bytes` per pixel) stay under [`MAX_WINDOW_BYTES`].
fn window_size(window: Window, pixel_bytes: usize) -> Result<(usize, usize)> {
    let Window {
        row0,
        row1,
        col0,
        col1,
    } = window;
    if row1 <= row0 || col1 <= col0 {
        return Err(GeoTiffError::Invalid(format!(
            "empty window rows {row0}..{row1}, cols {col0}..{col1}"
        )));
    }
    let size = |a: i64, b: i64| b.checked_sub(a).and_then(|n| usize::try_from(n).ok());
    let (Some(height), Some(width)) = (size(row0, row1), size(col0, col1)) else {
        return Err(GeoTiffError::Invalid("window is too large".to_owned()));
    };
    height
        .checked_mul(width)
        .and_then(|n| n.checked_mul(pixel_bytes))
        .filter(|&n| n <= MAX_WINDOW_BYTES)
        .ok_or_else(|| {
            GeoTiffError::Invalid(format!(
                "a {height}x{width} window is larger than the {MAX_WINDOW_BYTES}-byte limit; \
                 read it in pieces"
            ))
        })?;
    Ok((height, width))
}

/// The part of a window inside the raster.
struct Inside {
    /// First image row covered.
    ra: usize,
    /// One past the last image row covered.
    rb: usize,
    /// First image column covered.
    ca: usize,
    /// One past the last image column covered.
    cb: usize,
    /// Window row of image row `ra`.
    dr: usize,
    /// Window column of image column `ca`.
    dc: usize,
}

impl Inside {
    /// `None` when the window does not overlap the raster.
    fn new(window: Window, level: &Level) -> Option<Inside> {
        let clamp = |v: i64, max: usize| usize::try_from(v.max(0)).unwrap_or(usize::MAX).min(max);
        let (ra, rb) = (
            clamp(window.row0, level.height),
            clamp(window.row1, level.height),
        );
        let (ca, cb) = (
            clamp(window.col0, level.width),
            clamp(window.col1, level.width),
        );
        if ra >= rb || ca >= cb {
            return None;
        }
        let offset = |image: usize, start: i64| {
            i64::try_from(image)
                .ok()
                .and_then(|i| i.checked_sub(start))
                .and_then(|d| usize::try_from(d).ok())
        };
        Some(Inside {
            ra,
            rb,
            ca,
            cb,
            dr: offset(ra, window.row0)?,
            dc: offset(ca, window.col0)?,
        })
    }
}

/// One tile or strip (of one band plane) needed by a window read.
struct ChunkRequest {
    /// File byte range of the data to decode; `None` for a sparse (absent) chunk.
    range: Option<(u64, usize)>,
    /// Image row of the first decoded row.
    first_row: usize,
    /// Rows to decode.
    rows: usize,
    /// Whether `range` holds only `rows` uncompressed rows of a strip.
    partial_strip: bool,
}

/// The chunks under a window: `requests[(slot * ny + cy - cy0) * nx + cx - cx0]`
/// is chunk `(cy, cx)` of band plane `planes[slot]`.
struct Plan {
    planes: Vec<usize>,
    cy0: usize,
    cx0: usize,
    cx1: usize,
    ny: usize,
    nx: usize,
    requests: Vec<ChunkRequest>,
}

impl Plan {
    fn new(level: &Level, inside: &Inside, bands: &[usize]) -> Result<Plan> {
        let (cw, ch) = (level.chunk_width, level.chunk_height);
        let (cy0, cy1) = (inside.ra / ch, (inside.rb - 1) / ch);
        let (cx0, cx1) = (inside.ca / cw, (inside.cb - 1) / cw);
        // One plane per selected band for band-interleaved files, else the
        // single pixel-interleaved plane.
        let planes: Vec<usize> = if level.planar {
            let mut p = bands.to_vec();
            p.sort_unstable();
            p.dedup();
            p
        } else {
            vec![0]
        };
        let too_large = |rows: usize| {
            GeoTiffError::Invalid(format!(
                "a {cw}x{rows} block decodes to more than {MAX_CHUNK_BYTES} bytes; convert the \
                 file to a tiled GeoTIFF (e.g. a COG)"
            ))
        };
        let row_bytes = cw
            .checked_mul(level.chunk_samples())
            .and_then(|n| n.checked_mul(level.dtype.size()))
            .ok_or_else(|| too_large(1))?;
        // Bytes of `rows` decoded rows of one chunk, within the per-chunk limit.
        let decoded_bytes = |rows: usize| {
            rows.checked_mul(row_bytes)
                .filter(|&bytes| bytes <= MAX_CHUNK_BYTES)
                .ok_or_else(|| too_large(rows))
        };
        let per_plane = level.chunks_across() * level.chunks_down();
        // Uncompressed strips are read row-exact rather than whole.
        let partial = !level.tiled && level.compression == codec::compression::NONE;
        let mut requests = Vec::with_capacity(planes.len() * (cy1 - cy0 + 1) * (cx1 - cx0 + 1));
        for &plane in &planes {
            for cy in cy0..=cy1 {
                for cx in cx0..=cx1 {
                    let index = plane * per_plane + cy * level.chunks_across() + cx;
                    // `from_ifd` checked that both arrays hold one value per chunk.
                    let (Some(offset), Some(count)) =
                        (level.offsets.get(index), level.byte_counts.get(index))
                    else {
                        return Err(GeoTiffError::Invalid(format!(
                            "chunk {index} has no offset or byte count"
                        )));
                    };
                    let mut request = ChunkRequest {
                        range: None,
                        first_row: cy * ch,
                        rows: level.rows_in_chunk(cy),
                        partial_strip: false,
                    };
                    // GDAL writes absent ("sparse") chunks with offset and size 0.
                    if offset != 0 && count != 0 {
                        if partial {
                            let first = inside.ra.max(cy * ch);
                            let last = inside.rb.min(cy * ch + request.rows);
                            request.first_row = first;
                            request.rows = last - first;
                            request.partial_strip = true;
                            let len = decoded_bytes(request.rows)?;
                            // The first wanted row lies `skip` bytes into the strip.
                            let start = (first - cy * ch)
                                .checked_mul(row_bytes)
                                .and_then(|skip| u64::try_from(skip).ok())
                                .and_then(|skip| offset.checked_add(skip))
                                .ok_or_else(|| {
                                    GeoTiffError::Invalid(format!(
                                        "strip offset {offset} is out of range"
                                    ))
                                })?;
                            request.range = Some((start, len));
                        } else {
                            let len = usize::try_from(count).map_err(|_| {
                                GeoTiffError::Invalid("chunk byte count is too large".to_owned())
                            })?;
                            decoded_bytes(request.rows)?;
                            request.range = Some((offset, len));
                        }
                    }
                    requests.push(request);
                }
            }
        }
        Ok(Plan {
            planes,
            cy0,
            cx0,
            cx1,
            ny: cy1 - cy0 + 1,
            nx: cx1 - cx0 + 1,
            requests,
        })
    }
}

/// Copy decoded chunks into the window's `(rows, cols, bands)` sample bytes,
/// one output row per parallel task.
fn copy_chunks(
    level: &Level,
    plan: &Plan,
    decoded: &[Option<Vec<u8>>],
    bands: &[usize],
    inside: &Inside,
    out_width: usize,
    out: &mut [u8],
) {
    let sample_bytes = level.dtype.size();
    let chunk_samples = level.chunk_samples();
    let (cw, ch) = (level.chunk_width, level.chunk_height);
    let chunk_row_bytes = cw * chunk_samples * sample_bytes;
    let n_bands = bands.len();
    // Pixel-interleaved data with all bands in order is copied span by span.
    let identity =
        !level.planar && n_bands == level.samples && bands.iter().enumerate().all(|(i, &b)| i == b);
    out.par_chunks_mut(out_width * n_bands * sample_bytes)
        .enumerate()
        .skip(inside.dr)
        .take(inside.rb - inside.ra)
        .for_each(|(r, out_row)| {
            let image_row = inside.ra + (r - inside.dr);
            let cy = image_row / ch;
            for cx in plan.cx0..=plan.cx1 {
                let c_lo = inside.ca.max(cx * cw);
                let c_hi = inside.cb.min((cx + 1) * cw);
                let (src_x, dst_x, n) = (c_lo - cx * cw, c_lo - inside.ca + inside.dc, c_hi - c_lo);
                for (slot, &plane) in plan.planes.iter().enumerate() {
                    let index = (slot * plan.ny + (cy - plan.cy0)) * plan.nx + (cx - plan.cx0);
                    let Some(chunk) = &decoded[index] else {
                        continue;
                    };
                    let local_row = image_row - plan.requests[index].first_row;
                    let src = &chunk[local_row * chunk_row_bytes..][..chunk_row_bytes];
                    if identity {
                        let px = chunk_samples * sample_bytes;
                        out_row[dst_x * px..(dst_x + n) * px]
                            .copy_from_slice(&src[src_x * px..(src_x + n) * px]);
                        continue;
                    }
                    for (k, &band) in bands.iter().enumerate() {
                        let sample = match (level.planar, band == plane) {
                            (false, _) => band,
                            (true, true) => 0,
                            (true, false) => continue,
                        };
                        for i in 0..n {
                            let s = ((src_x + i) * chunk_samples + sample) * sample_bytes;
                            let d = ((dst_x + i) * n_bands + k) * sample_bytes;
                            out_row[d..d + sample_bytes].copy_from_slice(&src[s..s + sample_bytes]);
                        }
                    }
                }
            }
        });
}

#[cfg(test)]
#[allow(clippy::cast_possible_truncation)]
mod tests {
    use super::*;

    /// A little-endian TIFF (or BigTIFF) with one IFD whose tags each hold a single
    /// inline value `(tag, field type, value)`, followed by 8 bytes of "pixels".
    fn inline_tiff(bigtiff: bool, tags: &[(u16, u16, u64)]) -> Vec<u8> {
        let mut f = b"II".to_vec();
        if bigtiff {
            f.extend(43u16.to_le_bytes());
            f.extend(8u16.to_le_bytes());
            f.extend(0u16.to_le_bytes());
            f.extend(16u64.to_le_bytes());
            f.extend((tags.len() as u64).to_le_bytes());
        } else {
            f.extend(42u16.to_le_bytes());
            f.extend(8u32.to_le_bytes());
            f.extend((tags.len() as u16).to_le_bytes());
        }
        for &(tag, field_type, value) in tags {
            f.extend(tag.to_le_bytes());
            f.extend(field_type.to_le_bytes());
            if bigtiff {
                f.extend(1u64.to_le_bytes());
                f.extend(value.to_le_bytes());
            } else {
                f.extend(1u32.to_le_bytes());
                f.extend((value as u32).to_le_bytes());
            }
        }
        f.extend(if bigtiff { vec![0; 8] } else { vec![0; 4] });
        f.extend([0u8; 8]);
        f
    }

    const SHORT: u16 = 3;
    const LONG: u16 = 4;
    const LONG8: u16 = 16;
    const HUGE: u64 = 1 << 31;

    /// Read one pixel (band 0) at `row`, `col`.
    fn read_pixel(tiff: &GeoTiff, row: i64, col: i64) -> Result<WindowData> {
        let window = Window {
            row0: row,
            row1: row + 1,
            col0: col,
            col1: col + 1,
        };
        tiff.read_window(window, Some(&[0]), 0)
    }

    // Found by fuzzing (geotiff_structured): the chunk count overflowed `usize`.
    #[test]
    fn chunk_count_overflow_is_an_error() {
        // 2^31 x 2^31 pixels in 1x1 chunks, band-interleaved with 65535 bands.
        let file = inline_tiff(
            false,
            &[
                (256, LONG, HUGE),
                (257, LONG, HUGE),
                (258, SHORT, 8),
                (277, SHORT, 65535),
                (284, SHORT, 2),
                (322, LONG, 1),
                (323, LONG, 1),
                (324, LONG, 0),
                (325, LONG, 0),
            ],
        );
        let err = GeoTiff::from_bytes(file).err().expect("must be rejected");
        assert!(
            err.to_string().contains("impossible number of chunks"),
            "{err}"
        );
    }

    // Found by fuzzing (geotiff_structured): rows x row bytes overflowed `usize`.
    #[test]
    fn huge_tile_size_overflow_is_an_error() {
        // One 2^31 x 2^31 tile with 300 bands: 2^31 * 300 bytes per row.
        let file = inline_tiff(
            false,
            &[
                (256, LONG, HUGE),
                (257, LONG, HUGE),
                (258, SHORT, 8),
                (277, SHORT, 300),
                (322, LONG, HUGE),
                (323, LONG, HUGE),
                (324, LONG, 8),
                (325, LONG, 8),
            ],
        );
        let tiff = GeoTiff::from_bytes(file).unwrap();
        let err = read_pixel(&tiff, 0, 0).unwrap_err();
        assert!(err.to_string().contains("decodes to more than"), "{err}");
    }

    // Found by fuzzing (geotiff_structured): skipping rows of an uncompressed strip
    // multiplied the row count by the row size without a check.
    #[test]
    fn huge_strip_skip_overflow_is_an_error() {
        let file = inline_tiff(
            false,
            &[
                (256, LONG, HUGE),
                (257, LONG, HUGE),
                (258, SHORT, 8),
                (277, SHORT, 65535),
                (278, LONG, HUGE),
                (273, LONG, 8),
                (279, LONG, 8),
            ],
        );
        let tiff = GeoTiff::from_bytes(file).unwrap();
        let err = read_pixel(&tiff, 1 << 20, 0).unwrap_err();
        assert!(err.to_string().contains("decodes to more than"), "{err}");
    }

    // Found by auditing Plan::new while fuzzing: `offset + skip` overflowed `u64`.
    #[test]
    fn strip_offset_near_u64_max_is_an_error() {
        // 4x100 uncompressed bytes in one strip that "starts" 8 bytes before 2^64.
        let file = inline_tiff(
            true,
            &[
                (256, LONG, 4),
                (257, LONG, 100),
                (258, SHORT, 8),
                (259, SHORT, 1),
                (277, SHORT, 1),
                (278, LONG, 100),
                (273, LONG8, u64::MAX - 8),
                (279, LONG8, 400),
            ],
        );
        let tiff = GeoTiff::from_bytes(file).unwrap();
        let err = read_pixel(&tiff, 50, 0).unwrap_err();
        assert!(err.to_string().contains("out of range"), "{err}");
    }

    #[test]
    fn in_memory_tiff_reads_pixels() {
        // 2x2 uncompressed bytes at file offset 8 + 2 + 8 * 12 + 4 = 110.
        let mut file = inline_tiff(
            false,
            &[
                (256, LONG, 2),
                (257, LONG, 2),
                (258, SHORT, 8),
                (259, SHORT, 1),
                (277, SHORT, 1),
                (278, LONG, 2),
                (273, LONG, 110),
                (279, LONG, 4),
            ],
        );
        file.truncate(file.len() - 8);
        file.extend([1, 2, 3, 4]);
        let tiff = GeoTiff::from_bytes(file).unwrap();
        assert_eq!((tiff.width(), tiff.height(), tiff.band_count()), (2, 2, 1));
        let data = read_pixel(&tiff, 1, 0).unwrap();
        assert_eq!(data.samples, Samples::U8(vec![3]));
    }

    #[test]
    fn url_detection() {
        assert!(is_url("https://example.com/a.tif"));
        assert!(is_url("s3://bucket/key.tif"));
        assert!(!is_url("/data/a.tif"));
        assert!(!is_url("C:\\data\\a.tif"));
        assert!(!is_url("relative/a.tif"));
    }

    #[test]
    fn fill_values() {
        assert_eq!(int_fill::<u8>(Some(255.0)), 255);
        assert_eq!(int_fill::<u8>(Some(-9999.0)), 0);
        assert_eq!(int_fill::<i16>(Some(-9999.0)), -9999);
        assert_eq!(int_fill::<u16>(Some(f64::NAN)), 0);
        assert_eq!(int_fill::<u16>(Some(1.5)), 0);
        assert_eq!(int_fill::<i32>(None), 0);
        match Samples::filled(DType::F32, 2, Some(f64::NAN)) {
            Samples::F32(v) => assert!(v.iter().all(|x| x.is_nan())),
            other => panic!("unexpected {other:?}"),
        }
    }
}
