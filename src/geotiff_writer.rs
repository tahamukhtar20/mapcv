//! GDAL-free GeoTIFF writer for patches.
//!
//! [`encode`] turns one pixel-interleaved `(rows, cols, bands)` raster into the
//! bytes of a classic (32-bit offset), little-endian TIFF that GDAL, rasterio,
//! QGIS and the reader in [`crate::geotiff`] open with the same georeferencing:
//!
//! * **Layout**: striped, chunky (`PlanarConfiguration` 1), one strip per
//!   [`STRIP_BYTES`] of pixels (so a typical 256 x 256 patch is a single
//!   strip), `PhotometricInterpretation` RGB for 3-band 8/16-bit rasters and
//!   `MinIsBlack` otherwise.
//! * **Compression**: Deflate (`Compression` 8, the zlib stream GDAL calls
//!   `DEFLATE`) with the horizontal predictor (2) for 8, 16 and 32-bit
//!   integers and the floating-point predictor (3) for `float32`/`float64`. 64-bit
//!   integers are stored without a predictor, which libtiff does not apply to
//!   them everywhere. The zlib stream comes from `fdeflate` (fast, like PNG
//!   writing) for 8 and 16-bit integers and from zlib level 1 for wider types;
//!   see [`Options::level`]. Strips are compressed in parallel.
//! * **Georeferencing** (`PixelIsArea`): `GTModelType`, `GTRasterType` and the
//!   `ProjectedCSType` or `GeographicType` GeoKey carry the EPSG code. A
//!   north-up transform (`b = d = 0`, `a > 0`, `e < 0`) is stored as
//!   `ModelPixelScale` + `ModelTiepoint`; any other (rotated, sheared or
//!   south-up) transform as `ModelTransformation`. Both are what GDAL writes.
//! * **`GDAL_NODATA`** and per-band descriptions (`GDAL_METADATA`) when given.
//!
//! The container follows the TIFF 6.0 specification, the GeoTIFF 1.0
//! specification and Adobe's predictor note (TIFF Technical Note 3, for
//! predictor 3); no code is taken from another project.

use crate::geotiff::DType;
use flate2::write::ZlibEncoder;
use flate2::Compression;
use rayon::prelude::*;
use std::fmt::Write as _;
use std::io::Write;
use std::path::Path;

/// Uncompressed bytes per strip (rows are grouped until a strip reaches this).
pub const STRIP_BYTES: usize = 1 << 20;
/// Zlib level of the "auto" effort for data that compresses well: floating point
/// and 32/64-bit integers. `fdeflate` barely shrinks float bytes (7% against 29%
/// here), and level 1 is still above 100 MB/s per core.
const AUTO_ZLIB_LEVEL: u32 = 1;

/// TIFF field types.
const ASCII: u16 = 2;
const SHORT: u16 = 3;
const LONG: u16 = 4;
const DOUBLE: u16 = 12;

/// TIFF tags (see also `geotiff::ifd::tag` for those the reader loads).
const T_WIDTH: u16 = 256;
const T_LENGTH: u16 = 257;
const T_BITS: u16 = 258;
const T_COMPRESSION: u16 = 259;
const T_PHOTOMETRIC: u16 = 262;
const T_STRIP_OFFSETS: u16 = 273;
const T_SAMPLES: u16 = 277;
const T_ROWS_PER_STRIP: u16 = 278;
const T_STRIP_BYTE_COUNTS: u16 = 279;
const T_PLANAR: u16 = 284;
const T_PREDICTOR: u16 = 317;
const T_EXTRA_SAMPLES: u16 = 338;
const T_SAMPLE_FORMAT: u16 = 339;
const T_PIXEL_SCALE: u16 = 33550;
const T_TIEPOINT: u16 = 33922;
const T_TRANSFORMATION: u16 = 34264;
const T_GEO_KEYS: u16 = 34735;
const T_GDAL_METADATA: u16 = 42112;
const T_GDAL_NODATA: u16 = 42113;

/// Where a raster sits on the Earth.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Georef {
    /// EPSG code of the CRS (1..=32766).
    pub epsg: u32,
    /// Whether the CRS is geographic (angular units) rather than projected.
    pub geographic: bool,
    /// Affine transform `[a, b, c, d, e, f]` of pixel corners:
    /// `x = c + a * col + b * row`, `y = f + d * col + e * row`.
    pub transform: [f64; 6],
}

/// Shape and sample type of a raster, pixel-interleaved.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct RasterFormat {
    /// Columns.
    pub width: usize,
    /// Rows.
    pub height: usize,
    /// Bands (samples per pixel).
    pub bands: usize,
    /// Sample type.
    pub dtype: DType,
}

impl RasterFormat {
    /// Bytes of the whole raster, or `None` when it overflows `usize`.
    #[must_use]
    pub fn byte_len(&self) -> Option<usize> {
        self.width
            .checked_mul(self.height)?
            .checked_mul(self.bands)?
            .checked_mul(self.dtype.size())
    }
}

/// Optional tags and the compression effort.
#[derive(Clone, Debug)]
pub struct Options<'a> {
    /// `GDAL_NODATA` value, if the raster has one.
    pub nodata: Option<f64>,
    /// One description per band (empty for none).
    pub band_names: &'a [String],
    /// Compression level: 0 is `fdeflate`, the fast compressor behind `image`'s PNG
    /// writer; 1 to 9 are zlib (miniz) levels, a few percent smaller on 8-bit data
    /// and several times slower. `None` picks 0 for 8 and 16-bit integers (where it
    /// is as small as PNG and 3x quicker than zlib) and 1 for everything wider.
    pub level: Option<u32>,
}

/// Map a numpy dtype name to a [`DType`].
///
/// # Errors
/// Returns a message listing the supported names.
pub fn dtype_from_name(name: &str) -> Result<DType, String> {
    Ok(match name {
        "uint8" => DType::U8,
        "int8" => DType::I8,
        "uint16" => DType::U16,
        "int16" => DType::I16,
        "uint32" => DType::U32,
        "int32" => DType::I32,
        "uint64" => DType::U64,
        "int64" => DType::I64,
        "float32" => DType::F32,
        "float64" => DType::F64,
        other => {
            return Err(format!(
                "dtype '{other}' cannot be written to a GeoTIFF; use uint8, int8, uint16, int16, \
                 uint32, int32, uint64, int64, float32 or float64"
            ))
        }
    })
}

/// `(BitsPerSample, SampleFormat)` of a type.
fn sample_tags(dtype: DType) -> (u16, u16) {
    match dtype {
        DType::U8 => (8, 1),
        DType::I8 => (8, 2),
        DType::U16 => (16, 1),
        DType::I16 => (16, 2),
        DType::U32 => (32, 1),
        DType::I32 => (32, 2),
        DType::U64 => (64, 1),
        DType::I64 => (64, 2),
        DType::F32 => (32, 3),
        DType::F64 => (64, 3),
    }
}

/// The `Predictor` tag value to use for a type.
fn predictor_for(dtype: DType) -> u16 {
    match dtype {
        DType::U8 | DType::I8 | DType::U16 | DType::I16 | DType::U32 | DType::I32 => 2,
        DType::F32 | DType::F64 => 3,
        DType::U64 | DType::I64 => 1,
    }
}

/// Apply predictor 2 to one row in place: each sample becomes the difference
/// from the same band of the previous pixel (native-endian samples).
fn horizontal_difference(row: &mut [u8], sample_bytes: usize, bands: usize) {
    macro_rules! difference {
        ($t:ty) => {{
            const N: usize = std::mem::size_of::<$t>();
            let lag = bands * N;
            for i in (lag..row.len()).step_by(N).rev() {
                let prev = <$t>::from_ne_bytes(row[i - lag..i - lag + N].try_into().unwrap());
                let cur = <$t>::from_ne_bytes(row[i..i + N].try_into().unwrap());
                row[i..i + N].copy_from_slice(&cur.wrapping_sub(prev).to_ne_bytes());
            }
        }};
    }
    match sample_bytes {
        1 => {
            for i in (bands..row.len()).rev() {
                row[i] = row[i].wrapping_sub(row[i - bands]);
            }
        }
        2 => difference!(u16),
        _ => difference!(u32),
    }
}

/// Apply predictor 3 to one row in place: split the samples into byte planes
/// (most significant byte first), then difference the bytes with a lag of one
/// pixel.
fn floating_point_difference(
    row: &mut [u8],
    scratch: &mut [u8],
    sample_bytes: usize,
    bands: usize,
) {
    scratch.copy_from_slice(row);
    let count = row.len() / sample_bytes;
    for (i, sample) in scratch.chunks_exact(sample_bytes).enumerate() {
        for (byte, &value) in sample.iter().enumerate() {
            let plane = if cfg!(target_endian = "little") {
                sample_bytes - 1 - byte
            } else {
                byte
            };
            row[plane * count + i] = value;
        }
    }
    for i in (bands..row.len()).rev() {
        row[i] = row[i].wrapping_sub(row[i - bands]);
    }
}

/// Predict (rows of `row_bytes`) and deflate one strip.
fn compress_strip(
    raw: &[u8],
    row_bytes: usize,
    fmt: &RasterFormat,
    predictor: u16,
    level: u32,
) -> Result<Vec<u8>, String> {
    let sample_bytes = fmt.dtype.size();
    let mut work;
    let input = if predictor == 1 {
        raw
    } else {
        work = raw.to_vec();
        let mut scratch = vec![0u8; row_bytes];
        for row in work.chunks_exact_mut(row_bytes) {
            if predictor == 2 {
                horizontal_difference(row, sample_bytes, fmt.bands);
            } else {
                floating_point_difference(row, &mut scratch, sample_bytes, fmt.bands);
            }
        }
        &work
    };
    if level == 0 {
        return Ok(fdeflate::compress_to_vec(input));
    }
    let mut encoder = ZlibEncoder::new(
        Vec::with_capacity(input.len() / 2 + 64),
        Compression::new(level),
    );
    encoder.write_all(input).map_err(|e| e.to_string())?;
    encoder.finish().map_err(|e| e.to_string())
}

/// One IFD entry: tag, field type, value count and the little-endian value bytes.
struct Entry {
    tag: u16,
    kind: u16,
    count: usize,
    data: Vec<u8>,
}

impl Entry {
    fn shorts(tag: u16, values: &[u16]) -> Entry {
        Entry {
            tag,
            kind: SHORT,
            count: values.len(),
            data: values.iter().flat_map(|v| v.to_le_bytes()).collect(),
        }
    }

    fn longs(tag: u16, values: &[u32]) -> Entry {
        Entry {
            tag,
            kind: LONG,
            count: values.len(),
            data: values.iter().flat_map(|v| v.to_le_bytes()).collect(),
        }
    }

    fn doubles(tag: u16, values: &[f64]) -> Entry {
        Entry {
            tag,
            kind: DOUBLE,
            count: values.len(),
            data: values.iter().flat_map(|v| v.to_le_bytes()).collect(),
        }
    }

    /// An ASCII value; the count includes the terminating NUL.
    fn ascii(tag: u16, text: &str) -> Entry {
        let mut data = text.as_bytes().to_vec();
        data.push(0);
        Entry {
            tag,
            kind: ASCII,
            count: data.len(),
            data,
        }
    }
}

/// `GDAL_NODATA` text for a value: `nan`, an integer, or the shortest float that round-trips.
fn nodata_text(value: f64) -> String {
    if value.is_nan() {
        "nan".to_owned()
    } else if value.is_infinite() {
        if value > 0.0 { "inf" } else { "-inf" }.to_owned()
    } else {
        format!("{value}")
    }
}

fn xml_escape(text: &str) -> String {
    let mut out = String::with_capacity(text.len());
    for c in text.chars() {
        match c {
            '&' => out.push_str("&amp;"),
            '<' => out.push_str("&lt;"),
            '>' => out.push_str("&gt;"),
            '"' => out.push_str("&quot;"),
            '\0' => {}
            c => out.push(c),
        }
    }
    out
}

/// The `GDAL_METADATA` XML that names the bands.
fn band_description_xml(names: &[String]) -> String {
    let mut xml = String::from("<GDALMetadata>\n");
    for (i, name) in names.iter().enumerate() {
        let _ = writeln!(
            xml,
            "  <Item name=\"DESCRIPTION\" sample=\"{i}\" role=\"description\">{}</Item>",
            // GDAL escapes a description twice (it is escaped as metadata, then as XML) and
            // reads it back that way, so a name with `&` or `<` only survives if written alike.
            xml_escape(&xml_escape(name))
        );
    }
    xml.push_str("</GDALMetadata>\n");
    xml
}

/// The georeferencing tags: GeoKeys plus either scale and tie point or a full matrix.
#[allow(clippy::many_single_char_names)] // a..f are the names of the affine coefficients
fn georef_entries(georef: &Georef) -> Result<Vec<Entry>, String> {
    let [a, b, c, d, e, f] = georef.transform;
    if georef.transform.iter().any(|v| !v.is_finite()) {
        return Err("the affine transform must be finite".to_owned());
    }
    if a * e - b * d == 0.0 {
        return Err("the affine transform is singular".to_owned());
    }
    let code = u16::try_from(georef.epsg)
        .ok()
        .filter(|code| (1..32767).contains(code))
        .ok_or_else(|| format!("EPSG code {} is not in 1..=32766", georef.epsg))?;
    let (model_type, cs_key) = if georef.geographic {
        (2u16, 2048u16) // GeographicTypeGeoKey
    } else {
        (1u16, 3072u16) // ProjectedCSTypeGeoKey
    };
    let keys = [
        1, 1, 0, 3, // GeoKeyDirectory header: version 1.1.0, three keys
        1024, 0, 1, model_type, // GTModelTypeGeoKey
        1025, 0, 1, 1, // GTRasterTypeGeoKey: RasterPixelIsArea
        cs_key, 0, 1, code,
    ];
    let mut entries = vec![Entry::shorts(T_GEO_KEYS, &keys)];
    if b == 0.0 && d == 0.0 && a > 0.0 && e < 0.0 {
        entries.push(Entry::doubles(T_PIXEL_SCALE, &[a, -e, 0.0]));
        entries.push(Entry::doubles(T_TIEPOINT, &[0.0, 0.0, 0.0, c, f, 0.0]));
    } else {
        entries.push(Entry::doubles(
            T_TRANSFORMATION,
            &[
                a, b, 0.0, c, //
                d, e, 0.0, f, //
                0.0, 0.0, 0.0, 0.0, //
                0.0, 0.0, 0.0, 1.0,
            ],
        ));
    }
    Ok(entries)
}

/// Check that a sample count and dimensions fit the TIFF fields they go into.
fn check_format(fmt: &RasterFormat) -> Result<(usize, usize), String> {
    if fmt.width == 0 || fmt.height == 0 || fmt.bands == 0 {
        return Err("a GeoTIFF needs at least one row, column and band".to_owned());
    }
    if u32::try_from(fmt.width).is_err() || u32::try_from(fmt.height).is_err() {
        return Err("raster is too large for a GeoTIFF".to_owned());
    }
    if u16::try_from(fmt.bands).is_err() {
        return Err(format!(
            "{} bands do not fit a GeoTIFF (max 65535)",
            fmt.bands
        ));
    }
    let row_bytes = fmt
        .width
        .checked_mul(fmt.bands)
        .and_then(|n| n.checked_mul(fmt.dtype.size()))
        .ok_or("raster is too large for a GeoTIFF")?;
    fmt.byte_len().ok_or("raster is too large for a GeoTIFF")?;
    Ok((row_bytes, (STRIP_BYTES / row_bytes).clamp(1, fmt.height)))
}

/// The tags of the raster's structure, compression, georeferencing and metadata.
fn tag_entries(
    fmt: &RasterFormat,
    rows_per_strip: usize,
    georef: &Georef,
    options: &Options<'_>,
) -> Result<Vec<Entry>, String> {
    let (bits, format) = sample_tags(fmt.dtype);
    let bands = u16::try_from(fmt.bands).map_err(|e| e.to_string())?;
    let rgb = fmt.bands == 3 && matches!(fmt.dtype, DType::U8 | DType::U16);
    let mut entries = vec![
        Entry::longs(
            T_WIDTH,
            &[u32::try_from(fmt.width).map_err(|e| e.to_string())?],
        ),
        Entry::longs(
            T_LENGTH,
            &[u32::try_from(fmt.height).map_err(|e| e.to_string())?],
        ),
        Entry::shorts(T_BITS, &vec![bits; fmt.bands]),
        Entry::shorts(T_COMPRESSION, &[8]),
        Entry::shorts(T_PHOTOMETRIC, &[if rgb { 2 } else { 1 }]),
        Entry::shorts(T_SAMPLES, &[bands]),
        Entry::longs(
            T_ROWS_PER_STRIP,
            &[u32::try_from(rows_per_strip).map_err(|e| e.to_string())?],
        ),
        Entry::shorts(T_PLANAR, &[1]),
        Entry::shorts(T_PREDICTOR, &[predictor_for(fmt.dtype)]),
        Entry::shorts(T_SAMPLE_FORMAT, &vec![format; fmt.bands]),
    ];
    // Bands beyond the colour channels (none for RGB, all but one otherwise) are
    // "unspecified" extra samples, as GDAL writes them; libtiff warns without the tag.
    let extra = fmt.bands - if rgb { 3 } else { 1 };
    if extra > 0 {
        entries.push(Entry::shorts(T_EXTRA_SAMPLES, &vec![0; extra]));
    }
    entries.extend(georef_entries(georef)?);
    if !options.band_names.is_empty() {
        entries.push(Entry::ascii(
            T_GDAL_METADATA,
            &band_description_xml(options.band_names),
        ));
    }
    if let Some(nodata) = options.nodata {
        entries.push(Entry::ascii(T_GDAL_NODATA, &nodata_text(nodata)));
    }
    Ok(entries)
}

/// Lay out the file: header, strips, out-of-line tag values, then the IFD.
fn assemble(strips: &[Vec<u8>], mut entries: Vec<Entry>) -> Result<Vec<u8>, String> {
    fn pad(out: &mut Vec<u8>) {
        if out.len() % 2 == 1 {
            out.push(0);
        }
    }
    fn offset(out: &[u8]) -> Result<u32, String> {
        u32::try_from(out.len()).map_err(|_| "the GeoTIFF would exceed 4 GiB".to_owned())
    }
    let mut out: Vec<u8> = Vec::with_capacity(strips.iter().map(Vec::len).sum::<usize>() + 1024);
    out.extend_from_slice(&[b'I', b'I', 42, 0, 0, 0, 0, 0]);
    let mut offsets = Vec::with_capacity(strips.len());
    let mut counts = Vec::with_capacity(strips.len());
    for strip in strips {
        pad(&mut out);
        offsets.push(offset(&out)?);
        counts.push(u32::try_from(strip.len()).map_err(|e| e.to_string())?);
        out.extend_from_slice(strip);
    }
    entries.push(Entry::longs(T_STRIP_OFFSETS, &offsets));
    entries.push(Entry::longs(T_STRIP_BYTE_COUNTS, &counts));
    entries.sort_by_key(|e| e.tag);

    // Values of more than four bytes go out of line; shorter ones are stored in the entry.
    let mut values = Vec::with_capacity(entries.len());
    for entry in &entries {
        if entry.data.len() > 4 {
            pad(&mut out);
            values.push(offset(&out)?.to_le_bytes());
            out.extend_from_slice(&entry.data);
        } else {
            let mut inline = [0u8; 4];
            inline[..entry.data.len()].copy_from_slice(&entry.data);
            values.push(inline);
        }
    }
    pad(&mut out);
    let ifd = offset(&out)?;
    let n_entries = u16::try_from(entries.len()).map_err(|e| e.to_string())?;
    out.extend_from_slice(&n_entries.to_le_bytes());
    for (entry, value) in entries.iter().zip(&values) {
        out.extend_from_slice(&entry.tag.to_le_bytes());
        out.extend_from_slice(&entry.kind.to_le_bytes());
        let count = u32::try_from(entry.count).map_err(|e| e.to_string())?;
        out.extend_from_slice(&count.to_le_bytes());
        out.extend_from_slice(value);
    }
    out.extend_from_slice(&[0; 4]);
    offset(&out)?;
    out[4..8].copy_from_slice(&ifd.to_le_bytes());
    Ok(out)
}

/// Encode a pixel-interleaved raster as a GeoTIFF.
///
/// `pixels` holds `height * width * bands` native-endian samples of
/// `fmt.dtype`, row by row.
///
/// # Errors
/// Returns a message when `pixels` does not match `fmt`, the georeferencing or
/// options are invalid, or the file would exceed what a classic TIFF can
/// address (4 GiB).
pub fn encode(
    pixels: &[u8],
    fmt: &RasterFormat,
    georef: &Georef,
    options: &Options<'_>,
) -> Result<Vec<u8>, String> {
    let (row_bytes, rows_per_strip) = check_format(fmt)?;
    if Some(pixels.len()) != fmt.byte_len() {
        return Err(format!(
            "pixel buffer has {} bytes, expected {} for {} x {} x {} {}",
            pixels.len(),
            fmt.byte_len().unwrap_or(0),
            fmt.height,
            fmt.width,
            fmt.bands,
            fmt.dtype.name()
        ));
    }
    if options.level.is_some_and(|level| level > 9) {
        return Err(format!(
            "compression level must be 0 (fast) to 9, got {}",
            options.level.unwrap_or(0)
        ));
    }
    let level = options.level.unwrap_or(match fmt.dtype {
        DType::U8 | DType::I8 | DType::U16 | DType::I16 => 0,
        _ => AUTO_ZLIB_LEVEL,
    });
    if !options.band_names.is_empty() && options.band_names.len() != fmt.bands {
        return Err(format!(
            "{} band names for {} bands",
            options.band_names.len(),
            fmt.bands
        ));
    }
    let entries = tag_entries(fmt, rows_per_strip, georef, options)?;
    let predictor = predictor_for(fmt.dtype);
    let strips: Vec<Vec<u8>> = pixels
        .par_chunks(rows_per_strip * row_bytes)
        .map(|raw| compress_strip(raw, row_bytes, fmt, predictor, level))
        .collect::<Result<_, _>>()?;
    assemble(&strips, entries)
}

/// Write rasters of one shape, whose samples lie back to back in `pixels`, to
/// `dir`, in parallel: raster `i` goes to `files[i].0` with georeferencing
/// `files[i].1`.
///
/// Existing files are overwritten (a resumed run reuses the names of an
/// interrupted one).
///
/// # Errors
/// Returns a message when `pixels` does not hold `files.len()` rasters, a name
/// is not a plain file name, or any raster fails to encode or write.
pub fn write_all(
    pixels: &[u8],
    fmt: &RasterFormat,
    options: &Options<'_>,
    dir: &Path,
    files: &[(String, Georef)],
) -> Result<(), String> {
    if let Some((bad, _)) = files.iter().find(|(name, _)| {
        name.is_empty() || Path::new(name).file_name().and_then(|f| f.to_str()) != Some(name)
    }) {
        return Err(format!("'{bad}' is not a plain file name"));
    }
    let each = fmt.byte_len().ok_or("raster is too large for a GeoTIFF")?;
    if each == 0 {
        return Err("a GeoTIFF needs at least one row, column and band".to_owned());
    }
    if Some(pixels.len()) != each.checked_mul(files.len()) {
        return Err(format!(
            "pixel buffer has {} bytes, expected {} rasters of {each} bytes",
            pixels.len(),
            files.len()
        ));
    }
    pixels
        .par_chunks(each)
        .zip(files.par_iter())
        .try_for_each(|(chunk, (name, georef))| {
            let bytes = encode(chunk, fmt, georef, options)?;
            std::fs::write(dir.join(name), bytes).map_err(|e| format!("{name}: {e}"))
        })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::geotiff::{GeoTiff, Samples, Window};

    fn temp_dir(tag: &str) -> std::path::PathBuf {
        let dir = std::env::temp_dir().join(format!("mapcv-gtw-{tag}-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        dir
    }

    const NORTH_UP: [f64; 6] = [10.0, 0.0, 500_000.0, 0.0, -10.0, 4_000_000.0];

    fn bytes_of<T: bytemuck::NoUninit>(values: &[T]) -> Vec<u8> {
        bytemuck::cast_slice(values).to_vec()
    }

    fn write_and_open(
        pixels: &[u8],
        fmt: &RasterFormat,
        georef: &Georef,
        options: &Options<'_>,
        tag: &str,
    ) -> GeoTiff {
        let dir = temp_dir(tag);
        let path = dir.join("p.tif");
        std::fs::write(&path, encode(pixels, fmt, georef, options).unwrap()).unwrap();
        GeoTiff::open(path.to_str().unwrap(), 0).unwrap()
    }

    fn read_all(tiff: &GeoTiff) -> Samples {
        let window = Window {
            row0: 0,
            row1: i64::try_from(tiff.height()).unwrap(),
            col0: 0,
            col1: i64::try_from(tiff.width()).unwrap(),
        };
        tiff.read_window(window, None, 0).unwrap().samples
    }

    fn options(nodata: Option<f64>) -> Options<'static> {
        Options {
            nodata,
            band_names: &[],
            level: None,
        }
    }

    #[test]
    fn u8_rgb_round_trips_with_georeferencing() {
        let (w, h) = (37, 29);
        let pixels: Vec<u8> = (0..w * h * 3)
            .map(|i| u8::try_from(i * 7 % 251).unwrap())
            .collect();
        let fmt = RasterFormat {
            width: w,
            height: h,
            bands: 3,
            dtype: DType::U8,
        };
        let georef = Georef {
            epsg: 32633,
            geographic: false,
            transform: NORTH_UP,
        };
        let tiff = write_and_open(&pixels, &fmt, &georef, &options(None), "rgb");
        assert_eq!((tiff.width(), tiff.height(), tiff.band_count()), (w, h, 3));
        assert_eq!(tiff.dtype(), DType::U8);
        assert_eq!(tiff.epsg(), Ok(32633));
        assert_eq!(tiff.transform(), Some(NORTH_UP));
        assert_eq!(tiff.nodata(), None);
        assert_eq!(tiff.predictor(), 2);
        assert_eq!(tiff.photometric(), 2);
        assert_eq!(read_all(&tiff), Samples::U8(pixels));
    }

    #[test]
    fn every_dtype_round_trips_through_its_predictor() {
        let (w, h, bands) = (23, 17, 4);
        let n = w * h * bands;
        let cases: Vec<(DType, Vec<u8>, Samples)> = vec![
            {
                let v: Vec<i8> = (0..n)
                    .map(|i| i8::try_from(i % 200).unwrap_or(-5))
                    .collect();
                (DType::I8, bytes_of(&v), Samples::I8(v))
            },
            {
                let v: Vec<u16> = (0..n)
                    .map(|i| u16::try_from(i * 97 % 60_000).unwrap())
                    .collect();
                (DType::U16, bytes_of(&v), Samples::U16(v))
            },
            {
                let v: Vec<i16> = (0..n).map(|i| i16::try_from(i).unwrap() * -3).collect();
                (DType::I16, bytes_of(&v), Samples::I16(v))
            },
            {
                let v: Vec<u32> = (0..n).map(|i| u32::try_from(i).unwrap() * 70_001).collect();
                (DType::U32, bytes_of(&v), Samples::U32(v))
            },
            {
                let v: Vec<i32> = (0..n)
                    .map(|i| -i32::try_from(i).unwrap() * 12_345)
                    .collect();
                (DType::I32, bytes_of(&v), Samples::I32(v))
            },
            {
                let v: Vec<u64> = (0..n).map(|i| u64::try_from(i).unwrap() << 33).collect();
                (DType::U64, bytes_of(&v), Samples::U64(v))
            },
            {
                let v: Vec<i64> = (0..n).map(|i| -(i64::try_from(i).unwrap() << 35)).collect();
                (DType::I64, bytes_of(&v), Samples::I64(v))
            },
            {
                #[allow(clippy::cast_precision_loss)]
                let v: Vec<f32> = (0..n).map(|i| (i as f32).sin() * 1e4 + 0.25).collect();
                (DType::F32, bytes_of(&v), Samples::F32(v))
            },
            {
                #[allow(clippy::cast_precision_loss)]
                let v: Vec<f64> = (0..n).map(|i| (i as f64).cos() * 1e-3).collect();
                (DType::F64, bytes_of(&v), Samples::F64(v))
            },
        ];
        for (dtype, pixels, expected) in cases {
            let fmt = RasterFormat {
                width: w,
                height: h,
                bands,
                dtype,
            };
            let georef = Georef {
                epsg: 32633,
                geographic: false,
                transform: NORTH_UP,
            };
            let tiff = write_and_open(&pixels, &fmt, &georef, &options(None), dtype.name());
            assert_eq!(tiff.dtype(), dtype);
            assert_eq!(tiff.predictor(), predictor_for(dtype), "{}", dtype.name());
            assert_eq!(read_all(&tiff), expected, "{}", dtype.name());
        }
    }

    #[test]
    fn every_compression_level_round_trips() {
        let (w, h) = (31, 19);
        let pixels: Vec<u8> = (0..w * h * 4)
            .map(|i| u8::try_from(i / 3 % 251).unwrap())
            .collect();
        let fmt = RasterFormat {
            width: w,
            height: h,
            bands: 4,
            dtype: DType::U8,
        };
        let georef = Georef {
            epsg: 32633,
            geographic: false,
            transform: NORTH_UP,
        };
        for level in [0, 1, 6, 9] {
            let opts = Options {
                level: Some(level),
                ..options(None)
            };
            let tiff = write_and_open(&pixels, &fmt, &georef, &opts, &format!("lvl{level}"));
            assert_eq!(
                read_all(&tiff),
                Samples::U8(pixels.clone()),
                "level {level}"
            );
        }
    }

    #[test]
    fn float_nan_is_preserved_and_declared_as_nodata() {
        let mut v = vec![1.5f32; 8 * 8 * 2];
        v[5] = f32::NAN;
        v[100] = f32::NEG_INFINITY;
        let fmt = RasterFormat {
            width: 8,
            height: 8,
            bands: 2,
            dtype: DType::F32,
        };
        let georef = Georef {
            epsg: 4326,
            geographic: true,
            transform: [0.001, 0.0, 10.0, 0.0, -0.001, 45.0],
        };
        let tiff = write_and_open(
            &bytes_of(&v),
            &fmt,
            &georef,
            &options(Some(f64::NAN)),
            "nan",
        );
        assert!(tiff.nodata().unwrap().is_nan());
        assert_eq!(tiff.epsg(), Ok(4326));
        let Samples::F32(read) = read_all(&tiff) else {
            panic!()
        };
        assert!(read[5].is_nan());
        assert_eq!(read[100], f32::NEG_INFINITY);
        assert_eq!(read[0], 1.5);
    }

    #[test]
    fn integer_nodata_is_written_as_an_integer() {
        assert_eq!(nodata_text(255.0), "255");
        assert_eq!(nodata_text(-9999.0), "-9999");
        assert_eq!(nodata_text(0.1), "0.1");
        assert_eq!(nodata_text(f64::NAN), "nan");
        assert_eq!(nodata_text(f64::NEG_INFINITY), "-inf");
        let fmt = RasterFormat {
            width: 4,
            height: 4,
            bands: 1,
            dtype: DType::U8,
        };
        let georef = Georef {
            epsg: 3857,
            geographic: false,
            transform: NORTH_UP,
        };
        let tiff = write_and_open(&[255u8; 16], &fmt, &georef, &options(Some(255.0)), "nd");
        assert_eq!(tiff.nodata(), Some(255.0));
    }

    #[test]
    fn rotated_and_south_up_transforms_use_the_model_matrix() {
        let fmt = RasterFormat {
            width: 5,
            height: 5,
            bands: 1,
            dtype: DType::U8,
        };
        for transform in [
            [9.0, 3.0, 100.0, -2.0, -8.0, 200.0],
            [10.0, 0.0, 100.0, 0.0, 10.0, 200.0],
        ] {
            let georef = Georef {
                epsg: 32633,
                geographic: false,
                transform,
            };
            let tiff = write_and_open(&[7u8; 25], &fmt, &georef, &options(None), "rot");
            assert_eq!(tiff.transform(), Some(transform));
        }
    }

    #[test]
    fn many_strips_decode_in_order() {
        // 600 rows of 4096 x 3 bytes: a strip is 85 rows, so there are 8 strips.
        let (w, h) = (4096, 600);
        let pixels: Vec<u8> = (0..w * h * 3)
            .map(|i| u8::try_from((i / 5 + i / 4099) % 256).unwrap())
            .collect();
        let fmt = RasterFormat {
            width: w,
            height: h,
            bands: 3,
            dtype: DType::U8,
        };
        let georef = Georef {
            epsg: 3857,
            geographic: false,
            transform: NORTH_UP,
        };
        let tiff = write_and_open(&pixels, &fmt, &georef, &options(None), "strips");
        assert!(!tiff.tiled());
        assert_eq!(read_all(&tiff), Samples::U8(pixels));
    }

    #[test]
    fn invalid_inputs_are_errors() {
        let fmt = RasterFormat {
            width: 2,
            height: 2,
            bands: 1,
            dtype: DType::U8,
        };
        let good = Georef {
            epsg: 3857,
            geographic: false,
            transform: NORTH_UP,
        };
        let opts = options(None);
        assert!(encode(&[0; 3], &fmt, &good, &opts)
            .unwrap_err()
            .contains("pixel buffer"));
        let bad_epsg = Georef { epsg: 0, ..good };
        assert!(encode(&[0; 4], &fmt, &bad_epsg, &opts)
            .unwrap_err()
            .contains("EPSG"));
        let singular = Georef {
            transform: [1.0, 2.0, 0.0, 2.0, 4.0, 0.0],
            ..good
        };
        assert!(encode(&[0; 4], &fmt, &singular, &opts)
            .unwrap_err()
            .contains("singular"));
        let nan = Georef {
            transform: [f64::NAN, 0.0, 0.0, 0.0, -1.0, 0.0],
            ..good
        };
        assert!(encode(&[0; 4], &fmt, &nan, &opts)
            .unwrap_err()
            .contains("finite"));
        let one_name = vec!["a".to_owned()];
        let named = Options {
            nodata: None,
            band_names: &one_name,
            level: Some(3),
        };
        assert!(
            encode(&[0; 8], &RasterFormat { bands: 2, ..fmt }, &good, &named)
                .unwrap_err()
                .contains("band names")
        );
        assert!(dtype_from_name("float16").is_err());
        let level = Options {
            level: Some(10),
            ..opts
        };
        assert!(encode(&[0; 4], &fmt, &good, &level).is_err());
    }

    #[test]
    fn write_all_rejects_path_like_names() {
        let fmt = RasterFormat {
            width: 2,
            height: 2,
            bands: 1,
            dtype: DType::U8,
        };
        let georef = Georef {
            epsg: 3857,
            geographic: false,
            transform: NORTH_UP,
        };
        let dir = temp_dir("names");
        for name in ["../x.tif", "a/b.tif", ""] {
            let err = write_all(
                &[0; 4],
                &fmt,
                &options(None),
                &dir,
                &[(name.to_owned(), georef)],
            )
            .unwrap_err();
            assert!(err.contains("plain file name"), "{name}: {err}");
        }
        let files = [("a.tif".to_owned(), georef), ("b.tif".to_owned(), georef)];
        write_all(&[1; 8], &fmt, &options(None), &dir, &files).unwrap();
        assert!(dir.join("a.tif").exists() && dir.join("b.tif").exists());
        assert!(write_all(&[1; 7], &fmt, &options(None), &dir, &files).is_err());
    }
}
