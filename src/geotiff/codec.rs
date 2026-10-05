//! Decoding of one TIFF tile or strip: decompression, predictor and byte order.
//!
//! Each chunk is decoded independently from its raw bytes, so the window
//! reader can decode the chunks under a window in parallel.

use super::ifd::ByteOrder;
use super::GeoTiffError;
use std::io::Read;

type Result<T> = std::result::Result<T, GeoTiffError>;

/// TIFF `Compression` codes the reader decodes.
pub mod compression {
    /// No compression.
    pub const NONE: u16 = 1;
    /// LZW.
    pub const LZW: u16 = 5;
    /// JPEG ("new style", TIFF 6.0 technote 2).
    pub const JPEG: u16 = 7;
    /// Adobe Deflate (zlib).
    pub const DEFLATE: u16 = 8;
    /// `PackBits` run-length encoding.
    pub const PACKBITS: u16 = 32773;
    /// The original, non-registered Deflate code (zlib).
    pub const OLD_DEFLATE: u16 = 32946;
    /// Zstandard.
    pub const ZSTD: u16 = 50000;
    /// `WebP`.
    pub const WEBP: u16 = 50001;
}

/// Human-readable name of a TIFF compression code, for metadata and errors.
#[must_use]
pub fn compression_name(code: u16) -> String {
    match code {
        1 => "none",
        2 => "CCITT RLE",
        3 => "CCITT Group 3 fax",
        4 => "CCITT Group 4 fax",
        5 => "LZW",
        6 => "old-style JPEG",
        7 => "JPEG",
        8 | 32946 => "Deflate",
        32773 => "PackBits",
        32809 => "ThunderScan",
        32908 | 32909 => "PixarLog",
        34661 => "JBIG",
        34676 => "SGILog",
        34677 => "SGILog24",
        34712 => "JPEG 2000",
        34887 => "LERC",
        34925 => "LZMA",
        50000 => "ZSTD",
        50001 => "WebP",
        50002 | 52546 => "JPEG XL",
        _ => return format!("unknown ({code})"),
    }
    .to_owned()
}

/// Check that chunks with this compression can be decoded.
///
/// # Errors
/// Returns [`GeoTiffError::Invalid`] naming the codec when it is not supported,
/// or when JPEG/WebP is combined with a sample size other than 8 bits.
pub fn check_supported(code: u16, bits: u16) -> Result<()> {
    use compression::{DEFLATE, JPEG, LZW, NONE, OLD_DEFLATE, PACKBITS, WEBP, ZSTD};
    match code {
        NONE | LZW | DEFLATE | OLD_DEFLATE | PACKBITS | ZSTD => Ok(()),
        JPEG | WEBP if bits == 8 => Ok(()),
        JPEG | WEBP => Err(GeoTiffError::Invalid(format!(
            "{} compression with {bits}-bit samples is not supported",
            compression_name(code)
        ))),
        _ => Err(GeoTiffError::Invalid(format!(
            "{} compression (TIFF compression code {code}) is not supported; supported codecs \
             are none, Deflate, LZW, PackBits, ZSTD, JPEG and WebP (re-encode the file, e.g. \
             `gdal_translate -co COMPRESS=DEFLATE`)",
            compression_name(code)
        ))),
    }
}

/// Everything needed to decode one chunk.
#[derive(Clone, Copy)]
pub struct ChunkFormat<'a> {
    /// TIFF compression code.
    pub compression: u16,
    /// TIFF predictor (1 none, 2 horizontal, 3 floating point).
    pub predictor: u16,
    /// Byte order of the file.
    pub order: ByteOrder,
    /// Bytes per sample (1, 2, 4 or 8).
    pub sample_bytes: usize,
    /// Samples per pixel stored in the chunk (1 for band-interleaved files).
    pub samples: usize,
    /// Width of the chunk in pixels (the tile width, or the image width for strips).
    pub width: usize,
    /// TIFF photometric interpretation (6 = `YCbCr` is converted to RGB for JPEG).
    pub photometric: u16,
    /// Contents of the `JPEGTables` tag, if any.
    pub jpeg_tables: Option<&'a [u8]>,
}

/// Decode `rows` rows of a chunk from its raw bytes into native-endian samples
/// laid out row by row, pixel by pixel, sample by sample.
///
/// # Errors
/// Returns [`GeoTiffError::Invalid`] when the data is corrupt or decodes to
/// fewer bytes than the chunk needs.
pub fn decode_chunk(fmt: &ChunkFormat<'_>, raw: &[u8], rows: usize) -> Result<Vec<u8>> {
    let row_bytes = fmt.width * fmt.samples * fmt.sample_bytes;
    let expected = row_bytes * rows;
    let mut out = match fmt.compression {
        compression::NONE => {
            if raw.len() < expected {
                return Err(corrupt(&format!(
                    "uncompressed chunk has {} bytes, expected {expected}",
                    raw.len()
                )));
            }
            raw[..expected].to_vec()
        }
        compression::LZW => lzw(raw, expected)?,
        compression::DEFLATE | compression::OLD_DEFLATE => deflate(raw, expected)?,
        compression::ZSTD => zstd(raw, expected)?,
        compression::PACKBITS => packbits(raw, expected)?,
        compression::JPEG => return jpeg(fmt, raw, rows),
        compression::WEBP => return webp(fmt, raw, rows),
        code => {
            check_supported(code, 8)?;
            unreachable!("check_supported rejects unknown codecs")
        }
    };
    let predicted = matches!(
        fmt.compression,
        compression::LZW | compression::DEFLATE | compression::OLD_DEFLATE | compression::ZSTD
    );
    match (predicted, fmt.predictor) {
        (true, 2) => {
            to_native(&mut out, fmt.order, fmt.sample_bytes);
            for row in out.chunks_exact_mut(row_bytes) {
                horizontal_accumulate(row, fmt.sample_bytes, fmt.samples);
            }
        }
        (true, 3) => {
            let mut scratch = vec![0u8; row_bytes];
            for row in out.chunks_exact_mut(row_bytes) {
                floating_point_accumulate(row, &mut scratch, fmt.sample_bytes, fmt.samples);
            }
        }
        _ => to_native(&mut out, fmt.order, fmt.sample_bytes),
    }
    Ok(out)
}

fn corrupt(message: &str) -> GeoTiffError {
    GeoTiffError::Invalid(format!("corrupt TIFF chunk: {message}"))
}

fn short(codec: &str, got: usize, expected: usize) -> GeoTiffError {
    corrupt(&format!(
        "{codec} data decompressed to {got} bytes, expected {expected}"
    ))
}

fn lzw(raw: &[u8], expected: usize) -> Result<Vec<u8>> {
    if raw.len() >= 2 && raw[0] == 0 && raw[1] & 1 == 1 {
        return Err(GeoTiffError::Invalid(
            "old-style (pre-TIFF 6.0) LZW compression is not supported".to_owned(),
        ));
    }
    let mut out = vec![0u8; expected];
    let mut decoder = weezl::decode::Decoder::with_tiff_size_switch(weezl::BitOrder::Msb, 8);
    let (mut read, mut written) = (0, 0);
    while written < expected {
        let result = decoder.decode_bytes(&raw[read..], &mut out[written..]);
        read += result.consumed_in;
        written += result.consumed_out;
        match result.status {
            Ok(weezl::LzwStatus::Ok) if result.consumed_in + result.consumed_out > 0 => {}
            Ok(_) => break,
            Err(e) => return Err(corrupt(&format!("LZW: {e}"))),
        }
    }
    if written < expected {
        return Err(short("LZW", written, expected));
    }
    Ok(out)
}

fn deflate(raw: &[u8], expected: usize) -> Result<Vec<u8>> {
    let mut out = vec![0u8; expected];
    let mut inflater = flate2::Decompress::new(true);
    inflater
        .decompress(raw, &mut out, flate2::FlushDecompress::Finish)
        .map_err(|e| corrupt(&format!("Deflate: {e}")))?;
    let written = usize::try_from(inflater.total_out()).unwrap_or(usize::MAX);
    if written < expected {
        return Err(short("Deflate", written, expected));
    }
    Ok(out)
}

fn zstd(raw: &[u8], expected: usize) -> Result<Vec<u8>> {
    let mut out = vec![0u8; expected];
    let mut decoder = zstd::stream::read::Decoder::with_buffer(raw)
        .map_err(|e| corrupt(&format!("ZSTD: {e}")))?;
    let mut written = 0;
    while written < expected {
        match decoder.read(&mut out[written..]) {
            Ok(0) => break,
            Ok(n) => written += n,
            Err(e) if e.kind() == std::io::ErrorKind::Interrupted => {}
            Err(e) => return Err(corrupt(&format!("ZSTD: {e}"))),
        }
    }
    if written < expected {
        return Err(short("ZSTD", written, expected));
    }
    Ok(out)
}

fn packbits(raw: &[u8], expected: usize) -> Result<Vec<u8>> {
    let mut out = Vec::with_capacity(expected);
    let mut i = 0;
    while out.len() < expected && i < raw.len() {
        let n = raw[i].cast_signed();
        i += 1;
        if n >= 0 {
            let len = usize::from(n.unsigned_abs()) + 1;
            let end = (i + len).min(raw.len());
            out.extend_from_slice(&raw[i..end]);
            i = end;
        } else if n != -128 {
            let Some(&value) = raw.get(i) else { break };
            i += 1;
            let len = usize::from(n.unsigned_abs()) + 1;
            out.resize(out.len() + len, value);
        }
    }
    if out.len() < expected {
        return Err(short("PackBits", out.len(), expected));
    }
    out.truncate(expected);
    Ok(out)
}

fn jpeg(fmt: &ChunkFormat<'_>, raw: &[u8], rows: usize) -> Result<Vec<u8>> {
    use zune_jpeg::zune_core::bytestream::ZCursor;
    use zune_jpeg::zune_core::colorspace::ColorSpace;
    use zune_jpeg::zune_core::options::DecoderOptions;

    // JPEGTables holds a tables-only JPEG stream (SOI tables EOI) shared by all
    // chunks; splice it in front of the chunk's own stream without the
    // tables' EOI and the chunk's SOI.
    let stream = match fmt.jpeg_tables {
        Some(tables) if tables.len() > 4 => {
            if !tables.starts_with(&[0xFF, 0xD8]) || !raw.starts_with(&[0xFF, 0xD8]) {
                return Err(corrupt("JPEG data does not start with an SOI marker"));
            }
            let mut joined = Vec::with_capacity(tables.len() + raw.len());
            joined.extend_from_slice(&tables[..tables.len() - 2]);
            joined.extend_from_slice(&raw[2..]);
            joined
        }
        _ => raw.to_vec(),
    };
    let options = DecoderOptions::default()
        .set_max_width(1 << 20)
        .set_max_height(1 << 20);
    let mut decoder = zune_jpeg::JpegDecoder::new_with_options(ZCursor::new(stream), options);
    decoder
        .decode_headers()
        .map_err(|e| corrupt(&format!("JPEG: {e:?}")))?;
    let input = decoder
        .input_colorspace()
        .ok_or_else(|| corrupt("JPEG: no colour space"))?;
    // As libtiff does for GDAL: YCbCr is converted to RGB, every other
    // photometric interpretation gets the stored components unchanged.
    let output = if fmt.photometric == 6 {
        ColorSpace::RGB
    } else {
        input
    };
    decoder.set_options(options.jpeg_set_out_colorspace(output));
    let pixels = decoder
        .decode()
        .map_err(|e| corrupt(&format!("JPEG: {e:?}")))?;
    let info = decoder
        .info()
        .ok_or_else(|| corrupt("JPEG: no image information"))?;
    let (width, height) = (usize::from(info.width), usize::from(info.height));
    let components = output.num_components();
    if components != fmt.samples {
        return Err(corrupt(&format!(
            "JPEG has {components} components but the TIFF declares {} samples per pixel",
            fmt.samples
        )));
    }
    if width != fmt.width || height < rows || pixels.len() < width * height * components {
        return Err(corrupt(&format!(
            "JPEG is {width}x{height}, expected {}x{rows}",
            fmt.width
        )));
    }
    let mut out = pixels;
    out.truncate(width * rows * components);
    Ok(out)
}

fn webp(fmt: &ChunkFormat<'_>, raw: &[u8], rows: usize) -> Result<Vec<u8>> {
    let mut decoder = image_webp::WebPDecoder::new(std::io::Cursor::new(raw))
        .map_err(|e| corrupt(&format!("WebP: {e}")))?;
    let (w, h) = decoder.dimensions();
    let (width, height) = (w as usize, h as usize);
    if width != fmt.width || height < rows {
        return Err(corrupt(&format!(
            "WebP is {width}x{height}, expected {}x{rows}",
            fmt.width
        )));
    }
    let channels = if decoder.has_alpha() { 4 } else { 3 };
    let mut pixels = vec![0u8; width * height * channels];
    decoder
        .read_image(&mut pixels)
        .map_err(|e| corrupt(&format!("WebP: {e}")))?;
    pixels.truncate(width * rows * channels);
    match (channels, fmt.samples) {
        (c, s) if c == s => Ok(pixels),
        // An opaque image stored without alpha: alpha is 255 everywhere.
        (3, 4) => Ok(pixels
            .as_chunks::<3>()
            .0
            .iter()
            .flat_map(|p| [p[0], p[1], p[2], 255])
            .collect()),
        (c, s) => Err(corrupt(&format!(
            "WebP has {c} channels but the TIFF declares {s} samples per pixel"
        ))),
    }
}

/// Convert samples from the file byte order to the native one.
fn to_native(buf: &mut [u8], order: ByteOrder, sample_bytes: usize) {
    let native = if cfg!(target_endian = "little") {
        ByteOrder::Little
    } else {
        ByteOrder::Big
    };
    if order != native && sample_bytes > 1 {
        for sample in buf.chunks_exact_mut(sample_bytes) {
            sample.reverse();
        }
    }
}

/// Undo TIFF predictor 2 on one row of native-endian samples: each sample is
/// stored as the difference from the same band of the previous pixel.
fn horizontal_accumulate(row: &mut [u8], sample_bytes: usize, samples: usize) {
    macro_rules! accumulate {
        ($t:ty) => {{
            const N: usize = std::mem::size_of::<$t>();
            let lag = samples * N;
            for i in (lag..row.len()).step_by(N) {
                let prev = <$t>::from_ne_bytes(row[i - lag..i - lag + N].try_into().unwrap());
                let cur = <$t>::from_ne_bytes(row[i..i + N].try_into().unwrap());
                row[i..i + N].copy_from_slice(&cur.wrapping_add(prev).to_ne_bytes());
            }
        }};
    }
    match sample_bytes {
        1 => {
            for i in samples..row.len() {
                row[i] = row[i].wrapping_add(row[i - samples]);
            }
        }
        2 => accumulate!(u16),
        4 => accumulate!(u32),
        _ => accumulate!(u64),
    }
}

/// Undo TIFF predictor 3 on one row: the bytes were differenced with a lag of
/// one pixel after being split into planes, most significant byte first
/// (independent of the file's byte order). Output is native-endian.
fn floating_point_accumulate(
    row: &mut [u8],
    scratch: &mut [u8],
    sample_bytes: usize,
    samples: usize,
) {
    for i in samples..row.len() {
        row[i] = row[i].wrapping_add(row[i - samples]);
    }
    scratch.copy_from_slice(row);
    let count = row.len() / sample_bytes;
    for (i, sample) in row.chunks_exact_mut(sample_bytes).enumerate() {
        for (byte, value) in sample.iter_mut().enumerate() {
            // Plane 0 holds the most significant bytes.
            let plane = if cfg!(target_endian = "little") {
                sample_bytes - 1 - byte
            } else {
                byte
            };
            *value = scratch[plane * count + i];
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn fmt(
        compression: u16,
        predictor: u16,
        sample_bytes: usize,
        samples: usize,
    ) -> ChunkFormat<'static> {
        ChunkFormat {
            compression,
            predictor,
            order: ByteOrder::Big,
            sample_bytes,
            samples,
            width: 3,
            photometric: 1,
            jpeg_tables: None,
        }
    }

    #[test]
    fn packbits_matches_the_tiff_spec_example() {
        // TIFF 6.0 spec, section 9.
        let packed = [
            0xFE, 0xAA, 0x02, 0x80, 0x00, 0x2A, 0xFD, 0xAA, 0x03, 0x80, 0x00, 0x2A, 0x22, 0xF7,
            0xAA,
        ];
        let unpacked = [
            0xAA, 0xAA, 0xAA, 0x80, 0x00, 0x2A, 0xAA, 0xAA, 0xAA, 0xAA, 0x80, 0x00, 0x2A, 0x22,
            0xAA, 0xAA, 0xAA, 0xAA, 0xAA, 0xAA, 0xAA, 0xAA, 0xAA, 0xAA,
        ];
        assert_eq!(packbits(&packed, unpacked.len()).unwrap(), unpacked);
        assert!(packbits(&packed, unpacked.len() + 1).is_err());
    }

    #[test]
    fn big_endian_u16_with_horizontal_predictor() {
        // One row of 3 pixels x 2 bands: band 0 = 1, 3, 6; band 1 = 300, 200, 100.
        let diffs: [i16; 6] = [1, 300, 2, -100, 3, -100];
        let mut raw = Vec::new();
        for d in diffs {
            raw.extend(d.cast_unsigned().to_be_bytes());
        }
        let mut f = fmt(compression::NONE, 1, 2, 2);
        // Predictor is only applied with a compressing codec; use Deflate.
        f.compression = compression::DEFLATE;
        f.predictor = 2;
        let mut z = flate2::write::ZlibEncoder::new(Vec::new(), flate2::Compression::default());
        std::io::Write::write_all(&mut z, &raw).unwrap();
        let out = decode_chunk(&f, &z.finish().unwrap(), 1).unwrap();
        let values: Vec<u16> = out
            .as_chunks::<2>()
            .0
            .iter()
            .map(|b| u16::from_ne_bytes(*b))
            .collect();
        assert_eq!(values, [1, 300, 3, 200, 6, 100]);
    }

    #[test]
    fn floating_point_predictor_round_trip() {
        let values = [1.5f32, -2.25, 1e30];
        // Encode as libtiff does: big-endian byte planes, then byte differences.
        let count = values.len();
        let mut planes = vec![0u8; count * 4];
        for (i, v) in values.iter().enumerate() {
            for (b, byte) in v.to_be_bytes().iter().enumerate() {
                planes[b * count + i] = *byte;
            }
        }
        for i in (1..planes.len()).rev() {
            planes[i] = planes[i].wrapping_sub(planes[i - 1]);
        }
        let mut row = planes.clone();
        let mut scratch = vec![0u8; row.len()];
        floating_point_accumulate(&mut row, &mut scratch, 4, 1);
        let decoded: Vec<f32> = row
            .as_chunks::<4>()
            .0
            .iter()
            .map(|b| f32::from_ne_bytes(*b))
            .collect();
        assert_eq!(decoded, values);
    }

    #[test]
    fn unsupported_codecs_are_named() {
        let err = check_supported(34887, 8).unwrap_err().to_string();
        assert!(err.contains("LERC"), "{err}");
        let err = check_supported(34712, 16).unwrap_err().to_string();
        assert!(err.contains("JPEG 2000"), "{err}");
        let err = check_supported(7, 16).unwrap_err().to_string();
        assert!(err.contains("JPEG compression with 16-bit"), "{err}");
    }

    #[test]
    fn short_streams_are_errors() {
        let f = fmt(compression::NONE, 1, 1, 1);
        assert!(decode_chunk(&f, &[1, 2], 1).is_err());
    }
}
