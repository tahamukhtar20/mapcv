//! TIFF and BigTIFF container parsing: header, IFD chain and tag values.
//!
//! Only the tags the GeoTIFF reader uses are loaded; the values of all other
//! tags (XMP, ICC profiles, GDAL metadata, ...) are never read, which keeps
//! opening a remote file to a few small range requests.

use super::source::ByteSource;
use super::GeoTiffError;
use std::collections::{BTreeMap, HashSet};

type Result<T> = std::result::Result<T, GeoTiffError>;

/// Byte order of a TIFF file.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ByteOrder {
    /// `II`: least significant byte first.
    Little,
    /// `MM`: most significant byte first.
    Big,
}

/// Tag numbers used by the reader.
pub mod tag {
    /// `NewSubfileType`: bit 0 = reduced resolution, bit 2 = transparency mask.
    pub const NEW_SUBFILE_TYPE: u16 = 254;
    /// `ImageWidth`.
    pub const IMAGE_WIDTH: u16 = 256;
    /// `ImageLength`.
    pub const IMAGE_LENGTH: u16 = 257;
    /// `BitsPerSample`.
    pub const BITS_PER_SAMPLE: u16 = 258;
    /// `Compression`.
    pub const COMPRESSION: u16 = 259;
    /// `PhotometricInterpretation`.
    pub const PHOTOMETRIC: u16 = 262;
    /// `StripOffsets`.
    pub const STRIP_OFFSETS: u16 = 273;
    /// `SamplesPerPixel`.
    pub const SAMPLES_PER_PIXEL: u16 = 277;
    /// `RowsPerStrip`.
    pub const ROWS_PER_STRIP: u16 = 278;
    /// `StripByteCounts`.
    pub const STRIP_BYTE_COUNTS: u16 = 279;
    /// `PlanarConfiguration`: 1 = pixel interleaved, 2 = band interleaved.
    pub const PLANAR_CONFIGURATION: u16 = 284;
    /// `Predictor`.
    pub const PREDICTOR: u16 = 317;
    /// `TileWidth`.
    pub const TILE_WIDTH: u16 = 322;
    /// `TileLength`.
    pub const TILE_LENGTH: u16 = 323;
    /// `TileOffsets`.
    pub const TILE_OFFSETS: u16 = 324;
    /// `TileByteCounts`.
    pub const TILE_BYTE_COUNTS: u16 = 325;
    /// `SampleFormat`: 1 = unsigned, 2 = signed, 3 = IEEE float.
    pub const SAMPLE_FORMAT: u16 = 339;
    /// `JPEGTables`: quantisation and Huffman tables shared by all JPEG chunks.
    pub const JPEG_TABLES: u16 = 347;
    /// `ModelPixelScaleTag`.
    pub const MODEL_PIXEL_SCALE: u16 = 33550;
    /// `ModelTiepointTag`.
    pub const MODEL_TIEPOINT: u16 = 33922;
    /// `ModelTransformationTag`.
    pub const MODEL_TRANSFORMATION: u16 = 34264;
    /// `GeoKeyDirectoryTag`.
    pub const GEO_KEY_DIRECTORY: u16 = 34735;
    /// `GeoDoubleParamsTag`.
    pub const GEO_DOUBLE_PARAMS: u16 = 34736;
    /// `GeoAsciiParamsTag`.
    pub const GEO_ASCII_PARAMS: u16 = 34737;
    /// `GDAL_NODATA` (ASCII).
    pub const GDAL_NODATA: u16 = 42113;

    /// Tags whose values are loaded; all others are skipped.
    pub const LOADED: &[u16] = &[
        NEW_SUBFILE_TYPE,
        IMAGE_WIDTH,
        IMAGE_LENGTH,
        BITS_PER_SAMPLE,
        COMPRESSION,
        PHOTOMETRIC,
        STRIP_OFFSETS,
        SAMPLES_PER_PIXEL,
        ROWS_PER_STRIP,
        STRIP_BYTE_COUNTS,
        PLANAR_CONFIGURATION,
        PREDICTOR,
        TILE_WIDTH,
        TILE_LENGTH,
        TILE_OFFSETS,
        TILE_BYTE_COUNTS,
        SAMPLE_FORMAT,
        JPEG_TABLES,
        MODEL_PIXEL_SCALE,
        MODEL_TIEPOINT,
        MODEL_TRANSFORMATION,
        GEO_KEY_DIRECTORY,
        GEO_DOUBLE_PARAMS,
        GEO_ASCII_PARAMS,
        GDAL_NODATA,
    ];
}

/// Most IFDs followed in one file (COGs have one per overview and mask).
const MAX_IFDS: usize = 1024;
/// Most entries accepted in one IFD.
const MAX_ENTRIES: usize = 4096;
/// Largest tag value loaded (64 MiB holds 8 million 64-bit offsets).
const MAX_TAG_BYTES: usize = 64 * 1024 * 1024;
/// Out-of-line tag values of a file may hold this much more than the file itself:
/// values of distinct tags occupy distinct bytes of the file, and only small values
/// (`BitsPerSample` arrays, `JPEGTables`) are sometimes shared between IFDs.
const SHARED_TAG_BYTES: u64 = 16 * 1024 * 1024;

/// The most values a tag may hold when its size is set by the format rather than by
/// the image: one per band (at most 65535), or a GeoKey directory (a 4-value header and
/// at most 65535 4-value keys). `None` for the tags whose size grows with the image.
fn max_values(tag_id: u16) -> Option<u64> {
    match tag_id {
        tag::BITS_PER_SAMPLE | tag::SAMPLE_FORMAT => Some(65_535),
        tag::GEO_KEY_DIRECTORY => Some(4 + 4 * 65_535),
        _ => None,
    }
}

/// The raw value of one tag, in file byte order.
#[derive(Clone, Debug)]
struct Entry {
    field_type: u16,
    count: u64,
    data: Vec<u8>,
}

/// One image file directory with the values of the loaded tags.
#[derive(Clone, Debug)]
pub struct Ifd {
    order: ByteOrder,
    entries: BTreeMap<u16, Entry>,
}

/// The unsigned integer values of one tag (BYTE, SHORT, LONG, LONG8 or IFD types),
/// kept as stored in the file and decoded one at a time: a chunk offset array decodes
/// to up to eight times its size in the file.
#[derive(Clone, Debug)]
pub struct UintArray {
    order: ByteOrder,
    width: usize,
    data: Vec<u8>,
}

impl UintArray {
    /// Number of values.
    #[must_use]
    pub fn len(&self) -> usize {
        self.data.len() / self.width
    }

    /// Whether there are no values.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.data.len() < self.width
    }

    /// The value at `index`, `None` past the end.
    #[must_use]
    pub fn get(&self, index: usize) -> Option<u64> {
        let start = index.checked_mul(self.width)?;
        let b = self.data.get(start..start.checked_add(self.width)?)?;
        Some(decode_uint(self.order, self.width, b))
    }

    /// Every value, in order.
    pub fn iter(&self) -> impl Iterator<Item = u64> + '_ {
        (0..self.len()).filter_map(|i| self.get(i))
    }
}

/// Bytes per value of an unsigned integer tag, or an error for another field type.
fn uint_width(tag: u16, field_type: u16) -> Result<usize> {
    match field_type {
        1 | 7 => Ok(1),
        3 => Ok(2),
        4 | 13 => Ok(4),
        16 | 18 => Ok(8),
        t => Err(GeoTiffError::Invalid(format!(
            "TIFF tag {tag} has field type {t}, expected an unsigned integer type"
        ))),
    }
}

/// One unsigned value of `width` bytes (1, 2, 4 or 8) at the start of `b`.
fn decode_uint(order: ByteOrder, width: usize, b: &[u8]) -> u64 {
    match width {
        1 => u64::from(b[0]),
        2 => u64::from(u16_at(order, b)),
        4 => u64::from(u32_at(order, b)),
        _ => u64_at(order, b),
    }
}

/// Size in bytes of one value of a TIFF field type, `None` for unknown types.
fn type_size(field_type: u16) -> Option<usize> {
    match field_type {
        1 | 2 | 6 | 7 => Some(1),              // BYTE, ASCII, SBYTE, UNDEFINED
        3 | 8 => Some(2),                      // SHORT, SSHORT
        4 | 9 | 11 | 13 => Some(4),            // LONG, SLONG, FLOAT, IFD
        5 | 10 | 12 | 16 | 17 | 18 => Some(8), // RATIONAL, SRATIONAL, DOUBLE, LONG8, SLONG8, IFD8
        _ => None,
    }
}

fn u16_at(order: ByteOrder, b: &[u8]) -> u16 {
    let a = [b[0], b[1]];
    match order {
        ByteOrder::Little => u16::from_le_bytes(a),
        ByteOrder::Big => u16::from_be_bytes(a),
    }
}

fn u32_at(order: ByteOrder, b: &[u8]) -> u32 {
    let a = [b[0], b[1], b[2], b[3]];
    match order {
        ByteOrder::Little => u32::from_le_bytes(a),
        ByteOrder::Big => u32::from_be_bytes(a),
    }
}

fn u64_at(order: ByteOrder, b: &[u8]) -> u64 {
    let a = [b[0], b[1], b[2], b[3], b[4], b[5], b[6], b[7]];
    match order {
        ByteOrder::Little => u64::from_le_bytes(a),
        ByteOrder::Big => u64::from_be_bytes(a),
    }
}

impl Ifd {
    /// Whether the tag is present (and was loaded).
    #[must_use]
    pub fn has(&self, tag: u16) -> bool {
        self.entries.contains_key(&tag)
    }

    /// The tag's values as unsigned integers (BYTE, SHORT, LONG, LONG8, IFD types).
    ///
    /// # Errors
    /// Returns [`GeoTiffError::Invalid`] when the tag has a non-integer type.
    pub fn uint_array(&self, tag: u16) -> Result<Option<UintArray>> {
        let Some(entry) = self.entries.get(&tag) else {
            return Ok(None);
        };
        Ok(Some(UintArray {
            order: self.order,
            width: uint_width(tag, entry.field_type)?,
            data: entry.data.clone(),
        }))
    }

    /// The tag's values as unsigned integers, decoded into a vector.
    ///
    /// # Errors
    /// As [`Ifd::uint_array`].
    pub fn uints(&self, tag: u16) -> Result<Option<Vec<u64>>> {
        Ok(self.uint_array(tag)?.map(|values| values.iter().collect()))
    }

    /// The tag's first unsigned value, or `default` when absent.
    ///
    /// # Errors
    /// Returns [`GeoTiffError::Invalid`] for a non-integer type or an empty value.
    pub fn uint(&self, tag: u16, default: Option<u64>) -> Result<Option<u64>> {
        let Some(entry) = self.entries.get(&tag) else {
            return Ok(default);
        };
        // Decode only the first value, not the whole tag.
        let width = uint_width(tag, entry.field_type)?;
        entry
            .data
            .get(..width)
            .map(|b| Some(decode_uint(self.order, width, b)))
            .ok_or_else(|| GeoTiffError::Invalid(format!("TIFF tag {tag} has no value")))
    }

    /// The tag's values as doubles (DOUBLE or FLOAT types).
    ///
    /// # Errors
    /// Returns [`GeoTiffError::Invalid`] when the tag has another type.
    pub fn doubles(&self, tag: u16) -> Result<Option<Vec<f64>>> {
        let Some(entry) = self.entries.get(&tag) else {
            return Ok(None);
        };
        let o = self.order;
        let values = match entry.field_type {
            12 => entry
                .data
                .as_chunks::<8>()
                .0
                .iter()
                .map(|b| f64::from_bits(u64_at(o, b)))
                .collect(),
            11 => entry
                .data
                .as_chunks::<4>()
                .0
                .iter()
                .map(|b| f64::from(f32::from_bits(u32_at(o, b))))
                .collect(),
            t => {
                return Err(GeoTiffError::Invalid(format!(
                    "TIFF tag {tag} has field type {t}, expected DOUBLE"
                )))
            }
        };
        Ok(Some(values))
    }

    /// The tag's raw bytes (for BYTE, ASCII and UNDEFINED tags).
    #[must_use]
    pub fn bytes(&self, tag: u16) -> Option<&[u8]> {
        self.entries.get(&tag).map(|e| e.data.as_slice())
    }

    /// The tag's value as text, up to the first NUL.
    #[must_use]
    pub fn ascii(&self, tag: u16) -> Option<String> {
        let data = self.bytes(tag)?;
        let end = data.iter().position(|&b| b == 0).unwrap_or(data.len());
        Some(String::from_utf8_lossy(&data[..end]).into_owned())
    }

    /// Number of values of the tag.
    #[must_use]
    pub fn count(&self, tag: u16) -> Option<u64> {
        self.entries.get(&tag).map(|e| e.count)
    }
}

/// The parsed TIFF header and every IFD of the main chain, in file order.
#[derive(Debug)]
pub struct TiffFile {
    /// Byte order of all multi-byte values in the file.
    pub order: ByteOrder,
    /// Whether this is a BigTIFF (64-bit offsets).
    pub bigtiff: bool,
    /// The IFDs of the main chain.
    pub ifds: Vec<Ifd>,
}

/// Read the header and the IFD chain of a TIFF or BigTIFF file.
///
/// # Errors
/// Returns [`GeoTiffError::Invalid`] when the file is not a TIFF or its
/// directory structure is corrupt, and [`GeoTiffError::Io`] when reading fails.
pub fn read_tiff(source: &dyn ByteSource) -> Result<TiffFile> {
    let name = source.describe();
    if source.size() < 8 {
        return Err(GeoTiffError::Invalid(format!(
            "{name} is not a TIFF file (too short)"
        )));
    }
    let header = source.read_at(0, 16.min(usize::try_from(source.size()).unwrap_or(16)))?;
    let order = match &header[..2] {
        b"II" => ByteOrder::Little,
        b"MM" => ByteOrder::Big,
        _ => {
            return Err(GeoTiffError::Invalid(format!(
                "{name} is not a TIFF file (no II/MM byte-order mark)"
            )))
        }
    };
    let (bigtiff, mut next) = match u16_at(order, &header[2..]) {
        42 => (false, u64::from(u32_at(order, &header[4..]))),
        43 => {
            if header.len() < 16 || u16_at(order, &header[4..]) != 8 {
                return Err(GeoTiffError::Invalid(format!(
                    "{name} has a corrupt BigTIFF header"
                )));
            }
            (true, u64_at(order, &header[8..]))
        }
        magic => {
            return Err(GeoTiffError::Invalid(format!(
                "{name} is not a TIFF file (version {magic}, expected 42 or 43)"
            )))
        }
    };
    let mut ifds = Vec::new();
    let mut seen = HashSet::new();
    // Bytes of out-of-line tag values still allowed for the rest of the file.
    let mut tag_budget = source.size().saturating_add(SHARED_TAG_BYTES);
    while next != 0 {
        if !seen.insert(next) {
            return Err(GeoTiffError::Invalid(format!(
                "{name} has a loop in its IFD chain"
            )));
        }
        if ifds.len() >= MAX_IFDS {
            return Err(GeoTiffError::Invalid(format!(
                "{name} has more than {MAX_IFDS} IFDs"
            )));
        }
        let (ifd, following) = read_ifd(source, order, bigtiff, next, &mut tag_budget)?;
        ifds.push(ifd);
        next = following;
    }
    if ifds.is_empty() {
        return Err(GeoTiffError::Invalid(format!("{name} contains no image")));
    }
    Ok(TiffFile {
        order,
        bigtiff,
        ifds,
    })
}

/// Read the IFD at `offset`; returns it and the offset of the next IFD (0 at the end).
///
/// Its out-of-line values are taken from `tag_budget` before any is read, so that a
/// file cannot make the reader hold more tag data than the file is long.
fn read_ifd(
    source: &dyn ByteSource,
    order: ByteOrder,
    bigtiff: bool,
    offset: u64,
    tag_budget: &mut u64,
) -> Result<(Ifd, u64)> {
    let name = source.describe();
    // Sizes of the entry count, one entry, and an inline value or offset.
    let (count_size, entry_size, offset_size): (usize, usize, usize) =
        if bigtiff { (8, 20, 8) } else { (2, 12, 4) };
    let read_offset = |b: &[u8]| {
        if bigtiff {
            u64_at(order, b)
        } else {
            u64::from(u32_at(order, b))
        }
    };
    let count_bytes = source.read_at(offset, count_size)?;
    let n = if bigtiff {
        u64_at(order, &count_bytes)
    } else {
        u64::from(u16_at(order, &count_bytes))
    };
    let n = usize::try_from(n)
        .ok()
        .filter(|&n| n <= MAX_ENTRIES)
        .ok_or_else(|| {
            GeoTiffError::Invalid(format!(
                "{name} has an IFD with {n} entries (corrupt file?)"
            ))
        })?;
    // The entry table followed by the offset of the next IFD.
    let table_len = n * entry_size + offset_size;
    let table = source.read_at(offset + count_size as u64, table_len)?;
    let next = read_offset(&table[table_len - offset_size..]);

    let mut entries = BTreeMap::new();
    let mut deferred: Vec<(u16, u16, u64, u64, usize)> = Vec::new();
    let mut listed: Vec<u16> = Vec::new();
    for raw in table[..n * entry_size].chunks_exact(entry_size) {
        let tag_id = u16_at(order, raw);
        if !tag::LOADED.contains(&tag_id) {
            continue;
        }
        let field_type = u16_at(order, &raw[2..]);
        let Some(size) = type_size(field_type) else {
            continue;
        };
        // A tag appears once per IFD (TIFF 6.0, section 2). Like libtiff, keep the first
        // and ignore repeats, which would otherwise each be read.
        if listed.contains(&tag_id) {
            continue;
        }
        listed.push(tag_id);
        let (count, value) = if bigtiff {
            (u64_at(order, &raw[4..]), &raw[12..20])
        } else {
            (u64::from(u32_at(order, &raw[4..])), &raw[8..12])
        };
        let len = usize::try_from(count)
            .ok()
            .filter(|_| max_values(tag_id).is_none_or(|most| count <= most))
            .and_then(|c| c.checked_mul(size))
            .filter(|&l| l <= MAX_TAG_BYTES)
            .ok_or_else(|| {
                GeoTiffError::Invalid(format!(
                    "{name}: TIFF tag {tag_id} has {count} values, more than this reader accepts"
                ))
            })?;
        if len <= offset_size {
            let data = value[..len].to_vec();
            entries.insert(
                tag_id,
                Entry {
                    field_type,
                    count,
                    data,
                },
            );
        } else {
            deferred.push((tag_id, field_type, count, read_offset(value), len));
        }
    }
    // Fetch all out-of-line values of this IFD in one batch, once they fit the budget.
    let wanted: u64 = deferred.iter().map(|&(.., len)| len as u64).sum();
    *tag_budget = tag_budget.checked_sub(wanted).ok_or_else(|| {
        GeoTiffError::Invalid(format!(
            "{name}: its TIFF tags hold more data than the file itself, so the file is \
             corrupt; rewrite it (e.g. gdal_translate or rio convert) and try again"
        ))
    })?;
    let ranges: Vec<(u64, usize)> = deferred.iter().map(|&(.., at, len)| (at, len)).collect();
    let values = source.read_ranges(&ranges)?;
    for ((tag_id, field_type, count, ..), data) in deferred.into_iter().zip(values) {
        entries.insert(
            tag_id,
            Entry {
                field_type,
                count,
                data,
            },
        );
    }
    Ok((Ifd { order, entries }, next))
}

#[cfg(test)]
#[allow(clippy::cast_possible_truncation)]
mod tests {
    use super::super::source::MemorySource;
    use super::*;

    /// A classic little-endian TIFF with one IFD: width (SHORT), height
    /// (LONG), an out-of-line DOUBLE pair and a skipped unknown tag.
    fn tiny_tiff(next_ifd_points_to_self: bool) -> Vec<u8> {
        let mut f = b"II".to_vec();
        f.extend(42u16.to_le_bytes());
        f.extend(8u32.to_le_bytes());
        // IFD at 8 with 4 entries; values area at 8 + 2 + 4*12 + 4 = 62.
        f.extend(4u16.to_le_bytes());
        let mut entry = |tag: u16, ty: u16, count: u32, value: u32| {
            f.extend(tag.to_le_bytes());
            f.extend(ty.to_le_bytes());
            f.extend(count.to_le_bytes());
            f.extend(value.to_le_bytes());
        };
        entry(256, 3, 1, 300);
        entry(257, 4, 1, 200);
        entry(33550, 12, 2, 62);
        entry(65000, 2, 100, 9999);
        let next: u32 = if next_ifd_points_to_self { 8 } else { 0 };
        f.extend(next.to_le_bytes());
        f.extend(1.5f64.to_le_bytes());
        f.extend((-2.0f64).to_le_bytes());
        f
    }

    #[test]
    fn parses_inline_and_out_of_line_values() {
        let tiff = read_tiff(&MemorySource::new(tiny_tiff(false))).unwrap();
        assert!(!tiff.bigtiff);
        assert_eq!(tiff.ifds.len(), 1);
        let ifd = &tiff.ifds[0];
        assert_eq!(ifd.uint(tag::IMAGE_WIDTH, None).unwrap(), Some(300));
        assert_eq!(ifd.uint(tag::IMAGE_LENGTH, None).unwrap(), Some(200));
        assert_eq!(
            ifd.doubles(tag::MODEL_PIXEL_SCALE).unwrap(),
            Some(vec![1.5, -2.0])
        );
        assert!(!ifd.has(65000));
    }

    #[test]
    fn rejects_ifd_loops_and_non_tiffs() {
        let err = read_tiff(&MemorySource::new(tiny_tiff(true))).unwrap_err();
        assert!(err.to_string().contains("loop"), "{err}");
        let err = read_tiff(&MemorySource::new(b"\x89PNG\r\n\x1a\n0000".to_vec())).unwrap_err();
        assert!(err.to_string().contains("not a TIFF"), "{err}");
    }

    /// A TIFF with a 1 MiB block of zeros at offset 8, then `ifds` IFDs that each list
    /// `(tag, type, count, value)` entries.
    fn tiff_with_block(ifds: &[Vec<(u16, u16, u32, u32)>]) -> Vec<u8> {
        let block = 1 << 20;
        let mut f = b"II".to_vec();
        f.extend(42u16.to_le_bytes());
        f.extend((8 + block as u32).to_le_bytes());
        f.resize(8 + block, 0);
        for (i, entries) in ifds.iter().enumerate() {
            f.extend((entries.len() as u16).to_le_bytes());
            for &(tag, ty, count, value) in entries {
                f.extend(tag.to_le_bytes());
                f.extend(ty.to_le_bytes());
                f.extend(count.to_le_bytes());
                f.extend(value.to_le_bytes());
            }
            let next = if i + 1 == ifds.len() {
                0
            } else {
                f.len() as u32 + 4
            };
            f.extend(next.to_le_bytes());
        }
        f
    }

    #[test]
    fn reads_a_tag_repeated_in_one_ifd_once() {
        // Each of the 4096 entries read the 1 MiB block before: 4 GiB. Now the first
        // is kept, as libtiff does, and only it is read.
        let mut entries = vec![(tag::IMAGE_WIDTH, 3, 1, 7), (tag::IMAGE_WIDTH, 3, 1, 9)];
        entries.extend(vec![(tag::STRIP_OFFSETS, 1, 1 << 20, 8); MAX_ENTRIES - 2]);
        let tiff = read_tiff(&MemorySource::new(tiff_with_block(&[entries]))).unwrap();
        let ifd = &tiff.ifds[0];
        assert_eq!(ifd.uint(tag::IMAGE_WIDTH, None).unwrap(), Some(7));
        assert_eq!(
            ifd.uint_array(tag::STRIP_OFFSETS).unwrap().unwrap().len(),
            1 << 20
        );
    }

    #[test]
    fn tag_values_are_bounded_by_the_file_size() {
        // Each IFD lists its own tags once, but all point at the same 1 MiB block.
        let ifd = vec![
            (tag::IMAGE_WIDTH, 3, 1, 1),
            (tag::STRIP_OFFSETS, 1, 1 << 20, 8),
        ];
        let tiff = read_tiff(&MemorySource::new(tiff_with_block(&vec![ifd.clone(); 8])));
        assert_eq!(
            tiff.unwrap().ifds.len(),
            8,
            "shared values within the slack"
        );
        let err = read_tiff(&MemorySource::new(tiff_with_block(&vec![ifd; 64]))).unwrap_err();
        assert!(err.to_string().contains("more data than the file"), "{err}");
    }

    #[test]
    fn rejects_more_band_values_than_bands_can_exist() {
        let ifd = vec![(tag::BITS_PER_SAMPLE, 3, 65_536, 8)];
        let err = read_tiff(&MemorySource::new(tiff_with_block(&[ifd]))).unwrap_err();
        assert!(err.to_string().contains("65536 values"), "{err}");
    }

    #[test]
    fn decodes_integer_arrays_on_access() {
        let ifd = vec![
            (tag::IMAGE_WIDTH, 4, 1, 7),
            (tag::STRIP_OFFSETS, 3, 3, 8),
            (tag::STRIP_BYTE_COUNTS, 16, 2, 16),
        ];
        let mut f = tiff_with_block(&[ifd]);
        f[8..14].copy_from_slice(&[1, 0, 2, 0, 3, 0]);
        f[16..24].copy_from_slice(&2u64.to_le_bytes());
        f[24..32].copy_from_slice(&3u64.to_le_bytes());
        let tiff = read_tiff(&MemorySource::new(f)).unwrap();
        let ifd = &tiff.ifds[0];
        assert_eq!(ifd.uint(tag::IMAGE_WIDTH, None).unwrap(), Some(7));
        let offsets = ifd.uint_array(tag::STRIP_OFFSETS).unwrap().unwrap();
        assert_eq!(offsets.len(), 3);
        assert_eq!(offsets.iter().collect::<Vec<_>>(), [1, 2, 3]);
        assert_eq!(offsets.get(3), None);
        assert_eq!(ifd.uints(tag::STRIP_BYTE_COUNTS).unwrap(), Some(vec![2, 3]));
        assert_eq!(ifd.uint(tag::STRIP_OFFSETS, None).unwrap(), Some(1));
        assert_eq!(ifd.uint(tag::ROWS_PER_STRIP, Some(5)).unwrap(), Some(5));
    }
}
