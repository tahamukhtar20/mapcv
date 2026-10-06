//! Helpers shared by the fuzz targets: encoders for TIFF chunk codecs, a
//! structure-aware TIFF writer, and the common "open, then read some windows"
//! exercise run on every TIFF the targets produce.
//!
//! Raw random bytes almost never form a TIFF whose offsets, counts and codec
//! streams are mutually consistent, so the structured targets build the file
//! from arbitrary *parameters* instead and let the fuzzer mutate those.

use arbitrary::Arbitrary;
use mapcv::geotiff::{GeoTiff, Window};
use std::io::Write;

/// TIFF compression codes: those the reader decodes, then one it rejects.
pub const COMPRESSIONS: [u16; 9] = [1, 5, 7, 8, 32773, 32946, 50000, 50001, 34887];

/// Encode `raw` with the TIFF compression `code`.
///
/// JPEG, `WebP` and unknown codes return `raw` unchanged: the fuzzer cannot
/// build those streams from parameters, but it can mutate the seeds.
#[must_use]
pub fn compress(code: u16, raw: &[u8]) -> Vec<u8> {
    match code {
        5 => weezl::encode::Encoder::with_tiff_size_switch(weezl::BitOrder::Msb, 8)
            .encode(raw)
            .unwrap_or_default(),
        8 | 32946 => {
            let mut z = flate2::write::ZlibEncoder::new(Vec::new(), flate2::Compression::fast());
            z.write_all(raw)
                .and_then(|()| z.finish())
                .unwrap_or_default()
        }
        50000 => zstd::bulk::compress(raw, 1).unwrap_or_default(),
        32773 => {
            let mut out = Vec::new();
            for run in raw.chunks(128) {
                out.push(u8::try_from(run.len() - 1).unwrap_or(0));
                out.extend_from_slice(run);
            }
            out
        }
        _ => raw.to_vec(),
    }
}

/// One image (the first, or an overview) of a structured TIFF.
#[derive(Debug, Arbitrary)]
pub struct ImageSpec {
    width: u32,
    height: u32,
    /// 0 and 1: small dimensions, so most inputs decode; 2: the raw values; 3: up to
    /// 2^31 (the reader's limit), where products of dimensions overflow.
    scale: u8,
    bits: u8,
    samples: u16,
    format: u8,
    compression: u8,
    predictor: u8,
    photometric: u8,
    planar: bool,
    tiled: bool,
    chunk_width: u32,
    chunk_height: u32,
    /// `NewSubfileType` bit 0: a reduced-resolution overview.
    reduced: bool,
    /// Decoded bytes of the chunks, repeated and rotated to fill each chunk.
    payload: Vec<u8>,
    /// Store `payload` as the chunk bytes without compressing it.
    raw_chunks: bool,
    jpeg_tables: Option<Vec<u8>>,
    /// Add `delta` to the offset of chunk `index`, or to its byte count.
    corrupt: Option<(u8, u32, bool)>,
}

/// The georeferencing tags of the first image.
#[derive(Debug, Arbitrary)]
pub struct GeoSpec {
    /// A GeoKey directory: `(key, location, count, value)` entries.
    keys: Vec<(u16, u16, u16, u16)>,
    /// Header words (`version, major, minor`); the key count is appended.
    header: (u16, u16, u16),
    /// Override the declared key count (a malformed directory).
    key_count: Option<u16>,
    raw_directory: Option<Vec<u16>>,
    doubles: Vec<f64>,
    ascii: Vec<u8>,
    scale: Option<Vec<f64>>,
    tiepoints: Option<Vec<f64>>,
    matrix: Option<Vec<f64>>,
    nodata: Option<Vec<u8>>,
}

/// A whole structured TIFF.
#[derive(Debug, Arbitrary)]
pub struct TiffSpec {
    big_endian: bool,
    bigtiff: bool,
    images: Vec<ImageSpec>,
    geo: GeoSpec,
}

/// One IFD entry with its value in file byte order.
struct Entry {
    tag: u16,
    field_type: u16,
    count: u64,
    data: Vec<u8>,
}

/// Byte-order aware integer encoding.
#[derive(Clone, Copy)]
struct Enc {
    big: bool,
}

impl Enc {
    fn u16(self, v: u16) -> Vec<u8> {
        if self.big {
            v.to_be_bytes()
        } else {
            v.to_le_bytes()
        }
        .to_vec()
    }
    fn u32(self, v: u32) -> Vec<u8> {
        if self.big {
            v.to_be_bytes()
        } else {
            v.to_le_bytes()
        }
        .to_vec()
    }
    fn u64(self, v: u64) -> Vec<u8> {
        if self.big {
            v.to_be_bytes()
        } else {
            v.to_le_bytes()
        }
        .to_vec()
    }
    fn f64(self, v: f64) -> Vec<u8> {
        self.u64(v.to_bits())
    }
    fn shorts(self, v: &[u16]) -> Entry {
        let data = v.iter().flat_map(|&x| self.u16(x)).collect();
        Entry {
            tag: 0,
            field_type: 3,
            count: v.len() as u64,
            data,
        }
    }
    fn longs(self, v: &[u32]) -> Entry {
        let data = v.iter().flat_map(|&x| self.u32(x)).collect();
        Entry {
            tag: 0,
            field_type: 4,
            count: v.len() as u64,
            data,
        }
    }
    fn long8s(self, v: &[u64]) -> Entry {
        let data = v.iter().flat_map(|&x| self.u64(x)).collect();
        Entry {
            tag: 0,
            field_type: 16,
            count: v.len() as u64,
            data,
        }
    }
    fn doubles(self, v: &[f64]) -> Entry {
        let data = v.iter().flat_map(|&x| self.f64(x)).collect();
        Entry {
            tag: 0,
            field_type: 12,
            count: v.len() as u64,
            data,
        }
    }
}

fn bytes_entry(tag: u16, field_type: u16, data: &[u8]) -> Entry {
    Entry {
        tag,
        field_type,
        count: data.len() as u64,
        data: data.to_vec(),
    }
}

fn tagged(tag: u16, mut entry: Entry) -> Entry {
    entry.tag = tag;
    entry
}

fn pick<T: Copy>(options: &[T], i: u8) -> T {
    options[usize::from(i) % options.len()]
}

/// Bytes a chunk of the described shape decodes to.
fn chunk_len(width: u64, rows: u64, samples: u64, bytes: u64) -> usize {
    usize::try_from(
        width
            .saturating_mul(rows)
            .saturating_mul(samples)
            .saturating_mul(bytes),
    )
    .unwrap_or(usize::MAX)
    .min(1 << 14)
}

impl ImageSpec {
    /// Entries of this image's IFD; chunk data is appended to `blob`, whose
    /// first byte lies at file offset `blob_start`.
    fn entries(&self, enc: Enc, bigtiff: bool, blob: &mut Vec<u8>, blob_start: u64) -> Vec<Entry> {
        let (mut width, mut height) = (self.width, self.height);
        let (mut chunk_width, mut chunk_height) = (self.chunk_width, self.chunk_height);
        let small = self.scale % 4 < 2;
        let samples = if small {
            1 + self.samples % 6
        } else {
            self.samples
        };
        if small {
            width = width % 256 + 1;
            height = height % 256 + 1;
            chunk_width = chunk_width % 64 + 1;
            chunk_height = chunk_height % 64 + 1;
            // At most 32 chunks across and down.
            chunk_width = chunk_width.max(width.div_ceil(32));
            chunk_height = chunk_height.max(height.div_ceil(32));
        } else if self.scale % 4 == 3 {
            const LIMIT: u32 = 1 << 31;
            width = width % LIMIT + 1;
            height = height % LIMIT + 1;
            chunk_width = chunk_width % LIMIT + 1;
            chunk_height = chunk_height % LIMIT + 1;
        }
        let bits = pick(&[1u16, 8, 8, 8, 16, 16, 32, 64, 12], self.bits);
        let format = pick(&[1u16, 2, 3, 4, 1, 1, 2, 3, 0, 7], self.format);
        let compression = pick(&COMPRESSIONS, self.compression);
        let predictor = pick(&[1u16, 2, 3, 1, 2, 0, 4], self.predictor);
        let photometric = pick(&[0u16, 1, 2, 3, 5, 6, 1, 1], self.photometric);

        // Chunk geometry as the reader derives it.
        let (cw, ch) = if self.tiled {
            (chunk_width, chunk_height)
        } else {
            (width, chunk_height.clamp(1, height.max(1)))
        };
        let across = u64::from(width).div_ceil(u64::from(cw.max(1)));
        let down = u64::from(height).div_ceil(u64::from(ch.max(1)));
        let planes = if self.planar { u64::from(samples) } else { 1 };
        // Declare the right number of chunks, but never write more than 64.
        let declared = usize::try_from(across.saturating_mul(down).saturating_mul(planes))
            .unwrap_or(usize::MAX)
            .min(64);
        let per_chunk = if self.planar { 1 } else { u64::from(samples) };
        let decoded = chunk_len(
            u64::from(cw),
            u64::from(ch),
            per_chunk,
            u64::from(bits / 8).max(1),
        );
        // At most four distinct chunks are stored; the others point at them.
        let mut stored: Vec<(u64, u64)> = Vec::new();
        for i in 0..declared.min(4) {
            let mut raw: Vec<u8> = (0..decoded)
                .map(|j| {
                    *self
                        .payload
                        .get((j + i) % self.payload.len().max(1))
                        .unwrap_or(&0)
                })
                .collect();
            if self.raw_chunks {
                raw.clone_from(&self.payload);
            }
            let data = if self.raw_chunks {
                raw
            } else {
                compress(compression, &raw)
            };
            stored.push((blob_start + blob.len() as u64, data.len() as u64));
            blob.extend_from_slice(&data);
        }
        let mut offsets: Vec<u64> = (0..declared).map(|i| stored[i % stored.len()].0).collect();
        let mut counts: Vec<u64> = (0..declared).map(|i| stored[i % stored.len()].1).collect();
        if let Some((index, delta, on_count)) = self.corrupt {
            if !offsets.is_empty() {
                let i = usize::from(index) % offsets.len();
                let target = if on_count {
                    &mut counts[i]
                } else {
                    &mut offsets[i]
                };
                *target = target.wrapping_add(u64::from(delta));
            }
        }
        let offsets_entry = |tag: u16, values: &[u64]| {
            if bigtiff {
                tagged(tag, enc.long8s(values))
            } else {
                let v: Vec<u32> = values.iter().map(|&x| x as u32).collect();
                tagged(tag, enc.longs(&v))
            }
        };

        let mut e = vec![
            tagged(254, enc.longs(&[u32::from(self.reduced)])),
            tagged(256, enc.longs(&[width])),
            tagged(257, enc.longs(&[height])),
            tagged(258, enc.shorts(&vec![bits; usize::from(samples).min(8)])),
            tagged(259, enc.shorts(&[compression])),
            tagged(262, enc.shorts(&[photometric])),
            tagged(277, enc.shorts(&[samples])),
            tagged(284, enc.shorts(&[if self.planar { 2 } else { 1 }])),
            tagged(317, enc.shorts(&[predictor])),
            tagged(339, enc.shorts(&vec![format; usize::from(samples).min(8)])),
        ];
        if self.tiled {
            e.push(tagged(322, enc.longs(&[chunk_width])));
            e.push(tagged(323, enc.longs(&[chunk_height])));
            e.push(offsets_entry(324, &offsets));
            e.push(offsets_entry(325, &counts));
        } else {
            e.push(tagged(278, enc.longs(&[chunk_height])));
            e.push(offsets_entry(273, &offsets));
            e.push(offsets_entry(279, &counts));
        }
        if let Some(tables) = &self.jpeg_tables {
            e.push(bytes_entry(347, 7, tables));
        }
        e
    }
}

impl GeoSpec {
    fn entries(&self, enc: Enc) -> Vec<Entry> {
        let mut e = Vec::new();
        let directory: Vec<u16> = match &self.raw_directory {
            Some(raw) => raw.clone(),
            None => {
                let n = u16::try_from(self.keys.len()).unwrap_or(u16::MAX);
                let mut d = vec![
                    self.header.0,
                    self.header.1,
                    self.header.2,
                    self.key_count.unwrap_or(n),
                ];
                for &(a, b, c, v) in &self.keys {
                    d.extend([a, b, c, v]);
                }
                d
            }
        };
        e.push(tagged(34735, enc.shorts(&directory)));
        e.push(tagged(34736, enc.doubles(&self.doubles)));
        e.push(bytes_entry(34737, 2, &self.ascii));
        for (tag, values) in [
            (33550, &self.scale),
            (33922, &self.tiepoints),
            (34264, &self.matrix),
        ] {
            if let Some(v) = values {
                e.push(tagged(tag, enc.doubles(v)));
            }
        }
        if let Some(text) = &self.nodata {
            e.push(bytes_entry(42113, 2, text));
        }
        e
    }
}

/// Serialise `spec` to a TIFF file.
#[must_use]
pub fn build_tiff(spec: &TiffSpec) -> Vec<u8> {
    let enc = Enc {
        big: spec.big_endian,
    };
    let bigtiff = spec.bigtiff;
    let header_len: u64 = if bigtiff { 16 } else { 8 };
    let (count_size, entry_size, offset_size): (u64, u64, u64) =
        if bigtiff { (8, 20, 8) } else { (2, 12, 4) };
    let mut blob = Vec::new();
    let mut ifds: Vec<Vec<Entry>> = Vec::new();
    for (i, image) in spec.images.iter().take(4).enumerate() {
        let mut entries = image.entries(enc, bigtiff, &mut blob, header_len);
        if i == 0 {
            entries.extend(spec.geo.entries(enc));
        }
        entries.sort_by_key(|e| e.tag);
        ifds.push(entries);
    }
    let inline = offset_size as usize;
    let size_of = |entries: &[Entry]| -> u64 {
        let table = count_size + entries.len() as u64 * entry_size + offset_size;
        let values: u64 = entries
            .iter()
            .filter(|e| e.data.len() > inline)
            .map(|e| (e.data.len() as u64 + 1) & !1)
            .sum();
        table + values
    };
    let mut positions = Vec::new();
    let mut at = header_len + blob.len() as u64;
    for entries in &ifds {
        positions.push(at);
        at += size_of(entries);
    }

    let mut out = Vec::new();
    out.extend_from_slice(if spec.big_endian { b"MM" } else { b"II" });
    if bigtiff {
        out.extend(enc.u16(43));
        out.extend(enc.u16(8));
        out.extend(enc.u16(0));
        out.extend(enc.u64(positions.first().copied().unwrap_or(0)));
    } else {
        out.extend(enc.u16(42));
        out.extend(enc.u32(positions.first().copied().unwrap_or(0) as u32));
    }
    out.extend_from_slice(&blob);
    for (k, entries) in ifds.iter().enumerate() {
        let table = count_size + entries.len() as u64 * entry_size + offset_size;
        let mut values_at = positions[k] + table;
        let mut table_bytes = if bigtiff {
            enc.u64(entries.len() as u64)
        } else {
            enc.u16(entries.len() as u16)
        };
        let mut values = Vec::new();
        for entry in entries {
            table_bytes.extend(enc.u16(entry.tag));
            table_bytes.extend(enc.u16(entry.field_type));
            if bigtiff {
                table_bytes.extend(enc.u64(entry.count));
            } else {
                table_bytes.extend(enc.u32(entry.count as u32));
            }
            if entry.data.len() <= inline {
                let mut v = entry.data.clone();
                v.resize(inline, 0);
                table_bytes.extend(v);
            } else {
                if bigtiff {
                    table_bytes.extend(enc.u64(values_at));
                } else {
                    table_bytes.extend(enc.u32(values_at as u32));
                }
                values.extend_from_slice(&entry.data);
                if entry.data.len() % 2 == 1 {
                    values.push(0);
                }
                values_at += (entry.data.len() as u64 + 1) & !1;
            }
        }
        let next = positions.get(k + 1).copied().unwrap_or(0);
        if bigtiff {
            table_bytes.extend(enc.u64(next));
        } else {
            table_bytes.extend(enc.u32(next as u32));
        }
        out.extend(table_bytes);
        out.extend(values);
    }
    out
}

/// Open `data` as a GeoTIFF and read a few windows from every level, so the
/// header and IFD walk, the georeferencing, the chunk planning and every codec
/// run. Errors are expected and ignored; panics, hangs and huge allocations are
/// what the fuzzer is after.
pub fn exercise_tiff(data: Vec<u8>) {
    let Ok(tiff) = GeoTiff::from_bytes(data) else {
        return;
    };
    let _ = (
        tiff.epsg(),
        tiff.transform(),
        tiff.raster_type(),
        tiff.crs_citation().map(str::len),
        tiff.nodata(),
        tiff.compression(),
        tiff.predictor(),
        tiff.photometric(),
        tiff.dtype().name(),
    );
    let bands = tiff.band_count();
    for (overview, level) in tiff.levels().into_iter().enumerate() {
        let rows = i64::try_from(level.height.min(40)).unwrap_or(40);
        let cols = i64::try_from(level.width.min(40)).unwrap_or(40);
        let last_band = bands.saturating_sub(1);
        let windows = [
            (
                Window {
                    row0: 0,
                    row1: rows,
                    col0: 0,
                    col1: cols,
                },
                None,
            ),
            (
                Window {
                    row0: -3,
                    row1: rows / 2 + 1,
                    col0: -3,
                    col1: cols / 2 + 1,
                },
                Some(vec![last_band, 0]),
            ),
            (
                Window {
                    row0: i64::try_from(level.height).unwrap_or(0) - 3,
                    row1: i64::try_from(level.height).unwrap_or(0) + 5,
                    col0: i64::try_from(level.width).unwrap_or(0) - 3,
                    col1: i64::try_from(level.width).unwrap_or(0) + 5,
                },
                None,
            ),
        ];
        for (window, selected) in &windows {
            let _ = tiff.read_window(*window, selected.as_deref(), overview);
        }
    }
}
