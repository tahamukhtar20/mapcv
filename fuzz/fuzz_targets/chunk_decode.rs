//! One TIFF chunk through `decode_chunk`: every supported codec with an arbitrary
//! predictor, byte order, sample layout and plausible (bounded) dimensions.
//!
//! Input layout (so real JPEG, WebP, zlib, ... streams can be seeds): 11 parameter
//! bytes, `tables_len` bytes of `JPEGTables`, then the chunk bytes.
//!
//! ```text
//! 0 compression  1 predictor  2 flags (bit 0 big-endian, bit 1 re-encode the chunk,
//!                                      bits 2-3 extra rows the chunk may claim)
//! 3 sample_bytes 4 samples    5..7 width (LE u16)  7..9 rows (LE u16)
//! 9 photometric  10 tables_len
//! ```
#![no_main]

use libfuzzer_sys::fuzz_target;
use mapcv::geotiff::codec::{compression, decode_chunk, ChunkFormat};
use mapcv::geotiff::ifd::ByteOrder;
use mapcv_fuzz::{compress, COMPRESSIONS};

/// Largest decoded chunk exercised: keeps one run fast; the reader itself allows 1 GiB.
const MAX_DECODED: usize = 1 << 16;
const HEADER: usize = 11;

fuzz_target!(|data: &[u8]| {
    let Some((head, rest)) = data.split_at_checked(HEADER) else {
        return;
    };
    let tables_len = usize::from(head[10]).min(rest.len());
    let (tables, chunk) = rest.split_at(tables_len);
    let code = COMPRESSIONS[usize::from(head[0]) % COMPRESSIONS.len()];
    // JPEG and WebP only exist with 8-bit samples (`check_supported`).
    let sample_bytes = if matches!(code, compression::JPEG | compression::WEBP) {
        1
    } else {
        [1usize, 2, 4, 8][usize::from(head[3]) % 4]
    };
    let samples = 1 + usize::from(head[4]) % 8;
    let width = 1 + usize::from(u16::from_le_bytes([head[5], head[6]])) % 512;
    let mut rows = 1 + usize::from(u16::from_le_bytes([head[7], head[8]])) % 512;
    while width * samples * sample_bytes * rows > MAX_DECODED {
        rows = rows.div_ceil(2);
    }
    let expected = width * samples * sample_bytes * rows;
    let encoded;
    let chunk = if head[2] & 2 != 0 && !chunk.is_empty() {
        let raw: Vec<u8> = chunk.iter().copied().cycle().take(expected).collect();
        encoded = compress(code, &raw);
        &encoded
    } else {
        chunk
    };
    let format = ChunkFormat {
        compression: code,
        predictor: [1u16, 2, 3, 0][usize::from(head[1]) % 4],
        order: if head[2] & 1 != 0 {
            ByteOrder::Big
        } else {
            ByteOrder::Little
        },
        sample_bytes,
        samples,
        width,
        height: rows + usize::from((head[2] >> 2) & 3),
        photometric: [1u16, 6, 2, 0][usize::from(head[9]) % 4],
        jpeg_tables: (!tables.is_empty()).then_some(tables),
    };
    if let Ok(pixels) = decode_chunk(&format, chunk, rows) {
        assert_eq!(pixels.len(), expected, "decoded chunk has the wrong size");
    }
});
