//! Arbitrary bytes as a TIFF/BigTIFF/COG: header, IFD chain, tag values, GeoKeys,
//! the level structure, then a few window reads through the chunk decoders.
#![no_main]

use libfuzzer_sys::fuzz_target;

fuzz_target!(|data: &[u8]| {
    mapcv_fuzz::exercise_tiff(data.to_vec());
});
