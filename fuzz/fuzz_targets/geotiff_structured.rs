//! A TIFF built from arbitrary parameters (dimensions, sample layout, codec,
//! predictor, chunk geometry, offsets, GeoKeys, model tags), so the planner and
//! the codecs see files whose structure is valid enough to get past validation.
#![no_main]

use libfuzzer_sys::fuzz_target;
use mapcv_fuzz::{build_tiff, exercise_tiff, TiffSpec};

fuzz_target!(|spec: TiffSpec| {
    exercise_tiff(build_tiff(&spec));
});
