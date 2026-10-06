//! Arbitrary bytes as KML, with and without a label field.
#![no_main]

use libfuzzer_sys::fuzz_target;
use mapcv::kml_parser::parse_kml;

fuzz_target!(|data: &[u8]| {
    for label_field in [None, Some("name")] {
        if let Ok(result) = parse_kml(data, label_field) {
            for (group, _label) in &result.polygons {
                for polygon in group {
                    for ring in polygon {
                        assert!(ring.len() >= 3, "a ring has fewer than 3 points");
                        assert!(ring.iter().all(|(x, y)| x.is_finite() && y.is_finite()));
                    }
                }
            }
        }
    }
});
