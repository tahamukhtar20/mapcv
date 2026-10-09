//! Arbitrary bytes as KML, with and without a label field.
#![no_main]

use libfuzzer_sys::fuzz_target;
use mapcv::kml_parser::{kml_fields, parse_kml};

fuzz_target!(|data: &[u8]| {
    // The single-pass read of every field agrees with a parse for one field.
    if let (Ok(all), Ok(one)) = (kml_fields(data), parse_kml(data, Some("name"))) {
        let labels: Vec<_> = one.polygons.into_iter().map(|(_, label)| label).collect();
        let fields: Vec<_> = all.fields.iter().map(|f| f.get("name").cloned()).collect();
        assert_eq!(labels, fields, "kml_fields disagrees with parse_kml");
    }
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
