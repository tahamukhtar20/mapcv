//! Fast KML parser: extracts `(polygon_rings, label)` pairs using `quick-xml`.
//!
//! Supports `<Polygon>` and `<MultiGeometry>` placemarks. When `label_field`
//! is `None` every polygon gets class 1. When set, the value is read from
//! `<ExtendedData><Data name="…"><value>`.

use quick_xml::events::Event;
use quick_xml::Reader;
use std::collections::HashMap;

/// A single ring: list of `(lng, lat)` pairs.
pub type Ring = Vec<(f64, f64)>;
/// A single polygon: exterior ring first, then zero or more interior (hole) rings.
pub type Polygon = Vec<Ring>;

/// Parsed result: list of `(polygon_group, class_id)` and the class-name → id map.
pub struct KmlResult {
    /// `(polygon_group, class_id)` pairs in document order, one entry per `Placemark`.
    /// A group has one polygon for simple placemarks, many for `MultiGeometry`.
    pub polygons: Vec<(Vec<Polygon>, u8)>,
    /// Class name to integer id mapping (empty when `label_field` is `None`).
    pub class_map: HashMap<String, u8>,
}

/// Parse KML bytes and return polygon geometries with class labels.
///
/// # Errors
/// Returns an error string if the XML is malformed or coordinate parsing fails.
#[allow(clippy::too_many_lines)]
pub fn parse_kml(data: &[u8], label_field: Option<&str>) -> Result<KmlResult, String> {
    let mut reader = Reader::from_reader(data);
    reader.config_mut().trim_text(true);

    let mut polygons: Vec<(Vec<Polygon>, u8)> = Vec::new();
    let mut class_map: HashMap<String, u8> = HashMap::new();

    // State
    let mut in_placemark = false;
    let mut current_label: Option<String> = None;
    let mut current_polys: Vec<Polygon> = Vec::new();
    let mut current_rings: Vec<Ring> = Vec::new(); // rings for one <Polygon>
    let mut current_ring: Option<Ring> = None;
    let mut capture_coords = false;
    let mut capture_value = false;
    let mut current_data_name: Option<String> = None;
    let mut buf = Vec::new();

    loop {
        match reader.read_event_into(&mut buf) {
            Ok(Event::Start(ref e)) => {
                let tag = std::str::from_utf8(e.local_name().as_ref())
                    .unwrap_or("")
                    .to_owned();
                match tag.as_str() {
                    "Placemark" => {
                        in_placemark = true;
                        current_label = None;
                        current_polys.clear();
                    }
                    "Polygon" if in_placemark => {
                        current_rings.clear();
                    }
                    "outerBoundaryIs" | "innerBoundaryIs" if in_placemark => {
                        current_ring = Some(Vec::new());
                    }
                    "coordinates" if in_placemark => {
                        capture_coords = true;
                    }
                    "Data" if in_placemark => {
                        if let Some(name_attr) = e
                            .attributes()
                            .flatten()
                            .find(|a| std::str::from_utf8(a.key.as_ref()).unwrap_or("") == "name")
                        {
                            current_data_name = Some(
                                std::str::from_utf8(&name_attr.value)
                                    .unwrap_or("")
                                    .to_owned(),
                            );
                        }
                    }
                    "value" if in_placemark => {
                        if let Some(field) = label_field {
                            if current_data_name.as_deref() == Some(field) {
                                capture_value = true;
                            }
                        }
                    }
                    _ => {}
                }
            }
            Ok(Event::End(ref e)) => {
                let tag = std::str::from_utf8(e.local_name().as_ref())
                    .unwrap_or("")
                    .to_owned();
                match tag.as_str() {
                    "Placemark" => {
                        in_placemark = false;
                        let class_id =
                            resolve_class(current_label.as_deref(), label_field, &mut class_map);
                        if !current_polys.is_empty() {
                            polygons.push((std::mem::take(&mut current_polys), class_id));
                        }
                    }
                    "Polygon" if in_placemark && !current_rings.is_empty() => {
                        current_polys.push(std::mem::take(&mut current_rings));
                    }
                    "outerBoundaryIs" | "innerBoundaryIs" if in_placemark => {
                        if let Some(ring) = current_ring.take().filter(|r| !r.is_empty()) {
                            current_rings.push(ring);
                        }
                    }
                    "coordinates" => {
                        capture_coords = false;
                    }
                    "value" => {
                        capture_value = false;
                        current_data_name = None;
                    }
                    "Data" => {
                        current_data_name = None;
                    }
                    _ => {}
                }
            }
            Ok(Event::Text(ref e)) => {
                if capture_coords {
                    let text = e.unescape().map_err(|e| e.to_string())?;
                    if let Some(ring) = current_ring.as_mut() {
                        parse_coordinates(text.trim(), ring)?;
                    }
                } else if capture_value {
                    let text = e.unescape().map_err(|e| e.to_string())?;
                    current_label = Some(text.trim().to_owned());
                }
            }
            Ok(Event::CData(ref e)) => {
                let text = std::str::from_utf8(e.as_ref())
                    .map_err(|e| e.to_string())?
                    .to_owned();
                if capture_coords {
                    if let Some(ring) = current_ring.as_mut() {
                        parse_coordinates(text.trim(), ring)?;
                    }
                } else if capture_value {
                    current_label = Some(text.trim().to_owned());
                }
            }
            Ok(Event::Eof) => break,
            Err(e) => return Err(e.to_string()),
            _ => {}
        }
        buf.clear();
    }

    Ok(KmlResult {
        polygons,
        class_map,
    })
}

fn resolve_class(
    label: Option<&str>,
    label_field: Option<&str>,
    class_map: &mut HashMap<String, u8>,
) -> u8 {
    if label_field.is_none() {
        return 1;
    }
    match label {
        None => 0,
        Some(l) => {
            #[allow(clippy::cast_possible_truncation)]
            let next_id = class_map.len() as u8 + 1;
            *class_map.entry(l.to_owned()).or_insert(next_id)
        }
    }
}

fn parse_coordinates(text: &str, ring: &mut Ring) -> Result<(), String> {
    for token in text.split_whitespace() {
        let mut parts = token.split(',');
        let lng: f64 = parts
            .next()
            .ok_or("missing lng")?
            .parse()
            .map_err(|e: std::num::ParseFloatError| e.to_string())?;
        let lat: f64 = parts
            .next()
            .ok_or("missing lat")?
            .parse()
            .map_err(|e: std::num::ParseFloatError| e.to_string())?;
        ring.push((lng, lat));
    }
    Ok(())
}
