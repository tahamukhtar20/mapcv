//! Fast KML parser: extracts `(polygon_rings, label)` pairs using `quick-xml`.
//!
//! Supports `<Polygon>` and `<MultiGeometry>` placemarks. When `label_field`
//! is set, the label is read from `<ExtendedData><Data name="..."><value>` or
//! from `<SchemaData><SimpleData name="...">` (the form GDAL, QGIS and ogr2ogr
//! write). Class IDs are assigned by the Python caller.

use quick_xml::events::{BytesStart, Event};
use quick_xml::Reader;

/// A single ring: list of `(lng, lat)` pairs.
pub type Ring = Vec<(f64, f64)>;
/// A single polygon: exterior ring first, then zero or more interior (hole) rings.
pub type Polygon = Vec<Ring>;

/// Parsed result: one entry per polygon-bearing `Placemark`, in document order.
pub struct KmlResult {
    /// `(polygon_group, label)` pairs. A group has one polygon for simple
    /// placemarks, many for `MultiGeometry`. `label` is `None` when no
    /// `label_field` is requested or the placemark lacks it.
    pub polygons: Vec<(Vec<Polygon>, Option<String>)>,
    /// Placemarks skipped because they contain no polygon (points, lines).
    pub skipped_non_polygon: usize,
}

/// Minimum ring length: a closed triangle has 4 coordinates, an open one 3.
const MIN_RING_POINTS: usize = 3;

fn local_name(e: &BytesStart<'_>) -> String {
    String::from_utf8_lossy(e.local_name().as_ref()).into_owned()
}

fn name_attribute(e: &BytesStart<'_>) -> Result<Option<String>, String> {
    for attr in e.attributes().flatten() {
        if attr.key.local_name().as_ref() == b"name" {
            let raw = String::from_utf8_lossy(&attr.value);
            let value = quick_xml::escape::unescape(&raw).map_err(|err| err.to_string())?;
            return Ok(Some(value.into_owned()));
        }
    }
    Ok(None)
}

/// Mutable parser state for the placemark being read.
// Independent capture flags mirror the KML element nesting; an enum would not simplify it.
#[allow(clippy::struct_excessive_bools)]
#[derive(Default)]
struct PlacemarkState {
    in_placemark: bool,
    label: Option<String>,
    polys: Vec<Polygon>,
    has_geometry: bool,
    outer: Option<Ring>,
    holes: Vec<Ring>,
    ring: Option<Ring>,
    ring_is_outer: bool,
    coords: String,
    capture_coords: bool,
    capture_value: bool,
    value: String,
    data_name: Option<String>,
}

impl PlacemarkState {
    fn finish_ring(&mut self) -> Result<(), String> {
        let Some(mut ring) = self.ring.take() else {
            return Ok(());
        };
        parse_coordinates(&std::mem::take(&mut self.coords), &mut ring)?;
        if ring.len() < MIN_RING_POINTS {
            return Ok(());
        }
        if self.ring_is_outer {
            self.outer = Some(ring);
        } else {
            self.holes.push(ring);
        }
        Ok(())
    }

    fn finish_polygon(&mut self) {
        let holes = std::mem::take(&mut self.holes);
        if let Some(outer) = self.outer.take() {
            let mut rings = Vec::with_capacity(holes.len() + 1);
            rings.push(outer);
            rings.extend(holes);
            self.polys.push(rings);
        }
    }
}

/// Parse KML bytes and return polygon geometries with their raw labels.
///
/// # Errors
/// Returns an error string if the XML is malformed or truncated, or a
/// coordinate cannot be parsed as a finite number.
// The event loop is one state machine; splitting it would only move the match arms.
#[allow(clippy::too_many_lines)]
pub fn parse_kml(data: &[u8], label_field: Option<&str>) -> Result<KmlResult, String> {
    let mut reader = Reader::from_reader(data);
    reader.config_mut().trim_text(true);

    let mut polygons: Vec<(Vec<Polygon>, Option<String>)> = Vec::new();
    let mut skipped_non_polygon = 0usize;
    let mut state = PlacemarkState::default();
    let mut depth = 0usize;
    let mut buf = Vec::new();

    loop {
        match reader.read_event_into(&mut buf) {
            Ok(Event::Start(ref e)) => {
                depth += 1;
                let tag = local_name(e);
                match tag.as_str() {
                    "Placemark" => {
                        state = PlacemarkState {
                            in_placemark: true,
                            ..PlacemarkState::default()
                        };
                    }
                    _ if !state.in_placemark => {}
                    "Polygon" => {
                        state.outer = None;
                        state.holes.clear();
                    }
                    "Point" | "LineString" | "LinearRing" | "Track" => {
                        state.has_geometry = true;
                    }
                    "outerBoundaryIs" | "innerBoundaryIs" => {
                        state.ring = Some(Vec::new());
                        state.ring_is_outer = tag == "outerBoundaryIs";
                    }
                    "coordinates" => {
                        state.capture_coords = state.ring.is_some();
                        state.coords.clear();
                    }
                    "Data" => state.data_name = name_attribute(e)?,
                    "value" => {
                        state.capture_value =
                            label_field.is_some() && state.data_name.as_deref() == label_field;
                        state.value.clear();
                    }
                    "SimpleData" => {
                        state.capture_value =
                            label_field.is_some() && name_attribute(e)?.as_deref() == label_field;
                        state.value.clear();
                    }
                    _ => {}
                }
            }
            Ok(Event::End(ref e)) => {
                depth = depth.saturating_sub(1);
                let tag = String::from_utf8_lossy(e.local_name().as_ref()).into_owned();
                if !state.in_placemark {
                    buf.clear();
                    continue;
                }
                match tag.as_str() {
                    "Placemark" => {
                        if state.polys.is_empty() {
                            if state.has_geometry {
                                skipped_non_polygon += 1;
                            }
                        } else {
                            polygons.push((std::mem::take(&mut state.polys), state.label.take()));
                        }
                        state.in_placemark = false;
                    }
                    "Polygon" => state.finish_polygon(),
                    "outerBoundaryIs" | "innerBoundaryIs" => state.finish_ring()?,
                    "coordinates" => state.capture_coords = false,
                    "value" | "SimpleData" => {
                        if state.capture_value {
                            state.label = Some(state.value.trim().to_owned());
                        }
                        state.capture_value = false;
                    }
                    "Data" => state.data_name = None,
                    _ => {}
                }
            }
            Ok(Event::Text(ref e)) => {
                if state.capture_coords || state.capture_value {
                    let text = e.unescape().map_err(|err| err.to_string())?;
                    let target = if state.capture_coords {
                        &mut state.coords
                    } else {
                        &mut state.value
                    };
                    target.push_str(&text);
                    target.push(' ');
                }
            }
            Ok(Event::CData(ref e)) => {
                if state.capture_coords || state.capture_value {
                    let text = std::str::from_utf8(e.as_ref()).map_err(|err| err.to_string())?;
                    let target = if state.capture_coords {
                        &mut state.coords
                    } else {
                        &mut state.value
                    };
                    target.push_str(text);
                    target.push(' ');
                }
            }
            Ok(Event::Eof) => break,
            Err(e) => return Err(e.to_string()),
            _ => {}
        }
        buf.clear();
    }

    if depth != 0 || state.in_placemark {
        return Err("KML ended unexpectedly: the file is truncated or has unclosed tags".into());
    }

    Ok(KmlResult {
        polygons,
        skipped_non_polygon,
    })
}

/// Parse a KML `<coordinates>` text block (whitespace-separated `lng,lat[,alt]` tuples)
/// and append the resulting `(lng, lat)` pairs to *ring*.
///
/// Whitespace next to commas (`"0, 0 1, 0"`) is tolerated.
///
/// # Errors
/// Returns an error if a tuple is missing a component or a value is not a finite number.
fn parse_coordinates(text: &str, ring: &mut Ring) -> Result<(), String> {
    // Join tokens split by whitespace adjacent to a comma, e.g. "0," "0" -> "0,0".
    let mut tuples: Vec<String> = Vec::new();
    for token in text.split_whitespace() {
        match tuples.last_mut() {
            Some(last) if last.ends_with(',') || token.starts_with(',') => last.push_str(token),
            _ => tuples.push(token.to_owned()),
        }
    }
    for tuple in tuples {
        let mut parts = tuple.split(',');
        let mut next = |what: &str| -> Result<f64, String> {
            let raw = parts
                .next()
                .filter(|s| !s.is_empty())
                .ok_or_else(|| format!("coordinate '{tuple}' is missing {what}"))?;
            let value: f64 = raw
                .parse()
                .map_err(|_| format!("coordinate '{tuple}' has invalid {what} '{raw}'"))?;
            if value.is_finite() {
                Ok(value)
            } else {
                Err(format!("coordinate '{tuple}' has non-finite {what}"))
            }
        };
        let lng = next("longitude")?;
        let lat = next("latitude")?;
        ring.push((lng, lat));
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn placemark(body: &str) -> String {
        format!(
            r#"<?xml version="1.0"?><kml xmlns="http://www.opengis.net/kml/2.2"><Document><Placemark>{body}</Placemark></Document></kml>"#
        )
    }

    const SQUARE: &str = "<Polygon><outerBoundaryIs><LinearRing><coordinates>0,0 1,0 1,1 0,1 0,0</coordinates></LinearRing></outerBoundaryIs></Polygon>";

    #[test]
    fn reads_simple_data_labels() {
        let kml = placemark(&format!(
            r##"<ExtendedData><SchemaData schemaUrl="#s"><SimpleData name="kind">roof</SimpleData></SchemaData></ExtendedData>{SQUARE}"##
        ));
        let result = parse_kml(kml.as_bytes(), Some("kind")).unwrap();
        assert_eq!(result.polygons[0].1.as_deref(), Some("roof"));
    }

    #[test]
    fn unescapes_data_name_attribute() {
        let kml = placemark(&format!(
            r#"<ExtendedData><Data name="a&amp;b"><value>x</value></Data></ExtendedData>{SQUARE}"#
        ));
        let result = parse_kml(kml.as_bytes(), Some("a&b")).unwrap();
        assert_eq!(result.polygons[0].1.as_deref(), Some("x"));
    }

    #[test]
    fn tolerates_spaces_after_commas() {
        let mut ring = Ring::new();
        parse_coordinates("0, 0 1 ,0 1,1,5", &mut ring).unwrap();
        assert_eq!(ring, vec![(0.0, 0.0), (1.0, 0.0), (1.0, 1.0)]);
    }

    #[test]
    fn rejects_non_finite_coordinates() {
        let mut ring = Ring::new();
        assert!(parse_coordinates("nan,0", &mut ring).is_err());
        assert!(parse_coordinates("0,inf", &mut ring).is_err());
    }

    #[test]
    fn rejects_truncated_kml() {
        let kml = placemark(SQUARE);
        let truncated = &kml[..kml.len() - 30];
        assert!(parse_kml(truncated.as_bytes(), None).is_err());
    }

    #[test]
    fn puts_outer_ring_first_even_when_holes_come_first() {
        let kml = placemark("<Polygon><innerBoundaryIs><LinearRing><coordinates>0.2,0.2 0.4,0.2 0.4,0.4 0.2,0.2</coordinates></LinearRing></innerBoundaryIs><outerBoundaryIs><LinearRing><coordinates>0,0 1,0 1,1 0,0</coordinates></LinearRing></outerBoundaryIs></Polygon>");
        let result = parse_kml(kml.as_bytes(), None).unwrap();
        let rings = &result.polygons[0].0[0];
        assert_eq!(rings[0][1], (1.0, 0.0));
        assert_eq!(rings.len(), 2);
    }

    #[test]
    fn counts_non_polygon_placemarks_and_drops_degenerate_rings() {
        let kml = placemark("<Point><coordinates>0,0</coordinates></Point>");
        let result = parse_kml(kml.as_bytes(), None).unwrap();
        assert!(result.polygons.is_empty());
        assert_eq!(result.skipped_non_polygon, 1);

        let kml = placemark("<Polygon><outerBoundaryIs><LinearRing><coordinates>0,0 1,1</coordinates></LinearRing></outerBoundaryIs></Polygon>");
        assert!(parse_kml(kml.as_bytes(), None).unwrap().polygons.is_empty());
    }
}
