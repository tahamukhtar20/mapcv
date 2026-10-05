//! GeoTIFF georeferencing: GeoKeys to an EPSG code, model tags to an affine
//! transform, and the `GDAL_NODATA` tag.
//!
//! The rules follow what GDAL (and so rasterio) reports for the same file:
//!
//! * `ModelPixelScale` + `ModelTiepoint` take precedence over
//!   `ModelTransformation`. With the scale form, `e = -ScaleY`; a negative
//!   `ScaleY` is, as in GDAL's default (`GTIFF_HONOUR_NEGATIVE_SCALEY` unset),
//!   still read as a north-up image, i.e. `e = ScaleY`.
//! * For `RasterPixelIsPoint` the model coordinates name the *centre* of the
//!   tie-point pixel. GDAL (with its default `GTIFF_POINT_GEO_IGNORE=NO`)
//!   shifts the transform by half a pixel so that it maps pixel *corners*,
//!   like a `PixelIsArea` transform: `c -= (a + b) / 2`, `f -= (d + e) / 2`.
//!   The transform returned here is that corner-based transform, so the same
//!   pixel/world arithmetic applies to both raster types.
//! * Only CRSs given as an EPSG code (`ProjectedCSTypeGeoKey` or
//!   `GeographicTypeGeoKey`) are supported; user-defined CRSs are an error.

use super::ifd::{tag, Ifd};
use super::GeoTiffError;
use std::collections::BTreeMap;

/// GeoKey: `GTModelTypeGeoKey` (1 projected, 2 geographic, 3 geocentric).
const GT_MODEL_TYPE: u16 = 1024;
/// GeoKey: `GTRasterTypeGeoKey` (1 `PixelIsArea`, 2 `PixelIsPoint`).
const GT_RASTER_TYPE: u16 = 1025;
/// GeoKey: `GTCitationGeoKey`.
const GT_CITATION: u16 = 1026;
/// GeoKey: `GeographicTypeGeoKey`.
const GEOGRAPHIC_TYPE: u16 = 2048;
/// GeoKey: `GeogCitationGeoKey`.
const GEOG_CITATION: u16 = 2049;
/// GeoKey: `ProjectedCSTypeGeoKey`.
const PROJECTED_CS_TYPE: u16 = 3072;
/// GeoKey: `PCSCitationGeoKey`.
const PCS_CITATION: u16 = 3073;
/// GeoKey value meaning "user-defined".
const USER_DEFINED: u16 = 32767;

/// How pixel values relate to the model coordinates of the tie point.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum RasterType {
    /// `RasterPixelIsArea`: a pixel covers an area; the tie point is a corner.
    Area,
    /// `RasterPixelIsPoint`: a pixel is a point sample at the tie point.
    Point,
}

/// The value of one GeoKey.
#[derive(Clone, Debug, PartialEq)]
enum GeoKey {
    Short(Vec<u16>),
    Double(Vec<f64>),
    Ascii(String),
}

/// Georeferencing read from a GeoTIFF's first IFD.
#[derive(Clone, Debug)]
pub struct Georef {
    /// EPSG code of the CRS, or why there is none that this reader supports.
    pub epsg: Result<u32, String>,
    /// Affine transform `[a, b, c, d, e, f]` (pixel corner based), if any.
    pub transform: Option<[f64; 6]>,
    /// `PixelIsArea` or `PixelIsPoint`.
    pub raster_type: RasterType,
    /// The citation GeoKeys joined, if any (a human-readable CRS name).
    pub citation: Option<String>,
}

/// Parse the GeoKey directory into a map of key id to value.
fn parse_geokeys(ifd: &Ifd) -> Result<BTreeMap<u16, GeoKey>, String> {
    let Some(dir) = ifd
        .uints(tag::GEO_KEY_DIRECTORY)
        .map_err(|e| e.to_string())?
    else {
        return Ok(BTreeMap::new());
    };
    let dir: Vec<u16> = dir
        .into_iter()
        .map(|v| u16::try_from(v).map_err(|_| "GeoKeyDirectory value out of range".to_owned()))
        .collect::<Result<_, _>>()?;
    if dir.len() < 4 {
        return Err("GeoKeyDirectory tag is too short".to_owned());
    }
    let n = usize::from(dir[3]);
    if dir.len() < 4 + 4 * n {
        return Err(format!(
            "GeoKeyDirectory declares {n} keys but holds {} values",
            dir.len()
        ));
    }
    let doubles = ifd
        .doubles(tag::GEO_DOUBLE_PARAMS)
        .map_err(|e| e.to_string())?
        .unwrap_or_default();
    let ascii = ifd.bytes(tag::GEO_ASCII_PARAMS).unwrap_or_default();
    let mut keys = BTreeMap::new();
    for entry in dir[4..4 + 4 * n].as_chunks::<4>().0 {
        let (id, location, count, value) = (
            entry[0],
            entry[1],
            usize::from(entry[2]),
            usize::from(entry[3]),
        );
        let parsed = match location {
            0 => Some(GeoKey::Short(vec![entry[3]])),
            tag::GEO_KEY_DIRECTORY => dir
                .get(value..value + count)
                .map(|v| GeoKey::Short(v.to_vec())),
            tag::GEO_DOUBLE_PARAMS => doubles
                .get(value..value + count)
                .map(|v| GeoKey::Double(v.to_vec())),
            tag::GEO_ASCII_PARAMS => ascii.get(value..value + count).map(|v| {
                let text = String::from_utf8_lossy(v);
                // Each ASCII value ends with '|'.
                GeoKey::Ascii(text.trim_end_matches(['|', '\0']).to_owned())
            }),
            _ => None,
        };
        // Keys pointing outside their params tag are ignored, as GDAL's
        // libgeotiff does.
        if let Some(key) = parsed {
            keys.insert(id, key);
        }
    }
    Ok(keys)
}

fn short_key(keys: &BTreeMap<u16, GeoKey>, id: u16) -> Option<u16> {
    match keys.get(&id) {
        Some(GeoKey::Short(v)) => v.first().copied(),
        _ => None,
    }
}

fn ascii_key(keys: &BTreeMap<u16, GeoKey>, id: u16) -> Option<&str> {
    match keys.get(&id) {
        Some(GeoKey::Ascii(s)) if !s.is_empty() => Some(s),
        _ => None,
    }
}

/// Resolve the EPSG code from the GeoKeys.
fn epsg_from_keys(keys: &BTreeMap<u16, GeoKey>, citation: Option<&str>) -> Result<u32, String> {
    let named = citation.map_or_else(String::new, |c| format!(" ({c})"));
    let code_or_error = |key: u16, key_name: &str, kind: &str| -> Result<u32, String> {
        match short_key(keys, key) {
            Some(code @ 1..USER_DEFINED) => Ok(u32::from(code)),
            Some(USER_DEFINED) => Err(format!(
                "the GeoTIFF has a user-defined {kind} CRS{named} ({key_name} = 32767); only \
                 CRSs identified by an EPSG code are supported. Reproject or re-tag the file \
                 with an EPSG CRS, e.g. `gdalwarp -t_srs EPSG:<code>`"
            )),
            Some(code) => Err(format!(
                "the GeoTIFF's {kind} CRS code {code}{named} ({key_name}) is not an EPSG code; \
                 only CRSs identified by an EPSG code are supported"
            )),
            None => Err(format!(
                "the GeoTIFF declares a {kind} CRS{named} without {key_name}; only CRSs \
                 identified by an EPSG code are supported"
            )),
        }
    };
    let model = short_key(keys, GT_MODEL_TYPE);
    match model {
        Some(1) => code_or_error(PROJECTED_CS_TYPE, "ProjectedCSTypeGeoKey", "projected"),
        Some(2) => code_or_error(GEOGRAPHIC_TYPE, "GeographicTypeGeoKey", "geographic"),
        Some(3) => Err(format!(
            "the GeoTIFF has a geocentric CRS{named}, which is not supported"
        )),
        _ if keys.contains_key(&PROJECTED_CS_TYPE) => {
            code_or_error(PROJECTED_CS_TYPE, "ProjectedCSTypeGeoKey", "projected")
        }
        _ if keys.contains_key(&GEOGRAPHIC_TYPE) => {
            code_or_error(GEOGRAPHIC_TYPE, "GeographicTypeGeoKey", "geographic")
        }
        Some(other) => Err(format!(
            "the GeoTIFF has an unknown model type {other} (GTModelTypeGeoKey)"
        )),
        None => Err("the file has no CRS (no GeoKeyDirectory or no CRS GeoKeys)".to_owned()),
    }
}

/// The affine transform, following GDAL's reading of the model tags.
fn transform_from_tags(ifd: &Ifd, raster_type: RasterType) -> Result<Option<[f64; 6]>, String> {
    let scale = ifd
        .doubles(tag::MODEL_PIXEL_SCALE)
        .map_err(|e| e.to_string())?;
    let tiepoints = ifd
        .doubles(tag::MODEL_TIEPOINT)
        .map_err(|e| e.to_string())?;
    let matrix = ifd
        .doubles(tag::MODEL_TRANSFORMATION)
        .map_err(|e| e.to_string())?;
    let mut t = match (scale, tiepoints, matrix) {
        (Some(scale), tiepoints, _) if scale.len() >= 2 && scale[0] != 0.0 && scale[1] != 0.0 => {
            let Some(tp) = tiepoints.filter(|tp| tp.len() >= 6) else {
                return Ok(None);
            };
            let a = scale[0];
            // GDAL treats a negative ScaleY as a north-up image too.
            let e = if scale[1] < 0.0 { scale[1] } else { -scale[1] };
            [a, 0.0, tp[3] - tp[0] * a, 0.0, e, tp[4] - tp[1] * e]
        }
        (_, _, Some(m)) if m.len() == 16 => [m[0], m[1], m[3], m[4], m[5], m[7]],
        _ => return Ok(None),
    };
    if raster_type == RasterType::Point {
        t[2] -= t[0] * 0.5 + t[1] * 0.5;
        t[5] -= t[3] * 0.5 + t[4] * 0.5;
    }
    if t.iter().any(|v| !v.is_finite()) {
        return Err("the GeoTIFF's affine transform is not finite".to_owned());
    }
    Ok(Some(t))
}

/// Read the georeferencing of `ifd`. Malformed or unsupported georeferencing
/// is reported in `epsg` instead of failing, so pixels stay readable.
///
/// # Errors
/// Returns [`GeoTiffError::Invalid`] when the model tags are malformed.
pub fn read_georef(ifd: &Ifd) -> Result<Georef, GeoTiffError> {
    let (keys, key_error) = match parse_geokeys(ifd) {
        Ok(keys) => (keys, None),
        Err(e) => (BTreeMap::new(), Some(e)),
    };
    let raster_type = if short_key(&keys, GT_RASTER_TYPE) == Some(2) {
        RasterType::Point
    } else {
        RasterType::Area
    };
    let citations: Vec<&str> = [PCS_CITATION, GEOG_CITATION, GT_CITATION]
        .iter()
        .filter_map(|&k| ascii_key(&keys, k))
        .collect();
    let citation = (!citations.is_empty()).then(|| citations.join("; "));
    let epsg = match key_error {
        Some(e) => Err(format!("the GeoKeyDirectory is malformed: {e}")),
        None => epsg_from_keys(&keys, citation.as_deref()),
    };
    let transform = transform_from_tags(ifd, raster_type).map_err(GeoTiffError::Invalid)?;
    Ok(Georef {
        epsg,
        transform,
        raster_type,
        citation,
    })
}

/// Parse the `GDAL_NODATA` tag (`"0"`, `"-9999"`, `"nan"`, `"-inf"`, ...).
///
/// # Errors
/// Returns [`GeoTiffError::Invalid`] when the tag is not a number.
pub fn read_nodata(ifd: &Ifd) -> Result<Option<f64>, GeoTiffError> {
    let Some(text) = ifd.ascii(tag::GDAL_NODATA) else {
        return Ok(None);
    };
    let text = text.trim();
    if text.is_empty() {
        return Ok(None);
    }
    text.parse::<f64>()
        .map(Some)
        .map_err(|_| GeoTiffError::Invalid(format!("GDAL_NODATA tag {text:?} is not a number")))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn keys(entries: &[(u16, u16)]) -> BTreeMap<u16, GeoKey> {
        entries
            .iter()
            .map(|&(k, v)| (k, GeoKey::Short(vec![v])))
            .collect()
    }

    #[test]
    fn epsg_codes() {
        assert_eq!(
            epsg_from_keys(
                &keys(&[(GT_MODEL_TYPE, 1), (PROJECTED_CS_TYPE, 32633)]),
                None
            ),
            Ok(32633)
        );
        assert_eq!(
            epsg_from_keys(&keys(&[(GT_MODEL_TYPE, 2), (GEOGRAPHIC_TYPE, 4326)]), None),
            Ok(4326)
        );
        // Without a model type the CRS key decides.
        assert_eq!(
            epsg_from_keys(&keys(&[(PROJECTED_CS_TYPE, 3857)]), None),
            Ok(3857)
        );
    }

    #[test]
    fn user_defined_crs_is_a_clear_error() {
        let err = epsg_from_keys(
            &keys(&[(GT_MODEL_TYPE, 1), (PROJECTED_CS_TYPE, USER_DEFINED)]),
            Some("My LCC"),
        )
        .unwrap_err();
        assert!(err.contains("user-defined projected CRS (My LCC)"), "{err}");
        let err = epsg_from_keys(&keys(&[(GT_MODEL_TYPE, 1)]), None).unwrap_err();
        assert!(err.contains("without ProjectedCSTypeGeoKey"), "{err}");
        let err = epsg_from_keys(&keys(&[(GT_MODEL_TYPE, 3)]), None).unwrap_err();
        assert!(err.contains("geocentric"), "{err}");
        assert!(epsg_from_keys(&BTreeMap::new(), None).is_err());
    }
}
