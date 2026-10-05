//! Parallel patch writer: encodes and writes image/mask patches using rayon.

use image::codecs::jpeg::JpegEncoder;
use image::codecs::png::{CompressionType, FilterType, PngEncoder};
use image::{ImageBuffer, Luma, Rgb};
use rayon::prelude::*;
use std::fs::File;
use std::io::BufWriter;
use std::path::{Path, PathBuf};

/// Per-patch result returned after writing to disk.
pub struct PatchResult {
    /// Image filename (relative, e.g. `patch_0000001.png`).
    pub filename: String,
    /// Mask filename when a mask was provided (always `.png`).
    pub mask_filename: Option<String>,
    /// Top-left row of the patch in the source image.
    pub row: usize,
    /// Top-left column of the patch in the source image.
    pub col: usize,
    /// Whether the patch was zero-padded at the image boundary.
    pub padded: bool,
    /// Index of the strip this patch belongs to.
    pub strip_index: usize,
    /// `(class_id, pixel_count)` pairs for the classes present in the mask,
    /// in ascending class-id order (empty when no mask was provided).
    pub class_counts: Vec<(u8, u64)>,
    /// Fraction of all-zero (black) pixels in the image patch.
    pub empty_ratio: f64,
}

/// Check that `image_format` is one the writer encodes and `jpg_quality` is in `1..=100`.
///
/// Only `"png"` and `"jpg"` are accepted; anything else (for example `"jpeg"`)
/// would otherwise be written as PNG under a misleading name.
///
/// # Errors
/// Returns a message naming the unsupported value.
pub fn check_format(image_format: &str, jpg_quality: u8) -> Result<(), String> {
    if image_format != "png" && image_format != "jpg" {
        return Err(format!(
            "image_format must be 'png' or 'jpg', got '{image_format}'"
        ));
    }
    if !(1..=100).contains(&jpg_quality) {
        return Err(format!("jpg_quality must be in 1..=100, got {jpg_quality}"));
    }
    Ok(())
}

/// Check that the flat buffers and `meta` all describe `n_patches` patches of
/// `patch_size x patch_size` pixels, so that slicing them cannot go out of bounds.
fn check_buffers(
    image_len: usize,
    mask_len: Option<usize>,
    n_patches: usize,
    patch_size: usize,
    meta_len: usize,
    start_idx: usize,
) -> Result<(), String> {
    if patch_size == 0 {
        return Err("patch_size must be > 0".to_string());
    }
    if meta_len != n_patches {
        return Err(format!(
            "meta has {meta_len} entries but {n_patches} patches were given"
        ));
    }
    let pixels = patch_size
        .checked_mul(patch_size)
        .and_then(|p| p.checked_mul(n_patches))
        .ok_or_else(|| "patch buffer size overflows usize".to_string())?;
    if pixels.checked_mul(3) != Some(image_len) {
        return Err(format!(
            "image buffer has {image_len} bytes, expected {n_patches} x {patch_size} x {patch_size} x 3"
        ));
    }
    if let Some(len) = mask_len {
        if len != pixels {
            return Err(format!(
                "mask buffer has {len} bytes, expected {n_patches} x {patch_size} x {patch_size}"
            ));
        }
    }
    if start_idx.checked_add(n_patches).is_none() {
        return Err("start_idx + n_patches overflows usize".to_string());
    }
    Ok(())
}

/// Write `n_patches` image (and optionally mask) patches to disk in parallel.
///
/// # Errors
/// Returns an `Err` string if the buffers, `meta` or format arguments do not
/// match `n_patches` and `patch_size`, or if any patch fails to encode or
/// write to disk.
///
/// `image_data` is a flat `(N, ps, ps, 3)` C-order buffer.
/// `mask_data`  is a flat `(N, ps, ps)` C-order buffer (ignored when `has_mask` is false).
/// `meta`       is a slice of `(row, col, padded)` per patch, in the same order.
///
/// Files are named `patch_{global_idx:07}.{ext}` where `global_idx = start_idx + local_i`.
/// Masks are always written as PNG regardless of `image_format`.
/// Existing files are overwritten: the caller indexes from the manifest length, so any file
/// at these indices is an orphan from an interrupted run, never a recorded patch.
#[allow(clippy::too_many_arguments)]
pub fn write_patches(
    image_data: &[u8],
    mask_data: &[u8],
    has_mask: bool,
    n_patches: usize,
    patch_size: usize,
    meta: &[(usize, usize, bool)],
    start_idx: usize,
    strip_index: usize,
    images_dir: &Path,
    masks_dir: &Path,
    image_format: &str,
    jpg_quality: u8,
) -> Result<Vec<PatchResult>, String> {
    check_format(image_format, jpg_quality)?;
    check_buffers(
        image_data.len(),
        has_mask.then_some(mask_data.len()),
        n_patches,
        patch_size,
        meta.len(),
        start_idx,
    )?;
    let img_patch_bytes = patch_size * patch_size * 3;
    let msk_patch_bytes = patch_size * patch_size;
    let ext = if image_format == "jpg" { "jpg" } else { "png" };

    (0..n_patches)
        .into_par_iter()
        .map(|local_i| {
            let global_idx = start_idx + local_i;
            let (row, col, padded) = meta[local_i];

            let img_fname = format!("patch_{global_idx:07}.{ext}");
            let img_path = images_dir.join(&img_fname);

            let img_slice = &image_data[local_i * img_patch_bytes..(local_i + 1) * img_patch_bytes];

            encode_image(img_slice, patch_size, &img_path, image_format, jpg_quality)?;

            let mask_filename = if has_mask {
                let msk_fname = format!("patch_{global_idx:07}.png");
                let msk_path: PathBuf = masks_dir.join(&msk_fname);
                let msk_slice =
                    &mask_data[local_i * msk_patch_bytes..(local_i + 1) * msk_patch_bytes];
                encode_mask(msk_slice, patch_size, &msk_path)?;
                Some(msk_fname)
            } else {
                None
            };

            let class_counts = if has_mask {
                let msk_slice =
                    &mask_data[local_i * msk_patch_bytes..(local_i + 1) * msk_patch_bytes];
                compute_class_counts(msk_slice)
            } else {
                Vec::new()
            };
            let empty_ratio = compute_empty_ratio(img_slice, patch_size);

            Ok(PatchResult {
                filename: img_fname,
                mask_filename,
                row,
                col,
                padded,
                strip_index,
                class_counts,
                empty_ratio,
            })
        })
        .collect()
}

/// Encode *data* (raw RGB bytes) as PNG or JPEG and write to *path*.
fn encode_image(
    data: &[u8],
    patch_size: usize,
    path: &Path,
    format: &str,
    quality: u8,
) -> Result<(), String> {
    let ps = u32::try_from(patch_size)
        .map_err(|_| format!("patch size {patch_size} is too large to encode"))?;
    if format == "jpg" {
        let file = File::create(path).map_err(|e| e.to_string())?;
        let mut enc = JpegEncoder::new_with_quality(BufWriter::new(file), quality);
        enc.encode(data, ps, ps, image::ExtendedColorType::Rgb8)
            .map_err(|e| e.to_string())?;
    } else {
        let file = File::create(path).map_err(|e| e.to_string())?;
        let enc = PngEncoder::new_with_quality(
            BufWriter::new(file),
            CompressionType::Fast,
            FilterType::Sub,
        );
        let img: ImageBuffer<Rgb<u8>, _> =
            ImageBuffer::from_raw(ps, ps, data.to_vec()).ok_or("image buffer alloc failed")?;
        img.write_with_encoder(enc).map_err(|e| e.to_string())?;
    }
    Ok(())
}

/// Encode *data* (raw single-channel u8 bytes) as a lossless PNG and write to *path*.
fn encode_mask(data: &[u8], patch_size: usize, path: &Path) -> Result<(), String> {
    let ps = u32::try_from(patch_size)
        .map_err(|_| format!("patch size {patch_size} is too large to encode"))?;
    let file = File::create(path).map_err(|e| e.to_string())?;
    let enc =
        PngEncoder::new_with_quality(BufWriter::new(file), CompressionType::Fast, FilterType::Sub);
    let img: ImageBuffer<Luma<u8>, _> =
        ImageBuffer::from_raw(ps, ps, data.to_vec()).ok_or("mask buffer alloc failed")?;
    img.write_with_encoder(enc).map_err(|e| e.to_string())
}

/// Count pixels by class label in *mask*.
///
/// Uses a fixed-size 256-bin histogram (one bin per possible `u8` value), which
/// avoids hashing per pixel and yields a deterministic order: the result lists
/// `(class_id, count)` for every class present, in ascending class-id order.
fn compute_class_counts(mask: &[u8]) -> Vec<(u8, u64)> {
    let mut hist = [0u64; 256];
    for &v in mask {
        hist[usize::from(v)] += 1;
    }
    (0..=u8::MAX)
        .zip(hist)
        .filter(|&(_, count)| count > 0)
        .collect()
}

/// Return the fraction of all-black `(0, 0, 0)` pixels in the RGB image patch.
#[allow(clippy::cast_precision_loss)]
fn compute_empty_ratio(img: &[u8], patch_size: usize) -> f64 {
    let n_pixels = patch_size * patch_size;
    let empty = (0..n_pixels)
        .filter(|&i| img[i * 3] == 0 && img[i * 3 + 1] == 0 && img[i * 3 + 2] == 0)
        .count();
    empty as f64 / n_pixels as f64
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn class_counts_are_ascending_and_skip_absent_classes() {
        let mask = [10u8, 2, 255, 2, 0, 10, 10];
        assert_eq!(
            compute_class_counts(&mask),
            vec![(0, 1), (2, 2), (10, 3), (255, 1)]
        );
    }

    #[test]
    fn class_counts_empty_mask() {
        assert_eq!(compute_class_counts(&[]), Vec::<(u8, u64)>::new());
    }

    /// Run `write_patches` on PNG output and return its error message, or `""`.
    fn write_err(
        image: &[u8],
        mask: Option<&[u8]>,
        n: usize,
        ps: usize,
        meta_len: usize,
    ) -> String {
        let dir = std::env::temp_dir();
        let meta = vec![(0, 0, false); meta_len];
        write_patches(
            image,
            mask.unwrap_or(&[]),
            mask.is_some(),
            n,
            ps,
            &meta,
            0,
            0,
            &dir,
            &dir,
            "png",
            95,
        )
        .err()
        .unwrap_or_default()
    }

    #[test]
    fn mismatched_buffers_are_errors_not_panics() {
        let image = vec![0u8; 2 * 4 * 4 * 3];
        assert!(write_err(&image, None, 2, 4, 1).contains("meta has 1 entries"));
        assert!(write_err(&image, None, 3, 4, 3).contains("image buffer"));
        assert!(write_err(&image, Some(&[0u8; 4 * 4]), 2, 4, 2).contains("mask buffer"));
        assert!(write_err(&image, None, 2, 0, 2).contains("patch_size"));
    }

    #[test]
    fn unknown_format_and_quality_are_rejected() {
        assert!(check_format("jpeg", 95).is_err());
        assert!(check_format("PNG", 95).is_err());
        assert!(check_format("jpg", 0).is_err());
        assert!(check_format("jpg", 101).is_err());
        assert!(check_format("png", 95).is_ok());
        assert!(check_format("jpg", 1).is_ok());
    }
}
