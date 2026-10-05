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

/// Write `n_patches` image (and optionally mask) patches to disk in parallel.
///
/// # Errors
/// Returns an `Err` string if any patch fails to encode or write to disk.
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
#[allow(clippy::cast_possible_truncation)]
fn encode_image(
    data: &[u8],
    patch_size: usize,
    path: &Path,
    format: &str,
    quality: u8,
) -> Result<(), String> {
    let ps = patch_size as u32;
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
#[allow(clippy::cast_possible_truncation)]
fn encode_mask(data: &[u8], patch_size: usize, path: &Path) -> Result<(), String> {
    let ps = patch_size as u32;
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
    use super::compute_class_counts;

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
}
