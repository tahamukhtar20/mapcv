//! Parallel patch writer: encodes and writes image/mask patches using rayon.

use image::codecs::jpeg::JpegEncoder;
use image::codecs::png::{CompressionType, FilterType, PngEncoder};
use image::{ImageBuffer, Luma, Rgb};
use rayon::prelude::*;
use std::collections::HashMap;
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
    /// Pixel counts per class label (string key, e.g. `"0"`, `"1"`).
    pub class_counts: HashMap<String, u64>,
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
/// An existing file is left untouched (resume support).
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

            if !img_path.exists() {
                encode_image(img_slice, patch_size, &img_path, image_format, jpg_quality)?;
            }

            let mask_filename = if has_mask {
                let msk_fname = format!("patch_{global_idx:07}.png");
                let msk_path: PathBuf = masks_dir.join(&msk_fname);
                if !msk_path.exists() {
                    let msk_slice =
                        &mask_data[local_i * msk_patch_bytes..(local_i + 1) * msk_patch_bytes];
                    encode_mask(msk_slice, patch_size, &msk_path)?;
                }
                Some(msk_fname)
            } else {
                None
            };

            let class_counts = if has_mask {
                let msk_slice =
                    &mask_data[local_i * msk_patch_bytes..(local_i + 1) * msk_patch_bytes];
                compute_class_counts(msk_slice)
            } else {
                HashMap::new()
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
        enc.encode(data, ps, ps, image::ColorType::Rgb8)
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

fn compute_class_counts(mask: &[u8]) -> HashMap<String, u64> {
    let mut counts: HashMap<String, u64> = HashMap::new();
    for &v in mask {
        *counts.entry(v.to_string()).or_insert(0) += 1;
    }
    counts
}

#[allow(clippy::cast_precision_loss)]
fn compute_empty_ratio(img: &[u8], patch_size: usize) -> f64 {
    let n_pixels = patch_size * patch_size;
    let empty = (0..n_pixels)
        .filter(|&i| img[i * 3] == 0 && img[i * 3 + 1] == 0 && img[i * 3 + 2] == 0)
        .count();
    empty as f64 / n_pixels as f64
}
