//! Patch anchor generator.
//!
//! Computes top-left `(row, col)` positions for patch extraction from an
//! image of known `(height, width)` dimensions.
//!
//! Two sampling modes:
//! - **Grid / sliding window**: regular stride-based grid.
//! - **Random**: uniform random sampling via a seeded xorshift64 PRNG.
//!
//! Three edge strategies control what happens when a patch would extend past
//! the image boundary:
//! - `"pad"`: include the anchor; the caller pads the extracted patch.
//! - `"drop"`: skip anchors where the patch would extend out of bounds.
//! - `"shift"`: replace the last out-of-bounds anchor with one shifted inward
//!   so the patch stays within the image (may introduce overlap).

// u64 -> usize truncation is intentional (modulo reduction for PRNG output).
#![allow(clippy::cast_possible_truncation)]

/// xorshift64 PRNG step.
fn xorshift64(state: &mut u64) -> u64 {
    *state ^= *state << 13;
    *state ^= *state >> 7;
    *state ^= *state << 17;
    *state
}

fn uniform(state: &mut u64, n: usize) -> usize {
    debug_assert!(n > 0);
    (xorshift64(state) % n as u64) as usize
}

/// Compute anchor positions along one dimension for grid sampling.
fn dim_anchors(dim: usize, patch_size: usize, stride: usize, strategy: &str) -> Vec<usize> {
    match strategy {
        "drop" => {
            let mut v = Vec::new();
            let mut p = 0usize;
            while p + patch_size <= dim {
                v.push(p);
                p += stride;
            }
            v
        }
        "shift" => {
            let mut v = Vec::new();
            let mut p = 0usize;
            while p + patch_size <= dim {
                v.push(p);
                p += stride;
            }
            // Append a final anchor shifted inward if needed so that all
            // pixels are covered by at least one patch.
            if dim >= patch_size {
                let last = dim - patch_size;
                if v.last().copied().is_none_or(|q| q < last) {
                    v.push(last);
                }
            } else if v.is_empty() {
                // Image smaller than patch_size: single anchor at 0.
                v.push(0);
            }
            v
        }
        _ => {
            // "pad" (default): anchor at every stride step while inside image.
            let mut v = Vec::new();
            let mut p = 0usize;
            while p < dim {
                v.push(p);
                p += stride;
            }
            v
        }
    }
}

/// Generate grid (or sliding-window) anchor positions.
///
/// Returns `(row, col)` top-left corners for `patch_size x patch_size` patches
/// sampled with the given `stride` across an `height x width` image.
///
/// # Errors
/// Returns an error if `patch_size`, `stride`, `height`, or `width` is zero.
pub fn grid_anchors(
    height: usize,
    width: usize,
    patch_size: usize,
    stride: usize,
    strategy: &str,
) -> Result<Vec<(usize, usize)>, String> {
    if patch_size == 0 {
        return Err("patch_size must be > 0".to_string());
    }
    if stride == 0 {
        return Err("stride must be > 0".to_string());
    }
    if height == 0 || width == 0 {
        return Err("height and width must be > 0".to_string());
    }
    let rows = dim_anchors(height, patch_size, stride, strategy);
    let cols = dim_anchors(width, patch_size, stride, strategy);
    let mut anchors = Vec::with_capacity(rows.len() * cols.len());
    for &row in &rows {
        for &col in &cols {
            anchors.push((row, col));
        }
    }
    Ok(anchors)
}

/// Generate `count` random anchor positions using a seeded xorshift64 PRNG.
///
/// For `"drop"` and `"shift"` strategies the anchor is clamped so the patch
/// stays within the image. For `"pad"` the anchor may be anywhere in
/// `[0, height) x [0, width)`.
///
/// # Errors
/// Returns an error if `patch_size`, `height`, or `width` is zero.
pub fn random_anchors(
    height: usize,
    width: usize,
    patch_size: usize,
    count: usize,
    seed: u64,
    strategy: &str,
) -> Result<Vec<(usize, usize)>, String> {
    if patch_size == 0 {
        return Err("patch_size must be > 0".to_string());
    }
    if height == 0 || width == 0 {
        return Err("height and width must be > 0".to_string());
    }
    if count == 0 {
        return Ok(Vec::new());
    }
    // Mix seed to avoid the all-zeros xorshift state.
    let mut state = seed ^ 0x9e37_79b9_7f4a_7c15;
    if state == 0 {
        state = 1;
    }
    let mut anchors = Vec::with_capacity(count);
    match strategy {
        "drop" | "shift" => {
            // Clamp so patch stays in bounds.
            let row_max = height.saturating_sub(patch_size).saturating_add(1).max(1);
            let col_max = width.saturating_sub(patch_size).saturating_add(1).max(1);
            for _ in 0..count {
                anchors.push((uniform(&mut state, row_max), uniform(&mut state, col_max)));
            }
        }
        _ => {
            // "pad": anchor anywhere in [0, height) x [0, width).
            for _ in 0..count {
                anchors.push((uniform(&mut state, height), uniform(&mut state, width)));
            }
        }
    }
    Ok(anchors)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn grid_drop_only_full_patches() {
        let anchors = grid_anchors(10, 10, 4, 4, "drop").unwrap();
        assert_eq!(anchors.len(), 4); // 2 rows x 2 cols
        for (r, c) in &anchors {
            assert!(r + 4 <= 10 && c + 4 <= 10);
        }
    }

    #[test]
    fn grid_pad_includes_edge_anchors() {
        let anchors = grid_anchors(10, 10, 4, 4, "pad").unwrap();
        // rows: 0, 4, 8; cols: 0, 4, 8 -> 9 patches
        assert_eq!(anchors.len(), 9);
    }

    #[test]
    fn grid_shift_all_in_bounds() {
        let anchors = grid_anchors(10, 10, 4, 4, "shift").unwrap();
        for (r, c) in &anchors {
            assert!(r + 4 <= 10 && c + 4 <= 10);
        }
        let max_row = anchors.iter().map(|(r, _)| *r).max().unwrap();
        assert_eq!(max_row, 6); // 10 - 4 = 6
    }

    #[test]
    fn grid_exact_fit_no_duplicate_anchor() {
        // 8x8 image, patch=4, stride=4: anchors at 0 and 4 only.
        let anchors = grid_anchors(8, 8, 4, 4, "shift").unwrap();
        assert_eq!(anchors.len(), 4); // [0,4] x [0,4]
    }

    #[test]
    fn random_drop_in_bounds() {
        let anchors = random_anchors(100, 100, 32, 50, 42, "drop").unwrap();
        assert_eq!(anchors.len(), 50);
        for (r, c) in &anchors {
            assert!(r + 32 <= 100 && c + 32 <= 100);
        }
    }

    #[test]
    fn random_same_seed_reproducible() {
        let a = random_anchors(100, 100, 32, 10, 7, "pad").unwrap();
        let b = random_anchors(100, 100, 32, 10, 7, "pad").unwrap();
        assert_eq!(a, b);
    }

    #[test]
    fn zero_count_is_empty() {
        let anchors = random_anchors(100, 100, 32, 0, 42, "drop").unwrap();
        assert!(anchors.is_empty());
    }

    #[test]
    fn zero_patch_size_errors() {
        assert!(grid_anchors(10, 10, 0, 4, "pad").is_err());
        assert!(random_anchors(10, 10, 0, 10, 42, "pad").is_err());
    }
}
