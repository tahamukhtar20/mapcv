//! Patch anchor generator.
//!
//! Computes top-left `(row, col)` positions for patch extraction from an
//! image of known `(height, width)` dimensions.
//!
//! Two sampling modes:
//! - **Grid / sliding window**: regular stride-based grid.
//! - **Random**: uniform sampling *without replacement* of distinct anchors
//!   via a seeded splitmix64 PRNG.
//!
//! Three edge strategies control what happens when a patch would extend past
//! the image boundary:
//! - `"pad"`: include the anchor; the caller pads the extracted patch.
//! - `"drop"`: skip anchors where the patch would extend out of bounds.
//! - `"shift"`: replace the last out-of-bounds anchor with one shifted inward
//!   so the patch stays within the image (may introduce overlap).
//!
//! Random sampling treats every strategy the same way on rasters at least as
//! large as the patch: anchors are drawn uniformly from the positions whose
//! patch fits inside the raster. Only on a raster smaller than the patch do
//! the strategies differ (`"drop"` yields nothing, the others a single padded
//! anchor at `(0, 0)`), mirroring the grid sampler.

// u64 -> usize truncation is intentional (modulo reduction for PRNG output).
#![allow(clippy::cast_possible_truncation)]

/// splitmix64 step: a small, well-mixed PRNG that is safe for any seed
/// (including 0 and runs of consecutive seeds).
fn splitmix64(state: &mut u64) -> u64 {
    *state = state.wrapping_add(0x9e37_79b9_7f4a_7c15);
    let mut z = *state;
    z = (z ^ (z >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
    z ^ (z >> 31)
}

/// Uniform integer in `[0, n)` (multiply-high reduction, no modulo bias to
/// speak of for `n` far below 2^64).
fn uniform(state: &mut u64, n: u64) -> u64 {
    debug_assert!(n > 0);
    ((u128::from(splitmix64(state)) * u128::from(n)) >> 64) as u64
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

/// Number of positions along one axis from which random anchors are drawn.
fn random_axis_len(dim: usize, patch_size: usize, strategy: &str) -> usize {
    if dim >= patch_size {
        dim - patch_size + 1
    } else {
        // Raster smaller than the patch: nothing fits when dropping, else a
        // single padded anchor at 0, as in the grid sampler.
        usize::from(strategy != "drop")
    }
}

/// Number of distinct anchors [`random_anchors`] can return for a raster.
///
/// This is the cap on `count`: asking for more returns every one of them.
///
/// # Errors
/// Returns an error if `patch_size`, `height`, or `width` is zero, or the
/// number of anchors does not fit in 64 bits.
pub fn random_anchor_capacity(
    height: usize,
    width: usize,
    patch_size: usize,
    strategy: &str,
) -> Result<usize, String> {
    if patch_size == 0 {
        return Err("patch_size must be > 0".to_string());
    }
    if height == 0 || width == 0 {
        return Err("height and width must be > 0".to_string());
    }
    let rows = random_axis_len(height, patch_size, strategy) as u64;
    let cols = random_axis_len(width, patch_size, strategy) as u64;
    rows.checked_mul(cols)
        .and_then(|n| usize::try_from(n).ok())
        .ok_or_else(|| "raster is too large to enumerate anchors".to_string())
}

/// Generate up to `count` distinct random anchor positions using a seeded PRNG.
///
/// Anchors are drawn uniformly *without replacement* from every position whose
/// patch fits inside the image (all strategies; see the module docs for
/// rasters smaller than the patch), so no anchor repeats and the last
/// rows/columns are sampled as often as any other. If `count` is at least the
/// number of such positions, all of them are returned (see
/// [`random_anchor_capacity`]). The order is a seeded shuffle, so any prefix
/// is itself a uniform sample. The result is fully determined by the
/// arguments.
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
    let capacity = random_anchor_capacity(height, width, patch_size, strategy)? as u64;
    let cols = random_axis_len(width, patch_size, strategy) as u64;
    let wanted = (count as u64).min(capacity);
    if wanted == 0 {
        return Ok(Vec::new());
    }
    let mut state = seed;
    let mut flat: Vec<u64> = if wanted == capacity {
        (0..capacity).collect()
    } else {
        // Floyd's algorithm: `wanted` distinct values from [0, capacity) in
        // O(wanted) time and memory. The Vec keeps insertion order so the
        // output never depends on hash iteration order.
        let mut taken = std::collections::HashSet::with_capacity(wanted as usize);
        let mut picked = Vec::with_capacity(wanted as usize);
        for j in (capacity - wanted)..capacity {
            let t = uniform(&mut state, j + 1);
            let value = if taken.insert(t) {
                t
            } else {
                taken.insert(j);
                j
            };
            picked.push(value);
        }
        picked
    };
    // Fisher-Yates shuffle (Floyd's order is not uniform, and the full
    // enumeration is row-major).
    for i in (1..flat.len()).rev() {
        let j = uniform(&mut state, i as u64 + 1) as usize;
        flat.swap(i, j);
    }
    Ok(flat
        .into_iter()
        .map(|v| ((v / cols) as usize, (v % cols) as usize))
        .collect())
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

    use std::collections::HashSet;

    fn distinct(anchors: &[(usize, usize)]) -> usize {
        anchors.iter().collect::<HashSet<_>>().len()
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
    fn random_never_repeats_an_anchor() {
        for strategy in ["drop", "pad", "shift"] {
            for seed in 0..50 {
                // 300x300 with 256px patches has only 45*45 = 2025 positions.
                let anchors = random_anchors(300, 300, 256, 1500, seed, strategy).unwrap();
                assert_eq!(anchors.len(), 1500);
                assert_eq!(distinct(&anchors), 1500, "{strategy} seed {seed}");
            }
        }
    }

    #[test]
    fn random_caps_count_at_distinct_anchors() {
        // Regression: 100 draws from this raster used to give 96 unique ones.
        let capacity = random_anchor_capacity(300, 300, 256, "drop").unwrap();
        assert_eq!(capacity, 45 * 45);
        let anchors = random_anchors(300, 300, 256, 100_000, 1, "drop").unwrap();
        assert_eq!(anchors.len(), capacity);
        assert_eq!(distinct(&anchors), capacity);
        for (r, c) in &anchors {
            assert!(*r <= 44 && *c <= 44);
        }
        // Exactly at capacity and one under it.
        assert_eq!(random_anchors(10, 10, 8, 9, 3, "drop").unwrap().len(), 9);
        assert_eq!(random_anchors(10, 10, 8, 8, 3, "drop").unwrap().len(), 8);
        assert_eq!(random_anchors(10, 10, 8, 10, 3, "drop").unwrap().len(), 9);
    }

    #[test]
    fn random_raster_smaller_than_patch() {
        assert_eq!(
            random_anchors(20, 20, 32, 5, 1, "drop").unwrap(),
            Vec::new()
        );
        assert_eq!(random_anchor_capacity(20, 20, 32, "drop").unwrap(), 0);
        for strategy in ["pad", "shift"] {
            let anchors = random_anchors(20, 20, 32, 5, 1, strategy).unwrap();
            assert_eq!(anchors, vec![(0, 0)]);
        }
        // Narrow in one axis only.
        assert_eq!(
            random_anchors(100, 20, 32, 5, 1, "drop").unwrap(),
            Vec::new()
        );
        let anchors = random_anchors(40, 20, 32, 100, 1, "pad").unwrap();
        assert_eq!(anchors.len(), 9);
        assert!(anchors.iter().all(|&(r, c)| r <= 8 && c == 0));
    }

    #[test]
    fn random_all_strategies_stay_in_bounds_on_large_rasters() {
        for strategy in ["drop", "pad", "shift"] {
            let anchors = random_anchors(100, 80, 32, 300, 9, strategy).unwrap();
            assert_eq!(anchors.len(), 300);
            for (r, c) in &anchors {
                assert!(r + 32 <= 100 && c + 32 <= 80, "{strategy}: ({r}, {c})");
            }
        }
    }

    #[test]
    fn random_same_seed_reproducible() {
        for strategy in ["drop", "pad", "shift"] {
            let a = random_anchors(100, 100, 32, 10, 7, strategy).unwrap();
            let b = random_anchors(100, 100, 32, 10, 7, strategy).unwrap();
            assert_eq!(a, b);
        }
        // Also in the enumerate-everything path.
        assert_eq!(
            random_anchors(40, 40, 32, 1000, 7, "drop").unwrap(),
            random_anchors(40, 40, 32, 1000, 7, "drop").unwrap()
        );
    }

    #[test]
    fn random_different_seeds_differ() {
        let a = random_anchors(100, 100, 32, 20, 1, "pad").unwrap();
        let b = random_anchors(100, 100, 32, 20, 2, "pad").unwrap();
        assert_ne!(a, b);
        // Seed 0 is a perfectly good seed.
        assert_ne!(random_anchors(100, 100, 32, 20, 0, "pad").unwrap(), a);
    }

    #[test]
    fn random_full_enumeration_is_shuffled() {
        // The full enumeration is a shuffle, not row-major order.
        let all = random_anchors(40, 40, 32, 1000, 5, "drop").unwrap();
        assert_eq!(all.len(), 81);
        assert_ne!(all, {
            let mut sorted = all.clone();
            sorted.sort_unstable();
            sorted
        });
    }

    #[test]
    fn random_covers_edges_uniformly() {
        // 100x100, 20px patches: 81 positions per axis, the last 20 columns
        // of anchors (61..=80) are 20/81 of them. Sample 100 anchors for many
        // seeds and compare the share touching that band with the expectation.
        let positions = 81.0_f64;
        let band = 20.0_f64;
        let mut in_last_row_band = 0u32;
        let mut in_last_col_band = 0u32;
        let mut first_row = 0u32;
        let mut last_row = 0u32;
        let mut total = 0u32;
        for seed in 0..400 {
            for (r, c) in random_anchors(100, 100, 20, 100, seed, "drop").unwrap() {
                total += 1;
                in_last_row_band += u32::from(r >= 61);
                in_last_col_band += u32::from(c >= 61);
                first_row += u32::from(r == 0);
                last_row += u32::from(r == 80);
            }
        }
        let total = f64::from(total);
        let expected = band / positions;
        // Binomial std of the share is ~ sqrt(p(1-p)/40000) ~ 0.002.
        let row_share = f64::from(in_last_row_band) / total;
        let col_share = f64::from(in_last_col_band) / total;
        assert!(
            (row_share - expected).abs() < 0.01,
            "row {row_share} vs {expected}"
        );
        assert!(
            (col_share - expected).abs() < 0.01,
            "col {col_share} vs {expected}"
        );
        // The very first and very last anchor row are equally likely.
        let first = f64::from(first_row) / total;
        let last = f64::from(last_row) / total;
        assert!((first - 1.0 / positions).abs() < 0.005, "first row {first}");
        assert!((last - 1.0 / positions).abs() < 0.005, "last row {last}");
    }

    #[test]
    fn zero_count_is_empty() {
        let anchors = random_anchors(100, 100, 32, 0, 42, "drop").unwrap();
        assert_eq!(anchors, Vec::new());
    }

    #[test]
    fn huge_raster_small_sample_is_cheap_and_distinct() {
        let anchors = random_anchors(1_000_000, 1_000_000, 256, 500, 3, "drop").unwrap();
        assert_eq!(anchors.len(), 500);
        assert_eq!(distinct(&anchors), 500);
    }

    #[test]
    fn zero_patch_size_errors() {
        assert!(grid_anchors(10, 10, 0, 4, "pad").is_err());
        assert!(random_anchors(10, 10, 0, 10, 42, "pad").is_err());
        assert!(random_anchor_capacity(10, 10, 0, "pad").is_err());
        assert!(random_anchors(0, 10, 4, 10, 42, "pad").is_err());
    }
}
