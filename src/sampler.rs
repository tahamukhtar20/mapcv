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

/// Reject edge strategies other than `"pad"`, `"drop"` and `"shift"`.
fn check_strategy(strategy: &str) -> Result<(), String> {
    match strategy {
        "pad" | "drop" | "shift" => Ok(()),
        other => Err(format!(
            "edge_strategy must be 'pad', 'drop' or 'shift', got '{other}'"
        )),
    }
}

/// An empty vector with room for `n` items, or an error when that much memory
/// cannot be allocated (instead of aborting the process).
fn with_room<T>(n: usize, what: &str) -> Result<Vec<T>, String> {
    let mut v = Vec::new();
    v.try_reserve_exact(n)
        .map_err(|_| allocation_error(n, what))?;
    Ok(v)
}

fn allocation_error(n: usize, what: &str) -> String {
    format!("cannot allocate {n} {what}; the request is too large")
}

/// Compute anchor positions along one dimension for grid sampling.
///
/// `dim`, `patch_size` and `stride` must be non-zero. Anchors are `i * stride`
/// for a count computed up front, so huge strides or dimensions can neither
/// overflow nor loop for long.
fn dim_anchors(
    dim: usize,
    patch_size: usize,
    stride: usize,
    strategy: &str,
) -> Result<Vec<usize>, String> {
    // Anchors `p` with `p + patch_size <= dim`, i.e. patches fully inside the image.
    let inside = if dim >= patch_size {
        (dim - patch_size) / stride + 1
    } else {
        0
    };
    let count = match strategy {
        "drop" | "shift" => inside,
        // "pad": every stride step that starts inside the image (`p < dim`).
        _ => (dim - 1) / stride + 1,
    };
    let mut v = with_room(count.saturating_add(1), "patch anchors")?;
    // Each anchor is below `dim`, so `i * stride` cannot overflow.
    v.extend((0..count).map(|i| i * stride));
    if strategy == "shift" {
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
    }
    Ok(v)
}

/// Generate grid (or sliding-window) anchor positions.
///
/// Returns `(row, col)` top-left corners for `patch_size x patch_size` patches
/// sampled with the given `stride` across an `height x width` image.
///
/// # Errors
/// Returns an error if `patch_size`, `stride`, `height`, or `width` is zero,
/// if `strategy` is unknown, or if the anchors would not fit in memory.
pub fn grid_anchors(
    height: usize,
    width: usize,
    patch_size: usize,
    stride: usize,
    strategy: &str,
) -> Result<Vec<(usize, usize)>, String> {
    check_strategy(strategy)?;
    if patch_size == 0 {
        return Err("patch_size must be > 0".to_string());
    }
    if stride == 0 {
        return Err("stride must be > 0".to_string());
    }
    if height == 0 || width == 0 {
        return Err("height and width must be > 0".to_string());
    }
    let rows = dim_anchors(height, patch_size, stride, strategy)?;
    let cols = dim_anchors(width, patch_size, stride, strategy)?;
    let total = rows.len().checked_mul(cols.len()).ok_or_else(|| {
        format!(
            "cannot allocate {} x {} patch anchors; the request is too large",
            rows.len(),
            cols.len()
        )
    })?;
    let mut anchors = with_room(total, "patch anchors")?;
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
/// Returns an error if `patch_size`, `height`, or `width` is zero, if
/// `strategy` is unknown, or if `count` anchors would not fit in memory.
pub fn random_anchors(
    height: usize,
    width: usize,
    patch_size: usize,
    count: usize,
    seed: u64,
    strategy: &str,
) -> Result<Vec<(usize, usize)>, String> {
    check_strategy(strategy)?;
    let capacity = random_anchor_capacity(height, width, patch_size, strategy)? as u64;
    let cols = random_axis_len(width, patch_size, strategy) as u64;
    let wanted = (count as u64).min(capacity);
    if wanted == 0 {
        return Ok(Vec::new());
    }
    let mut state = seed;
    // Sized with `try_reserve` so a request too large to hold is an error, not
    // an allocation abort.
    let wanted_len = wanted as usize;
    let mut flat: Vec<u64> = if wanted == capacity {
        let mut all = with_room(wanted_len, "random patch anchors")?;
        all.extend(0..capacity);
        all
    } else {
        // Floyd's algorithm: `wanted` distinct values from [0, capacity) in
        // O(wanted) time and memory. The Vec keeps insertion order so the
        // output never depends on hash iteration order.
        let mut taken = std::collections::HashSet::new();
        taken
            .try_reserve(wanted_len)
            .map_err(|_| allocation_error(wanted_len, "random patch anchors"))?;
        let mut picked = with_room(wanted_len, "random patch anchors")?;
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
    let mut anchors = with_room(flat.len(), "random patch anchors")?;
    anchors.extend(
        flat.into_iter()
            .map(|v| ((v / cols) as usize, (v % cols) as usize)),
    );
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

    /// The step-by-step loop `dim_anchors` used before counting up front.
    fn loop_anchors(dim: usize, patch_size: usize, stride: usize, strategy: &str) -> Vec<usize> {
        let mut v = Vec::new();
        let mut p = 0usize;
        if strategy == "pad" {
            while p < dim {
                v.push(p);
                p += stride;
            }
            return v;
        }
        while p + patch_size <= dim {
            v.push(p);
            p += stride;
        }
        if strategy == "shift" {
            if dim >= patch_size {
                let last = dim - patch_size;
                if v.last().copied().is_none_or(|q| q < last) {
                    v.push(last);
                }
            } else if v.is_empty() {
                v.push(0);
            }
        }
        v
    }

    #[test]
    fn counted_anchors_match_the_stepping_loop() {
        for strategy in ["pad", "drop", "shift"] {
            for dim in 1..40 {
                for patch_size in 1..20 {
                    for stride in 1..20 {
                        assert_eq!(
                            dim_anchors(dim, patch_size, stride, strategy).unwrap(),
                            loop_anchors(dim, patch_size, stride, strategy),
                            "{strategy} dim={dim} ps={patch_size} stride={stride}"
                        );
                    }
                }
            }
        }
    }

    #[test]
    fn unknown_strategy_errors() {
        assert!(grid_anchors(10, 10, 4, 4, "wrap").is_err());
        assert!(random_anchors(10, 10, 4, 1, 42, "Pad").is_err());
    }

    #[test]
    fn huge_values_error_or_stay_small_instead_of_overflowing() {
        // Strides near usize::MAX used to overflow `p += stride`.
        let anchors = grid_anchors(10, 10, 5, usize::MAX, "drop").unwrap();
        assert_eq!(anchors, vec![(0, 0)]);
        let anchors =
            grid_anchors(usize::MAX, usize::MAX, usize::MAX, usize::MAX, "shift").unwrap();
        assert_eq!(anchors, vec![(0, 0)]);
        // Grids and counts that cannot be allocated are errors, not aborts.
        assert!(grid_anchors(usize::MAX, usize::MAX, 1, 1, "pad").is_err());
        // A count above the raster's capacity is capped, not an allocation request.
        assert_eq!(
            random_anchors(10, 10, 4, usize::MAX, 42, "pad")
                .unwrap()
                .len(),
            49
        );
        // Only a raster with too many positions to hold is an error.
        assert!(random_anchors(1 << 30, 1 << 30, 1, usize::MAX, 42, "pad").is_err());
        assert!(random_anchors(usize::MAX, usize::MAX, 1, 1, 42, "pad").is_err());
    }
}
