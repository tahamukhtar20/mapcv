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

/// The most anchors one run enumerates (grid or random).
///
/// A run holds its anchors compactly: a grid as its row and column positions, random
/// anchors and those over an area of interest as one `u64` each (see
/// [`random_positions`]), so this keeps them near 400 MB. A larger request is refused
/// before anything is allocated: how many anchors there would be is computed from the
/// dimensions alone, so a crafted size cannot ask for memory the system only runs out
/// of while filling it.
pub const MAX_ANCHORS: u64 = 50_000_000;

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

/// Number of grid anchors along one dimension, without building them.
///
/// `dim`, `patch_size` and `stride` must be non-zero. This is the length of
/// [`dim_anchors`]; a `u128` because `shift` can add one to a count of `usize::MAX`.
fn dim_count(dim: usize, patch_size: usize, stride: usize, strategy: &str) -> u128 {
    // Anchors `p` with `p + patch_size <= dim`, i.e. patches fully inside the image.
    let inside = if dim >= patch_size {
        (dim - patch_size) / stride + 1
    } else {
        0
    } as u128;
    match strategy {
        "drop" => inside,
        "shift" => {
            if dim >= patch_size {
                // One more, shifted inward, unless the stride steps already end flush.
                let last = (dim - patch_size) as u128;
                inside + u128::from((inside - 1) * (stride as u128) < last)
            } else {
                // Image smaller than patch_size: a single anchor at 0.
                1
            }
        }
        // "pad": every stride step that starts inside the image (`p < dim`).
        _ => ((dim - 1) / stride + 1) as u128,
    }
}

/// Number of patches [`grid_anchors`] returns, computed from the dimensions alone (no
/// anchor is built): the per-axis counts multiplied.
///
/// Returns `(rows, cols, rows * cols)`; the product saturates at `u128::MAX`.
///
/// # Errors
/// Returns an error if `patch_size`, `stride`, `height` or `width` is zero, or if
/// `strategy` is unknown.
pub fn grid_anchor_count(
    height: usize,
    width: usize,
    patch_size: usize,
    stride: usize,
    strategy: &str,
) -> Result<(u128, u128, u128), String> {
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
    let rows = dim_count(height, patch_size, stride, strategy);
    let cols = dim_count(width, patch_size, stride, strategy);
    Ok((rows, cols, rows.saturating_mul(cols)))
}

/// Compute anchor positions along one dimension for grid sampling.
///
/// `dim`, `patch_size` and `stride` must be non-zero, and the caller has checked the
/// count against [`MAX_ANCHORS`]. Anchors are `i * stride` for a count computed up
/// front, so huge strides or dimensions can neither overflow nor loop for long.
fn dim_anchors(
    dim: usize,
    patch_size: usize,
    stride: usize,
    strategy: &str,
) -> Result<Vec<usize>, String> {
    let total = dim_count(dim, patch_size, stride, strategy);
    let count = match strategy {
        "drop" | "shift" => {
            if dim >= patch_size {
                (dim - patch_size) / stride + 1
            } else {
                0
            }
        }
        // "pad": every stride step that starts inside the image (`p < dim`).
        _ => (dim - 1) / stride + 1,
    };
    let mut v = with_room(
        usize::try_from(total).map_err(|_| allocation_error(usize::MAX, "patch anchors"))?,
        "patch anchors",
    )?;
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
    debug_assert_eq!(v.len() as u128, total);
    Ok(v)
}

/// Generate grid (or sliding-window) anchor positions.
///
/// Returns `(row, col)` top-left corners for `patch_size x patch_size` patches
/// sampled with the given `stride` across an `height x width` image.
///
/// # Errors
/// Returns an error if `patch_size`, `stride`, `height`, or `width` is zero,
/// if `strategy` is unknown, or if there would be more than [`MAX_ANCHORS`]
/// anchors (checked before anything is allocated).
pub fn grid_anchors(
    height: usize,
    width: usize,
    patch_size: usize,
    stride: usize,
    strategy: &str,
) -> Result<Vec<(usize, usize)>, String> {
    let (rows_n, cols_n, total) = grid_anchor_count(height, width, patch_size, stride, strategy)?;
    if total > u128::from(MAX_ANCHORS) {
        return Err(format!(
            "the grid has {rows_n} x {cols_n} = {total} patches, more than the limit of \
             {MAX_ANCHORS}; raise the stride, use a smaller area or split it into parts"
        ));
    }
    let rows = dim_anchors(height, patch_size, stride, strategy)?;
    let cols = dim_anchors(width, patch_size, stride, strategy)?;
    let mut anchors = with_room(rows.len() * cols.len(), "patch anchors")?;
    for &row in &rows {
        for &col in &cols {
            anchors.push((row, col));
        }
    }
    Ok(anchors)
}

/// The grid anchor positions along one axis: the rows (or columns) of [`grid_anchors`].
///
/// A grid is every pair of a row and a column position, so a run can hold the two axes
/// instead of every anchor.
///
/// # Errors
/// Returns an error if `dim`, `patch_size` or `stride` is zero, if `strategy` is
/// unknown, or if the axis alone has more than [`MAX_ANCHORS`] positions.
pub fn grid_axis(
    dim: usize,
    patch_size: usize,
    stride: usize,
    strategy: &str,
) -> Result<Vec<u64>, String> {
    let (count, _, _) = grid_anchor_count(dim, 1, patch_size, stride, strategy)?;
    if count > u128::from(MAX_ANCHORS) {
        return Err(format!(
            "the grid has {count} positions along one axis, more than the limit of \
             {MAX_ANCHORS}; raise the stride, use a smaller area or split it into parts"
        ));
    }
    let positions = dim_anchors(dim, patch_size, stride, strategy)?;
    let mut out = with_room(positions.len(), "patch anchors")?;
    out.extend(positions.into_iter().map(|p| p as u64));
    Ok(out)
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

/// Which values Floyd's algorithm has picked: an open-addressing table of indices into
/// the picked values, so each value is stored once (4 bytes a slot instead of the 9 to
/// 18 bytes a value a `HashSet<u64>` takes, on top of the list).
struct PickedSet {
    slots: Vec<u32>,
}

impl PickedSet {
    /// An empty slot (an index is below [`MAX_ANCHORS`], far below `u32::MAX`).
    const EMPTY: u32 = u32::MAX;

    /// Room for `n` values at a load of at most two thirds.
    fn with_room(n: usize) -> Result<Self, String> {
        let len = n + n / 2 + 1;
        let mut slots = with_room(len, "random patch anchors")?;
        slots.resize(len, Self::EMPTY);
        Ok(Self { slots })
    }

    /// The slot a value's probe starts at.
    fn start(&self, value: u64) -> usize {
        let mut mixed = value;
        let hash = splitmix64(&mut mixed);
        ((u128::from(hash) * self.slots.len() as u128) >> 64) as usize
    }

    /// Whether `value` is among `picked`, the values inserted so far in order.
    fn contains(&self, picked: &[u64], value: u64) -> bool {
        let mut slot = self.start(value);
        loop {
            let index = self.slots[slot];
            if index == Self::EMPTY {
                return false;
            }
            if picked[index as usize] == value {
                return true;
            }
            slot = (slot + 1) % self.slots.len();
        }
    }

    /// Record that `picked[index]` holds `value`, which is not in the set yet.
    fn insert(&mut self, index: usize, value: u64) {
        let mut slot = self.start(value);
        while self.slots[slot] != Self::EMPTY {
            slot = (slot + 1) % self.slots.len();
        }
        self.slots[slot] = index as u32;
    }
}

/// Up to `count` distinct positions `row * cols + col` on the grid random anchors are
/// drawn from (`cols` wide), in a seeded shuffle; returns them with `cols`.
fn random_flat(
    height: usize,
    width: usize,
    patch_size: usize,
    count: usize,
    seed: u64,
    strategy: &str,
) -> Result<(Vec<u64>, u64), String> {
    check_strategy(strategy)?;
    let capacity = random_anchor_capacity(height, width, patch_size, strategy)? as u64;
    let cols = random_axis_len(width, patch_size, strategy) as u64;
    let wanted = (count as u64).min(capacity);
    if wanted == 0 {
        return Ok((Vec::new(), cols));
    }
    if wanted > MAX_ANCHORS {
        return Err(format!(
            "{wanted} random patches were asked for (the raster has room for {capacity}), \
             more than the limit of {MAX_ANCHORS}; ask for fewer"
        ));
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
        let mut taken = PickedSet::with_room(wanted_len)?;
        let mut picked = with_room(wanted_len, "random patch anchors")?;
        for j in (capacity - wanted)..capacity {
            let t = uniform(&mut state, j + 1);
            // `j` is above every value picked before it, so it is never taken yet.
            let value = if taken.contains(&picked, t) { j } else { t };
            taken.insert(picked.len(), value);
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
    Ok((flat, cols))
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
/// `strategy` is unknown, or if more than [`MAX_ANCHORS`] anchors are asked for
/// (and available).
pub fn random_anchors(
    height: usize,
    width: usize,
    patch_size: usize,
    count: usize,
    seed: u64,
    strategy: &str,
) -> Result<Vec<(usize, usize)>, String> {
    let (flat, cols) = random_flat(height, width, patch_size, count, seed, strategy)?;
    let mut anchors = with_room(flat.len(), "random patch anchors")?;
    anchors.extend(
        flat.into_iter()
            .map(|v| ((v / cols) as usize, (v % cols) as usize)),
    );
    Ok(anchors)
}

/// The anchors of [`random_anchors`] as `row * width + col`, one `u64` each, grouped by
/// the chunk of `chunk_rows` rows they fall in.
///
/// Chunks come top to bottom; within a chunk the anchors keep the order of
/// [`random_anchors`]. This is how a run holds random anchors: a quarter of the memory
/// of two `usize` each, and none of the Python objects a list of pairs would need.
///
/// # Errors
/// The errors of [`random_anchors`], and an error if `chunk_rows` is zero or
/// `height * width` does not fit in 64 bits.
pub fn random_positions(
    height: usize,
    width: usize,
    patch_size: usize,
    count: usize,
    seed: u64,
    strategy: &str,
    chunk_rows: usize,
) -> Result<Vec<u64>, String> {
    if chunk_rows == 0 {
        return Err("chunk_rows must be > 0".to_string());
    }
    if (height as u64).checked_mul(width as u64).is_none() {
        return Err("raster is too large to enumerate anchors".to_string());
    }
    let (mut flat, cols) = random_flat(height, width, patch_size, count, seed, strategy)?;
    let width = width as u64;
    // Below `height * width`, which fits: every anchor is inside the raster.
    for value in &mut flat {
        *value = *value / cols * width + *value % cols;
    }
    let chunk_rows = chunk_rows as u64;
    // Stable, so each chunk keeps the shuffled order.
    flat.sort_by_key(|value| value / width / chunk_rows);
    Ok(flat)
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
    fn grid_count_matches_the_enumeration() {
        for strategy in ["pad", "drop", "shift"] {
            for (height, width) in [(1, 1), (5, 40), (37, 11), (64, 64), (100, 73)] {
                for patch_size in [1, 3, 8, 32, 64, 120] {
                    for stride in [1, 2, 7, 16, 33, 64, 500] {
                        let (rows, cols, total) =
                            grid_anchor_count(height, width, patch_size, stride, strategy).unwrap();
                        let anchors =
                            grid_anchors(height, width, patch_size, stride, strategy).unwrap();
                        assert_eq!(
                            total,
                            anchors.len() as u128,
                            "{strategy} {height}x{width} ps={patch_size} stride={stride}"
                        );
                        assert_eq!(rows * cols, total);
                    }
                }
            }
        }
    }

    #[test]
    fn grid_count_handles_sizes_beyond_memory() {
        // 2^30 x 2^30 positions: counted without building anything.
        let (_, _, total) = grid_anchor_count(1 << 30, 1 << 30, 64, 1, "pad").unwrap();
        assert_eq!(total, 1u128 << 60);
        let (_, _, total) = grid_anchor_count(usize::MAX, usize::MAX, 1, 1, "shift").unwrap();
        assert!(total > u128::from(u64::MAX));
        assert!(grid_anchor_count(10, 10, 4, 0, "pad").is_err());
        assert!(grid_anchor_count(10, 10, 0, 4, "pad").is_err());
        assert!(grid_anchor_count(0, 10, 4, 4, "pad").is_err());
        assert!(grid_anchor_count(10, 10, 4, 4, "wrap").is_err());
    }

    #[test]
    fn grid_over_the_limit_is_refused_before_allocating() {
        // 7,072 x 7,072 = 50,013,184 anchors: over the limit, refused before any is built.
        let err = grid_anchors(7_072, 7_072, 64, 1, "pad").unwrap_err();
        assert!(err.contains("more than the limit of 50000000"), "{err}");
        assert!(err.contains("7072 x 7072 = 50013184"), "{err}");
        // Exactly at the limit is fine: 5,000 x 10,000 = 50,000,000 (counted, not built).
        let (_, _, total) = grid_anchor_count(5_000, 10_000, 1, 1, "pad").unwrap();
        assert_eq!(total, u128::from(MAX_ANCHORS));
        assert!(grid_anchors(5_000, 10_001, 1, 1, "pad").is_err());
    }

    #[test]
    fn grid_axes_are_the_rows_and_columns_of_the_grid() {
        for strategy in ["pad", "drop", "shift"] {
            for (height, width, patch, stride) in [
                (10, 10, 4, 4),
                (37, 11, 8, 3),
                (64, 64, 64, 7),
                (5, 130, 32, 33),
            ] {
                let rows = grid_axis(height, patch, stride, strategy).unwrap();
                let cols = grid_axis(width, patch, stride, strategy).unwrap();
                let product: Vec<(usize, usize)> = rows
                    .iter()
                    .flat_map(|&r| cols.iter().map(move |&c| (r as usize, c as usize)))
                    .collect();
                assert_eq!(
                    product,
                    grid_anchors(height, width, patch, stride, strategy).unwrap()
                );
            }
        }
        let err = grid_axis(MAX_ANCHORS as usize + 1, 1, 1, "pad").unwrap_err();
        assert!(err.contains("more than the limit"), "{err}");
        assert!(grid_axis(10, 4, 0, "pad").is_err());
        assert!(grid_axis(10, 4, 4, "wrap").is_err());
    }

    /// Floyd's algorithm with a `HashSet`, as before the compact set: the reference.
    fn floyd_with_hash_set(capacity: u64, wanted: u64, seed: u64) -> Vec<u64> {
        let mut state = seed;
        let mut taken = HashSet::new();
        let mut picked = Vec::new();
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
        for i in (1..picked.len()).rev() {
            let j = uniform(&mut state, i as u64 + 1) as usize;
            picked.swap(i, j);
        }
        picked
    }

    #[test]
    fn the_compact_set_picks_what_a_hash_set_picks() {
        // Dense draws (many collisions) and sparse ones, on a 100 x 100 position grid.
        for (count, seed) in [(1, 0), (50, 3), (5_000, 7), (9_999, 11), (2_000, 42)] {
            let (flat, cols) = random_flat(131, 131, 32, count, seed, "drop").unwrap();
            assert_eq!(cols, 100);
            assert_eq!(flat, floyd_with_hash_set(10_000, count as u64, seed));
        }
    }

    #[test]
    fn random_positions_are_the_anchors_grouped_by_chunk() {
        for (height, width, patch, count, chunk_rows) in [
            (300, 200, 16, 500, 64),
            (300, 200, 16, 100_000, 7),
            (20, 20, 32, 5, 4),
            (1_000, 3, 1, 2_000, 1_000),
        ] {
            let anchors = random_anchors(height, width, patch, count, 9, "pad").unwrap();
            let mut expected: Vec<(usize, usize)> = anchors.clone();
            expected.sort_by_key(|&(row, _)| row / chunk_rows);
            let positions =
                random_positions(height, width, patch, count, 9, "pad", chunk_rows).unwrap();
            let got: Vec<(usize, usize)> = positions
                .iter()
                .map(|&p| ((p / width as u64) as usize, (p % width as u64) as usize))
                .collect();
            assert_eq!(got, expected);
        }
        assert!(random_positions(10, 10, 4, 3, 1, "pad", 0).is_err());
        assert!(random_positions(usize::MAX, usize::MAX, 1, 1, 1, "pad", 1).is_err());
    }

    #[test]
    fn random_over_the_limit_is_refused() {
        let err = random_anchors(100_000, 100_000, 64, 50_000_001, 1, "drop").unwrap_err();
        assert!(err.contains("more than the limit of 50000000"), "{err}");
        // A count above the raster's capacity is capped first, so this stays valid.
        assert_eq!(
            random_anchors(10, 10, 4, 50_000_001, 1, "drop")
                .unwrap()
                .len(),
            49
        );
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
