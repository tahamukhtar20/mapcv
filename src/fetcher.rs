//! Asynchronous tile fetcher using reqwest and tokio.

use crate::tile_math::TileIndex;
use futures::stream::{self, StreamExt};
use image::{DynamicImage, ImageFormat};
use pyo3::exceptions::PyRuntimeError;
use pyo3::prelude::*;
use reqwest::Client;
use std::io::Cursor;
use std::sync::mpsc;
use std::thread;
use std::time::Duration;

/// How tile fetch failures are handled.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum FailurePolicy {
    /// Abort on the first failure.
    Strict,
    /// Omit failed tiles from results.
    Lenient,
    /// Return a black 256x256 PNG for failed tiles instead of omitting them.
    Ignore,
}

impl std::str::FromStr for FailurePolicy {
    type Err = String;

    /// # Errors
    /// Returns an error if the string is not a valid policy name.
    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s.to_lowercase().as_str() {
            "strict" => Ok(FailurePolicy::Strict),
            "lenient" => Ok(FailurePolicy::Lenient),
            "ignore" => Ok(FailurePolicy::Ignore),
            _ => Err(format!("Unknown policy: {s}")),
        }
    }
}

// 1 initial attempt + MAX_RETRIES retries = MAX_RETRIES + 1 total attempts.
const MAX_RETRIES: u32 = 3;
const RETRY_BACKOFF_MS: u64 = 500;

/// Standard tile pixel dimension used by XYZ tile servers.
pub(crate) const TILE_PX: usize = 256;
/// `TILE_PX` as `f64`, derived from `TILE_PX` to stay in sync.
// 256 is exactly representable in f64 (2^8), so no precision is lost.
#[allow(clippy::cast_precision_loss)]
pub(crate) const TILE_PX_F: f64 = TILE_PX as f64;

/// Outcome of a single tile fetch attempt.
enum TileOutcome {
    /// Tile fetched successfully; contains the raw PNG bytes.
    Success(Vec<u8>),
    /// Black-fill PNG returned under the Ignore policy; still counts toward the failed ratio.
    BlackFill(Vec<u8>),
    /// Tile was not found or failed; omitted from results under the Lenient policy.
    Missing,
}

/// Remove query parameters from a URL string to prevent leaking sensitive API keys in error messages.
fn sanitize_url(url: &str) -> String {
    url.split('?').next().unwrap_or(url).to_owned()
}

/// Return a solid-black `TILE_PX x TILE_PX` PNG buffer used as a `NoData` placeholder.
fn black_tile_png() -> Vec<u8> {
    // TILE_PX is 256, well within u32 range.
    #[allow(clippy::cast_possible_truncation)]
    let px = TILE_PX as u32;
    let img = DynamicImage::new_rgb8(px, px);
    let mut buf = Cursor::new(Vec::new());
    // In-memory cursor; I/O cannot fail here.
    img.write_to(&mut buf, ImageFormat::Png)
        .expect("black tile PNG encoding failed");
    buf.into_inner()
}

/// Channel messages sent from the async fetch worker back to the Python thread.
enum Event {
    /// Number of tiles completed so far (for progress callbacks).
    Progress(usize),
    /// A fatal error aborted the fetch; contains the message.
    Error(String),
    /// All tiles processed; contains results and the count of failed tiles.
    Done(Vec<(TileIndex, Vec<u8>)>, usize),
}

/// Fetch a single tile with retries, applying *policy* on failure.
async fn fetch_single_tile(
    client: Client,
    tile: TileIndex,
    url_template: String,
    policy: FailurePolicy,
) -> Result<(TileIndex, TileOutcome), String> {
    let url = url_template
        .replace("{z}", &tile.z.to_string())
        .replace("{x}", &tile.x.to_string())
        .replace("{y}", &tile.y.to_string());

    let mut retries: u32 = 0;
    loop {
        let resp = client.get(&url).send().await;
        match resp {
            Ok(r) if r.status().is_success() => {
                let bytes = r.bytes().await.map_err(|e| e.to_string())?;
                return Ok((tile, TileOutcome::Success(bytes.to_vec())));
            }
            Ok(r) if r.status() == reqwest::StatusCode::NOT_FOUND => match policy {
                FailurePolicy::Strict => {
                    return Err(format!("Tile 404 Not Found: {}", sanitize_url(&url)))
                }
                FailurePolicy::Lenient => return Ok((tile, TileOutcome::Missing)),
                FailurePolicy::Ignore => {
                    return Ok((tile, TileOutcome::BlackFill(black_tile_png())))
                }
            },
            // Non-retryable client errors (4xx except 404 above and 429 which may clear).
            Ok(r)
                if r.status().is_client_error()
                    && r.status() != reqwest::StatusCode::TOO_MANY_REQUESTS =>
            {
                match policy {
                    FailurePolicy::Strict => {
                        return Err(format!(
                            "HTTP {} for URL: {}",
                            r.status(),
                            sanitize_url(&url)
                        ))
                    }
                    FailurePolicy::Lenient => return Ok((tile, TileOutcome::Missing)),
                    FailurePolicy::Ignore => {
                        return Ok((tile, TileOutcome::BlackFill(black_tile_png())))
                    }
                }
            }
            Ok(r) => {
                if retries >= MAX_RETRIES {
                    match policy {
                        FailurePolicy::Strict => {
                            return Err(format!(
                                "HTTP {} for URL: {}",
                                r.status(),
                                sanitize_url(&url)
                            ))
                        }
                        FailurePolicy::Lenient => return Ok((tile, TileOutcome::Missing)),
                        FailurePolicy::Ignore => {
                            return Ok((tile, TileOutcome::BlackFill(black_tile_png())))
                        }
                    }
                }
            }
            Err(e) => {
                if retries >= MAX_RETRIES {
                    match policy {
                        FailurePolicy::Strict => {
                            // reqwest's error message may contain the URL. We attempt to redact it.
                            let err_str = e.to_string();
                            let sanitized_msg = if err_str.contains(&url) {
                                err_str.replace(&url, &sanitize_url(&url))
                            } else {
                                err_str
                            };
                            return Err(format!("Network error: {sanitized_msg}"));
                        }
                        FailurePolicy::Lenient => return Ok((tile, TileOutcome::Missing)),
                        FailurePolicy::Ignore => {
                            return Ok((tile, TileOutcome::BlackFill(black_tile_png())))
                        }
                    }
                }
            }
        }
        retries += 1;
        tokio::time::sleep(Duration::from_millis(RETRY_BACKOFF_MS * u64::from(retries))).await;
    }
}

/// Returns `(results, failed_count)` where `failed_count` includes both
/// omitted tiles (Lenient) and black-fill tiles (Ignore).
///
/// # Errors
/// Returns a `PyResult` error if the policy string is invalid or if the
/// background Tokio runtime fails.
///
/// # Panics
/// Panics if the internal cross-thread channel mutex is poisoned.
#[allow(clippy::needless_pass_by_value, clippy::type_complexity)]
pub fn fetch_tiles(
    py: Python,
    tiles: Vec<TileIndex>,
    url_template: String,
    callback: Option<PyObject>,
    max_connections: usize,
    policy_str: &str,
) -> PyResult<(Vec<(TileIndex, Vec<u8>)>, usize)> {
    let policy = policy_str
        .parse::<FailurePolicy>()
        .map_err(PyRuntimeError::new_err)?;

    if max_connections == 0 {
        return Err(PyRuntimeError::new_err(
            "max_connections must be at least 1",
        ));
    }

    let (tx, rx) = mpsc::channel();

    let _ = thread::spawn(move || {
        let rt = match tokio::runtime::Runtime::new() {
            Ok(rt) => rt,
            Err(e) => {
                let _ = tx.send(Event::Error(format!("Failed to create tokio runtime: {e}")));
                return;
            }
        };

        rt.block_on(async {
            let client = match Client::builder()
                .timeout(Duration::from_secs(10))
                .user_agent(concat!("mapcv-fetcher/", env!("CARGO_PKG_VERSION")))
                .build()
            {
                Ok(c) => c,
                Err(e) => {
                    let _ = tx.send(Event::Error(format!("Failed to build HTTP client: {e}")));
                    return;
                }
            };

            let mut stream = stream::iter(tiles)
                .map(|tile| {
                    let client_clone = client.clone();
                    let url_clone = url_template.clone();
                    async move { fetch_single_tile(client_clone, tile, url_clone, policy).await }
                })
                .buffer_unordered(max_connections);

            let mut results = Vec::new();
            let mut completed: usize = 0;
            let mut failed: usize = 0;

            while let Some(res) = stream.next().await {
                match res {
                    Ok((tile, TileOutcome::Success(bytes))) => {
                        results.push((tile, bytes));
                    }
                    Ok((tile, TileOutcome::BlackFill(bytes))) => {
                        results.push((tile, bytes));
                        failed += 1;
                    }
                    Ok((_tile, TileOutcome::Missing)) => {
                        failed += 1;
                    }
                    Err(e) => {
                        let _ = tx.send(Event::Error(e));
                        return;
                    }
                }
                completed += 1;
                if tx.send(Event::Progress(completed)).is_err() {
                    return;
                }
            }

            let _ = tx.send(Event::Done(results, failed));
        });
    });

    let rx = std::sync::Mutex::new(rx);

    loop {
        let event = py.allow_threads(|| -> PyResult<Event> {
            let guard = rx
                .lock()
                .map_err(|_| PyRuntimeError::new_err("internal fetch mutex was poisoned"))?;
            guard
                .recv()
                .map_err(|_| PyRuntimeError::new_err("Background thread died unexpectedly"))
        })?;

        match event {
            Event::Progress(c) => {
                if let Some(ref cb) = callback {
                    cb.call1(py, (c,))?;
                }
            }
            Event::Error(err) => {
                return Err(PyRuntimeError::new_err(err));
            }
            Event::Done(results, failed) => {
                return Ok((results, failed));
            }
        }
    }
}
