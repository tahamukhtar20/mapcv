//! Asynchronous tile fetcher using reqwest and tokio.

use crate::tile_math::TileIndex;
use futures::stream::{self, StreamExt};
use image::{DynamicImage, ImageFormat};
use pyo3::exceptions::{PyRuntimeError, PyValueError};
use pyo3::prelude::*;
use reqwest::{Client, Url};
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
    /// Black-fill PNG returned under the Ignore policy; counted as failed but never
    /// subject to `max_failed_ratio`.
    BlackFill(Vec<u8>),
    /// Tile was not found or failed; omitted from results under the Lenient policy.
    Missing,
}

/// Remove credentials, query parameters, and fragments before a URL is logged.
fn sanitize_url(url: &str) -> String {
    if let Ok(mut parsed) = Url::parse(url) {
        let _ = parsed.set_username("");
        let _ = parsed.set_password(None);
        parsed.set_query(None);
        parsed.set_fragment(None);
        return parsed.to_string();
    }

    url.split(['?', '#']).next().unwrap_or(url).to_owned()
}

fn network_error_message(error: &reqwest::Error, url: &str) -> String {
    let reason = if error.is_timeout() {
        "request timed out"
    } else if error.is_connect() {
        "connection failed"
    } else if error.is_request() {
        "request could not be sent"
    } else {
        "request failed"
    };
    format!("Network error for {}: {reason}", sanitize_url(url))
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

/// Longest server-requested `Retry-After` delay mapcv will honour.
const MAX_RETRY_AFTER: Duration = Duration::from_secs(30);

/// Apply *policy* to a failed tile.
fn on_failure(
    tile: TileIndex,
    policy: FailurePolicy,
    message: impl FnOnce() -> String,
) -> Result<(TileIndex, TileOutcome), String> {
    match policy {
        FailurePolicy::Strict => Err(message()),
        FailurePolicy::Lenient => Ok((tile, TileOutcome::Missing)),
        FailurePolicy::Ignore => Ok((tile, TileOutcome::BlackFill(black_tile_png()))),
    }
}

/// True when *bytes* start with a PNG, JPEG, WebP or GIF signature.
fn looks_like_image(bytes: &[u8]) -> bool {
    bytes.starts_with(b"\x89PNG\r\n\x1a\n")
        || bytes.starts_with(&[0xFF, 0xD8, 0xFF])
        || (bytes.len() >= 12 && &bytes[..4] == b"RIFF" && &bytes[8..12] == b"WEBP")
        || bytes.starts_with(b"GIF8")
}

/// Delay before the next attempt: the server's `Retry-After` (seconds, capped)
/// when given, otherwise linear backoff with up to 50% jitter.
fn retry_delay(retries: u32, retry_after: Option<&reqwest::header::HeaderValue>) -> Duration {
    if let Some(seconds) = retry_after
        .and_then(|value| value.to_str().ok())
        .and_then(|value| value.trim().parse::<u64>().ok())
    {
        return Duration::from_secs(seconds).min(MAX_RETRY_AFTER);
    }
    let base = RETRY_BACKOFF_MS * u64::from(retries);
    let jitter_seed = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map_or(0, |elapsed| u64::from(elapsed.subsec_nanos()));
    Duration::from_millis(base + jitter_seed % (base / 2 + 1))
}

/// Fetch a single tile with retries, applying *policy* on failure.
///
/// Network errors, truncated bodies, 429 and 5xx responses are retried. A
/// success response whose body is not an image (an HTML error page, an empty
/// 204) and other 4xx responses fail immediately.
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
        let (retryable_error, retry_after) = match client.get(&url).send().await {
            Ok(r) if r.status().is_success() => match r.bytes().await {
                Ok(bytes) if looks_like_image(&bytes) => {
                    return Ok((tile, TileOutcome::Success(bytes.to_vec())));
                }
                Ok(bytes) => {
                    return on_failure(tile, policy, || {
                        format!(
                            "Response for {} is not an image ({} bytes); the server may be \
                             returning an error page or rate-limiting",
                            sanitize_url(&url),
                            bytes.len()
                        )
                    });
                }
                Err(e) => (network_error_message(&e, &url), None),
            },
            Ok(r)
                if r.status().is_client_error()
                    && r.status() != reqwest::StatusCode::TOO_MANY_REQUESTS =>
            {
                let status = r.status();
                return on_failure(tile, policy, || {
                    format!("HTTP {status} for URL: {}", sanitize_url(&url))
                });
            }
            Ok(r) => {
                let retry_after = r.headers().get(reqwest::header::RETRY_AFTER).cloned();
                (
                    format!("HTTP {} for URL: {}", r.status(), sanitize_url(&url)),
                    retry_after,
                )
            }
            Err(e) => (network_error_message(&e, &url), None),
        };
        if retries >= MAX_RETRIES {
            return on_failure(tile, policy, || retryable_error);
        }
        retries += 1;
        tokio::time::sleep(retry_delay(retries, retry_after.as_ref())).await;
    }
}

/// Returns `(results, failed_count)` where `failed_count` includes both
/// omitted tiles (Lenient) and black-fill tiles (Ignore).
///
/// # Errors
/// Returns a `PyResult` error if the policy string is invalid or if the
/// background Tokio runtime fails.
///
/// Raises `ValueError` for an unknown policy or zero connections.
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
        .map_err(PyValueError::new_err)?;

    if max_connections == 0 {
        return Err(PyValueError::new_err("max_connections must be at least 1"));
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
                .connect_timeout(Duration::from_secs(10))
                .timeout(Duration::from_secs(30))
                .user_agent(concat!(
                    "mapcv/",
                    env!("CARGO_PKG_VERSION"),
                    " (+https://github.com/tahamukhtar20/mapcv)"
                ))
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
