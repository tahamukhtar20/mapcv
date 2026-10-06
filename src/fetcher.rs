//! Asynchronous tile fetcher using reqwest and tokio.

use crate::tile_math::{TileIndex, TILE_PX};
use futures::stream::{self, StreamExt};
use image::{DynamicImage, ImageFormat};
use pyo3::exceptions::{PyRuntimeError, PyValueError};
use pyo3::prelude::*;
use reqwest::{Client, Url};
use std::collections::BTreeMap;
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

/// Outcome of a single tile fetch attempt.
enum TileOutcome {
    /// Tile fetched successfully; contains the raw PNG bytes.
    Success(Vec<u8>),
    /// Black-fill PNG returned under the Ignore policy; counted as failed but never
    /// subject to `max_failed_ratio`. Carries the failure.
    BlackFill(Vec<u8>, Failure),
    /// Tile was not found or failed; omitted from results under the Lenient policy.
    Missing(Failure),
}

/// Why a tile failed: a short kind for grouping and the full message for one example.
struct Failure {
    kind: String,
    message: String,
}

/// Failed tiles grouped by kind, with one example message, for error reports.
#[derive(Default)]
pub struct FailureSummary {
    counts: BTreeMap<String, usize>,
    example: Option<String>,
}

impl FailureSummary {
    fn add(&mut self, failure: Failure) {
        *self.counts.entry(failure.kind).or_insert(0) += 1;
        self.example.get_or_insert(failure.message);
    }

    /// Failed-tile counts per cause.
    #[must_use]
    pub fn counts(&self) -> Vec<(String, usize)> {
        self.counts.iter().map(|(k, v)| (k.clone(), *v)).collect()
    }

    /// The first failure's full message, if any tile failed.
    #[must_use]
    pub fn example(&self) -> Option<String> {
        self.example.clone()
    }

    /// One line such as `8 x HTTP 503 Service Unavailable, 2 x request timed out
    /// (e.g. HTTP 503 ... for URL: ...)`, most common first; empty without failures.
    #[must_use]
    pub fn describe(&self) -> String {
        let mut counts: Vec<(&String, &usize)> = self.counts.iter().collect();
        counts.sort_by(|a, b| b.1.cmp(a.1).then_with(|| a.0.cmp(b.0)));
        let kinds: Vec<String> = counts
            .iter()
            .map(|(kind, count)| format!("{count} x {kind}"))
            .collect();
        match &self.example {
            Some(example) => format!("{} (e.g. {example})", kinds.join(", ")),
            None => String::new(),
        }
    }
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

fn network_error_kind(error: &reqwest::Error) -> &'static str {
    if error.is_timeout() {
        "request timed out"
    } else if error.is_connect() {
        "connection failed"
    } else if error.is_request() {
        "request could not be sent"
    } else {
        "request failed"
    }
}

fn network_error_message(error: &reqwest::Error, url: &str) -> String {
    format!(
        "Network error for {}: {}",
        sanitize_url(url),
        network_error_kind(error)
    )
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
    /// All tiles processed; contains results, the count of failed tiles and why they failed.
    Done(Vec<(TileIndex, Vec<u8>)>, usize, FailureSummary),
}

/// Longest server-requested `Retry-After` delay mapcv will honour.
const MAX_RETRY_AFTER: Duration = Duration::from_secs(30);

/// Apply *policy* to a failed tile.
fn on_failure(
    tile: TileIndex,
    policy: FailurePolicy,
    kind: String,
    message: String,
) -> Result<(TileIndex, TileOutcome), String> {
    let failure = Failure { kind, message };
    match policy {
        FailurePolicy::Strict => Err(failure.message),
        FailurePolicy::Lenient => Ok((tile, TileOutcome::Missing(failure))),
        FailurePolicy::Ignore => Ok((tile, TileOutcome::BlackFill(black_tile_png(), failure))),
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
        let (kind, retryable_error, retry_after) = match client.get(&url).send().await {
            Ok(r) if r.status().is_success() => match r.bytes().await {
                Ok(bytes) if looks_like_image(&bytes) => {
                    return Ok((tile, TileOutcome::Success(bytes.to_vec())));
                }
                Ok(bytes) => {
                    let message = format!(
                        "Response for {} is not an image ({} bytes); the server may be \
                         returning an error page or rate-limiting",
                        sanitize_url(&url),
                        bytes.len()
                    );
                    return on_failure(tile, policy, "response is not an image".into(), message);
                }
                Err(e) => (
                    network_error_kind(&e).to_owned(),
                    network_error_message(&e, &url),
                    None,
                ),
            },
            Ok(r)
                if r.status().is_client_error()
                    && r.status() != reqwest::StatusCode::TOO_MANY_REQUESTS =>
            {
                let status = r.status();
                let message = format!("HTTP {status} for URL: {}", sanitize_url(&url));
                return on_failure(tile, policy, format!("HTTP {status}"), message);
            }
            Ok(r) => {
                let retry_after = r.headers().get(reqwest::header::RETRY_AFTER).cloned();
                (
                    format!("HTTP {}", r.status()),
                    format!("HTTP {} for URL: {}", r.status(), sanitize_url(&url)),
                    retry_after,
                )
            }
            Err(e) => (
                network_error_kind(&e).to_owned(),
                network_error_message(&e, &url),
                None,
            ),
        };
        if retries >= MAX_RETRIES {
            return on_failure(tile, policy, kind, retryable_error);
        }
        retries += 1;
        tokio::time::sleep(retry_delay(retries, retry_after.as_ref())).await;
    }
}

/// Returns `(results, failed_count, failures)` where `failed_count` includes both
/// omitted tiles (Lenient) and black-fill tiles (Ignore), and `failures` groups them
/// by reason.
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
    callback: Option<Py<PyAny>>,
    max_connections: usize,
    policy_str: &str,
) -> PyResult<(Vec<(TileIndex, Vec<u8>)>, usize, FailureSummary)> {
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
            let mut failures = FailureSummary::default();

            while let Some(res) = stream.next().await {
                match res {
                    Ok((tile, TileOutcome::Success(bytes))) => {
                        results.push((tile, bytes));
                    }
                    Ok((tile, TileOutcome::BlackFill(bytes, failure))) => {
                        results.push((tile, bytes));
                        failed += 1;
                        failures.add(failure);
                    }
                    Ok((_tile, TileOutcome::Missing(failure))) => {
                        failed += 1;
                        failures.add(failure);
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

            let _ = tx.send(Event::Done(results, failed, failures));
        });
    });

    let rx = std::sync::Mutex::new(rx);

    loop {
        let event = py.detach(|| -> PyResult<Event> {
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
            Event::Done(results, failed, failures) => {
                return Ok((results, failed, failures));
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::{Failure, FailureSummary};

    fn failure(kind: &str, message: &str) -> Failure {
        Failure {
            kind: kind.to_owned(),
            message: message.to_owned(),
        }
    }

    #[test]
    fn summary_lists_most_common_reasons_first_with_one_example() {
        let mut summary = FailureSummary::default();
        assert_eq!(summary.describe(), "");
        summary.add(failure(
            "request timed out",
            "Network error for A: request timed out",
        ));
        summary.add(failure(
            "HTTP 503 Service Unavailable",
            "HTTP 503 for URL: B",
        ));
        summary.add(failure(
            "HTTP 503 Service Unavailable",
            "HTTP 503 for URL: C",
        ));
        assert_eq!(
            summary.describe(),
            "2 x HTTP 503 Service Unavailable, 1 x request timed out \
             (e.g. Network error for A: request timed out)"
        );
    }
}
