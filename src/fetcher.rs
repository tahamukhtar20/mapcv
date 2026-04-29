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

enum TileOutcome {
    Success(Vec<u8>),
    /// Black-fill PNG returned by Ignore policy. Still counts toward failed ratio.
    BlackFill(Vec<u8>),
    Missing,
}

fn black_tile_png() -> Vec<u8> {
    let img = DynamicImage::new_rgb8(256, 256);
    let mut buf = Cursor::new(Vec::new());
    img.write_to(&mut buf, ImageFormat::Png).unwrap_or(());
    buf.into_inner()
}

enum Event {
    Progress(usize),
    Error(String),
    Done(Vec<(TileIndex, Vec<u8>)>, usize),
}

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
                FailurePolicy::Strict => return Err(format!("Tile 404 Not Found: {url}")),
                FailurePolicy::Lenient => return Ok((tile, TileOutcome::Missing)),
                FailurePolicy::Ignore => {
                    return Ok((tile, TileOutcome::BlackFill(black_tile_png())))
                }
            },
            Ok(r) => {
                if retries >= 3 {
                    match policy {
                        FailurePolicy::Strict => {
                            return Err(format!("HTTP {} for URL: {}", r.status(), url))
                        }
                        FailurePolicy::Lenient => return Ok((tile, TileOutcome::Missing)),
                        FailurePolicy::Ignore => {
                            return Ok((tile, TileOutcome::BlackFill(black_tile_png())))
                        }
                    }
                }
            }
            Err(e) => {
                if retries >= 3 {
                    match policy {
                        FailurePolicy::Strict => return Err(format!("Network error: {e}")),
                        FailurePolicy::Lenient => return Ok((tile, TileOutcome::Missing)),
                        FailurePolicy::Ignore => {
                            return Ok((tile, TileOutcome::BlackFill(black_tile_png())))
                        }
                    }
                }
            }
        }
        retries += 1;
        tokio::time::sleep(Duration::from_millis(500 * u64::from(retries))).await;
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
                .user_agent("mapcv-fetcher/0.1.0")
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
        let event = py.allow_threads(|| {
            rx.lock()
                .unwrap()
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
