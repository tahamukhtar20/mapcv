//! Asynchronous tile fetcher using reqwest and tokio.

use crate::tile_math::TileIndex;
use futures::stream::{self, StreamExt};
use pyo3::exceptions::PyRuntimeError;
use pyo3::prelude::*;
use reqwest::Client;
use std::sync::mpsc;
use std::thread;
use std::time::Duration;

/// Determines how fetch failures (e.g. 404 Not Found) are handled.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum FailurePolicy {
    /// Fails the entire fetch operation immediately.
    Strict,
    /// Ignores the failed tile and omits it from the result.
    Lenient,
    /// Returns an empty/black tile buffer. (Currently not fully implemented, behaves like Lenient).
    Ignore,
}

impl FailurePolicy {
    /// Parse a FailurePolicy from a string.
    pub fn from_str(s: &str) -> Result<Self, String> {
        match s.to_lowercase().as_str() {
            "strict" => Ok(FailurePolicy::Strict),
            "lenient" => Ok(FailurePolicy::Lenient),
            "ignore" => Ok(FailurePolicy::Ignore),
            _ => Err(format!("Unknown policy: {}", s)),
        }
    }
}

enum Event {
    Progress(usize),
    Error(String),
    Done(Vec<(TileIndex, Vec<u8>)>),
}

async fn fetch_single_tile(
    client: Client,
    tile: TileIndex,
    url_template: String,
    policy: FailurePolicy,
) -> Result<(TileIndex, Option<Vec<u8>>), String> {
    let url = url_template
        .replace("{z}", &tile.z.to_string())
        .replace("{x}", &tile.x.to_string())
        .replace("{y}", &tile.y.to_string());

    let mut retries = 0;
    loop {
        let resp = client.get(&url).send().await;
        match resp {
            Ok(r) if r.status().is_success() => {
                let bytes = r.bytes().await.map_err(|e| e.to_string())?;
                return Ok((tile, Some(bytes.to_vec())));
            }
            Ok(r) if r.status() == reqwest::StatusCode::NOT_FOUND => match policy {
                FailurePolicy::Strict => return Err(format!("Tile 404 Not Found: {}", url)),
                FailurePolicy::Lenient | FailurePolicy::Ignore => return Ok((tile, None)),
            },
            Ok(r) => {
                if retries >= 3 {
                    match policy {
                        FailurePolicy::Strict => {
                            return Err(format!("HTTP {} for URL: {}", r.status(), url))
                        }
                        FailurePolicy::Lenient | FailurePolicy::Ignore => return Ok((tile, None)),
                    }
                }
            }
            Err(e) => {
                if retries >= 3 {
                    match policy {
                        FailurePolicy::Strict => return Err(format!("Network error: {}", e)),
                        FailurePolicy::Lenient | FailurePolicy::Ignore => return Ok((tile, None)),
                    }
                }
            }
        }
        retries += 1;
        tokio::time::sleep(Duration::from_millis(500 * retries as u64)).await;
    }
}

/// Fetches multiple tiles concurrently.
/// This spawns a background Tokio runtime and sends progress via a cross-thread channel back to Python.
pub fn fetch_tiles(
    py: Python,
    tiles: Vec<TileIndex>,
    url_template: String,
    callback: Option<PyObject>,
    max_connections: usize,
    policy_str: &str,
) -> PyResult<Vec<(TileIndex, Vec<u8>)>> {
    let policy = FailurePolicy::from_str(policy_str).map_err(|e| PyRuntimeError::new_err(e))?;

    let (tx, rx) = mpsc::channel();

    thread::spawn(move || {
        let rt = match tokio::runtime::Runtime::new() {
            Ok(rt) => rt,
            Err(e) => {
                let _ = tx.send(Event::Error(format!(
                    "Failed to create tokio runtime: {}",
                    e
                )));
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
                    let _ = tx.send(Event::Error(format!("Failed to build HTTP client: {}", e)));
                    return;
                }
            };

            let stream = stream::iter(tiles)
                .map(|tile| {
                    let client_clone = client.clone();
                    let url_clone = url_template.clone();
                    async move { fetch_single_tile(client_clone, tile, url_clone, policy).await }
                })
                .buffer_unordered(max_connections);

            let mut stream = stream;
            let mut results = Vec::new();
            let mut completed = 0;

            while let Some(res) = stream.next().await {
                match res {
                    Ok((tile, Some(bytes))) => {
                        results.push((tile, bytes));
                    }
                    Ok((_tile, None)) => {}
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

            let _ = tx.send(Event::Done(results));
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
            Event::Done(results) => {
                return Ok(results);
            }
        }
    }
}
