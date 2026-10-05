//! Byte sources for the GeoTIFF reader: local files and HTTP(S)/S3 range requests.
//!
//! The reader asks a source for a batch of `(offset, length)` byte ranges at a
//! time: the IFD values of one directory, or the tiles/strips under one window.
//! A remote source turns each batch into as few HTTP range requests as it can
//! (adjacent missing blocks are fetched together, several requests in flight)
//! and keeps recently used blocks in a bounded LRU cache.

use super::GeoTiffError;
use futures::stream::{self, StreamExt, TryStreamExt};
use reqwest::header::{CONTENT_RANGE, RANGE};
use reqwest::{Client, StatusCode, Url};
use std::collections::{BTreeMap, BTreeSet, HashMap};
use std::fs::File;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Mutex};
use std::time::Duration;

type Result<T> = std::result::Result<T, GeoTiffError>;

/// Random access to the bytes of a TIFF file.
pub trait ByteSource: Send + Sync {
    /// Size of the file in bytes.
    fn size(&self) -> u64;

    /// Read each `(offset, length)` range; the result has one buffer per range.
    ///
    /// # Errors
    /// Returns [`GeoTiffError::Invalid`] for a range past the end of the file
    /// and [`GeoTiffError::Io`] when reading fails.
    fn read_ranges(&self, ranges: &[(u64, usize)]) -> Result<Vec<Vec<u8>>>;

    /// Read `length` bytes at `offset`.
    ///
    /// # Errors
    /// As [`ByteSource::read_ranges`].
    fn read_at(&self, offset: u64, length: usize) -> Result<Vec<u8>> {
        let mut out = self.read_ranges(&[(offset, length)])?;
        out.pop()
            .ok_or_else(|| GeoTiffError::Io("internal error: empty read result".to_owned()))
    }

    /// The path or URL for error messages, without credentials or query string.
    fn describe(&self) -> String;
}

/// Check that a range lies inside a file of `size` bytes.
fn check_range(offset: u64, length: usize, size: u64, what: &str) -> Result<()> {
    let end = offset.checked_add(length as u64);
    match end {
        Some(end) if end <= size => Ok(()),
        _ => Err(GeoTiffError::Invalid(format!(
            "{what} is truncated or corrupt: it references bytes {offset}..{} but the file has \
             {size} bytes",
            offset.saturating_add(length as u64)
        ))),
    }
}

/// A local file read with positioned reads, so several threads can share it.
pub struct LocalFile {
    file: File,
    size: u64,
    path: String,
}

impl LocalFile {
    /// Open the file at `path`.
    ///
    /// # Errors
    /// Returns [`GeoTiffError::Io`] when the file cannot be opened.
    pub fn open(path: &str) -> Result<Self> {
        let file =
            File::open(path).map_err(|e| GeoTiffError::Io(format!("cannot open {path}: {e}")))?;
        let size = file
            .metadata()
            .map_err(|e| GeoTiffError::Io(format!("cannot read the size of {path}: {e}")))?
            .len();
        Ok(LocalFile {
            file,
            size,
            path: path.to_owned(),
        })
    }

    #[cfg(unix)]
    fn read_exact_at(&self, buf: &mut [u8], offset: u64) -> std::io::Result<()> {
        use std::os::unix::fs::FileExt;
        self.file.read_exact_at(buf, offset)
    }

    #[cfg(windows)]
    fn read_exact_at(&self, mut buf: &mut [u8], mut offset: u64) -> std::io::Result<()> {
        use std::os::windows::fs::FileExt;
        while !buf.is_empty() {
            match self.file.seek_read(buf, offset) {
                Ok(0) => return Err(std::io::ErrorKind::UnexpectedEof.into()),
                Ok(n) => {
                    buf = &mut buf[n..];
                    offset += n as u64;
                }
                Err(e) if e.kind() == std::io::ErrorKind::Interrupted => {}
                Err(e) => return Err(e),
            }
        }
        Ok(())
    }
}

impl ByteSource for LocalFile {
    fn size(&self) -> u64 {
        self.size
    }

    fn read_ranges(&self, ranges: &[(u64, usize)]) -> Result<Vec<Vec<u8>>> {
        ranges
            .iter()
            .map(|&(offset, length)| {
                check_range(offset, length, self.size, &self.path)?;
                let mut buf = vec![0u8; length];
                self.read_exact_at(&mut buf, offset).map_err(|e| {
                    GeoTiffError::Io(format!(
                        "cannot read {length} bytes at offset {offset} of {}: {e}",
                        self.path
                    ))
                })?;
                Ok(buf)
            })
            .collect()
    }

    fn describe(&self) -> String {
        self.path.clone()
    }
}

/// Bytes per cache block; every HTTP request covers whole blocks.
const BLOCK_BYTES: usize = 16 * 1024;
const BLOCK_SIZE: u64 = BLOCK_BYTES as u64;
/// Bytes requested when a remote file is opened: COGs keep their IFDs here.
const INITIAL_FETCH: u64 = 4 * BLOCK_SIZE;
/// Largest single range request, so one response cannot use unbounded memory.
const MAX_REQUEST_BYTES: u64 = 8 * 1024 * 1024;
/// Range requests in flight at once for one read.
const MAX_CONCURRENT_REQUESTS: usize = 8;
/// Retries after the first attempt for a timeout, connection error, 429 or 5xx.
const MAX_RETRIES: u32 = 3;
const RETRY_BACKOFF_MS: u64 = 500;

/// Least-recently-used cache of fixed-size blocks of a remote file.
struct BlockCache {
    blocks: HashMap<u64, (Arc<Vec<u8>>, u64)>,
    by_age: BTreeMap<u64, u64>,
    tick: u64,
    capacity: usize,
}

impl BlockCache {
    fn new(capacity_bytes: usize) -> Self {
        BlockCache {
            blocks: HashMap::new(),
            by_age: BTreeMap::new(),
            tick: 0,
            capacity: capacity_bytes / BLOCK_BYTES,
        }
    }

    fn get(&mut self, block: u64) -> Option<Arc<Vec<u8>>> {
        self.tick += 1;
        let tick = self.tick;
        let (data, age) = self.blocks.get_mut(&block)?;
        self.by_age.remove(age);
        *age = tick;
        self.by_age.insert(tick, block);
        Some(Arc::clone(data))
    }

    fn insert(&mut self, block: u64, data: Arc<Vec<u8>>) {
        if self.capacity == 0 {
            return;
        }
        self.tick += 1;
        if let Some((_, old_age)) = self.blocks.insert(block, (data, self.tick)) {
            self.by_age.remove(&old_age);
        }
        self.by_age.insert(self.tick, block);
        while self.blocks.len() > self.capacity {
            let Some((_, oldest)) = self.by_age.pop_first() else {
                break;
            };
            self.blocks.remove(&oldest);
        }
    }
}

/// Turn `s3://bucket/key` into its public HTTPS URL and reject URLs that
/// carry credentials or use another scheme.
///
/// # Errors
/// Returns [`GeoTiffError::Invalid`] for an unparsable URL, an unsupported
/// scheme, embedded credentials or an S3 URL without bucket or key.
pub fn resolve_url(url: &str) -> Result<Url> {
    let parsed = Url::parse(url)
        .map_err(|e| GeoTiffError::Invalid(format!("invalid URL {}: {e}", sanitize_url(url))))?;
    if !parsed.username().is_empty() || parsed.password().is_some() {
        return Err(GeoTiffError::Invalid(format!(
            "URLs with embedded credentials are not supported (only anonymous access is): {}",
            sanitize_url(url)
        )));
    }
    match parsed.scheme() {
        "http" | "https" => Ok(parsed),
        "s3" => {
            let bucket = parsed.host_str().unwrap_or_default();
            let key = parsed.path().trim_start_matches('/');
            if bucket.is_empty() || key.is_empty() {
                return Err(GeoTiffError::Invalid(format!(
                    "S3 URL must look like s3://bucket/key, got {}",
                    sanitize_url(url)
                )));
            }
            // Virtual-hosted style on the global endpoint, which routes to the
            // bucket's region. Bucket names with dots do not match the
            // wildcard certificate, so those use path style.
            let base = if bucket.contains('.') {
                format!("https://s3.amazonaws.com/{bucket}/")
            } else {
                format!("https://{bucket}.s3.amazonaws.com/")
            };
            let mut https = Url::parse(&base).map_err(|e| {
                GeoTiffError::Invalid(format!("invalid S3 bucket name in {url}: {e}"))
            })?;
            // The key is already percent-encoded by the s3:// URL parser.
            https.set_path(&format!("{}{key}", https.path()));
            https.set_query(parsed.query());
            Ok(https)
        }
        other => Err(GeoTiffError::Invalid(format!(
            "unsupported URL scheme {other:?}: use a local path, http(s):// or s3://"
        ))),
    }
}

/// Strip credentials, query string and fragment from a URL for messages.
#[must_use]
pub fn sanitize_url(url: &str) -> String {
    if let Ok(mut parsed) = Url::parse(url) {
        let _ = parsed.set_username("");
        let _ = parsed.set_password(None);
        parsed.set_query(None);
        parsed.set_fragment(None);
        return parsed.to_string();
    }
    url.split(['?', '#']).next().unwrap_or(url).to_owned()
}

/// A remote file read with HTTP range requests through a block cache.
pub struct HttpSource {
    url: Url,
    display: String,
    size: u64,
    client: Client,
    runtime: tokio::runtime::Runtime,
    cache: Mutex<BlockCache>,
    requests: AtomicU64,
}

/// One validated `206 Partial Content` response.
struct RangeResponse {
    start: u64,
    data: Vec<u8>,
    total: u64,
}

impl HttpSource {
    /// Open a remote file, fetching its first blocks to learn its size.
    ///
    /// `cache_bytes` bounds the memory kept for blocks between reads.
    ///
    /// # Errors
    /// Returns [`GeoTiffError::Invalid`] for an unsupported URL and
    /// [`GeoTiffError::Io`] when the server cannot be reached, answers with an
    /// error, or does not support range requests.
    pub fn open(url: &str, cache_bytes: usize) -> Result<Self> {
        let resolved = resolve_url(url)?;
        let display = sanitize_url(resolved.as_str());
        let runtime = tokio::runtime::Builder::new_current_thread()
            .enable_all()
            .build()
            .map_err(|e| GeoTiffError::Io(format!("cannot start the HTTP runtime: {e}")))?;
        let client = Client::builder()
            .connect_timeout(Duration::from_secs(10))
            .timeout(Duration::from_secs(60))
            .user_agent(concat!(
                "mapcv/",
                env!("CARGO_PKG_VERSION"),
                " (+https://github.com/tahamukhtar20/mapcv)"
            ))
            .build()
            .map_err(|e| GeoTiffError::Io(format!("cannot build the HTTP client: {e}")))?;
        let mut source = HttpSource {
            url: resolved,
            display,
            size: 0,
            client,
            runtime,
            cache: Mutex::new(BlockCache::new(cache_bytes)),
            requests: AtomicU64::new(0),
        };
        let first = source
            .runtime
            .block_on(source.fetch(0, INITIAL_FETCH, None))?;
        source.size = first.total;
        source.store_blocks(0, &first.data, &mut HashMap::new())?;
        Ok(source)
    }

    /// Number of HTTP requests sent so far (including retries).
    pub fn request_count(&self) -> u64 {
        self.requests.load(Ordering::Relaxed)
    }

    /// Fetch bytes `start..start + length` (clamped to the file size once known).
    async fn fetch(
        &self,
        start: u64,
        length: u64,
        expected_total: Option<u64>,
    ) -> Result<RangeResponse> {
        let mut attempt = 0;
        loop {
            match self.fetch_once(start, length, expected_total).await {
                Ok(response) => return Ok(response),
                Err((error, retryable)) => {
                    if !retryable || attempt >= MAX_RETRIES {
                        return Err(error);
                    }
                    attempt += 1;
                    tokio::time::sleep(Duration::from_millis(RETRY_BACKOFF_MS << (attempt - 1)))
                        .await;
                }
            }
        }
    }

    /// One attempt of [`HttpSource::fetch`]; the flag says whether a retry may help.
    async fn fetch_once(
        &self,
        start: u64,
        length: u64,
        expected_total: Option<u64>,
    ) -> std::result::Result<RangeResponse, (GeoTiffError, bool)> {
        let end = start + length - 1;
        self.requests.fetch_add(1, Ordering::Relaxed);
        let mut response = self
            .client
            .get(self.url.clone())
            .header(RANGE, format!("bytes={start}-{end}"))
            .send()
            .await
            .map_err(|e| {
                let retry = e.is_timeout() || e.is_connect() || e.is_request();
                (
                    GeoTiffError::Io(format!("request to {} failed: {}", self.display, kind(&e))),
                    retry,
                )
            })?;
        let status = response.status();
        if status == StatusCode::OK {
            return Err((
                GeoTiffError::Io(format!(
                    "{} does not support HTTP range requests (it answered 200 OK to a Range \
                     request); remote GeoTIFFs are read in ranges, so download the file and open \
                     the local copy instead",
                    self.display
                )),
                false,
            ));
        }
        if status != StatusCode::PARTIAL_CONTENT {
            let retry = status == StatusCode::TOO_MANY_REQUESTS || status.is_server_error();
            return Err((
                GeoTiffError::Io(format!("HTTP {status} for {}", self.display)),
                retry,
            ));
        }
        let header = response
            .headers()
            .get(CONTENT_RANGE)
            .and_then(|v| v.to_str().ok())
            .unwrap_or_default()
            .to_owned();
        let (got_start, got_end, total) = parse_content_range(&header).ok_or_else(|| {
            (
                GeoTiffError::Io(format!(
                    "{} answered 206 with an invalid Content-Range header {header:?}",
                    self.display
                )),
                false,
            )
        })?;
        if let Some(expected) = expected_total {
            if total != expected {
                return Err((
                    GeoTiffError::Io(format!(
                        "{} changed size while being read ({expected} -> {total} bytes)",
                        self.display
                    )),
                    false,
                ));
            }
        }
        let want_end = end.min(total.saturating_sub(1));
        if got_start != start || got_end != want_end {
            return Err((
                GeoTiffError::Io(format!(
                    "{} answered bytes {got_start}-{got_end} to a request for {start}-{want_end}",
                    self.display
                )),
                false,
            ));
        }
        let expected_len = usize::try_from(want_end - start + 1).unwrap_or(usize::MAX);
        let data = self.read_body(&mut response, expected_len).await?;
        Ok(RangeResponse { start, data, total })
    }

    /// Read a response body that must be exactly `expected_len` bytes long,
    /// stopping early if the server sends more.
    async fn read_body(
        &self,
        response: &mut reqwest::Response,
        expected_len: usize,
    ) -> std::result::Result<Vec<u8>, (GeoTiffError, bool)> {
        let mut data = Vec::with_capacity(expected_len);
        loop {
            let chunk = response.chunk().await.map_err(|e| {
                let message = format!(
                    "reading the response from {} failed: {}",
                    self.display,
                    kind(&e)
                );
                (GeoTiffError::Io(message), true)
            })?;
            let Some(chunk) = chunk else { break };
            if data.len() + chunk.len() > expected_len {
                let message = format!("{} sent more bytes than the requested range", self.display);
                return Err((GeoTiffError::Io(message), false));
            }
            data.extend_from_slice(&chunk);
        }
        if data.len() != expected_len {
            let message = format!(
                "{} sent {} bytes for a {expected_len}-byte range",
                self.display,
                data.len()
            );
            return Err((GeoTiffError::Io(message), true));
        }
        Ok(data)
    }

    /// Split fetched bytes starting at block-aligned `start` into blocks, keep
    /// them in `local` for the current read and add them to the cache.
    fn store_blocks(
        &self,
        start: u64,
        data: &[u8],
        local: &mut HashMap<u64, Arc<Vec<u8>>>,
    ) -> Result<()> {
        let mut cache = self.lock_cache()?;
        for (i, piece) in data.chunks(BLOCK_BYTES).enumerate() {
            let block = start / BLOCK_SIZE + i as u64;
            let piece = Arc::new(piece.to_vec());
            cache.insert(block, Arc::clone(&piece));
            local.insert(block, piece);
        }
        Ok(())
    }

    fn lock_cache(&self) -> Result<std::sync::MutexGuard<'_, BlockCache>> {
        self.cache
            .lock()
            .map_err(|_| GeoTiffError::Io("internal error: block cache lock poisoned".to_owned()))
    }
}

/// Group sorted block numbers into runs of consecutive blocks, each at most
/// [`MAX_REQUEST_BYTES`] long; returns `(first_block, block_count)` pairs.
fn coalesce(blocks: &[u64]) -> Vec<(u64, u64)> {
    let max_blocks = MAX_REQUEST_BYTES / BLOCK_SIZE;
    let mut runs: Vec<(u64, u64)> = Vec::new();
    for &block in blocks {
        match runs.last_mut() {
            Some((first, count)) if *first + *count == block && *count < max_blocks => *count += 1,
            _ => runs.push((block, 1)),
        }
    }
    runs
}

/// Parse `bytes start-end/total` (the total must be known).
fn parse_content_range(header: &str) -> Option<(u64, u64, u64)> {
    let rest = header.trim().strip_prefix("bytes")?.trim_start();
    let (range, total) = rest.split_once('/')?;
    let (start, end) = range.split_once('-')?;
    let start: u64 = start.trim().parse().ok()?;
    let end: u64 = end.trim().parse().ok()?;
    let total: u64 = total.trim().parse().ok()?;
    (start <= end && end < total).then_some((start, end, total))
}

fn kind(error: &reqwest::Error) -> &'static str {
    if error.is_timeout() {
        "request timed out"
    } else if error.is_connect() {
        "connection failed"
    } else if error.is_body() || error.is_decode() {
        "response body could not be read"
    } else {
        "request failed"
    }
}

impl ByteSource for HttpSource {
    fn size(&self) -> u64 {
        self.size
    }

    fn read_ranges(&self, ranges: &[(u64, usize)]) -> Result<Vec<Vec<u8>>> {
        let mut needed = BTreeSet::new();
        for &(offset, length) in ranges {
            check_range(offset, length, self.size, &self.display)?;
            if length > 0 {
                let last = offset + length as u64 - 1;
                needed.extend(offset / BLOCK_SIZE..=last / BLOCK_SIZE);
            }
        }
        let mut local: HashMap<u64, Arc<Vec<u8>>> = HashMap::new();
        let mut missing = Vec::new();
        {
            let mut cache = self.lock_cache()?;
            for &block in &needed {
                match cache.get(block) {
                    Some(data) => {
                        local.insert(block, data);
                    }
                    None => missing.push(block),
                }
            }
        }
        if !missing.is_empty() {
            let runs = coalesce(&missing);
            let size = self.size;
            let responses: Vec<RangeResponse> = self.runtime.block_on(
                stream::iter(runs)
                    .map(|(first, count)| {
                        let start = first * BLOCK_SIZE;
                        let length = (count * BLOCK_SIZE).min(size - start);
                        self.fetch(start, length, Some(size))
                    })
                    .buffer_unordered(MAX_CONCURRENT_REQUESTS)
                    .try_collect(),
            )?;
            for response in responses {
                self.store_blocks(response.start, &response.data, &mut local)?;
            }
        }
        ranges
            .iter()
            .map(|&(offset, length)| {
                let mut out = Vec::with_capacity(length);
                let mut pos = offset;
                let end = offset + length as u64;
                while pos < end {
                    let block = pos / BLOCK_SIZE;
                    let data = local.get(&block).ok_or_else(|| {
                        GeoTiffError::Io(format!("internal error: block {block} was not fetched"))
                    })?;
                    let within = usize::try_from(pos - block * BLOCK_SIZE)
                        .map_err(|e| GeoTiffError::Io(e.to_string()))?;
                    let take =
                        (data.len() - within).min(usize::try_from(end - pos).unwrap_or(usize::MAX));
                    out.extend_from_slice(&data[within..within + take]);
                    pos += take as u64;
                }
                Ok(out)
            })
            .collect()
    }

    fn describe(&self) -> String {
        self.display.clone()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn s3_urls_map_to_public_https() {
        assert_eq!(
            resolve_url("s3://sentinel-cogs/sentinel-s2-l2a-cogs/x/B04.tif")
                .unwrap()
                .as_str(),
            "https://sentinel-cogs.s3.amazonaws.com/sentinel-s2-l2a-cogs/x/B04.tif"
        );
        assert_eq!(
            resolve_url("s3://my.bucket/a b/c.tif").unwrap().as_str(),
            "https://s3.amazonaws.com/my.bucket/a%20b/c.tif"
        );
        assert!(resolve_url("s3://bucket").is_err());
    }

    #[test]
    fn credentials_and_other_schemes_are_rejected() {
        let err = resolve_url("https://user:secret@example.com/a.tif").unwrap_err();
        assert!(err.to_string().contains("credentials"));
        assert!(!err.to_string().contains("secret"));
        assert!(resolve_url("ftp://example.com/a.tif").is_err());
        assert!(resolve_url("https://example.com/a.tif?sig=1").is_ok());
    }

    #[test]
    fn content_range_parsing() {
        assert_eq!(parse_content_range("bytes 0-99/1000"), Some((0, 99, 1000)));
        assert_eq!(parse_content_range("bytes 0-99/*"), None);
        assert_eq!(parse_content_range("bytes 5-4/10"), None);
        assert_eq!(parse_content_range("bytes 0-10/10"), None);
    }

    #[test]
    fn adjacent_blocks_coalesce_into_bounded_runs() {
        assert_eq!(
            coalesce(&[1, 2, 3, 7, 8, 10]),
            vec![(1, 3), (7, 2), (10, 1)]
        );
        let max = MAX_REQUEST_BYTES / BLOCK_SIZE;
        let many: Vec<u64> = (0..max + 5).collect();
        assert_eq!(coalesce(&many), vec![(0, max), (max, 5)]);
    }

    #[test]
    fn block_cache_evicts_least_recently_used() {
        let mut cache = BlockCache::new(2 * BLOCK_BYTES);
        cache.insert(1, Arc::new(vec![1]));
        cache.insert(2, Arc::new(vec![2]));
        assert!(cache.get(1).is_some());
        cache.insert(3, Arc::new(vec![3]));
        assert!(cache.get(2).is_none());
        assert!(cache.get(1).is_some());
        assert!(cache.get(3).is_some());
    }
}
