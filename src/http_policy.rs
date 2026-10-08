//! Which HTTP redirects mapcv follows.
//!
//! A tile URL template or a remote GeoTIFF URL is checked when the config is
//! validated, but the server it names decides where a redirect goes. These
//! policies check every hop again, so a redirect cannot take a request (and,
//! for a tile template, its key) somewhere the configured URL could not go.
//! The clients that use them also never send a `Referer` header, which would
//! carry the previous URL, query string included, to the next host.
//!
//! A host *name* is judged by the addresses it resolves to, at connect time:
//! [`PublicResolver`] drops every address on this machine or a private network,
//! so neither a name pointing there nor a DNS answer that changes between a
//! check and the connection reaches one. A request that starts at such an
//! address named literally (a local test server) may stay there, unless the
//! process only allows public addresses ([`set_public_only`], used by the MCP
//! server) or `MAPCV_ALLOW_LOCAL_URLS=1` lets every request reach them.

use reqwest::dns::{Addrs, Name, Resolve, Resolving};
use reqwest::redirect::{Attempt, Policy};
use reqwest::Url;
use std::net::{IpAddr, Ipv4Addr, Ipv6Addr, SocketAddr, ToSocketAddrs};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;

/// Longest redirect chain followed (as `reqwest`'s default policy).
const MAX_REDIRECTS: usize = 10;

/// Why a redirect was refused; the message names hosts only, never paths or queries.
#[derive(Debug)]
pub struct RedirectRefused(pub String);

impl std::fmt::Display for RedirectRefused {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(&self.0)
    }
}

impl std::error::Error for RedirectRefused {}

/// `scheme://host[:port]` of a URL: what messages show of a redirect target.
fn origin(url: &Url) -> String {
    let host = url.host_str().unwrap_or("");
    match url.port() {
        Some(port) => format!("{}://{host}:{port}", url.scheme()),
        None => format!("{}://{host}", url.scheme()),
    }
}

fn internal_v4(ip: Ipv4Addr) -> bool {
    ip.is_loopback()
        || ip.is_private()
        || ip.is_link_local()
        || ip.is_unspecified()
        || ip.is_broadcast()
        || ip.is_multicast()
        // 100.64.0.0/10 (carrier-grade NAT; 100.100.100.200 is a cloud metadata address).
        || (ip.octets()[0] == 100 && (ip.octets()[1] & 0xC0) == 64)
}

fn internal_v6(ip: Ipv6Addr) -> bool {
    if let Some(v4) = ip.to_ipv4_mapped() {
        return internal_v4(v4);
    }
    let first = ip.segments()[0];
    ip.is_loopback()
        || ip.is_unspecified()
        || ip.is_multicast()
        || (first & 0xFE00) == 0xFC00 // unique local, fc00::/7
        || (first & 0xFFC0) == 0xFE80 // link local, fe80::/10
}

/// Whether an address is on this machine or a private network: loopback, private
/// (RFC 1918, `fc00::/7`), link-local, carrier-grade NAT, unspecified, broadcast or
/// multicast, IPv4-mapped IPv6 addresses included.
#[must_use]
pub fn is_internal_ip(ip: IpAddr) -> bool {
    match ip {
        IpAddr::V4(ip) => internal_v4(ip),
        IpAddr::V6(ip) => internal_v6(ip),
    }
}

/// Set when every request of this process must stay on public addresses.
static PUBLIC_ONLY: AtomicBool = AtomicBool::new(false);

/// The environment variable that lets requests reach this machine and private networks.
pub const ALLOW_LOCAL_ENV: &str = "MAPCV_ALLOW_LOCAL_URLS";

/// Make every request of this process stay on public addresses (`true`), even one whose
/// URL names this machine, or go back to the default (`false`). The MCP server sets it.
pub fn set_public_only(on: bool) {
    PUBLIC_ONLY.store(on, Ordering::SeqCst);
}

/// Whether every request of this process must stay on public addresses.
#[must_use]
pub fn public_only() -> bool {
    PUBLIC_ONLY.load(Ordering::SeqCst)
}

fn local_allowed_by_env() -> bool {
    std::env::var(ALLOW_LOCAL_ENV).is_ok_and(|value| value.trim() == "1")
}

/// Whether a request that started at `start` may reach this machine or a private
/// network: when `start` names one literally (an IP address or `localhost`) or
/// `MAPCV_ALLOW_LOCAL_URLS=1` is set, and the process is not limited to public addresses.
#[must_use]
pub fn may_reach_internal(start: &Url) -> bool {
    !public_only() && (is_internal(start) || local_allowed_by_env())
}

/// Why `host` may not be reached; the same text whatever is (or is not) listening there.
fn internal_refusal(host: &str) -> String {
    if public_only() {
        format!(
            "{host} is on this machine or a private network, and this mapcv MCP server \
             connects to public addresses only (start it with --allow-local-urls to use a \
             local server)"
        )
    } else {
        format!(
            "{host} resolves to an address on this machine or a private network, which \
             mapcv does not connect to for a URL that names a public host; to use a server \
             on your own network, put its IP address in the URL or set {ALLOW_LOCAL_ENV}=1"
        )
    }
}

/// Why a request may not start at `start`, or `None`: when the process is limited to
/// public addresses, a URL naming this machine or a private network is refused before
/// anything is sent.
#[must_use]
pub fn start_refusal(start: &Url) -> Option<String> {
    (public_only() && is_internal(start)).then(|| internal_refusal(&origin(start)))
}

/// A name that resolved only to addresses on this machine or a private network.
#[derive(Debug)]
pub struct AddressRefused(pub String);

impl std::fmt::Display for AddressRefused {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(&self.0)
    }
}

impl std::error::Error for AddressRefused {}

/// How a name is turned into addresses (replaceable in tests).
pub type Lookup = dyn Fn(&str) -> std::io::Result<Vec<IpAddr>> + Send + Sync;

fn system_lookup(host: &str) -> std::io::Result<Vec<IpAddr>> {
    Ok((host, 0)
        .to_socket_addrs()?
        .map(|address| address.ip())
        .collect())
}

/// The hosts of the proxies `reqwest` takes from the environment: the client
/// resolves a proxy's name itself, and the operator chose it, so it is not filtered.
fn proxy_hosts() -> Vec<String> {
    [
        "HTTP_PROXY",
        "http_proxy",
        "HTTPS_PROXY",
        "https_proxy",
        "ALL_PROXY",
        "all_proxy",
    ]
    .iter()
    .filter_map(|name| std::env::var(name).ok())
    .filter_map(|value| {
        let value = value.trim().to_owned();
        let text = if value.contains("://") {
            value
        } else {
            format!("http://{value}")
        };
        Url::parse(&text).ok()?.host_str().map(str::to_owned)
    })
    .collect()
}

/// A DNS resolver that keeps only public addresses.
///
/// The addresses are checked when the connection is made, after every lookup, so a
/// name (or a redirect to a name) cannot lead a request to this machine or a private
/// network, and an answer that changes between a check and the connection does not
/// either. A name with no public address fails with [`AddressRefused`].
pub struct PublicResolver {
    lookup: Arc<Lookup>,
    exempt: Vec<String>,
    /// Which addresses are dropped ([`is_internal_ip`]; tests pick their own).
    internal: fn(IpAddr) -> bool,
}

impl PublicResolver {
    /// A resolver that asks the system resolver.
    #[must_use]
    pub fn new() -> Self {
        Self::with_lookup(Arc::new(system_lookup))
    }

    /// A resolver that asks `lookup` instead of the system (for tests).
    #[must_use]
    pub fn with_lookup(lookup: Arc<Lookup>) -> Self {
        Self {
            lookup,
            exempt: proxy_hosts(),
            internal: is_internal_ip,
        }
    }
}

impl Default for PublicResolver {
    fn default() -> Self {
        Self::new()
    }
}

impl Resolve for PublicResolver {
    fn resolve(&self, name: Name) -> Resolving {
        let host = name.as_str().to_owned();
        let lookup = Arc::clone(&self.lookup);
        let internal = self.internal;
        let exempt = self
            .exempt
            .iter()
            .any(|proxy| proxy.eq_ignore_ascii_case(&host));
        Box::pin(async move {
            let asked = host.clone();
            let found = tokio::task::spawn_blocking(move || lookup(&asked)).await??;
            if found.is_empty() {
                return Err(format!("{host} has no address").into());
            }
            let kept: Vec<SocketAddr> = found
                .into_iter()
                .filter(|ip| exempt || !internal(*ip))
                .map(|ip| SocketAddr::new(ip, 0))
                .collect();
            if kept.is_empty() {
                return Err(AddressRefused(internal_refusal(&host)).into());
            }
            let addresses: Addrs = Box::new(kept.into_iter());
            Ok(addresses)
        })
    }
}

/// `builder` with the resolver a request starting at `start` needs: one that keeps
/// only public addresses, unless the request may reach this machine or a private
/// network ([`may_reach_internal`]).
pub fn with_resolver(builder: reqwest::ClientBuilder, start: &Url) -> reqwest::ClientBuilder {
    if may_reach_internal(start) {
        builder
    } else {
        builder.dns_resolver(PublicResolver::new())
    }
}

/// The host of a URL as an IP address, if it is one (`[::1]` without brackets).
fn host_ip(url: &Url) -> Option<IpAddr> {
    let host = url.host_str()?;
    host.trim_start_matches('[')
        .trim_end_matches(']')
        .parse()
        .ok()
}

fn is_localhost_name(url: &Url) -> bool {
    url.host_str().is_some_and(|name| {
        let name = name.trim_end_matches('.').to_ascii_lowercase();
        name == "localhost" || name.ends_with(".localhost")
    })
}

/// Whether a URL names this machine or a private network: a loopback, private,
/// link-local or unspecified IP address, or `localhost`.
#[must_use]
pub fn is_internal(url: &Url) -> bool {
    match host_ip(url) {
        Some(IpAddr::V4(ip)) => internal_v4(ip),
        Some(IpAddr::V6(ip)) => internal_v6(ip),
        None => url.host_str().is_none() || is_localhost_name(url),
    }
}

/// Whether a URL names this machine (`localhost` or a loopback address).
#[must_use]
pub fn is_loopback(url: &Url) -> bool {
    match host_ip(url) {
        Some(IpAddr::V4(ip)) => ip.is_loopback(),
        Some(IpAddr::V6(ip)) => {
            ip.is_loopback() || ip.to_ipv4_mapped().is_some_and(|v4| v4.is_loopback())
        }
        None => url
            .host_str()
            .is_some_and(|name| name.eq_ignore_ascii_case("localhost")),
    }
}

fn same_host(a: &Url, b: &Url) -> bool {
    a.host_str()
        .zip(b.host_str())
        .is_some_and(|(x, y)| x.eq_ignore_ascii_case(y))
}

/// Rules both policies share: http(s) only, no credentials, no `https -> http`
/// downgrade, and no hop from a public host into this machine or a private network.
fn common_refusal(from: &Url, to: &Url) -> Option<String> {
    if !matches!(to.scheme(), "http" | "https") {
        return Some(format!(
            "the server redirected to a {} URL; only http(s) redirects are followed",
            to.scheme()
        ));
    }
    if !to.username().is_empty() || to.password().is_some() {
        return Some(format!(
            "the server redirected to {} with credentials in the URL, which mapcv does not follow",
            origin(to)
        ));
    }
    if from.scheme() == "https" && to.scheme() == "http" {
        return Some(format!(
            "the server redirected from https to plain http ({}), which mapcv does not follow",
            origin(to)
        ));
    }
    if is_internal(to) && !may_reach_internal(from) {
        if public_only() {
            return Some(internal_refusal(&origin(to)));
        }
        return Some(format!(
            "the server redirected to {}, an address on this machine or a private network, \
             which mapcv does not follow from a public server",
            origin(to)
        ));
    }
    None
}

/// Why a tile request must not follow a redirect from `from` (the URL the
/// template gave) to `to`, or `None` to follow it.
///
/// Besides the shared rules, a URL with a query string (where providers put
/// keys) is redirected only within its own host, so the key's server is the
/// only one that ever sees a request.
#[must_use]
pub fn tile_redirect_refusal(from: &Url, to: &Url) -> Option<String> {
    if let Some(reason) = common_refusal(from, to) {
        return Some(reason);
    }
    if from.query().is_some() && !same_host(from, to) {
        return Some(format!(
            "the tile server redirected to another host ({}); mapcv follows redirects of a \
             url_template with a query string only within its own host. Put the final tile \
             URL in imagery.url_template",
            origin(to)
        ));
    }
    None
}

/// Why a GeoTIFF range request must not follow a redirect from `from` (the
/// configured URL) to `to`, or `None` to follow it.
///
/// Besides the shared rules, the target gets the rule the configured URL had:
/// plain http only from a loopback URL to a loopback URL.
#[must_use]
pub fn geotiff_redirect_refusal(from: &Url, to: &Url) -> Option<String> {
    if let Some(reason) = common_refusal(from, to) {
        return Some(reason);
    }
    if to.scheme() == "http" && !(is_loopback(from) && is_loopback(to)) {
        return Some(format!(
            "the server redirected to plain http ({}); remote GeoTIFFs are read over https \
             (plain http only on this machine)",
            origin(to)
        ));
    }
    None
}

fn policy(refusal: fn(&Url, &Url) -> Option<String>) -> Policy {
    Policy::custom(move |attempt: Attempt<'_>| {
        // The first of `previous` is the URL first requested (credentials already moved
        // to a header by reqwest), the last the one that answered with this redirect.
        let Some(from) = attempt.previous().first() else {
            return attempt.follow();
        };
        if attempt.previous().len() > MAX_REDIRECTS {
            return attempt.error(RedirectRefused("too many redirects".to_owned()));
        }
        match refusal(from, attempt.url()) {
            Some(reason) => attempt.error(RedirectRefused(reason)),
            None => attempt.follow(),
        }
    })
}

/// The redirect policy of the tile fetcher; see [`tile_redirect_refusal`].
#[must_use]
pub fn tile_redirect_policy() -> Policy {
    policy(tile_redirect_refusal)
}

/// The redirect policy of the remote GeoTIFF reader; see [`geotiff_redirect_refusal`].
#[must_use]
pub fn geotiff_redirect_policy() -> Policy {
    policy(geotiff_redirect_refusal)
}

/// The reason a redirect was refused, if that is what failed `error`.
#[must_use]
pub fn redirect_refusal(error: &reqwest::Error) -> Option<String> {
    if !error.is_redirect() {
        return None;
    }
    let mut source = std::error::Error::source(error);
    while let Some(inner) = source {
        if let Some(refused) = inner.downcast_ref::<RedirectRefused>() {
            return Some(refused.0.clone());
        }
        source = inner.source();
    }
    Some("redirect refused".to_owned())
}

/// The reason a host was refused for its addresses, if that is what failed `error`.
#[must_use]
pub fn address_refusal(error: &reqwest::Error) -> Option<String> {
    let mut source = std::error::Error::source(error);
    while let Some(inner) = source {
        if let Some(refused) = inner.downcast_ref::<AddressRefused>() {
            return Some(refused.0.clone());
        }
        source = inner.source();
    }
    None
}

#[cfg(test)]
mod tests {
    use super::*;

    fn url(text: &str) -> Url {
        Url::parse(text).unwrap()
    }

    #[test]
    fn tile_redirects_within_the_host_are_followed() {
        let from = url("https://tiles.example.com/1/2/3.png?key=SECRET");
        assert!(
            tile_redirect_refusal(&from, &url("https://tiles.example.com/v2/1/2/3.png")).is_none()
        );
        let plain = url("http://tiles.example.com/1/2/3.png");
        assert!(
            tile_redirect_refusal(&plain, &url("https://tiles.example.com/1/2/3.png")).is_none()
        );
        // Without a query string, a public CDN is fine.
        assert!(tile_redirect_refusal(&plain, &url("https://cdn.example.net/1/2/3.png")).is_none());
    }

    #[test]
    fn tile_redirects_that_could_leak_are_refused() {
        let keyed = url("https://tiles.example.com/1/2/3.png?key=SECRET");
        let reason = tile_redirect_refusal(&keyed, &url("https://other.example.net/x?y")).unwrap();
        assert!(reason.contains("https://other.example.net"));
        assert!(!reason.contains("SECRET") && !reason.contains("x?y"));
        let https = url("https://tiles.example.com/1/2/3.png");
        assert!(
            tile_redirect_refusal(&https, &url("http://tiles.example.com/1/2/3.png")).is_some()
        );
        assert!(tile_redirect_refusal(&https, &url("http://169.254.169.254/latest")).is_some());
        assert!(tile_redirect_refusal(&https, &url("https://[fe80::1]/x")).is_some());
        assert!(tile_redirect_refusal(&https, &url("file:///etc/passwd")).is_some());
        assert!(tile_redirect_refusal(&https, &url("https://u:p@tiles.example.com/x")).is_some());
        // A local test server may redirect on the same machine.
        let local = url("http://127.0.0.1:8711/1/2/3.png");
        assert!(tile_redirect_refusal(&local, &url("http://localhost:8712/1/2/3.png")).is_none());
    }

    #[test]
    fn geotiff_redirects_keep_the_configured_url_rules() {
        let local = url("http://127.0.0.1:8743/remote.tif");
        assert!(
            geotiff_redirect_refusal(&local, &url("http://localhost:8744/remote.tif")).is_none()
        );
        assert!(
            geotiff_redirect_refusal(&local, &url("http://172.17.0.4:8744/remote.tif")).is_some()
        );
        assert!(geotiff_redirect_refusal(&local, &url("http://example.com/remote.tif")).is_some());
        assert!(geotiff_redirect_refusal(&local, &url("https://example.com/remote.tif")).is_none());
        let remote = url("https://data.example.com/cog.tif");
        // Signed object-store URLs carry a query string: that is fine for a redirect target.
        assert!(geotiff_redirect_refusal(
            &remote,
            &url("https://bucket.s3.example.com/cog.tif?X-Sig=1")
        )
        .is_none());
        assert!(
            geotiff_redirect_refusal(&remote, &url("http://data.example.com/cog.tif")).is_some()
        );
        assert!(geotiff_redirect_refusal(
            &remote,
            &url("http://169.254.169.254/latest/meta-data/x.tif")
        )
        .is_some());
        assert!(geotiff_redirect_refusal(&remote, &url("https://10.0.0.5/cog.tif")).is_some());
        assert!(
            geotiff_redirect_refusal(&remote, &url("https://[::ffff:127.0.0.1]/cog.tif")).is_some()
        );
    }

    #[test]
    fn internal_addresses() {
        for text in [
            "http://127.0.0.1/",
            "http://localhost/",
            "http://10.1.2.3/",
            "http://172.16.0.1/",
            "http://192.168.1.1/",
            "http://169.254.169.254/",
            "http://100.100.100.200/",
            "http://0.0.0.0/",
            "http://[::1]/",
            "http://[fd00:ec2::254]/",
            "http://[fe80::1]/",
        ] {
            assert!(is_internal(&url(text)), "{text}");
        }
        for text in [
            "https://example.com/",
            "http://8.8.8.8/",
            "http://[2001:db8::1]/",
        ] {
            assert!(!is_internal(&url(text)), "{text}");
        }
    }

    #[test]
    fn internal_ips() {
        for text in [
            "127.0.0.1",
            "127.1.2.3",
            "10.0.0.1",
            "172.31.255.255",
            "192.168.0.1",
            "169.254.169.254",
            "100.64.0.1",
            "100.127.255.255",
            "0.0.0.0",
            "255.255.255.255",
            "224.0.0.1",
            "::1",
            "::",
            "fc00::1",
            "fd12:3456::1",
            "fe80::1",
            "ff02::1",
            "::ffff:127.0.0.1",
            "::ffff:10.1.2.3",
            "::ffff:169.254.169.254",
        ] {
            assert!(is_internal_ip(text.parse().unwrap()), "{text}");
        }
        for text in [
            "8.8.8.8",
            "100.63.255.255",
            "100.128.0.0",
            "172.32.0.1",
            "2001:4860::8888",
            "::ffff:8.8.8.8",
        ] {
            assert!(!is_internal_ip(text.parse().unwrap()), "{text}");
        }
    }

    /// A resolver whose answers come from `table` instead of DNS.
    fn resolver(table: &[(&str, &[&str])]) -> PublicResolver {
        let table: Vec<(String, Vec<IpAddr>)> = table
            .iter()
            .map(|(name, ips)| {
                let ips = ips.iter().map(|ip| ip.parse().unwrap()).collect();
                ((*name).to_owned(), ips)
            })
            .collect();
        let mut resolver = PublicResolver::with_lookup(Arc::new(move |host: &str| {
            table
                .iter()
                .find(|(name, _)| name == host)
                .map(|(_, ips)| ips.clone())
                .ok_or_else(|| std::io::Error::new(std::io::ErrorKind::NotFound, "unknown name"))
        }));
        resolver.exempt.clear(); // whatever proxy the environment names
        resolver
    }

    fn runtime() -> tokio::runtime::Runtime {
        tokio::runtime::Builder::new_current_thread()
            .enable_all()
            .build()
            .unwrap()
    }

    fn lookup(resolver: &PublicResolver, name: &str) -> Result<Vec<SocketAddr>, String> {
        let name: Name = name.parse().unwrap();
        runtime()
            .block_on(resolver.resolve(name))
            .map(Iterator::collect)
            .map_err(|e| e.to_string())
    }

    #[test]
    fn resolver_keeps_public_addresses_only() {
        let resolver = resolver(&[
            ("public.test", &["93.184.216.34", "2606:2800::1"]),
            ("mixed.test", &["127.0.0.1", "93.184.216.34", "fd00::1"]),
            ("loopback.test", &["127.0.0.1", "::1"]),
            ("metadata.test", &["169.254.169.254"]),
            ("private.test", &["10.0.0.7", "::ffff:192.168.1.1"]),
            ("cgnat.test", &["100.100.100.200"]),
        ]);
        assert_eq!(lookup(&resolver, "public.test").unwrap().len(), 2);
        let mixed = lookup(&resolver, "mixed.test").unwrap();
        assert_eq!(mixed, vec!["93.184.216.34:0".parse().unwrap()]);
        for name in [
            "loopback.test",
            "metadata.test",
            "private.test",
            "cgnat.test",
        ] {
            let error = lookup(&resolver, name).unwrap_err();
            assert!(error.contains("private network"), "{name}: {error}");
        }
        assert!(lookup(&resolver, "unknown.test").is_err());
    }

    #[test]
    fn a_name_resolving_to_this_machine_is_never_connected() {
        let listener = std::net::TcpListener::bind("127.0.0.1:0").unwrap();
        listener.set_nonblocking(true).unwrap();
        let port = listener.local_addr().unwrap().port();
        let client = reqwest::Client::builder()
            .dns_resolver(resolver(&[("internal.test", &["127.0.0.1"])]))
            .redirect(geotiff_redirect_policy())
            .build()
            .unwrap();
        let error = runtime()
            .block_on(
                client
                    .get(format!("http://internal.test:{port}/latest/meta-data"))
                    .send(),
            )
            .unwrap_err();
        let reason = address_refusal(&error).unwrap();
        assert!(reason.starts_with("internal.test resolves to"), "{reason}");
        // Refused before connecting: the listener saw nothing.
        assert!(listener.accept().is_err());
    }

    #[test]
    fn a_redirect_to_a_name_resolving_internally_is_never_connected() {
        // Stand-ins on this machine: 127.0.0.1 plays a public tile server, 127.0.0.2
        // the internal service; "internal" means 127.0.0.2 only.
        let Ok(internal_service) = std::net::TcpListener::bind("127.0.0.2:0") else {
            return; // 127.0.0.2 is not configured (macOS): nothing to check here
        };
        internal_service.set_nonblocking(true).unwrap();
        let internal_port = internal_service.local_addr().unwrap().port();
        let public = std::net::TcpListener::bind("127.0.0.1:0").unwrap();
        let public_port = public.local_addr().unwrap().port();
        let server = std::thread::spawn(move || {
            use std::io::{Read, Write};
            let (mut stream, _) = public.accept().unwrap();
            let mut buffer = [0_u8; 1024];
            let _ = stream.read(&mut buffer).unwrap();
            let answer = format!(
                "HTTP/1.1 302 Found\r\nLocation: http://internal.test:{internal_port}/latest/\
                 meta-data\r\nContent-Length: 0\r\nConnection: close\r\n\r\n"
            );
            stream.write_all(answer.as_bytes()).unwrap();
        });
        let mut resolver = resolver(&[
            ("public.test", &["127.0.0.1"]),
            ("internal.test", &["127.0.0.2"]),
        ]);
        resolver.internal = |ip| ip == IpAddr::from([127, 0, 0, 2]);
        let client = reqwest::Client::builder()
            .dns_resolver(resolver)
            .redirect(tile_redirect_policy())
            .build()
            .unwrap();
        let error = runtime()
            .block_on(
                client
                    .get(format!("http://public.test:{public_port}/1/2/3.png"))
                    .send(),
            )
            .unwrap_err();
        server.join().unwrap();
        let reason = address_refusal(&error).unwrap();
        assert!(reason.starts_with("internal.test resolves to"), "{reason}");
        assert!(internal_service.accept().is_err());
    }

    #[test]
    fn a_literal_local_start_may_stay_local() {
        // `MAPCV_ALLOW_LOCAL_URLS` and the MCP switch are off in tests.
        assert!(may_reach_internal(&url("http://127.0.0.1:8000/x.tif")));
        assert!(may_reach_internal(&url("http://localhost:8000/x.tif")));
        assert!(!may_reach_internal(&url("https://tiles.example.com/x.png")));
        assert!(start_refusal(&url("http://127.0.0.1:8000/x.tif")).is_none());
    }
}
