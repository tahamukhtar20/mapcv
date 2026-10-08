//! Which HTTP redirects mapcv follows.
//!
//! A tile URL template or a remote GeoTIFF URL is checked when the config is
//! validated, but the server it names decides where a redirect goes. These
//! policies check every hop again, so a redirect cannot take a request (and,
//! for a tile template, its key) somewhere the configured URL could not go.
//! The clients that use them also never send a `Referer` header, which would
//! carry the previous URL, query string included, to the next host.

use reqwest::redirect::{Attempt, Policy};
use reqwest::Url;
use std::net::{IpAddr, Ipv4Addr, Ipv6Addr};

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
        || (first & 0xFE00) == 0xFC00 // unique local, fc00::/7
        || (first & 0xFFC0) == 0xFE80 // link local, fe80::/10
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
    if is_internal(to) && !is_internal(from) {
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
}
