"""Where mapcv's Python HTTP requests may connect: the rules of ``src/http_policy.rs``.

The host a URL in the config names is the user's choice: mapcv connects to it (that
host name and port) wherever it resolves, a docker-compose service or an on-prem server
included. Any *other* host, reached through a redirect or named by a STAC catalog, must
not be on this machine or a private network: a name is judged by the addresses it
resolves to *when connecting* (every address, IPv4 and IPv6), so a DNS answer that
changes between a check and the connection does not help. A request that starts at a
URL naming such an address literally (an IP address or ``localhost``: a local test
server) may go on to others.

Inside :func:`public_addresses_only` (the MCP server, unless started with
``--allow-local-urls``) nothing is trusted: every request, the Rust tile fetcher's and
GeoTIFF reader's included, reaches public addresses only.

:func:`urlopen` is ``urllib.request.urlopen`` with these rules; :func:`fsspec_options`
gives the same rules to fsspec's HTTP file system (aiohttp).
"""

from __future__ import annotations

import errno
import ipaddress
import socket
import threading
import urllib.error
import urllib.request
from collections.abc import Iterator
from contextlib import contextmanager
from functools import partial
from http.client import HTTPConnection, HTTPResponse, HTTPSConnection
from typing import Any
from urllib.parse import urlsplit

__all__ = [
    "AddressRefused",
    "fsspec_options",
    "is_internal_host",
    "is_internal_ip",
    "linked_url_refusal",
    "may_reach_internal",
    "public_addresses_only",
    "public_only",
    "redirect_refusal",
    "start_refusal",
    "trusted_endpoints",
    "urlopen",
]

#: ``(host, port)`` pairs a request connects to wherever they resolve.
Trusted = frozenset[tuple[str, int]]
_MAX_REDIRECTS = 10
_CGNAT = ipaddress.ip_network("100.64.0.0/10")
_UNIQUE_LOCAL = ipaddress.ip_network("fc00::/7")
_LINK_LOCAL_V6 = ipaddress.ip_network("fe80::/10")
_PRIVATE_V4 = tuple(
    ipaddress.ip_network(net) for net in ("10.0.0.0/8", "172.16.0.0/12", "192.168.0.0/16")
)

_DEFAULT_TIMEOUT: Any = socket._GLOBAL_DEFAULT_TIMEOUT  # type: ignore[attr-defined]

_lock = threading.Lock()
_public_only_depth = 0


class AddressRefused(OSError):
    """A request mapcv does not make: its target is on this machine or a private network,
    or a redirect breaks the rules. The message names the host only."""

    def __init__(self, message: str) -> None:
        # As (errno, strerror), so libraries that show only strerror (aiohttp) show it.
        super().__init__(errno.EACCES, message)

    def __str__(self) -> str:
        return str(self.strerror)


def is_internal_ip(ip: ipaddress.IPv4Address | ipaddress.IPv6Address) -> bool:
    """Loopback, private (RFC 1918, ``fc00::/7``), link-local, carrier-grade NAT,
    unspecified, broadcast or multicast; IPv4-mapped IPv6 addresses by their IPv4 address."""
    if isinstance(ip, ipaddress.IPv6Address):
        if ip.ipv4_mapped is not None:
            return is_internal_ip(ip.ipv4_mapped)
        return (
            ip.is_loopback
            or ip.is_unspecified
            or ip.is_multicast
            or ip in _UNIQUE_LOCAL
            or ip in _LINK_LOCAL_V6
        )
    return (
        ip.is_loopback
        or any(ip in net for net in _PRIVATE_V4)
        or ip.is_link_local
        or ip.is_unspecified
        or ip == ipaddress.IPv4Address("255.255.255.255")
        or ip.is_multicast
        or ip in _CGNAT
    )


def _ip(host: str) -> ipaddress.IPv4Address | ipaddress.IPv6Address | None:
    try:
        return ipaddress.ip_address(host.strip("[]").partition("%")[0])
    except ValueError:
        return None


def is_internal_host(host: str | None) -> bool:
    """Whether a URL's host *text* names this machine or a private network: such an IP
    address, ``localhost`` or a name under ``.localhost``. Other names are judged by
    the addresses they resolve to."""
    if not host:
        return True
    ip = _ip(host)
    if ip is not None:
        return is_internal_ip(ip)
    name = host.rstrip(".").lower()
    return name == "localhost" or name.endswith(".localhost")


def public_only() -> bool:
    """Whether requests are limited to public addresses (inside :func:`public_addresses_only`)."""
    return _public_only_depth > 0


@contextmanager
def public_addresses_only() -> Iterator[None]:
    """Limit every request of this process to public addresses while the block runs.

    Blocks may nest and overlap across threads; the limit holds while any is open.
    """
    global _public_only_depth
    from mapcv._mapcv_rs import set_public_only

    with _lock:
        _public_only_depth += 1
        set_public_only(True)
    try:
        yield
    finally:
        with _lock:
            _public_only_depth -= 1
            if _public_only_depth == 0:
                set_public_only(False)


def may_reach_internal(start: str) -> bool:
    """Whether a request that started at ``start`` may reach this machine or a private
    network anywhere: when ``start`` names one literally and requests are not limited to
    public addresses."""
    return not public_only() and is_internal_host(urlsplit(start).hostname)


def trusted_endpoints(start: str) -> Trusted:
    """The host name and port of ``start``, which a request connects to wherever they
    resolve: the user wrote them. Also port 443 of a plain-http URL on port 80, for the
    usual upgrade to https. Nothing while requests are limited to public addresses."""
    if public_only():
        return frozenset()
    parts = urlsplit(start)
    host = (parts.hostname or "").rstrip(".").lower()
    try:
        port = parts.port or (443 if parts.scheme == "https" else 80)
    except ValueError:
        return frozenset()
    if not host:
        return frozenset()
    found = {(host, port)}
    if parts.scheme == "http" and port == 80:
        found.add((host, 443))
    return frozenset(found)


def _is_trusted(trusted: Trusted, host: str, port: int) -> bool:
    return (host.rstrip(".").lower(), int(port)) in trusted


def _origin(url: str) -> str:
    parts = urlsplit(url)
    return f"{parts.scheme}://{parts.netloc.rpartition('@')[2]}"


def _internal_refusal(host: str) -> str:
    """Why ``host`` may not be reached; the same text whatever listens there."""
    if public_only():
        return (
            f"{host} is on this machine or a private network, and this mapcv MCP server "
            "connects to public addresses only (start it with --allow-local-urls to use a "
            "local server)"
        )
    return (
        f"{host} resolves to an address on this machine or a private network; mapcv "
        "connects there only to the host a URL in the config names, not to another host "
        "reached by a redirect or named by a catalog"
    )


def start_refusal(url: str) -> str | None:
    """Why a request may not start at ``url`` (only when limited to public addresses)."""
    if public_only() and is_internal_host(urlsplit(url).hostname):
        return _internal_refusal(_origin(url))
    return None


def linked_url_refusal(source: str, url: str) -> str | None:
    """Why a URL that ``source`` (a STAC catalog) links to must not be opened: it names
    this machine or a private network and ``source`` may not lead there."""
    parts = urlsplit(url)
    if (
        parts.scheme not in ("http", "https")
        or not is_internal_host(parts.hostname)
        or may_reach_internal(source)
    ):
        return None
    if public_only():
        return _internal_refusal(_origin(url))
    return (
        f"it points to {_origin(url)}, an address on this machine or a private network, "
        "which mapcv does not open for a catalog on a public host"
    )


def redirect_refusal(start: str, current: str, target: str) -> str | None:
    """Why a redirect from ``current`` to ``target`` must not be followed, for a request
    that started at ``start``; ``None`` to follow it."""
    parts = urlsplit(target)
    if parts.scheme not in ("http", "https"):
        return (
            f"the server redirected to a {parts.scheme or 'relative'} URL; only http(s) is followed"
        )
    if parts.username or parts.password:
        return f"the server redirected to {_origin(target)} with credentials in the URL"
    if urlsplit(current).scheme == "https" and parts.scheme == "http":
        return f"the server redirected from https to plain http ({_origin(target)})"
    if is_internal_host(parts.hostname) and not may_reach_internal(start):
        if public_only():
            return _internal_refusal(_origin(target))
        return (
            f"the server redirected to {_origin(target)}, an address on this machine or a "
            "private network, which mapcv does not follow from a public server"
        )
    return None


def _public(address: str) -> bool:
    ip = _ip(str(address))
    return ip is not None and not is_internal_ip(ip)


def _connect(
    address: tuple[str, int],
    timeout: Any = _DEFAULT_TIMEOUT,
    source_address: tuple[str, int] | None = None,
    *,
    filtered: bool,
    trusted: Trusted = frozenset(),
) -> socket.socket:
    """``socket.create_connection``, connecting only to public addresses when ``filtered``,
    except to a ``trusted`` host and port."""
    host, port = address
    if not filtered or _is_trusted(trusted, host, port):
        return socket.create_connection(address, timeout, source_address)
    kept = [
        info
        for info in socket.getaddrinfo(host, port, 0, socket.SOCK_STREAM)
        if _public(str(info[4][0]))
    ]
    if not kept:
        raise AddressRefused(_internal_refusal(host))
    error: OSError | None = None
    for family, kind, proto, _, sockaddr in kept:
        sock = socket.socket(family, kind, proto)
        try:
            if timeout is not _DEFAULT_TIMEOUT:
                sock.settimeout(timeout)
            if source_address:
                sock.bind(source_address)
            sock.connect(sockaddr)
        except OSError as exc:
            sock.close()
            error = exc
            continue
        return sock
    assert error is not None
    raise error


class _Connection(HTTPConnection):
    def __init__(self, *args: Any, filtered: bool, trusted: Trusted, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self._create_connection = partial(_connect, filtered=filtered, trusted=trusted)


class _SecureConnection(HTTPSConnection):
    def __init__(self, *args: Any, filtered: bool, trusted: Trusted, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self._create_connection = partial(_connect, filtered=filtered, trusted=trusted)


def _through_proxy(request: urllib.request.Request) -> bool:
    return request.has_proxy() or bool(getattr(request, "_tunnel_host", None))


class _HTTPHandler(urllib.request.HTTPHandler):
    def __init__(self, filtered: bool, trusted: Trusted) -> None:
        super().__init__()
        self._filtered = filtered
        self._trusted = trusted

    def http_open(self, req: urllib.request.Request) -> HTTPResponse:
        # Through a proxy, the proxy resolves the target; the proxy itself is the operator's.
        filtered = self._filtered and not _through_proxy(req)
        return self.do_open(partial(_Connection, filtered=filtered, trusted=self._trusted), req)


class _HTTPSHandler(urllib.request.HTTPSHandler):
    def __init__(self, filtered: bool, trusted: Trusted) -> None:
        super().__init__()
        self._filtered = filtered
        self._trusted = trusted

    def https_open(self, req: urllib.request.Request) -> HTTPResponse:
        filtered = self._filtered and not _through_proxy(req)
        context = getattr(self, "_context", None)
        connection = partial(_SecureConnection, filtered=filtered, trusted=self._trusted)
        return self.do_open(connection, req, context=context)


class _RedirectHandler(urllib.request.HTTPRedirectHandler):
    max_redirections = _MAX_REDIRECTS

    def __init__(self, start: str) -> None:
        super().__init__()
        self._start = start

    def redirect_request(
        self,
        req: urllib.request.Request,
        fp: Any,
        code: int,
        msg: str,
        headers: Any,
        newurl: str,
    ) -> urllib.request.Request | None:
        reason = redirect_refusal(self._start, req.full_url, newurl)
        if reason is not None:
            raise AddressRefused(reason)
        return super().redirect_request(req, fp, code, msg, headers, newurl)


def urlopen(
    request: str | urllib.request.Request, timeout: float, *, trust_host: bool = True
) -> Any:
    """``urllib.request.urlopen`` for http(s) with the connection rules of this module.

    ``trust_host=False`` for a URL the user did not write (one a catalog links to): its
    host is then filtered like any other.

    Raises:
        AddressRefused: The URL, a redirect or the addresses of a name break the rules.
        OSError: As ``urllib.request.urlopen`` (``URLError``, ``HTTPError``).
    """
    url = request.full_url if isinstance(request, urllib.request.Request) else request
    reason = start_refusal(url)
    if reason is not None:
        raise AddressRefused(reason)
    filtered = not may_reach_internal(url)
    trusted = trusted_endpoints(url) if trust_host else frozenset()
    opener = urllib.request.OpenerDirector()
    for handler in (
        urllib.request.ProxyHandler(),
        urllib.request.UnknownHandler(),
        _HTTPHandler(filtered, trusted),
        _HTTPSHandler(filtered, trusted),
        urllib.request.HTTPDefaultErrorHandler(),
        _RedirectHandler(url),
        urllib.request.HTTPErrorProcessor(),
    ):
        opener.add_handler(handler)
    try:
        return opener.open(request, timeout=timeout)
    except urllib.error.URLError as exc:
        if isinstance(exc.reason, AddressRefused):
            raise exc.reason from None
        raise


# ── fsspec (aiohttp) ─────────────────────────────────────────────────────────


def fsspec_options(url: str, *, trust_host: bool = True) -> dict[str, Any]:
    """Options for fsspec's HTTP file system that apply these rules to ``url`` and
    every request it leads to, or ``{}`` when the request may reach internal addresses.
    ``trust_host`` as for :func:`urlopen`.

    Raises:
        AddressRefused: ``url`` itself is refused.
    """
    reason = start_refusal(url)
    if reason is not None:
        raise AddressRefused(reason)
    if may_reach_internal(url):
        return {}
    trusted = trusted_endpoints(url) if trust_host else frozenset()
    return {"get_client": partial(_public_client, trusted=trusted)}


#: Stands for the public URL a session of :func:`_public_client` started at.
_PUBLIC_START = "https://public.invalid/"


async def _public_client(*, trusted: Trusted = frozenset(), **kwargs: Any) -> Any:
    """An aiohttp session that connects to public addresses only, and to the ``trusted``
    host and port (for ``get_client``).

    Names go through a resolver that keeps public addresses; redirects (literal IP
    addresses included, which aiohttp does not resolve) through :func:`redirect_refusal`.
    """
    import aiohttp
    from yarl import URL

    async def on_redirect(
        session: Any, context: Any, params: aiohttp.TraceRequestRedirectParams
    ) -> None:
        location = params.response.headers.get("Location", "")
        target = str(params.url.join(URL(location)))
        reason = redirect_refusal(_PUBLIC_START, str(params.url), target)
        if reason is not None:
            raise AddressRefused(reason)

    trace = aiohttp.TraceConfig()
    trace.on_request_redirect.append(on_redirect)
    kwargs.pop("connector", None)
    kwargs.pop("trust_env", None)  # a proxy would resolve names itself
    connector = aiohttp.TCPConnector(resolver=_public_resolver(trusted))
    return aiohttp.ClientSession(connector=connector, trace_configs=[trace], **kwargs)


def _public_resolver(trusted: Trusted) -> Any:
    """An aiohttp resolver that keeps public addresses only, except for ``trusted``."""
    import aiohttp

    class PublicResolver(aiohttp.ThreadedResolver):
        async def resolve(
            self, host: str, port: int = 0, family: socket.AddressFamily = socket.AF_INET
        ) -> list[Any]:
            found = await super().resolve(host, port, family)
            if _is_trusted(trusted, host, port):
                return list(found)
            kept = [entry for entry in found if _public(entry["host"])]
            if not kept:
                raise AddressRefused(_internal_refusal(host))
            return kept

    return PublicResolver()
