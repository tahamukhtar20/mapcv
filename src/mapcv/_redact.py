"""Credentials out of everything mapcv shows or writes.

One :class:`Redactor` serves every output path: the CLI's messages and its debug log
(:data:`REDACTOR`, taught each config the CLI loads), the MCP server's results and
errors (one per server), and the settings a manifest records (:func:`redact_query`).

What counts as a secret: the user info and query values of any URL in a config, and,
for a tile ``url_template``, also its longer path segments (providers put keys there:
``/v1/<key>/{z}/{x}/{y}``) and every URL on its host.
"""

from __future__ import annotations

import logging
import re
import threading
from typing import Any
from urllib.parse import parse_qsl, urlencode, urlsplit, urlunsplit

__all__ = ["REDACTOR", "RedactingFormatter", "Redactor", "redact_query", "redact_url"]

_TEMPLATE_IN_TEXT = re.compile(r"""https?://[^\s"'<>]*\{[xyz]\}[^\s"'<>]*""")
_URL_IN_TEXT = re.compile(r"""https?://[^\s"'<>)\]]+""")
_MIN_QUERY_SECRET = 3
_MIN_PATH_SECRET = 6
#: Config keys whose value is a tile URL template (path segments may be keys).
_TEMPLATE_KEYS = frozenset({"url_template"})


def redact_url(url: str) -> str:
    """Only the scheme and host of a URL that may embed credentials."""
    parsed = urlsplit(url)
    if not parsed.scheme or not parsed.hostname:
        return url
    return f"{parsed.scheme}://{parsed.hostname}/..."


def redact_query(url: str) -> str:
    """``url`` with user info removed and every query value replaced by ``***``.

    For settings a dataset records (an Overpass URL, say): the address stays readable,
    the key in its query string does not ship with the dataset.
    """
    parts = urlsplit(url)
    if not parts.scheme or not parts.netloc:
        return url
    netloc = parts.hostname or ""
    if ":" in netloc:  # an IPv6 address
        netloc = f"[{netloc}]"
    if parts.port is not None:
        netloc += f":{parts.port}"
    pairs = parse_qsl(parts.query, keep_blank_values=True)
    query = urlencode([(name, "***") for name, _ in pairs], safe="*")
    return urlunsplit((parts.scheme, netloc, parts.path, query, ""))


def _secret_tokens(url: str, template: bool) -> set[str]:
    """The parts of a URL that may be credentials."""
    parts = urlsplit(url)
    tokens: set[str] = set()
    for value in (parts.username, parts.password, parts.fragment):
        if value:
            tokens.add(value)
    for pair in parts.query.split("&"):
        _, _, value = pair.partition("=")
        if len(value) >= _MIN_QUERY_SECRET and "{" not in value:
            tokens.add(value)
    if template:
        for segment in parts.path.split("/"):
            if len(segment) >= _MIN_PATH_SECRET and "{" not in segment:
                tokens.add(segment)
    return tokens


class Redactor:
    """Removes the credentials of every URL it has been taught from text.

    Output is built from fields that never carry a credential; this is the second
    line of defence for text mapcv does not control: error messages, warnings and
    tracebacks of the libraries it calls.
    """

    def __init__(self) -> None:
        self._templates: set[str] = set()
        self._hosts: set[str] = set()
        self._tokens: set[str] = set()
        self._lock = threading.Lock()

    def learn_url(self, url: str, template: bool = True) -> None:
        """Remember a URL so its secrets are hidden from later output.

        A tile ``template`` is hidden whole, with every URL on its host; another URL
        (an Overpass endpoint, a GeoTIFF) only loses its user info and query values.
        """
        if not url or "://" not in url:
            return
        parts = urlsplit(url)
        tokens = _secret_tokens(url, template)
        with self._lock:
            if template:
                self._templates.add(url)
                if parts.hostname:
                    self._hosts.add(parts.hostname)
            self._tokens |= tokens

    def learn_text(self, text: str) -> None:
        """Remember every tile URL template written in a piece of config text."""
        for match in _TEMPLATE_IN_TEXT.finditer(text):
            self.learn_url(match.group(0).rstrip(",;"))

    def learn_data(self, data: Any, key: str = "") -> None:
        """Remember the URLs of a parsed (maybe invalid) config: every string value
        that holds one, at any depth."""
        if isinstance(data, dict):
            for name, value in data.items():
                self.learn_data(value, str(name))
        elif isinstance(data, (list, tuple)):
            for value in data:
                self.learn_data(value, key)
        elif isinstance(data, str) and "://" in data:
            self.learn_url(data, template=key in _TEMPLATE_KEYS)

    def scrub(self, text: str) -> str:
        """``text`` without known templates, their secrets, or credentials in URLs."""
        with self._lock:
            templates = sorted(self._templates, key=len, reverse=True)
            hosts = set(self._hosts)
            tokens = sorted(self._tokens, key=len, reverse=True)
        for template in templates:
            text = text.replace(template, redact_url(template))

        def url(match: re.Match[str]) -> str:
            found = match.group(0)
            parts = urlsplit(found)
            if parts.hostname in hosts or parts.username or parts.password or parts.query:
                return redact_url(found)
            return found

        text = _URL_IN_TEXT.sub(url, text)
        for token in tokens:
            text = text.replace(token, "***")
        return text

    def scrub_data(self, data: Any) -> Any:
        """:meth:`scrub` applied to every string inside nested lists and dicts."""
        if isinstance(data, str):
            return self.scrub(data)
        if isinstance(data, dict):
            return {key: self.scrub_data(value) for key, value in data.items()}
        if isinstance(data, (list, tuple)):
            return [self.scrub_data(value) for value in data]
        return data


#: The process's redactor: the CLI teaches it each config it loads, and its debug
#: log passes through it.
REDACTOR = Redactor()


class RedactingFormatter(logging.Formatter):
    """A formatter whose output (message and traceback) went through a redactor."""

    def __init__(self, fmt: str, redactor: Redactor = REDACTOR) -> None:
        super().__init__(fmt)
        self._redactor = redactor

    def format(self, record: logging.LogRecord) -> str:
        return self._redactor.scrub(super().format(record))
