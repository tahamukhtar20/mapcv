"""On-disk cache of XYZ tiles, so re-runs, resumes and parameter sweeps don't download again.

Tiles are kept under ``<cache dir>/tiles/<key>/<z>/<x>/<y>.tile``, where ``<key>`` is a
hash of the URL template: the template (which may hold an API key) is never written
to disk. Each file is a short header (format, expiry time, payload length) and the
tile's bytes as the server sent them. Only tiles the server sent are cached, never
black fills of failed tiles.

A tile stays fresh as long as the server's caching headers allow (RFC 9111):
``Cache-Control: max-age`` less ``Age``, else ``Expires`` less ``Date``. A response
marked ``no-store`` or ``no-cache`` is not kept (mapcv does not revalidate), and one
without any of these headers is kept for :data:`DEFAULT_TTL` (7 days, the minimum the
OpenStreetMap tile usage policy asks clients to cache for). Expired tiles are
downloaded again and replaced.

The cache directory is ``$MAPCV_CACHE_DIR`` when set, else the platform's user cache
folder: ``$XDG_CACHE_HOME/mapcv`` or ``~/.cache/mapcv`` on Linux,
``~/Library/Caches/mapcv`` on macOS and ``%LOCALAPPDATA%\\mapcv\\Cache`` on Windows.
"""

from __future__ import annotations

import hashlib
import math
import os
import struct
import sys
import time
import warnings
from collections.abc import Callable, Iterator
from dataclasses import dataclass
from email.utils import parsedate_to_datetime
from pathlib import Path

CACHE_ENV = "MAPCV_CACHE_DIR"
DEFAULT_TTL = 7 * 24 * 3600.0
#: The longest a tile is kept, whatever the server's headers ask for (RFC 9111 allows a
#: cache to keep a response for less than it is fresh).
MAX_TTL = 365 * 24 * 3600.0
_MAGIC = b"mapcv-tile\x01"
_HEADER = struct.Struct(">dQ")  # expiry (Unix seconds), payload length
_SUFFIX = ".tile"

# ``(cache_control, expires, date, age)`` of a tile response, each ``None`` when absent.
CacheHeaders = tuple[str | None, str | None, str | None, str | None]


def cache_dir() -> Path:
    """The folder mapcv caches in (see the module docs)."""
    configured = os.environ.get(CACHE_ENV)
    if configured:
        return Path(configured).expanduser()
    home = Path.home()
    if sys.platform == "win32":
        base = os.environ.get("LOCALAPPDATA")
        return (Path(base) if base else home / "AppData" / "Local") / "mapcv" / "Cache"
    if sys.platform == "darwin":
        return home / "Library" / "Caches" / "mapcv"
    xdg = os.environ.get("XDG_CACHE_HOME")
    return (Path(xdg) if xdg else home / ".cache") / "mapcv"


def tiles_dir() -> Path:
    return cache_dir() / "tiles"


def _http_time(value: str | None) -> float | None:
    if not value:
        return None
    try:
        parsed = parsedate_to_datetime(value)
        if parsed.tzinfo is None:  # an HTTP date is always GMT
            return None
        return parsed.timestamp()
    except (TypeError, ValueError, IndexError, OverflowError, OSError):
        return None


def freshness_lifetime(headers: CacheHeaders, now: float) -> float | None:
    """Seconds a response stays fresh from ``now``; ``None`` if it must not be cached.

    Without ``Cache-Control: max-age`` or ``Expires``, :data:`DEFAULT_TTL`.
    """
    cache_control, expires, date, age = headers
    directives = {}
    for part in (cache_control or "").split(","):
        name, _, value = part.strip().partition("=")
        if name:
            directives[name.strip().lower()] = value.strip().strip('"')
    if "no-store" in directives or "no-cache" in directives:
        return None
    try:
        current_age = max(0.0, float(age)) if age else 0.0
    except (ValueError, OverflowError):
        current_age = 0.0
    if math.isnan(current_age):
        current_age = 0.0
    lifetime: float | None = None
    if "max-age" in directives:
        try:
            lifetime = float(int(directives["max-age"]))
        except ValueError:
            lifetime = 0.0  # an invalid max-age makes the response stale (RFC 9111 4.2.1)
        except OverflowError:
            lifetime = MAX_TTL  # too large for a float: as long as a cache may keep it
    elif expires is not None:
        expiry = _http_time(expires)
        # An invalid Expires (such as "0") means already expired.
        lifetime = 0.0 if expiry is None else expiry - (_http_time(date) or now)
    if lifetime is None:
        return DEFAULT_TTL
    remaining = min(lifetime - current_age, MAX_TTL)
    return remaining if remaining > 0 else None


@dataclass
class CacheUsage:
    """What :func:`usage` found in the cache."""

    path: Path
    tiles: int = 0
    expired: int = 0
    bytes: int = 0


def _tile_files(root: Path) -> Iterator[Path]:
    if root.is_dir():
        yield from (path for path in root.rglob(f"*{_SUFFIX}") if path.is_file())


def _expiry(path: Path) -> float | None:
    try:
        with path.open("rb") as handle:
            head = handle.read(len(_MAGIC) + _HEADER.size)
    except OSError:  # pragma: no cover - removed or unreadable since it was listed
        return None
    if len(head) != len(_MAGIC) + _HEADER.size or not head.startswith(_MAGIC):
        return None
    expiry, _ = _HEADER.unpack(head[len(_MAGIC) :])
    return float(expiry)


def usage(now: float | None = None) -> CacheUsage:
    """Count the cached tiles, the expired ones among them and their size on disk."""
    now = time.time() if now is None else now
    found = CacheUsage(tiles_dir())
    for path in _tile_files(found.path):
        found.tiles += 1
        try:
            found.bytes += path.stat().st_size
        except OSError:  # pragma: no cover - removed since it was listed
            pass
        expiry = _expiry(path)
        if expiry is None or expiry <= now:
            found.expired += 1
    return found


def clear(expired_only: bool = False, now: float | None = None) -> int:
    """Delete cached tiles (only the expired ones with ``expired_only``); returns how many."""
    now = time.time() if now is None else now
    root = tiles_dir()
    removed = 0
    for path in list(_tile_files(root)):
        if expired_only:
            expiry = _expiry(path)
            if expiry is not None and expiry > now:
                continue
        try:
            path.unlink()
            removed += 1
        except FileNotFoundError:  # pragma: no cover - removed by another process
            pass
    # Drop the folders the deletion emptied, deepest first; leave anything else.
    if root.is_dir():
        for folder in sorted((p for p in root.rglob("*") if p.is_dir()), reverse=True):
            try:
                folder.rmdir()
            except OSError:
                pass
    return removed


class TileCache:
    """The cached tiles of one URL template."""

    def __init__(
        self,
        url_template: str,
        root: Path | None = None,
        clock: Callable[[], float] | None = None,
    ) -> None:
        key = hashlib.sha256(url_template.encode("utf-8")).hexdigest()[:24]
        self.folder = (root if root is not None else tiles_dir()) / key
        self._clock = clock if clock is not None else lambda: time.time()
        self._broken = False

    def _path(self, x: int, y: int, z: int) -> Path:
        return self.folder / str(z) / str(x) / f"{y}{_SUFFIX}"

    def get(self, x: int, y: int, z: int) -> bytes | None:
        """The tile's bytes if it is cached and still fresh, else ``None``."""
        try:
            data = self._path(x, y, z).read_bytes()
        except OSError:
            return None
        start = len(_MAGIC) + _HEADER.size
        if len(data) < start or not data.startswith(_MAGIC):
            return None
        expiry, length = _HEADER.unpack(data[len(_MAGIC) : start])
        if expiry <= self._clock() or len(data) - start != length:
            return None
        return data[start:]

    def discard(self, x: int, y: int, z: int) -> bool:
        """Remove a tile from the cache (one that turned out not to be a usable image);
        ``True`` if it was cached."""
        try:
            self._path(x, y, z).unlink()
        except OSError:
            return False
        return True

    def put(self, x: int, y: int, z: int, payload: bytes, headers: CacheHeaders) -> bool:
        """Cache a tile the server sent, unless its headers forbid it; ``True`` if kept.

        A cache that can't be written (no space, no permission) is switched off for the
        rest of the run with one warning; the run itself goes on.
        """
        if self._broken:
            return False
        now = self._clock()
        lifetime = freshness_lifetime(headers, now)
        if lifetime is None:
            return False
        path = self._path(x, y, z)
        temp = path.with_name(f".{path.name}.{os.getpid()}.tmp")
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            temp.write_bytes(_MAGIC + _HEADER.pack(now + lifetime, len(payload)) + payload)
            os.replace(temp, path)
        except OSError as exc:
            self._broken = True
            try:
                temp.unlink()
            except OSError:
                pass
            warnings.warn(
                f"The tile cache in {self.folder} can't be written ({exc}); tiles are "
                f"downloaded without caching. Set {CACHE_ENV} to another folder or "
                "imagery.cache: false.",
                UserWarning,
                stacklevel=2,
            )
            return False
        return True
