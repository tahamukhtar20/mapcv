"""Reading a Zarr store someone else made without running code from it.

Zarr 2 decodes an array through the codecs its own metadata names, and one of them
(numcodecs ``pickle``, for object arrays) unpickles the stored bytes, which runs code.
So a downloaded store is checked before any of its arrays is opened:

* :func:`check_group` reads the metadata of every array of a group (as JSON, without
  opening the array) and refuses the store unless each array holds numbers (bool, int,
  uint, float) and names only plain compressors and filters: what
  ``mapcv export -f zarr`` writes (Blosc), and the other common numeric codecs.
* :func:`pickle_codec_refused` keeps the ``pickle`` codec from being created while a
  block runs, for stores opened by other libraries (an EOPF product through xarray),
  whose arrays mapcv does not choose.
"""

from __future__ import annotations

import json
import threading
from collections.abc import Iterator
from contextlib import contextmanager
from typing import Any

import numpy as np

__all__ = ["SAFE_CODECS", "array_problem", "check_group", "pickle_codec_refused"]

#: Compressors and filters that only transform numbers or bytes.
SAFE_CODECS = frozenset({"blosc", "zstd", "zlib", "gzip", "lz4", "bz2", "lzma", "delta", "shuffle"})
_NUMERIC_KINDS = frozenset("biuf")
#: Arrays and groups visited at most; a mapcv export has a handful.
_MAX_NODES = 10_000


def array_problem(meta: Any) -> str | None:
    """What makes a Zarr 2 array's metadata (``.zarray``) unsafe to read, or ``None``."""
    if not isinstance(meta, dict):
        return "has metadata that is not a JSON object"
    dtype = meta.get("dtype")
    if not isinstance(dtype, str):
        return "has a structured data type"
    try:
        kind = np.dtype(dtype).kind
    except TypeError:
        return f"has an unknown data type {dtype[:40]!r}"
    if kind not in _NUMERIC_KINDS:
        return f"holds {dtype[:40]!r} values, not numbers"
    filters = meta.get("filters")
    if filters is not None and not isinstance(filters, list):
        return "has filters that are not a list"
    for codec in [meta.get("compressor"), *(filters or [])]:
        if codec is None:
            continue
        name = codec.get("id") if isinstance(codec, dict) else None
        if name not in SAFE_CODECS:
            shown = name[:40] if isinstance(name, str) else "?"
            return f"is encoded with the codec {shown!r}"
    return None


def check_group(group: Any, where: str) -> None:
    """Refuse a Zarr 2 group unless all its arrays are safe to read (:func:`array_problem`).

    Only metadata is read; no array is opened.

    Raises:
        ValueError: An array is not numeric or names another codec.
    """
    store = group.store
    pending = [group]
    visited = 0
    while pending:
        current = pending.pop()
        prefix = f"{current.path}/" if current.path else ""
        for name in current.array_keys():
            visited += 1
            path = prefix + name
            try:
                meta = json.loads(store[f"{path}/.zarray"])
            except (KeyError, ValueError, UnicodeDecodeError):
                problem: str | None = "has metadata that cannot be read"
            else:
                problem = array_problem(meta)
            if problem is not None:
                raise ValueError(
                    f"{where}: the Zarr array '{path}' {problem}. mapcv reads Zarr arrays of "
                    "numbers (bool, int, uint, float) encoded with "
                    f"{', '.join(sorted(SAFE_CODECS))} only, as `mapcv export -f zarr` writes "
                    "them, so it did not open this store"
                )
        for name in current.group_keys():
            visited += 1
            pending.append(current[name])
        if visited > _MAX_NODES:
            raise ValueError(
                f"{where}: the Zarr store has more than {_MAX_NODES} arrays and groups; a mapcv "
                "export has a few, so it did not open this store"
            )


class _RefusedPickle:
    """Stands in for numcodecs' ``pickle`` codec: creating it fails."""

    codec_id = "pickle"

    @classmethod
    def from_config(cls, config: Any) -> Any:
        raise ValueError(
            "this Zarr store encodes data with the 'pickle' codec, which can run code when "
            "it is read; mapcv does not read it"
        )


_lock = threading.Lock()
_depth = 0
_saved: Any = None


@contextmanager
def pickle_codec_refused() -> Iterator[None]:
    """Make numcodecs refuse to create the ``pickle`` codec while the block runs.

    Zarr creates an array's codecs when it opens the array, so a store opened inside the
    block cannot decode anything with ``pickle`` later either. Blocks may nest and
    overlap across threads.
    """
    global _depth, _saved
    from numcodecs.registry import codec_registry

    with _lock:
        if _depth == 0:
            _saved = codec_registry.get("pickle")
            codec_registry["pickle"] = _RefusedPickle
        _depth += 1
    try:
        yield
    finally:
        with _lock:
            _depth -= 1
            if _depth == 0:
                if _saved is None:
                    codec_registry.pop("pickle", None)
                else:
                    codec_registry["pickle"] = _saved
                _saved = None
