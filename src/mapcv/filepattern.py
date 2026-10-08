"""Local file patterns for GeoTIFF mosaics (``imagery.path: survey/**/*.tif``).

A path that exists on disk is read as itself, even when its name holds pattern
characters (``img[12].tif``, ``survey [2024]/ortho.tif``). Only the parts of a path that
do not exist literally are patterns: ``*``, ``?``, ``[...]`` within one name and ``**``
for any number of folders.

``**`` does not follow symbolic links to folders (so a link back to a parent folder
cannot loop), and a file reached through several links is listed once.
"""

from __future__ import annotations

import fnmatch
import glob
import os
from collections.abc import Iterator
from pathlib import Path


def _components(path: Path) -> list[tuple[str, bool]]:
    """``(name, is_pattern)`` for each part of ``path``.

    A part with pattern characters is literal when it exists, on the way to it, as
    written; from the first part that is a real pattern on, the rest cannot be looked up
    and counts as a pattern if it has pattern characters.
    """
    parts = path.parts
    if not parts:
        return []
    found: list[tuple[str, bool]] = [(parts[0], False)]
    prefix: Path | None = Path(parts[0])
    for part in parts[1:]:
        if not glob.has_magic(part):
            found.append((part, False))
            prefix = prefix / part if prefix is not None else None
        elif part != "**" and prefix is not None and (prefix / part).exists():
            found.append((part, False))
            prefix = prefix / part
        else:
            found.append((part, True))
            prefix = None
    return found


def is_pattern(path: Path) -> bool:
    """Whether ``path`` has a part that must be matched against names on disk."""
    return any(magic for _, magic in _components(path))


def _is_hidden(name: str) -> bool:
    return name.startswith(".")


def _folders(base: str) -> Iterator[str]:
    """``base`` and the folders below it, without following links to folders."""
    for root, names, _ in os.walk(base, followlinks=False):
        names[:] = sorted(name for name in names if not _is_hidden(name))
        yield root


def _expand(base: str, parts: list[tuple[str, bool]]) -> Iterator[str]:
    if not parts:
        yield base
        return
    (name, magic), rest = parts[0], parts[1:]
    if not magic:
        yield from _expand(os.path.join(base, name), rest)
    elif name == "**":
        for folder in _folders(base):
            if rest:
                yield from _expand(folder, rest)
            else:
                try:
                    entries = sorted(os.listdir(folder))
                except OSError:
                    continue
                yield from (
                    os.path.join(folder, entry) for entry in entries if not _is_hidden(entry)
                )
    else:
        try:
            entries = sorted(os.listdir(base))
        except OSError:
            return
        for entry in entries:
            if _is_hidden(entry) and not _is_hidden(name):
                continue
            if fnmatch.fnmatch(entry, name):
                yield from _expand(os.path.join(base, entry), rest)


def matching_files(path: Path) -> list[str]:
    """The files ``path`` names, sorted: itself when it exists as written, else the files
    its pattern parts match. One entry per real file."""
    parts = _components(path)
    if not parts:
        return []
    seen: set[str] = set()
    matches: list[str] = []
    for found in sorted(set(_expand(parts[0][0], parts[1:]))):
        if not os.path.isfile(found):
            continue
        real = os.path.realpath(found)
        if real not in seen:
            seen.add(real)
            matches.append(found)
    return matches
