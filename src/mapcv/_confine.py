"""Keeping reads and writes inside a dataset folder.

A dataset folder may come from elsewhere (a download, an archive, a shared drive).
Two things in it could make mapcv touch files outside it:

* a path in ``manifest.json`` or ``SHA256SUMS`` that is absolute or climbs out with
  ``..`` (:func:`is_dataset_path`);
* a symbolic link that resolves outside the folder, or a hard link (a file with
  another name somewhere else), which a write would go through
  (:func:`check_folder_links`).
"""

from __future__ import annotations

import os
from pathlib import Path

__all__ = ["LinkEscapeError", "check_folder_links", "first_outside_path", "is_dataset_path"]

# With every path between newlines, one of these is in the text exactly when a path is
# not an :func:`is_dataset_path` (a newline inside a path changes the line count).
_OUTSIDE_TOKENS = (
    "\\",
    ":",
    "\x00",
    "//",
    "\n/",
    "\n\n",
    "/\n",
    "/./",
    "/../",
    "\n./",
    "\n../",
    "/.\n",
    "/..\n",
    "\n.\n",
    "\n..\n",
)


def is_dataset_path(rel: str) -> bool:
    """Whether ``rel`` is a relative path that stays inside the dataset folder.

    mapcv writes ``/``-separated names like ``Images/patch_0000000.png``: no leading
    ``/``, no drive or ``\\``, no empty, ``.`` or ``..`` parts.
    """
    if not rel or rel[0] == "/" or any(char in rel for char in "\\:\x00\n"):
        return False
    return all(part not in ("", ".", "..") for part in rel.split("/"))


def first_outside_path(paths: list[str]) -> str | None:
    """The first of ``paths`` that is not a :func:`is_dataset_path`, or ``None``.

    A few substring searches over all of them at once: a manifest may list millions.
    """
    if not paths:
        return None
    joined = "\n" + "\n".join(paths) + "\n"
    if joined.count("\n") == len(paths) + 1 and not any(t in joined for t in _OUTSIDE_TOKENS):
        return None
    return next((rel for rel in paths if not is_dataset_path(rel)), None)


class LinkEscapeError(ValueError):
    """A folder holds a link that would let a read or write leave it."""


def check_folder_links(folder: Path, what: str, hard_links: bool = False) -> None:
    """Fail if something below ``folder`` lets a file operation leave it.

    A symbolic link counts when it resolves outside ``folder`` (links inside it are
    fine); with ``hard_links``, so does every regular file with more than one name,
    since its other names may be anywhere. A folder that does not exist passes.

    Raises:
        LinkEscapeError: Naming the first such entry and what to do.
    """
    try:
        root = folder.resolve()
    except (OSError, RuntimeError):
        return
    if not root.is_dir():
        return
    stack = [root]
    while stack:
        current = stack.pop()
        try:
            with os.scandir(current) as entries:
                found = sorted(entries, key=lambda entry: entry.name)
        except OSError:
            continue
        for entry in found:
            path = Path(entry.path)
            if entry.is_symlink():
                try:
                    target = path.resolve()
                except (OSError, RuntimeError):
                    target = None
                if target is None or not target.is_relative_to(root):
                    raise LinkEscapeError(
                        f"{what}: {path.relative_to(root).as_posix()} is a link to "
                        f"{target if target is not None else 'a path that cannot be resolved'}, "
                        f"outside {folder}. mapcv does not read or write through links that "
                        "leave the folder: replace the link with the file itself, or use "
                        "another folder."
                    )
            elif entry.is_dir(follow_symlinks=False):
                stack.append(path)
            elif hard_links:
                try:
                    links = os.lstat(entry.path).st_nlink  # DirEntry.stat has none on Windows
                except OSError:
                    continue
                if links > 1:
                    raise LinkEscapeError(
                        f"{what}: {path.relative_to(root).as_posix()} is a hard link (the "
                        f"same file has {links - 1} other name(s), possibly outside {folder}), "
                        "so writing it would change those too. Replace it with a copy, or use "
                        "another folder."
                    )
