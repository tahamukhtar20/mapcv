"""Input files that mapcv reads must be ordinary files.

A path that names a pipe (FIFO), a device such as ``/dev/zero`` or a socket makes a read
block for ever or fill the memory, so the files a user names (a config, a region, labels,
imagery) are checked before they are opened.
"""

from __future__ import annotations

import os
import stat
from pathlib import Path

__all__ = ["check_regular_file"]

_KINDS = (
    (stat.S_ISDIR, "a folder"),
    (stat.S_ISFIFO, "a named pipe (FIFO)"),
    (stat.S_ISCHR, "a character device"),
    (stat.S_ISBLK, "a block device"),
    (stat.S_ISSOCK, "a socket"),
)


def check_regular_file(path: str | os.PathLike[str], what: str) -> None:
    """Raise ``ValueError`` when ``path`` exists but is not an ordinary file.

    Links are followed, so a link to ``/dev/zero`` is refused. A path that does not exist
    (or cannot be examined) passes: the caller reports that in its own words.

    Args:
        path: The file the user named.
        what: How the message calls it, such as ``labels.path`` or ``the config file``.
    """
    try:
        mode = Path(path).stat().st_mode
    except (OSError, ValueError):
        return
    if stat.S_ISREG(mode):
        return
    kind = next((name for test, name in _KINDS if test(mode)), "not an ordinary file")
    raise ValueError(
        f"{what} '{os.fspath(path)}' is {kind}, not a regular file; mapcv reads ordinary "
        "files only. Point it at a file (copy the data into one if needed)."
    )
