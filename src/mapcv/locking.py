"""One writer per dataset folder: ``generate`` and ``split`` hold ``.mapcv.lock`` while they
write, so two runs into one folder cannot interleave their files.

The lock is an operating-system file lock (``flock`` on POSIX, ``msvcrt.locking`` on
Windows) on ``<dataset>/.mapcv.lock``. The system releases it when the process ends,
however it ends, so a crashed run never leaves a stale lock behind: a lock file left
on disk is simply locked again by the next run. The file is removed when the lock is
released.
"""

from __future__ import annotations

import os
import sys
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path

LOCK_FILENAME = ".mapcv.lock"


class DatasetBusyError(RuntimeError):
    """Another mapcv process is writing to the dataset folder."""


class StagingDirError(ValueError):
    """The dataset folder holds files mapcv did not write, and no manifest."""


if sys.platform == "win32":  # pragma: no cover - exercised on the Windows CI runners
    import msvcrt

    def _try_lock(fd: int) -> bool:
        os.lseek(fd, 0, os.SEEK_SET)
        try:
            msvcrt.locking(fd, msvcrt.LK_NBLCK, 1)
        except OSError:
            return False
        return True

    def _unlock(fd: int) -> None:
        os.lseek(fd, 0, os.SEEK_SET)
        try:
            msvcrt.locking(fd, msvcrt.LK_UNLCK, 1)
        except OSError:
            pass

else:
    import fcntl

    def _try_lock(fd: int) -> bool:
        try:
            fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError:
            return False
        return True

    def _unlock(fd: int) -> None:
        fcntl.flock(fd, fcntl.LOCK_UN)


def _holder(path: Path) -> str:
    """`` (process 1234)`` from the lock file, or nothing when it cannot be read."""
    try:
        pid = path.read_text(encoding="utf-8").strip()
    except (OSError, ValueError):
        return ""
    return f" (process {pid})" if pid.isdigit() else ""


@contextmanager
def dataset_lock(folder: Path) -> Iterator[None]:
    """Hold the dataset folder's lock for the block.

    Raises:
        DatasetBusyError: Another process holds it.
    """
    path = folder / LOCK_FILENAME
    while True:
        fd = os.open(path, os.O_RDWR | os.O_CREAT | getattr(os, "O_BINARY", 0), 0o644)
        if not _try_lock(fd):
            os.close(fd)
            raise DatasetBusyError(
                f"another mapcv command is writing to {folder}{_holder(path)}; wait for it "
                "to finish, or use another writer.staging_dir"
            )
        # The holder that released the lock just before may have removed the file, so the
        # lock taken may be on a file no longer at ``path``: then take it again.
        try:
            current = os.stat(path)
        except FileNotFoundError:
            current = None
        if current is not None and os.path.samestat(os.fstat(fd), current):
            break
        _unlock(fd)
        os.close(fd)
    try:
        os.ftruncate(fd, 0)
        os.lseek(fd, 0, os.SEEK_SET)
        os.write(fd, f"{os.getpid()}\n".encode())
        yield
    finally:
        if sys.platform != "win32":
            # Removed while still locked, so no other process can lock this file after.
            path.unlink(missing_ok=True)
        _unlock(fd)
        os.close(fd)
        if sys.platform == "win32":  # pragma: no cover - Windows cannot remove an open file
            try:
                path.unlink(missing_ok=True)
            except OSError:
                pass  # another run opened it meanwhile; it is that run's lock now
