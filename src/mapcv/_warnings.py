"""Record the warnings of the code inside a block, also when other threads record theirs.

``warnings.catch_warnings(record=True)`` swaps process-wide state, so two blocks that
overlap in time (a ``plan`` next to a ``generate`` in the MCP server) take each other's
warnings and, when they end out of order, leave the swapped state behind. :func:`capture`
installs one hook while any block is open and hands every warning to the block of the
thread that raised it. In a single-threaded program (the CLI) it behaves like
``catch_warnings(record=True)`` with every warning shown.
"""

from __future__ import annotations

import threading
import warnings
from collections.abc import Iterator
from contextlib import AbstractContextManager, contextmanager
from typing import Any

__all__ = ["capture"]


class _Router:
    """The hook, and the blocks that are open.

    A warning goes to the innermost block of the thread that raised it. One from a thread
    no block owns (a worker pool) goes to the ``broad`` blocks (a generation), or to every
    block when there is none.
    """

    def __init__(self) -> None:
        self._lock = threading.RLock()
        self._captures: list[tuple[int, bool, list[warnings.WarningMessage]]] = []
        self._context: warnings.catch_warnings | None = None

    def _show(
        self,
        message: Warning | str,
        category: type[Warning],
        filename: str,
        lineno: int,
        file: Any = None,
        line: str | None = None,
    ) -> None:
        record = warnings.WarningMessage(message, category, filename, lineno, file, line)
        me = threading.get_ident()
        with self._lock:
            mine = [log for thread, _, log in self._captures if thread == me]
            if mine:
                targets = mine[-1:]  # the innermost block, as nested catch_warnings would
            else:
                broad = [log for _, wide, log in self._captures if wide]
                targets = broad or [log for _, _, log in self._captures]
            for log in targets:
                log.append(record)

    @contextmanager
    def capture(self, broad: bool = False) -> Iterator[list[warnings.WarningMessage]]:
        log: list[warnings.WarningMessage] = []
        entry = (threading.get_ident(), broad, log)
        with self._lock:
            if not self._captures:
                self._context = warnings.catch_warnings()
                self._context.__enter__()
                warnings.simplefilter("always")
                warnings.showwarning = self._show
            self._captures.append(entry)
        try:
            yield log
        finally:
            with self._lock:
                self._captures.remove(entry)
                if not self._captures and self._context is not None:
                    self._context.__exit__(None, None, None)
                    self._context = None


_ROUTER = _Router()


def capture(broad: bool = False) -> AbstractContextManager[list[warnings.WarningMessage]]:
    """Record the warnings raised inside the block, by this thread only.

    ``broad`` also takes the warnings of threads that belong to no block, such as the
    workers a generation starts. The list fills while the block runs.
    """
    return _ROUTER.capture(broad)
