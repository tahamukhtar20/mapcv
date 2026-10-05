"""Dataset writers: output layouts behind one interface."""

from __future__ import annotations

from mapcv.writer import WriterConfig
from mapcv.writers.base import Writer
from mapcv.writers.files import FilesWriter

__all__ = ["FilesWriter", "Writer", "create_writer"]


def create_writer(config: WriterConfig) -> Writer:
    """The writer a ``writer:`` block asks for (today always one file per patch)."""
    return FilesWriter(config)
