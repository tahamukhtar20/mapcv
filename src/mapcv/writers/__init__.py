"""Dataset writers: output layouts behind one interface."""

from __future__ import annotations

from mapcv.targets.base import Target
from mapcv.writer import WriterConfig
from mapcv.writers.base import Writer
from mapcv.writers.files import FilesWriter

__all__ = ["FilesWriter", "Writer", "check_compatible", "create_writer"]


def create_writer(config: WriterConfig) -> Writer:
    """The writer a ``writer:`` block asks for (today always one file per patch)."""
    return FilesWriter(config)


def check_compatible(target: Target, writer: Writer) -> None:
    """Fail before any imagery is read when ``writer`` cannot store ``target``'s annotations.

    Raises:
        ValueError: The writer's layout does not support the target.
    """
    if writer.supports(target.type):
        return
    what = f"{target.type} targets" if target.type is not None else "image-only datasets"
    raise ValueError(
        f"the '{writer.layout}' writer layout cannot write {what}; "
        "choose a layout that supports this task"
    )
