"""Dataset writers: output layouts behind one interface."""

from __future__ import annotations

from pathlib import Path
from typing import Optional

from mapcv.manifest import Manifest
from mapcv.splitter import SplitLists
from mapcv.targets.base import Target
from mapcv.targets.detection import DetectionTarget
from mapcv.writer import WriterConfig
from mapcv.writers.base import Writer
from mapcv.writers.detection import DetectionWriter
from mapcv.writers.files import FilesWriter

__all__ = [
    "DetectionWriter",
    "FilesWriter",
    "Writer",
    "check_compatible",
    "create_writer",
    "refresh_split_outputs",
]


def create_writer(config: WriterConfig, target: Optional[Target] = None) -> Writer:
    """The writer a ``writer:`` block asks for, for ``target``'s annotations.

    One file per patch; detection targets also get COCO/YOLO annotation files.
    """
    if isinstance(target, DetectionTarget):
        return DetectionWriter(config, target.options)
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


def refresh_split_outputs(
    manifest: Manifest, staging_dir: Path, split_lists: Optional[SplitLists]
) -> None:
    """Rebuild the outputs that depend on the split after ``mapcv split``.

    Detection datasets get new per-split COCO files, YOLO image lists and
    ``dataset.yaml``; other tasks have none. (``run_split`` updates
    ``patches.geojson`` itself, for every task.)
    """
    if manifest.task == "detection":
        writer = DetectionWriter.from_manifest(manifest, staging_dir)
        writer.write_annotations(manifest, split_lists)
