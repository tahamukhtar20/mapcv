"""Dataset writers: output layouts behind one interface."""

from __future__ import annotations

from pathlib import Path

from mapcv.manifest import Manifest
from mapcv.splitter import SplitLists
from mapcv.targets.base import Target
from mapcv.targets.classification import ClassificationTarget
from mapcv.targets.detection import DetectionTarget
from mapcv.targets.instance import InstanceTarget
from mapcv.writer import WriterConfig
from mapcv.writers.base import Writer
from mapcv.writers.change import ChangeWriter
from mapcv.writers.classification import ClassificationWriter
from mapcv.writers.detection import DetectionWriter
from mapcv.writers.files import FilesWriter
from mapcv.writers.instance import InstanceWriter

__all__ = [
    "ChangeWriter",
    "ClassificationWriter",
    "DetectionWriter",
    "FilesWriter",
    "InstanceWriter",
    "Writer",
    "check_compatible",
    "create_writer",
    "refresh_split_outputs",
]


def create_writer(
    config: WriterConfig,
    target: Target | None = None,
    sources: list[str] | None = None,
) -> Writer:
    """The writer a ``writer:`` block asks for, for ``target``'s annotations.

    One file per patch; detection targets also get COCO/YOLO annotation files,
    instance targets COCO files with RLE masks and classification targets label tables.
    ``sources`` names the imagery sources of a multi-source dataset (``imagery`` as a
    list), whose patches go to ``Images/<name>/``; ``None`` for one ``imagery`` block.

    Raises:
        ValueError: Several sources for a task whose layout holds one image per patch.
    """
    target_type = target.type if target is not None else None
    if target_type == "change":
        if sources is None:
            raise ValueError("task: change needs two imagery sources: before and after")
        return ChangeWriter(config, sources)
    if sources is not None and target_type not in FilesWriter.TARGET_TYPES:
        raise ValueError(
            f"task: {target_type} writes one image per patch; several imagery sources are "
            "supported for segmentation and change datasets"
        )
    if sources is not None:
        return FilesWriter(config, sources)
    if isinstance(target, DetectionTarget):
        return DetectionWriter(config, target.options)
    if isinstance(target, InstanceTarget):
        return InstanceWriter(config, target.options)
    if isinstance(target, ClassificationTarget):
        return ClassificationWriter(config, target.options)
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
    manifest: Manifest, staging_dir: Path, split_lists: SplitLists | None
) -> None:
    """Rebuild the outputs that depend on the split after ``mapcv split``.

    Detection datasets get new per-split COCO files, YOLO image lists and
    ``dataset.yaml``, instance datasets new per-split COCO files and classification datasets
    new label tables (``labels.csv``'s split column and ``labels_<split>.csv``); segmentation
    has none. (``run_split`` updates
    ``patches.geojson`` itself, for every task.)
    """
    if manifest.task == "detection":
        writer = DetectionWriter.from_manifest(manifest, staging_dir)
        writer.write_annotations(manifest, split_lists)
    elif manifest.task == "instance":
        instance_writer = InstanceWriter.from_manifest(manifest, staging_dir)
        instance_writer.write_annotations(manifest, split_lists)
    elif manifest.task == "classification":
        classification_writer = ClassificationWriter.from_manifest(manifest, staging_dir)
        classification_writer.write_annotations(manifest, split_lists)
