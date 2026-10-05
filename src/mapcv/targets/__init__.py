"""Dataset targets: the per-patch annotation a task produces next to the image."""

from __future__ import annotations

from mapcv.config import MapcvConfig
from mapcv.targets.base import Annotation, AnnotationBatch, Target, WindowTarget
from mapcv.targets.none import ImageOnlyTarget
from mapcv.targets.segmentation import SegmentationTarget

__all__ = [
    "Annotation",
    "AnnotationBatch",
    "ImageOnlyTarget",
    "SegmentationTarget",
    "Target",
    "WindowTarget",
    "create_target",
]


def create_target(config: MapcvConfig) -> Target:
    """The target ``config.task`` asks for; without labels, patches carry no annotation."""
    if config.task != "segmentation":  # the config only accepts supported tasks
        raise ValueError(f"task '{config.task}' has no target")
    if config.labels is None:
        return ImageOnlyTarget()
    return SegmentationTarget(config.labels)
