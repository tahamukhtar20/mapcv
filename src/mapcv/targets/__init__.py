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
    """The target a configuration asks for: segmentation with labels, else image-only."""
    if config.labels is None:
        return ImageOnlyTarget()
    return SegmentationTarget(config.labels)
