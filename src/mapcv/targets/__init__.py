"""Dataset targets: the per-patch annotation a task produces next to the image."""

from __future__ import annotations

from mapcv.config import MapcvConfig
from mapcv.targets.base import Annotation, AnnotationBatch, Target, WindowTarget
from mapcv.targets.detection import DetectedObject, DetectionTarget, PatchObjects
from mapcv.targets.none import ImageOnlyTarget
from mapcv.targets.segmentation import SegmentationTarget

__all__ = [
    "Annotation",
    "AnnotationBatch",
    "DetectedObject",
    "DetectionTarget",
    "ImageOnlyTarget",
    "PatchObjects",
    "SegmentationTarget",
    "Target",
    "WindowTarget",
    "create_target",
]


def create_target(config: MapcvConfig) -> Target:
    """The target ``config.task`` asks for; without labels, patches carry no annotation."""
    if config.task == "detection":
        if config.labels is None:  # the config refuses this; keep the type checker honest
            raise ValueError("task: detection needs labels")
        return DetectionTarget(config.labels, config.detection_options, config.sampler.patch_size)
    if config.task != "segmentation":  # the config only accepts supported tasks
        raise ValueError(f"task '{config.task}' has no target")
    if config.labels is None:
        return ImageOnlyTarget()
    return SegmentationTarget(config.labels)
