"""Dataset targets: the per-patch annotation a task produces next to the image."""

from __future__ import annotations

from mapcv.config import ContinuousLabelsConfig, LabelsConfig, MapcvConfig, RasterLabelsConfig
from mapcv.targets.base import Annotation, AnnotationBatch, Target, WindowTarget
from mapcv.targets.change import ChangeTarget
from mapcv.targets.classification import ClassificationTarget, PatchLabels
from mapcv.targets.detection import DetectedObject, DetectionTarget, PatchObjects
from mapcv.targets.instance import Instance, InstanceTarget, PatchInstances
from mapcv.targets.none import ImageOnlyTarget
from mapcv.targets.raster_labels import RasterSegmentationTarget
from mapcv.targets.regression import RegressionTarget
from mapcv.targets.segmentation import SegmentationTarget

__all__ = [
    "Annotation",
    "AnnotationBatch",
    "ChangeTarget",
    "ClassificationTarget",
    "DetectedObject",
    "DetectionTarget",
    "ImageOnlyTarget",
    "Instance",
    "InstanceTarget",
    "PatchInstances",
    "PatchLabels",
    "PatchObjects",
    "RasterSegmentationTarget",
    "RegressionTarget",
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
        if not isinstance(config.labels, LabelsConfig):  # refused by the config too
            raise ValueError("task: detection needs vector labels, not a label raster")
        return DetectionTarget(config.labels, config.detection_options, config.sampler.patch_size)
    if config.task == "instance":
        if config.labels is None:  # the config refuses this; keep the type checker honest
            raise ValueError("task: instance needs labels")
        if not isinstance(config.labels, LabelsConfig):  # refused by the config too
            raise ValueError("task: instance needs vector labels, not a label raster")
        return InstanceTarget(config.labels, config.instance_options, config.sampler.patch_size)
    if config.task == "classification":
        if config.labels is None:  # the config refuses this; keep the type checker honest
            raise ValueError("task: classification needs labels")
        if isinstance(config.labels, ContinuousLabelsConfig):  # refused by the config too
            raise ValueError("task: classification needs class labels, not values")
        return ClassificationTarget(config.labels, config.classification_options)
    if config.task == "change":
        if isinstance(config.labels, ContinuousLabelsConfig):  # refused by the config
            raise ValueError("task: change needs vector labels or a classified label raster")
        return ChangeTarget(config.change_options, config.labels)
    if config.task == "regression":
        if not isinstance(config.labels, ContinuousLabelsConfig):  # refused by the config
            raise ValueError("task: regression needs labels.type: continuous")
        return RegressionTarget(config.labels)
    if config.task != "segmentation":  # the config only accepts supported tasks
        raise ValueError(f"task '{config.task}' has no target")
    if config.labels is None:
        return ImageOnlyTarget()
    if isinstance(config.labels, RasterLabelsConfig):
        return RasterSegmentationTarget(config.labels)
    if isinstance(config.labels, ContinuousLabelsConfig):  # refused by the config
        raise ValueError("task: segmentation needs class labels; values need task: regression")
    return SegmentationTarget(config.labels)
