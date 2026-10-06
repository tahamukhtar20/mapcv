"""Instance segmentation datasets: image files plus COCO JSON with RLE masks.

Layout (relative to ``writer.staging_dir``)::

    images/patch_0000000.png          one image per patch (the ``files`` layout)
    masks/patch_0000000.png           optional (instance.id_mask): 16-bit instance IDs
    annotations/instances_train.json  COCO, one file per split (``instances_all.json`` without a split)
    annotations/objects/chunk_000000.json  the instances of each chunk (internal)

Each COCO annotation is one instance: ``segmentation`` is its mask as a compressed RLE
(``{"size": [height, width], "counts": "..."}``, column-major, as ``pycocotools``
reads it), ``bbox`` the tight ``[x, y, width, height]`` of the mask, ``area`` its pixel
count and ``iscrowd`` is 0. ``truncated`` (true when part of the feature is not
visible in the patch) is an extra field that COCO readers ignore. The optional
instance-ID mask holds 0 for background and ``k`` for the ``k``-th annotation of the
image (annotations are in feature order, and their ``id`` ascends); where instances
overlap, the later one wins there, while the COCO masks stay exact.

``write`` stores each chunk's instances in ``annotations/objects/`` next to the images.
``finalize`` (after the split, and again after ``mapcv split``) builds the per-split COCO
files from those chunk files and the manifest, so a resumed run or a re-split always
describes every patch. IDs are stable as for detection: a COCO image's ``id`` is its patch
number + 1 and annotation IDs count the instances of all patches in manifest order from 1.
"""

from __future__ import annotations

import json
import posixpath
import warnings
from pathlib import Path
from typing import Any, Dict, FrozenSet, List, Optional, Sequence

import numpy as np
import numpy.typing as npt

from mapcv.config import InstanceOptions
from mapcv.footprints import FOOTPRINTS_FILENAME, write_footprints
from mapcv.imagery import RasterMetadata
from mapcv.manifest import Manifest, PatchSummary
from mapcv.sampler import PatchMeta
from mapcv.splitter import SplitLists
from mapcv.targets.instance import Instance, PatchInstances
from mapcv.writer import WriterConfig, write_patches
from mapcv.writers.detection import (
    ANNOTATIONS_DIR,
    DEFAULT_CATEGORY,
    IMAGES_DIR,
    SPLIT_NAMES,
    _chunk_file,
    _write_text,
    categories,
    dump_coco,
    load_objects,
)
from mapcv.writers.files import FilesWriter

MASKS_DIR = "masks"

# One stored instance: [category_id, x, y, width, height, area, truncated (0/1), rle counts].
InstanceRow = List[Any]


def _row(item: Instance) -> InstanceRow:
    return [item.category_id, *item.bbox, item.area, int(item.truncated), item.counts]


def coco_document(
    manifest: Manifest,
    instances: Sequence[Sequence[InstanceRow]],
    names: Optional[FrozenSet[str]] = None,
    description: str = "",
) -> Dict[str, Any]:
    """A COCO instance segmentation document for the patches whose image file name is in
    ``names`` (all patches when ``None``), with IDs that do not depend on the selection."""
    size = int((manifest.sampler or {}).get("patch_size") or 0)
    images: List[Dict[str, Any]] = []
    annotations: List[Dict[str, Any]] = []
    next_id = 1
    for index, (entry, rows) in enumerate(zip(manifest.patches, instances)):
        file_name = posixpath.basename(entry["files"]["image"])
        first_id, next_id = next_id, next_id + len(rows)
        if names is not None and file_name not in names:
            continue
        image_id = index + 1
        images.append({"id": image_id, "file_name": file_name, "width": size, "height": size})
        for offset, (cid, x, y, w, h, area, truncated, counts) in enumerate(rows):
            annotations.append(
                {
                    "id": first_id + offset,
                    "image_id": image_id,
                    "category_id": cid,
                    "segmentation": {"size": [size, size], "counts": counts},
                    "bbox": [x, y, w, h],
                    "area": area,
                    "iscrowd": 0,
                    "truncated": bool(truncated),
                }
            )
    return {
        "info": {"description": description, "mapcv_version": manifest.mapcv_version},
        "licenses": [],
        "categories": [
            {"id": cid, "name": name, "supercategory": DEFAULT_CATEGORY}
            for cid, name in categories(manifest.class_map).items()
        ],
        "images": images,
        "annotations": annotations,
    }


class InstanceWriter:
    """Image files in ``images/`` plus COCO RLE annotations (see the module docs).

    Annotations must be the instance target's collated :class:`PatchInstances`.
    """

    TARGET_TYPES: FrozenSet[Optional[str]] = frozenset({"instance"})

    def __init__(self, config: WriterConfig, options: InstanceOptions) -> None:
        self._config = config
        self._options = options

    @classmethod
    def from_manifest(cls, manifest: Manifest, staging_dir: Path) -> "InstanceWriter":
        """The writer of an existing instance dataset, to rebuild its split outputs."""
        options = manifest.target.options if manifest.target is not None else {}
        return cls(WriterConfig(staging_dir=staging_dir), InstanceOptions.model_validate(options))

    @property
    def layout(self) -> str:
        return "files"

    def supports(self, target_type: Optional[str]) -> bool:
        return target_type in self.TARGET_TYPES

    def fingerprint(self) -> Dict[str, Any]:
        # As the files layout records it; mask_format only matters with instance-ID masks.
        exclude = {"staging_dir", "world_files", "footprints"}
        if not self._options.id_mask:
            exclude.add("mask_format")
        block = {"layout": self.layout, **self._config.model_dump(mode="json", exclude=exclude)}
        if self._config.world_files:
            block["world_files"] = True
        return block

    def patch_shape(self, source: RasterMetadata, patch_size: int) -> List[int]:
        return FilesWriter(self._config).patch_shape(source, patch_size)

    def write(
        self,
        images: npt.NDArray[np.generic],
        annotations: List[PatchInstances],
        metadata: List[PatchMeta],
        manifest: Manifest,
        chunk_index: int,
    ) -> None:
        if not isinstance(annotations, list) or len(annotations) != len(metadata):
            raise TypeError("InstanceWriter writes instance targets: one PatchInstances per patch")
        if not metadata:
            return
        for annotation in annotations:
            if not isinstance(annotation, PatchInstances):
                raise TypeError("InstanceWriter writes instance targets: PatchInstances expected")
        staging = self._config.staging_dir
        start = len(manifest.patches)
        id_masks: Optional[npt.NDArray[np.uint16]] = None
        if self._options.id_mask:
            id_masks = np.stack([_id_mask(annotation) for annotation in annotations])
        write_patches(
            images,
            id_masks,
            metadata,
            self._config,
            manifest,
            chunk_index,
            images_dir=IMAGES_DIR,
            masks_dir=MASKS_DIR,
        )
        stored: Dict[str, List[InstanceRow]] = {}
        for entry, annotation in zip(manifest.patches[start:], annotations):
            entry["summary"] = PatchSummary(
                class_objects=annotation.class_counts, empty_ratio=entry["summary"]["empty_ratio"]
            )
            name = posixpath.basename(entry["files"]["image"])
            stored[name] = [_row(item) for item in annotation.instances]
        chunk_file = _chunk_file(staging, chunk_index)
        chunk_file.parent.mkdir(parents=True, exist_ok=True)
        if chunk_file.exists():
            # A resumed run can finish a chunk whose first patches are already recorded.
            previous = json.loads(chunk_file.read_text(encoding="utf-8"))["patches"]
            stored = {**previous, **stored}
        patches = {name: stored[name] for name in sorted(stored)}
        _write_text(
            chunk_file,
            json.dumps({"chunk": chunk_index, "patches": patches}, separators=(",", ":")) + "\n",
        )

    def finalize(self, manifest: Manifest, split_lists: Optional[SplitLists]) -> None:
        """Write ``patches.geojson`` (``writer.footprints``) and the annotation files."""
        if self._config.footprints:
            path = self._config.staging_dir / FOOTPRINTS_FILENAME
            try:
                write_footprints(manifest, split_lists, path)
            except (RuntimeError, ValueError) as exc:
                warnings.warn(
                    f"{FOOTPRINTS_FILENAME} was not written: {exc}", UserWarning, stacklevel=2
                )
        self.write_annotations(manifest, split_lists)

    def write_annotations(self, manifest: Manifest, split_lists: Optional[SplitLists]) -> None:
        """Write the COCO files for the split."""
        staging = self._config.staging_dir
        instances = load_objects(manifest, staging, "instances")
        annotations_dir = staging / ANNOTATIONS_DIR
        annotations_dir.mkdir(parents=True, exist_ok=True)
        wanted: Dict[str, Optional[FrozenSet[str]]]
        if split_lists is None:
            wanted = {"instances_all.json": None}
        else:
            wanted = {
                f"instances_{name}.json": frozenset(getattr(split_lists, name))
                for name in SPLIT_NAMES
            }
        for file_name, names in wanted.items():
            split = file_name[len("instances_") : -len(".json")]
            document = coco_document(
                manifest, instances, names, f"mapcv instance segmentation dataset, {split} patches"
            )
            _write_text(annotations_dir / file_name, dump_coco(document))
        for stale in {f"instances_{name}.json" for name in (*SPLIT_NAMES, "all")} - set(wanted):
            (annotations_dir / stale).unlink(missing_ok=True)


def _id_mask(annotation: PatchInstances) -> npt.NDArray[np.uint16]:
    if annotation.id_mask is None:
        raise ValueError("instance.id_mask is on, but the patch carries no instance-ID mask")
    return annotation.id_mask
