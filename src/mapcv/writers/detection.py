"""Detection datasets: image files plus COCO JSON and YOLO labels.

Layout (relative to ``writer.staging_dir``)::

    images/patch_0000000.png        one image per patch (the ``files`` layout)
    labels/patch_0000000.txt        YOLO: one line per object; no file for a patch without objects
    annotations/instances_train.json  COCO, one file per split (``instances_all.json`` without a split)
    annotations/objects/chunk_000000.json  the exact boxes of each chunk (internal)
    train.txt, val.txt, test.txt    YOLO image lists (``./images/...``), one per split
    dataset.yaml                    Ultralytics dataset file (written with a split; no ``path``,
                                    so paths are relative to its folder)

``write`` stores each chunk's boxes in ``annotations/objects/`` with full
precision next to the images and YOLO labels. ``finalize`` (after the split, and
again after ``mapcv split``) builds the per-split COCO files, the YOLO image
lists and ``dataset.yaml`` from those chunk files and the manifest, so a resumed
run or a re-split always describes every patch.

IDs are stable: a COCO image's ``id`` is its patch number + 1 (``patch_0000041``
is image 42) and annotation IDs count the objects of all patches in manifest
order from 1, so an object keeps its ID in every split file and after a re-split.
COCO ``category_id`` is the class ID from the class map; YOLO class indices are
0-based, in ascending class-ID order (``dataset.yaml``'s ``names``).
"""

from __future__ import annotations

import json
import os
import posixpath
import warnings
from pathlib import Path
from typing import Any, Dict, FrozenSet, List, Optional, Sequence

import numpy as np
import numpy.typing as npt
import yaml

from mapcv.config import DetectionOptions
from mapcv.footprints import FOOTPRINTS_FILENAME, write_footprints
from mapcv.imagery import RasterMetadata
from mapcv.labels import ClassMap
from mapcv.manifest import Manifest, PatchSummary
from mapcv.sampler import PatchMeta
from mapcv.splitter import SplitLists
from mapcv.targets.detection import DetectedObject, PatchObjects
from mapcv.writer import WriterConfig, write_patches
from mapcv.writers.files import FilesWriter

IMAGES_DIR = "images"
LABELS_DIR = "labels"
ANNOTATIONS_DIR = "annotations"
OBJECTS_DIR = "annotations/objects"
DATASET_YAML = "dataset.yaml"
SPLIT_NAMES = ("train", "val", "test")
# The category name when the labels have no label_field (every feature is class 1).
DEFAULT_CATEGORY = "object"

# One stored object: [category_id, x, y, width, height, area, truncated (0/1)].
ObjectRow = List[Any]


def categories(class_map: ClassMap) -> Dict[int, str]:
    """Class ID to name, ascending by ID; ``{1: "object"}`` without a class map."""
    if not class_map:
        return {1: DEFAULT_CATEGORY}
    return {cid: name for name, cid in sorted(class_map.items(), key=lambda item: item[1])}


def yolo_indices(class_map: ClassMap) -> Dict[int, int]:
    """Class ID to YOLO class index: 0, 1, ... in ascending class-ID order."""
    return {cid: index for index, cid in enumerate(categories(class_map))}


def _number(value: float) -> str:
    """A normalized coordinate with 10 decimals and no trailing zeros."""
    text = format(min(max(value, 0.0), 1.0), ".10f").rstrip("0").rstrip(".")
    return text or "0"


def yolo_lines(
    objects: Sequence[ObjectRow], indices: Dict[int, int], width: int, height: int
) -> str:
    """YOLO label text: ``class cx cy w h`` per object, normalized by the image size."""
    lines = []
    for cid, x, y, w, h, *_ in objects:
        lines.append(
            f"{indices[cid]} {_number((x + w / 2) / width)} {_number((y + h / 2) / height)} "
            f"{_number(w / width)} {_number(h / height)}\n"
        )
    return "".join(lines)


def _row(item: DetectedObject) -> ObjectRow:
    return [item.category_id, *item.bbox, item.area, int(item.truncated)]


def _write_text(path: Path, text: str) -> None:
    """Write atomically, so an interrupted run never leaves half a file."""
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(text, encoding="utf-8", newline="\n")
    os.replace(tmp, path)


def _chunk_file(staging: Path, chunk: int) -> Path:
    return staging / OBJECTS_DIR / f"chunk_{chunk:06d}.json"


def load_objects(manifest: Manifest, staging: Path) -> List[List[ObjectRow]]:
    """Every patch's stored objects, in manifest order.

    Raises:
        ValueError: A chunk file or a patch in it is missing (the dataset is incomplete).
    """
    by_chunk: Dict[int, Dict[str, List[ObjectRow]]] = {}
    result: List[List[ObjectRow]] = []
    for entry in manifest.patches:
        chunk = entry["chunk"]
        if chunk not in by_chunk:
            path = _chunk_file(staging, chunk)
            if not path.exists():
                raise ValueError(
                    f"{path} is missing, so the boxes of chunk {chunk} are unknown; "
                    "regenerate the dataset into a new writer.staging_dir"
                )
            by_chunk[chunk] = json.loads(path.read_text(encoding="utf-8"))["patches"]
        name = posixpath.basename(entry["files"]["image"])
        stored = by_chunk[chunk].get(name)
        if stored is None:
            raise ValueError(
                f"{_chunk_file(staging, chunk)} has no boxes for {name}; regenerate the "
                "dataset into a new writer.staging_dir"
            )
        result.append(stored)
    return result


def coco_document(
    manifest: Manifest,
    objects: Sequence[Sequence[ObjectRow]],
    names: Optional[FrozenSet[str]] = None,
    description: str = "",
) -> Dict[str, Any]:
    """A COCO detection document for the patches whose image file name is in ``names``
    (all patches when ``None``), with IDs that do not depend on the selection."""
    size = int((manifest.sampler or {}).get("patch_size") or 0)
    images: List[Dict[str, Any]] = []
    annotations: List[Dict[str, Any]] = []
    next_id = 1
    for index, (entry, rows) in enumerate(zip(manifest.patches, objects)):
        file_name = posixpath.basename(entry["files"]["image"])
        first_id, next_id = next_id, next_id + len(rows)
        if names is not None and file_name not in names:
            continue
        image_id = index + 1
        images.append({"id": image_id, "file_name": file_name, "width": size, "height": size})
        for offset, (cid, x, y, w, h, area, truncated) in enumerate(rows):
            annotations.append(
                {
                    "id": first_id + offset,
                    "image_id": image_id,
                    "category_id": cid,
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


def dump_coco(document: Dict[str, Any]) -> str:
    """COCO JSON with one image, annotation or category per line (diffable, greppable)."""
    parts = []
    for key, value in document.items():
        if isinstance(value, list) and value:
            rows = ",\n    ".join(json.dumps(item, ensure_ascii=False) for item in value)
            parts.append(f'  "{key}": [\n    {rows}\n  ]')
        else:
            parts.append(f'  "{key}": {json.dumps(value, ensure_ascii=False)}')
    return "{\n" + ",\n".join(parts) + "\n}\n"


class DetectionWriter:
    """Image files in ``images/`` plus COCO and/or YOLO annotations (see the module docs).

    Annotations must be the detection target's collated :class:`PatchObjects`.
    """

    TARGET_TYPES: FrozenSet[Optional[str]] = frozenset({"detection"})

    def __init__(self, config: WriterConfig, options: DetectionOptions) -> None:
        self._config = config
        self._formats = tuple(options.formats)

    @classmethod
    def from_manifest(cls, manifest: Manifest, staging_dir: Path) -> "DetectionWriter":
        """The writer of an existing detection dataset, to rebuild its split outputs."""
        options = manifest.target.options if manifest.target is not None else {}
        formats = options.get("formats") or list(DetectionOptions().formats)
        return cls(WriterConfig(staging_dir=staging_dir), DetectionOptions(formats=formats))

    @property
    def layout(self) -> str:
        return "files"

    def supports(self, target_type: Optional[str]) -> bool:
        return target_type in self.TARGET_TYPES

    def fingerprint(self) -> Dict[str, Any]:
        # As the files layout records it, minus mask_format: detection writes no masks.
        block = {
            "layout": self.layout,
            **self._config.model_dump(
                mode="json", exclude={"staging_dir", "mask_format", "world_files", "footprints"}
            ),
        }
        if self._config.world_files:
            block["world_files"] = True
        return block

    def patch_shape(self, source: RasterMetadata, patch_size: int) -> List[int]:
        return FilesWriter(self._config).patch_shape(source, patch_size)

    def write(
        self,
        images: npt.NDArray[np.generic],
        annotations: List[PatchObjects],
        metadata: List[PatchMeta],
        manifest: Manifest,
        chunk_index: int,
    ) -> None:
        if not isinstance(annotations, list) or len(annotations) != len(metadata):
            raise TypeError("DetectionWriter writes detection targets: one PatchObjects per patch")
        if not metadata:
            return
        staging = self._config.staging_dir
        start = len(manifest.patches)
        write_patches(
            images, None, metadata, self._config, manifest, chunk_index, images_dir=IMAGES_DIR
        )
        indices = yolo_indices(manifest.class_map)
        labels_dir = staging / LABELS_DIR
        if "yolo" in self._formats:
            labels_dir.mkdir(parents=True, exist_ok=True)
        stored: Dict[str, List[ObjectRow]] = {}
        for entry, annotation in zip(manifest.patches[start:], annotations):
            if not isinstance(annotation, PatchObjects):
                raise TypeError("DetectionWriter writes detection targets: PatchObjects expected")
            entry["summary"] = PatchSummary(
                class_objects=annotation.class_counts, empty_ratio=entry["summary"]["empty_ratio"]
            )
            name = posixpath.basename(entry["files"]["image"])
            rows = [_row(item) for item in annotation.objects]
            stored[name] = rows
            if "yolo" in self._formats:
                label = labels_dir / f"{posixpath.splitext(name)[0]}.txt"
                if rows:
                    size = annotation.patch_size
                    _write_text(label, yolo_lines(rows, indices, size, size))
                else:  # Ultralytics' convention: no label file for an image without objects
                    label.unlink(missing_ok=True)
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
        """Write the COCO files, YOLO image lists and ``dataset.yaml`` for the split."""
        staging = self._config.staging_dir
        objects = load_objects(manifest, staging)
        annotations_dir = staging / ANNOTATIONS_DIR
        annotations_dir.mkdir(parents=True, exist_ok=True)
        splits: Dict[str, List[str]] = (
            {name: list(getattr(split_lists, name)) for name in SPLIT_NAMES}
            if split_lists is not None
            else {}
        )

        coco_files = {f"instances_{name}.json" for name in (*SPLIT_NAMES, "all")}
        if "coco" in self._formats:
            wanted: Dict[str, Optional[FrozenSet[str]]]
            if split_lists is None:
                wanted = {"instances_all.json": None}
            else:
                wanted = {f"instances_{name}.json": frozenset(splits[name]) for name in SPLIT_NAMES}
            for file_name, names in wanted.items():
                split = file_name[len("instances_") : -len(".json")]
                document = coco_document(
                    manifest, objects, names, f"mapcv object detection dataset, {split} patches"
                )
                _write_text(annotations_dir / file_name, dump_coco(document))
            coco_files -= set(wanted)
        for stale in coco_files:
            (annotations_dir / stale).unlink(missing_ok=True)

        yolo_outputs = [staging / DATASET_YAML, *(staging / f"{name}.txt" for name in SPLIT_NAMES)]
        if "yolo" not in self._formats:
            return
        if split_lists is None:
            for stale_path in yolo_outputs:
                stale_path.unlink(missing_ok=True)
            warnings.warn(
                "the dataset has no split, so dataset.yaml (which Ultralytics needs with train "
                "and val lists) was not written; add a split: block or run 'mapcv split'",
                UserWarning,
                stacklevel=2,
            )
            return
        for name in SPLIT_NAMES:
            lines = "".join(f"./{IMAGES_DIR}/{image}\n" for image in splits[name])
            _write_text(staging / f"{name}.txt", lines)
        _write_text(staging / DATASET_YAML, dataset_yaml(manifest, splits))


def dataset_yaml(manifest: Manifest, splits: Dict[str, List[str]]) -> str:
    """Ultralytics' dataset file: image lists per split and class ``names``.

    There is no ``path`` key: Ultralytics then takes the dataset root from the
    folder of the YAML file itself, so the dataset keeps working when it is moved
    or copied to another machine. The image lists sit next to it.
    """
    data: Dict[str, Any] = {"train": "train.txt", "val": "val.txt"}
    if splits.get("test"):
        data["test"] = "test.txt"
    data["names"] = {
        index: categories(manifest.class_map)[cid]
        for cid, index in yolo_indices(manifest.class_map).items()
    }
    header = (
        "# Ultralytics YOLO dataset written by mapcv. Train with:\n"
        "#   yolo detect train data=<this file>\n"
        "# Paths are relative to this file's folder (no `path:` key), so the dataset can move.\n"
    )
    body: str = yaml.safe_dump(data, sort_keys=False, allow_unicode=True)
    return header + body
