"""Classification datasets: image files plus a CSV and a JSON of patch labels.

Layout (relative to ``writer.staging_dir``), flat as EuroSAT and BigEarthNet are::

    images/patch_0000000.png   one image per patch (the ``files`` layout)
    labels.csv                 image,labels  (plus split once the dataset has splits)
    labels_train.csv           image,labels of the train patches; also labels_val.csv and
    labels_test.csv            labels_test.csv (written with a split only)
    classes.txt                one class name per line, in class ID order
    labels.json                {"classes": [...], "images": {"patch_0000000.png": [...]}}

``image`` is the image's file name inside ``images/``. ``labels`` holds the patch's class
names joined by one space (a single name with ``classification.mode: single``); CSV
quoting applies where a name needs it. With ``classification.empty: background`` the
label ``background`` is the first class and the label of patches that no class
qualifies for. ``split`` is ``train``, ``val`` or ``test``, and empty for a patch a
spatial split left out. Rows follow the manifest's patch order. ``labels.json`` lists
each image's labels as an array, for single-label datasets too, and does not depend on
the split.

``write`` records every patch's coverage and labels in the manifest summary, so
``finalize`` (after the split, and again after ``mapcv split`` or a resumed run) builds
all of these files from the manifest alone. Text is written atomically, UTF-8, with
``\\n`` line ends on every OS.
"""

from __future__ import annotations

import csv
import io
import json
import posixpath
import warnings
from pathlib import Path
from typing import Any, Dict, FrozenSet, List, Optional

import numpy as np
import numpy.typing as npt

from mapcv.config import BACKGROUND_LABEL, ClassificationOptions
from mapcv.footprints import FOOTPRINTS_FILENAME, write_footprints
from mapcv.imagery import RasterMetadata
from mapcv.manifest import Manifest, PatchSummary
from mapcv.sampler import PatchMeta
from mapcv.splitter import SplitLists
from mapcv.targets.classification import PatchLabels, class_names
from mapcv.writer import WriterConfig, write_patches
from mapcv.writers.detection import IMAGES_DIR, SPLIT_NAMES, _write_text
from mapcv.writers.files import FilesWriter

LABELS_CSV = "labels.csv"
LABELS_JSON = "labels.json"
CLASSES_TXT = "classes.txt"


def split_csv_name(split: str) -> str:
    """The file name of one split's labels, ``labels_train.csv``."""
    return f"labels_{split}.csv"


def label_names(manifest: Manifest, options: ClassificationOptions) -> Dict[int, str]:
    """Class ID to label name, ascending: the classes, plus ``background`` as ID 0 when
    ``classification.empty`` is ``background``."""
    names = class_names(manifest.class_map)
    if options.empty == "background":
        names = {0: BACKGROUND_LABEL, **names}
    return names


def _csv_text(rows: List[List[str]]) -> str:
    buffer = io.StringIO()
    csv.writer(buffer, lineterminator="\n").writerows(rows)
    return buffer.getvalue()


class ClassificationWriter:
    """Image files in ``images/`` plus label tables (see the module docs).

    Annotations must be the classification target's collated :class:`PatchLabels`.
    """

    TARGET_TYPES: FrozenSet[Optional[str]] = frozenset({"classification"})

    def __init__(self, config: WriterConfig, options: ClassificationOptions) -> None:
        self._config = config
        self._options = options

    @classmethod
    def from_manifest(cls, manifest: Manifest, staging_dir: Path) -> "ClassificationWriter":
        """The writer of an existing classification dataset, to rebuild its split outputs."""
        options = manifest.target.options if manifest.target is not None else {}
        return cls(
            WriterConfig(staging_dir=staging_dir), ClassificationOptions.model_validate(options)
        )

    @property
    def layout(self) -> str:
        return "files"

    def supports(self, target_type: Optional[str]) -> bool:
        return target_type in self.TARGET_TYPES

    def fingerprint(self) -> Dict[str, Any]:
        # As the files layout records it, minus mask_format: classification writes no masks.
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
        annotations: List[PatchLabels],
        metadata: List[PatchMeta],
        manifest: Manifest,
        chunk_index: int,
    ) -> None:
        if not isinstance(annotations, list) or len(annotations) != len(metadata):
            raise TypeError(
                "ClassificationWriter writes classification targets: one PatchLabels per patch"
            )
        for annotation in annotations:
            if not isinstance(annotation, PatchLabels):
                raise TypeError(
                    "ClassificationWriter writes classification targets: PatchLabels expected"
                )
        if not metadata:
            return
        start = len(manifest.patches)
        write_patches(
            images, None, metadata, self._config, manifest, chunk_index, images_dir=IMAGES_DIR
        )
        for entry, annotation in zip(manifest.patches[start:], annotations):
            entry["summary"] = PatchSummary(
                class_coverage={str(cid): share for cid, share in annotation.coverage.items()},
                labels=list(annotation.labels),
                empty_ratio=entry["summary"]["empty_ratio"],
            )

    def finalize(self, manifest: Manifest, split_lists: Optional[SplitLists]) -> None:
        """Write ``patches.geojson`` (``writer.footprints``) and the label files."""
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
        """Write ``labels.csv``, the per-split CSVs, ``classes.txt`` and ``labels.json``."""
        staging = self._config.staging_dir
        staging.mkdir(parents=True, exist_ok=True)
        names = label_names(manifest, self._options)
        images: List[str] = []
        labeled: List[List[str]] = []  # each image's label names
        for entry in manifest.patches:
            image = posixpath.basename(entry["files"]["image"])
            assigned = entry["summary"].get("labels")
            if assigned is None:
                raise ValueError(
                    f"{image} has no labels in the manifest; regenerate the dataset into a new "
                    "writer.staging_dir"
                )
            images.append(image)
            labeled.append([names[cid] for cid in assigned])

        split_of: Dict[str, str] = {}
        if split_lists is not None:
            for split in SPLIT_NAMES:
                split_of.update({name: split for name in getattr(split_lists, split)})
        header = ["image", "labels"] + (["split"] if split_lists is not None else [])
        table = [header]
        for image, names_of in zip(images, labeled):
            row = [image, " ".join(names_of)]
            if split_lists is not None:
                row.append(split_of.get(image, ""))
            table.append(row)
        _write_text(staging / LABELS_CSV, _csv_text(table))

        for split in SPLIT_NAMES:
            path = staging / split_csv_name(split)
            if split_lists is None:
                path.unlink(missing_ok=True)  # left over from an earlier split
                continue
            per_split = [["image", "labels"]]
            per_split.extend(
                [image, " ".join(names_of)]
                for image, names_of in zip(images, labeled)
                if split_of.get(image) == split
            )
            _write_text(path, _csv_text(per_split))

        _write_text(staging / CLASSES_TXT, "".join(f"{name}\n" for name in names.values()))
        _write_text(staging / LABELS_JSON, _labels_json(list(names.values()), images, labeled))


def _labels_json(classes: List[str], images: List[str], labeled: List[List[str]]) -> str:
    """``labels.json``: the class names, and each image's labels one per line."""
    rows = ",\n    ".join(
        f"{json.dumps(image, ensure_ascii=False)}: {json.dumps(assigned, ensure_ascii=False)}"
        for image, assigned in zip(images, labeled)
    )
    body = f"\n    {rows}\n  " if rows else ""
    return (
        f'{{\n  "classes": {json.dumps(classes, ensure_ascii=False)},\n  "images": {{{body}}}\n}}\n'
    )
