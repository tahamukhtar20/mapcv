"""The dataset manifest (``manifest.json``): what a dataset holds and how it was made.

Version 3 records the task, the imagery sources, the target, the writer and the
sampler once, and per patch its position, the files it was written to and a
summary of its annotation. Versions 1 (mapcv 0.1) and 2 (mapcv 0.2) are read and
upgraded in memory; see :meth:`Manifest.load`.
"""

from __future__ import annotations

import json
import math
import os
import posixpath
import re
from importlib.metadata import PackageNotFoundError
from importlib.metadata import version as _package_version
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from pydantic import BaseModel, ConfigDict, Field, PrivateAttr, model_serializer, with_config
from pydantic_core import to_json
from typing_extensions import TypedDict

from mapcv.labels import ClassMap

MANIFEST_VERSION: int = 3
Transform = Tuple[float, float, float, float, float, float]

# A manifest whose first key is ``"version": 3`` (as mapcv writes it).
_CURRENT_VERSION_FIRST = re.compile(rf'\s*\{{\s*"version"\s*:\s*{MANIFEST_VERSION}\s*,')

# Folder names of the ``files`` layout; v1/v2 entries name files inside them.
IMAGES_DIR = "Images"
MASKS_DIR = "Masks"


def mapcv_version() -> Optional[str]:
    """The installed mapcv version, or ``None`` when running from an uninstalled tree."""
    try:
        return _package_version("mapcv")
    except PackageNotFoundError:
        return None


@with_config(ConfigDict(extra="allow"))
class PatchSummary(TypedDict, total=False):
    """What a patch holds, readable without opening its files.

    ``empty_ratio`` is always present. Segmentation adds ``class_pixels``: the
    mask's pixel count per class ID (as strings, ascending), ignore value
    included. Other tasks add their own keys.
    """

    class_pixels: Dict[str, int]
    empty_ratio: float


@with_config(ConfigDict(extra="allow"))
class ManifestEntry(TypedDict):
    """One patch (a plain ``dict`` at runtime).

    ``files`` maps a role to a path relative to the dataset folder, with ``/``
    separators: ``image`` (the source's name) and, for segmentation, ``mask``.
    """

    row: int
    col: int
    padded: bool
    chunk: int
    files: Dict[str, str]
    summary: PatchSummary


class SourceRecord(BaseModel):
    """One imagery source: its raster grid and how its patches are stored.

    ``name`` is also the key of the source's patch in each entry's ``files``.
    """

    model_config = ConfigDict(extra="allow")

    name: str = "image"
    source_type: str = "xyz"
    product_id: Optional[str] = None
    bands: List[str] = Field(default_factory=list)
    dtype: Optional[str] = None
    crs: Optional[str] = None
    transform: Optional[Transform] = None
    patch_shape: List[int] = Field(default_factory=list)
    # Identity of the input file (a GeoTIFF's size, time and header hash, or a URL's ETag),
    # so a resumed run refuses a different file. Absent for sources that have none.
    fingerprint: Optional[Dict[str, Any]] = None

    @model_serializer(mode="wrap")
    def _omit_missing_fingerprint(self, handler: Any) -> Dict[str, Any]:
        data: Dict[str, Any] = handler(self)
        if data.get("fingerprint") is None:
            data.pop("fingerprint", None)
        return data


class TargetRecord(BaseModel):
    """What each patch is annotated with.

    ``labels`` holds the label settings (without the path) and the ``sha256`` of
    the label file, so a resumed run notices edited labels. ``options`` holds
    task-specific settings (none for segmentation).
    """

    model_config = ConfigDict(extra="allow")

    type: str
    class_map: ClassMap = Field(default_factory=dict)
    ignore_index: Optional[int] = None
    dtype: Optional[str] = None
    labels: Optional[Dict[str, Any]] = None
    options: Dict[str, Any] = Field(default_factory=dict)


class ManifestMismatchError(ValueError):
    """A manifest cannot be read, or cannot be resumed with the current configuration."""


class Manifest(BaseModel):
    """Dataset manifest: dataset-wide records plus one entry per patch.

    Keys unknown to this mapcv are kept, so a manifest round-trips unchanged.
    """

    model_config = ConfigDict(extra="allow")

    version: int = MANIFEST_VERSION
    mapcv_version: Optional[str] = None
    task: str = "segmentation"
    sources: List[SourceRecord] = Field(default_factory=list)
    target: Optional[TargetRecord] = None
    writer: Optional[Dict[str, Any]] = None
    sampler: Optional[Dict[str, Any]] = None
    patches: List[ManifestEntry] = Field(default_factory=list)

    _upgraded_from: Optional[int] = PrivateAttr(default=None)

    # ── convenience views ────────────────────────────────────────────────────

    @property
    def source(self) -> SourceRecord:
        """The first (today: the only) imagery source."""
        if not self.sources:
            return SourceRecord()
        return self.sources[0]

    @property
    def class_map(self) -> ClassMap:
        """Class name to ID; empty without a target or when every polygon is class 1."""
        return self.target.class_map if self.target is not None else {}

    @property
    def ignore_index(self) -> Optional[int]:
        """Mask value of pixels without imagery, or ``None``."""
        return self.target.ignore_index if self.target is not None else None

    @property
    def upgraded_from(self) -> Optional[int]:
        """The on-disk version (1 or 2) when this manifest was upgraded on load."""
        return self._upgraded_from

    @property
    def loaded_version(self) -> int:
        """The version the manifest had on disk (``version`` unless it was upgraded)."""
        return self._upgraded_from or self.version

    def patch_name(self, entry: ManifestEntry) -> str:
        """The name split lists use for a patch: the file name of its first source's image."""
        return posixpath.basename(entry["files"][self.source.name])

    def patch_transform(self, entry: ManifestEntry) -> Transform:
        """Affine transform of a patch's pixels, in the source CRS.

        Derived from the source transform and the entry's ``row``/``col``.
        """
        transform = self.source.transform
        if transform is None:
            raise ValueError("the manifest records no transform (made by mapcv 0.1)")
        a, b, c, d, e, f = transform
        row, col = entry["row"], entry["col"]
        return (a, b, c + a * col + b * row, d, e, f + d * col + e * row)

    def patch_bounds(self, entry: ManifestEntry) -> Tuple[float, float, float, float]:
        """``(left, bottom, right, top)`` of a patch in the source CRS."""
        size = (self.sampler or {}).get("patch_size")
        if size is None:
            raise ValueError("the manifest records no patch size (made by mapcv 0.1)")
        a, b, c, d, e, f = self.patch_transform(entry)
        xs = [c + a * x + b * y for x in (0, size) for y in (0, size)]
        ys = [f + d * x + e * y for x in (0, size) for y in (0, size)]
        return min(xs), min(ys), max(xs), max(ys)

    # ── reading and writing ──────────────────────────────────────────────────

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Manifest":
        """Validate a parsed manifest of any supported version, upgrading 1 and 2 to 3.

        Raises:
            ManifestMismatchError: The manifest was written by a newer mapcv.
        """
        version = data.get("version", 1)
        if not isinstance(version, int) or isinstance(version, bool) or version < 1:
            raise ManifestMismatchError(f"unknown manifest version {version!r}")
        if version > MANIFEST_VERSION:
            raise ManifestMismatchError(
                f"manifest version {version} was written by a newer mapcv; upgrade mapcv to read it"
            )
        if version == MANIFEST_VERSION:
            return cls.model_validate(data)
        manifest = cls.model_validate(_upgrade_v2(data))
        manifest._upgraded_from = version
        return manifest

    @classmethod
    def load(cls, path: Path) -> "Manifest":
        """Read a version-1, -2 or -3 manifest; older versions are upgraded in memory.

        The file is not modified. A version-1 or -2 manifest becomes version 3
        on disk only when :meth:`save` writes it, for example after a resumed
        run added patches.

        Raises:
            ManifestMismatchError: The manifest was written by a newer mapcv.
        """
        text = path.read_text(encoding="utf-8")
        if _CURRENT_VERSION_FIRST.match(text):
            # What mapcv writes: validate straight from JSON, the fast path for big manifests.
            return cls.model_validate_json(text)
        data = json.loads(text)
        if not isinstance(data, dict):
            raise ManifestMismatchError(f"{path} is not a mapcv manifest")
        try:
            return cls.from_dict(data)
        except ManifestMismatchError as exc:
            raise ManifestMismatchError(f"{path}: {exc}") from None

    def to_json(self) -> str:
        """The manifest as JSON: indented header, one line per patch."""
        header = self.model_dump(mode="json", exclude={"patches"})
        header["version"] = MANIFEST_VERSION
        text = json.dumps(header, indent=2, ensure_ascii=False)
        head = text[: text.rfind("}")].rstrip()
        rows = ",\n    ".join(to_json(entry).decode("utf-8") for entry in self.patches)
        body = f"\n    {rows}\n  " if rows else ""
        return f'{head},\n  "patches": [{body}]\n}}\n'

    def save(self, path: Path) -> None:
        """Atomically write the manifest (always as version 3)."""
        tmp_path = path.with_name(path.name + ".tmp")
        tmp_path.write_text(self.to_json(), encoding="utf-8", newline="\n")
        os.replace(tmp_path, path)


# ── upgrading versions 1 and 2 ───────────────────────────────────────────────


def _upgrade_entry(entry: Dict[str, Any], has_target: bool) -> Dict[str, Any]:
    files = {"image": f"{IMAGES_DIR}/{entry['filename']}"}
    if entry.get("mask_filename"):
        files["mask"] = f"{MASKS_DIR}/{entry['mask_filename']}"
    summary: Dict[str, Any] = {}
    if has_target:
        summary["class_pixels"] = dict(entry.get("per_class_pixel_counts") or {})
    summary["empty_ratio"] = entry.get("empty_ratio", 0.0)
    return {
        "row": entry["row"],
        "col": entry["col"],
        "padded": entry.get("padded", False),
        "chunk": entry.get("strip_index", 0),
        "files": files,
        "summary": summary,
    }


def _upgrade_v2(data: Dict[str, Any]) -> Dict[str, Any]:
    """A version-1 or -2 manifest in the version-3 shape.

    Version 2 recorded ``labels.ignore_index`` only from the 0.3 development
    line on; mapcv 0.2 had no ignore index (pixels without imagery were written
    as background), so an absent value upgrades to ``None``.
    """
    patches = data.get("patches") or []
    labels = data.get("labels")
    class_map = data.get("class_map") or {}
    has_masks = any(entry.get("mask_filename") for entry in patches)
    target: Optional[Dict[str, Any]] = None
    if labels is not None or has_masks or class_map:
        settings = dict(labels) if labels is not None else None
        ignore_index = settings.pop("ignore_index", None) if settings is not None else None
        target = {
            "type": "segmentation",
            "class_map": class_map,
            "ignore_index": ignore_index,
            "dtype": "uint8",
            "labels": settings,
            "options": {},
        }
    writer = data.get("writer")
    if writer is not None:
        writer = {"layout": "files", **writer, "mask_format": "png"}
        # mapcv 0.2 wrote JPEG patches with full-resolution chroma and had no setting for it.
        is_jpg = writer.get("image_format") == "jpg"
        writer.setdefault("jpg_subsampling", "4:4:4" if is_jpg else "4:2:0")
    source = {
        "name": "image",
        "source_type": data.get("source_type", "xyz"),
        "product_id": data.get("product_id"),
        "bands": data.get("bands") or [],
        "dtype": data.get("dtype"),
        "crs": data.get("crs"),
        "transform": data.get("transform"),
        "patch_shape": data.get("patch_shape") or [],
    }
    return {
        "version": MANIFEST_VERSION,
        "mapcv_version": None,
        "task": "segmentation",
        "sources": [source],
        "target": target,
        "writer": writer,
        "sampler": data.get("sampler"),
        "patches": [_upgrade_entry(entry, target is not None) for entry in patches],
    }


# ── resuming ─────────────────────────────────────────────────────────────────


def _transforms_differ(a: Optional[Transform], b: Optional[Transform]) -> bool:
    if a is None or b is None:
        return a != b
    return not all(math.isclose(x, y, rel_tol=1e-9, abs_tol=1e-9) for x, y in zip(a, b))


_SOURCE_FIELDS = (
    "source_type",
    "product_id",
    "bands",
    "dtype",
    "crs",
    "patch_shape",
    "fingerprint",
)
# Fields whose difference is easier to act on under another name.
_SOURCE_FIELD_LABELS = {"fingerprint": "imagery file or read settings"}
_TARGET_FIELDS = {
    "type": "target type",
    "class_map": "class_map",
    "ignore_index": "ignore_index",
    "dtype": "mask dtype",
    "labels": "labels",
    "options": "task options",
}


def _resume_mismatches(manifest: Manifest, expected: Manifest) -> List[str]:
    mismatches: List[str] = []
    if manifest.task != expected.task:
        mismatches.append("task")
    if [s.name for s in manifest.sources] != [s.name for s in expected.sources]:
        mismatches.append("sources")
    else:
        for have, want in zip(manifest.sources, expected.sources):
            prefix = f"{have.name}: " if len(expected.sources) > 1 else ""
            mismatches.extend(
                prefix + _SOURCE_FIELD_LABELS.get(name, name)
                for name in _SOURCE_FIELDS
                if getattr(have, name) != getattr(want, name)
            )
            if _transforms_differ(have.transform, want.transform):
                mismatches.append(prefix + "transform")
    if (manifest.target is None) != (expected.target is None):
        mismatches.append("labels")
    elif manifest.target is not None and expected.target is not None:
        mismatches.extend(
            label
            for name, label in _TARGET_FIELDS.items()
            if getattr(manifest.target, name) != getattr(expected.target, name)
        )
    if manifest.sampler != expected.sampler:
        mismatches.append("sampler")
    if manifest.writer != expected.writer:
        mismatches.append("writer")
    return mismatches


def load_or_create_manifest(path: Path, expected: Manifest) -> Manifest:
    """Load the manifest at ``path`` for resuming, or return ``expected`` when there is none.

    ``expected`` describes the run about to start (no patches). An existing
    manifest is resumed only when it records the same task, sources, target,
    sampler and writer. Version-2 manifests (mapcv 0.2) are compared after
    upgrading: mapcv 0.2 had no ignore index, so they resume only with
    ``labels.ignore_index: null``.

    Raises:
        ManifestMismatchError: The existing manifest is version 1, a version-2
            manifest of random sampling, or was generated with a different
            configuration (the message names what differs).
    """
    if not path.exists():
        return expected

    manifest = Manifest.load(path)
    if manifest.upgraded_from == 1:
        raise ManifestMismatchError(
            f"{path} is a version-1 manifest from mapcv 0.1.x and cannot be resumed; "
            "generate into a new writer.staging_dir ('mapcv split' still reads it)"
        )
    if manifest.upgraded_from == 2 and (manifest.sampler or {}).get("mode") == "random":
        raise ManifestMismatchError(
            f"{path} was made by mapcv 0.2 with sampler.mode 'random', whose anchors mapcv "
            "0.3 draws differently, so resuming would mix two samples; generate into a new "
            "writer.staging_dir ('mapcv split' still reads it)"
        )
    mismatches = _resume_mismatches(manifest, expected)
    if mismatches:
        hint = ""
        if (
            manifest.upgraded_from == 2
            and "ignore_index" in mismatches
            and manifest.ignore_index is None
        ):
            hint = (
                ". mapcv 0.2 wrote background where there is no imagery: set "
                "labels.ignore_index: null to resume this dataset"
            )
        raise ManifestMismatchError(
            f"{path} was generated with a different configuration "
            f"({', '.join(mismatches)}); use a new writer.staging_dir or remove the old "
            f"dataset{hint}"
        )
    return manifest


def patch_folders(manifest: Manifest) -> List[str]:
    """The top-level folders the patches' files live in, in first-seen order."""
    folders: Dict[str, None] = {}
    for entry in manifest.patches:
        for path in entry["files"].values():
            folders.setdefault(path.split("/", 1)[0], None)
    return list(folders)
