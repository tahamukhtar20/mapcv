"""Dataset checks (``mapcv verify``): is everything the manifest lists there and intact?

The manifest lists every patch's files and the split lists name patches. ``verify``
checks that ``generate`` finished the dataset (the manifest's ``complete``), that each
listed file exists, is not empty and is not cut short (a PNG ends with its ``IEND``
chunk, a JPEG with its end marker, an NPY file holds as many bytes as its header
says, a TIFF starts with a TIFF header), that split lists only name patches of the
manifest, do not share patches and match ``splits/split.json``, and reports files in
the patch folders that no patch lists (left by an interrupted run). With ``deep`` it
also decodes each image and checks its shape against the source's ``patch_shape``,
and decodes each mask, checking its size and, when the manifest records them, that its
pixel count per class is the recorded one (so a mask with another class, or values
that are no class, is found). ``SHA256SUMS`` (written by :func:`write_checksums`) pins
the contents of every file in the dataset folder; when it exists, ``verify`` checks
each hash, so a copied or downloaded dataset can be proven complete and unchanged.
"""

from __future__ import annotations

import fnmatch
import hashlib
import json
import math
from dataclasses import dataclass, field
from itertools import combinations
from pathlib import Path
from typing import Any

import numpy as np

from mapcv.locking import LOCK_FILENAME
from mapcv.manifest import Manifest, ManifestEntry, ManifestMismatchError, patch_folders

CHECKSUMS_FILENAME = "SHA256SUMS"
_SPLITS = ("train", "val", "test")
# Files that mapcv's own commands rewrite after generate: split lists and the outputs that
# follow the split (mapcv split), stats.json (mapcv stats) and README.md (mapcv card).
_REWRITTEN = (
    "splits/*",
    "patches.geojson",
    "stats.json",
    "README.md",
    "annotations/instances_*.json",
    "dataset.yaml",
    "train.txt",
    "val.txt",
    "test.txt",
    "labels.csv",
    "labels_*.csv",
)

_PNG_SIGNATURE = b"\x89PNG\r\n\x1a\n"
_PNG_END = b"\x00\x00\x00\x00IEND\xaeB`\x82"
_TIFF_SIGNATURES = (b"II*\x00", b"MM\x00*", b"II+\x00", b"MM\x00+")


@dataclass
class VerifyReport:
    """What :func:`verify_dataset` found: problems make a dataset unusable, notes do not.

    ``incomplete`` says the manifest records an unfinished ``generate``; ``rewritten``
    lists the files that changed since ``SHA256SUMS`` was written and that mapcv's own
    commands (``split``, ``stats``, ``card``) rewrite. Both are among the problems too.
    """

    patches: int = 0
    files: int = 0
    checked_hashes: int = 0
    problems: list[str] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)
    incomplete: bool = False
    rewritten: list[str] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return not self.problems


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _listed_files(manifest: Manifest) -> list[str]:
    """Every file the manifest lists, in patch order, without duplicates."""
    seen: dict[str, None] = {}
    for entry in manifest.patches:
        for path in entry["files"].values():
            seen.setdefault(path, None)
    return list(seen)


def _dataset_files(staging_dir: Path, manifest: Manifest) -> list[str]:
    """The files a checksum list covers: every file of the dataset folder, the patches in
    manifest order first, then the rest (manifest, split lists, annotations, label tables,
    footprints, ...) sorted. Unfinished ``.tmp`` files, a running command's lock and the
    checksum list itself are left out."""
    listed = _listed_files(manifest)
    known = set(listed)
    rest = sorted(
        rel
        for path in staging_dir.rglob("*")
        if path.is_file()
        and path.suffix != ".tmp"
        and (rel := path.relative_to(staging_dir).as_posix())
        not in (CHECKSUMS_FILENAME, LOCK_FILENAME)
        and rel not in known
    )
    return listed + rest


def write_checksums(staging_dir: Path) -> Path:
    """Write ``SHA256SUMS`` (``<hash>  <path>`` per line, as ``sha256sum`` writes it)."""
    manifest = Manifest.load(staging_dir / "manifest.json")
    lines = [
        f"{_sha256(staging_dir / rel)}  {rel}" for rel in _dataset_files(staging_dir, manifest)
    ]
    path = staging_dir / CHECKSUMS_FILENAME
    path.write_text("\n".join(lines) + "\n", encoding="utf-8", newline="\n")
    return path


def _cut_short(path: Path, size: int) -> str | None:
    """Why a patch file is not whole, read from its first and last bytes; ``None`` if it
    looks whole (or is of a kind not checked)."""
    suffix = path.suffix.lower()
    if suffix not in (".png", ".jpg", ".jpeg", ".npy", ".tif", ".tiff"):
        return None
    with path.open("rb") as handle:
        head = handle.read(4096)
        handle.seek(max(0, size - len(_PNG_END)))
        tail = handle.read()
    if suffix == ".png":
        if not head.startswith(_PNG_SIGNATURE):
            return "is not a PNG file"
        return None if tail.endswith(_PNG_END) else "is cut short (no PNG end chunk)"
    if suffix in (".jpg", ".jpeg"):
        if not head.startswith(b"\xff\xd8"):
            return "is not a JPEG file"
        return None if tail.endswith(b"\xff\xd9") else "is cut short (no JPEG end marker)"
    if suffix in (".tif", ".tiff"):
        return None if head[:4] in _TIFF_SIGNATURES else "is not a TIFF file"
    return _npy_cut_short(path, size)


def _npy_cut_short(path: Path, size: int) -> str | None:
    try:
        with path.open("rb") as handle:
            version = np.lib.format.read_magic(handle)
            if version == (1, 0):
                shape, _, dtype = np.lib.format.read_array_header_1_0(handle)
            else:
                shape, _, dtype = np.lib.format.read_array_header_2_0(handle)
            start = handle.tell()
    except (ValueError, OSError, SyntaxError):
        return "is not an NPY file"
    expected = start + math.prod(shape) * dtype.itemsize
    if size < expected:
        return f"is cut short ({size:,} of {expected:,} bytes)"
    return None


def _check_image(path: Path, shape: list[int]) -> str | None:
    """``None`` if the file decodes to ``shape`` (the source's ``patch_shape``)."""
    from mapcv.stats import _read_array

    try:
        array, _ = _read_array(path)
    except Exception as exc:  # noqa: BLE001 - any decoder error means a broken file
        return f"cannot be read ({exc})"
    if len(shape) != 3 or array.ndim != 3:
        return None
    # patch_shape is [H, W, C] for PNG/JPG and [C, H, W] for NPY and GeoTIFF; read back,
    # every format is (H, W, C).
    if path.suffix.lower() in (".png", ".jpg", ".jpeg"):
        want = (shape[0], shape[1], shape[2])
    else:
        want = (shape[1], shape[2], shape[0])
    if tuple(array.shape) != want:
        return f"has shape {tuple(array.shape)} (H, W, C), the manifest says {want}"
    return None


def _check_mask(path: Path, entry: ManifestEntry, patch_size: int | None) -> str | None:
    """``None`` if the mask decodes to ``patch_size`` x ``patch_size`` and holds the pixel
    count per class the manifest records for it (when it records one)."""
    from mapcv.stats import _read_array

    try:
        array, _ = _read_array(path)
    except Exception as exc:  # noqa: BLE001 - any decoder error means a broken file
        return f"cannot be read ({exc})"
    if array.ndim != 3 or array.shape[2] != 1:
        return f"has shape {tuple(array.shape)} (H, W, C); a mask has one band"
    if patch_size is not None and array.shape[:2] != (patch_size, patch_size):
        return (
            f"is {array.shape[0]}x{array.shape[1]} pixels, the manifest says "
            f"{patch_size}x{patch_size}"
        )
    recorded: dict[str, Any] | None = entry["summary"].get("class_pixels")
    if recorded is None or array.dtype.kind not in "biu":
        return None
    values, counts = np.unique(array, return_counts=True)
    found = {str(int(value)): int(count) for value, count in zip(values, counts)}
    if found != {key: int(count) for key, count in recorded.items()}:
        unknown = sorted(set(found) - set(recorded), key=int)
        if unknown:
            return (
                f"holds value(s) {', '.join(unknown[:5])} that the manifest records for no "
                "pixel of this patch (not a class of the dataset, or changed)"
            )
        return "has other pixel counts per class than the manifest records"
    return None


def _check_splits(staging_dir: Path, manifest: Manifest, report: VerifyReport) -> None:
    """Split lists: only patches of the manifest, no patch in two splits, as many patches
    as ``split.json`` says (it is written last, so it is missing after an interrupted
    ``mapcv split``)."""
    splits_dir = staging_dir / "splits"
    names = {manifest.patch_name(entry) for entry in manifest.patches}
    lists: dict[str, list[str]] = {}
    for split in _SPLITS:
        split_file = splits_dir / f"{split}.txt"
        if not split_file.exists():
            continue
        lists[split] = split_file.read_text(encoding="utf-8").split()
        unknown = [n for n in lists[split] if n not in names]
        if unknown:
            report.problems.append(
                f"splits/{split}.txt names {len(unknown):,} "
                f"{'patch' if len(unknown) == 1 else 'patches'} the manifest does not "
                f"list (first: {unknown[0]}); re-split with mapcv split"
            )
    if not lists:
        return
    for first, second in combinations(lists, 2):
        shared = set(lists[first]) & set(lists[second])
        if shared:
            report.problems.append(
                f"{len(shared):,} {'patch is' if len(shared) == 1 else 'patches are'} in both "
                f"splits/{first}.txt and splits/{second}.txt (first: {min(shared)}); "
                "re-split with mapcv split"
            )
    record_path = splits_dir / "split.json"
    if not record_path.exists():
        if manifest.loaded_version >= 2:  # mapcv 0.1 wrote no split.json
            report.problems.append(
                "splits/split.json is missing, so the split lists may be unfinished (an "
                "interrupted mapcv split); re-split with mapcv split"
            )
        return
    try:
        record = json.loads(record_path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        report.problems.append("splits/split.json cannot be read; re-split with mapcv split")
        return
    for split, members in lists.items():
        recorded = record.get(split) if isinstance(record, dict) else None
        if isinstance(recorded, int) and recorded != len(members):
            report.problems.append(
                f"splits/{split}.txt lists {len(members):,} patches, splits/split.json says "
                f"{recorded:,}; re-split with mapcv split"
            )


def verify_dataset(staging_dir: Path, deep: bool = False) -> VerifyReport:
    """Check the dataset in ``staging_dir``; see the module docs."""
    report = VerifyReport()
    manifest_path = staging_dir / "manifest.json"
    if not manifest_path.exists():
        report.problems.append(f"no manifest.json in {staging_dir}")
        return report
    try:
        manifest = Manifest.load(manifest_path)
    except (ManifestMismatchError, ValueError) as exc:
        report.problems.append(f"manifest.json cannot be read: {exc}")
        return report
    report.patches = len(manifest.patches)
    if manifest.complete is False:
        report.incomplete = True
        report.problems.append(
            f"the dataset is incomplete: mapcv generate stopped before it finished "
            f"({report.patches:,} patches so far); run mapcv generate again with the same "
            "config to finish it"
        )
    shapes = {record.name: record.patch_shape for record in manifest.sources}
    stacked = bool((manifest.writer or {}).get("stack_sources"))
    patch_size = (manifest.sampler or {}).get("patch_size")
    listed = _listed_files(manifest)
    report.files = len(listed)
    for entry in manifest.patches:
        for key, rel in entry["files"].items():
            path = staging_dir / rel
            if not path.is_file():
                report.problems.append(f"{rel} is missing")
                continue
            size = path.stat().st_size
            if size == 0:
                report.problems.append(f"{rel} is empty")
                continue
            problem = _cut_short(path, size)
            if problem is None and deep:
                if key == "mask":
                    problem = _check_mask(path, entry, patch_size)
                elif key in shapes and not stacked:
                    problem = _check_image(path, shapes[key])
            if problem is not None:
                report.problems.append(f"{rel} {problem}")

    _check_splits(staging_dir, manifest, report)

    referenced: set[str] = set(listed)
    orphans = sorted(
        path.relative_to(staging_dir).as_posix()
        for folder in patch_folders(manifest)
        if (staging_dir / folder).is_dir()
        for path in (staging_dir / folder).rglob("*")
        if path.is_file() and path.relative_to(staging_dir).as_posix() not in referenced
    )
    if orphans:
        report.notes.append(
            f"{len(orphans):,} {'file' if len(orphans) == 1 else 'files'} in the patch folders "
            f"{'is' if len(orphans) == 1 else 'are'} not in the manifest (left by an "
            f"interrupted run or added by hand; first: {orphans[0]})"
        )

    checksums = staging_dir / CHECKSUMS_FILENAME
    if checksums.exists():
        for line in checksums.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            expected, _, rel = line.partition("  ")
            path = staging_dir / rel
            if not path.is_file():
                report.problems.append(f"{rel} is listed in {CHECKSUMS_FILENAME} but missing")
            elif _sha256(path) != expected:
                if any(fnmatch.fnmatchcase(rel, pattern) for pattern in _REWRITTEN):
                    report.rewritten.append(rel)
                    report.problems.append(
                        f"{rel} changed since {CHECKSUMS_FILENAME} was written (mapcv split, "
                        "stats and card rewrite it)"
                    )
                else:
                    report.problems.append(f"{rel} does not match its {CHECKSUMS_FILENAME} hash")
            report.checked_hashes += 1
    return report
