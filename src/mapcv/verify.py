"""Dataset checks (``mapcv verify``): is everything the manifest lists there and intact?

The manifest lists every patch's files and the split lists name patches. ``verify``
checks that each listed file exists and is not empty, that split lists only name
patches of the manifest, and reports files in the patch folders that no patch lists
(left by an interrupted run). With ``deep`` it also decodes each image and checks its
shape against the source's ``patch_shape``. ``SHA256SUMS`` (written by
:func:`write_checksums`) pins the contents of every file in the dataset folder; when it exists, ``verify`` checks
each hash, so a copied or downloaded dataset can be proven complete and unchanged.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from pathlib import Path

from mapcv.manifest import Manifest, ManifestMismatchError, patch_folders

CHECKSUMS_FILENAME = "SHA256SUMS"
_SPLITS = ("train", "val", "test")


@dataclass
class VerifyReport:
    """What :func:`verify_dataset` found: problems make a dataset unusable, notes do not."""

    patches: int = 0
    files: int = 0
    checked_hashes: int = 0
    problems: list[str] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)

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
    footprints, ...) sorted. Unfinished ``.tmp`` files and the checksum list itself are left
    out."""
    listed = _listed_files(manifest)
    known = set(listed)
    rest = sorted(
        rel
        for path in staging_dir.rglob("*")
        if path.is_file()
        and path.suffix != ".tmp"
        and (rel := path.relative_to(staging_dir).as_posix()) != CHECKSUMS_FILENAME
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
    shapes = {record.name: record.patch_shape for record in manifest.sources}
    listed = _listed_files(manifest)
    report.files = len(listed)
    for entry in manifest.patches:
        for key, rel in entry["files"].items():
            path = staging_dir / rel
            if not path.is_file():
                report.problems.append(f"{rel} is missing")
            elif path.stat().st_size == 0:
                report.problems.append(f"{rel} is empty")
            elif deep and key in shapes and not (manifest.writer or {}).get("stack_sources"):
                problem = _check_image(path, shapes[key])
                if problem is not None:
                    report.problems.append(f"{rel} {problem}")

    names = {manifest.patch_name(entry) for entry in manifest.patches}
    for split in _SPLITS:
        split_file = staging_dir / "splits" / f"{split}.txt"
        if split_file.exists():
            unknown = [n for n in split_file.read_text(encoding="utf-8").split() if n not in names]
            if unknown:
                report.problems.append(
                    f"splits/{split}.txt names {len(unknown)} patch(es) the manifest does not "
                    f"list (first: {unknown[0]}); re-split with mapcv split"
                )

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
            f"{len(orphans)} file(s) in the patch folders are not in the manifest (left by an "
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
                report.problems.append(f"{rel} does not match its {CHECKSUMS_FILENAME} hash")
            report.checked_hashes += 1
    return report
