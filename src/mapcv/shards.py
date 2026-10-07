"""Pack a dataset into few large files: WebDataset tar shards or one Zarr store.

A dataset of hundreds of thousands of patch files is slow to copy, list and read from
network or object storage. These exports (``mapcv export --format webdataset|zarr``)
hold the same bytes and arrays in a handful of files; the dataset folder stays as it
is, so every other command keeps working on it. Re-split the dataset with
``mapcv split`` and export again to change split membership.

``webdataset``
    ``<out>/<split>-000000.tar``, ... of at most ``shard_bytes`` each (one sample is
    never split). A sample is a patch: ``<name>.<ext>`` (the image file as written),
    ``<name>.mask.<ext>``, ``<name>.<source>.<ext>`` for further sources, and
    ``<name>.json`` (row, col, transform, CRS, split and the manifest summary). Tars
    are deterministic: members in a fixed order, owner and times zeroed.
    ``<out>/shards.json`` lists the shards of each split and their sample counts.
``zarr``
    ``<out>`` is a Zarr group (zarr 2 format, the ``zarr`` extra): ``images/<source>``
    ``(N, C, H, W)`` (``(N, T, C, H, W)`` for stacks), ``masks`` ``(N, H, W)``,
    ``split`` (``uint8``: 0 train, 1 val, 2 test, 255 none), ``row``/``col``, each chunked
    one patch per chunk; a copy of ``manifest.json`` sits next to the arrays (patch
    ``i`` of the arrays is entry ``i``). Random access by index, for NPY-style
    workflows; :class:`mapcv.data.MapcvDataset` reads it like a dataset folder.
"""

from __future__ import annotations

import io
import json
import tarfile
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

from mapcv.data import read_image, read_mask, splits_of
from mapcv.manifest import Manifest, ManifestEntry

SHARD_BYTES = 1_000_000_000
SPLIT_CODES = {"train": 0, "val": 1, "test": 2}
NO_SPLIT = 255


def _load(root: Path) -> Manifest:
    path = root / "manifest.json"
    if not path.exists():
        raise FileNotFoundError(f"No manifest found at {path}")
    return Manifest.load(path)


def _split_lists(root: Path, manifest: Manifest) -> Dict[str, List[ManifestEntry]]:
    """Each split's entries in its list's order (``all``, in manifest order, without lists)."""
    splits = splits_of(root)
    if splits == ("all",):
        return {"all": list(manifest.patches)}
    by_name = {manifest.patch_name(entry): entry for entry in manifest.patches}
    return {
        split: [
            by_name[name]
            for name in (root / "splits" / f"{split}.txt").read_text(encoding="utf-8").split()
            if name in by_name
        ]
        for split in splits
    }


def _check_out(root: Path, out: Path) -> None:
    if out.resolve() == root.resolve() or root.resolve() in out.resolve().parents:
        raise ValueError("export to a folder outside the dataset folder")


def _coco_by_image(root: Path) -> Dict[str, List[Dict[str, Any]]]:
    """Detection and instance annotations by image file name, from the COCO files."""
    found: Dict[str, List[Dict[str, Any]]] = {}
    for path in sorted((root / "annotations").glob("instances_*.json")):
        coco = json.loads(path.read_text(encoding="utf-8"))
        names = {image["id"]: image["file_name"] for image in coco["images"]}
        for annotation in coco["annotations"]:
            found.setdefault(names[annotation["image_id"]], []).append(annotation)
    return found


def _sample_json(
    manifest: Manifest,
    entry: ManifestEntry,
    split: str,
    objects: Optional[Dict[str, List[Dict[str, Any]]]] = None,
) -> bytes:
    record: Dict[str, Any] = {
        "name": manifest.patch_name(entry),
        "row": entry["row"],
        "col": entry["col"],
        "padded": entry["padded"],
        "transform": list(manifest.patch_transform(entry)),
        "crs": manifest.source.crs,
        "split": split,
        "summary": entry["summary"],
    }
    if objects is not None:
        record["annotations"] = objects.get(manifest.patch_name(entry), [])
    return json.dumps(record, sort_keys=True).encode("utf-8")


def _members(
    manifest: Manifest,
    entry: ManifestEntry,
    root: Path,
    split: str,
    objects: Optional[Dict[str, List[Dict[str, Any]]]],
) -> List[tuple[str, bytes]]:
    stem = Path(manifest.patch_name(entry)).stem
    first = manifest.sources[0].name
    members = []
    for key, rel in entry["files"].items():
        suffix = Path(rel).suffix.lstrip(".")
        if key in ("image", first):
            name = f"{stem}.{suffix}"
        else:
            name = f"{stem}.{key}.{suffix}"
        members.append((name, (root / rel).read_bytes()))
    members.append((f"{stem}.json", _sample_json(manifest, entry, split, objects)))
    return members


def _add(tar: tarfile.TarFile, name: str, data: bytes) -> None:
    info = tarfile.TarInfo(name)
    info.size = len(data)
    info.mtime = 0
    info.mode = 0o644
    info.uid = info.gid = 0
    info.uname = info.gname = ""
    tar.addfile(info, io.BytesIO(data))


def export_webdataset(root: Path, out: Path, shard_bytes: int = SHARD_BYTES) -> List[Path]:
    """Write the tar shards and ``shards.json``; returns the shards (see the module docs)."""
    if shard_bytes < 1:
        raise ValueError("shard_bytes must be positive")
    _check_out(root, out)
    manifest = _load(root)
    objects = _coco_by_image(root) if manifest.task in ("detection", "instance") else None
    out.mkdir(parents=True, exist_ok=True)
    index: Dict[str, Any] = {
        "splits": {},
        "manifest_version": manifest.version,
        "task": manifest.task,
    }
    written: List[Path] = []
    for split, entries in _split_lists(root, manifest).items():
        shards: List[Dict[str, Any]] = []
        tar: Optional[tarfile.TarFile] = None
        size = 0
        for entry in entries:
            members = _members(manifest, entry, root, split, objects)
            sample_size = sum(512 + -(-len(data) // 512) * 512 for _, data in members)
            if tar is None or (size and size + sample_size > shard_bytes):
                if tar is not None:
                    tar.close()
                path = out / f"{split}-{len(shards):06d}.tar"
                tar = tarfile.open(path, "w", format=tarfile.USTAR_FORMAT)
                shards.append({"file": path.name, "samples": 0})
                written.append(path)
                size = 0
            for name, data in members:
                _add(tar, name, data)
            size += sample_size
            shards[-1]["samples"] += 1
        if tar is not None:
            tar.close()
        index["splits"][split] = shards
    (out / "shards.json").write_text(json.dumps(index, indent=2) + "\n", encoding="utf-8")
    return written


def _zarr() -> Any:
    try:
        import zarr
    except ImportError as exc:
        raise RuntimeError(
            "the zarr export needs zarr: pip install 'mapcv[zarr]' (Python 3.10-3.13)"
        ) from exc
    return zarr


def export_zarr(root: Path, out: Path) -> Path:
    """Write the Zarr group (see the module docs); returns ``out``."""
    zarr = _zarr()
    _check_out(root, out)
    manifest = _load(root)
    split_of = {
        manifest.patch_name(entry): split
        for split, members in _split_lists(root, manifest).items()
        for entry in members
    }
    entries = list(manifest.patches)
    if not entries:
        raise ValueError("the dataset has no patches")
    group = zarr.open_group(str(out), mode="w")
    count = len(entries)
    first = entries[0]["files"]
    image_keys = [key for key in first if key != "mask"]
    images = group.require_group("images")
    for key in image_keys:
        sample = read_image(root / first[key])
        array = images.create_dataset(
            key, shape=(count, *sample.shape), chunks=(1, *sample.shape), dtype=sample.dtype
        )
        for index, entry in enumerate(entries):
            array[index] = read_image(root / entry["files"][key])
    if "mask" in first:
        sample = read_mask(root / first["mask"])
        masks = group.create_dataset(
            "masks", shape=(count, *sample.shape), chunks=(1, *sample.shape), dtype=sample.dtype
        )
        for index, entry in enumerate(entries):
            masks[index] = read_mask(root / entry["files"]["mask"])
    codes = [SPLIT_CODES.get(split_of.get(manifest.patch_name(e), ""), NO_SPLIT) for e in entries]
    group.create_dataset("split", data=np.array(codes, dtype=np.uint8), chunks=(count,))
    group.create_dataset(
        "row", data=np.array([e["row"] for e in entries], dtype=np.int64), chunks=(count,)
    )
    group.create_dataset(
        "col", data=np.array([e["col"] for e in entries], dtype=np.int64), chunks=(count,)
    )
    group.attrs["mapcv"] = {"split_codes": SPLIT_CODES, "image_keys": image_keys}
    # The manifest goes next to the arrays as a file (Zarr ignores it), not into the
    # attributes, which are read whole whenever the group is opened.
    (out / "manifest.json").write_bytes((root / "manifest.json").read_bytes())
    # The split lists too, so MapcvDataset reads the store's splits (and labeled
    # subsets) exactly as the dataset folder's.
    splits = root / "splits"
    for listed in sorted(splits.rglob("*")) if splits.is_dir() else []:
        if listed.is_file():
            target = out / listed.relative_to(root)
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(listed.read_bytes())
    # Boxes and instances stay in their COCO files, which MapcvDataset reads.
    for coco in sorted((root / "annotations").glob("instances_*.json")):
        (out / "annotations").mkdir(exist_ok=True)
        (out / "annotations" / coco.name).write_bytes(coco.read_bytes())
    return out
