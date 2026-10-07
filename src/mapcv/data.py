"""Read a mapcv dataset for training: :class:`MapcvDataset`.

A map-style dataset (``len`` and indexing), so ``torch.utils.data.DataLoader`` takes it
directly; mapcv itself does not depend on torch. Each item is a dict:

``image``
    ``(C, H, W)`` (``(T, C, H, W)`` for stacked sources), as stored or normalised.
``images``
    Every source's image by name, for datasets with several sources.
``mask``
    ``(H, W)`` class IDs (segmentation, change) or ``float32`` values (regression).
``labels``
    Class IDs of the patch (classification), ``int64``.
``boxes`` / ``categories`` / ``annotations``
    ``(N, 4)`` ``[x, y, width, height]`` boxes in pixels, their class IDs and the COCO
    records (detection, instance; read from the COCO files of the split).
``name``, ``row``, ``col``, ``transform``, ``crs``
    Which patch it is and where: its file name, top-left pixel on the source grid, its
    affine transform and the CRS.

Arrays are numpy, or torch tensors when torch is installed and ``as_tensors`` is not
``False``.
"""

from __future__ import annotations

import importlib
import importlib.util
import json
import os
from pathlib import Path
from typing import Any, Callable, Dict, Iterator, List, Optional, Sequence, Union

import numpy as np
import numpy.typing as npt
from PIL import Image

from mapcv.manifest import Manifest, ManifestEntry

_SPLITS = ("train", "val", "test")


def _read(path: Path) -> npt.NDArray[Any]:
    """A patch file as ``(C, H, W)``, or as stored for NPY (``(H, W)`` masks, ``(T, C, H,
    W)`` stacks). PNG/JPEG via Pillow, GeoTIFF via mapcv's own reader (no GDAL)."""
    suffix = path.suffix.lower()
    if suffix == ".npy":
        array: npt.NDArray[Any] = np.load(path, allow_pickle=False)
        return array
    if suffix in (".tif", ".tiff"):
        from mapcv.geotiff import GeoTiff

        tif = GeoTiff(path)
        data, _ = tif.read_window(0, tif.info.height, 0, tif.info.width)
        return np.moveaxis(data, -1, 0)
    with Image.open(path) as image:
        loaded = np.array(image)  # writable, unlike np.asarray of a Pillow image
    return loaded[np.newaxis] if loaded.ndim == 2 else np.moveaxis(loaded, -1, 0)


def read_image(path: Path) -> npt.NDArray[Any]:
    """An image patch as ``(C, H, W)`` (``(T, C, H, W)`` for a stack of sources)."""
    array = _read(path)
    return array[np.newaxis] if array.ndim == 2 else array


def read_mask(path: Path) -> npt.NDArray[Any]:
    """A mask or target patch as ``(H, W)``."""
    array = _read(path)
    return array[0] if array.ndim == 3 else array


class MapcvDataset:
    """The patches of one split of a mapcv dataset (see the module docs).

    Args:
        root: The dataset folder (with ``manifest.json``).
        split: ``train``, ``val``, ``test``, a labeled subset such as ``10/labeled``, or
            ``all``. A dataset without split lists only has ``all``.
        normalize: Standardise each band with the train split's mean and standard
            deviation from ``stats.json`` (computed with :func:`mapcv.stats.dataset_stats`
            when the file is missing); the image becomes ``float32`` and pixels without
            data (``NaN``) become 0.
        transform: Called with each item dict; its return value is the item.
        as_tensors: ``True`` for torch tensors (torch must be installed), ``False`` for
            numpy, ``None`` (default) for tensors when torch is installed.

    Raises:
        FileNotFoundError: No manifest, or no list for ``split``.
    """

    def __init__(
        self,
        root: Union[str, "os.PathLike[str]"],
        split: str = "train",
        *,
        normalize: bool = False,
        transform: Optional[Callable[[Dict[str, Any]], Any]] = None,
        as_tensors: Optional[bool] = None,
    ) -> None:
        self.root = Path(root)
        manifest_path = self.root / "manifest.json"
        if not manifest_path.exists():
            raise FileNotFoundError(f"No manifest found at {manifest_path}")
        self.manifest = Manifest.load(manifest_path)
        self.split = split
        # A Zarr export (mapcv export --format zarr) holds the patches as arrays.
        self._zarr: Any = None
        if (self.root / ".zgroup").exists():
            import zarr

            self._zarr = zarr.open_group(str(self.root), mode="r")
        self._positions = {id(entry): index for index, entry in enumerate(self.manifest.patches)}
        self.entries = self._entries(split)
        self.transform = transform
        torch_available = importlib.util.find_spec("torch") is not None
        if as_tensors and not torch_available:
            raise ImportError("as_tensors=True needs PyTorch: pip install torch")
        self.as_tensors = torch_available if as_tensors is None else as_tensors
        self._stats: Optional[Dict[str, Any]] = self._load_stats() if normalize else None
        self._objects = (
            self._load_objects() if self.manifest.task in ("detection", "instance") else {}
        )

    # ── which patches ────────────────────────────────────────────────────────

    def _entries(self, split: str) -> List[ManifestEntry]:
        if split == "all":
            return list(self.manifest.patches)
        listed = self.root / "splits" / f"{split}.txt"
        if not listed.exists():
            available = (
                sorted(
                    p.relative_to(self.root / "splits").with_suffix("").as_posix()
                    for p in (self.root / "splits").rglob("*.txt")
                )
                if (self.root / "splits").is_dir()
                else []
            )
            raise FileNotFoundError(
                f"No split list {listed}; available: {', '.join(available + ['all'])}"
            )
        names = listed.read_text(encoding="utf-8").split()
        by_name = {self.manifest.patch_name(entry): entry for entry in self.manifest.patches}
        return [by_name[name] for name in names if name in by_name]

    @property
    def classes(self) -> Dict[int, str]:
        """Class ID to name (``0`` is background for masks)."""
        return {cid: name for name, cid in self.manifest.class_map.items()}

    @property
    def ignore_index(self) -> Optional[int]:
        """The mask value of pixels to leave out of the loss, or ``None``."""
        return self.manifest.ignore_index

    # ── side data ────────────────────────────────────────────────────────────

    def _load_stats(self) -> Dict[str, Any]:
        path = self.root / "stats.json"
        if path.exists():
            stats: Dict[str, Any] = json.loads(path.read_text(encoding="utf-8"))
            return stats
        from mapcv.stats import dataset_stats

        return dataset_stats(self.root, "train")

    def _load_objects(self) -> Dict[str, List[Dict[str, Any]]]:
        """COCO annotations by image file name, from the COCO files of the dataset."""
        folder = self.root / "annotations"
        documents = sorted(folder.glob("instances_*.json")) if folder.is_dir() else []
        if not documents:
            raise FileNotFoundError(
                f"No COCO files in {folder}: MapcvDataset reads boxes from them, so write "
                "the dataset with task_options formats including coco"
            )
        objects: Dict[str, List[Dict[str, Any]]] = {}
        for document in documents:
            coco = json.loads(document.read_text(encoding="utf-8"))
            names = {image["id"]: image["file_name"] for image in coco["images"]}
            for name in names.values():
                objects.setdefault(name, [])
            for annotation in coco["annotations"]:
                objects[names[annotation["image_id"]]].append(annotation)
        return objects

    # ── items ────────────────────────────────────────────────────────────────

    def __len__(self) -> int:
        return len(self.entries)

    def _normalised(self, source: str, image: npt.NDArray[Any]) -> npt.NDArray[np.float32]:
        assert self._stats is not None
        values = self._stats["sources"].get(source)
        if values is None:
            raise KeyError(f"stats.json has no statistics for source '{source}'")
        mean = np.array([m if m is not None else 0.0 for m in values["mean"]], dtype=np.float32)
        std = np.array([s if s else 1.0 for s in values["std"]], dtype=np.float32)
        shape = (-1,) + (1,) * 2  # broadcast over (C, H, W) and (T, C, H, W)
        result = (image.astype(np.float32) - mean.reshape(shape)) / std.reshape(shape)
        return np.nan_to_num(result, nan=0.0).astype(np.float32)

    def _image(self, key: str, entry: ManifestEntry) -> npt.NDArray[Any]:
        if self._zarr is not None:
            return np.asarray(self._zarr["images"][key][self._positions[id(entry)]])
        return read_image(self.root / entry["files"][key])

    def _mask(self, entry: ManifestEntry) -> npt.NDArray[Any]:
        if self._zarr is not None:
            return np.asarray(self._zarr["masks"][self._positions[id(entry)]])
        return read_mask(self.root / entry["files"]["mask"])

    def __getitem__(self, index: int) -> Any:
        entry = self.entries[index]
        files = entry["files"]
        manifest = self.manifest
        names = [record.name for record in manifest.sources]
        item: Dict[str, Any] = {
            "name": manifest.patch_name(entry),
            "row": entry["row"],
            "col": entry["col"],
            "transform": manifest.patch_transform(entry),
            "crs": manifest.source.crs,
        }
        stacked = bool((manifest.writer or {}).get("stack_sources"))
        if stacked:
            image = self._image("image", entry)
            if self._stats is not None:
                image = np.stack([self._normalised(name, image[t]) for t, name in enumerate(names)])
            item["image"] = image
        else:
            images = {}
            for name in names:
                key = name if name in files else "image"
                image = self._image(key, entry)
                images[name] = self._normalised(name, image) if self._stats is not None else image
            item["image"] = images[names[0]]
            if len(images) > 1:
                item["images"] = images
        if "mask" in files:
            item["mask"] = self._mask(entry)
        task = manifest.task
        if task == "classification":
            labels = entry["summary"].get("labels") or []
            item["labels"] = np.array([int(label) for label in labels], dtype=np.int64)
        elif task in ("detection", "instance"):
            annotations = self._objects.get(item["name"], [])
            item["annotations"] = annotations
            item["boxes"] = np.array([a["bbox"] for a in annotations], dtype=np.float32).reshape(
                -1, 4
            )
            item["categories"] = np.array([a["category_id"] for a in annotations], dtype=np.int64)
        if self.as_tensors:
            item = _to_tensors(item)
        return self.transform(item) if self.transform is not None else item

    def __iter__(self) -> Iterator[Any]:
        return (self[index] for index in range(len(self)))

    def __repr__(self) -> str:
        return f"MapcvDataset({str(self.root)!r}, split={self.split!r}, {len(self)} patches)"


def _to_tensors(item: Dict[str, Any]) -> Dict[str, Any]:
    torch: Any = importlib.import_module("torch")

    def convert(value: Any) -> Any:
        if isinstance(value, np.ndarray):
            if value.dtype == np.uint16:  # torch has no general uint16 support
                value = value.astype(np.int32)
            value = np.ascontiguousarray(value)
            # torch warns about (and must not write to) read-only arrays.
            return torch.from_numpy(value if value.flags.writeable else value.copy())
        if isinstance(value, dict):
            return {key: convert(each) for key, each in value.items()}
        return value

    return {
        key: (value if key in ("annotations",) else convert(value)) for key, value in item.items()
    }


def splits_of(root: Union[str, "os.PathLike[str]"]) -> Sequence[str]:
    """The split lists a dataset has (``train``, ``val``, ``test``), or ``("all",)``."""
    folder = Path(root) / "splits"
    present = [split for split in _SPLITS if (folder / f"{split}.txt").exists()]
    return present or ("all",)
