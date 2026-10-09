"""Dataset statistics (``mapcv stats``): what training needs to know about a dataset.

``stats.json`` holds, for each imagery source, the per-band mean, standard deviation,
minimum and maximum over the pixels that have imagery (for normalising inputs, as
TerraTorch and most training code expect), and the class balance of the task with
median-frequency class weights. By default only the ``train`` split is counted, so
validation and test data do not leak into the normalisation.

A pixel has imagery when the mask does not mark it as ignored (``ignore_index``) and,
without such a mask, when its bands are finite and not all the source's NoData value
(all zero for XYZ tiles, where failed tiles are black).
"""

from __future__ import annotations

import json
import math
from collections.abc import Sequence
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import numpy.typing as npt
from PIL import Image

from mapcv.manifest import Manifest, ManifestEntry, warn_if_incomplete

STATS_FILENAME = "stats.json"
_READ_THREADS = 8


@dataclass
class Moments:
    """Count, mean, sum of squared deviations, minimum and maximum of each band.

    Batches are merged with the parallel form of Welford's update (Chan et al.), which
    stays accurate over billions of pixels where a sum of squares would not.
    """

    bands: int
    count: npt.NDArray[np.int64] = field(init=False)
    mean: npt.NDArray[np.float64] = field(init=False)
    m2: npt.NDArray[np.float64] = field(init=False)
    low: npt.NDArray[np.float64] = field(init=False)
    high: npt.NDArray[np.float64] = field(init=False)

    def __post_init__(self) -> None:
        self.count = np.zeros(self.bands, dtype=np.int64)
        self.mean = np.zeros(self.bands, dtype=np.float64)
        self.m2 = np.zeros(self.bands, dtype=np.float64)
        self.low = np.full(self.bands, np.inf)
        self.high = np.full(self.bands, -np.inf)

    def add(self, values: npt.NDArray[Any], valid: npt.NDArray[np.bool_]) -> None:
        """Add the ``valid`` pixels of ``values`` (``(H, W, bands)``); NaN is never counted."""
        flat = values.reshape(-1, self.bands).astype(np.float64)
        keep = valid.reshape(-1)
        for band in range(self.bands):
            column = flat[keep, band]
            column = column[np.isfinite(column)]
            n = column.size
            if not n:
                continue
            mean = float(column.mean())
            m2 = float(((column - mean) ** 2).sum())
            total = self.count[band] + n
            delta = mean - self.mean[band]
            self.mean[band] += delta * n / total
            self.m2[band] += m2 + delta * delta * self.count[band] * n / total
            self.count[band] = total
            self.low[band] = min(self.low[band], float(column.min()))
            self.high[band] = max(self.high[band], float(column.max()))

    def summary(self, names: Sequence[str]) -> dict[str, Any]:
        """Per-band values as lists in band order (``None`` for a band without pixels)."""

        def values(array: npt.NDArray[np.float64]) -> list[float | None]:
            return [float(v) if c else None for v, c in zip(array, self.count)]

        std = np.sqrt(
            np.divide(self.m2, self.count, out=np.zeros_like(self.m2), where=self.count > 0)
        )
        return {
            "bands": list(names),
            "mean": values(self.mean),
            "std": values(std),
            "min": values(self.low),
            "max": values(self.high),
            "pixels": [int(c) for c in self.count],
        }


def _read_array(path: Path) -> tuple[npt.NDArray[Any], float | None]:
    """A patch file as ``(H, W, C)`` (or ``(T, C, H, W)`` for a stack) and its NoData."""
    suffix = path.suffix.lower()
    if suffix == ".npy":
        array: npt.NDArray[Any] = np.load(path, allow_pickle=False)
        if array.ndim == 2:
            return array[:, :, np.newaxis], None
        if array.ndim == 3:
            return np.moveaxis(array, 0, -1), None
        return array, None
    if suffix in (".tif", ".tiff"):
        from mapcv.geotiff import GeoTiff

        tif = GeoTiff(path)
        data, _ = tif.read_window(0, tif.info.height, 0, tif.info.width)
        return data, tif.info.nodata
    with Image.open(path) as image:
        loaded = np.asarray(image)
    return (loaded[:, :, np.newaxis] if loaded.ndim == 2 else loaded), None


def _valid_pixels(
    image: npt.NDArray[Any], nodata: float | None, black_is_empty: bool
) -> npt.NDArray[np.bool_]:
    valid = np.ones(image.shape[:2], dtype=np.bool_)
    if image.dtype.kind == "f":
        valid &= np.all(np.isfinite(image), axis=-1)
    if nodata is not None and not math.isnan(nodata):
        valid &= ~np.all(image == nodata, axis=-1)
    if black_is_empty:
        valid &= ~np.all(image == 0, axis=-1)
    return valid


def _split_entries(
    manifest: Manifest, staging_dir: Path, split: str
) -> tuple[list[ManifestEntry], str]:
    """The entries of ``split`` (``all`` for every patch); ``all`` when there are no splits."""
    if split == "all":
        return list(manifest.patches), "all"
    listed = staging_dir / "splits" / f"{split}.txt"
    if not listed.exists():
        return list(manifest.patches), "all"
    names = set(listed.read_text(encoding="utf-8").split())
    return [entry for entry in manifest.patches if manifest.patch_name(entry) in names], split


def _class_names(manifest: Manifest) -> dict[str, str]:
    """Class ID (as a string) to name; background and IDs without a name get one too."""
    names = {str(cid): name for name, cid in manifest.class_map.items()}
    names.setdefault("0", "background")
    return names


def _median_frequency_weights(
    pixels: dict[str, int], present_in: dict[str, int]
) -> dict[str, float]:
    """Eigen & Fergus: ``median(freq) / freq(c)``, ``freq(c)`` = the class's pixels over the
    valid pixels of the patches it appears in."""
    freq = {c: pixels[c] / present_in[c] for c in pixels if present_in.get(c) and pixels[c]}
    if not freq:
        return {}
    median = float(np.median(list(freq.values())))
    return {c: median / f for c, f in sorted(freq.items(), key=lambda item: int(item[0]))}


def _class_balance(manifest: Manifest, entries: list[ManifestEntry]) -> dict[str, Any]:
    names = _class_names(manifest)
    task = manifest.task
    ignore = manifest.ignore_index
    if task in ("detection", "instance"):
        objects: dict[str, int] = {}
        for entry in entries:
            for cid, count in (entry["summary"].get("class_objects") or {}).items():
                objects[cid] = objects.get(cid, 0) + int(count)
        return {
            "objects": {
                names.get(c, f"class_{c}"): n
                for c, n in sorted(objects.items(), key=lambda i: int(i[0]))
            }
        }
    if task == "classification":
        patches: dict[str, int] = {}
        for entry in entries:
            for label in entry["summary"].get("labels") or []:
                patches[str(label)] = patches.get(str(label), 0) + 1
        return {
            "patches": {
                names.get(c, f"class_{c}"): n
                for c, n in sorted(patches.items(), key=lambda i: int(i[0]))
            }
        }
    if task == "regression":
        return {}
    pixels: dict[str, int] = {}
    present_in: dict[str, int] = {}
    for entry in entries:
        counts = {
            cid: int(n)
            for cid, n in (entry["summary"].get("class_pixels") or {}).items()
            if ignore is None or cid != str(ignore)
        }
        valid = sum(counts.values())
        for cid, n in counts.items():
            pixels[cid] = pixels.get(cid, 0) + n
            if n:
                present_in[cid] = present_in.get(cid, 0) + valid
    if not pixels:
        return {}
    total = sum(pixels.values())
    ordered = sorted(pixels, key=int)
    weights = _median_frequency_weights(pixels, present_in)
    return {
        "pixels": {names.get(c, f"class_{c}"): pixels[c] for c in ordered},
        "frequency": {names.get(c, f"class_{c}"): pixels[c] / total for c in ordered},
        "median_frequency_weights": {names.get(c, f"class_{c}"): w for c, w in weights.items()},
    }


def dataset_stats(staging_dir: Path, split: str = "train") -> dict[str, Any]:
    """Statistics of the dataset in ``staging_dir`` over ``split`` (or ``all``).

    Raises:
        FileNotFoundError: The folder has no ``manifest.json``.
    """
    manifest_path = staging_dir / "manifest.json"
    if not manifest_path.exists():
        raise FileNotFoundError(f"No manifest found at {manifest_path}")
    manifest = Manifest.load(manifest_path)
    warn_if_incomplete(manifest, "these statistics cover only the patches written so far")
    entries, used = _split_entries(manifest, staging_dir, split)
    sources = manifest.sources
    stacked = bool((manifest.writer or {}).get("stack_sources"))
    ignore = manifest.ignore_index
    has_mask = any("mask" in entry["files"] for entry in entries) and manifest.task not in (
        "regression",
        "instance",
    )
    moments: dict[str, Moments] = {}
    targets = Moments(1) if manifest.task == "regression" else None

    def keys_for(entry: ManifestEntry) -> list[str]:
        if stacked:
            return ["image"]
        return [record.name if record.name in entry["files"] else "image" for record in sources]

    def read(rel: str) -> tuple[npt.NDArray[Any], float | None]:
        try:
            return _read_array(staging_dir / rel)
        except (OSError, ValueError) as exc:  # missing, truncated or not an image
            raise ValueError(
                f"{rel} cannot be read ({exc}); check the dataset with mapcv verify --deep"
            ) from exc

    def load(entry: ManifestEntry) -> dict[str, Any]:
        loaded: dict[str, Any] = {key: read(entry["files"][key]) for key in keys_for(entry)}
        if "mask" in entry["files"]:
            loaded["mask"] = read(entry["files"]["mask"])
        return loaded

    with ThreadPoolExecutor(max_workers=_READ_THREADS) as pool:
        for entry, loaded in zip(entries, pool.map(load, entries)):
            mask = loaded.get("mask")
            mask_valid = None
            if mask is not None and has_mask and ignore is not None:
                mask_valid = mask[0][:, :, 0] != ignore
            if targets is not None and mask is not None:
                values = mask[0][:, :, :1].astype(np.float64)
                targets.add(values, np.isfinite(values[:, :, 0]))
            if stacked:
                stack, _ = loaded["image"]
                # (T, C, H, W): source t holds channels t * C .. (t + 1) * C.
                per_source = [np.moveaxis(stack[index], 0, -1) for index in range(stack.shape[0])]
                images = [(record, image, None) for record, image in zip(sources, per_source)]
            else:
                images = [(record, *loaded[key]) for record, key in zip(sources, keys_for(entry))]
            for record, image, nodata in images:
                declared = (record.fingerprint or {}).get("nodata")
                if nodata is None and declared not in (None, "nan"):
                    nodata = float(declared)
                valid = (
                    mask_valid
                    if mask_valid is not None
                    else _valid_pixels(image, nodata, record.source_type == "xyz")
                )
                bands = image.shape[-1]
                state = moments.setdefault(record.name, Moments(bands))
                state.add(image, valid)

    stats: dict[str, Any] = {
        "mapcv_version": manifest.mapcv_version,
        "task": manifest.task,
        "split": used,
        "patches": len(entries),
        "sources": {},
    }
    for record in sources:
        found = moments.get(record.name)
        if found is None:
            continue
        state = found
        names = (
            record.bands
            if len(record.bands) == state.bands
            else [str(b + 1) for b in range(state.bands)]
        )
        stats["sources"][record.name] = state.summary(names)
    balance = _class_balance(manifest, entries)
    if balance:
        stats["classes"] = balance
    if targets is not None:
        summary = targets.summary(["target"])
        stats["targets"] = {key: value[0] for key, value in summary.items() if key != "bands"}
    return stats


def write_stats(staging_dir: Path, split: str = "train") -> tuple[Path, dict[str, Any]]:
    """Compute :func:`dataset_stats` and write it to ``stats.json`` in the dataset folder."""
    stats = dataset_stats(staging_dir, split)
    path = staging_dir / STATS_FILENAME
    path.write_text(json.dumps(stats, indent=2) + "\n", encoding="utf-8", newline="\n")
    return path, stats
