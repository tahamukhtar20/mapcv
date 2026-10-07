"""A small, reusable dataset class for mapcv output.

Copy this file into your project. It reads a mapcv dataset directory::

    dataset/
      manifest.json        version 3: task, sources (bands, CRS), target (class map,
                           ignore index) and one entry per patch with its files
      Images/              patch_0000000.png | .jpg | .npy
      Masks/               patch_0000000.png (uint8 class ids, 0 = background,
                           255 = no imagery: pass ignore_index=255 to your loss)
      splits/              train.txt, val.txt, test.txt, 10/labeled.txt, ...

and yields one dict per patch::

    {"image": float32 (C, H, W), "mask": int64 (H, W), "filename": str}

File paths come from each manifest entry's ``files`` (relative to the dataset
folder), never from guessing names.

PNG/JPG images are scaled to 0..1. NPY images (Sentinel-2) are returned as
stored: decoded reflectance, bands-first, with NaN where the product has no
data. ``"mask"`` is absent for image-only datasets.

It only needs numpy and Pillow. When PyTorch is installed the class is a
``torch.utils.data.Dataset``, and ``DataLoader``'s default collate turns the
numpy arrays into tensors::

    from torch.utils.data import DataLoader
    loader = DataLoader(MapcvDataset("dataset", "train"), batch_size=8, shuffle=True)

Run it to summarise a dataset::

    python torch_dataset.py ../quickstart/dataset
"""

from __future__ import annotations

import argparse
import json
from collections.abc import Callable
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
import numpy.typing as npt
from PIL import Image

if TYPE_CHECKING:
    _Base = object
else:
    try:
        from torch.utils.data import Dataset as _Base
    except ImportError:  # torch is optional
        _Base = object

Sample = dict[str, Any]
Transform = Callable[[Sample], Sample]


class MapcvDataset(_Base):
    """Patches and masks of one split of a mapcv dataset.

    Args:
        root: Dataset directory (``writer.staging_dir``) containing manifest.json.
        split: ``"train"``, ``"val"``, ``"test"``, a semi-supervised list such as
            ``"10/labeled"`` (any ``splits/<name>.txt``), or ``None`` for every patch.
        transform: Optional callable that takes and returns the sample dict, for
            augmentation or normalisation. It must transform image and mask together.
    """

    def __init__(
        self,
        root: str | Path,
        split: str | None = "train",
        transform: Transform | None = None,
    ) -> None:
        self.root = Path(root)
        self.transform = transform
        manifest_path = self.root / "manifest.json"
        if not manifest_path.exists():
            raise FileNotFoundError(f"{manifest_path} not found; run `mapcv generate` first")
        self.manifest: dict[str, Any] = json.loads(manifest_path.read_text())
        version = self.manifest.get("version", 1)
        if version != 3:
            raise ValueError(
                f"{manifest_path} is manifest version {version}; this class reads version 3 "
                "(mapcv 0.3). Upgrade it in place with mapcv: "
                "mapcv.Manifest.load(path).save(path)"
            )

        target: dict[str, Any] = self.manifest.get("target") or {}
        class_map: dict[str, int] = target.get("class_map") or {}
        # Mask value of pixels without imagery; pass it to the loss as ignore_index.
        self.ignore_index: int | None = target.get("ignore_index")
        patches: list[dict[str, Any]] = self.manifest["patches"]
        # Without labels.label_field every polygon is class 1 and class_map is empty.
        if not class_map and self.manifest.get("target"):
            class_map = {"foreground": 1}
        self.class_names: dict[int, str] = {0: "background"}
        self.class_names.update({class_id: name for name, class_id in class_map.items()})
        self.num_classes = max(self.class_names) + 1

        # Split lists name each patch by its image's file name.
        by_name = {Path(patch["files"]["image"]).name: patch for patch in patches}
        if split is None:
            self.patches = patches
        else:
            split_file = self.root / "splits" / f"{split}.txt"
            if not split_file.exists():
                raise FileNotFoundError(f"{split_file} not found; run `mapcv split {self.root}`")
            names = split_file.read_text().split()
            self.patches = [by_name[name] for name in names]

    def __len__(self) -> int:
        return len(self.patches)

    def __getitem__(self, index: int) -> Sample:
        files: dict[str, str] = self.patches[index]["files"]
        sample: Sample = {
            "image": self.load_image(files["image"]),
            "filename": Path(files["image"]).name,
        }
        if "mask" in files:
            sample["mask"] = self.load_mask(files["mask"])
        if self.transform is not None:
            sample = self.transform(sample)
        return sample

    def load_image(self, path_in_dataset: str) -> npt.NDArray[np.float32]:
        """Read one image patch as float32, bands-first ``(C, H, W)``."""
        path = self.root / path_in_dataset
        if path.suffix == ".npy":
            array: npt.NDArray[Any] = np.load(path, allow_pickle=False)
            return array.astype(np.float32, copy=False)
        with Image.open(path) as image:
            rgb = np.asarray(image.convert("RGB"), dtype=np.float32) / 255.0
        return np.ascontiguousarray(rgb.transpose(2, 0, 1))

    def load_mask(self, path_in_dataset: str) -> npt.NDArray[np.int64]:
        """Read one mask patch as int64 class ids ``(H, W)``."""
        with Image.open(self.root / path_in_dataset) as mask:
            return np.asarray(mask, dtype=np.int64)

    def class_pixel_counts(self) -> dict[str, int]:
        """Pixels per class name in this split, from the manifest (no images read)."""
        counts: dict[str, int] = {}
        for patch in self.patches:
            for class_id, pixels in patch["summary"].get("class_pixels", {}).items():
                if int(class_id) == self.ignore_index:
                    name = "no imagery"
                else:
                    name = self.class_names.get(int(class_id), f"class {class_id}")
                counts[name] = counts.get(name, 0) + int(pixels)
        return counts


def _summary(root: Path) -> None:
    for split in ("train", "val", "test"):
        dataset = MapcvDataset(root, split)
        counts = dataset.class_pixel_counts()
        total = sum(counts.values()) or 1
        balance = ", ".join(f"{name} {pixels / total:.1%}" for name, pixels in counts.items())
        print(f"{split:>5}: {len(dataset):4d} patches  ({balance})")

    sample = MapcvDataset(root, "train")[0]
    image = sample["image"]
    print(f"\nsample {sample['filename']}: image {image.shape} {image.dtype}", end="")
    if "mask" in sample:
        print(f", mask {sample['mask'].shape} {sample['mask'].dtype}", end="")
    print(f", value range {np.nanmin(image):.3f}..{np.nanmax(image):.3f}")

    try:
        from torch.utils.data import DataLoader
    except ImportError:
        print("\nPyTorch is not installed; install it to batch samples with a DataLoader.")
        return
    batch = next(iter(DataLoader(MapcvDataset(root, "train"), batch_size=4, shuffle=True)))
    shapes = {key: tuple(value.shape) for key, value in batch.items() if hasattr(value, "shape")}
    print(f"\nDataLoader batch: {shapes}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Summarise a mapcv dataset.")
    default = Path(__file__).resolve().parent.parent / "quickstart" / "dataset"
    parser.add_argument("root", nargs="?", type=Path, default=default, help="dataset directory")
    _summary(parser.parse_args().root)


if __name__ == "__main__":
    main()
