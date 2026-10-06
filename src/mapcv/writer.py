"""Writing patches as files (``Images/``, ``Masks/``) and their manifest entries."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Literal, Optional, Tuple

import numpy as np
import numpy.typing as npt
from PIL import Image
from pydantic import BaseModel, ConfigDict, Field, model_validator

from mapcv._georef import epsg_code, is_geographic, world_file_text
from mapcv._mapcv_rs import write_geotiffs_rs, write_patches_rs
from mapcv.manifest import (
    IMAGES_DIR,
    MANIFEST_VERSION,
    MASKS_DIR,
    Manifest,
    ManifestEntry,
    ManifestMismatchError,
    PatchSummary,
    SourceRecord,
    Transform,
    load_or_create_manifest,
)
from mapcv.sampler import PatchMeta

__all__ = [
    "MANIFEST_VERSION",
    "Manifest",
    "ManifestEntry",
    "ManifestMismatchError",
    "PatchSummary",
    "Transform",
    "WriterConfig",
    "load_or_create_manifest",
    "write_patches",
    "write_source_images",
]

# Sample types the GeoTIFF writer stores; NPY takes any numeric type, PNG only the first two.
_PNG_MASK_DTYPES = ("uint8", "uint16")
_TIF_DTYPES = (
    "uint8",
    "int8",
    "uint16",
    "int16",
    "uint32",
    "int32",
    "uint64",
    "int64",
    "float32",
    "float64",
)


class WriterConfig(BaseModel):
    """Configuration for writing patches to disk."""

    # Unknown keys are errors, so typos and newer-version options are not silently ignored.
    model_config = ConfigDict(extra="forbid")

    staging_dir: Path
    image_format: Literal["png", "jpg", "npy", "tif"] = "png"
    jpg_quality: int = Field(default=95, ge=1, le=100)
    # 4:2:0 (Pillow's default) halves the chroma resolution for a much smaller file; 4:4:4 keeps it.
    jpg_subsampling: Literal["4:2:0", "4:4:4"] = "4:2:0"
    # PNG holds 8 and 16-bit masks; NPY any dtype; TIF (GeoTIFF) any dtype, georeferenced.
    mask_format: Literal["png", "npy", "tif"] = "png"
    # A .pgw / .jgw world file next to every PNG / JPG patch (images and masks).
    world_files: bool = False
    # patches.geojson in the dataset folder: one WGS-84 footprint per patch.
    footprints: bool = True

    @model_validator(mode="after")
    def _world_files_need_a_png_or_jpg(self) -> "WriterConfig":
        if (
            self.world_files
            and self.image_format not in ("png", "jpg")
            and self.mask_format != "png"
        ):
            raise ValueError(
                "writer.world_files adds .pgw/.jgw files to PNG and JPG patches, but "
                f"image_format '{self.image_format}' and mask_format '{self.mask_format}' "
                "are neither (GeoTIFF patches carry their georeferencing inside the file)"
            )
        return self


def _entry(
    image_name: str,
    mask_name: Optional[str],
    row: int,
    col: int,
    padded: bool,
    chunk: int,
    counts: Optional[Dict[str, int]],
    empty_ratio: float,
    extra_files: Optional[Dict[str, str]] = None,
    images_dir: str = IMAGES_DIR,
    masks_dir: str = MASKS_DIR,
    image_key: str = "image",
) -> ManifestEntry:
    files = {image_key: f"{images_dir}/{image_name}"}
    summary = PatchSummary(empty_ratio=empty_ratio)
    if mask_name is not None:
        files["mask"] = f"{masks_dir}/{mask_name}"
    if extra_files:
        files.update(extra_files)
    if counts is not None:
        summary = PatchSummary(class_pixels=counts, empty_ratio=empty_ratio)
    return ManifestEntry(row=row, col=col, padded=padded, chunk=chunk, files=files, summary=summary)


def _class_counts(mask: npt.NDArray[Any]) -> Optional[Dict[str, int]]:
    """Pixels per class ID, keys ascending; ``None`` for a float mask (a regression target)."""
    if mask.dtype.kind not in "biu":
        return None
    # Ascending order makes the keys numerically ordered ("2" before "10"), so identical
    # runs serialise identically.
    if mask.dtype.kind in "bu" and mask.dtype.itemsize <= 2:
        histogram = np.bincount(mask.ravel())
        present = np.flatnonzero(histogram)
        return {str(int(value)): int(histogram[value]) for value in present}
    values, counts = np.unique(mask, return_counts=True)
    return {str(int(value)): int(count) for value, count in zip(values, counts)}


def _empty_ratio(image: npt.NDArray[Any]) -> float:
    if image.ndim == 2:
        empty = ~np.isfinite(image) | (image == 0)
    else:
        empty = np.all(~np.isfinite(image), axis=-1)
    return float(np.count_nonzero(empty) / empty.size) if empty.size else 1.0


def _check_masks(masks: Any, count: int, mask_format: str) -> None:
    """Check ``masks`` is an ``(N, H, W)`` array a ``mask_format`` file can hold."""
    if not isinstance(masks, np.ndarray) or masks.ndim != 3 or masks.shape[0] != count:
        raise ValueError(f"mask patches must be an ({count}, H, W) array")
    name = masks.dtype.name
    if mask_format == "png" and name not in _PNG_MASK_DTYPES:
        raise ValueError(
            f"PNG masks hold uint8 or uint16 values, not {name}; "
            "set writer.mask_format to 'npy' or 'tif'"
        )
    if mask_format == "tif" and name not in _TIF_DTYPES:
        raise ValueError(f"a GeoTIFF mask cannot hold {name} values")
    if mask_format == "npy" and masks.dtype.kind not in "biuf":
        raise ValueError(f"an NPY mask must be numeric, got {name}")


def _mask_nodata(ignore_index: Optional[int], dtype: "np.dtype[Any]") -> Optional[float]:
    """The ``GDAL_NODATA`` of a GeoTIFF mask: its ``ignore_index`` (pixels without imagery)."""
    if ignore_index is None:
        return None
    if dtype.kind in "iu":
        info = np.iinfo(dtype)
        if not info.min <= ignore_index <= info.max:
            raise ValueError(f"ignore_index {ignore_index} does not fit a {dtype.name} mask")
    return float(ignore_index)


def _image_nodata(dtype: "np.dtype[Any]", source: SourceRecord) -> Optional[float]:
    """The ``GDAL_NODATA`` of a GeoTIFF image of ``source``.

    A GeoTIFF source's own no-data value (``fingerprint.nodata``) when it has one the
    output type can hold. Otherwise NaN for floats (what no-data pixels hold) and none for
    integers: 0 is also a legitimate value there, and a no-data value applies per band.
    """
    fingerprint = source.fingerprint or {}
    declared = fingerprint.get("nodata")
    if declared is not None:
        value = float("nan") if declared == "nan" else float(declared)
        if dtype.kind == "f":
            return value
        if dtype.kind in "iu" and np.isfinite(value) and value == int(value):
            info = np.iinfo(dtype)
            return value if info.min <= value <= info.max else None
        return None
    return float("nan") if dtype.kind == "f" else None


def _georeference(manifest: Manifest) -> Tuple[int, bool]:
    """EPSG code of the source CRS and whether it is geographic."""
    crs = manifest.source.crs
    if crs is None:
        raise ValueError("the manifest records no CRS, so GeoTIFF patches cannot be georeferenced")
    code = epsg_code(crs)
    return code, is_geographic(code)


def _write_geotiffs(
    patches: npt.NDArray[Any],
    names: List[str],
    directory: Path,
    meta: List[PatchMeta],
    manifest: Manifest,
    nodata: Optional[float],
    band_names: Optional[List[str]],
) -> None:
    """Write ``(N, H, W)`` or ``(N, H, W, C)`` patches as GeoTIFFs georeferenced from ``meta``."""
    epsg, geographic = _georeference(manifest)
    array = patches if patches.dtype.isnative else patches.astype(patches.dtype.newbyteorder("="))
    array = np.ascontiguousarray(array)
    count, height, width = array.shape[:3]
    bands = array.shape[3] if array.ndim == 4 else 1
    transforms = [manifest.transform_at(item["row"], item["col"]) for item in meta]
    write_geotiffs_rs(
        array.reshape(-1).view(np.uint8),
        array.dtype.name,
        (count, height, width, bands),
        transforms,
        names,
        str(directory),
        epsg,
        geographic,
        nodata,
        band_names if band_names is not None and len(band_names) == bands else None,
    )


def _write_world_files(
    directory: Path, stems: List[str], suffix: str, meta: List[PatchMeta], manifest: Manifest
) -> None:
    for stem, item in zip(stems, meta):
        text = world_file_text(manifest.transform_at(item["row"], item["col"]))
        (directory / f"{stem}.{suffix}").write_text(text, encoding="ascii", newline="\n")


def _write_python_masks(
    masks: npt.NDArray[Any],
    stems: List[str],
    directory: Path,
    config: WriterConfig,
    meta: List[PatchMeta],
    manifest: Manifest,
) -> None:
    """Masks the Rust PNG writer does not take: NPY, GeoTIFF and 16-bit or NPY-image PNG."""
    if config.mask_format == "tif":
        _write_geotiffs(
            masks,
            [f"{stem}.tif" for stem in stems],
            directory,
            meta,
            manifest,
            _mask_nodata(manifest.ignore_index, masks.dtype),
            None,
        )
    elif config.mask_format == "npy":
        for stem, mask in zip(stems, masks):
            np.save(directory / f"{stem}.npy", np.ascontiguousarray(mask), allow_pickle=False)
    else:
        for stem, mask in zip(stems, masks):
            # uint8 gives mode "L", uint16 "I;16": 8 and 16-bit grayscale PNGs.
            image = Image.fromarray(mask)
            image.save(directory / f"{stem}.png", format="PNG")


def _write_npy_images(images: npt.NDArray[Any], stems: List[str], directory: Path) -> None:
    for stem, image in zip(stems, images):
        channels_first = image[np.newaxis, ...] if image.ndim == 2 else np.moveaxis(image, -1, 0)
        np.save(directory / f"{stem}.npy", np.ascontiguousarray(channels_first), allow_pickle=False)


def write_patches(
    image_patches: npt.NDArray[Any],
    mask_patches: Optional[npt.NDArray[Any]],
    meta: List[PatchMeta],
    config: WriterConfig,
    manifest: Manifest,
    chunk_index: int = 0,
    *,
    images_dir: str = IMAGES_DIR,
    masks_dir: str = MASKS_DIR,
    image_key: str = "image",
    source: Optional[SourceRecord] = None,
) -> None:
    """Write patches to ``Images/`` and ``Masks/`` and append their entries to ``manifest``.

    ``images_dir`` names the image folder (detection datasets use ``images``, the
    folder name Ultralytics expects next to ``labels``) and ``masks_dir`` the mask folder
    (instance datasets use ``masks``, next to ``images``). ``image_key`` is the key of
    the image in each entry's ``files`` (the source's name) and ``source`` the imagery
    source the images come from (default: the manifest's first).

    Images: PNG/JPG (uint8 RGB) are encoded by the Rust writer; NPY keeps any
    channel count and dtype, stored bands-first; TIF is a GeoTIFF (also any
    channel count and dtype, bands-first when read) georeferenced from the
    manifest's source CRS and transform. Masks (``(N, H, W)``, any integer or,
    for NPY and TIF, float dtype): PNG holds uint8 and uint16, NPY and TIF any
    dtype; a TIF mask declares ``ignore_index`` as its no-data value. With
    ``config.world_files`` every PNG and JPG also gets a world file.

    Files are numbered from ``len(manifest.patches)``, so any existing file at
    those numbers is an orphan from an interrupted run and is overwritten.
    """
    if len(meta) == 0:
        return
    if mask_patches is not None:
        _check_masks(mask_patches, len(meta), config.mask_format)

    images_path = config.staging_dir / images_dir
    masks_path = config.staging_dir / masks_dir
    start = len(manifest.patches)
    stems = [f"patch_{start + index:07d}" for index in range(len(meta))]
    image_suffix = config.image_format
    mask_suffix = config.mask_format

    # The Rust PNG/JPG writer also encodes uint8 PNG masks; every other combination
    # writes its masks separately.
    rust_images = config.image_format in ("png", "jpg")
    rust_masks = (
        rust_images
        and mask_patches is not None
        and config.mask_format == "png"
        and mask_patches.dtype == np.uint8
    )

    if rust_images and (
        image_patches.dtype != np.uint8 or image_patches.ndim != 4 or image_patches.shape[-1] != 3
    ):
        raise ValueError("PNG/JPG output requires uint8 image patches shaped (N, H, W, 3)")

    images_path.mkdir(parents=True, exist_ok=True)
    if mask_patches is not None:
        masks_path.mkdir(parents=True, exist_ok=True)

    counts: List[Optional[Dict[str, int]]] = [None] * len(meta)
    if rust_images:
        results = write_patches_rs(
            np.ascontiguousarray(image_patches),
            np.ascontiguousarray(mask_patches) if rust_masks else None,
            [(item["row"], item["col"], item["padded"]) for item in meta],
            start,
            chunk_index,
            str(images_path),
            str(masks_path),
            config.image_format,
            config.jpg_quality,
            config.jpg_subsampling,
        )
        # The sampler's ratio counts pixels without valid imagery (NoData, NaN, outside the
        # raster), the same measure the masks use; the Rust writer's all-black ratio is the
        # fallback for callers that bring no metadata. For XYZ the two are identical.
        empty_ratios = [
            float(item["empty_ratio"]) if "empty_ratio" in item else float(result[7])
            for item, result in zip(meta, results)
        ]
        if rust_masks:
            counts = [
                {str(class_id): int(count) for class_id, count in result[6]} for result in results
            ]
    else:
        if config.image_format == "npy":
            _write_npy_images(image_patches, stems, images_path)
        else:
            _write_geotiffs(
                image_patches,
                [f"{stem}.tif" for stem in stems],
                images_path,
                meta,
                manifest,
                _image_nodata(image_patches.dtype, source or manifest.source),
                (source or manifest.source).bands or None,
            )
        empty_ratios = [
            item["empty_ratio"] if "empty_ratio" in item else _empty_ratio(image)
            for item, image in zip(meta, image_patches)
        ]

    if mask_patches is not None and not rust_masks:
        _write_python_masks(mask_patches, stems, masks_path, config, meta, manifest)
        counts = [_class_counts(mask) for mask in mask_patches]

    world: List[Dict[str, str]] = [{} for _ in meta]
    if config.world_files:
        if rust_images:
            _write_world_files(
                images_path, stems, "pgw" if image_suffix == "png" else "jgw", meta, manifest
            )
            for entry_files, stem in zip(world, stems):
                entry_files[f"{image_key}_world"] = (
                    f"{images_dir}/{stem}.{'pgw' if image_suffix == 'png' else 'jgw'}"
                )
        if mask_patches is not None and mask_suffix == "png":
            _write_world_files(masks_path, stems, "pgw", meta, manifest)
            for entry_files, stem in zip(world, stems):
                entry_files["mask_world"] = f"{masks_dir}/{stem}.pgw"

    for index, item in enumerate(meta):
        manifest.patches.append(
            _entry(
                f"{stems[index]}.{image_suffix}",
                f"{stems[index]}.{mask_suffix}" if mask_patches is not None else None,
                item["row"],
                item["col"],
                item["padded"],
                chunk_index,
                counts[index],
                empty_ratios[index],
                world[index],
                images_dir,
                masks_dir,
                image_key,
            )
        )


def write_source_images(
    image_patches: npt.NDArray[Any],
    meta: List[PatchMeta],
    config: WriterConfig,
    manifest: Manifest,
    start: int,
    chunk_index: int,
    *,
    source: SourceRecord,
    images_dir: str,
) -> List[Dict[str, str]]:
    """Write a further imagery source's patches, numbered like the first source's.

    ``image_patches`` are the source's patches at the positions in ``meta`` (read on
    the first source's grid), written to ``images_dir`` as ``writer.image_format``
    files named from ``start``, as :func:`write_patches` names the first source's.
    Returns each patch's ``files`` entries for this source: the image under the
    source's name and, with ``writer.world_files`` and PNG/JPG, its world file.
    """
    if len(meta) == 0:
        return []
    images_path = config.staging_dir / images_dir
    images_path.mkdir(parents=True, exist_ok=True)
    stems = [f"patch_{start + index:07d}" for index in range(len(meta))]
    suffix = config.image_format
    if suffix in ("png", "jpg"):
        if (
            image_patches.dtype != np.uint8
            or image_patches.ndim != 4
            or image_patches.shape[-1] != 3
        ):
            raise ValueError(
                f"PNG/JPG output requires uint8 image patches shaped (N, H, W, 3); imagery "
                f"'{source.name}' gives {image_patches.dtype.name} {image_patches.shape[1:]}"
            )
        write_patches_rs(
            np.ascontiguousarray(image_patches),
            None,
            [(item["row"], item["col"], item["padded"]) for item in meta],
            start,
            chunk_index,
            str(images_path),
            str(images_path),
            suffix,
            config.jpg_quality,
            config.jpg_subsampling,
        )
    elif suffix == "npy":
        _write_npy_images(image_patches, stems, images_path)
    else:
        _write_geotiffs(
            image_patches,
            [f"{stem}.tif" for stem in stems],
            images_path,
            meta,
            manifest,
            _image_nodata(image_patches.dtype, source),
            source.bands or None,
        )
    files = [{source.name: f"{images_dir}/{stem}.{suffix}"} for stem in stems]
    if config.world_files and suffix in ("png", "jpg"):
        world = "pgw" if suffix == "png" else "jgw"
        _write_world_files(images_path, stems, world, meta, manifest)
        for entry_files, stem in zip(files, stems):
            entry_files[f"{source.name}_world"] = f"{images_dir}/{stem}.{world}"
    return files
