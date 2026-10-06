"""Dry-run planning: estimate the size and cost of a dataset before generating it."""

from __future__ import annotations

import math
import warnings
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np

from mapcv._mapcv_rs import grid_sample_anchors, snap_bbox, tile
from mapcv.config import (
    EOPFZarrImageryConfig,
    GeoTiffImageryConfig,
    MapcvConfig,
    RasterLabelsConfig,
    XYZImageryConfig,
    eopf_local_path,
)
from mapcv.imagery import GeoTiffRasterSource
from shapely.geometry import box

from mapcv.labels import load_vector_labels
from mapcv.sampler import random_patch_capacity

# Earth radius used by Web Mercator; ground resolution at zoom z is
# 2 * pi * R * cos(lat) / (256 * 2**z) metres per pixel.
_EARTH_RADIUS_M = 6_378_137.0
_TILE_PX = 256
# Typical encoded sizes, used only for rough estimates.
_XYZ_TILE_BYTES = 25_000
_PNG_COMPRESSION = 0.55
_JPG_COMPRESSION = 0.15
_MASK_COMPRESSION = 0.05
# GeoTIFF (Deflate with a predictor): like PNG for 8-bit data, and about a fifth off float32 bytes.
_TIF_FLOAT_COMPRESSION = 0.8
_OBJECT_BYTES = 250
# An instance: its COCO annotation with the RLE mask, and the same again in the chunk store.
_INSTANCE_BYTES = 500
# A classification patch: its row in labels.csv, labels_<split>.csv and labels.json, and its
# coverage in the manifest.
_CLASSIFICATION_BYTES = 250

# Jobs above either threshold ask for confirmation before downloading.
LARGE_JOB_TILES = 20_000
LARGE_JOB_BYTES = 5 * 1024**3


@dataclass
class LabelSummary:
    """What the configured label file contains."""

    path: str
    polygons: int
    classes: Dict[str, int]
    warnings: List[str] = field(default_factory=list)
    # Features whose geometry intersects the region (each is one detection or instance object).
    in_region: int = 0
    #: What the label raster is (CRS, size, pixel size), for raster labels.
    raster: Optional[str] = None


@dataclass
class Plan:
    """Estimated size and cost of generating a dataset from a config."""

    task: str
    region_km: Tuple[float, float]
    imagery: str
    resolution_m: float
    raster_px: Tuple[int, int]
    patches: int
    patch_size: int
    tiles: Optional[int]
    download_bytes: Optional[int]
    output_bytes: int
    chunk_memory_bytes: int
    labels: Optional[LabelSummary]
    warnings: List[str] = field(default_factory=list)
    # Detection and instance segmentation: label features in the region, each one object
    # (``None`` for other tasks).
    objects: Optional[int] = None

    @property
    def is_large(self) -> bool:
        """Whether the job is big enough to confirm before downloading."""
        return (self.tiles or 0) > LARGE_JOB_TILES or (
            self.download_bytes or 0
        ) + self.output_bytes > LARGE_JOB_BYTES


def ground_resolution_m(zoom: int, latitude: float) -> float:
    """Web Mercator ground resolution in metres per pixel at *latitude*."""
    circumference = 2 * math.pi * _EARTH_RADIUS_M * math.cos(math.radians(latitude))
    return float(circumference / (_TILE_PX * 2**zoom))


def region_size_km(west: float, south: float, east: float, north: float) -> Tuple[float, float]:
    """Approximate (width, height) of a lon/lat box in kilometres."""
    mid_lat = math.radians((south + north) / 2)
    width = (east - west) * 111.320 * math.cos(mid_lat)
    height = (north - south) * 110.574
    return width, height


def _xyz_raster(config: MapcvConfig, imagery: XYZImageryConfig) -> Tuple[int, int, int]:
    region = config.region
    snapped = snap_bbox(region.west, region.south, region.east, region.north, imagery.zoom)
    eps = 1e-9
    top_left = tile(snapped.west + eps, snapped.north - eps, imagery.zoom)
    bottom_right = tile(snapped.east - eps, snapped.south + eps, imagery.zoom)
    cols = bottom_right.x - top_left.x + 1
    rows = bottom_right.y - top_left.y + 1
    return rows * _TILE_PX, cols * _TILE_PX, rows * cols


def _eopf_raster(config: MapcvConfig, imagery: EOPFZarrImageryConfig) -> Tuple[int, int]:
    """Raster size in the product's UTM grid, snapped outward to whole pixels."""
    region = config.region
    res = imagery.resolution
    try:
        from pyproj import Transformer
    except ImportError:  # without the zarr extra: kilometre approximation
        width_km, height_km = region_size_km(region.west, region.south, region.east, region.north)
        return (
            max(1, math.ceil(height_km * 1000 / res)),
            max(1, math.ceil(width_km * 1000 / res)),
        )
    lon = (region.west + region.east) / 2
    lat = (region.south + region.north) / 2
    zone = min(60, int((lon + 180) // 6) + 1)
    epsg = (32600 if lat >= 0 else 32700) + zone
    transformer = Transformer.from_crs("EPSG:4326", f"EPSG:{epsg}", always_xy=True)
    left, bottom, right, top = transformer.transform_bounds(
        region.west, region.south, region.east, region.north, densify_pts=21
    )
    cols = math.ceil(right / res) - math.floor(left / res)
    rows = math.ceil(top / res) - math.floor(bottom / res)
    return max(1, rows), max(1, cols)


def _pixel_size_m(crs: str, transform: Tuple[float, float, float, float, float, float]) -> float:
    """Ground size of one pixel of a raster in ``crs``, in metres (at the raster's origin)."""
    from pyproj import CRS

    a, b, _, d, e, f = transform  # f is the top latitude for a geographic CRS
    size = math.sqrt(abs(a * e - b * d))
    parsed = CRS.from_user_input(crs)
    if parsed.is_geographic:
        return size * 111_320.0 * math.cos(math.radians(f))
    return size * float(parsed.axis_info[0].unit_conversion_factor)


def _geotiff_raster(
    config: MapcvConfig, imagery: GeoTiffImageryConfig, warned: List[str]
) -> Tuple[int, int, float, int, int, str]:
    """Open the file's header and size the region's window: ``(height, width, metres per
    pixel, channels, bytes per value, description)``."""
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always", UserWarning)
        source = GeoTiffRasterSource(
            config.region, imagery, image_format=config.writer.image_format
        )
    warned.extend(str(warning.message) for warning in caught)
    meta = source.metadata
    source.close()
    resolution = _pixel_size_m(meta.crs, meta.transform)
    description = f"GeoTIFF {meta.product_id} · {meta.crs} · {len(meta.bands)} band(s) {meta.dtype}"
    return (
        meta.height,
        meta.width,
        resolution,
        len(meta.bands),
        int(np.dtype(meta.dtype).itemsize),
        description,
    )


def _patch_count(height: int, width: int, config: MapcvConfig) -> int:
    sampler = config.sampler
    if sampler.mode == "random":
        return random_patch_capacity(height, width, sampler)
    return len(
        grid_sample_anchors(
            height, width, sampler.patch_size, sampler.stride, sampler.edge_strategy
        )
    )


def _summarize_label_raster(config: MapcvConfig, labels: RasterLabelsConfig) -> LabelSummary:
    """Open the label raster's header (no pixels are read) and check it covers the region."""
    from pyproj import Transformer

    from mapcv.targets.raster_labels import LabelRasterSampler

    local = eopf_local_path(labels.path)
    if local is not None and not local.exists():
        return LabelSummary(
            labels.path, 0, labels.class_map(), [f"label raster not found: {local}"]
        )
    # The same checks generate makes (CRS, band, integer values); the CRS argument only
    # matters for sampling, which planning does not do.
    sampler = LabelRasterSampler(labels, "EPSG:4326")
    info = sampler.info
    a, b, c, d, e, f = sampler.transform
    xs = [c + a * col + b * row for col in (0, info.width) for row in (0, info.height)]
    ys = [f + d * col + e * row for col in (0, info.width) for row in (0, info.height)]
    to_wgs84 = Transformer.from_crs(sampler.crs, "EPSG:4326", always_xy=True)
    west, south, east, north = to_wgs84.transform_bounds(
        min(xs), min(ys), max(xs), max(ys), densify_pts=21
    )
    region = config.region
    messages: List[str] = []
    if not box(west, south, east, north).intersects(
        box(region.west, region.south, region.east, region.north)
    ):
        outcome = (
            "no patch would get a label"
            if config.task == "classification"
            else "every mask pixel would be ignored"
        )
        messages.append(
            f"the label raster does not overlap the region, so {outcome}. "
            "Check labels.path and the region."
        )
    description = (
        f"raster {sampler.name} · {sampler.crs} · {info.width:,} × {info.height:,} px · "
        f"{info.dtype} · ≈ {_pixel_size_m(sampler.crs, sampler.transform):.2f} m/px"
    )
    return LabelSummary(labels.path, 0, labels.class_map(), messages, raster=description)


def summarize_labels(config: MapcvConfig) -> Optional[LabelSummary]:
    """Parse the configured label file and summarize it, or ``None`` without labels."""
    labels = config.labels
    if labels is None:
        return None
    if isinstance(labels, RasterLabelsConfig):
        return _summarize_label_raster(config, labels)
    if not labels.path.exists():
        return LabelSummary(str(labels.path), 0, {}, [f"label file not found: {labels.path}"])
    points = config.task == "detection" and config.detection_options.point_box_size is not None
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always", UserWarning)
        geometries, class_map = load_vector_labels(
            labels.path, labels.label_field, labels.classes, points=points, layer=labels.layer
        )
    messages = [str(warning.message) for warning in caught]
    region = config.region
    area = box(region.west, region.south, region.east, region.north)
    in_region = sum(1 for geometry, _ in geometries if geometry.intersects(area))
    if geometries and not in_region:
        if config.task == "detection":
            what, outcome = "feature", "no patch would have objects"
        elif config.task == "instance":
            what, outcome = "feature", "no patch would have instances"
        elif config.task == "classification":
            what, outcome = "polygon", "no patch would get a label"
        else:
            what, outcome = "polygon", "every mask would be background"
        messages.append(
            f"no label {what} intersects the region, so {outcome}. "
            "Check that labels are longitude/latitude (not swapped) and cover the region."
        )
    return LabelSummary(str(labels.path), len(geometries), class_map, messages, in_region)


def plan(config: MapcvConfig) -> Plan:
    """Estimate a generation run without downloading any imagery."""
    imagery = config.imagery
    region = config.region
    region_km = region_size_km(region.west, region.south, region.east, region.north)
    plan_warnings: List[str] = []
    patch_size = config.sampler.patch_size

    tiles: Optional[int] = None
    download: Optional[int] = None
    if isinstance(imagery, XYZImageryConfig):
        height, width, tiles = _xyz_raster(config, imagery)
        download = tiles * _XYZ_TILE_BYTES
        resolution = ground_resolution_m(imagery.zoom, (region.south + region.north) / 2)
        name = imagery.source or "custom XYZ template"
        description = f"{name} · zoom {imagery.zoom}"
        channels, bytes_per_value = 3, 1
        chunk_rows = imagery.strip_rows * _TILE_PX
    elif isinstance(imagery, GeoTiffImageryConfig):
        height, width, resolution, channels, bytes_per_value, description = _geotiff_raster(
            config, imagery, plan_warnings
        )
        chunk_rows = imagery.chunk_rows
    else:
        height, width = _eopf_raster(config, imagery)
        resolution = float(imagery.resolution)
        description = f"Sentinel-2 L2A (EOPF) · {len(imagery.bands)} bands"
        channels, bytes_per_value = len(imagery.bands), 4
        chunk_rows = imagery.chunk_rows

    patches = _patch_count(height, width, config)
    pixels_per_patch = patch_size * patch_size
    image_format = config.writer.image_format
    if image_format == "npy":
        image_bytes = channels * pixels_per_patch * bytes_per_value
    elif image_format == "tif":
        ratio = _PNG_COMPRESSION if bytes_per_value == 1 else _TIF_FLOAT_COMPRESSION
        image_bytes = int(channels * pixels_per_patch * bytes_per_value * ratio)
    elif image_format == "jpg":
        image_bytes = int(3 * pixels_per_patch * _JPG_COMPRESSION)
    else:
        image_bytes = int(3 * pixels_per_patch * _PNG_COMPRESSION)
    # Segmentation writes one uint8 mask per patch when there are labels: compressed as
    # PNG or GeoTIFF, raw as NPY.
    has_masks = config.task == "segmentation" and config.labels is not None
    mask_ratio = 1.0 if config.writer.mask_format == "npy" else _MASK_COMPRESSION
    mask_bytes = int(pixels_per_patch * mask_ratio) if has_masks else 0
    if config.task == "instance" and config.instance_options.id_mask:
        # One 16-bit instance-ID mask per patch (compressed PNG or GeoTIFF, raw as NPY).
        mask_bytes = int(2 * pixels_per_patch * mask_ratio)
    output = patches * (image_bytes + mask_bytes)
    if config.task == "classification":
        output += patches * _CLASSIFICATION_BYTES
    window_rows = min(height, chunk_rows + patch_size)
    # Window, validity mask, label mask and extracted patches each hold a copy.
    chunk_memory = window_rows * width * (channels * bytes_per_value * 2 + 2)

    labels = summarize_labels(config)
    if labels is not None:
        plan_warnings.extend(labels.warnings)
        if config.task == "detection":
            # A box in the COCO file, the YOLO label and the chunk store.
            output += labels.in_region * _OBJECT_BYTES
        elif config.task == "instance":
            output += labels.in_region * _INSTANCE_BYTES
    if config.sampler.mode == "random" and 0 < patches < config.sampler.random_count:
        plan_warnings.append(
            f"random_count is {config.sampler.random_count} but only {patches} distinct patch "
            "positions exist on this raster; all of them will be used"
        )
    if patches == 0:
        plan_warnings.append(
            "no patch fits the region with these sampler settings; enlarge the region or "
            "use edge_strategy: pad"
        )
    if resolution > patch_size * 10:
        plan_warnings.append("each patch covers more than 10 km; consider a higher zoom")

    return Plan(
        task=config.task,
        region_km=region_km,
        imagery=description,
        resolution_m=resolution,
        raster_px=(width, height),
        patches=patches,
        patch_size=patch_size,
        tiles=tiles,
        download_bytes=download,
        output_bytes=output,
        chunk_memory_bytes=chunk_memory,
        labels=labels,
        warnings=plan_warnings,
        objects=labels.in_region
        if labels is not None and config.task in ("detection", "instance")
        else None,
    )


def human_bytes(size: float) -> str:
    """Format a byte count, e.g. ``1.5 GB``."""
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if size < 1000 or unit == "TB":
            return f"{size:.0f} {unit}" if unit == "B" else f"{size:.1f} {unit}"
        size /= 1000
    return f"{size:.1f} TB"
