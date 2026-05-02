"""
mapcv - A satellite imagery dataset creation tool for segmentation.
"""

from mapcv._mapcv_rs import bounds, hello, snap_bbox
from mapcv.config import LabelsConfig, MapcvConfig, RegionConfig, TilesConfig
from mapcv.downloader import (
    URL_TEMPLATES,
    download_region,
    download_region_strips,
    iter_tile_strips,
    resolve_url_template,
    stitch_region,
)
from mapcv.labels import (
    ClassMap,
    GeomWithClass,
    parse_geojson,
    parse_kml,
    transform_to_mercator,
)
from mapcv.rasterizer import rasterize
from mapcv.sampler import SamplerConfig, sample_patches
from mapcv.splitter import SplitterConfig, split_dataset
from mapcv.writer import (
    Manifest,
    ManifestEntry,
    WriterConfig,
    load_or_create_manifest,
    write_patches,
)

__all__ = [
    "URL_TEMPLATES",
    "LabelsConfig",
    "MapcvConfig",
    "RegionConfig",
    "TilesConfig",
    "ClassMap",
    "GeomWithClass",
    "Manifest",
    "ManifestEntry",
    "WriterConfig",
    "download_region",
    "download_region_strips",
    "hello",
    "bounds",
    "load_or_create_manifest",
    "snap_bbox",
    "iter_tile_strips",
    "parse_geojson",
    "parse_kml",
    "rasterize",
    "resolve_url_template",
    "sample_patches",
    "SamplerConfig",
    "split_dataset",
    "SplitterConfig",
    "stitch_region",
    "transform_to_mercator",
    "write_patches",
]
