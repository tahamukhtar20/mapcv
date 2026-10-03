"""
mapcv - A satellite imagery dataset creation tool for segmentation.
"""

from mapcv._mapcv_rs import bounds, snap_bbox
from mapcv.config import (
    DEFAULT_SENTINEL2_L2A_BANDS,
    EOPFZarrImageryConfig,
    ImageryConfig,
    LabelsConfig,
    MapcvConfig,
    RegionConfig,
    XYZImageryConfig,
)
from mapcv.downloader import (
    URL_TEMPLATES,
    download_region,
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
from mapcv.sampler import SamplerConfig, sample_patches, sample_patches_at_anchors
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
    "ImageryConfig",
    "XYZImageryConfig",
    "EOPFZarrImageryConfig",
    "DEFAULT_SENTINEL2_L2A_BANDS",
    "MapcvConfig",
    "RegionConfig",
    "ClassMap",
    "GeomWithClass",
    "Manifest",
    "ManifestEntry",
    "WriterConfig",
    "download_region",
    "bounds",
    "load_or_create_manifest",
    "snap_bbox",
    "parse_geojson",
    "parse_kml",
    "rasterize",
    "resolve_url_template",
    "sample_patches",
    "sample_patches_at_anchors",
    "SamplerConfig",
    "split_dataset",
    "SplitterConfig",
    "stitch_region",
    "transform_to_mercator",
    "write_patches",
]
