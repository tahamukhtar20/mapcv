"""
mapcv - A satellite imagery dataset creation tool for segmentation, detection and instance segmentation.
"""

from importlib.metadata import PackageNotFoundError
from importlib.metadata import version as _package_version

from mapcv._mapcv_rs import bounds, snap_bbox
from mapcv.config import (
    DEFAULT_SENTINEL2_L2A_BANDS,
    DetectionOptions,
    EOPFZarrImageryConfig,
    GeoTiffImageryConfig,
    ImageryConfig,
    InstanceOptions,
    LabelsConfig,
    MapcvConfig,
    RasterClass,
    RasterLabelsConfig,
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
from mapcv.manifest import (
    MANIFEST_VERSION,
    Manifest,
    ManifestEntry,
    ManifestMismatchError,
    PatchSummary,
    SourceRecord,
    TargetRecord,
    load_or_create_manifest,
)
from mapcv.pipeline import GenerateResult, run_generate, run_split
from mapcv.planning import Plan, plan
from mapcv.rasterizer import rasterize
from mapcv.sampler import SamplerConfig, sample_patches, sample_patches_at_anchors
from mapcv.splitter import SplitterConfig, split_dataset
from mapcv.writer import WriterConfig, write_patches

try:
    __version__ = _package_version("mapcv")
except PackageNotFoundError:  # running from a source tree without installing
    __version__ = "0+unknown"

__all__ = [
    "__version__",
    "GenerateResult",
    "Plan",
    "plan",
    "run_generate",
    "run_split",
    "URL_TEMPLATES",
    "LabelsConfig",
    "DetectionOptions",
    "InstanceOptions",
    "RasterClass",
    "RasterLabelsConfig",
    "ImageryConfig",
    "XYZImageryConfig",
    "EOPFZarrImageryConfig",
    "GeoTiffImageryConfig",
    "DEFAULT_SENTINEL2_L2A_BANDS",
    "MapcvConfig",
    "RegionConfig",
    "ClassMap",
    "GeomWithClass",
    "MANIFEST_VERSION",
    "Manifest",
    "ManifestMismatchError",
    "ManifestEntry",
    "PatchSummary",
    "SourceRecord",
    "TargetRecord",
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
