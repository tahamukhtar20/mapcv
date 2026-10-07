"""
mapcv - Remote-sensing training datasets for segmentation, detection, instance segmentation,
classification, change detection and regression.
"""

from importlib.metadata import PackageNotFoundError
from importlib.metadata import version as _package_version

from mapcv._mapcv_rs import bounds, snap_bbox
from mapcv.config import (
    DEFAULT_SENTINEL2_L2A_BANDS,
    ClassificationOptions,
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
    StacCogImageryConfig,
    XYZImageryConfig,
)
from mapcv.data import MapcvDataset
from mapcv.downloader import (
    URL_TEMPLATES,
    download_region,
    resolve_url_template,
    stitch_region,
)
from mapcv.infer import predict_raster
from mapcv.labels import (
    ClassMap,
    GeomWithClass,
    load_vector_labels,
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
from mapcv.pipeline import (
    GenerateResult,
    Patch,
    generate,
    iter_patches,
    run_generate,
    run_split,
    split,
)
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
    "DEFAULT_SENTINEL2_L2A_BANDS",
    "MANIFEST_VERSION",
    "URL_TEMPLATES",
    "ClassMap",
    "ClassificationOptions",
    "DetectionOptions",
    "EOPFZarrImageryConfig",
    "GenerateResult",
    "GeoTiffImageryConfig",
    "GeomWithClass",
    "ImageryConfig",
    "InstanceOptions",
    "LabelsConfig",
    "Manifest",
    "ManifestEntry",
    "ManifestMismatchError",
    "MapcvConfig",
    "MapcvDataset",
    "Patch",
    "PatchSummary",
    "Plan",
    "RasterClass",
    "RasterLabelsConfig",
    "RegionConfig",
    "SamplerConfig",
    "SourceRecord",
    "SplitterConfig",
    "StacCogImageryConfig",
    "TargetRecord",
    "WriterConfig",
    "XYZImageryConfig",
    "__version__",
    "bounds",
    "download_region",
    "generate",
    "iter_patches",
    "load_or_create_manifest",
    "load_vector_labels",
    "parse_geojson",
    "parse_kml",
    "plan",
    "predict_raster",
    "rasterize",
    "resolve_url_template",
    "run_generate",
    "run_split",
    "sample_patches",
    "sample_patches_at_anchors",
    "snap_bbox",
    "split",
    "split_dataset",
    "stitch_region",
    "transform_to_mercator",
    "write_patches",
]
