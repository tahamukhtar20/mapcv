"""
mapcv - A satellite imagery dataset creation tool for segmentation, detection, instance segmentation and classification.
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
from mapcv.downloader import (
    URL_TEMPLATES,
    download_region,
    resolve_url_template,
    stitch_region,
)
from mapcv.infer import predict_raster
from mapcv.data import MapcvDataset
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
    "__version__",
    "GenerateResult",
    "MapcvDataset",
    "Patch",
    "Plan",
    "generate",
    "iter_patches",
    "plan",
    "predict_raster",
    "split",
    "run_generate",
    "run_split",
    "URL_TEMPLATES",
    "LabelsConfig",
    "ClassificationOptions",
    "DetectionOptions",
    "InstanceOptions",
    "RasterClass",
    "RasterLabelsConfig",
    "ImageryConfig",
    "XYZImageryConfig",
    "StacCogImageryConfig",
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
    "load_vector_labels",
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
