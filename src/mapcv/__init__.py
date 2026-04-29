"""
mapcv - A satellite imagery dataset creation tool for segmentation.
"""

from mapcv._mapcv_rs import bounds, hello, snap_bbox
from mapcv.downloader import (
    URL_TEMPLATES,
    download_region,
    download_region_strips,
    iter_tile_strips,
    resolve_url_template,
)
from mapcv.labels import (
    ClassMap,
    GeomWithClass,
    parse_geojson,
    parse_kml,
    transform_to_mercator,
)

__all__ = [
    "URL_TEMPLATES",
    "ClassMap",
    "GeomWithClass",
    "download_region",
    "download_region_strips",
    "hello",
    "bounds",
    "snap_bbox",
    "iter_tile_strips",
    "parse_geojson",
    "parse_kml",
    "resolve_url_template",
    "transform_to_mercator",
]
