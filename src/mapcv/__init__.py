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

__all__ = [
    "URL_TEMPLATES",
    "download_region",
    "download_region_strips",
    "hello",
    "bounds",
    "snap_bbox",
    "iter_tile_strips",
    "resolve_url_template",
]
