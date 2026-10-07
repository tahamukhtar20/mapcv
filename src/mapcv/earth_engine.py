"""Google Earth Engine imagery as XYZ tiles (``imagery.earth_engine``).

Earth Engine renders an image with visualization parameters and serves it as XYZ tiles
under a *map ID* that ``getMapId`` creates. mapcv asks for that URL every time it opens
the imagery, with the user's own Earth Engine credentials (``earthengine authenticate``
or ``ee.Authenticate()``); the URL holds the map ID, which acts as a short-lived
credential, so it is never written to the config, the manifest or the logs.

Earth Engine's terms, quotas and any billing apply to the user's own account and Cloud
project: mapcv only makes the requests the user asks for. Needs the ``gee`` extra
(``pip install "mapcv[gee]"``, the Apache-2.0 ``earthengine-api``).
"""

from __future__ import annotations

import importlib
from typing import Any

from mapcv.config import EarthEngineImageryConfig

_INSTALL_HINT = 'Earth Engine imagery needs the gee extra: pip install "mapcv[gee]"'
_LOGIN_HINT = (
    "Log in once with `earthengine authenticate` (or ee.Authenticate() in Python) and set "
    "imagery.earth_engine.project to a Cloud project registered for Earth Engine"
)


def _ee() -> Any:
    try:
        return importlib.import_module("ee")
    except ImportError as exc:
        raise RuntimeError(_INSTALL_HINT) from exc


# Google's per-pixel cloud scores for Sentinel-2 (cs_cdf: 1 = clear, 0 = cloud).
CLOUD_SCORE_PLUS = "GOOGLE/CLOUD_SCORE_PLUS/V1/S2_HARMONIZED"


def _image(
    ee: Any,
    config: EarthEngineImageryConfig,
    bounds: tuple[float, float, float, float] | None = None,
) -> Any:
    """The ee.Image to render: the asset, or the collection filtered and reduced."""
    if config.image is not None:
        return ee.Image(config.image)
    collection = ee.ImageCollection(config.collection)
    if bounds is not None:
        # Only scenes over the region: the same pixels there, far less work for Earth Engine.
        collection = collection.filterBounds(ee.Geometry.Rectangle(list(bounds)))
    if config.start is not None or config.end is not None:
        collection = collection.filterDate(config.start or "1970-01-01", config.end or "2100-01-01")
    if config.max_cloud is not None:
        collection = collection.filter(
            ee.Filter.lte(config.cloud_filter_property, config.max_cloud)
        )
    if config.cloud_score_plus is not None:
        threshold = config.cloud_score_plus
        collection = collection.linkCollection(
            ee.ImageCollection(CLOUD_SCORE_PLUS), ["cs_cdf"]
        ).map(lambda image: image.updateMask(image.select("cs_cdf").gte(threshold)))
    return getattr(collection, config.reducer)()


def tile_url(
    config: EarthEngineImageryConfig, bounds: tuple[float, float, float, float] | None = None
) -> str:
    """An XYZ ``{z}/{x}/{y}`` URL template for ``config``'s image, freshly created.

    ``bounds`` (west, south, east, north in degrees) limits a collection to the scenes
    over the region, which changes nothing there but saves Earth Engine work.

    Raises:
        RuntimeError: The ``gee`` extra is missing, or Earth Engine refused (not logged
            in, no project, an unknown asset): the message says what to do.
    """
    ee = _ee()
    try:
        ee.Initialize(project=config.project)
        map_id = _image(ee, config, bounds).getMapId(config.vis.params())
    except Exception as exc:
        what = config.image or config.collection
        raise RuntimeError(
            f"Earth Engine could not render {what!r}: {exc}. {_LOGIN_HINT}."
        ) from exc
    template = str(map_id["tile_fetcher"].url_format)
    for placeholder in ("{z}", "{x}", "{y}"):
        if placeholder not in template:
            raise RuntimeError(
                f"Earth Engine returned a tile URL without {placeholder}; "
                "upgrade earthengine-api (pip install -U earthengine-api)"
            )
    return template


def product_id(config: EarthEngineImageryConfig) -> str:
    """What the manifest records as the product: the asset, never the map URL."""
    return f"earth-engine:{config.image or config.collection}"
