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
import re
from typing import Any

from mapcv.config import EarthEngineImageryConfig

_INSTALL_HINT = 'Earth Engine imagery needs the gee extra: pip install "mapcv[gee]"'
_LOGIN_HINT = (
    "Log in once with `earthengine authenticate` (or ee.Authenticate() in Python) and set "
    "imagery.earth_engine.project to a Cloud project registered for Earth Engine"
)
# What Earth Engine says when the login or the Cloud project is the problem. Anything
# else (a wrong band, an unknown asset) is about the config, and its message says so.
_LOGIN_ERRORS = re.compile(
    r"authoriz|authenticat|credential|not signed up|not registered|no project found"
    r"|quota project|serviceusage|permission to use project|caller does not have permission"
    r"|permission denied on resource project|has not been used in project"
    r"|project\b[^.]*\bnot found or deleted",
    re.IGNORECASE,
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
        # Only scenes over the tiles: the same pixels there, far less work for Earth Engine.
        # BBox is a plain lon/lat box like the tiles' edges; a Rectangle of numbers would
        # be geodesic, its east-west edges bowing away from the tiles' parallels.
        collection = collection.filterBounds(ee.Geometry.BBox(*bounds))
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


def _detail(exc: BaseException) -> str:
    """Earth Engine's message on one line, without its closing period."""
    return " ".join(str(exc).split()).rstrip(". ") or type(exc).__name__


def _is_login_error(exc: BaseException) -> bool:
    # google.auth raises its own errors for a missing or expired login.
    return type(exc).__module__.startswith("google.auth") or bool(_LOGIN_ERRORS.search(str(exc)))


def tile_url(
    config: EarthEngineImageryConfig, bounds: tuple[float, float, float, float] | None = None
) -> str:
    """An XYZ ``{z}/{x}/{y}`` URL template for ``config``'s image, freshly created.

    ``bounds`` (west, south, east, north in degrees) limits a collection to the scenes
    over that box, which changes nothing inside it but saves Earth Engine work; pixels
    outside it may miss scenes, so it must cover every tile that will be read.

    Raises:
        RuntimeError: The ``gee`` extra is missing, or Earth Engine refused. The message
            is Earth Engine's own; the login hint follows it only when the login or the
            project failed, not for a wrong band or an unknown asset.
    """
    ee = _ee()
    what = config.image or config.collection
    try:
        ee.Initialize(project=config.project)
    except Exception as exc:
        # Initialize only reads the credentials and the project.
        raise RuntimeError(
            f"Earth Engine could not render {what!r}: {_detail(exc)}. {_LOGIN_HINT}."
        ) from exc
    try:
        map_id = _image(ee, config, bounds).getMapId(config.vis.params())
    except Exception as exc:
        hint = f" {_LOGIN_HINT}." if _is_login_error(exc) else ""
        raise RuntimeError(
            f"Earth Engine could not render {what!r}: {_detail(exc)}.{hint}"
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
