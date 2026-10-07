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


def _image(ee: Any, config: EarthEngineImageryConfig) -> Any:
    if config.image is not None:
        return ee.Image(config.image)
    collection = ee.ImageCollection(config.collection)
    if config.start is not None or config.end is not None:
        collection = collection.filterDate(config.start or "1970-01-01", config.end or "2100-01-01")
    return getattr(collection, config.reducer)()


def tile_url(config: EarthEngineImageryConfig) -> str:
    """An XYZ ``{z}/{x}/{y}`` URL template for ``config``'s image, freshly created.

    Raises:
        RuntimeError: The ``gee`` extra is missing, or Earth Engine refused (not logged
            in, no project, an unknown asset): the message says what to do.
    """
    ee = _ee()
    try:
        ee.Initialize(project=config.project)
        map_id = _image(ee, config).getMapId(config.vis.params())
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
