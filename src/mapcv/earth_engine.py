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

# Earth Engine reports everything as an ``ee.EEException`` that carries only a message, so
# the kind of failure is read from the text. These are the texts of login and project
# errors (``ee.data.get_persistent_credentials``, ``ee.Initialize`` and the Cloud API): no
# stored or expired credentials, no project, an unknown or unregistered project, a project
# without the Earth Engine API, or no permission to use the project.
_LOGIN_ERRORS = re.compile(
    r"authori[sz]e access|authenticat|credential|reauth|invalid_grant|oauth|refresh token"
    r"|no project found|not initialized|not registered|not signed up|signed up for earth engine"
    r"|api has not been used|api .{0,40}is disabled|serviceusage|quota project"
    r"|permission[^.]{0,80}\bproject|project[^.]{0,80}(not found|does not exist|is invalid|deleted)"
    r"|permission denied[^.]{0,40}(project|resource)|status code 40[13]",
    re.IGNORECASE,
)


def _is_login_problem(exc: BaseException, *, initialized: bool) -> bool:
    """Whether ``exc`` is about logging in or the Cloud project, not about the image.

    Before ``ee.Initialize`` succeeds anything may be a login problem, so it is assumed
    to be (the hint stays when unsure). After it, Earth Engine's own ``EEException`` about
    the image (a wrong band, an unknown asset, a timeout) is shown as it is, unless its
    text says the login or the project is the trouble. An error of another kind there
    (a lost connection, a Google auth error) keeps the hint.
    """
    if _LOGIN_ERRORS.search(str(exc)) or not initialized:
        return True
    return not any(cls.__name__ == "EEException" for cls in type(exc).__mro__)


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


def tile_url(
    config: EarthEngineImageryConfig, bounds: tuple[float, float, float, float] | None = None
) -> str:
    """An XYZ ``{z}/{x}/{y}`` URL template for ``config``'s image, freshly created.

    ``bounds`` (west, south, east, north in degrees) limits a collection to the scenes
    over that box, which changes nothing inside it but saves Earth Engine work; pixels
    outside it may miss scenes, so it must cover every tile that will be read.

    Raises:
        RuntimeError: The ``gee`` extra is missing, or Earth Engine refused (not logged
            in, no project, an unknown asset): the message says what to do, with the login hint
            only when logging in or the project is the problem.
    """
    ee = _ee()
    initialized = False
    try:
        ee.Initialize(project=config.project)
        initialized = True
        map_id = _image(ee, config, bounds).getMapId(config.vis.params())
    except Exception as exc:
        what = config.image or config.collection
        # Earth Engine's messages often end in a period already.
        message = f"Earth Engine could not render {what!r}: {str(exc).strip().rstrip('.')}."
        if _is_login_problem(exc, initialized=initialized):
            message += f" {_LOGIN_HINT}."
        raise RuntimeError(message) from exc
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
