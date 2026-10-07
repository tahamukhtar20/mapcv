"""CRS helpers for the georeferenced outputs: GeoTIFF tags, world files, WGS-84 footprints."""

from __future__ import annotations

import math
import re
from collections.abc import Callable

import numpy as np
import numpy.typing as npt

Transform = tuple[float, float, float, float, float, float]
LonLat = Callable[
    [npt.NDArray[np.float64], npt.NDArray[np.float64]],
    tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]],
]

_EPSG = re.compile(r"^\s*EPSG:(\d+)\s*$", re.IGNORECASE)
_EARTH_RADIUS = 6378137.0  # EPSG:3857's sphere


def epsg_code(crs: str) -> int:
    """The EPSG code of a CRS written ``"EPSG:<code>"``.

    Raises:
        ValueError: The CRS is anything else, which GeoTIFF GeoKeys cannot name.
    """
    match = _EPSG.match(crs)
    if match is None:
        raise ValueError(
            f"cannot write georeferencing for CRS {crs!r}: only CRSs given as an EPSG code "
            "(for example 'EPSG:32633') are supported"
        )
    code = int(match.group(1))
    if not 1 <= code < 32767:
        raise ValueError(f"EPSG code {code} is out of range for a GeoTIFF GeoKey (1..32766)")
    return code


def is_geographic(code: int) -> bool:
    """Whether EPSG ``code`` is a geographic (angular) CRS rather than a projected one.

    Uses pyproj when it is installed; without it, the 4000-4999 range, which holds
    the EPSG geographic CRSs in common use (4326, 4269, 4258, ...).
    """
    try:
        from pyproj import CRS
    except ImportError:
        return 4000 <= code < 5000
    return bool(CRS.from_epsg(code).is_geographic)


def world_file_text(transform: Transform) -> str:
    """The six lines of a world file (``.pgw``, ``.jgw``) for a pixel-corner ``transform``.

    World files locate the *centre* of the top-left pixel, so the tie point moves
    by half a pixel. Lines: x pixel size, y rotation, x rotation, y pixel size
    (negative for north-up), x and y of that centre.
    """
    a, b, c, d, e, f = transform
    cx = c + 0.5 * (a + b)
    cy = f + 0.5 * (d + e)
    return "".join(f"{value!r}\n" for value in (a, d, b, e, cx, cy))


def _mercator_to_lonlat(
    x: npt.NDArray[np.float64], y: npt.NDArray[np.float64]
) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
    lon = np.degrees(x / _EARTH_RADIUS)
    lat = np.degrees(2.0 * np.arctan(np.exp(y / _EARTH_RADIUS)) - math.pi / 2.0)
    return lon, lat


def _identity(
    x: npt.NDArray[np.float64], y: npt.NDArray[np.float64]
) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
    return x, y


def to_lonlat(crs: str) -> LonLat:
    """A vectorised ``(x, y) -> (lon, lat)`` converter from ``crs`` to WGS-84.

    Web Mercator and WGS-84 need nothing else; other CRSs need pyproj (installed
    with ``pip install mapcv[zarr]``).

    Raises:
        RuntimeError: pyproj is missing and ``crs`` is neither EPSG:3857 nor EPSG:4326.
    """
    normalised = crs.strip().upper()
    if normalised == "EPSG:3857":
        return _mercator_to_lonlat
    if normalised == "EPSG:4326":
        return _identity
    try:
        from pyproj import Transformer
    except ImportError as exc:
        raise RuntimeError(
            f"converting {crs} coordinates to WGS-84 needs pyproj; "
            "install it with 'pip install mapcv[zarr]'"
        ) from exc
    transformer = Transformer.from_crs(crs, "EPSG:4326", always_xy=True)

    def convert(
        x: npt.NDArray[np.float64], y: npt.NDArray[np.float64]
    ) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
        lon, lat = transformer.transform(x, y)
        return np.asarray(lon, dtype=np.float64), np.asarray(lat, dtype=np.float64)

    return convert
