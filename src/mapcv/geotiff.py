"""Read GeoTIFF and Cloud Optimized GeoTIFF rasters with mapcv's own Rust reader.

A thin typed wrapper over the Rust reader in ``mapcv._mapcv_rs``. It opens
local files, ``http(s)://`` URLs and public ``s3://bucket/key`` objects (read
with HTTP range requests, anonymous access only), reports the georeferencing
GDAL would report for the same file, and decodes only the tiles or strips
under a requested pixel window::

    from mapcv.geotiff import GeoTiff

    tif = GeoTiff("ortho.tif")
    tif.info.epsg, tif.info.transform
    data, valid = tif.read_window(0, 512, 0, 512)  # (512, 512, bands)

Conventions:

* ``transform`` is ``(a, b, c, d, e, f)`` in rasterio's ``Affine`` order:
  ``x = a * col + b * row + c`` and ``y = d * col + e * row + f`` for pixel
  corners. For ``PixelIsPoint`` files it is shifted by half a pixel, as GDAL
  and rasterio do, so it is corner-based too.
* Windows are half-open ``[row0, row1) x [col0, col1)`` and may extend past
  the raster; pixels outside it are filled with the nodata value (0 without
  one) and are ``False`` in ``valid``.
* ``bands`` are 0-based (rasterio's band indexes are 1-based).
"""

from __future__ import annotations

import os
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any, Literal

import numpy as np
import numpy.typing as npt

from mapcv._inputs import check_regular_file
from mapcv._mapcv_rs import GeoTiff as _RustGeoTiff

Transform = tuple[float, float, float, float, float, float]

DEFAULT_CACHE_BYTES = 64 * 1024 * 1024


@dataclass(frozen=True)
class GeoTiffInfo:
    """Structure and georeferencing of a GeoTIFF."""

    width: int
    height: int
    count: int
    dtype: np.dtype[Any]
    #: EPSG code of the CRS; ``None`` when the file has no CRS or one that is
    #: not identified by an EPSG code (see ``crs_error``).
    epsg: int | None
    crs_error: str | None
    crs_citation: str | None
    transform: Transform | None
    raster_type: Literal["area", "point"]
    nodata: float | None
    tiled: bool
    #: ``(rows, cols)`` of one tile or strip, as rasterio's ``block_shapes``.
    block_size: tuple[int, int]
    #: ``(height, width)`` of each overview, largest first.
    overviews: tuple[tuple[int, int], ...]
    compression: str
    predictor: int
    planar: bool
    photometric: str
    byte_order: Literal["little", "big"]
    bigtiff: bool

    def overview_transform(self, overview: int) -> Transform | None:
        """The transform of overview level ``overview`` (0 = full resolution).

        Scaled from the full-resolution transform by the size ratios, as GDAL
        does for overview datasets.
        """
        if overview < 0 or overview > len(self.overviews):
            raise ValueError(
                f"overview {overview} does not exist: the file has "
                f"{len(self.overviews)} overview level(s)"
            )
        if self.transform is None or overview == 0:
            return self.transform
        height, width = self.overviews[overview - 1]
        sx = self.width / width
        sy = self.height / height
        a, b, c, d, e, f = self.transform
        return (a * sx, b * sy, c, d * sx, e * sy, f)


class GeoTiff:
    """A GeoTIFF or COG opened for windowed reading.

    ``source`` is a local path, an ``http(s)://`` URL or ``s3://bucket/key``
    for a public bucket. ``cache_bytes`` bounds the memory kept between reads
    for a remote file. A URL's host is connected wherever it resolves; with
    ``trust_host=False`` (a URL the user did not write, such as one a STAC catalog
    names) it must resolve to a public address, as a redirect target must.

    Raises ``FileNotFoundError`` for a missing local file, ``ValueError`` for a
    file that is not a GeoTIFF the reader supports (unsupported compression,
    sample type, ...) and ``RuntimeError`` when reading fails.
    """

    def __init__(
        self,
        source: str | os.PathLike[str],
        *,
        cache_bytes: int = DEFAULT_CACHE_BYTES,
        trust_host: bool = True,
    ) -> None:
        path = os.fspath(source)
        if "://" not in path:
            if not os.path.exists(path):
                raise FileNotFoundError(f"No such file: {path}")
            check_regular_file(path, "the GeoTIFF")
        self._inner = _RustGeoTiff(path, cache_bytes, trust_host)
        meta = self._inner.metadata()
        self.info = GeoTiffInfo(
            width=meta["width"],
            height=meta["height"],
            count=meta["count"],
            dtype=np.dtype(meta["dtype"]),
            epsg=meta["epsg"],
            crs_error=meta["crs_error"],
            crs_citation=meta["crs_citation"],
            transform=meta["transform"],
            raster_type=meta["raster_type"],
            nodata=meta["nodata"],
            tiled=meta["tiled"],
            block_size=meta["block_size"],
            overviews=tuple(meta["overviews"]),
            compression=meta["compression"],
            predictor=meta["predictor"],
            planar=meta["planar"],
            photometric=meta["photometric"],
            byte_order=meta["byte_order"],
            bigtiff=meta["bigtiff"],
        )

    @property
    def epsg(self) -> int:
        """The EPSG code of the CRS; ``ValueError`` explains why there is none."""
        if self.info.epsg is None:
            raise ValueError(self.info.crs_error or "the file has no CRS")
        return self.info.epsg

    def read_window(
        self,
        row0: int,
        row1: int,
        col0: int,
        col1: int,
        *,
        bands: Sequence[int] | None = None,
        overview: int = 0,
    ) -> tuple[npt.NDArray[Any], npt.NDArray[np.bool_]]:
        """Read rows ``row0:row1`` and columns ``col0:col1`` of overview ``overview``.

        Returns ``(data, valid)``: ``data`` is ``(rows, cols, bands)`` in the
        file's dtype and ``valid`` is ``(rows, cols)``, ``False`` where the
        window lies outside the raster. Only the tiles or strips intersecting
        the window are read and decoded (in parallel, without the GIL).
        """
        band_list = None if bands is None else [int(b) for b in bands]
        data, valid = self._inner.read_window(
            int(row0), int(row1), int(col0), int(col1), band_list, int(overview)
        )
        return data, valid
