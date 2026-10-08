"""``patches.geojson``: one WGS-84 footprint polygon per patch, for QGIS and GIS joins."""

from __future__ import annotations

import filecmp
import json
import os
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import numpy as np
import numpy.typing as npt

from mapcv._georef import to_lonlat
from mapcv.manifest import Manifest
from mapcv.splitter import SplitLists

FOOTPRINTS_FILENAME = "patches.geojson"
# Nine decimal degrees is about a tenth of a millimetre.
_DECIMALS = 9


def _split_of(split_lists: SplitLists | None) -> dict[str, str]:
    if split_lists is None:
        return {}
    split: dict[str, str] = {}
    for name, names in (
        ("train", split_lists.train),
        ("val", split_lists.val),
        ("test", split_lists.test),
    ):
        split.update(dict.fromkeys(names, name))
    return split


def _rings(manifest: Manifest, patch_size: int) -> npt.NDArray[np.float64]:
    """``(N, 5, 2)`` closed lon/lat rings of every patch, counter-clockwise."""
    transform = manifest.source.transform
    if transform is None or manifest.source.crs is None:
        raise ValueError("the manifest records no transform or CRS (made by mapcv 0.1)")
    a, b, c, d, e, f = transform
    rows = np.array([entry["row"] for entry in manifest.patches], dtype=np.float64)
    cols = np.array([entry["col"] for entry in manifest.patches], dtype=np.float64)
    # Bottom-left, bottom-right, top-right, top-left of a north-up patch.
    corners = [(0, patch_size), (patch_size, patch_size), (patch_size, 0), (0, 0)]
    if a * e - b * d > 0:  # a south-up grid flips the winding
        corners.reverse()
    corners.append(corners[0])
    convert = to_lonlat(manifest.source.crs)
    rings = np.empty((len(rows), len(corners), 2), dtype=np.float64)
    for index, (u, v) in enumerate(corners):
        x = c + a * (cols + u) + b * (rows + v)
        y = f + d * (cols + u) + e * (rows + v)
        rings[:, index, 0], rings[:, index, 1] = convert(x, y)
    return np.round(rings, _DECIMALS)


def _features(manifest: Manifest, patch_size: int, split_lists: SplitLists | None) -> Iterator[str]:
    split = _split_of(split_lists)
    rings = _rings(manifest, patch_size)
    for entry, ring in zip(manifest.patches, rings):
        name = manifest.patch_name(entry)
        properties: dict[str, Any] = {
            "filename": name,
            "split": split.get(name),
            "row": entry["row"],
            "col": entry["col"],
            "padded": entry["padded"],
            "empty_ratio": entry["summary"].get("empty_ratio"),
        }
        class_pixels = entry["summary"].get("class_pixels")
        if class_pixels is not None:
            properties["class_pixels"] = class_pixels
        yield json.dumps(
            {
                "type": "Feature",
                "properties": properties,
                "geometry": {"type": "Polygon", "coordinates": [ring.tolist()]},
            },
            separators=(",", ":"),
        )


def write_footprints(manifest: Manifest, split_lists: SplitLists | None, path: Path) -> None:
    """Write the footprint index of ``manifest``'s patches to ``path`` (atomically).

    One Feature per patch, in manifest order. Each footprint is the patch's four
    corners converted to WGS-84 longitude/latitude (a closed counter-clockwise
    ring), with the properties ``filename`` (the name split lists use), ``split``
    (``"train"``, ``"val"``, ``"test"``, or ``null`` without a split or for a
    dropped patch), ``row``, ``col``, ``padded``, ``empty_ratio`` and, for
    segmentation, ``class_pixels``.

    Raises:
        ValueError: The manifest records no transform, CRS or patch size.
        RuntimeError: The CRS needs pyproj and it is not installed.
    """
    patch_size = (manifest.sampler or {}).get("patch_size")
    if patch_size is None:
        raise ValueError("the manifest records no patch size (made by mapcv 0.1)")
    tmp = path.with_name(path.name + ".tmp")
    with tmp.open("w", encoding="utf-8", newline="\n") as out:
        out.write('{"type":"FeatureCollection","features":[')
        first = True
        for feature in _features(manifest, int(patch_size), split_lists):
            out.write(("\n" if first else ",\n") + feature)
            first = False
        out.write("\n]}\n")
    if path.is_file() and filecmp.cmp(tmp, path, shallow=False):
        # Same bytes: leave the file as it is, so a run that changes nothing leaves the
        # dataset (and its checksums and modification times) untouched.
        tmp.unlink()
        return
    os.replace(tmp, path)
