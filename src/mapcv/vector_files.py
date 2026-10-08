"""Readers for GeoPackage, Shapefile and GeoParquet label files.

Each reader returns a :class:`VectorTable`: one shapely geometry per feature
(``None`` where the feature has none) in WGS-84 longitude/latitude, and the
attribute columns that were asked for. Files in another CRS are reprojected
with pyproj (``always_xy=True``: x is longitude/easting whatever the CRS
authority says about axis order). A file whose CRS is not stated is an error
rather than a guess, because a wrong guess shifts every label silently.

Every reader raises ``ValueError`` for a missing, unreadable or corrupt file,
with a message that says what to do next.
"""

from __future__ import annotations

import codecs
import difflib
import importlib
import json
import sqlite3
from collections.abc import Iterator, Sequence
from contextlib import closing
from dataclasses import dataclass, field
from itertools import zip_longest
from pathlib import Path
from typing import Any

import numpy as np
import numpy.typing as npt
import shapely
from shapely.geometry import LineString, MultiLineString, MultiPoint, MultiPolygon, Point, Polygon
from shapely.geometry.base import BaseGeometry

#: Shapefile geometry type codes (ESRI spec), by the geometry they hold.
_SHP_NULL = 0
_SHP_POINT_TYPES = frozenset({1, 11, 21})
_SHP_POLYLINE_TYPES = frozenset({3, 13, 23})
_SHP_POLYGON_TYPES = frozenset({5, 15, 25})
_SHP_MULTIPOINT_TYPES = frozenset({8, 18, 28})

#: GeoPackage binary header: size of the envelope by the envelope indicator in the flags.
_GPKG_ENVELOPE_BYTES = {0: 0, 1: 32, 2: 48, 3: 48, 4: 64}

_READ_CHUNK = 50_000


@dataclass
class VectorTable:
    """Features of a vector file: geometries in lon/lat and the requested columns."""

    #: One entry per feature; ``None`` for a feature without geometry.
    geometries: list[BaseGeometry | None] = field(default_factory=list)
    #: Attribute values by column name, each a list as long as ``geometries``.
    columns: dict[str, list[Any]] = field(default_factory=dict)
    #: Features whose geometry is a kind the file format stores but mapcv cannot read
    #: (a shapefile MultiPatch); they are counted as lacking polygon geometry.
    unreadable: set[int] = field(default_factory=set)


# --- shared helpers ------------------------------------------------------------------


def _check_file(path: Path) -> None:
    if not path.is_file():
        raise FileNotFoundError(f"label file not found: {path}")


def _require_columns(
    wanted: Sequence[str] | None, available: Sequence[str], what: str, kind: str
) -> list[str]:
    """The columns to read: ``wanted`` (checked against ``available``), or all of them."""
    if wanted is None:
        return list(available)
    for name in wanted:
        if name not in available:
            shown = ", ".join(repr(column) for column in available) or "none"
            raise ValueError(
                f"labels.label_field '{name}' is not a column of {what}. "
                f"The {kind} has these columns: {shown}.{did_you_mean(name, available)}"
            )
    return list(wanted)


def did_you_mean(name: str, choices: Sequence[str]) -> str:
    """`` Did you mean 'x'?`` for the choice closest to a misspelled ``name`` (also one
    that differs in case only), else a hint to check the spelling."""
    close = difflib.get_close_matches(name, list(choices), n=1)
    if not close:
        lowered = {choice.lower(): choice for choice in choices}
        close = [lowered[name.lower()]] if name.lower() in lowered else []
    return f" Did you mean '{close[0]}'?" if close else " Check the spelling and the case."


def _wgs84_lonlat(crs: Any) -> bool:
    from pyproj import CRS

    return bool(crs.equals(CRS.from_epsg(4326), ignore_axis_order=True))


def _to_wgs84(geometries: list[BaseGeometry | None], crs: Any, what: str) -> None:
    """Reproject ``geometries`` in place from ``crs`` to EPSG:4326 longitude/latitude."""
    if _wgs84_lonlat(crs):
        return
    from pyproj import Transformer
    from pyproj.exceptions import ProjError

    try:
        transformer = Transformer.from_crs(crs, "EPSG:4326", always_xy=True)
    except ProjError as exc:
        raise ValueError(
            f"{what}: cannot reproject from its CRS ({crs.name}) to WGS-84: {exc}. "
            "Reproject the file to EPSG:4326 first."
        ) from exc

    def project(coords: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
        x, y = transformer.transform(coords[:, 0], coords[:, 1])
        out: npt.NDArray[np.float64] = np.column_stack((x, y))
        if not np.isfinite(out).all():
            raise ValueError(
                f"{what}: some coordinates are outside the area of use of its CRS "
                f"({crs.name}) and cannot be reprojected to WGS-84. Check that the file's "
                "CRS is the one its coordinates are really in."
            )
        return out

    present = [index for index, geometry in enumerate(geometries) if geometry is not None]
    if not present:
        return
    array = np.empty(len(present), dtype=object)
    array[:] = [geometries[index] for index in present]
    for index, geometry in zip(present, shapely.transform(array, project)):
        geometries[index] = geometry


def _parse_wkb(blobs: Sequence[bytes | None], what: str) -> list[BaseGeometry | None]:
    """Decode WKB blobs (``None`` stays ``None``)."""
    present = [index for index, blob in enumerate(blobs) if blob is not None]
    result: list[BaseGeometry | None] = [None] * len(blobs)
    if not present:
        return result
    array = np.empty(len(present), dtype=object)
    array[:] = [blobs[index] for index in present]
    try:
        decoded = shapely.from_wkb(array)
    except Exception as exc:
        raise ValueError(
            f"{what}: a geometry is not valid WKB ({exc}). The file may be corrupt, or hold "
            "curved geometries (CircularString and the like), which mapcv does not read; "
            "convert them to straight segments first."
        ) from exc
    for index, geometry in zip(present, decoded):
        result[index] = geometry
    return result


def _crs_from(source: Any, what: str) -> Any:
    from pyproj import CRS
    from pyproj.exceptions import CRSError

    try:
        return CRS.from_user_input(source)
    except CRSError as exc:
        raise ValueError(f"{what}: its CRS definition cannot be understood ({exc}).") from exc


# --- GeoPackage ----------------------------------------------------------------------


def _quote(identifier: str) -> str:
    return '"' + identifier.replace('"', '""') + '"'


# A GeoPackage is an SQLite database, and its tables may be views: SQL that runs when the
# layer is read, and that may never end (a recursive view yields rows forever). Reading
# is bounded by the size of the file: at most this many SQLite instructions (checked
# every _PROGRESS_STEP) and rows per byte, plus a floor, which is far more than reading
# any real table takes.
_PROGRESS_STEP = 10_000
_INSTRUCTIONS_PER_BYTE = 500
_INSTRUCTIONS_FLOOR = 20_000_000
_ROWS_PER_BYTE = 0.25
_ROWS_FLOOR = 10_000


@dataclass
class _GpkgBudget:
    """How much work reading a GeoPackage may take, and whether it ran out."""

    instructions: int
    rows: int
    exceeded: bool = False

    def tick(self) -> int:
        """SQLite's progress handler: a non-zero result interrupts the statement."""
        self.instructions -= _PROGRESS_STEP
        if self.instructions < 0:
            self.exceeded = True
            return 1
        return 0


def _budget_message(what: str) -> str:
    return (
        f"{what}: reading it did not finish within the work a file of this size can need, "
        "so its layer is probably a view whose query does not end. Export the layer to a "
        "new GeoPackage as a table (QGIS: Export, Save Features As; or ogr2ogr) and read "
        "that."
    )


def _open_gpkg(path: Path) -> tuple[sqlite3.Connection, _GpkgBudget]:
    _check_file(path)
    size = path.stat().st_size
    wal = path.with_name(path.name + "-wal")
    if wal.is_file():
        size += wal.stat().st_size
    budget = _GpkgBudget(
        instructions=_INSTRUCTIONS_FLOOR + _INSTRUCTIONS_PER_BYTE * size,
        rows=_ROWS_FLOOR + int(_ROWS_PER_BYTE * size),
    )
    try:
        connection = sqlite3.connect(f"{path.resolve().as_uri()}?mode=ro", uri=True)
    except sqlite3.Error as exc:  # pragma: no cover - connect() is lazy and rarely fails
        raise ValueError(f"{path.name}: cannot open the GeoPackage ({exc}).") from exc
    connection.set_progress_handler(budget.tick, _PROGRESS_STEP)
    return connection, budget


def _srs_id(value: Any, table: str, path: Path) -> int:
    """The ``srs_id`` of a layer as an integer; the spec makes it one, but SQLite stores
    any value."""
    if isinstance(value, int) and not isinstance(value, bool):
        return value
    if isinstance(value, str) and value.strip().lstrip("+-").isdigit() and value.isascii():
        return int(value)
    shown = "missing (NULL)" if value is None else f"{value!r}, not an integer"
    raise ValueError(
        f"{path.name} (layer '{table}'): its srs_id in gpkg_geometry_columns is {shown}, so "
        "the layer's CRS is unknown. Assign the CRS in QGIS (Layer, Set Layer CRS, then "
        "export again) or with ogr2ogr -a_srs."
    )


def _gpkg_layers(
    connection: sqlite3.Connection, path: Path, budget: _GpkgBudget
) -> list[tuple[str, str, Any]]:
    """Feature tables as ``(table, geometry column, srs_id)``, sorted by table name. The
    ``srs_id`` is as stored; :func:`_srs_id` checks it for the layer that is read."""
    try:
        rows = connection.execute(
            "SELECT c.table_name, g.column_name, g.srs_id "
            "FROM gpkg_contents AS c JOIN gpkg_geometry_columns AS g "
            "ON c.table_name = g.table_name WHERE c.data_type = 'features' "
            "ORDER BY c.table_name"
        ).fetchmany(budget.rows + 1)
    except sqlite3.Error as exc:
        if budget.exceeded:
            raise ValueError(_budget_message(path.name)) from None
        raise ValueError(
            f"{path.name} is not a readable GeoPackage ({exc}). Check that the file is a "
            ".gpkg and is not truncated; re-export it from QGIS or ogr2ogr if unsure."
        ) from exc
    if len(rows) > budget.rows:
        raise ValueError(_budget_message(path.name))
    return [(str(name), str(column), srs_id) for name, column, srs_id in rows]


def gpkg_layer_names(path: Path) -> list[str]:
    """Names of the feature tables (layers) of a GeoPackage, sorted.

    Raises:
        ValueError: The file is not a readable GeoPackage.
    """
    connection, budget = _open_gpkg(path)
    with closing(connection):
        return [name for name, _, _ in _gpkg_layers(connection, path, budget)]


def _gpkg_crs(connection: sqlite3.Connection, srs_id: int, what: str) -> Any:
    if srs_id in (0, -1):
        raise ValueError(
            f"{what}: the layer has no CRS (srs_id {srs_id}, 'undefined'). mapcv will not "
            "guess one; assign the real CRS in QGIS (Layer, Set Layer CRS, then export again) "
            "or with ogr2ogr -a_srs."
        )
    try:
        row = connection.execute(
            "SELECT organization, organization_coordsys_id, definition "
            "FROM gpkg_spatial_ref_sys WHERE srs_id = ?",
            (srs_id,),
        ).fetchone()
    except sqlite3.Error as exc:
        raise ValueError(f"{what}: cannot read gpkg_spatial_ref_sys ({exc}).") from exc
    if row is None:
        raise ValueError(
            f"{what}: the layer's srs_id {srs_id} is not in gpkg_spatial_ref_sys, so its CRS "
            "is unknown. Re-export the file with a CRS."
        )
    organization, code, definition = row
    from pyproj.exceptions import CRSError

    if isinstance(organization, str) and organization.upper() != "NONE" and code is not None:
        try:
            return _crs_from(f"{organization}:{code}", what)
        except ValueError:
            pass  # not a code pyproj knows: fall back to the stored definition
    if isinstance(definition, str) and definition.strip().lower() not in ("", "undefined"):
        try:
            return _crs_from(definition, what)
        except (ValueError, CRSError):
            pass
    raise ValueError(
        f"{what}: the CRS of srs_id {srs_id} ({organization}:{code}) is not one mapcv can "
        "interpret. Reproject the layer to EPSG:4326 and export it again."
    )


def _gpkg_order(connection: sqlite3.Connection, table: str, info: list[Any]) -> str:
    """The ``ORDER BY`` clause that gives a feature table its stable, file order.

    Feature order decides which polygon wins where polygons overlap and the order and IDs
    of instances, and SQLite returns rows in any order without ``ORDER BY``. A feature
    table has an integer primary key (``fid``): rows are ordered by it, as GDAL does. A
    table without one is ordered by ``rowid``.
    """
    keys = [row for row in info if row[5]]
    if len(keys) == 1 and str(keys[0][2]).upper() == "INTEGER":
        return f" ORDER BY {_quote(str(keys[0][1]))}"
    try:
        connection.execute(f"SELECT rowid FROM {_quote(table)} LIMIT 0")
    except sqlite3.OperationalError:
        # A WITHOUT ROWID table with no integer key (not a valid GeoPackage feature table)
        # has no row number to order by; SQLite returns it in primary-key order anyway.
        return ""
    return " ORDER BY rowid"


def _gpkg_wkb(blob: Any, what: str) -> bytes | None:
    """The WKB inside a GeoPackage geometry blob, or ``None`` for an empty geometry."""
    if blob is None:
        return None
    if not isinstance(blob, (bytes, bytearray)) or len(blob) < 8 or blob[:2] != b"GP":
        raise ValueError(
            f"{what}: a geometry is not GeoPackage binary (it must start with 'GP'). "
            "The file may be corrupt or not a GeoPackage."
        )
    version, flags = blob[2], blob[3]
    if version != 0:
        raise ValueError(f"{what}: GeoPackage binary version {version} is not supported.")
    if flags & 0x20:
        raise ValueError(
            f"{what}: the layer uses ExtendedGeoPackageBinary geometries, which mapcv does not "
            "read. Re-export it with standard geometries."
        )
    envelope = _GPKG_ENVELOPE_BYTES.get((flags >> 1) & 0x07)
    if envelope is None:
        raise ValueError(f"{what}: a geometry has an invalid envelope indicator.")
    if flags & 0x10:
        return None
    start = 8 + envelope
    if len(blob) <= start:
        raise ValueError(f"{what}: a geometry blob is truncated.")
    return bytes(blob[start:])


def read_gpkg(
    path: Path, layer: str | None = None, fields: Sequence[str] | None = None
) -> VectorTable:
    """Read one feature table of a GeoPackage.

    Args:
        path: The ``.gpkg`` file.
        layer: The feature table to read. Optional when the file has just one.
        fields: Attribute columns to read; ``None`` reads every non-key column.

    Raises:
        ValueError: Not a GeoPackage, a missing or ambiguous layer, an unknown column,
            an undefined CRS, or a corrupt geometry.
    """
    what = f"{path.name}"
    connection, budget = _open_gpkg(path)
    with closing(connection):
        layers = _gpkg_layers(connection, path, budget)
        names = [name for name, _, _ in layers]
        if not layers:
            raise ValueError(f"{what} has no feature tables, so there are no labels in it.")
        if layer is None:
            if len(layers) > 1:
                raise ValueError(
                    f"{what} has {len(layers)} layers: {', '.join(names)}. "
                    "Set labels.layer to the one with the labels."
                )
            table, geometry_column, srs_id = layers[0]
        else:
            chosen = [entry for entry in layers if entry[0] == layer]
            if not chosen:
                raise ValueError(
                    f"{what} has no layer '{layer}'. Its layers are: {', '.join(names)}. "
                    "Set labels.layer to one of them."
                )
            table, geometry_column, srs_id = chosen[0]
        what = f"{path.name} (layer '{table}')"
        crs = _gpkg_crs(connection, _srs_id(srs_id, table, path), what)
        try:
            info = connection.execute(f"PRAGMA table_info({_quote(table)})").fetchall()
        except sqlite3.Error as exc:  # pragma: no cover - PRAGMA table_info does not fail
            raise ValueError(f"{what}: cannot read the table ({exc}).") from exc
        attributes = [str(row[1]) for row in info if str(row[1]) != geometry_column and not row[5]]
        columns = _require_columns(fields, attributes, f"layer '{table}'", "layer")
        select = ", ".join(_quote(name) for name in [geometry_column, *columns])
        order = _gpkg_order(connection, table, info)
        blobs: list[bytes | None] = []
        values: dict[str, list[Any]] = {name: [] for name in columns}
        try:
            cursor = connection.execute(f"SELECT {select} FROM {_quote(table)}{order}")
            while True:
                rows = cursor.fetchmany(_READ_CHUNK)
                if not rows:
                    break
                if len(blobs) + len(rows) > budget.rows:
                    raise ValueError(_budget_message(what))
                for row in rows:
                    blobs.append(_gpkg_wkb(row[0], what))
                    for name, value in zip(columns, row[1:]):
                        values[name].append(value)
        except sqlite3.Error as exc:
            if budget.exceeded:
                raise ValueError(_budget_message(what)) from None
            raise ValueError(
                f"{what}: cannot read the features ({exc}). The file may be corrupt."
            ) from exc
    geometries = _parse_wkb(blobs, what)
    _to_wgs84(geometries, crs, what)
    return VectorTable(geometries, values)


# --- Shapefile -----------------------------------------------------------------------


def _sidecar(path: Path, suffix: str) -> Path | None:
    """The ``.dbf``/``.prj``/... file next to a ``.shp``, whatever the case of its suffix."""
    for candidate in (path.with_suffix(suffix), path.with_suffix(suffix.upper())):
        if candidate.is_file():
            return candidate
    try:
        for sibling in sorted(path.parent.iterdir()):
            if sibling.stem == path.stem and sibling.suffix.lower() == suffix and sibling.is_file():
                return sibling
    except OSError:  # pragma: no cover - an unreadable folder
        return None
    return None


def shapefile_files(path: Path) -> list[Path]:
    """The files that make up a shapefile (the ``.shp`` first), those that exist."""
    files = [path]
    for suffix in (".shx", ".dbf", ".prj", ".cpg"):
        sidecar = _sidecar(path, suffix)
        if sidecar is not None:
            files.append(sidecar)
    return files


def _shapefile_encoding(cpg: Path | None) -> str:
    """The attribute encoding: the ``.cpg`` file's, else UTF-8."""
    if cpg is None:
        return "utf-8"
    name = cpg.read_text(encoding="ascii", errors="replace").strip()
    lowered = name.lower().removeprefix("ansi").strip()
    for candidate in (name, lowered, f"cp{lowered}" if lowered.isdigit() else lowered):
        try:
            return codecs.lookup(candidate).name
        except LookupError:
            continue
    raise ValueError(
        f"{cpg.name} names the encoding '{name}', which Python does not know. "
        "Replace its content with a codec name such as UTF-8 or ISO-8859-1."
    )


def _parts(record: Any) -> list[npt.NDArray[np.float64]]:
    """The vertex arrays (``(n, 2)``, x and y only) of a shape's parts (rings or lines)."""
    points = np.asarray(record.points, dtype=np.float64)
    if points.size == 0:
        return []
    points = points.reshape(len(record.points), -1)[:, :2]
    starts = [int(start) for start in record.parts] or [0]
    ends = [*starts[1:], len(points)]
    if starts != sorted(starts) or starts[0] < 0 or starts[-1] > len(points):
        raise ValueError("the part offsets of a shape are invalid")
    return [points[start:end] for start, end in zip(starts, ends)]


def organize_rings(rings: list[npt.NDArray[np.float64]]) -> BaseGeometry | None:
    """Polygons with holes from the rings of a shapefile polygon shape.

    The format only says that exterior rings run clockwise and holes counter-clockwise,
    and writers do not always follow it; as GDAL does, containment decides instead of the
    winding: a ring inside an even number of other rings is an exterior, one inside an odd
    number is a hole of the smallest ring around it (so an island in a hole works).
    """
    shells = [Polygon(ring) for ring in rings if len(ring) >= 3]
    if not shells:
        return None
    if len(shells) == 1:
        return shells[0]
    inner, outer = shapely.STRtree(shells).query(shells, predicate="within")
    keep = inner != outer
    inner, outer = inner[keep], outer[keep]
    depth = np.bincount(inner, minlength=len(shells))
    areas = shapely.area(shells)
    holes: dict[int, list[Any]] = {}
    for index in np.flatnonzero(depth % 2 == 1):
        around = outer[inner == index]
        parent = min(around, key=lambda candidate: (-depth[candidate], areas[candidate]))
        holes.setdefault(int(parent), []).append(shells[int(index)].exterior.coords)
    polygons = [
        Polygon(shells[int(index)].exterior.coords, holes.get(int(index), []))
        for index in np.flatnonzero(depth % 2 == 0)
    ]
    return polygons[0] if len(polygons) == 1 else MultiPolygon(polygons)


class _SingleRing:
    """A polygon shape with one ring: built in bulk after the read (much faster)."""

    def __init__(self, ring: npt.NDArray[np.float64]) -> None:
        self.ring = ring


def _shape_geometry(record: Any) -> BaseGeometry | _SingleRing | None:
    """A shapely geometry (x and y only) from a pyshp shape; ``None`` for a null shape.

    A polygon with a single ring is returned as :class:`_SingleRing`, for the caller to
    build with the others in one vectorized call.

    Raises:
        TypeError: A shape type mapcv cannot read (MultiPatch), or a shape without usable
            geometry.
    """
    shape_type = int(record.shapeType)
    if shape_type == _SHP_NULL:
        return None
    if shape_type in _SHP_POINT_TYPES | _SHP_MULTIPOINT_TYPES:
        parts = _parts(record)
        if not parts:
            raise TypeError("a shape without points")
        if shape_type in _SHP_POINT_TYPES:
            return Point(parts[0][0])
        return MultiPoint(parts[0])
    if shape_type in _SHP_POLYLINE_TYPES:
        lines = [part for part in _parts(record) if len(part) >= 2]
        if not lines:
            raise TypeError("a line without two points")
        return LineString(lines[0]) if len(lines) == 1 else MultiLineString(lines)
    if shape_type in _SHP_POLYGON_TYPES:
        rings = _parts(record)
        if len(rings) == 1 and len(rings[0]) >= 3:
            return _SingleRing(rings[0])
        polygon = organize_rings(rings)
        if polygon is None:
            raise TypeError("a polygon without a ring")
        return polygon
    raise TypeError(f"shape type {shape_type}")


def _build_single_rings(table: VectorTable, pending: list[tuple[int, _SingleRing]]) -> None:
    """Fill the polygons of single-ring shapes into ``table`` with one vectorized call."""
    if not pending:
        return
    coordinates = np.concatenate([item.ring for _, item in pending])
    indices = np.repeat(np.arange(len(pending)), [len(item.ring) for _, item in pending])
    polygons = shapely.polygons(shapely.linearrings(coordinates, indices=indices))
    for (index, _), polygon in zip(pending, polygons):
        table.geometries[index] = polygon


def read_shapefile(path: Path, fields: Sequence[str] | None = None) -> VectorTable:
    """Read a Shapefile (``.shp`` with its ``.dbf``, ``.prj`` and ``.cpg``) with pyshp.

    Args:
        path: The ``.shp`` file; the files next to it are found by name, whatever the
            case of their suffix.
        fields: Attribute columns to read; ``None`` reads all. Needs the ``.dbf``.

    Raises:
        ValueError: A missing ``.prj`` (the CRS would be unknown), an unreadable file, an
            unknown column or an attribute encoding that does not decode.
    """
    import shapefile

    _check_file(path)
    prj = _sidecar(path, ".prj")
    if prj is None:
        raise ValueError(
            f"{path.name} has no .prj file, so its CRS is unknown and mapcv will not guess "
            f"one: a wrong guess shifts every label. Put the .prj next to it ({path.stem}.prj), "
            "or assign the CRS with QGIS (Layer, Set Layer CRS, then export again) or "
            "ogr2ogr -a_srs EPSG:xxxx."
        )
    try:
        # utf-8-sig: a .prj saved by a Windows editor may start with a byte-order mark.
        wkt = prj.read_text(encoding="utf-8-sig", errors="replace").strip()
    except OSError as exc:  # pragma: no cover - the file vanished or is unreadable
        raise ValueError(f"{prj.name}: cannot read it ({exc}).") from exc
    crs = _crs_from(wkt, prj.name)
    dbf = _sidecar(path, ".dbf")
    cpg = _sidecar(path, ".cpg")
    encoding = _shapefile_encoding(cpg)
    shx = _sidecar(path, ".shx")

    table = VectorTable()
    handles: list[Any] = []
    try:
        shp_file = path.open("rb")
        handles.append(shp_file)
        dbf_file = dbf.open("rb") if dbf is not None else None
        if dbf_file is not None:
            handles.append(dbf_file)
        shx_file = shx.open("rb") if shx is not None else None
        if shx_file is not None:
            handles.append(shx_file)
        try:
            # Open the files here and pass them in, so pyshp never treats a path as a URL.
            reader = shapefile.Reader(shp=shp_file, dbf=dbf_file, shx=shx_file, encoding=encoding)
            handles.append(reader)
            # Without a .dbf there are no columns, so a named label field is reported missing.
            available = [] if dbf_file is None else [str(item[0]) for item in reader.fields[1:]]
            columns = _require_columns(fields, available, path.name, "shapefile")
            table.columns = {name: [] for name in columns}
            _read_shapefile_features(reader, dbf_file is not None, columns, table, path.name)
        except ValueError:
            raise
        except Exception as exc:
            if "decode" in str(exc).lower():
                raise ValueError(
                    f"{path.name}: its attributes do not decode as {encoding} ({exc}). "
                    f"Put the file's real encoding in {path.stem}.cpg (for example "
                    "ISO-8859-1 or windows-1252)."
                ) from exc
            raise ValueError(
                f"{path.name}: cannot read the shapefile ({type(exc).__name__}: {exc}). "
                "The file may be corrupt or truncated."
            ) from exc
    except OSError as exc:  # pragma: no cover - a file vanished or is unreadable
        raise ValueError(f"{path.name}: cannot read the shapefile ({exc}).") from exc
    finally:
        for handle in reversed(handles):
            handle.close()
    _to_wgs84(table.geometries, crs, path.name)
    return table


def _read_shapefile_features(
    reader: Any, has_dbf: bool, columns: list[str], table: VectorTable, name: str
) -> None:
    """Fill ``table`` from ``reader``, keeping shapes and attribute records in step.

    A record deleted in the dbf is dropped with its shape, as GDAL does.
    """
    missing = object()
    pairs: Iterator[tuple[Any, Any]]
    if has_dbf:
        records = reader.iterRecords(fields=columns, deleted_as_None=True)
        pairs = zip_longest(reader.iterShapes(), records, fillvalue=missing)
    else:
        pairs = ((shape, None) for shape in reader.iterShapes())
    pending: list[tuple[int, _SingleRing]] = []
    for shape, record in pairs:
        if shape is missing or record is missing:
            raise ValueError(
                f"{name}: the .shp and the .dbf hold different numbers of features. "
                "The file may be corrupt; re-export it."
            )
        if has_dbf and record is None:
            continue  # deleted in the dbf
        try:
            geometry = _shape_geometry(shape)
        except TypeError:
            geometry = None
            table.unreadable.add(len(table.geometries))
        except ValueError as exc:
            raise ValueError(
                f"{name}: a shape is corrupt ({exc}). The file may be truncated; re-export it."
            ) from exc
        if isinstance(geometry, _SingleRing):
            pending.append((len(table.geometries), geometry))
            geometry = None
        table.geometries.append(geometry)
        for index, column in enumerate(columns):
            table.columns[column].append(record[index])
    _build_single_rings(table, pending)


# --- GeoParquet ----------------------------------------------------------------------

_PARQUET_INSTALL = (
    "GeoParquet labels need the pyarrow package: install it with "
    "pip install 'mapcv[parquet]' (or pip install pyarrow)."
)


def _pyarrow() -> tuple[Any, Any]:
    try:
        return importlib.import_module("pyarrow"), importlib.import_module("pyarrow.parquet")
    except ImportError as exc:
        raise ValueError(_PARQUET_INSTALL) from exc


def _geo_metadata(schema: Any, name: str) -> tuple[str, dict[str, Any]]:
    """``(primary column, its metadata)`` from the file's ``geo`` key."""
    raw = (schema.metadata or {}).get(b"geo")
    if raw is None:
        raise ValueError(
            f"{name} is a Parquet file without GeoParquet 'geo' metadata, so mapcv cannot tell "
            "which column holds the geometry or its CRS. Write it with geopandas "
            "(GeoDataFrame.to_parquet) or another GeoParquet writer."
        )
    try:
        geo = json.loads(raw)
        primary = geo["primary_column"]
        column = geo["columns"][primary]
        if not isinstance(column, dict):
            raise TypeError("column metadata is not an object")
    except (ValueError, KeyError, TypeError) as exc:
        raise ValueError(
            f"{name}: its GeoParquet 'geo' metadata is malformed ({exc}). "
            "Re-write the file with a GeoParquet writer."
        ) from exc
    return str(primary), column


def read_geoparquet(path: Path, fields: Sequence[str] | None = None) -> VectorTable:
    """Read a GeoParquet file (WKB geometry) with pyarrow, an optional dependency.

    Args:
        path: The ``.parquet`` or ``.geoparquet`` file.
        fields: Attribute columns to read; ``None`` reads every scalar column.

    Raises:
        ValueError: pyarrow is not installed, the file is not GeoParquet or not WKB, its
            CRS is stated as null (unknown), or it is corrupt.
    """
    # A missing file is reported as missing, with or without pyarrow.
    _check_file(path)
    pa, pq = _pyarrow()
    try:
        parquet = pq.ParquetFile(str(path))
        schema = parquet.schema_arrow
    except Exception as exc:
        raise ValueError(
            f"{path.name}: cannot read it as a Parquet file ({exc}). "
            "The file may be corrupt or truncated."
        ) from exc
    primary, meta = _geo_metadata(schema, path.name)
    encoding = meta.get("encoding")
    if encoding != "WKB":
        raise ValueError(
            f"{path.name}: geometry column '{primary}' uses the {encoding!r} encoding; mapcv "
            "reads WKB. Re-write it with geopandas (GeoDataFrame.to_parquet(..., "
            "geometry_encoding='WKB')) or convert it with ogr2ogr."
        )
    if meta.get("edges", "planar") != "planar":
        raise ValueError(
            f"{path.name}: geometry column '{primary}' has {meta.get('edges')!r} edges; mapcv "
            "treats edges as planar straight lines. Densify the geometries and re-write the file."
        )
    if "crs" not in meta:
        crs = _crs_from("OGC:CRS84", path.name)  # the GeoParquet default
    elif meta["crs"] is None:
        raise ValueError(
            f"{path.name}: its GeoParquet metadata sets crs to null, so the CRS is unknown and "
            "mapcv will not guess one: a wrong guess shifts every label. Re-write the file "
            "with the real CRS (GeoDataFrame.set_crs(...).to_parquet(...))."
        )
    else:
        crs = _crs_from(meta["crs"], path.name)
    if primary not in schema.names:
        raise ValueError(f"{path.name}: the geometry column '{primary}' is not in the file.")

    scalar = [
        field.name
        for field in schema
        if field.name != primary
        and (
            pa.types.is_integer(field.type)
            or pa.types.is_floating(field.type)
            or pa.types.is_boolean(field.type)
            or pa.types.is_string(field.type)
            or pa.types.is_large_string(field.type)
            or pa.types.is_dictionary(field.type)
            or pa.types.is_decimal(field.type)
            or pa.types.is_date(field.type)
            or pa.types.is_timestamp(field.type)
        )
    ]
    # A named field is read whatever its type; the wizard's all-columns read takes scalars.
    available = scalar if fields is None else [name for name in schema.names if name != primary]
    columns = _require_columns(fields, available, path.name, "file")
    blobs: list[bytes | None] = []
    values: dict[str, list[Any]] = {name: [] for name in columns}
    try:
        for batch in parquet.iter_batches(batch_size=_READ_CHUNK, columns=[primary, *columns]):
            blobs.extend(batch.column(0).to_pylist())
            for index, name in enumerate(columns, start=1):
                values[name].extend(batch.column(index).to_pylist())
    except Exception as exc:
        raise ValueError(
            f"{path.name}: cannot read its features ({exc}). The file may be corrupt."
        ) from exc
    geometries = _parse_wkb(blobs, path.name)
    _to_wgs84(geometries, crs, path.name)
    return VectorTable(geometries, values)


__all__ = [
    "VectorTable",
    "gpkg_layer_names",
    "read_geoparquet",
    "read_gpkg",
    "read_shapefile",
    "shapefile_files",
]
