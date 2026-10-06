"""Top-level Pydantic configuration models and YAML loading."""

from __future__ import annotations

import math
import os
import re
import unicodedata
from pathlib import Path
from typing import Annotated, Any, Dict, Iterable, List, Literal, Optional, Tuple, Union
from urllib.parse import unquote, urlsplit

import yaml
from pydantic import (
    BaseModel,
    ConfigDict,
    Discriminator,
    Field,
    SerializerFunctionWrapHandler,
    Tag,
    field_validator,
    model_serializer,
    model_validator,
)

from mapcv.downloader import URL_TEMPLATES
from mapcv.labels import VECTOR_LABEL_SUFFIXES, _normalize_label
from mapcv.sampler import SamplerConfig
from mapcv.splitter import SplitterConfig
from mapcv.writer import WriterConfig


DEFAULT_SENTINEL2_L2A_BANDS: List[str] = [
    "b01",
    "b02",
    "b03",
    "b04",
    "b05",
    "b06",
    "b07",
    "b08",
    "b8a",
    "b09",
    "b11",
    "b12",
]


_RASTER_LABEL_SUFFIXES = frozenset({".tif", ".tiff"})


def eopf_local_path(path: str) -> Optional[Path]:
    """Return the filesystem path of a local EOPF product, or ``None`` for a remote URL."""
    parsed = urlsplit(path)
    if parsed.scheme == "" or (len(parsed.scheme) == 1 and parsed.scheme.isalpha()):
        # Plain path, including Windows drive paths such as C:\data\S2.zarr.
        return Path(path)
    if parsed.scheme == "file":
        local = unquote(parsed.path)
        if re.match(r"^/[A-Za-z]:", local):  # file:///C:/data/S2.zarr
            local = local[1:]
        return Path(local)
    return None


def _validate_eopf_path(path: str) -> str:
    if eopf_local_path(path) is not None:
        return path
    parsed = urlsplit(path)
    if parsed.scheme not in ("https", "s3"):
        raise ValueError(
            "imagery.path must be a local path, file://, https://, or anonymous s3:// URL"
        )
    if parsed.username or parsed.password or parsed.query or parsed.fragment:
        raise ValueError(
            "imagery.path must not contain credentials, query strings, or fragments; "
            "private-store authentication is not supported"
        )
    return path


_LOOPBACK_HOSTS = frozenset({"localhost", "127.0.0.1", "::1"})


def _check_geotiff_location(path: str, field: str) -> str:
    """A GeoTIFF/COG location: local path, ``file://``, ``https://``, anonymous ``s3://``.

    Plain ``http://`` is accepted only for a loopback host (a local test server): the
    rest of the internet gets the same rules as EOPF products.
    """
    if not path.strip():
        raise ValueError(f"{field} must not be empty")
    if eopf_local_path(path) is not None:
        return path
    parsed = urlsplit(path)
    loopback_http = parsed.scheme == "http" and (parsed.hostname or "") in _LOOPBACK_HOSTS
    if parsed.scheme not in ("https", "s3") and not loopback_http:
        raise ValueError(f"{field} must be a local path, file://, https://, or anonymous s3:// URL")
    if parsed.username or parsed.password or parsed.query or parsed.fragment:
        raise ValueError(
            f"{field} must not contain credentials, query strings, or fragments; "
            "private-store authentication is not supported"
        )
    return path


def _validate_geotiff_path(path: str) -> str:
    return _check_geotiff_location(path, "imagery.path")


def _validate_label_raster_path(path: str) -> str:
    return _check_geotiff_location(path, "labels.path")


_REMOVED_SOURCES = {
    "google_satellite": "Google does not permit downloading its imagery for datasets",
    "osm": (
        "the OpenStreetMap Foundation tile servers forbid bulk downloading; use a commercial "
        "or self-hosted OSM tile service through imagery.url_template"
    ),
}


def _validate_tile_source(source: Optional[str]) -> Optional[str]:
    if source is None or source in URL_TEMPLATES:
        return source
    if source in _REMOVED_SOURCES:
        raise ValueError(
            f"the built-in '{source}' source was removed in mapcv 0.2.0: "
            f"{_REMOVED_SOURCES[source]} (see MIGRATION.md and PROVIDERS.md)"
        )
    raise ValueError(
        f"unknown tile source '{source}'; built-in sources: {', '.join(sorted(URL_TEMPLATES))}"
    )


_TEMPLATE_PLACEHOLDER = re.compile(r"\{([^{}]*)\}")


def _validate_url_template(template: Optional[str]) -> Optional[str]:
    if template is None:
        return None
    if urlsplit(template).scheme not in ("http", "https"):
        raise ValueError("url_template must be an http:// or https:// URL")
    placeholders = set(_TEMPLATE_PLACEHOLDER.findall(template))
    missing = {"x", "y", "z"} - placeholders
    if missing:
        raise ValueError(
            f"url_template must contain {{x}}, {{y}} and {{z}}; missing "
            f"{', '.join('{' + name + '}' for name in sorted(missing))}"
        )
    unknown = placeholders - {"x", "y", "z"}
    if unknown:
        hint = " (replace {s} with one subdomain, e.g. 'a')" if "s" in unknown else ""
        raise ValueError(
            f"url_template has unsupported placeholder(s) "
            f"{', '.join('{' + name + '}' for name in sorted(unknown))}{hint}"
        )
    return template


def _join(base: Path, value: object) -> object:
    """``base / value`` for a relative path string, else ``value`` unchanged."""
    if not isinstance(value, str) or not value or Path(value).expanduser().is_absolute():
        return value
    return os.path.normpath(base / value)


def area_polygons(
    path: Path, name_field: Optional[str] = None, layer: Optional[str] = None
) -> List[Tuple[Any, str]]:
    """The polygons of an area-of-interest file in WGS-84 lon/lat, with their region names:
    the ``name_field`` value, or the polygon's number (1, 2, ...) in file order."""
    from mapcv.labels import load_vector_labels

    try:
        raw, names = load_vector_labels(path, name_field, layer=layer)
    except ValueError as exc:
        if "distinct values" in str(exc):
            raise ValueError(
                f"region.name_field '{name_field}' has more than 255 distinct names; give "
                "polygons of one region the same name, or leave name_field out to number them"
            ) from None
        raise
    if name_field is None:
        return [(geometry, str(index)) for index, (geometry, _) in enumerate(raw, start=1)]
    by_id = {class_id: name for name, class_id in names.items()}
    return [(geometry, by_id[class_id]) for geometry, class_id in raw]


def _resolve_relative_paths(data: Dict[str, Any], base: Path) -> None:
    region = data.get("region")
    if isinstance(region, dict) and isinstance(region.get("path"), str):
        region["path"] = _join(base, region["path"])
    labels = data.get("labels")
    if isinstance(labels, dict) and "path" in labels:
        path = labels["path"]
        # A label raster may be a URL, which is not a path to resolve.
        if not isinstance(path, str) or urlsplit(path).scheme == "":
            labels["path"] = _join(base, path)
    if isinstance(labels, dict) and "annotated_area" in labels:
        labels["annotated_area"] = _join(base, labels["annotated_area"])
    for file in labels.get("files") or [] if isinstance(labels, dict) else []:
        if isinstance(file, dict) and "path" in file:
            file["path"] = _join(base, file["path"])
    change = data.get("change")
    if isinstance(change, dict):
        for key in ("before", "after"):
            label_set = change.get(key)
            if isinstance(label_set, dict) and "path" in label_set:
                label_set["path"] = _join(base, label_set["path"])
    writer = data.get("writer")
    if isinstance(writer, dict) and "staging_dir" in writer:
        writer["staging_dir"] = _join(base, writer["staging_dir"])
    imagery = data.get("imagery")
    for source in imagery if isinstance(imagery, list) else [imagery]:
        if isinstance(source, dict) and isinstance(source.get("path"), str):
            path = source["path"]
            if urlsplit(path).scheme == "" and eopf_local_path(path) is not None:
                source["path"] = _join(base, path)


# A source name is a folder name (``Images/<name>/``) and a key of each patch's ``files``.
_SOURCE_NAME = re.compile(r"[a-z0-9][a-z0-9_-]{0,31}")
# Keys of a patch's ``files`` that are not source names.
_RESERVED_SOURCE_NAMES = frozenset({"mask"})


def _validate_source_name(name: Optional[str]) -> Optional[str]:
    if name is None:
        return name
    if not _SOURCE_NAME.fullmatch(name):
        raise ValueError(
            f"imagery name '{name}' must be 1-32 lowercase letters, digits, '_' or '-', "
            "starting with a letter or digit (it names the folder Images/<name>/)"
        )
    if name in _RESERVED_SOURCE_NAMES or name.endswith("_world"):
        raise ValueError(f"imagery name '{name}' is reserved; choose another name")
    return name


_REGION_ZOOM_REMOVED = (
    "`region.zoom` was moved to `imagery.zoom` in mapcv 0.2 and removed in 0.3; "
    "delete it and set `imagery: {type: xyz, zoom: ..., source: ...}` instead "
    "(see MIGRATION.md)"
)

_TILES_REMOVED = (
    "`tiles:` was replaced by `imagery:` in mapcv 0.2 and removed in 0.3; "
    "move it under `imagery: {type: xyz, zoom: ..., source: ...}` and delete `region.zoom` "
    "(see MIGRATION.md)"
)


# Web Mercator (EPSG:3857) is undefined at the poles; XYZ tiles stop at this latitude.
WEB_MERCATOR_MAX_LATITUDE = 85.05112878


class RegionConfig(BaseModel):
    """The area of interest: a WGS-84 bounding box, or polygons in a vector file.

    With ``path`` (an area-of-interest file, any vector label format) the box is the
    polygons' bounds, and only patches that touch a polygon are made. Each polygon is
    a region: named by its ``name_field`` value, or numbered 1, 2, ... in file order.
    """

    # Unknown keys are errors, so typos and newer-version options are not silently ignored.
    model_config = ConfigDict(extra="forbid")

    west: float
    south: float
    east: float
    north: float
    path: Optional[Path] = None
    name_field: Optional[str] = None
    layer: Optional[str] = None

    @model_validator(mode="before")
    @classmethod
    def _reject_removed_zoom(cls, raw: Any) -> Any:
        if isinstance(raw, dict) and "zoom" in raw:
            raise ValueError(_REGION_ZOOM_REMOVED)
        return raw

    @model_validator(mode="before")
    @classmethod
    def _bounds_from_area(cls, raw: Any) -> Any:
        if not isinstance(raw, dict) or raw.get("path") is None:
            return raw
        if any(key in raw for key in ("west", "south", "east", "north")):
            raise ValueError(
                "give region either path (area-of-interest polygons) or west/south/east/north, "
                "not both"
            )
        path = Path(raw["path"])
        if path.suffix.lower() not in VECTOR_LABEL_SUFFIXES:
            raise ValueError(
                "region.path must be a polygon file (.geojson, .json, .kml, .gpkg, .shp, "
                f".parquet or .geoparquet), got '{path.name}'"
            )
        if not path.exists():
            raise ValueError(f"region.path not found: {path}")
        polygons = area_polygons(path, raw.get("name_field"), raw.get("layer"))
        if not polygons:
            raise ValueError(f"region.path '{path.name}' holds no polygon")
        west = min(geometry.bounds[0] for geometry, _ in polygons)
        south = min(geometry.bounds[1] for geometry, _ in polygons)
        east = max(geometry.bounds[2] for geometry, _ in polygons)
        north = max(geometry.bounds[3] for geometry, _ in polygons)
        return {**raw, "west": west, "south": south, "east": east, "north": north}

    @model_serializer(mode="wrap")
    def _omit_unset_area(self, handler: SerializerFunctionWrapHandler) -> Dict[str, Any]:
        data: Dict[str, Any] = handler(self)
        for key in ("path", "name_field", "layer"):
            if data.get(key, 0) is None:
                del data[key]
        return data

    @model_validator(mode="after")
    def _validate_bounds(self) -> "RegionConfig":
        for name in ("west", "east"):
            value = getattr(self, name)
            if not -180.0 <= value <= 180.0:
                raise ValueError(f"region.{name} must be a longitude in -180..180, got {value}")
        for name in ("south", "north"):
            value = getattr(self, name)
            if not -90.0 <= value <= 90.0:
                raise ValueError(
                    f"region.{name} must be a latitude in -90..90, got {value}; check that "
                    "longitude and latitude are not swapped"
                )
        if self.path is None and (self.name_field is not None or self.layer is not None):
            raise ValueError("region.name_field and region.layer need region.path")
        if self.west >= self.east:
            raise ValueError("region.west must be less than region.east")
        if self.south >= self.north:
            raise ValueError("region.south must be less than region.north")
        return self


class XYZImageryConfig(BaseModel):
    """XYZ tile imagery source and fetch settings."""

    # Unknown keys are errors, so typos and newer-version options are not silently ignored.
    model_config = ConfigDict(extra="forbid")

    type: Literal["xyz"] = "xyz"
    # Required when imagery is a list of sources: the folder Images/<name>/.
    name: Optional[str] = None
    zoom: int = Field(ge=1, le=22)
    source: Optional[str] = None
    url_template: Optional[str] = None
    max_connections: int = Field(default=16, ge=1)
    policy: Literal["strict", "lenient", "ignore"] = "lenient"
    max_failed_ratio: float = Field(default=0.05, ge=0.0, le=1.0)
    strip_rows: int = Field(default=4, ge=1)

    _check_source = field_validator("source")(_validate_tile_source)
    _check_name = field_validator("name")(_validate_source_name)
    _check_template = field_validator("url_template")(_validate_url_template)

    @model_validator(mode="after")
    def _require_source_or_template(self) -> "XYZImageryConfig":
        if self.source is None and self.url_template is None:
            raise ValueError("imagery: provide either 'source' or 'url_template'")
        if self.source is not None and self.url_template is not None:
            raise ValueError("imagery: set 'source' or 'url_template', not both")
        return self


class EOPFZarrImageryConfig(BaseModel):
    """One local or anonymous public Sentinel-2 L2A EOPF Zarr product."""

    # Unknown keys are errors, so typos and newer-version options are not silently ignored.
    model_config = ConfigDict(extra="forbid")

    type: Literal["eopf_zarr"] = "eopf_zarr"
    # Required when imagery is a list of sources: the folder Images/<name>/.
    name: Optional[str] = None
    path: str
    resolution: Literal[10, 20, 60] = 10
    bands: List[str] = Field(default_factory=lambda: list(DEFAULT_SENTINEL2_L2A_BANDS))
    chunk_rows: int = Field(default=1024, ge=1)

    _check_path = field_validator("path")(_validate_eopf_path)
    _check_name = field_validator("name")(_validate_source_name)

    @model_validator(mode="after")
    def _validate_bands(self) -> "EOPFZarrImageryConfig":
        normalized = [band.lower() for band in self.bands]
        if not normalized:
            raise ValueError("imagery.bands must contain at least one variable")
        if len(normalized) != len(set(normalized)):
            raise ValueError("imagery.bands must not contain duplicates")
        self.bands = normalized
        return self


class GeoTiffImageryConfig(BaseModel):
    """One local or remote GeoTIFF / Cloud Optimized GeoTIFF, read as it is (no resampling).

    ``bands`` are 1-based band numbers in the order they should be written
    (default: every band). ``overview`` is the overview level to read (0 = full
    resolution). ``nodata`` overrides the file's NoData value; pixels where every
    selected band equals it are treated as having no imagery.
    """

    # Unknown keys are errors, so typos and newer-version options are not silently ignored.
    model_config = ConfigDict(extra="forbid")

    type: Literal["geotiff"] = "geotiff"
    # Required when imagery is a list of sources: the folder Images/<name>/.
    name: Optional[str] = None
    path: str
    bands: Optional[List[int]] = None
    overview: int = Field(default=0, ge=0)
    nodata: Optional[float] = None
    chunk_rows: int = Field(default=1024, ge=1)

    _check_path = field_validator("path")(_validate_geotiff_path)
    _check_name = field_validator("name")(_validate_source_name)

    @field_validator("nodata")
    @classmethod
    def _finite_or_nan_nodata(cls, value: Optional[float]) -> Optional[float]:
        if value is not None and value in (float("inf"), float("-inf")):
            raise ValueError("imagery.nodata must be a number or .nan")
        return value

    @model_validator(mode="after")
    def _validate_bands(self) -> "GeoTiffImageryConfig":
        if self.bands is None:
            return self
        if not self.bands:
            raise ValueError("imagery.bands must contain at least one band (or be omitted)")
        if any(isinstance(band, bool) or band < 1 for band in self.bands):
            raise ValueError("imagery.bands are 1-based band numbers (1 is the first band)")
        if len(self.bands) != len(set(self.bands)):
            raise ValueError("imagery.bands must not contain duplicates")
        return self


ImageryConfig = Annotated[
    Union[XYZImageryConfig, EOPFZarrImageryConfig, GeoTiffImageryConfig],
    Field(discriminator="type"),
]


def _imagery_form(value: Any) -> str:
    return "source-list" if isinstance(value, list) else "single-source"


# One source, or a list of named sources. The form is chosen from the input, so a
# mistake is reported for that form only. Error locations carry the tag
# ("single-source", "source-list"); messages leave it out, as they leave out "xyz".
AnyImageryConfig = Annotated[
    Union[
        Annotated[ImageryConfig, Tag("single-source")],
        Annotated[List[ImageryConfig], Tag("source-list")],
    ],
    Discriminator(_imagery_form),
]

#: Parts of a validation error's location that are union tags, not config keys.
UNION_TAGS = frozenset({"xyz", "eopf_zarr", "geotiff", "single-source", "source-list"})


class BufferConfig(BaseModel):
    """Distances, in metres on the ground, that turn lines and points into polygons."""

    # Unknown keys are errors, so typos and newer-version options are not silently ignored.
    model_config = ConfigDict(extra="forbid")

    line: Optional[float] = Field(default=None, gt=0)
    point: Optional[float] = Field(default=None, gt=0)

    @model_validator(mode="after")
    def _one_distance(self) -> "BufferConfig":
        if self.line is None and self.point is None:
            raise ValueError("labels.buffer needs a line or point distance in metres")
        for name in ("line", "point"):
            value = getattr(self, name)
            if value is not None and not math.isfinite(value):
                raise ValueError(f"labels.buffer.{name} must be a finite number of metres")
        return self


def _check_vector_suffix(path: Path, key: str) -> Path:
    if path.suffix.lower() in _RASTER_LABEL_SUFFIXES:
        raise ValueError(
            f"{key} '{path.name}' is a raster: set labels.type: raster and map its "
            "values with labels.classes"
        )
    if path.suffix.lower() not in VECTOR_LABEL_SUFFIXES:
        raise ValueError(
            f"{key} must be a .geojson, .json, .kml, .gpkg, .shp, .parquet or "
            f".geoparquet file, got '{path.name}' (convert KMZ to KML first)"
        )
    return path


class LabelFile(BaseModel):
    """One of several vector label files (``labels.files``).

    Each feature's class is its ``label_field`` value or, with ``class``, the same
    class for the whole file (all buildings, all roads). ``layer`` and ``buffer`` work
    as under ``labels``.
    """

    # Unknown keys are errors, so typos and newer-version options are not silently ignored.
    model_config = ConfigDict(extra="forbid", populate_by_name=True)

    path: Path
    layer: Optional[str] = None
    label_field: Optional[str] = None
    class_name: Optional[str] = Field(default=None, alias="class")
    buffer: Optional[BufferConfig] = None

    @model_serializer(mode="wrap")
    def _omit_unset(self, handler: SerializerFunctionWrapHandler) -> Dict[str, Any]:
        data: Dict[str, Any] = handler(self)
        return {key: value for key, value in data.items() if value is not None}

    @field_validator("path")
    @classmethod
    def _check_suffix(cls, path: Path) -> Path:
        return _check_vector_suffix(path, "labels.files path")

    @field_validator("class_name")
    @classmethod
    def _normalize_class(cls, value: Optional[str]) -> Optional[str]:
        if value is None:
            return value
        name = _normalize_label(value)
        if name is None:
            raise ValueError("labels.files class must be a non-empty name")
        return name

    @model_validator(mode="after")
    def _one_class_source(self) -> "LabelFile":
        if (self.label_field is None) == (self.class_name is None):
            raise ValueError(
                f"labels.files entry '{self.path.name}' needs exactly one of label_field (the "
                "attribute holding each feature's class) or class (one class for the file)"
            )
        if self.layer is not None and self.path.suffix.lower() != ".gpkg":
            raise ValueError(
                f"labels.files entry '{self.path.name}': layer picks a table of a GeoPackage"
            )
        if self.buffer is not None and self.path.suffix.lower() == ".kml":
            raise ValueError(
                f"labels.files entry '{self.path.name}': buffering needs line and point "
                "features, which mapcv does not read from KML"
            )
        return self


class LabelsConfig(BaseModel):
    """Vector label file settings: polygons burned into the masks (or boxed, for detection).

    The file is GeoJSON, KML, a GeoPackage, a Shapefile or GeoParquet, chosen by its suffix.
    ``layer`` picks the table of a GeoPackage that has several.

    ``classes`` maps label values to mask IDs (1..255). Without it, integer
    labels in 1..255 are used as-is and other labels get IDs in sorted order.
    ``ignore_index`` (default 255) is written where a mask has no imagery under
    it: padding beyond the raster edge, NoData and failed tiles. ``null`` writes
    background (0) there instead, as mapcv 0.2 did.
    """

    # Unknown keys are errors, so typos and newer-version options are not silently ignored.
    model_config = ConfigDict(extra="forbid")

    type: Literal["vector"] = "vector"
    # One label file, or several in ``files`` (later files win where features overlap).
    path: Optional[Path] = None
    files: Optional[List[LabelFile]] = None
    label_field: Optional[str] = None
    classes: Optional[Dict[str, int]] = None
    all_touched: bool = False
    ignore_index: Optional[int] = Field(default=255, ge=1, le=255)
    layer: Optional[str] = None
    # Lines and points become polygons this many metres wide (lines) or across (points).
    buffer: Optional[BufferConfig] = None
    # Polygons of the area that was labeled; mask pixels outside it get ignore_index.
    annotated_area: Optional[Path] = None

    @model_serializer(mode="wrap")
    def _omit_unset_options(self, handler: SerializerFunctionWrapHandler) -> Dict[str, Any]:
        # Records and manifests written before these options existed have no such keys.
        data: Dict[str, Any] = handler(self)
        for key in ("layer", "buffer", "annotated_area", "files"):
            if key in data and data[key] is None:
                del data[key]
        if data.get("path", 0) is None:
            del data["path"]
        return data

    @property
    def label_files(self) -> List[LabelFile]:
        """Every label file in order: ``files``, or one entry made from ``path``."""
        if self.files is not None:
            return list(self.files)
        assert self.path is not None  # the validator requires path or files
        # Built without validation: one label file may have no label_field (all class 1).
        return [
            LabelFile.model_construct(
                path=self.path,
                layer=self.layer,
                label_field=self.label_field,
                class_name=None,
                buffer=self.buffer,
            )
        ]

    def keyed_files(self, prefix: str = "labels") -> List[Tuple[str, Path]]:
        """Each label file's path with its config key (``labels.path``, ``labels.files[1].path``)."""
        if self.files is None:
            assert self.path is not None
            return [(f"{prefix}.path", self.path)]
        return [
            (f"{prefix}.files[{index}].path", file.path) for index, file in enumerate(self.files)
        ]

    @property
    def first_path(self) -> Path:
        """``path``, or the first of ``files`` (for messages that name one file)."""
        return self.label_files[0].path

    @field_validator("annotated_area")
    @classmethod
    def _check_area_suffix(cls, path: Optional[Path]) -> Optional[Path]:
        if path is not None and path.suffix.lower() not in VECTOR_LABEL_SUFFIXES:
            raise ValueError(
                "labels.annotated_area must be a polygon file (.geojson, .json, .kml, .gpkg, "
                f".shp, .parquet or .geoparquet), got '{path.name}'"
            )
        return path

    @field_validator("path")
    @classmethod
    def _check_suffix(cls, path: Optional[Path]) -> Optional[Path]:
        return None if path is None else _check_vector_suffix(path, "labels.path")

    @field_validator("layer")
    @classmethod
    def _check_layer_name(cls, layer: Optional[str]) -> Optional[str]:
        if layer is not None and not layer.strip():
            raise ValueError("labels.layer must not be empty; omit it to use the only layer")
        return layer

    @field_validator("classes", mode="before")
    @classmethod
    def _normalize_classes(cls, classes: Any) -> Any:
        if not isinstance(classes, dict):
            return classes
        normalized: Dict[str, int] = {}
        for key, value in classes.items():
            name = _normalize_label(key)
            if name is None:
                raise ValueError("labels.classes keys must be non-empty label values")
            if isinstance(value, bool) or not isinstance(value, int) or not 1 <= value <= 255:
                raise ValueError(f"labels.classes['{name}'] must be an integer in 1..255")
            normalized[name] = value
        return normalized

    @model_validator(mode="after")
    def _classes_need_field(self) -> "LabelsConfig":
        if (self.path is None) == (self.files is None):
            raise ValueError("labels needs path (one label file) or files (several), and not both")
        if self.files is not None:
            if not self.files:
                raise ValueError("labels.files is empty; list at least one label file")
            for key in ("label_field", "layer", "buffer"):
                if getattr(self, key) is not None:
                    raise ValueError(
                        f"with labels.files, set {key} on each file instead of under labels"
                    )
        elif (
            self.path is not None and self.layer is not None and self.path.suffix.lower() != ".gpkg"
        ):
            raise ValueError(
                "labels.layer picks a table of a GeoPackage (.gpkg); "
                f"'{self.path.name}' has just one, so remove labels.layer"
            )
        if self.classes is not None and self.label_field is None and self.files is None:
            raise ValueError("labels.classes requires labels.label_field")
        if self.classes is not None and self.ignore_index in self.classes.values():
            raise ValueError(
                f"labels.classes uses {self.ignore_index}, which is labels.ignore_index; "
                "pick another class ID or set ignore_index to a free value (or null)"
            )
        return self


class RasterClass(BaseModel):
    """Where one label-raster value goes: mask ``id`` (0 = background) and class ``name``."""

    model_config = ConfigDict(extra="forbid")

    id: int = Field(ge=0, le=255)
    name: Optional[str] = None


class RasterLabelsConfig(BaseModel):
    """A classified label raster (GeoTIFF / COG): land cover, a previous model's output, ...

    ``classes`` maps raster values to mask IDs, either as ``value: id`` or as
    ``value: {id: ..., name: ...}``; several values may share an ID. ID 0 is
    background. After validation every entry is a :class:`RasterClass` with a
    name (``value_<v>`` when none is given, joined with ``_`` for merged values).

    Each imagery pixel takes the label value at its centre (nearest neighbour, never
    averaged), whatever the label raster's CRS, resolution and origin. Pixels outside
    the label raster, label NoData (``nodata``, default the file's) and
    ``ignore_values`` get ``ignore_index``; values missing from ``classes`` get
    background or ``ignore_index`` (``unmapped``).
    """

    # Unknown keys are errors, so typos and newer-version options are not silently ignored.
    model_config = ConfigDict(extra="forbid")

    type: Literal["raster"]
    path: str
    band: int = Field(default=1, ge=1)
    classes: Dict[int, RasterClass]
    nodata: Optional[int] = None
    ignore_values: List[int] = Field(default_factory=list)
    unmapped: Literal["background", "ignore"] = "background"
    resampling: Literal["nearest"] = "nearest"
    ignore_index: Optional[int] = Field(default=255, ge=1, le=255)

    _check_path = field_validator("path")(_validate_label_raster_path)

    @field_validator("classes", mode="before")
    @classmethod
    def _expand_short_classes(cls, classes: Any) -> Any:
        if not isinstance(classes, dict):
            return classes
        expanded: Dict[Any, Any] = {}
        for value, target in classes.items():
            if isinstance(value, bool) or isinstance(target, bool):
                raise ValueError("labels.classes maps integer raster values to integer IDs")
            expanded[value] = {"id": target} if isinstance(target, int) else target
        return expanded

    @field_validator("nodata", "ignore_values", mode="before")
    @classmethod
    def _no_booleans(cls, value: Any) -> Any:
        values = value if isinstance(value, list) else [value]
        if any(isinstance(item, bool) for item in values):
            raise ValueError("raster values must be integers")
        return value

    @model_validator(mode="after")
    def _validate_classes(self) -> "RasterLabelsConfig":
        if not self.classes:
            raise ValueError(
                "labels.classes must map at least one raster value to a mask ID, "
                "e.g. {10: {id: 1, name: tree_cover}}"
            )
        names: Dict[int, str] = {}
        for value, target in sorted(self.classes.items()):
            if target.id == 0:
                if target.name is not None:
                    raise ValueError(
                        f"labels.classes[{value}] maps to ID 0, which is background; "
                        "it takes no name"
                    )
                continue
            if target.id == self.ignore_index:
                raise ValueError(
                    f"labels.classes[{value}] uses {self.ignore_index}, which is "
                    "labels.ignore_index; pick another class ID or set ignore_index to a free "
                    "value (or null)"
                )
            if target.name is not None:
                name = _normalize_label(target.name)
                if name is None:
                    raise ValueError(f"labels.classes[{value}].name must not be empty")
                if names.setdefault(target.id, name) != name:
                    raise ValueError(
                        f"labels.classes gives ID {target.id} two names, "
                        f"'{names[target.id]}' and '{name}'"
                    )
        by_id: Dict[int, List[int]] = {}
        for value, target in sorted(self.classes.items()):
            if target.id:
                by_id.setdefault(target.id, []).append(value)
        resolved: Dict[int, str] = {}
        for class_id, values in by_id.items():
            resolved[class_id] = names.get(class_id) or "value_" + "_".join(map(str, values))
        seen: Dict[str, int] = {}
        for class_id, name in sorted(resolved.items()):
            if seen.setdefault(name, class_id) != class_id:
                raise ValueError(
                    f"labels.classes uses the name '{name}' for IDs {seen[name]} and {class_id}"
                )
        self.classes = {
            value: RasterClass(id=target.id, name=resolved.get(target.id))
            for value, target in sorted(self.classes.items())
        }
        overlap = sorted(set(self.ignore_values) & set(self.classes))
        if overlap:
            raise ValueError(
                f"labels.ignore_values and labels.classes both list the value {overlap[0]}"
            )
        if self.nodata is not None and self.nodata in self.classes:
            raise ValueError(
                f"labels.nodata is {self.nodata}, which labels.classes maps to a class"
            )
        if self.unmapped == "ignore" and self.ignore_index is None:
            raise ValueError(
                "labels.unmapped: ignore needs labels.ignore_index; set one or use "
                "unmapped: background"
            )
        return self

    def class_map(self) -> Dict[str, int]:
        """Class name to mask ID, in ID order (background is not a class)."""
        pairs = {target.name: target.id for target in self.classes.values() if target.id}
        return {
            name: class_id
            for name, class_id in sorted(pairs.items(), key=lambda item: (item[1], item[0]))
            if name is not None
        }


class ContinuousLabelsConfig(BaseModel):
    """A raster of continuous values for ``task: regression``: canopy height, biomass,
    elevation, a previous model's scores ...

    Each imagery pixel takes the value at its centre (nearest neighbour, never
    interpolated), whatever the raster's CRS, resolution and origin, the same lookup
    as classified label rasters. The target is ``value * scale + offset`` as float32.
    Pixels outside the raster, on its NoData value (``nodata``, default the file's),
    ``NaN`` or outside ``valid_min``..``valid_max`` (raw values), and pixels without
    imagery, are ``NaN``.
    """

    # Unknown keys are errors, so typos and newer-version options are not silently ignored.
    model_config = ConfigDict(extra="forbid")

    type: Literal["continuous"]
    path: str
    band: int = Field(default=1, ge=1)
    nodata: Optional[float] = None
    scale: float = 1.0
    offset: float = 0.0
    valid_min: Optional[float] = None
    valid_max: Optional[float] = None

    _check_path = field_validator("path")(_validate_label_raster_path)

    @model_validator(mode="after")
    def _validate_values(self) -> "ContinuousLabelsConfig":
        if not math.isfinite(self.scale) or self.scale == 0.0:
            raise ValueError("labels.scale must be a finite number other than 0")
        if not math.isfinite(self.offset):
            raise ValueError("labels.offset must be a finite number")
        for name in ("valid_min", "valid_max"):
            value = getattr(self, name)
            if value is not None and not math.isfinite(value):
                raise ValueError(f"labels.{name} must be a finite number")
        if (
            self.valid_min is not None
            and self.valid_max is not None
            and self.valid_min > self.valid_max
        ):
            raise ValueError("labels.valid_min must not be above labels.valid_max")
        return self


AnyLabelsConfig = Annotated[
    Union[LabelsConfig, RasterLabelsConfig, ContinuousLabelsConfig],
    Field(discriminator="type"),
]

#: Label settings read from a raster (a path or URL string, not a vector file).
RASTER_LABEL_TYPES = (RasterLabelsConfig, ContinuousLabelsConfig)


SUPPORTED_TASKS: Tuple[str, ...] = (
    "segmentation",
    "detection",
    "instance",
    "classification",
    "change",
    "regression",
)
# Tasks on the roadmap, named in the error so a config written for them fails clearly.
PLANNED_TASKS: Tuple[str, ...] = ()
# Tasks whose datasets can hold several imagery sources (``imagery`` as a list).
MULTI_SOURCE_TASKS: Tuple[str, ...] = ("segmentation", "change", "regression")

DetectionFormat = Literal["coco", "yolo"]
DETECTION_FORMATS: Tuple[DetectionFormat, ...] = ("coco", "yolo")


def _all_formats() -> List[DetectionFormat]:
    return list(DETECTION_FORMATS)


class DetectionOptions(BaseModel):
    """Settings of ``task: detection`` (the ``detection:`` block).

    One object is one label feature (a MultiPolygon is one object). Its box is the
    axis-aligned box of the part that is visible in the patch: inside the patch,
    inside the raster and over pixels that have imagery.
    """

    # Unknown keys are errors, so typos and newer-version options are not silently ignored.
    model_config = ConfigDict(extra="forbid")

    # Keep an object only if at least this fraction of its area is visible in the patch
    # (0 keeps every object with a visible area).
    min_visible: float = Field(default=0.3, ge=0.0, le=1.0)
    # Drop boxes narrower or shorter than this many pixels (slivers at patch edges).
    min_box_pixels: float = Field(default=2.0, ge=0.0)
    # Output formats, written in this order; both by default.
    formats: List[DetectionFormat] = Field(default_factory=lambda: _all_formats())
    # Side in pixels of the square box drawn around each point feature (not KML:
    # points are not read there). Unset: point features are skipped with a warning.
    point_box_size: Optional[float] = Field(default=None, gt=0.0)

    @field_validator("formats")
    @classmethod
    def _check_formats(cls, formats: List[DetectionFormat]) -> List[DetectionFormat]:
        if not formats:
            raise ValueError("detection.formats needs at least one of: coco, yolo")
        if len(formats) != len(set(formats)):
            raise ValueError("detection.formats must not contain duplicates")
        # One canonical order, so the manifest records the same options for [yolo, coco].
        return [name for name in DETECTION_FORMATS if name in formats]


class InstanceOptions(BaseModel):
    """Settings of ``task: instance`` (the ``instance:`` block).

    One instance is one label feature (a MultiPolygon is one instance, and so is a
    feature that a patch edge cuts into several pieces). Its mask is the part of the
    feature that is visible in the patch: inside the patch, inside the raster and
    over pixels that have imagery.
    """

    # Unknown keys are errors, so typos and newer-version options are not silently ignored.
    model_config = ConfigDict(extra="forbid")

    # Keep an instance only if at least this fraction of its area is visible in the patch
    # (0 keeps every instance with a visible mask).
    min_visible: float = Field(default=0.3, ge=0.0, le=1.0)
    # Drop instances whose mask has fewer pixels than this (slivers at patch edges).
    min_area: int = Field(default=4, ge=1)
    # Also write one 16-bit instance-ID PNG per patch (masks/), next to the COCO RLE masks.
    id_mask: bool = False


class ClassificationOptions(BaseModel):
    """Settings of ``task: classification`` (the ``classification:`` block).

    A patch's label is decided by how much of it each class covers. Coverage is the
    class's pixels divided by the patch's valid pixels (pixels that have imagery and
    are not ``labels.ignore_index``), counted on the mask that segmentation would
    write for the same labels.

    ``single`` gives the patch the class with the largest coverage (ties go to the
    lowest class ID); ``multi`` gives it every class whose coverage reaches
    ``min_fraction``. A class qualifies when its coverage is at least ``min_fraction``
    and above zero, so the default 0 means "any labeled pixel". A patch no class
    qualifies for is dropped (``empty: skip``) or labeled ``background``.
    """

    # Unknown keys are errors, so typos and newer-version options are not silently ignored.
    model_config = ConfigDict(extra="forbid")

    mode: Literal["single", "multi"] = "single"
    # A class needs at least this share of the patch's valid pixels (0: any labeled pixel).
    min_fraction: float = Field(default=0.0, ge=0.0, le=1.0)
    empty: Literal["skip", "background"] = "skip"


class ChangeOptions(BaseModel):
    """Settings of ``task: change`` (the ``change:`` block).

    The change mask comes either from ``labels`` (features, or a label raster, that
    mark what changed: every labeled pixel is change) or from two label sets,
    ``before`` and ``after``: a pixel changed where they differ (an object appeared or
    disappeared there or, when both sets map ``label_field`` with the same
    ``classes``, changed class). Changed pixels get ``change_value``, others 0, and
    pixels without imagery in either image the ignore value.
    """

    # Unknown keys are errors, so typos and newer-version options are not silently ignored.
    model_config = ConfigDict(extra="forbid")

    # The mask value of changed pixels: 1, or 255 for loaders that expect 0/255 masks.
    change_value: int = Field(default=1, ge=1, le=255)
    before: Optional[LabelsConfig] = None
    after: Optional[LabelsConfig] = None

    @model_validator(mode="before")
    @classmethod
    def _vector_label_sets(cls, raw: Any) -> Any:
        # Label sets are vector files; ``type: vector`` may be left out, as under ``labels``.
        if not isinstance(raw, dict):
            return raw
        out = dict(raw)
        for key in ("before", "after"):
            value = out.get(key)
            if isinstance(value, dict):
                if value.get("type", "vector") != "vector":
                    raise ValueError(
                        f"change.{key} must be vector labels (features that mark objects); "
                        "use labels with type: raster for a change raster instead"
                    )
                out[key] = {"type": "vector", **value}
        return out


# The label of classification patches that no class qualifies for (``empty: background``).
BACKGROUND_LABEL = "background"


def classification_name_problem(
    names: Iterable[str], options: ClassificationOptions
) -> Optional[str]:
    """Why a class name cannot be a classification label, or ``None`` when all can.

    Labels go to CSV and text files one per field or line, and a multi-label patch's
    labels are joined with a space, so names must not hold control characters (or
    whitespace, with ``mode: multi``). With ``empty: background`` the name
    ``background`` is taken by the label of patches without a class.
    """
    for name in names:
        if any(unicodedata.category(char).startswith("C") for char in name):
            return (
                f"class {name!r} contains a control character (a tab or a line break), which "
                "labels.csv and classes.txt cannot hold; rename the label value"
            )
        if options.mode == "multi" and len(name.split()) != 1:
            return (
                f"class {name!r} contains whitespace, which separates the labels of a patch in "
                "labels.csv in classification.mode: multi; rename the label value (for example "
                "with underscores) or use classification.mode: single"
            )
        if options.empty == "background" and name == BACKGROUND_LABEL:
            return (
                f"class {name!r} is also the label of patches without a class "
                "(classification.empty: background); rename the label value or use "
                "classification.empty: skip"
            )
    return None


def _validate_task(task: Any) -> Any:
    if not isinstance(task, str) or task in SUPPORTED_TASKS:
        return task
    supported = ", ".join(SUPPORTED_TASKS)
    planned = f" (planned: {', '.join(PLANNED_TASKS)})" if PLANNED_TASKS else ""
    if task in PLANNED_TASKS:  # pragma: no cover - none planned now
        raise ValueError(f"task '{task}' is not supported yet; supported: {supported}{planned}")
    raise ValueError(f"unknown task '{task}'; supported: {supported}{planned}")


class MapcvConfig(BaseModel):
    """Full mapcv pipeline configuration."""

    # Unknown keys are errors, so typos and newer-version options are not silently ignored.
    model_config = ConfigDict(extra="forbid")

    # What the dataset is for; decides the target each patch is annotated with.
    task: Literal[
        "segmentation", "detection", "instance", "classification", "change", "regression"
    ] = "segmentation"
    region: RegionConfig
    # One source, or a list of named sources sampled on the first one's grid.
    imagery: AnyImageryConfig
    labels: Optional[AnyLabelsConfig] = None
    sampler: SamplerConfig
    writer: WriterConfig
    split: Optional[SplitterConfig] = None
    # Options of task: detection; defaults apply when the block is omitted.
    detection: Optional[DetectionOptions] = None
    # Options of task: instance; defaults apply when the block is omitted.
    instance: Optional[InstanceOptions] = None
    # Options of task: classification; defaults apply when the block is omitted.
    classification: Optional[ClassificationOptions] = None
    # Options of task: change; defaults apply when the block is omitted.
    change: Optional[ChangeOptions] = None

    _check_task = field_validator("task", mode="before")(_validate_task)

    @model_validator(mode="before")
    @classmethod
    def _reject_removed_keys(cls, raw: Any) -> Any:
        if not isinstance(raw, dict):
            return raw
        if "tiles" in raw:
            raise ValueError(_TILES_REMOVED)
        imagery = raw.get("imagery")
        for source in imagery if isinstance(imagery, list) else [imagery]:
            if isinstance(source, dict) and "type" not in source:
                raise ValueError("imagery.type is required: 'xyz', 'eopf_zarr' or 'geotiff'")
        labels = raw.get("labels")
        if isinstance(labels, dict) and "type" not in labels:
            # Polygon labels predate labels.type; configs without it keep working.
            raw = {**raw, "labels": {"type": "vector", **labels}}
        return raw

    @model_validator(mode="after")
    def _validate_sources(self) -> "MapcvConfig":
        if not isinstance(self.imagery, list):
            if self.imagery.name is not None:
                raise ValueError(
                    "imagery.name only applies when imagery is a list of sources; remove it, "
                    "or write imagery as a list"
                )
            return self
        if not self.imagery:
            raise ValueError("imagery is an empty list; give at least one source")
        names: List[str] = []
        for index, source in enumerate(self.imagery):
            if source.name is None:
                raise ValueError(
                    f"imagery source {index + 1} has no name; every source of a list needs "
                    "one (it names the folder Images/<name>/)"
                )
            if source.name in names:
                raise ValueError(f"imagery name '{source.name}' is used twice; names must differ")
            names.append(source.name)
        if self.task not in MULTI_SOURCE_TASKS:
            raise ValueError(
                f"task: {self.task} reads one imagery source; several sources are supported "
                f"for task: {', '.join(MULTI_SOURCE_TASKS)}"
            )
        return self

    @model_validator(mode="after")
    def _validate_stacking(self) -> "MapcvConfig":
        if not self.writer.stack_sources:
            return self
        if not self.multi_source or len(self.sources) < 2:
            raise ValueError(
                "writer.stack_sources puts several imagery sources in one file per patch; "
                "write imagery as a list of two or more named sources"
            )
        if self.task not in ("segmentation", "regression"):
            raise ValueError(
                "writer.stack_sources applies to segmentation and regression datasets "
                f"(task is '{self.task}')"
            )
        if self.writer.image_format not in ("npy", "tif"):
            raise ValueError(
                "writer.stack_sources needs writer.image_format npy (T, C, H, W arrays) or tif "
                f"(T x C bands), not '{self.writer.image_format}'"
            )
        return self

    @model_validator(mode="after")
    def _validate_source_writer_pair(self) -> "MapcvConfig":
        for source in self.sources:
            # Messages name the source when there are several.
            where = f"imagery '{source.name}'" if self.multi_source else "imagery"
            self._check_source_writer_pair(source, where)
        return self

    def _check_source_writer_pair(self, imagery: Any, where: str) -> None:
        if isinstance(imagery, EOPFZarrImageryConfig):
            if self.writer.image_format not in ("npy", "tif"):
                kind = "EOPF Zarr imagery" if where == "imagery" else f"{where} (EOPF Zarr)"
                raise ValueError(f"{kind} requires writer.image_format='npy' (or 'tif')")
        elif isinstance(imagery, GeoTiffImageryConfig):
            bands = imagery.bands
            if (
                self.writer.image_format not in ("npy", "tif")
                and bands is not None
                and len(bands) not in (1, 3)
            ):
                raise ValueError(
                    f"{where}.bands selects {len(bands)} bands, but writer.image_format "
                    f"'{self.writer.image_format}' writes 1 or 3 bands of uint8; select 1 or 3 "
                    "bands or set writer.image_format: npy (or tif)"
                )
        elif self.writer.image_format == "npy":
            kind = "XYZ imagery" if where == "imagery" else f"{where} (XYZ)"
            raise ValueError(f"{kind} supports writer.image_format 'png', 'jpg' or 'tif'")
        if isinstance(imagery, XYZImageryConfig):
            limit = WEB_MERCATOR_MAX_LATITUDE
            if self.region.north > limit or self.region.south < -limit:
                raise ValueError(
                    f"XYZ tiles cover latitudes -{limit:.4f}..{limit:.4f} (Web Mercator); "
                    "shrink the region or use imagery that covers the poles"
                )

    @property
    def multi_source(self) -> bool:
        """Whether ``imagery`` is a list of named sources (patches go to ``Images/<name>/``)."""
        return isinstance(self.imagery, list)

    @property
    def sources(self) -> List[Union[XYZImageryConfig, EOPFZarrImageryConfig, GeoTiffImageryConfig]]:
        """The imagery sources in order: a list of one for a single ``imagery`` block."""
        return list(self.imagery) if isinstance(self.imagery, list) else [self.imagery]

    @property
    def primary_imagery(
        self,
    ) -> Union[XYZImageryConfig, EOPFZarrImageryConfig, GeoTiffImageryConfig]:
        """The first imagery source: its grid is the dataset's grid."""
        return self.sources[0]

    @property
    def source_names(self) -> List[str]:
        """Names of the sources: ``["image"]`` for a single ``imagery`` block."""
        if not isinstance(self.imagery, list):
            return ["image"]
        return [source.name or "" for source in self.imagery]

    @model_validator(mode="after")
    def _validate_label_options(self) -> "MapcvConfig":
        labels = self.labels
        if not isinstance(labels, LabelsConfig):
            return self
        if labels.annotated_area is not None:
            if self.task not in ("segmentation", "classification"):
                raise ValueError(
                    "labels.annotated_area marks mask pixels outside the labeled area as "
                    f"ignored; it applies to segmentation and classification, not {self.task}"
                )
            if labels.ignore_index is None:
                raise ValueError(
                    "labels.annotated_area needs labels.ignore_index: pixels outside the area "
                    "get that value (255 by default; null would make them background)"
                )
        if labels.buffer is not None:
            if labels.first_path.suffix.lower() == ".kml":
                raise ValueError(
                    "labels.buffer needs line and point features, which mapcv does not read "
                    "from KML; convert the labels to GeoJSON or GeoPackage"
                )
            if (
                self.task == "detection"
                and labels.buffer.point is not None
                and self.detection_options.point_box_size is not None
            ):
                raise ValueError(
                    "detection.point_box_size and labels.buffer.point both turn points into "
                    "shapes; keep one"
                )
        return self

    @model_validator(mode="after")
    def _validate_task_settings(self) -> "MapcvConfig":
        if isinstance(self.labels, ContinuousLabelsConfig) and self.task != "regression":
            raise ValueError(
                "labels.type: continuous holds values to predict, a regression target; set "
                f"task: regression (task is '{self.task}'), or use type: raster with classes"
            )
        if self.detection is not None and self.task != "detection":
            raise ValueError(
                f"the detection block only applies to task: detection (task is '{self.task}'); "
                "remove it or set task: detection"
            )
        if self.instance is not None and self.task != "instance":
            raise ValueError(
                f"the instance block only applies to task: instance (task is '{self.task}'); "
                "remove it or set task: instance"
            )
        if self.classification is not None and self.task != "classification":
            raise ValueError(
                "the classification block only applies to task: classification "
                f"(task is '{self.task}'); remove it or set task: classification"
            )
        if self.task == "detection":
            self._check_detection()
        if self.task == "instance":
            self._check_instance()
        if self.task == "classification":
            self._check_classification()
        if self.change is not None and self.task != "change":
            raise ValueError(
                f"the change block only applies to task: change (task is '{self.task}'); "
                "remove it or set task: change"
            )
        if self.task == "change":
            self._check_change()
        if self.task == "regression":
            self._check_regression()
        return self

    def _check_regression(self) -> None:
        if not isinstance(self.labels, ContinuousLabelsConfig):
            raise ValueError(
                "task: regression needs labels.type: continuous, a raster of the values to "
                "predict (labels: {type: continuous, path: canopy_height.tif})"
            )
        if "mask_format" not in self.writer.model_fields_set:
            # Float targets do not fit a PNG: default to a georeferenced GeoTIFF.
            self.writer.mask_format = "tif"
        elif self.writer.mask_format == "png":
            raise ValueError(
                "regression targets are float32, which PNG cannot hold; set "
                "writer.mask_format: tif or npy"
            )

    def _check_change(self) -> None:
        if not self.multi_source or len(self.sources) != 2:
            raise ValueError(
                "task: change needs two imagery sources as a list, the image before and the "
                "image after (for example names before and after)"
            )
        options = self.change_options
        before, after = options.before, options.after
        if (before is None) != (after is None):
            raise ValueError(
                "change.before and change.after come together: give both label sets, or "
                "neither and labels that mark the change"
            )
        if before is not None and after is not None:
            if self.labels is not None:
                raise ValueError(
                    "give either labels (what changed) or change.before and change.after (two "
                    "label sets whose difference is the change), not both"
                )
            if (before.label_field is None) != (after.label_field is None) or (
                before.label_field is not None
                and (before.classes is None or before.classes != after.classes)
            ):
                raise ValueError(
                    "to compare classes, change.before and change.after both need label_field "
                    "and the same classes mapping (so a class has one ID in both); without "
                    "label_field only the presence of objects is compared"
                )
            if before.ignore_index != after.ignore_index:
                raise ValueError(
                    "change.before and change.after need the same ignore_index (the mask "
                    "value of pixels without imagery)"
                )
            ignore = before.ignore_index
        elif self.labels is None:
            raise ValueError(
                "task: change needs labels that mark what changed, or change.before and "
                "change.after (two label sets whose difference is the change)"
            )
        elif isinstance(self.labels, ContinuousLabelsConfig):  # pragma: no cover - refused first
            raise ValueError("task: change needs vector labels or a classified label raster")
        else:
            ignore = self.labels.ignore_index
        if ignore is not None and options.change_value == ignore:
            raise ValueError(
                f"change.change_value {options.change_value} is also the ignore value; set "
                "labels.ignore_index (or change.before/after.ignore_index) to another value, "
                "or null"
            )

    def _check_detection(self) -> None:
        labels = self.labels
        if labels is None:
            raise ValueError("task: detection needs labels: the boxes come from the label features")
        if not isinstance(labels, LabelsConfig):  # a raster (continuous ones fail earlier)
            raise ValueError(
                "task: detection needs vector labels (one feature per object), not a label "
                "raster; use task: segmentation for labels.type: raster, or polygonize the "
                "raster's objects into a GeoJSON first"
            )
        if "ignore_index" in labels.model_fields_set:
            raise ValueError(
                "labels.ignore_index marks mask pixels without imagery and detection writes no "
                "masks; remove it (objects over no-imagery areas keep only their visible part)"
            )
        if labels.all_touched:
            raise ValueError(
                "labels.all_touched is a rasterization setting and detection does not "
                "rasterize; remove it"
            )
        if "mask_format" in self.writer.model_fields_set:
            raise ValueError(
                "writer.mask_format sets the format of segmentation masks and detection writes "
                "no masks; remove it (boxes go to COCO and YOLO files, see detection.formats)"
            )
        if self.writer.world_files and self.writer.image_format not in ("png", "jpg"):
            raise ValueError(
                "writer.world_files adds .pgw/.jgw files to PNG and JPG patches and detection "
                f"writes no masks, so with image_format '{self.writer.image_format}' it would "
                "write none; remove it (GeoTIFF patches carry their georeferencing)"
            )
        options = self.detection_options
        if options.point_box_size is not None and any(
            file.path.suffix.lower() == ".kml" for file in labels.label_files
        ):
            raise ValueError(
                "detection.point_box_size reads point features from GeoJSON; KML points are "
                "not supported, so convert the labels to GeoJSON or remove point_box_size"
            )
        if self.sampler.edge_strategy == "pad" and self.sampler.pad_mode == "reflect":
            raise ValueError(
                "sampler.pad_mode: reflect mirrors objects into the padding of edge patches, "
                "where they would have no box; use pad_mode: zero, or edge_strategy: shift "
                "or drop"
            )

    def _check_instance(self) -> None:
        labels = self.labels
        if labels is None:
            raise ValueError("task: instance needs labels: the masks come from the label features")
        if not isinstance(labels, LabelsConfig):  # a raster (continuous ones fail earlier)
            raise ValueError(
                "task: instance needs vector labels (one feature per instance), not a label "
                "raster; use task: segmentation for labels.type: raster, or polygonize the "
                "raster's objects into a GeoJSON first"
            )
        if "ignore_index" in labels.model_fields_set:
            raise ValueError(
                "labels.ignore_index marks mask pixels without imagery and instance masks have "
                "no such value (an instance keeps only its pixels over imagery); remove it"
            )
        id_mask = self.instance_options.id_mask
        if "mask_format" in self.writer.model_fields_set and not id_mask:
            raise ValueError(
                "writer.mask_format sets the format of the instance-ID masks, which are only "
                "written with instance.id_mask: true (the COCO masks are RLE inside the "
                "annotation files); set instance.id_mask: true or remove writer.mask_format"
            )
        if (
            self.writer.world_files
            and not id_mask
            and self.writer.image_format not in ("png", "jpg")
        ):
            raise ValueError(
                "writer.world_files adds .pgw/.jgw files to PNG and JPG patches and, without "
                "instance.id_mask, instance datasets write no masks, so with image_format "
                f"'{self.writer.image_format}' it would write none; remove it (GeoTIFF patches "
                "carry their georeferencing)"
            )
        if self.sampler.edge_strategy == "pad" and self.sampler.pad_mode == "reflect":
            raise ValueError(
                "sampler.pad_mode: reflect mirrors instances into the padding of edge patches, "
                "where they would have no mask; use pad_mode: zero, or edge_strategy: shift "
                "or drop"
            )

    def _check_classification(self) -> None:
        if self.labels is None:
            raise ValueError(
                "task: classification needs labels: a patch's label comes from the label "
                "coverage (vector features or a label raster)"
            )
        if "mask_format" in self.writer.model_fields_set:
            raise ValueError(
                "writer.mask_format sets the format of segmentation masks and classification "
                "writes no masks (labels go to labels.csv and labels.json); remove it"
            )
        if self.writer.world_files and self.writer.image_format not in ("png", "jpg"):
            raise ValueError(
                "writer.world_files adds .pgw/.jgw files to PNG and JPG patches and "
                f"classification writes no masks, so with image_format '{self.writer.image_format}' "
                "it would write none; remove it (GeoTIFF patches carry their georeferencing)"
            )
        options = self.classification_options
        if options.empty == "background" and self.sampler.min_label_ratio > 0.0:
            raise ValueError(
                "sampler.min_label_ratio drops patches with little labeled area before "
                "classification.empty applies, so with empty: background it would drop the "
                "background patches; set sampler.min_label_ratio: 0 (and use "
                "classification.min_fraction to set how much a class needs)"
            )
        if isinstance(self.labels, RasterLabelsConfig):
            problem = classification_name_problem(self.labels.class_map(), options)
            if problem is not None:
                raise ValueError(problem)
        if self.sampler.edge_strategy == "pad" and self.sampler.pad_mode == "reflect":
            raise ValueError(
                "sampler.pad_mode: reflect mirrors imagery into the padding of edge patches, "
                "where no label is counted, so the image would show more than the label "
                "describes; use pad_mode: zero, or edge_strategy: shift or drop"
            )

    @property
    def detection_options(self) -> DetectionOptions:
        """The ``detection`` block, or its defaults when it is omitted."""
        return self.detection if self.detection is not None else DetectionOptions()

    @property
    def instance_options(self) -> InstanceOptions:
        """The ``instance`` block, or its defaults when it is omitted."""
        return self.instance if self.instance is not None else InstanceOptions()

    @property
    def change_options(self) -> ChangeOptions:
        """The ``change`` block, or its defaults when it is omitted."""
        return self.change if self.change is not None else ChangeOptions()

    @property
    def classification_options(self) -> ClassificationOptions:
        """The ``classification`` block, or its defaults when it is omitted."""
        return self.classification if self.classification is not None else ClassificationOptions()

    @classmethod
    def from_yaml(cls, path: Union[str, "os.PathLike[str]"]) -> "MapcvConfig":
        """Load and validate a mapcv YAML file.

        Relative paths in the file (``labels.path``, ``writer.staging_dir`` and a
        local ``imagery.path``) are resolved against the file's folder, so a
        config works from any working directory.
        """
        path = Path(path)
        data = yaml.safe_load(path.read_text(encoding="utf-8"))
        if isinstance(data, dict):
            _resolve_relative_paths(data, path.parent)
        return cls.model_validate(data)
