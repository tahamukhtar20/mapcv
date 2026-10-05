"""Top-level Pydantic configuration models and YAML loading."""

from __future__ import annotations

import os
import re
from pathlib import Path
from typing import Annotated, Any, Dict, List, Literal, Optional, Tuple, Union
from urllib.parse import unquote, urlsplit

import yaml
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from mapcv.downloader import URL_TEMPLATES
from mapcv.labels import _normalize_label
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


_LABEL_SUFFIXES = frozenset({".kml", ".geojson", ".json"})


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


def _validate_geotiff_path(path: str) -> str:
    """A GeoTIFF/COG location: local path, ``file://``, ``https://``, anonymous ``s3://``.

    Plain ``http://`` is accepted only for a loopback host (a local test server): the
    rest of the internet gets the same rules as EOPF products.
    """
    if not path.strip():
        raise ValueError("imagery.path must not be empty")
    if eopf_local_path(path) is not None:
        return path
    parsed = urlsplit(path)
    loopback_http = parsed.scheme == "http" and (parsed.hostname or "") in _LOOPBACK_HOSTS
    if parsed.scheme not in ("https", "s3") and not loopback_http:
        raise ValueError(
            "imagery.path must be a local path, file://, https://, or anonymous s3:// URL"
        )
    if parsed.username or parsed.password or parsed.query or parsed.fragment:
        raise ValueError(
            "imagery.path must not contain credentials, query strings, or fragments; "
            "private-store authentication is not supported"
        )
    return path


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


def _resolve_relative_paths(data: Dict[str, Any], base: Path) -> None:
    labels = data.get("labels")
    if isinstance(labels, dict) and "path" in labels:
        labels["path"] = _join(base, labels["path"])
    writer = data.get("writer")
    if isinstance(writer, dict) and "staging_dir" in writer:
        writer["staging_dir"] = _join(base, writer["staging_dir"])
    imagery = data.get("imagery")
    if isinstance(imagery, dict) and isinstance(imagery.get("path"), str):
        path = imagery["path"]
        if urlsplit(path).scheme == "" and eopf_local_path(path) is not None:
            imagery["path"] = _join(base, path)


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
    """Geographic bounding box in WGS-84 degrees."""

    # Unknown keys are errors, so typos and newer-version options are not silently ignored.
    model_config = ConfigDict(extra="forbid")

    west: float
    south: float
    east: float
    north: float

    @model_validator(mode="before")
    @classmethod
    def _reject_removed_zoom(cls, raw: Any) -> Any:
        if isinstance(raw, dict) and "zoom" in raw:
            raise ValueError(_REGION_ZOOM_REMOVED)
        return raw

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
    zoom: int = Field(ge=1, le=22)
    source: Optional[str] = None
    url_template: Optional[str] = None
    max_connections: int = Field(default=16, ge=1)
    policy: Literal["strict", "lenient", "ignore"] = "lenient"
    max_failed_ratio: float = Field(default=0.05, ge=0.0, le=1.0)
    strip_rows: int = Field(default=4, ge=1)

    _check_source = field_validator("source")(_validate_tile_source)
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
    path: str
    resolution: Literal[10, 20, 60] = 10
    bands: List[str] = Field(default_factory=lambda: list(DEFAULT_SENTINEL2_L2A_BANDS))
    chunk_rows: int = Field(default=1024, ge=1)

    _check_path = field_validator("path")(_validate_eopf_path)

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
    path: str
    bands: Optional[List[int]] = None
    overview: int = Field(default=0, ge=0)
    nodata: Optional[float] = None
    chunk_rows: int = Field(default=1024, ge=1)

    _check_path = field_validator("path")(_validate_geotiff_path)

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


class LabelsConfig(BaseModel):
    """Label file (KML or GeoJSON) settings.

    ``classes`` maps label values to mask IDs (1..255). Without it, integer
    labels in 1..255 are used as-is and other labels get IDs in sorted order.
    ``ignore_index`` (default 255) is written where a mask has no imagery under
    it: padding beyond the raster edge, NoData and failed tiles. ``null`` writes
    background (0) there instead, as mapcv 0.2 did.
    """

    # Unknown keys are errors, so typos and newer-version options are not silently ignored.
    model_config = ConfigDict(extra="forbid")

    path: Path
    label_field: Optional[str] = None
    classes: Optional[Dict[str, int]] = None
    all_touched: bool = False
    ignore_index: Optional[int] = Field(default=255, ge=1, le=255)

    @field_validator("path")
    @classmethod
    def _check_suffix(cls, path: Path) -> Path:
        if path.suffix.lower() not in _LABEL_SUFFIXES:
            raise ValueError(
                f"labels.path must be a .kml, .geojson, or .json file, got '{path.name}' "
                "(convert KMZ or Shapefiles to GeoJSON first)"
            )
        return path

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
        if self.classes is not None and self.label_field is None:
            raise ValueError("labels.classes requires labels.label_field")
        if self.classes is not None and self.ignore_index in self.classes.values():
            raise ValueError(
                f"labels.classes uses {self.ignore_index}, which is labels.ignore_index; "
                "pick another class ID or set ignore_index to a free value (or null)"
            )
        return self


SUPPORTED_TASKS: Tuple[str, ...] = ("segmentation",)
# Tasks on the roadmap, named in the error so a config written for them fails clearly.
PLANNED_TASKS: Tuple[str, ...] = ("detection", "instance", "classification", "change", "regression")


def _validate_task(task: Any) -> Any:
    if not isinstance(task, str) or task in SUPPORTED_TASKS:
        return task
    supported = ", ".join(SUPPORTED_TASKS)
    planned = ", ".join(PLANNED_TASKS)
    if task in PLANNED_TASKS:
        raise ValueError(
            f"task '{task}' is not supported yet; supported: {supported} (planned: {planned})"
        )
    raise ValueError(f"unknown task '{task}'; supported: {supported} (planned: {planned})")


class MapcvConfig(BaseModel):
    """Full mapcv pipeline configuration."""

    # Unknown keys are errors, so typos and newer-version options are not silently ignored.
    model_config = ConfigDict(extra="forbid")

    # What the dataset is for; decides the target each patch is annotated with.
    task: Literal["segmentation"] = "segmentation"
    region: RegionConfig
    imagery: ImageryConfig
    labels: Optional[LabelsConfig] = None
    sampler: SamplerConfig
    writer: WriterConfig
    split: Optional[SplitterConfig] = None

    _check_task = field_validator("task", mode="before")(_validate_task)

    @model_validator(mode="before")
    @classmethod
    def _reject_removed_keys(cls, raw: Any) -> Any:
        if not isinstance(raw, dict):
            return raw
        if "tiles" in raw:
            raise ValueError(_TILES_REMOVED)
        imagery = raw.get("imagery")
        if isinstance(imagery, dict) and "type" not in imagery:
            raise ValueError("imagery.type is required: 'xyz', 'eopf_zarr' or 'geotiff'")
        return raw

    @model_validator(mode="after")
    def _validate_source_writer_pair(self) -> "MapcvConfig":
        if isinstance(self.imagery, EOPFZarrImageryConfig):
            if self.writer.image_format != "npy":
                raise ValueError("EOPF Zarr imagery requires writer.image_format='npy'")
        elif isinstance(self.imagery, GeoTiffImageryConfig):
            bands = self.imagery.bands
            if self.writer.image_format != "npy" and bands is not None and len(bands) not in (1, 3):
                raise ValueError(
                    f"imagery.bands selects {len(bands)} bands, but writer.image_format "
                    f"'{self.writer.image_format}' writes 1 or 3 bands of uint8; select 1 or 3 "
                    "bands or set writer.image_format: npy"
                )
        elif self.writer.image_format == "npy":
            raise ValueError("XYZ imagery supports writer.image_format 'png' or 'jpg'")
        if isinstance(self.imagery, XYZImageryConfig):
            limit = WEB_MERCATOR_MAX_LATITUDE
            if self.region.north > limit or self.region.south < -limit:
                raise ValueError(
                    f"XYZ tiles cover latitudes -{limit:.4f}..{limit:.4f} (Web Mercator); "
                    "shrink the region or use imagery that covers the poles"
                )
        return self

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
