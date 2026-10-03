"""Top-level Pydantic configuration models and YAML loading."""

from __future__ import annotations

import re
import warnings
from pathlib import Path
from typing import Annotated, Any, Dict, List, Literal, Optional, Union
from urllib.parse import unquote, urlsplit

import yaml
from pydantic import BaseModel, Field, field_validator, model_validator

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


def _warn_deprecated(message: str) -> None:
    # FutureWarning, not DeprecationWarning: this targets end users editing YAML,
    # and Python hides DeprecationWarning raised outside __main__ by default.
    warnings.warn(f"{message} Removed in mapcv 0.3.0.", FutureWarning, stacklevel=2)


class RegionConfig(BaseModel):
    """Geographic bounding box in WGS-84 degrees."""

    west: float
    south: float
    east: float
    north: float
    zoom: Optional[int] = Field(default=None, ge=1, le=22)

    @model_validator(mode="after")
    def _validate_bounds(self) -> "RegionConfig":
        if self.west >= self.east:
            raise ValueError("region.west must be less than region.east")
        if self.south >= self.north:
            raise ValueError("region.south must be less than region.north")
        return self


class XYZImageryConfig(BaseModel):
    """XYZ tile imagery source and fetch settings."""

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
        return self


class EOPFZarrImageryConfig(BaseModel):
    """One local or anonymous public Sentinel-2 L2A EOPF Zarr product."""

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


ImageryConfig = Annotated[
    Union[XYZImageryConfig, EOPFZarrImageryConfig],
    Field(discriminator="type"),
]


class LabelsConfig(BaseModel):
    """Label file (KML or GeoJSON) settings.

    ``classes`` maps label values to mask IDs (1..255). Without it, integer
    labels in 1..255 are used as-is and other labels get IDs in sorted order.
    """

    path: Path
    label_field: Optional[str] = None
    classes: Optional[Dict[str, int]] = None
    all_touched: bool = False

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
        return self


class MapcvConfig(BaseModel):
    """Full mapcv pipeline configuration."""

    region: RegionConfig
    imagery: ImageryConfig
    labels: Optional[LabelsConfig] = None
    sampler: SamplerConfig
    writer: WriterConfig
    split: Optional[SplitterConfig] = None

    @model_validator(mode="before")
    @classmethod
    def _normalize_legacy_tiles(cls, raw: Any) -> Any:
        if not isinstance(raw, dict):
            return raw

        data = dict(raw)
        legacy_tiles = data.get("tiles")
        imagery = data.get("imagery")
        region = data.get("region")
        region_zoom = region.get("zoom") if isinstance(region, dict) else None
        if legacy_tiles is not None and imagery is not None:
            raise ValueError("provide 'imagery' or legacy 'tiles', not both")
        if isinstance(imagery, dict) and "type" not in imagery:
            raise ValueError("imagery.type is required: 'xyz' or 'eopf_zarr'")

        if imagery is None and legacy_tiles is not None:
            if region_zoom is None:
                raise ValueError("legacy tiles configuration requires region.zoom")
            data["imagery"] = {"type": "xyz", "zoom": region_zoom, **dict(legacy_tiles)}
            data.pop("tiles")
            _warn_deprecated(
                "'tiles' and 'region.zoom' are deprecated; move them into an 'imagery' block "
                "with type: xyz and zoom (see MIGRATION.md)."
            )
        elif region_zoom is not None:
            if isinstance(imagery, dict) and imagery.get("type") == "xyz" and "zoom" not in imagery:
                data["imagery"] = {**imagery, "zoom": region_zoom}
                _warn_deprecated("'region.zoom' is deprecated; set imagery.zoom instead.")
            else:
                _warn_deprecated("'region.zoom' is ignored with this imagery block; remove it.")

        if region_zoom is not None and isinstance(region, dict):
            data["region"] = {key: value for key, value in region.items() if key != "zoom"}

        return data

    @model_validator(mode="after")
    def _validate_source_writer_pair(self) -> "MapcvConfig":
        if isinstance(self.imagery, EOPFZarrImageryConfig):
            if self.writer.image_format != "npy":
                raise ValueError("EOPF Zarr imagery requires writer.image_format='npy'")
        elif self.writer.image_format == "npy":
            raise ValueError("XYZ imagery supports writer.image_format 'png' or 'jpg'")
        return self

    @classmethod
    def from_yaml(cls, path: Path) -> "MapcvConfig":
        """Load and validate a mapcv YAML file."""
        data = yaml.safe_load(path.read_text())
        return cls.model_validate(data)
