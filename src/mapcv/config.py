"""Top-level Pydantic configuration models and YAML loading."""

from __future__ import annotations

import warnings
from pathlib import Path
from typing import Annotated, Any, List, Literal, Optional, Union

import yaml
from pydantic import BaseModel, Field, model_validator

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


class TilesConfig(BaseModel):
    """Legacy XYZ tile configuration accepted until mapcv 0.3.0."""

    source: Optional[str] = None
    url_template: Optional[str] = None
    max_connections: int = Field(default=16, ge=1)
    policy: Literal["strict", "lenient", "ignore"] = "lenient"
    max_failed_ratio: float = Field(default=0.05, ge=0.0, le=1.0)
    strip_rows: int = Field(default=4, ge=1)

    @model_validator(mode="after")
    def _require_source_or_template(self) -> "TilesConfig":
        if self.source is None and self.url_template is None:
            raise ValueError("tiles: provide either 'source' or 'url_template'")
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
    """Label file (KML or GeoJSON) settings."""

    path: Path
    label_field: Optional[str] = None
    all_touched: bool = False


class MapcvConfig(BaseModel):
    """Full mapcv pipeline configuration."""

    region: RegionConfig
    imagery: ImageryConfig
    tiles: Optional[TilesConfig] = None
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
        if legacy_tiles is not None and imagery is not None:
            raise ValueError("provide 'imagery' or legacy 'tiles', not both")

        if imagery is None and legacy_tiles is not None:
            region = data.get("region")
            zoom = region.get("zoom") if isinstance(region, dict) else None
            if zoom is None:
                raise ValueError("legacy tiles configuration requires region.zoom")
            data["imagery"] = {"type": "xyz", "zoom": zoom, **dict(legacy_tiles)}
            warnings.warn(
                "'tiles' and 'region.zoom' are deprecated; use imagery.type='xyz'. "
                "Legacy configuration support will be removed in mapcv 0.3.0.",
                DeprecationWarning,
                stacklevel=2,
            )
        elif isinstance(imagery, dict) and imagery.get("type") == "xyz":
            if "zoom" not in imagery:
                region = data.get("region")
                zoom = region.get("zoom") if isinstance(region, dict) else None
                if zoom is not None:
                    data["imagery"] = {**imagery, "zoom": zoom}

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
