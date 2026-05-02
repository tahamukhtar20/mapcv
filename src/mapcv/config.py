"""Top-level Pydantic config model and YAML loader."""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import yaml
from pydantic import BaseModel, Field, model_validator

from mapcv.sampler import SamplerConfig
from mapcv.splitter import SplitterConfig
from mapcv.writer import WriterConfig


class RegionConfig(BaseModel):
    """Geographic bounding box and zoom level."""

    west: float
    south: float
    east: float
    north: float
    zoom: int = Field(ge=1, le=22)


class TilesConfig(BaseModel):
    """Tile source and fetch settings."""

    source: Optional[str] = None
    url_template: Optional[str] = None
    max_connections: int = Field(default=16, ge=1)
    policy: str = "lenient"
    max_failed_ratio: float = Field(default=0.05, ge=0.0, le=1.0)
    strip_rows: int = Field(default=4, ge=1)

    @model_validator(mode="after")
    def _require_source_or_template(self) -> "TilesConfig":
        if self.source is None and self.url_template is None:
            raise ValueError("tiles: provide either 'source' or 'url_template'")
        return self


class LabelsConfig(BaseModel):
    """Label file (KML or GeoJSON) settings."""

    path: Path
    label_field: Optional[str] = None
    all_touched: bool = False


class MapcvConfig(BaseModel):
    """Full pipeline configuration."""

    region: RegionConfig
    tiles: TilesConfig
    labels: Optional[LabelsConfig] = None
    sampler: SamplerConfig
    writer: WriterConfig
    split: Optional[SplitterConfig] = None

    @classmethod
    def from_yaml(cls, path: Path) -> "MapcvConfig":
        """Load and validate config from a YAML file."""
        data = yaml.safe_load(path.read_text())
        return cls.model_validate(data)
