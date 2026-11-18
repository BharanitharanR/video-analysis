"""Pydantic models for deterministic video analysis output."""
from __future__ import annotations

from typing import Any
from pydantic import BaseModel, ConfigDict, Field


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class DimensionDocument(StrictModel):
    """Flexible dimension payloads; concrete dimensions own their inner schemas."""
    pass


class VideoDocument(StrictModel):
    path: str
    dimensions: dict[str, Any] = Field(default_factory=dict)


class AnalysisDocument(StrictModel):
    schema_version: str = "0.1"
    video: VideoDocument
