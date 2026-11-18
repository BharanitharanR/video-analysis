"""Separate JSON contract returned by a multimodal model.

This contract intentionally does not mirror or mutate the deterministic report.
The deterministic analyzer owns its document; this model owns only semantic
observations inferred from the audiovisual content.
"""
from __future__ import annotations

from typing import Literal
from pydantic import ConfigDict, Field, model_validator
from ..models import StrictModel

Mood = Literal[
    "joyful", "excited", "happy", "calm", "peaceful", "curious",
    "nostalgic", "tense", "dramatic", "sad", "fearful", "angry",
    "surprised", "neutral", "mixed", "unknown",
]

Mode = Literal[
    "real_world", "screen_recording", "presentation", "static_visual",
    "mixed", "unknown",
]

class SemanticSegment(StrictModel):
    start_seconds: float = Field(ge=0)
    end_seconds: float = Field(gt=0)
    mode: Mode = "unknown"
    activity: str = Field(min_length=1, max_length=120)
    mood: Mood = "unknown"
    mood_intensity: float = Field(ge=0, le=1)
    confidence: float = Field(ge=0, le=1)
    visual_observations: list[str] = Field(default_factory=list, max_length=12)
    audio_observations: list[str] = Field(default_factory=list, max_length=12)

    @model_validator(mode="after")
    def validate_interval(self):
        if self.end_seconds <= self.start_seconds:
            raise ValueError("end_seconds must be greater than start_seconds")
        return self


class SemanticEvent(StrictModel):
    start_seconds: float = Field(ge=0)
    end_seconds: float = Field(gt=0)
    event: str = Field(min_length=1, max_length=160)
    confidence: float = Field(ge=0, le=1)
    evidence: list[str] = Field(default_factory=list, max_length=8)

    @model_validator(mode="after")
    def validate_interval(self):
        if self.end_seconds <= self.start_seconds:
            raise ValueError("end_seconds must be greater than start_seconds")
        return self


class SemanticContent(StrictModel):
    environment: str = "unknown"
    subjects: list[str] = Field(default_factory=list, max_length=30)
    objects: list[str] = Field(default_factory=list, max_length=50)
    summary: str = Field(default="", max_length=1200)


class SemanticAnalysis(StrictModel):
    schema_version: Literal["0.1"] = "0.1"
    analysis_type: Literal["semantic"] = "semantic"
    duration_seconds: float = Field(gt=0)
    overall_mood: Mood = "unknown"
    overall_mood_confidence: float = Field(ge=0, le=1)
    content: SemanticContent = Field(default_factory=SemanticContent)
    timeline: list[SemanticSegment] = Field(min_length=1, max_length=500)
    events: list[SemanticEvent] = Field(default_factory=list, max_length=500)

    @model_validator(mode="after")
    def validate_timeline(self):
        previous_end = -1.0
        for segment in self.timeline:
            if segment.end_seconds > self.duration_seconds + 0.25:
                raise ValueError("timeline segment exceeds video duration")
            if segment.start_seconds < previous_end - 0.05:
                raise ValueError("timeline segments must be ordered and non-overlapping")
            previous_end = segment.end_seconds
        for event in self.events:
            if event.end_seconds > self.duration_seconds + 0.25:
                raise ValueError("event exceeds video duration")
        return self
