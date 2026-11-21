"""LLM composition-planning contract.

The LLM makes creative musical decisions here. A deterministic compiler later
turns this plan into the provider-neutral SongSpec 1.0 consumed by text-to-compose.
"""
from __future__ import annotations

from pydantic import Field, model_validator
from ..models import StrictModel


class CompositionSection(StrictModel):
    start_seconds: float = Field(ge=0)
    end_seconds: float = Field(gt=0)
    # Short label such as "Scenic Breath". The music provider gets the section
    # name, so a full purpose sentence here would read as a stage direction.
    name: str = Field(default="", max_length=60)
    purpose: str = Field(min_length=1, max_length=240)
    energy: float = Field(ge=0, le=1)
    mood: str = Field(default="neutral", min_length=1, max_length=80)
    instruments: list[str] = Field(default_factory=list, max_length=12)
    texture: str = Field(default="", max_length=240)
    transition: str = Field(default="", max_length=160)

    @model_validator(mode="after")
    def validate_interval(self):
        if self.end_seconds <= self.start_seconds:
            raise ValueError("end_seconds must be greater than start_seconds")
        return self


class CompositionPlan(StrictModel):
    schema_version: str = "0.1"
    duration_seconds: float = Field(gt=0)
    title: str = Field(default="Video Score", min_length=1, max_length=160)
    language: str = "Instrumental"
    world: str = Field(min_length=1, max_length=1000)
    style: str = Field(min_length=1, max_length=500)
    tempo_bpm: float = Field(gt=30, le=220)
    time_signature: list[int] = Field(min_length=2, max_length=2)
    key_or_raga: str = Field(default="G major", min_length=1, max_length=80)
    motif: str = Field(default="", max_length=240)
    ending: str = Field(min_length=1, max_length=500)
    instruments: list[str] = Field(default_factory=list, max_length=16)
    sections: list[CompositionSection] = Field(min_length=1, max_length=100)
    # True when the music is a soft bed under spoken voiceover. Set in code
    # when a voice timeline is supplied, not trusted from the model.
    narration_bed: bool = False

    @model_validator(mode="after")
    def validate_plan(self):
        if self.time_signature[0] <= 0 or self.time_signature[1] <= 0:
            raise ValueError("time_signature values must be positive")
        previous_end = -1.0
        for section in self.sections:
            if section.end_seconds > self.duration_seconds + 0.25:
                raise ValueError("composition section exceeds video duration")
            if section.start_seconds < previous_end - 0.05:
                raise ValueError("composition sections must be ordered and non-overlapping")
            previous_end = section.end_seconds

        if self.sections[0].start_seconds > 0.25:
            raise ValueError("composition must start at the beginning of the video")
        if self.sections[-1].end_seconds < self.duration_seconds - 0.25:
            raise ValueError("composition must cover the full video duration")
        return self
