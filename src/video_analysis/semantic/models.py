"""Separate semantic artifact and video content-type classification."""
from __future__ import annotations
from typing import Literal
from pydantic import Field, model_validator
from ..models import StrictModel
Mood = Literal["joyful","excited","happy","calm","peaceful","curious","nostalgic","tense","dramatic","sad","fearful","angry","surprised","neutral","mixed","unknown"]
Mode = Literal["real_world","screen_recording","presentation","static_visual","mixed","unknown"]
EditPointType = Literal["cut","shorten","transition","emphasize","opening","ending","unknown"]
VideoType = Literal["demo","tutorial","walkthrough","presentation","documentation","screen_recording","vlog","conversation","gameplay","review","advertisement","real_world","music_video","animation","news","meeting","event","interview","educational","mixed","unknown"]
SemanticSegmentMode = Mode
class VideoTypeAlternative(StrictModel):
    type: VideoType
    confidence: float = Field(ge=0, le=1)
class VideoTypeClassification(StrictModel):
    primary: VideoType = "unknown"
    confidence: float = Field(ge=0, le=1)
    alternatives: list[VideoTypeAlternative] = Field(default_factory=list, max_length=5)
    reason: str = Field(default="", max_length=500)
class SemanticSegment(StrictModel):
    start_seconds: float = Field(ge=0)
    end_seconds: float = Field(gt=0)
    mode: SemanticSegmentMode = "unknown"
    activity: str = Field(min_length=1, max_length=160)
    mood: Mood = "unknown"
    mood_intensity: float = Field(ge=0, le=1)
    confidence: float = Field(ge=0, le=1)
    story_context: str = Field(default="", max_length=500)
    visual_observations: list[str] = Field(default_factory=list, max_length=20)
    audio_observations: list[str] = Field(default_factory=list, max_length=12)
    @model_validator(mode="after")
    def validate_interval(self):
        if self.end_seconds <= self.start_seconds: raise ValueError("end_seconds must be greater than start_seconds")
        return self
class SemanticEvent(StrictModel):
    start_seconds: float = Field(ge=0)
    end_seconds: float = Field(gt=0)
    event: str = Field(min_length=1, max_length=200)
    confidence: float = Field(ge=0, le=1)
    evidence: list[str] = Field(default_factory=list, max_length=10)
    @model_validator(mode="after")
    def validate_interval(self):
        if self.end_seconds <= self.start_seconds: raise ValueError("end_seconds must be greater than start_seconds")
        return self
class SemanticMoment(StrictModel):
    start_seconds: float = Field(ge=0)
    end_seconds: float = Field(gt=0)
    description: str = Field(min_length=1, max_length=240)
    importance: float = Field(ge=0, le=1)
    reason: str = Field(min_length=1, max_length=240)
    confidence: float = Field(ge=0, le=1)
    @model_validator(mode="after")
    def validate_interval(self):
        if self.end_seconds <= self.start_seconds: raise ValueError("end_seconds must be greater than start_seconds")
        return self
class SemanticEditPoint(StrictModel):
    timestamp_seconds: float = Field(ge=0)
    type: EditPointType = "unknown"
    reason: str = Field(min_length=1, max_length=240)
    confidence: float = Field(ge=0, le=1)
class SemanticContent(StrictModel):
    environment: str = "unknown"
    people: list[str] = Field(default_factory=list, max_length=40)
    vehicles: list[str] = Field(default_factory=list, max_length=30)
    objects: list[str] = Field(default_factory=list, max_length=60)
    activities: list[str] = Field(default_factory=list, max_length=40)
    interactions: list[str] = Field(default_factory=list, max_length=40)
    text: list[str] = Field(default_factory=list, max_length=30)
    summary: str = Field(default="", max_length=1600)
class SemanticAnalysis(StrictModel):
    schema_version: Literal["0.3"] = "0.3"
    analysis_type: Literal["semantic"] = "semantic"
    duration_seconds: float = Field(gt=0)
    video_type: VideoTypeClassification = Field(default_factory=VideoTypeClassification)
    overall_mood: Mood = "unknown"
    overall_mood_confidence: float = Field(ge=0, le=1)
    content: SemanticContent = Field(default_factory=SemanticContent)
    story: str = Field(default="", max_length=1600)
    timeline: list[SemanticSegment] = Field(min_length=1, max_length=500)
    events: list[SemanticEvent] = Field(default_factory=list, max_length=500)
    moments: list[SemanticMoment] = Field(default_factory=list, max_length=200)
    edit_points: list[SemanticEditPoint] = Field(default_factory=list, max_length=200)
    @model_validator(mode="after")
    def validate_timeline(self):
        previous_end=-1.0
        for seg in self.timeline:
            if seg.end_seconds > self.duration_seconds+0.25: raise ValueError("timeline exceeds duration")
            if seg.start_seconds < previous_end-0.05: raise ValueError("timeline must be ordered and non-overlapping")
            previous_end=seg.end_seconds
        for item in [*self.events,*self.moments]:
            if item.end_seconds > self.duration_seconds+0.25: raise ValueError("semantic interval exceeds duration")
        for point in self.edit_points:
            if point.timestamp_seconds > self.duration_seconds+0.25: raise ValueError("edit point exceeds duration")
        return self
