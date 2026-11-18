"""Deterministic video analysis orchestration."""
from __future__ import annotations
from pathlib import Path
from .dimensions import technical, quality, audio, location, scenes, temporal

class VideoAnalyzer:
    def __init__(self, *, scene_detection=True, sample_interval=1.0):
        self.scene_detection=scene_detection
        self.sample_interval=sample_interval

    def analyze(self, path: str | Path) -> dict:
        path=Path(path)
        tech=technical.analyze(path)
        return {
            "schema_version":"0.1",
            "video":{
                "path":str(path),
                "dimensions":{
                    "technical":tech,
                    "audio":audio.analyze(path),
                    "location":location.analyze(path),
                    "quality":quality.analyze(path,self.sample_interval),
                    "scenes":scenes.analyze(path) if self.scene_detection else {"status":"disabled","scenes":[]},
                    "temporal":temporal.analyze(path,self.sample_interval),
                }
            }
        }
