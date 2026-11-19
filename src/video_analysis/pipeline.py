"""End-to-end deterministic + semantic video analysis pipeline."""
from __future__ import annotations
from pathlib import Path
from typing import Any, Protocol
from .analyzer import VideoAnalyzer
from .semantic.merge import merge_semantic_analysis
from .semantic.models import SemanticAnalysis
class SemanticProvider(Protocol):
    def analyze(self,video_path:str|Path,duration_seconds:float,evidence:dict[str,Any]|None=None)->SemanticAnalysis: ...
class VideoAnalysisPipeline:
    """Run deterministic analysis, then optional multimodal semantics and classification."""
    def __init__(self,*,deterministic_analyzer:VideoAnalyzer|None=None,semantic_provider:SemanticProvider|None=None):
        self.deterministic_analyzer=deterministic_analyzer or VideoAnalyzer()
        self.semantic_provider=semantic_provider
    def analyze(self,path:str|Path)->dict[str,Any]:
        deterministic=self.deterministic_analyzer.analyze(path)
        if self.semantic_provider is None: return deterministic
        dims=deterministic["video"]["dimensions"]
        duration=dims["technical"].get("duration_seconds")
        if duration is None: raise ValueError("Deterministic technical analysis did not provide video duration.")
        evidence={"temporal":dims.get("temporal",{}),"scenes":dims.get("scenes",{})}
        semantic=self.semantic_provider.analyze(path,duration_seconds=float(duration),evidence=evidence)
        return merge_semantic_analysis(deterministic,semantic)
