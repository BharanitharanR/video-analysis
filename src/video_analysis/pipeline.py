"""End-to-end deterministic + semantic video analysis pipeline."""
from __future__ import annotations

from pathlib import Path
from typing import Any, Protocol

from .analyzer import VideoAnalyzer
from .semantic.merge import merge_semantic_analysis
from .semantic.models import SemanticAnalysis


class SemanticProvider(Protocol):
    """Provider interface required by the pipeline."""

    def analyze(
        self,
        video_path: str | Path,
        duration_seconds: float,
        evidence: dict[str, Any] | None = None,
    ) -> SemanticAnalysis:
        ...


class VideoAnalysisPipeline:
    """Run deterministic analysis first, then optional semantic inference.

    The deterministic report is always produced by VideoAnalyzer. When a semantic
    provider is supplied, its validated output is merged into the deterministic
    report under video.dimensions.semantic. No deterministic dimension is
    overwritten by the semantic provider.
    """

    def __init__(
        self,
        *,
        deterministic_analyzer: VideoAnalyzer | None = None,
        semantic_provider: SemanticProvider | None = None,
    ) -> None:
        self.deterministic_analyzer = deterministic_analyzer or VideoAnalyzer()
        self.semantic_provider = semantic_provider

    def analyze(self, path: str | Path) -> dict[str, Any]:
        deterministic = self.deterministic_analyzer.analyze(path)

        if self.semantic_provider is None:
            return deterministic

        dimensions = deterministic["video"]["dimensions"]
        duration = deterministic["video"]["dimensions"]["technical"].get(
            "duration_seconds"
        )
        if duration is None:
            raise ValueError(
                "Deterministic technical analysis did not provide video duration."
            )

        # Temporal measurements are read-only evidence for the semantic model.
        # They help Gemini reason about when changes occur without allowing the
        # model to become the owner of deterministic measurements.
        evidence = {
            "temporal": dimensions.get("temporal", {}),
            "scenes": dimensions.get("scenes", {}),
        }

        semantic = self.semantic_provider.analyze(
            path,
            duration_seconds=float(duration),
            evidence=evidence,
        )

        # The merge function validates the semantic contract and creates a new
        # document. The original deterministic report remains unchanged.
        return merge_semantic_analysis(deterministic, semantic)
