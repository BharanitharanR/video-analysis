import pytest

from video_analysis.pipeline import VideoAnalysisPipeline
from video_analysis.semantic.models import SemanticAnalysis


class FakeDeterministicAnalyzer:
    def __init__(self):
        self.calls = []

    def analyze(self, path):
        self.calls.append(path)
        return {
            "schema_version": "0.1",
            "video": {
                "path": str(path),
                "dimensions": {
                    "technical": {"duration_seconds": 5.0},
                    "temporal": {"status": "ok", "evidence": [{"time_seconds": 1.0, "motion": 2.0}]},
                    "scenes": {"status": "ok", "scenes": []},
                },
            },
        }


class FakeSemanticProvider:
    def __init__(self):
        self.calls = []

    def analyze(self, video_path, duration_seconds, evidence=None):
        self.calls.append((video_path, duration_seconds, evidence))
        return SemanticAnalysis.model_validate({
            "schema_version": "0.2",
            "analysis_type": "semantic",
            "duration_seconds": duration_seconds,
            "overall_mood": "happy",
            "overall_mood_confidence": 0.9,
            "content": {
                "environment": "vehicle interior",
                "people": ["person wearing a green shirt"],
                "vehicles": ["auto-rickshaw"],
                "objects": ["camera"],
                "activities": ["filming a selfie"],
                "interactions": ["person looks toward camera"],
                "text": [],
                "summary": "A person films a selfie inside an auto-rickshaw.",
            },
            "story": "A person films a selfie while riding in an auto-rickshaw.",
            "timeline": [{
                "start_seconds": 0.0,
                "end_seconds": 5.0,
                "mode": "real_world",
                "activity": "filming a selfie",
                "mood": "happy",
                "mood_intensity": 0.7,
                "confidence": 0.9,
                "story_context": "The person films a selfie while riding.",
                "visual_observations": ["Auto-rickshaw interior is visible."],
                "audio_observations": [],
            }],
            "events": [],
            "moments": [],
            "edit_points": [],
        })


def test_pipeline_merges_semantic_without_mutating_deterministic():
    deterministic = FakeDeterministicAnalyzer()
    semantic = FakeSemanticProvider()
    pipeline = VideoAnalysisPipeline(
        deterministic_analyzer=deterministic,
        semantic_provider=semantic,
    )

    result = pipeline.analyze("clip.mov")

    assert result["video"]["dimensions"]["technical"]["duration_seconds"] == 5.0
    assert result["video"]["dimensions"]["semantic"]["overall_mood"] == "happy"
    assert result["video"]["dimensions"]["semantic"]["duration_seconds"] == 5.0
    assert semantic.calls[0][1] == 5.0
    assert semantic.calls[0][2]["temporal"]["evidence"][0]["motion"] == 2.0


def test_pipeline_without_semantic_provider_is_deterministic_only():
    deterministic = FakeDeterministicAnalyzer()
    result = VideoAnalysisPipeline(deterministic_analyzer=deterministic).analyze("clip.mov")

    assert "semantic" not in result["video"]["dimensions"]
