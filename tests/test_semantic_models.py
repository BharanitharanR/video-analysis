import pytest
from pydantic import ValidationError

from video_analysis.semantic.models import SemanticAnalysis
from video_analysis.semantic.merge import merge_semantic_analysis


def valid_semantic():
    return {
        "schema_version": "0.1",
        "analysis_type": "semantic",
        "duration_seconds": 5.0,
        "overall_mood": "curious",
        "overall_mood_confidence": 0.8,
        "content": {
            "environment": "outdoor",
            "subjects": [],
            "objects": [],
            "summary": "A short outdoor clip.",
        },
        "timeline": [{
            "start_seconds": 0.0,
            "end_seconds": 5.0,
            "mode": "real_world",
            "activity": "observing",
            "mood": "curious",
            "mood_intensity": 0.5,
            "confidence": 0.8,
            "visual_observations": ["A person looks toward the camera."],
            "audio_observations": [],
        }],
        "events": [],
    }


def test_semantic_contract_accepts_valid_json():
    assert SemanticAnalysis.model_validate(valid_semantic()).analysis_type == "semantic"


def test_semantic_contract_rejects_out_of_order_segments():
    data = valid_semantic()
    data["timeline"] = [
        {**data["timeline"][0], "start_seconds": 2.0, "end_seconds": 3.0},
        {**data["timeline"][0], "start_seconds": 1.0, "end_seconds": 4.0},
    ]
    with pytest.raises(ValidationError):
        SemanticAnalysis.model_validate(data)


def test_merge_preserves_deterministic_dimensions():
    deterministic = {
        "schema_version": "0.1",
        "video": {
            "path": "clip.mov",
            "dimensions": {
                "technical": {"duration_seconds": 5.0},
                "quality": {"sharpness": {"median": 12}},
            },
        },
    }
    merged = merge_semantic_analysis(deterministic, valid_semantic())
    assert merged["video"]["dimensions"]["technical"]["duration_seconds"] == 5.0
    assert merged["video"]["dimensions"]["quality"]["sharpness"]["median"] == 12
    assert merged["video"]["dimensions"]["semantic"]["overall_mood"] == "curious"
    assert "semantic" not in deterministic["video"]["dimensions"]
