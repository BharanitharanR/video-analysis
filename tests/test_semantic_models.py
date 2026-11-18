import pytest
from pydantic import ValidationError

from video_analysis.semantic.models import SemanticAnalysis
from video_analysis.semantic.merge import merge_semantic_analysis
from video_analysis.semantic.gemini import _gemini_response_schema


def valid_semantic():
    return {
        "schema_version": "0.2",
        "analysis_type": "semantic",
        "duration_seconds": 5.0,
        "overall_mood": "curious",
        "overall_mood_confidence": 0.8,
        "content": {
            "environment": "outdoor",
            "people": [],
            "vehicles": [],
            "objects": [],
            "activities": [],
            "interactions": [],
            "text": [],
            "summary": "A short outdoor clip.",
        },
        "story": "A person observes an outdoor scene.",
        "timeline": [{
            "start_seconds": 0.0,
            "end_seconds": 5.0,
            "mode": "real_world",
            "activity": "observing",
            "mood": "curious",
            "mood_intensity": 0.5,
            "confidence": 0.8,
            "story_context": "A person observes the scene.",
            "visual_observations": ["A person looks toward the camera."],
            "audio_observations": [],
        }],
        "events": [],
        "moments": [],
        "edit_points": [],
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


def test_gemini_schema_omits_incompatible_constraints():
    schema = _gemini_response_schema()

    def walk(node):
        if isinstance(node, dict):
            yield node
            for value in node.values():
                yield from walk(value)
        elif isinstance(node, list):
            for value in node:
                yield from walk(value)

    nodes = list(walk(schema))
    assert all("exclusiveMinimum" not in node for node in nodes)
    assert all("maxItems" not in node for node in nodes)


def test_gemini_schema_excludes_contract_owned_fields():
    schema = _gemini_response_schema()
    properties = schema["properties"]
    required = schema.get("required", [])

    assert "schema_version" not in properties
    assert "analysis_type" not in properties
    assert "duration_seconds" not in properties
    assert "schema_version" not in required
    assert "analysis_type" not in required
    assert "duration_seconds" not in required
