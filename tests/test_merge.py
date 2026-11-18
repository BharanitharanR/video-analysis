import json

import pytest

from video_analysis.merge import merge_analysis_artifacts, write_merged_analysis


def deterministic_artifact():
    return {
        "schema_version": "0.1",
        "video": {
            "path": "video.mov",
            "dimensions": {
                "technical": {"duration_seconds": 87.0},
                "audio": {"present": True},
                "quality": {"sampling": {"sample_count": 87}},
                "scenes": {"status": "ok", "scenes": []},
                "temporal": {"status": "ok", "evidence": []},
            },
        },
    }


def semantic_artifact():
    return {
        "schema_version": "0.2",
        "analysis_type": "semantic",
        "duration_seconds": 87.0,
        "overall_mood": "joyful",
        "overall_mood_confidence": 0.88,
        "content": {
            "environment": "Outdoor hillside viewpoint",
            "people": ["Child filming", "Man", "Woman"],
            "vehicles": ["White car"],
            "objects": ["smartphone"],
            "activities": ["walking", "joking"],
            "interactions": ["Child jokes with father"],
            "text": ["GOKU"],
            "summary": "A child films a family visit.",
        },
        "story": "A child films a family visit and playful interaction.",
        "timeline": [{
            "start_seconds": 0.0,
            "end_seconds": 87.0,
            "mode": "real_world",
            "activity": "filming",
            "mood": "joyful",
            "mood_intensity": 0.7,
            "confidence": 0.9,
            "story_context": "Family visit.",
            "visual_observations": [],
            "audio_observations": [],
        }],
        "events": [],
        "moments": [],
        "edit_points": [],
    }


def test_merge_analysis_artifacts_keeps_deterministic_dimensions(tmp_path):
    deterministic_path = tmp_path / "deterministic.json"
    semantic_path = tmp_path / "semantic.json"

    deterministic_path.write_text(json.dumps(deterministic_artifact()))
    semantic_path.write_text(json.dumps(semantic_artifact()))

    result = merge_analysis_artifacts(deterministic_path, semantic_path)

    dimensions = result["video"]["dimensions"]
    assert dimensions["technical"]["duration_seconds"] == 87.0
    assert dimensions["quality"]["sampling"]["sample_count"] == 87
    assert dimensions["semantic"]["overall_mood"] == "joyful"


def test_write_merged_analysis_creates_json(tmp_path):
    deterministic_path = tmp_path / "deterministic.json"
    semantic_path = tmp_path / "semantic.json"
    output_path = tmp_path / "analysis.json"

    deterministic_path.write_text(json.dumps(deterministic_artifact()))
    semantic_path.write_text(json.dumps(semantic_artifact()))

    write_merged_analysis(deterministic_path, semantic_path, output_path)

    saved = json.loads(output_path.read_text())
    assert saved["video"]["dimensions"]["semantic"]["analysis_type"] == "semantic"


def test_merge_rejects_invalid_semantic_artifact(tmp_path):
    deterministic_path = tmp_path / "deterministic.json"
    semantic_path = tmp_path / "semantic.json"

    deterministic_path.write_text(json.dumps(deterministic_artifact()))
    semantic_path.write_text(json.dumps({"analysis_type": "semantic"}))

    with pytest.raises(Exception):
        merge_analysis_artifacts(deterministic_path, semantic_path)
