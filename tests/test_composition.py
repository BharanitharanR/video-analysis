import json

from video_analysis.composition.compiler import compile_song_spec
from video_analysis.composition.models import CompositionPlan


def sample_plan():
    return CompositionPlan.model_validate({
        "schema_version": "0.1",
        "duration_seconds": 87.0,
        "title": "Family Hillside",
        "language": "Instrumental",
        "world": "A warm family journey through a bright hillside viewpoint.",
        "style": "Cinematic acoustic travel score with Indian colors",
        "tempo_bpm": 92,
        "time_signature": [4, 4],
        "key_or_raga": "G major",
        "motif": "G-B-A-G",
        "ending": "Gentle sustained guitar chord fading into natural ambience.",
        "instruments": ["acoustic guitar", "piano", "soft percussion"],
        "sections": [
            {
                "start_seconds": 0,
                "end_seconds": 15.8,
                "purpose": "curious introduction",
                "energy": 0.35,
                "mood": "curious",
                "instruments": ["acoustic guitar", "soft piano"],
            },
            {
                "start_seconds": 15.8,
                "end_seconds": 27.2,
                "purpose": "playful family interaction",
                "energy": 0.75,
                "mood": "joyful",
                "instruments": ["acoustic guitar", "light percussion"],
            },
        ],
    })


def test_compile_song_spec_uses_song_spec_1_0():
    result = compile_song_spec(sample_plan())

    assert result["song_spec_version"] == "1.0"
    assert result["metadata"]["duration_seconds"] == 87.0
    assert result["creative"]["vocals"]["required"] is False
    assert result["lyrics"]["sections"] == []
    assert result["arrangement"]["sections"][0]["start_beat"] == 0
    assert result["arrangement"]["sections"][0]["end_beat"] == round(15.8 * 92 / 60)
    assert result["providers"]["requested_capabilities"] == [
        "instrumental", "full_duration", "original_music"
    ]


def test_compiled_song_spec_is_json_serializable():
    json.dumps(compile_song_spec(sample_plan()))
