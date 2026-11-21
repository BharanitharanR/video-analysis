import json
from types import SimpleNamespace

from video_analysis.composition import gemini as gem
from video_analysis.composition.compiler import compile_song_spec
from video_analysis.composition.models import CompositionPlan
from video_analysis.composition.prompt import build_composition_prompt
from video_analysis.composition.voice import (
    NARRATION_ENERGY_CAP, GAP_ENERGY_CAP, apply_narration_bed, narration_windows,
    speaking_share, summarize_voice_timeline,
)

TIMELINE = {
    "schema_version": "1.0", "duration_seconds": 60.0,
    "lines": [
        {"segment_index": 0, "turn_index": None, "speaker": "A", "text": "x", "start_seconds": 5.0, "end_seconds": 12.0},
        {"segment_index": 1, "turn_index": 0, "speaker": "B", "text": "y", "start_seconds": 12.2, "end_seconds": 20.0},
        {"segment_index": 2, "turn_index": None, "speaker": "A", "text": "z", "start_seconds": 40.0, "end_seconds": 50.0},
    ],
}


def plan():
    return CompositionPlan.model_validate({
        "duration_seconds": 60.0, "world": "w", "style": "s", "tempo_bpm": 90,
        "time_signature": [4, 4], "ending": "e",
        "sections": [
            {"start_seconds": 0, "end_seconds": 5, "purpose": "intro", "energy": 0.9},
            {"start_seconds": 5, "end_seconds": 20, "purpose": "talk", "energy": 0.8},
            {"start_seconds": 20, "end_seconds": 40, "purpose": "gap", "energy": 0.9},
            {"start_seconds": 40, "end_seconds": 60, "purpose": "talk2", "energy": 0.7},
        ],
    })


def test_windows_merge_close_lines_and_share():
    w = narration_windows(TIMELINE)
    assert [(x["start"], x["end"]) for x in w] == [(5.0, 20.0), (40.0, 50.0)]
    assert w[0]["speakers"] == ["A", "B"]
    assert abs(speaking_share(w, 60.0) - 25 / 60) < 1e-9


def test_prompt_includes_summary_only_when_given():
    analysis = {"video": {"dimensions": {"technical": {"duration_seconds": 60}}}}
    plain = build_composition_prompt(analysis)
    assert "NARRATION BED" not in plain
    with_voice = build_composition_prompt(analysis, summarize_voice_timeline(TIMELINE, 60.0))
    assert "NARRATION BED" in with_voice and "5.0s to 20.0s" in with_voice and "42%" in with_voice


def test_energy_capped_in_code_and_spec_flagged():
    p = apply_narration_bed(plan(), TIMELINE)
    e = [s.energy for s in p.sections]
    assert e == [GAP_ENERGY_CAP, NARRATION_ENERGY_CAP, GAP_ENERGY_CAP, NARRATION_ENERGY_CAP]
    spec = compile_song_spec(p)
    assert spec["extra"]["narration_bed"] is True
    assert spec["arrangement"]["sections"][1]["energy"] == NARRATION_ENERGY_CAP
    json.dumps(spec)


def test_spec_without_voice_has_no_extra():
    assert "extra" not in compile_song_spec(plan())


def test_provider_compose_with_fake(monkeypatch):
    raw = plan().model_dump()
    monkeypatch.setattr(gem, "create_client", lambda: object())
    monkeypatch.setattr(gem, "run_with_model_fallback", lambda chain, fn: (SimpleNamespace(output_text=json.dumps(raw)), "fake"))
    monkeypatch.setattr(gem, "persist_raw_response", lambda *a, **k: None)
    monkeypatch.setattr(gem, "parse_json_response", lambda i, artifact_name, raw_path, validator: validator(json.loads(i.output_text)))
    analysis = {"video": {"dimensions": {"technical": {"duration_seconds": 60}}}}
    p = gem.GeminiCompositionProvider().compose(analysis, voice_timeline=TIMELINE)
    assert p.narration_bed and max(s.energy for s in p.sections) <= GAP_ENERGY_CAP
    q = gem.GeminiCompositionProvider().compose(analysis)
    assert not q.narration_bed and q.sections[0].energy == 0.9
