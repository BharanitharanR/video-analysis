"""An answer that fails validation moves up to the next model.

Gemini is always a fake here: each test scripts what each model "answers".
No test makes a network call.
"""
import json
from types import SimpleNamespace

import pytest

from video_analysis.composition import gemini as composition_gemini
from video_analysis.composition.models import CompositionPlan
from video_analysis.gemini import (
    InvalidModelResponse,
    is_retryable_model_error,
    rejected_response_path,
    run_with_model_fallback,
)
from video_analysis.semantic.gemini import GeminiSemanticProvider

DURATION = 173.04


def semantic_answer(event_start, event_end):
    """A semantic answer shaped like the real 2026-11-21 one."""
    return {
        "overall_mood": "excited",
        "overall_mood_confidence": 0.8,
        "timeline": [{"start_seconds": 0, "end_seconds": DURATION, "activity": "demo",
                      "mood_intensity": 0.5, "confidence": 0.9}],
        "events": [{"start_seconds": event_start, "end_seconds": event_end,
                    "event": "Hotel booking confirmation received", "confidence": 0.9,
                    "evidence": ["PNR shown"]}],
    }


# The real failure: an event timed at 240 s in a 173 s video.
OUT_OF_RANGE = json.dumps(semantic_answer(240, 243))
VALID = json.dumps(semantic_answer(168, 171))


class FakeInteractions:
    def __init__(self, answers):
        self.answers = dict(answers)
        self.models = []

    def create(self, **kwargs):
        model = kwargs["model"]
        self.models.append(model)
        return SimpleNamespace(id=f"i-{model}", status="completed", output_text=self.answers[model])


def semantic_provider(answers):
    provider = object.__new__(GeminiSemanticProvider)
    provider._client = SimpleNamespace(interactions=FakeInteractions(answers), files=None)
    provider.model = None
    provider.file_poll_interval_seconds = 0
    provider.file_processing_timeout_seconds = 1
    provider.interaction_poll_interval_seconds = 0
    provider.interaction_processing_timeout_seconds = 1
    provider.webhook_url = None
    provider.make_copy = False
    return provider


@pytest.fixture
def video(tmp_path, monkeypatch):
    for name in ("VIDEO_ANALYSIS_SEMANTIC_MODELS", "VIDEO_ANALYSIS_MAX_MODEL_ATTEMPTS",
                 "VIDEO_ANALYSIS_MODEL_FALLBACK"):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("VIDEO_ANALYSIS_GEMINI_RAW_RESPONSE", str(tmp_path / "gemini-response.json"))
    path = tmp_path / "input.mp4"
    path.write_bytes(b"not really a video; the fake never reads it")
    return path


def test_fixtures_are_what_they_claim():
    from video_analysis.semantic.models import SemanticAnalysis

    def check(text):
        payload = json.loads(text)
        payload.update(schema_version="0.3", analysis_type="semantic", duration_seconds=DURATION)
        return SemanticAnalysis.model_validate(payload)

    check(VALID)
    with pytest.raises(ValueError, match="event exceeds video duration"):
        check(OUT_OF_RANGE)


def test_out_of_range_answer_moves_up_to_the_next_model(video, tmp_path):
    provider = semantic_provider({"gemini-3.5-flash-lite": OUT_OF_RANGE, "gemini-2.5-flash": VALID})

    semantic = provider.analyze(video, duration_seconds=DURATION)

    assert provider._client.interactions.models == ["gemini-3.5-flash-lite", "gemini-2.5-flash"]
    assert provider.model_used == "gemini-2.5-flash"
    assert semantic.events[0].start_seconds == 168
    # The accepted answer is the raw response; the rejected one is kept beside it.
    raw = tmp_path / "gemini-response.json"
    assert json.loads(raw.read_text())["events"][0]["start_seconds"] == 168
    rejected = rejected_response_path(raw, "gemini-3.5-flash-lite")
    assert rejected.name == "gemini-response.gemini-3.5-flash-lite.rejected.json"
    assert json.loads(rejected.read_text())["events"][0]["start_seconds"] == 240


def test_a_valid_first_answer_makes_exactly_one_call(video):
    provider = semantic_provider({"gemini-3.5-flash-lite": VALID})
    provider.analyze(video, duration_seconds=DURATION)
    assert provider._client.interactions.models == ["gemini-3.5-flash-lite"]


def test_invalid_answers_never_exceed_the_attempt_cap(video):
    bad = {m: OUT_OF_RANGE for m in ("gemini-3.5-flash-lite", "gemini-2.5-flash",
                                     "gemini-3.8-flash", "gemini-3.6-flash", "gemini-4-hydrogen")}
    provider = semantic_provider(bad)

    with pytest.raises(RuntimeError, match="Every Gemini model") as info:
        provider.analyze(video, duration_seconds=DURATION)

    assert len(provider._client.interactions.models) == 3  # the default cap
    assert "gemini-4-hydrogen" not in provider._client.interactions.models
    assert "event exceeds video duration" in str(info.value)
    assert isinstance(info.value.__cause__, InvalidModelResponse)


def test_attempt_cap_of_one_means_no_escalation(video, monkeypatch):
    monkeypatch.setenv("VIDEO_ANALYSIS_MAX_MODEL_ATTEMPTS", "1")
    provider = semantic_provider({"gemini-3.5-flash-lite": OUT_OF_RANGE, "gemini-2.5-flash": VALID})
    with pytest.raises(RuntimeError):
        provider.analyze(video, duration_seconds=DURATION)
    assert provider._client.interactions.models == ["gemini-3.5-flash-lite"]


@pytest.mark.parametrize("answer", ["", "{not json"])
def test_empty_or_non_json_answers_also_move_up(video, answer):
    provider = semantic_provider({"gemini-3.5-flash-lite": answer, "gemini-2.5-flash": VALID})
    provider.analyze(video, duration_seconds=DURATION)
    assert provider.model_used == "gemini-2.5-flash"


def test_log_says_the_answer_was_invalid():
    logs = []

    def attempt(model):
        if model == "small":
            raise InvalidModelResponse("Gemini semantic response failed local validation")
        return "ok"

    assert run_with_model_fallback(["small", "big"], attempt, log=logs.append) == ("ok", "big")
    assert logs[0].startswith("Gemini model small gave an invalid answer")
    assert logs[-1] == "Gemini model used: big"


def test_only_invalid_answers_are_retried_not_other_value_errors():
    assert is_retryable_model_error(InvalidModelResponse("bad answer"))
    assert not is_retryable_model_error(ValueError("bad input"))
    assert not is_retryable_model_error(TimeoutError("too slow"))


def test_composition_also_moves_up_on_an_invalid_answer(monkeypatch, tmp_path):
    for name in ("VIDEO_ANALYSIS_COMPOSITION_MODELS", "VIDEO_ANALYSIS_MAX_MODEL_ATTEMPTS",
                 "VIDEO_ANALYSIS_MODEL_FALLBACK"):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("VIDEO_ANALYSIS_COMPOSITION_RAW_RESPONSE", str(tmp_path / "composition.json"))
    monkeypatch.setattr(composition_gemini, "create_client", lambda: object())

    valid_plan = {"duration_seconds": 60.0, "world": "w", "style": "bold", "tempo_bpm": 100,
                  "time_signature": [4, 4], "ending": "e",
                  "sections": [{"start_seconds": 0, "end_seconds": 60, "purpose": "intro",
                                "energy": 0.5}]}
    CompositionPlan.model_validate(valid_plan)  # guard: the fake "good" answer really is valid
    answers = {"gemini-3.5-flash-lite": json.dumps({"world": "missing everything else"}),
               "gemini-2.5-flash": json.dumps(valid_plan)}
    called = []

    def fake_interaction(client, *, model, **kwargs):
        called.append(model)
        return SimpleNamespace(output_text=answers[model])

    monkeypatch.setattr(composition_gemini, "run_background_interaction", fake_interaction)
    provider = composition_gemini.GeminiCompositionProvider()
    plan = provider.compose({"video": {"dimensions": {"technical": {"duration_seconds": 60}}}})

    assert called == ["gemini-3.5-flash-lite", "gemini-2.5-flash"]
    assert provider.model_used == "gemini-2.5-flash"
    assert plan.style == "bold"
