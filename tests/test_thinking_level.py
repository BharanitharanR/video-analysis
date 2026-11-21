"""Thinking level per step, and models that reject a request shape.

Gemini is a fake; the config path is a temp file (see conftest)."""
import json
from types import SimpleNamespace

import pytest

from video_analysis import model_config
from video_analysis.gemini import (
    DEFAULT_COMPOSITION_MODELS,
    DEFAULT_SEMANTIC_MODELS,
    is_quota_error,
    is_retryable_model_error,
    run_with_key_and_model_fallback,
)
from video_analysis.semantic.gemini import GeminiSemanticProvider

ANSWER = json.dumps({"overall_mood": "calm", "overall_mood_confidence": 0.5,
                     "timeline": [{"start_seconds": 0, "end_seconds": 5, "activity": "x",
                                   "mood_intensity": 0.2, "confidence": 0.5}]})


class ClientError(Exception):
    code = 400


UNSUPPORTED = ClientError("400 - {'error': {'message': \"Model 'gemini-2.5-flash' does not support "
                          "background interactions.\", 'code': 'invalid_request'}}")


@pytest.fixture
def cfg(monkeypatch, tmp_path):
    monkeypatch.delenv(model_config.THINKING_ENV, raising=False)
    for name in ("VIDEO_ANALYSIS_SEMANTIC_MODELS", "VIDEO_ANALYSIS_MAX_MODEL_ATTEMPTS"):
        monkeypatch.delenv(name, raising=False)
    path = tmp_path / "gemini-models.json"
    monkeypatch.setenv(model_config.CONFIG_ENV, str(path))
    return lambda data: path.write_text(json.dumps(data))


def run_provider(tmp_path, failures=None):
    sent = []

    def create(**kwargs):
        sent.append(kwargs)
        if failures and kwargs["model"] in failures:
            raise failures[kwargs["model"]]
        return SimpleNamespace(id="i", status="completed", output_text=ANSWER)

    provider = object.__new__(GeminiSemanticProvider)
    provider._client = SimpleNamespace(interactions=SimpleNamespace(create=create), files=None)
    provider.model = None
    provider.interaction_poll_interval_seconds = 0
    provider.interaction_processing_timeout_seconds = 1
    provider.file_poll_interval_seconds = 0
    provider.file_processing_timeout_seconds = 1
    provider.webhook_url = None
    provider.make_copy = False
    video = tmp_path / "input.mp4"
    video.write_bytes(b"fake")
    provider.analyze(video, duration_seconds=5)
    return sent, provider


def test_default_is_low_and_is_sent_to_gemini(cfg, tmp_path):
    sent, _ = run_provider(tmp_path)
    assert sent[0]["generation_config"] == {"thinking_level": "low"}
    assert model_config.thinking_setting("analysis") == ("low", "built-in default")


def test_config_file_sets_each_step(cfg, tmp_path):
    cfg({"thinking_level": {"analysis": "medium", "composition": "minimal", "narration": "high"}})
    sent, _ = run_provider(tmp_path)
    assert sent[0]["generation_config"] == {"thinking_level": "medium"}
    assert model_config.thinking_setting("composition")[0] == "minimal"
    assert model_config.thinking_setting("narration")[0] == "high"


def test_default_sends_nothing_and_env_beats_file(cfg, tmp_path, monkeypatch):
    cfg({"thinking_level": {"analysis": "default"}})
    sent, _ = run_provider(tmp_path)
    assert "generation_config" not in sent[0]
    monkeypatch.setenv(model_config.THINKING_ENV, "minimal")
    assert model_config.thinking_setting("analysis")[0] == "minimal"


@pytest.mark.parametrize("bad", [{"thinking_level": "low"}, {"thinking_level": {"analysis": "max"}},
                                 {"thinking_level": {"video": "low"}}])
def test_bad_thinking_settings_are_refused(bad):
    with pytest.raises(model_config.ModelConfigError):
        model_config.validate(bad)


def test_unsupported_model_steps_down_not_sideways():
    assert is_retryable_model_error(UNSUPPORTED)
    assert not is_quota_error(UNSUPPORTED)
    calls = []

    def attempt(model, key):
        calls.append((model, key))
        if model == "gemini-2.5-flash":
            raise UNSUPPORTED
        return "ok"

    logs = []
    _, model, _ = run_with_key_and_model_fallback(["gemini-2.5-flash", "gemini-3.8-flash"], attempt,
                                                  key_count=4, log=logs.append)
    assert calls == [("gemini-2.5-flash", 0), ("gemini-3.8-flash", 0)]
    assert model == "gemini-3.8-flash"
    assert "does not support this request" in logs[0]


def test_other_400s_still_stop_at_once():
    bad_input = ClientError("400 - {'error': {'message': 'Request contains an invalid argument.'}}")
    assert not is_retryable_model_error(bad_input)


def test_provider_skips_a_model_that_rejects_background_mode(cfg, tmp_path, monkeypatch):
    monkeypatch.setenv("VIDEO_ANALYSIS_SEMANTIC_MODELS", "gemini-2.5-flash,gemini-3.8-flash")
    sent, provider = run_provider(tmp_path, failures={"gemini-2.5-flash": UNSUPPORTED})
    assert [s["model"] for s in sent] == ["gemini-2.5-flash", "gemini-3.8-flash"]
    assert provider.model_used == "gemini-3.8-flash"


def test_defaults_leave_out_the_2_5_models():
    assert not [m for m in DEFAULT_SEMANTIC_MODELS + DEFAULT_COMPOSITION_MODELS if m.startswith("gemini-2.5")]
