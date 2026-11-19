import pytest

from video_analysis.gemini import (
    is_retryable_model_error,
    model_chain,
    run_with_model_fallback,
)


INTERNAL = RuntimeError(
    "Gemini interaction v1_x ended with status=failed; "
    "errors=[Error(code='INTERNAL', message='There was a problem processing your request.')]"
)


class QuotaError(Exception):
    code = 429


def test_falls_through_to_next_model_on_server_and_quota_errors():
    calls, logs = [], []

    def attempt(model):
        calls.append(model)
        if model == "a":
            raise INTERNAL
        if model == "b":
            raise QuotaError("RESOURCE_EXHAUSTED")
        return "ok"

    result, used = run_with_model_fallback(["a", "b", "c", "d"], attempt, log=logs.append)
    assert (result, used) == ("ok", "c")
    assert calls == ["a", "b", "c"]
    assert logs[-1] == "Gemini model used: c"


def test_request_errors_are_not_retried_on_other_models():
    calls = []

    def attempt(model):
        calls.append(model)
        raise ValueError("Gemini semantic response failed local validation")

    with pytest.raises(ValueError):
        run_with_model_fallback(["a", "b"], attempt, log=lambda m: None)
    assert calls == ["a"]


def test_timeouts_are_not_retried():
    assert not is_retryable_model_error(TimeoutError("did not complete within 1800 seconds"))


def test_all_models_failing_reports_each_one():
    with pytest.raises(RuntimeError, match="a: .*INTERNAL.*b: .*INTERNAL"):
        run_with_model_fallback(["a", "b"], lambda m: (_ for _ in ()).throw(INTERNAL), log=lambda m: None)


def test_chain_puts_requested_model_first_without_repeats(monkeypatch):
    monkeypatch.delenv("VIDEO_ANALYSIS_MODEL_FALLBACK", raising=False)
    monkeypatch.delenv("X_MODELS", raising=False)
    monkeypatch.delenv("VIDEO_ANALYSIS_MAX_MODEL_ATTEMPTS", raising=False)
    assert model_chain("b", "X_MODELS", ["a", "b", "c", "d", "e"]) == ["b", "a", "c"]
    monkeypatch.setenv("VIDEO_ANALYSIS_MAX_MODEL_ATTEMPTS", "5")
    assert model_chain("b", "X_MODELS", ["a", "b", "c", "d", "e"]) == ["b", "a", "c", "d", "e"]
    monkeypatch.setenv("X_MODELS", "z, y")
    assert model_chain("b", "X_MODELS", ["a"]) == ["b", "z", "y"]
    monkeypatch.setenv("VIDEO_ANALYSIS_MODEL_FALLBACK", "0")
    assert model_chain("b", "X_MODELS", ["a"]) == ["b"]
