"""Several API keys: each model is tried on key 1, 2, ... before stepping down.

Keys here are fake strings and Gemini is always a fake. No network calls.
"""
import json
from types import SimpleNamespace

import pytest

from video_analysis import gemini as shared
from video_analysis.gemini import (
    InvalidModelResponse,
    api_keys,
    is_quota_error,
    run_with_key_and_model_fallback,
)
from video_analysis.semantic import gemini as semantic_gemini
from video_analysis.semantic.gemini import GeminiSemanticProvider

FAKE_KEYS = [f"fake-key-{n}-do-not-log" for n in range(1, 6)]


class QuotaError(Exception):
    code = 429


QUOTA = QuotaError("429 RESOURCE_EXHAUSTED: quota exceeded for model")
INTERNAL = RuntimeError("Gemini interaction ended with status=failed; errors=[Error(code='INTERNAL')]")
DEADLINE = RuntimeError("504 DEADLINE_EXCEEDED")


@pytest.fixture(autouse=True)
def clean_env(monkeypatch):
    for n in range(1, 21):
        monkeypatch.delenv(f"GEMINI_API_KEY_{n}", raising=False)
    for name in ("GEMINI_API_KEY", "GOOGLE_API_KEY", "VIDEO_ANALYSIS_MAX_MODEL_ATTEMPTS",
                 "VIDEO_ANALYSIS_SEMANTIC_MODELS", "VIDEO_ANALYSIS_MODEL_FALLBACK"):
        monkeypatch.delenv(name, raising=False)


def scripted(outcomes):
    """attempt(model, key) that raises or returns per (model, key); records calls."""
    calls = []

    def attempt(model, key):
        calls.append((model, key))
        outcome = outcomes.get((model, key), outcomes.get(model, "ok"))
        if isinstance(outcome, BaseException):
            raise outcome
        return f"{model}@{key}"

    return attempt, calls


# Which keys, in which order.

def test_numbered_keys_in_order_without_repeats(monkeypatch):
    monkeypatch.setenv("GEMINI_API_KEY_1", "k1")
    monkeypatch.setenv("GEMINI_API_KEY_2", "k2")
    monkeypatch.setenv("GEMINI_API_KEY_3", "k1")  # duplicate of key 1
    monkeypatch.setenv("GEMINI_API_KEY_5", "k5")  # a gap is skipped, not the end
    monkeypatch.setenv("GEMINI_API_KEY", "single")
    assert api_keys() == ["k1", "k2", "k5"]


def test_single_key_when_no_numbered_keys(monkeypatch):
    monkeypatch.setenv("GEMINI_API_KEY", "single")
    assert api_keys() == ["single"]


def test_no_keys_fails_before_any_call():
    with pytest.raises(RuntimeError, match="GEMINI_API_KEY"):
        shared.create_clients()


def test_each_client_gets_its_key_explicitly_not_google_api_key(monkeypatch):
    from google import genai

    made = []
    monkeypatch.setattr(genai, "Client", lambda **kwargs: made.append(kwargs) or object())
    monkeypatch.setenv("GOOGLE_API_KEY", "some-other-project")
    for n, key in enumerate(FAKE_KEYS[:3], start=1):
        monkeypatch.setenv(f"GEMINI_API_KEY_{n}", key)
    assert len(shared.create_clients()) == 3
    assert [m["api_key"] for m in made] == FAKE_KEYS[:3]


# Which failures rotate keys.

def test_only_quota_errors_rotate_keys():
    assert is_quota_error(QUOTA)
    assert is_quota_error(RuntimeError("RESOURCE_EXHAUSTED"))
    assert not is_quota_error(INTERNAL)
    assert not is_quota_error(DEADLINE)
    assert not is_quota_error(InvalidModelResponse("bad answer"))
    assert not is_quota_error(TimeoutError("did not complete"))


# The order of attempts.

MODELS = ["gemini-4-hydrogen", "gemini-3.8-flash", "gemini-3.6-flash"]


def test_quota_tries_the_same_model_on_the_next_key():
    attempt, calls = scripted({("gemini-4-hydrogen", 0): QUOTA, ("gemini-4-hydrogen", 1): QUOTA})
    result, model, key = run_with_key_and_model_fallback(MODELS, attempt, key_count=5, log=lambda m: None)
    assert calls == [("gemini-4-hydrogen", 0), ("gemini-4-hydrogen", 1), ("gemini-4-hydrogen", 2)]
    assert (model, key) == ("gemini-4-hydrogen", 2)


def test_every_key_out_of_quota_steps_down_a_model_from_key_1():
    attempt, calls = scripted({"gemini-4-hydrogen": QUOTA})
    result, model, key = run_with_key_and_model_fallback(MODELS, attempt, key_count=5, log=lambda m: None)
    assert calls == [("gemini-4-hydrogen", k) for k in range(5)] + [("gemini-3.8-flash", 0)]
    assert (model, key) == ("gemini-3.8-flash", 0)


def test_second_model_also_rotates_keys_on_quota():
    attempt, calls = scripted({"gemini-4-hydrogen": QUOTA, ("gemini-3.8-flash", 0): QUOTA})
    _, model, key = run_with_key_and_model_fallback(MODELS, attempt, key_count=2, log=lambda m: None)
    assert calls == [("gemini-4-hydrogen", 0), ("gemini-4-hydrogen", 1),
                     ("gemini-3.8-flash", 0), ("gemini-3.8-flash", 1)]
    assert (model, key) == ("gemini-3.8-flash", 1)


@pytest.mark.parametrize("error", [INTERNAL, DEADLINE, InvalidModelResponse("bad answer")],
                         ids=["internal", "deadline", "invalid-answer"])
def test_non_quota_failures_step_down_a_model_without_trying_other_keys(error):
    attempt, calls = scripted({"gemini-4-hydrogen": error})
    _, model, key = run_with_key_and_model_fallback(MODELS, attempt, key_count=5, log=lambda m: None)
    assert calls == [("gemini-4-hydrogen", 0), ("gemini-3.8-flash", 0)]
    assert (model, key) == ("gemini-3.8-flash", 0)


def test_default_cap_is_six_attempts_in_all():
    attempt, calls = scripted({m: QUOTA for m in MODELS})
    with pytest.raises(RuntimeError, match="cap of 6 attempts"):
        run_with_key_and_model_fallback(MODELS, attempt, key_count=5, log=lambda m: None)
    assert len(calls) == 6
    assert calls[-1] == ("gemini-3.8-flash", 0)


def test_cap_can_be_changed(monkeypatch):
    monkeypatch.setenv("VIDEO_ANALYSIS_MAX_MODEL_ATTEMPTS", "2")
    attempt, calls = scripted({m: QUOTA for m in MODELS})
    with pytest.raises(RuntimeError):
        run_with_key_and_model_fallback(MODELS, attempt, key_count=5, log=lambda m: None)
    assert len(calls) == 2


def test_non_retryable_errors_stop_at_once():
    attempt, calls = scripted({"gemini-4-hydrogen": TimeoutError("did not complete within 1800 seconds")})
    with pytest.raises(TimeoutError):
        run_with_key_and_model_fallback(MODELS, attempt, key_count=5, log=lambda m: None)
    assert calls == [("gemini-4-hydrogen", 0)]


def test_logs_name_key_numbers_never_key_values():
    logs = []
    attempt, _ = scripted({"gemini-4-hydrogen": QUOTA})
    run_with_key_and_model_fallback(MODELS, attempt, key_count=2, log=logs.append)
    assert logs[0].startswith("Gemini model gemini-4-hydrogen (key 1/2) out of quota")
    assert logs[0].endswith("trying key 2/2")
    assert "is out of quota on every key" in logs[1] and logs[1].endswith("trying gemini-3.8-flash from key 1")
    assert logs[-1] == "Gemini model used: gemini-3.8-flash (key 1/2)"
    assert not any("fake-key" in line for line in logs)


# End to end through the semantic provider, with one fake client per key.

ANSWER = json.dumps({
    "overall_mood": "excited", "overall_mood_confidence": 0.8,
    "timeline": [{"start_seconds": 0, "end_seconds": 10, "activity": "demo",
                  "mood_intensity": 0.5, "confidence": 0.9}],
})


class FakeClient:
    def __init__(self, name, failures=None):
        self.name = name
        self.failures = failures or {}
        self.calls = []
        self.uploads = 0
        self.interactions = SimpleNamespace(create=self._create)
        self.files = SimpleNamespace(upload=self._upload, get=None)

    def _create(self, **kwargs):
        self.calls.append((kwargs["model"], kwargs["input"][0]))
        failure = self.failures.get(kwargs["model"])
        if failure:
            raise failure
        return SimpleNamespace(id="i", status="completed", output_text=ANSWER)

    def _upload(self, file):
        self.uploads += 1
        return SimpleNamespace(state="ACTIVE", uri=f"files/{self.name}-{self.uploads}", mime_type="video/mp4")


def provider_with(clients, tmp_path, monkeypatch):
    monkeypatch.setenv("VIDEO_ANALYSIS_GEMINI_RAW_RESPONSE", str(tmp_path / "gemini-response.json"))
    provider = object.__new__(GeminiSemanticProvider)
    provider._clients = clients
    provider._client = clients[0]
    provider.model = None
    provider.file_poll_interval_seconds = 0
    provider.file_processing_timeout_seconds = 1
    provider.interaction_poll_interval_seconds = 0
    provider.interaction_processing_timeout_seconds = 1
    provider.webhook_url = None
    provider.make_copy = False
    video = tmp_path / "input.mp4"
    video.write_bytes(b"fake video bytes")
    return provider, video


def test_provider_moves_to_the_next_key_on_quota(tmp_path, monkeypatch):
    clients = [FakeClient("k1", {"gemini-4-hydrogen": QUOTA}), FakeClient("k2"), FakeClient("k3")]
    provider, video = provider_with(clients, tmp_path, monkeypatch)
    monkeypatch.setenv("VIDEO_ANALYSIS_SEMANTIC_MODELS", "gemini-4-hydrogen,gemini-3.8-flash")

    provider.analyze(video, duration_seconds=10)

    assert [c[0] for c in clients[0].calls] == ["gemini-4-hydrogen"]
    assert [c[0] for c in clients[1].calls] == ["gemini-4-hydrogen"]
    assert clients[2].calls == []
    assert (provider.model_used, provider.key_used) == ("gemini-4-hydrogen", 1)


def test_large_video_is_uploaded_once_per_key_used(tmp_path, monkeypatch):
    monkeypatch.setattr(semantic_gemini, "MAX_INLINE_VIDEO_BYTES", 1)  # force the Files API path
    clients = [FakeClient("k1", {"gemini-4-hydrogen": QUOTA, "gemini-3.8-flash": QUOTA}),
               FakeClient("k2", {"gemini-4-hydrogen": QUOTA}),
               FakeClient("k3")]
    provider, video = provider_with(clients, tmp_path, monkeypatch)
    monkeypatch.setenv("VIDEO_ANALYSIS_SEMANTIC_MODELS", "gemini-4-hydrogen,gemini-3.8-flash")

    provider.analyze(video, duration_seconds=10)

    # hydrogen: k1, k2 (quota), k3 succeeds. Only k1..k3 were used.
    assert [c.uploads for c in clients] == [1, 1, 1]
    # Each key's request references that key's own upload.
    assert clients[2].calls[0][1]["uri"] == "files/k3-1"
    assert (provider.model_used, provider.key_used) == ("gemini-4-hydrogen", 2)
