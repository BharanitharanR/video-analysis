"""The model order comes from a config file, not code.

The config path is a temp file per test (see conftest); no network calls.
"""
import json
import subprocess
import sys

import pytest

from video_analysis import model_config
from video_analysis.gemini import (
    DEFAULT_COMPOSITION_MODELS,
    DEFAULT_SEMANTIC_MODELS,
    max_attempts,
    model_chain,
)

SEM = "VIDEO_ANALYSIS_SEMANTIC_MODELS"
COMP = "VIDEO_ANALYSIS_COMPOSITION_MODELS"


@pytest.fixture
def config(monkeypatch, tmp_path):
    for name in (SEM, COMP, "VIDEO_ANALYSIS_MAX_MODEL_ATTEMPTS", "VIDEO_ANALYSIS_MODEL_FALLBACK"):
        monkeypatch.delenv(name, raising=False)
    path = tmp_path / "gemini-models.json"
    monkeypatch.setenv(model_config.CONFIG_ENV, str(path))

    def write(data):
        path.write_text(json.dumps(data), encoding="utf-8")
        return path

    return write


def test_no_file_uses_the_built_in_order(config):
    assert model_chain(None, SEM, DEFAULT_SEMANTIC_MODELS)[0] == "gemini-3.8-flash"
    assert model_config.step_models("analysis", DEFAULT_SEMANTIC_MODELS)[1] == "built-in defaults"


def test_file_order_is_the_order_tried(config):
    config({"analysis": ["gemini-2.5-flash", "gemini-4-hydrogen", "gemini-3.8-flash"]})
    assert model_chain(None, SEM, DEFAULT_SEMANTIC_MODELS) == [
        "gemini-2.5-flash", "gemini-4-hydrogen", "gemini-3.8-flash"]
    # composition is not in the file, so it keeps its default
    assert model_chain(None, COMP, DEFAULT_COMPOSITION_MODELS)[0] == "gemini-3.8-flash"


def test_steps_have_independent_orders(config):
    config({"analysis": ["gemini-3.6-flash"], "composition": ["gemini-2.5-flash", "gemini-3.8-flash"]})
    assert model_chain(None, SEM, DEFAULT_SEMANTIC_MODELS) == ["gemini-3.6-flash"]
    assert model_chain(None, COMP, DEFAULT_COMPOSITION_MODELS) == ["gemini-2.5-flash", "gemini-3.8-flash"]


def test_switched_off_models_are_never_tried(config):
    config({"analysis": ["gemini-3.8-flash", "gemini-4-hydrogen", "gemini-2.5-flash"],
            "disabled": {"analysis": ["gemini-4-hydrogen"]}})
    assert model_chain(None, SEM, DEFAULT_SEMANTIC_MODELS) == ["gemini-3.8-flash", "gemini-2.5-flash"]


def test_edits_apply_on_the_next_read_without_restart(config):
    config({"analysis": ["gemini-3.8-flash"]})
    assert model_chain(None, SEM, DEFAULT_SEMANTIC_MODELS) == ["gemini-3.8-flash"]
    config({"analysis": ["gemini-3.6-flash"]})
    assert model_chain(None, SEM, DEFAULT_SEMANTIC_MODELS) == ["gemini-3.6-flash"]


def test_environment_variable_beats_the_file(config, monkeypatch):
    config({"analysis": ["gemini-3.6-flash"]})
    monkeypatch.setenv(SEM, "gemini-2.5-flash")
    assert model_chain(None, SEM, DEFAULT_SEMANTIC_MODELS) == ["gemini-2.5-flash"]


def test_forced_model_goes_first_then_the_file_order(config):
    config({"analysis": ["gemini-3.6-flash", "gemini-2.5-flash"]})
    assert model_chain("gemini-4-hydrogen", SEM, DEFAULT_SEMANTIC_MODELS) == [
        "gemini-4-hydrogen", "gemini-3.6-flash", "gemini-2.5-flash"]


def test_fallback_off_uses_the_first_model_in_the_file(config, monkeypatch):
    config({"analysis": ["gemini-3.6-flash", "gemini-2.5-flash"]})
    monkeypatch.setenv("VIDEO_ANALYSIS_MODEL_FALLBACK", "0")
    assert model_chain(None, SEM, DEFAULT_SEMANTIC_MODELS) == ["gemini-3.6-flash"]


def test_attempt_cap_from_file_and_env_beats_it(config, monkeypatch):
    config({"max_attempts": 2})
    assert max_attempts() == 2
    assert len(model_chain(None, SEM, DEFAULT_SEMANTIC_MODELS)) == 2
    monkeypatch.setenv("VIDEO_ANALYSIS_MAX_MODEL_ATTEMPTS", "4")
    assert max_attempts() == 4


@pytest.mark.parametrize("bad", [
    "{not json",
    json.dumps(["gemini-3.8-flash"]),
    json.dumps({"analysis": "gemini-3.8-flash"}),
    json.dumps({"analysis": ["gemini 3.8 flash; rm -rf"]}),
    json.dumps({"analysis": ["gemini-3.8-flash"], "disabled": {"analysis": ["gemini-3.8-flash"]}}),
    json.dumps({"max_attempts": 0}),
    json.dumps({"max_attempts": True}),
], ids=["not-json", "not-object", "not-list", "bad-name", "all-switched-off", "cap-zero", "cap-bool"])
def test_a_broken_file_fails_before_any_gemini_call(config, bad):
    path = config({})
    path.write_text(bad, encoding="utf-8")
    with pytest.raises(model_config.ModelConfigError):
        model_chain(None, SEM, DEFAULT_SEMANTIC_MODELS)


def test_broken_file_error_is_not_retried_on_other_models():
    from video_analysis.gemini import is_retryable_model_error
    assert not is_retryable_model_error(model_config.ModelConfigError("bad file"))


def test_save_validates_dedupes_and_round_trips(config, tmp_path):
    path = model_config.save({"analysis": ["gemini-3.8-flash", "gemini-3.8-flash", " gemini-2.5-flash "],
                              "max_attempts": 3})
    assert json.loads(path.read_text()) == {"analysis": ["gemini-3.8-flash", "gemini-2.5-flash"],
                                            "max_attempts": 3}
    assert not list(path.parent.glob(".gemini-models.*.tmp"))
    with pytest.raises(model_config.ModelConfigError):
        model_config.save({"analysis": []})
    # the failed save left the previous file in place
    assert json.loads(path.read_text())["analysis"][0] == "gemini-3.8-flash"


def test_provider_uses_the_file_order(config, tmp_path, monkeypatch):
    from types import SimpleNamespace

    from video_analysis.semantic.gemini import GeminiSemanticProvider

    config({"analysis": ["gemini-2.5-flash", "gemini-3.8-flash"]})
    answer = json.dumps({"overall_mood": "calm", "overall_mood_confidence": 0.5,
                         "timeline": [{"start_seconds": 0, "end_seconds": 5, "activity": "x",
                                       "mood_intensity": 0.2, "confidence": 0.5}]})
    models_called = []

    def create(**kwargs):
        models_called.append(kwargs["model"])
        return SimpleNamespace(id="i", status="completed", output_text=answer)

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
    assert models_called == ["gemini-2.5-flash"]


# The `video-analysis models` command, for use without the studio.

def run_models(*args, env_path):
    import os

    env = {k: v for k, v in os.environ.items()
           if not k.startswith("GEMINI_API_KEY") and k not in (SEM, COMP, "VIDEO_ANALYSIS_MAX_MODEL_ATTEMPTS")}
    env[model_config.CONFIG_ENV] = str(env_path)
    env["GEMINI_API_KEY_1"] = "fake-1"
    env["GEMINI_API_KEY_2"] = "fake-2"
    return subprocess.run([sys.executable, "-m", "video_analysis.cli", "models", *args],
                          capture_output=True, text=True, env=env)


def test_models_command_reports_order_source_and_key_count(config):
    path = config({"analysis": ["gemini-3.6-flash", "gemini-4-hydrogen"],
                   "disabled": {"analysis": ["gemini-4-hydrogen"]}})
    out = run_models("--json", env_path=path)
    assert out.returncode == 0, out.stderr
    report = json.loads(out.stdout)
    assert report["key_count"] == 2
    assert report["config_exists"] is True
    analysis = report["steps"]["analysis"]
    assert analysis["order"] == ["gemini-3.6-flash"]
    assert analysis["configured"] == ["gemini-3.6-flash", "gemini-4-hydrogen"]
    assert analysis["disabled"] == ["gemini-4-hydrogen"]
    assert analysis["source"].startswith("config file")
    assert report["steps"]["composition"]["source"] == "built-in defaults"
    assert "fake-1" not in out.stdout + out.stderr


def test_models_command_text_output_and_write_defaults(config, tmp_path):
    path = tmp_path / "new" / "gemini-models.json"
    out = run_models("--write-defaults", env_path=path)
    assert out.returncode == 0, out.stderr
    assert json.loads(path.read_text())["analysis"][0] == "gemini-3.8-flash"
    assert "1. gemini-3.8-flash" in out.stdout
    # a second run never overwrites the user's file
    path.write_text(json.dumps({"analysis": ["gemini-2.5-flash"]}))
    out = run_models("--write-defaults", env_path=path)
    assert json.loads(path.read_text()) == {"analysis": ["gemini-2.5-flash"]}
    assert "left unchanged" in out.stderr


def test_models_command_reports_a_broken_file(config, tmp_path):
    path = tmp_path / "gemini-models.json"
    path.write_text("{oops")
    out = run_models(env_path=path)
    assert out.returncode == 2
    assert "Model config error" in out.stderr


def test_models_command_saves_a_validated_config(config, tmp_path):
    import os

    path = tmp_path / "gemini-models.json"
    env = dict(os.environ, **{model_config.CONFIG_ENV: str(path)})
    doc = {"analysis": ["gemini-2.5-flash", "gemini-3.8-flash"], "max_attempts": 4}
    out = subprocess.run([sys.executable, "-m", "video_analysis.cli", "models", "--save", "-", "--json"],
                         input=json.dumps(doc), capture_output=True, text=True, env=env)
    assert out.returncode == 0, out.stderr
    assert json.loads(path.read_text()) == doc
    assert json.loads(out.stdout)["steps"]["analysis"]["order"] == ["gemini-2.5-flash", "gemini-3.8-flash"]
    bad = subprocess.run([sys.executable, "-m", "video_analysis.cli", "models", "--save", "-"],
                         input=json.dumps({"analysis": []}), capture_output=True, text=True, env=env)
    assert bad.returncode == 2
    assert json.loads(path.read_text()) == doc  # unchanged
