"""The deterministic Gemini copy and the cheapest-first model order.

Real FFmpeg on small generated clips; Gemini is always a fake. No test here
makes a network call.
"""
import base64
import json
import shutil
import subprocess
import sys

import pytest

from video_analysis import gemini_input
from video_analysis.gemini import DEFAULT_COMPOSITION_MODELS, DEFAULT_SEMANTIC_MODELS, model_chain
from video_analysis.semantic import gemini as semantic_gemini
from video_analysis.semantic.gemini import GeminiSemanticProvider

pytestmark = pytest.mark.skipif(
    not (shutil.which("ffmpeg") and shutil.which("ffprobe")), reason="ffmpeg not installed"
)


def make_clip(path, width, height, seconds=2.0, audio=True):
    cmd = ["ffmpeg", "-nostdin", "-loglevel", "error", "-y",
           "-f", "lavfi", "-i", f"testsrc2=size={width}x{height}:rate=30:duration={seconds}"]
    if audio:
        cmd += ["-f", "lavfi", "-i", f"sine=frequency=440:duration={seconds}"]
    cmd += ["-c:v", "libx264", "-pix_fmt", "yuv420p"]
    if audio:
        cmd += ["-c:a", "aac", "-shortest"]
    cmd.append(str(path))
    subprocess.run(cmd, check=True)
    return path


def dimensions(path):
    out = subprocess.run(
        ["ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries",
         "stream=width,height", "-of", "csv=p=0", str(path)],
        capture_output=True, text=True, check=True,
    ).stdout.strip()
    width, height = out.split(",")[:2]
    return int(width), int(height)


def has_audio(path):
    out = subprocess.run(
        ["ffprobe", "-v", "error", "-select_streams", "a", "-show_entries", "stream=index",
         "-of", "csv=p=0", str(path)], capture_output=True, text=True, check=True,
    ).stdout.strip()
    return bool(out)


def test_copy_is_byte_identical_across_runs(tmp_path):
    source = make_clip(tmp_path / "input.mp4", 1920, 1080)
    first = gemini_input.make_gemini_copy(source, tmp_path / "a.mp4")
    second = gemini_input.make_gemini_copy(source, tmp_path / "b.mp4")
    assert first["sha256"] == second["sha256"]
    assert (tmp_path / "a.mp4").read_bytes() == (tmp_path / "b.mp4").read_bytes()


def test_copy_leaves_original_untouched_and_records_it(tmp_path):
    source = make_clip(tmp_path / "input.mp4", 1920, 1080)
    before = gemini_input.sha256_of(source)
    record = gemini_input.make_gemini_copy(source)
    assert gemini_input.sha256_of(source) == before
    assert record["source_sha256"] == before
    assert record["path"] == "input.gemini-input.mp4"
    assert (tmp_path / record["path"]).is_file()
    assert record["ffmpeg"].startswith("ffmpeg version")
    assert record["settings"]["max_short_side"] == 720
    assert abs(record["duration_seconds"] - gemini_input.probe_duration(source)) <= 0.25


def test_landscape_1080p_becomes_720p_with_audio(tmp_path):
    source = make_clip(tmp_path / "input.mp4", 1920, 1080)
    gemini_input.make_gemini_copy(source, tmp_path / "copy.mp4")
    assert dimensions(tmp_path / "copy.mp4") == (1280, 720)
    assert has_audio(tmp_path / "copy.mp4")


def test_portrait_keeps_orientation(tmp_path):
    source = make_clip(tmp_path / "input.mp4", 1080, 1920)
    gemini_input.make_gemini_copy(source, tmp_path / "copy.mp4")
    assert dimensions(tmp_path / "copy.mp4") == (720, 1280)


def test_small_video_is_not_upscaled(tmp_path):
    source = make_clip(tmp_path / "input.mp4", 640, 360)
    gemini_input.make_gemini_copy(source, tmp_path / "copy.mp4")
    assert dimensions(tmp_path / "copy.mp4") == (640, 360)


def test_video_without_audio_is_copied(tmp_path):
    source = make_clip(tmp_path / "input.mp4", 1280, 720, audio=False)
    gemini_input.make_gemini_copy(source, tmp_path / "copy.mp4")
    assert not has_audio(tmp_path / "copy.mp4")


def test_copy_refuses_to_overwrite_original(tmp_path):
    source = make_clip(tmp_path / "input.mp4", 640, 360)
    with pytest.raises(ValueError):
        gemini_input.make_gemini_copy(source, source)


def test_copy_can_be_switched_off(monkeypatch):
    monkeypatch.setenv(gemini_input.COPY_ENV, "0")
    assert gemini_input.copy_enabled() is False
    monkeypatch.delenv(gemini_input.COPY_ENV)
    assert gemini_input.copy_enabled() is True


def _semantic_payload():
    return {
        "overall_mood": "curious",
        "overall_mood_confidence": 0.8,
        "timeline": [{"start_seconds": 0.0, "end_seconds": 2.0, "activity": "test pattern",
                      "mood_intensity": 0.3, "confidence": 0.9}],
    }


def _fake_provider(make_copy):
    class FakeInteraction:
        id = "interaction-1"
        status = "completed"
        output_text = json.dumps(_semantic_payload())

    class FakeFiles:
        def upload(self, file):
            raise AssertionError("the small test clip should be sent inline")

    class FakeInteractions:
        def __init__(self):
            self.calls = []

        def create(self, **kwargs):
            self.calls.append(kwargs)
            return FakeInteraction()

    class FakeClient:
        files = FakeFiles()
        interactions = FakeInteractions()

    provider = object.__new__(GeminiSemanticProvider)
    provider._client = FakeClient()
    provider.model = None
    provider.file_poll_interval_seconds = 0
    provider.file_processing_timeout_seconds = 1
    provider.interaction_poll_interval_seconds = 0
    provider.interaction_processing_timeout_seconds = 1
    provider.webhook_url = None
    provider.make_copy = make_copy
    return provider


def test_gemini_receives_the_copy_not_the_original(tmp_path):
    source = make_clip(tmp_path / "input.mp4", 1920, 1080)
    original_bytes = source.read_bytes()
    provider = _fake_provider(make_copy=True)

    provider.analyze(source, duration_seconds=2.0)

    sent = base64.b64decode(provider._client.interactions.calls[0]["input"][0]["data"])
    copy_file = tmp_path / provider.gemini_input["path"]
    assert sent == copy_file.read_bytes()
    assert sent != original_bytes
    assert source.read_bytes() == original_bytes


def test_copy_off_sends_the_original(tmp_path):
    source = make_clip(tmp_path / "input.mp4", 640, 360)
    provider = _fake_provider(make_copy=False)

    provider.analyze(source, duration_seconds=2.0)

    sent = base64.b64decode(provider._client.interactions.calls[0]["input"][0]["data"])
    assert sent == source.read_bytes()
    assert provider.gemini_input is None


def test_pipeline_records_what_gemini_was_sent(tmp_path):
    from video_analysis.pipeline import VideoAnalysisPipeline

    source = make_clip(tmp_path / "input.mp4", 1920, 1080)
    provider = _fake_provider(make_copy=True)
    result = VideoAnalysisPipeline(semantic_provider=provider).analyze(source)

    record = result["video"]["gemini_input"]
    assert record["path"] == "input.gemini-input.mp4"
    assert record["source_sha256"] == gemini_input.sha256_of(source)


# Built-in order: gemini-3.8-flash is the baseline (see test_model_config.py for the config file).

def test_semantic_chain_starts_with_the_baseline_model(monkeypatch):
    monkeypatch.delenv("VIDEO_ANALYSIS_SEMANTIC_MODELS", raising=False)
    monkeypatch.delenv("VIDEO_ANALYSIS_MAX_MODEL_ATTEMPTS", raising=False)
    assert model_chain(None, "VIDEO_ANALYSIS_SEMANTIC_MODELS", DEFAULT_SEMANTIC_MODELS)[:3] == [
        "gemini-3.8-flash", "gemini-3.6-flash", "gemini-4-hydrogen"]
    assert DEFAULT_SEMANTIC_MODELS[-1].endswith("flash-lite")


def test_composition_chain_starts_with_the_baseline_model(monkeypatch):
    monkeypatch.delenv("VIDEO_ANALYSIS_COMPOSITION_MODELS", raising=False)
    monkeypatch.delenv("VIDEO_ANALYSIS_MAX_MODEL_ATTEMPTS", raising=False)
    assert model_chain(None, "VIDEO_ANALYSIS_COMPOSITION_MODELS", DEFAULT_COMPOSITION_MODELS)[0] == \
        "gemini-3.8-flash"


def test_chain_is_cut_at_the_attempt_cap(monkeypatch):
    monkeypatch.setenv("VIDEO_ANALYSIS_MAX_MODEL_ATTEMPTS", "2")
    assert model_chain(None, "VIDEO_ANALYSIS_SEMANTIC_MODELS", DEFAULT_SEMANTIC_MODELS) == [
        "gemini-3.8-flash", "gemini-3.6-flash"]


def test_fallback_off_without_primary_uses_first_of_chain(monkeypatch):
    monkeypatch.setenv("VIDEO_ANALYSIS_MODEL_FALLBACK", "0")
    assert model_chain(None, "VIDEO_ANALYSIS_SEMANTIC_MODELS", DEFAULT_SEMANTIC_MODELS) == [
        "gemini-3.8-flash"]


def test_cli_no_longer_forces_a_model(monkeypatch):
    from video_analysis import cli

    captured = {}

    class StopHere(Exception):
        pass

    class FakeSemantic:
        def __init__(self, model):
            captured["semantic"] = model
            raise StopHere

    monkeypatch.setattr(sys, "argv", ["video-analysis", "analyze", "x.mp4", "--semantic", "gemini"])
    monkeypatch.setattr(semantic_gemini, "GeminiSemanticProvider", FakeSemantic)
    with pytest.raises(StopHere):
        cli.main()
    assert captured["semantic"] is None

    class FakeComposition:
        def __init__(self, model):
            captured["composition"] = model
            raise StopHere

    monkeypatch.setattr(sys, "argv", ["video-analysis", "compose", "analysis.json"])
    monkeypatch.setattr(cli, "GeminiCompositionProvider", FakeComposition)
    with pytest.raises(StopHere):
        cli.main()
    assert captured["composition"] is None
