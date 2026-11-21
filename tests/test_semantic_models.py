import json

import pytest
from pydantic import ValidationError

from video_analysis.semantic.models import SemanticAnalysis
from video_analysis.semantic.merge import merge_semantic_analysis
from video_analysis.semantic import gemini
from video_analysis.semantic.gemini import GeminiSemanticProvider, _gemini_response_schema


def valid_semantic():
    return {
        "schema_version": "0.3",
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


def test_gemini_large_video_uses_files_api_uri(monkeypatch, tmp_path):
    video = tmp_path / "large.mov"
    video.write_bytes(b"video-bytes")

    class FakeFile:
        def __init__(self, state, uri=None):
            self.state = state
            self.name = "files/test-video"
            self.uri = uri
            self.mime_type = "video/quicktime"

    class FakeInteraction:
        id = "interaction-test"
        status = "completed"
        output_text = json.dumps(valid_semantic())

    class FakeFiles:
        def __init__(self):
            self.uploaded = []
            self.get_calls = 0

        def upload(self, file):
            self.uploaded.append(file)
            return FakeFile("PROCESSING")

        def get(self, name):
            self.get_calls += 1
            return FakeFile("ACTIVE", "https://generativelanguage.googleapis.com/v1beta/files/test-video")

    class FakeInteractions:
        def __init__(self):
            self.calls = []

        def create(self, **kwargs):
            self.calls.append(kwargs)
            return FakeInteraction()

    class FakeClient:
        def __init__(self):
            self.files = FakeFiles()
            self.interactions = FakeInteractions()

    provider = object.__new__(GeminiSemanticProvider)
    provider._client = FakeClient()
    provider.model = "gemini-test"
    provider.file_poll_interval_seconds = 0
    provider.file_processing_timeout_seconds = 1
    provider.interaction_poll_interval_seconds = 0
    provider.interaction_processing_timeout_seconds = 1
    provider.webhook_url = None

    monkeypatch.setattr(gemini, "MAX_INLINE_VIDEO_BYTES", 1)

    result = provider.analyze(video, duration_seconds=5.0)

    assert result.analysis_type == "semantic"
    assert provider._client.files.uploaded == [str(video)]
    assert provider._client.files.get_calls == 1

    video_input = provider._client.interactions.calls[0]["input"][0]
    assert video_input["type"] == "video"
    assert video_input["uri"].endswith("/files/test-video")
    assert video_input["mime_type"] == "video/quicktime"
    assert "data" not in video_input


def test_gemini_small_video_stays_inline(monkeypatch, tmp_path):
    video = tmp_path / "small.mp4"
    video.write_bytes(b"small-video")

    class FakeInteraction:
        # Background interactions return an id and a status; "completed"
        # skips polling.
        id = "interaction-1"
        status = "completed"
        output_text = json.dumps(valid_semantic())

    class FakeFiles:
        def upload(self, file):
            raise AssertionError("small videos should not use Files API")

    class FakeInteractions:
        def __init__(self):
            self.calls = []

        def create(self, **kwargs):
            self.calls.append(kwargs)
            return FakeInteraction()

    class FakeClient:
        def __init__(self):
            self.files = FakeFiles()
            self.interactions = FakeInteractions()

    provider = object.__new__(GeminiSemanticProvider)
    provider._client = FakeClient()
    provider.model = "gemini-test"
    provider.file_poll_interval_seconds = 0
    provider.file_processing_timeout_seconds = 1
    provider.interaction_poll_interval_seconds = 0
    provider.interaction_processing_timeout_seconds = 1
    provider.webhook_url = None

    monkeypatch.setattr(gemini, "MAX_INLINE_VIDEO_BYTES", 1024)

    result = provider.analyze(video, duration_seconds=5.0)

    assert result.analysis_type == "semantic"
    video_input = provider._client.interactions.calls[0]["input"][0]
    assert video_input["type"] == "video"
    assert video_input["mime_type"] == "video/mp4"
    assert "data" in video_input
    assert "uri" not in video_input


def test_gemini_background_interaction_uses_webhook(monkeypatch, tmp_path):
    video = tmp_path / "video.mp4"
    video.write_bytes(b"video")

    class FakeInteraction:
        id = "interaction-webhook"
        status = "completed"
        output_text = json.dumps(valid_semantic())

    class FakeInteractions:
        def __init__(self):
            self.calls = []

        def create(self, **kwargs):
            self.calls.append(kwargs)
            return FakeInteraction()

        def get(self, interaction_id):
            raise AssertionError("completed interactions should not be polled")

    class FakeClient:
        def __init__(self):
            self.interactions = FakeInteractions()
            self.files = None

    provider = object.__new__(GeminiSemanticProvider)
    provider._client = FakeClient()
    provider.model = "gemini-test"
    provider.interaction_poll_interval_seconds = 0
    provider.interaction_processing_timeout_seconds = 1
    provider.webhook_url = "https://example.test/gemini-webhook"
    provider.file_poll_interval_seconds = 0
    provider.file_processing_timeout_seconds = 1

    monkeypatch.setattr(gemini, "MAX_INLINE_VIDEO_BYTES", 1024)

    result = provider.analyze(video, duration_seconds=5.0)

    assert result.analysis_type == "semantic"
    kwargs = provider._client.interactions.calls[0]
    assert kwargs["background"] is True
    assert kwargs["store"] is True
    assert kwargs["webhook_config"]["uris"] == ["https://example.test/gemini-webhook"]
    assert kwargs["webhook_config"]["user_metadata"]["analysis_type"] == "semantic"


def test_video_type_defaults_when_missing():
    s = SemanticAnalysis.model_validate(valid_semantic())
    assert s.video_type.primary == "unknown"
    assert s.video_type.confidence == 0.0
    assert s.video_type.alternatives == []
    assert s.screen_actions == []


def test_screen_actions_validate():
    payload = valid_semantic()
    payload["screen_actions"] = [{"timestamp_seconds": 2.0, "action": "click", "target": "Save"}]
    s = SemanticAnalysis.model_validate(payload)
    assert s.screen_actions[0].visible_text == ""
    payload["screen_actions"][0]["timestamp_seconds"] = 99
    with pytest.raises(ValidationError):
        SemanticAnalysis.model_validate(payload)
