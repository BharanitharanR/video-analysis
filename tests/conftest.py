import pytest


@pytest.fixture(autouse=True)
def _isolate_raw_response_logs(monkeypatch, tmp_path):
    """Keep Gemini raw-response logs out of the repo working directory."""
    monkeypatch.setenv("VIDEO_ANALYSIS_GEMINI_RAW_RESPONSE", str(tmp_path / "gemini-response.raw.json"))
    monkeypatch.setenv("VIDEO_ANALYSIS_COMPOSITION_RAW_RESPONSE", str(tmp_path / "composition-response.raw.json"))
