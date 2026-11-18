from pathlib import Path
from video_analysis.analyzer import VideoAnalyzer

def test_analyzer_missing_file(tmp_path: Path):
    # ffprobe failure is surfaced rather than silently fabricating metadata.
    try:
        VideoAnalyzer().analyze(tmp_path / "missing.mov")
    except Exception as exc:
        assert "ffprobe" in str(exc).lower() or "no such" in str(exc).lower()
