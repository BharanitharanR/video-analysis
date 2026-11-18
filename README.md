# video-analysis

A standalone, dimension-based video analysis engine. It separates deterministic
media measurements from AI-generated semantic interpretation.

## Design

The engine produces two independent artifacts:

1. **Deterministic report** — technical metadata, audio stream metadata, GPS
   metadata, quality measurements, scene detection, and temporal evidence.
2. **Semantic report** — a separate JSON document containing temporal mode,
   activity, mood, observations, events, and content inferred from the video.

The Gemini provider returns only the semantic document. It does not append to,
rewrite, or reproduce the deterministic report. The application validates the
semantic response with Pydantic and the merger adds it under
`video.dimensions.semantic`, preserving all existing deterministic dimensions.

## Install

Python 3.10+ is required. FFprobe (from FFmpeg) must be available on PATH for
technical media inspection.

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -e '.[scenes,gemini,dev]'
```

For deterministic analysis without Gemini:

```bash
python -m pip install -e '.[scenes,dev]'
```

Set `GEMINI_API_KEY` in your environment before using the Gemini provider.
Do not commit API keys.

## Semantic-only provider

```python
from video_analysis.semantic.gemini import GeminiSemanticProvider

provider = GeminiSemanticProvider(model="gemini-2.5-flash")
semantic = provider.analyze("clip.mov", duration_seconds=9.568333)
semantic_json = semantic.model_dump(mode="json")
```

The semantic response is a separate artifact with `analysis_type: "semantic"`.
It contains the duration constraint, overall mood, content summary, temporal
segments, and localized events. It intentionally excludes codec, resolution,
FPS, bitrate, quality metrics, and GPS metadata.

## Merge

```python
from video_analysis.semantic.merge import merge_semantic_analysis

final_report = merge_semantic_analysis(
    deterministic_report=deterministic_json,
    semantic=semantic,
)
```

The merge is additive and namespaced. Existing deterministic dimensions are
never overwritten.

## Gemini configuration

Install the optional integration using `.[gemini]`. The provider uses the
Google GenAI Python SDK, reads `GEMINI_API_KEY` from the environment, requests
JSON output conforming to the Pydantic schema, parses the response, and validates
it again locally. Gemini is optional; deterministic analysis does not require
the Google SDK.

## Supported containers

MOV, MP4, MKV, WebM, AVI, and M4V work through FFprobe/FFmpeg and the installed
OpenCV decoder. Actual support depends on codecs available in the FFmpeg/OpenCV
build. Files are not transcoded by default.
