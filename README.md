# video-analysis

A standalone, dimension-based video analysis engine. It separates deterministic
media measurements from AI-generated semantic interpretation.

## Architecture

The skill has two independent analysis paths:

1. **Deterministic video analysis** — FFprobe, OpenCV and optional PySceneDetect
   measure the media and produce technical, audio, location, quality, scenes and
   temporal dimensions.
2. **Semantic AI analysis** — an optional Gemini provider watches the video and
   returns a separate semantic JSON artifact containing mode, activity, mood,
   observations and events.

The AI artifact is validated independently and then merged under
`video.dimensions.semantic`. AI never overwrites deterministic dimensions.

## Install

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -e '.[scenes,gemini,dev]'
```

FFprobe must be available on PATH for deterministic media inspection.

## Deterministic analysis

```bash
video-analysis analyze IMG_0771.mov -o analysis.json
```

The deterministic analyzer supports MOV, MP4, MKV, WebM, AVI and M4V containers
subject to codecs supported by the installed FFmpeg/OpenCV builds.

## End-to-end pipeline

The pipeline runs deterministic analysis first, then optionally invokes a semantic provider with the measured duration and read-only temporal/scene evidence. The final report keeps deterministic dimensions intact and adds the validated semantic artifact under `video.dimensions.semantic`.

```bash
video-analysis analyze IMG_0771.mov -o analysis.json --semantic gemini
```

Use a different Gemini model when needed:

```bash
video-analysis analyze IMG_0771.mov --semantic gemini --gemini-model gemini-4-hydrogen
```

Without `--semantic`, the CLI remains deterministic-only.

## Large video transport

Gemini video input uses two transports automatically:

- Videos up to 20 MiB are sent inline.
- Larger videos are uploaded through the Gemini Files API, polled until the file is `ACTIVE`, and then referenced by URI.

This means normal phone-camera videos can be analyzed without the caller first transcoding them just to satisfy the inline request limit.

## Semantic analysis

```python
from video_analysis.semantic.gemini import GeminiSemanticProvider

provider = GeminiSemanticProvider(model="gemini-2.5-flash")
semantic = provider.analyze("IMG_0771.mov", duration_seconds=9.568333)
semantic_json = semantic.model_dump(mode="json")
```

The semantic artifact is intentionally separate from the deterministic report.

## Merge

```python
from video_analysis.semantic.merge import merge_semantic_analysis
final = merge_semantic_analysis(deterministic_json, semantic)
```

For applications that want the complete orchestration:

```python
from video_analysis.pipeline import VideoAnalysisPipeline
from video_analysis.semantic.gemini import GeminiSemanticProvider

pipeline = VideoAnalysisPipeline(
    semantic_provider=GeminiSemanticProvider(model="gemini-4-hydrogen"),
)
final = pipeline.analyze("IMG_0771.mov")
```

## Development

```bash
pytest
```

Set `GEMINI_API_KEY` only in the environment. Never commit API keys.
