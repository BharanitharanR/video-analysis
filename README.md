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

## Background Gemini inference and webhooks

Semantic inference is submitted as a background Gemini Interaction rather than keeping the initial HTTP request open for the entire video analysis. The provider polls the stored interaction until completion, which avoids the server-side request deadline that can affect long video inference.

A webhook can also be configured for completion/failure notifications:

```bash
export VIDEO_ANALYSIS_GEMINI_WEBHOOK_URL="https://example.com/gemini-webhook"
```

When configured, the provider sends this URL in Gemini's dynamic `webhook_config` along with metadata containing the local video path and analysis type. Gemini requires `background=True` for webhook-configured interactions.

The current synchronous `analyze()` API still waits for the completed interaction so existing callers receive a `SemanticAnalysis` object. The webhook is an additional notification channel and can be used by a future job-oriented API.

## Semantic analysis

```python
from video_analysis.semantic.gemini import GeminiSemanticProvider

provider = GeminiSemanticProvider(model="gemini-2.5-flash")
semantic = provider.analyze("IMG_0771.mov", duration_seconds=9.568333)
semantic_json = semantic.model_dump(mode="json")
```

The semantic artifact is intentionally separate from the deterministic report.

## Merge artifacts

Deterministic and semantic analysis can also be persisted as independent artifacts and merged later. This keeps the two contracts separate and makes the merge reproducible without rerunning Gemini.

```bash
video-analysis merge deterministic.json semantic.json -o analysis.json
```

The command validates `semantic.json` against the semantic contract, preserves all deterministic dimensions, and writes the semantic artifact under `video.dimensions.semantic`.

Python API:

```python
from video_analysis.merge import merge_analysis_artifacts, write_merged_analysis

final = merge_analysis_artifacts("deterministic.json", "semantic.json")
write_merged_analysis("deterministic.json", "semantic.json", "analysis.json")
```

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
## Video-to-music composition

The completed `analysis.json` can be turned into a provider-neutral SongSpec 1.0
for the existing **text-to-compose** skill.

The architecture deliberately uses two stages:

1. Gemini receives the completed analysis artifact and creates a `CompositionPlan`.
2. Local deterministic code compiles that plan into SongSpec 1.0, including exact
   seconds-to-beats conversion.

This keeps creative decisions in the LLM while keeping schema compliance and timing
arithmetic deterministic.

```text
analysis.json
     |
     v
Gemini CompositionPlan
     |
     v
deterministic compiler
     |
     v
song-spec.json (SongSpec 1.0)
     |
     v
text-to-compose
     |
     v
ACE-Step / Eleven Music / Suno
```

Run:

```bash
video-analysis compose analysis.json -o song-spec.json
```

Use a different composition model:

```bash
video-analysis compose analysis.json \
  --gemini-model gemini-3.6-flash \
  -o song-spec.json
```

The generated SongSpec is instrumental by default because it is intended as an
overlay soundtrack. The composition planner uses the semantic timeline, story,
moods, moments, events, edit points and environment from `analysis.json` to make
the music evolve with the video.

The generated `song-spec.json` is intentionally compatible with the existing
text-to-compose contract rather than introducing another audio-generation system.
## Model fallback

Both Gemini steps (`analyze --semantic gemini` and `compose`) try the requested
`--gemini-model` first and, if Gemini answers with a quota, rate-limit or server
error (`429`, `RESOURCE_EXHAUSTED`, `INTERNAL`, `UNAVAILABLE`, ...), move to the
next model in a chain of free-tier models. Bad input, a response that fails
validation, and timeouts are not retried. The model that answered is printed to
stderr as `Gemini model used: <model>`.

| Variable | Effect |
|---|---|
| `VIDEO_ANALYSIS_SEMANTIC_MODELS` | comma-separated fallbacks for video analysis |
| `VIDEO_ANALYSIS_COMPOSITION_MODELS` | comma-separated fallbacks for composition |
| `VIDEO_ANALYSIS_MAX_MODEL_ATTEMPTS` | most models tried per step (default 3; each try is a paid call) |
| `VIDEO_ANALYSIS_MODEL_FALLBACK=0` | use only the requested model |

Default chains (free-tier models with quota, 2026-11-20):

- analysis: gemini-4-hydrogen, gemini-3.8-flash, gemini-3.6-flash, gemini-2.5-flash, gemini-3.5-flash-lite, gemini-3.1-flash-lite
- composition: gemini-3.6-flash, gemini-3.8-flash, gemini-4-hydrogen, gemini-2.5-flash, gemini-3.5-flash-lite, gemini-3.1-flash-lite, gemini-2.5-flash-lite
