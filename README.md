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
`video.dimensions.semantic`. It also classifies overall content type at
`video.dimensions.semantic.video_type`, copied to `video.dimensions.video_type`
as `{primary, confidence, alternatives:[{type, confidence}], reason}`. A missing
classification reads as primary `unknown`, confidence 0. Types include (demo, tutorial, walkthrough, presentation, vlog,
gameplay, meeting, review, real-world footage, and more). This is distinct from
visual mode in individual timeline segments, such as `screen_recording` or
`real_world`. AI never overwrites deterministic dimensions.

For demo, tutorial, walkthrough and screen-recording content the same semantic
call also fills `video.dimensions.semantic.screen_actions`, a list of
`{timestamp_seconds, action, target, visible_text, result}` (strings may be
empty). For other video types it stays empty.

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

## What Gemini is sent: a deterministic 720p copy

Before the semantic call, the original video is copied with fixed FFmpeg
settings (`src/video_analysis/gemini_input.py`): the shorter side down to 720
pixels (never upscaled, orientation kept), H.264 CRF 28 preset medium, AAC 96k
mono 48 kHz, frame rate unchanged, all metadata stripped with the bitexact
flags. Gemini receives `<name>.gemini-input.mp4`; the original is never
modified, and the deterministic measurements still use the original.

The same input and the same FFmpeg build give a byte-identical copy. The
duration must match the original within 0.25 s, so Gemini's timestamps apply
to the original timeline. `analysis.json` records what was sent under
`video.gemini_input`: file name, size, SHA-256 of the copy and of the original,
FFmpeg version and settings. A 173 s, 100 MB 1080p clip became a 5 MB copy in
about 10 s on this Mac.

`VIDEO_ANALYSIS_GEMINI_COPY=0` sends the original instead.

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
## Narration bed

When a voiceover will play over the score, pass its timeline so the music sits
underneath it:

```bash
video-analysis compose analysis.json --voice-timeline voiceover-timeline.json -o song-spec.json
```

The timeline's lines are merged into narration windows (start, end, speaker) and the
speaking share of the video, and both go into the composition prompt. Gemini is asked
for a sparse, low-energy bed with nothing prominent in roughly 200 Hz to 4 kHz during
speech, lifting slightly in gaps and at the intro and outro. After Gemini answers, code
sets `narration_bed: true` and caps section energy at 0.45 for sections that overlap
speech and 0.60 for the others. The SongSpec gets `extra.narration_bed: true` and a
`narration-bed` tag. Without `--voice-timeline` nothing changes.

## Model order: a config file, not code

The order Gemini models are tried in is set in one config file, shared by the
Video Score Studio **Models** window, this command line and the MCP wrapper.
The order of each list is the order of tries:

```json
{
  "analysis":    ["gemini-3.8-flash", "gemini-3.6-flash", "gemini-2.5-flash", "gemini-4-hydrogen", "gemini-3.5-flash-lite"],
  "composition": ["gemini-3.8-flash", "gemini-3.6-flash", "gemini-2.5-flash", "gemini-4-hydrogen", "gemini-3.5-flash-lite"],
  "disabled":    {"analysis": [], "composition": []},
  "max_attempts": 6
}
```

- Location: `~/.config/video-score/gemini-models.json`, or the path in
  `VIDEO_ANALYSIS_MODELS_CONFIG`. It is outside the repos, so pulls never
  overwrite it and it is never committed.
- `analysis` is the video analysis step, `composition` the music plan.
- `disabled` keeps models switched off in the studio window; they are never
  tried. Every field is optional; a missing list uses the built-in default.
- It is read fresh at the start of every Gemini step, so an edit applies to the
  next run without a restart.
- A broken file (not JSON, a bad model name, every model switched off, a cap
  outside 1 to 50) stops the step before any Gemini call, with the reason.

Without the studio:

```bash
video-analysis models                  # the order each step will use, and where it comes from
video-analysis models --write-defaults # create the file with the built-in order (never overwrites)
video-analysis models --json           # the same report as JSON
echo '{"analysis": ["gemini-3.6-flash", "gemini-3.8-flash"]}' | video-analysis models --save -
```

Which setting wins, per step: `--gemini-model` (moved to the front), then
`VIDEO_ANALYSIS_SEMANTIC_MODELS` / `VIDEO_ANALYSIS_COMPOSITION_MODELS`, then
the config file, then the built-in defaults. Every run prints the order it
uses, for example
`Gemini model order for Video analysis (config file ...): gemini-3.8-flash, ...`.

Built-in defaults (when nothing else sets an order), with `gemini-3.8-flash`
as the baseline:

- analysis: gemini-3.8-flash, gemini-3.6-flash, gemini-2.5-flash, gemini-4-hydrogen, gemini-3.5-flash-lite, gemini-3.1-flash-lite
- composition: gemini-3.8-flash, gemini-3.6-flash, gemini-2.5-flash, gemini-4-hydrogen, gemini-3.5-flash-lite, gemini-3.1-flash-lite, gemini-2.5-flash-lite

## Fallback across models and API keys

Keys come from `GEMINI_API_KEY_1`, `GEMINI_API_KEY_2`, ... in that order
(repeats are ignored); when none is set, `GEMINI_API_KEY` alone. Each key gets
its own client with the key passed explicitly, so a `GOOGLE_API_KEY` in the
environment is never used instead. A second key only helps with quota when it
belongs to a different Google Cloud project, since quota is counted per
project.

The order of attempts:

1. The first model on key 1.
2. A quota or rate-limit error (`429`, `RESOURCE_EXHAUSTED`) tries the **same
   model on the next key**.
3. When every key is out of quota for that model, or the model fails in a way
   another key cannot fix (a server error such as `INTERNAL` or
   `UNAVAILABLE`, `DEADLINE_EXCEEDED`, or an answer that is empty, not JSON,
   or fails local validation), the **next model** is tried, starting again
   from key 1.
4. At most `max_attempts` attempts (default **6**) are made in all, across
   models and keys. Every attempt may be a paid call.

Bad input and our own wait running out (`TimeoutError`) stop the step at once.

Large videos sent through the Files API are uploaded once per key actually
used, since an upload belongs to its key's project. Inline videos (up to
20 MiB, such as the 720p copy) need no upload.

Logs name keys by number, never by value:

```
Gemini model gemini-3.8-flash (key 1/4) out of quota (...); trying key 2/4
Gemini model gemini-3.8-flash (key 4/4) is out of quota on every key (...); trying gemini-3.6-flash from key 1
Gemini model used: gemini-3.6-flash (key 1/4)
```

A rejected answer is logged as `... gave an invalid answer (...)` and kept
beside the raw response as `<raw>.<model>.rejected.json`.

| Variable | Effect |
|---|---|
| `VIDEO_ANALYSIS_MODELS_CONFIG` | path of the model config file |
| `GEMINI_API_KEY_1` .. `GEMINI_API_KEY_<n>` | keys to rotate through, in order |
| `GEMINI_API_KEY` | the single key, used when no numbered key is set |
| `VIDEO_ANALYSIS_SEMANTIC_MODELS` | comma-separated model order for video analysis (beats the file) |
| `VIDEO_ANALYSIS_COMPOSITION_MODELS` | comma-separated model order for composition (beats the file) |
| `VIDEO_ANALYSIS_MAX_MODEL_ATTEMPTS` | most attempts per step across models and keys (beats the file) |
| `VIDEO_ANALYSIS_MODEL_FALLBACK=0` | use only the requested model, or the first of the order |
