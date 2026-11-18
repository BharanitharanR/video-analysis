---
name: video-analysis
description: Analyze video files using deterministic media measurements and optional AI semantic interpretation. Returns structured temporal evidence and a separate semantic JSON artifact.
---

# Video Analysis Skill

## Purpose

Analyze video without coupling the engine to a specific agent harness. The skill
can be invoked from a CLI, Python application, or external agent.

## Analysis contract

The deterministic analyzer owns its report. An optional multimodal provider
returns a separate semantic JSON document. The merger adds that document under
`video.dimensions.semantic`; the provider never edits deterministic fields.

## Workflow

1. Inspect the input with FFprobe.
2. Run enabled deterministic dimensions (technical, audio metadata, location
   metadata, quality, scenes, and temporal evidence).
3. Optionally send the video to a configured semantic provider.
4. Validate the provider response with the semantic Pydantic contract.
5. Merge semantic output under its own namespace.
6. Return or persist the final JSON.

## Semantic interpretation rules

- Distinguish mode, activity, and mood.
- Do not infer mood from motion metrics alone.
- Ground observations in visible/audible evidence.
- Use timestamps in seconds and ordered non-overlapping intervals.
- Use unknown when evidence is insufficient.
- Treat mood as apparent audiovisual tone, not verified internal emotional state.

## Privacy

Do not log API keys. Do not upload videos to a cloud provider unless the caller
explicitly enables the AI dimension. Uploaded provider-side files are deleted
best-effort after analysis.
