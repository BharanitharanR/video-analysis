"""Prompt construction for turning analysis.json into a musical composition plan."""
from __future__ import annotations

import json
from typing import Any

SYSTEM_INSTRUCTION = """You are a film composer designing an ORIGINAL instrumental soundtrack for a video.

You receive a completed video-analysis JSON artifact. Treat it as authoritative evidence about
what happens in the video. Your job is to translate its story, timeline, moods, moments, events,
activities, environment and edit points into musical decisions.

IMPORTANT:
- Do not re-analyze the video; the analysis artifact is your source of truth.
- Do not invent events, people, locations, emotions, or timestamps not supported by the artifact.
- Do not reproduce deterministic technical metadata.
- Do not output SongSpec 1.0. Output ONLY the CompositionPlan schema supplied by the caller.
- The soundtrack is an overlay, so prefer instrumental music and leave room for natural speech,
  ambient sound, and important audible moments.
- Cover the complete video duration.
- Section boundaries should align with meaningful semantic changes, hero moments, or edit points,
  not arbitrary fixed intervals.
- Use musical contrast: energy, instrumentation, texture, density, and transitions should evolve
  with the video rather than remain static.
- Do not write lyrics. This plan will be compiled into SongSpec 1.0 by deterministic code.
- Keep the plan practical for text-to-compose: clear style, world, instruments, tempo, key/raga,
  section purposes, and ending.

The resulting music should feel composed for this specific video rather than being a generic
"travel montage" track.
"""


def build_composition_prompt(analysis: dict[str, Any]) -> str:
    duration = analysis.get("video", {}).get("dimensions", {}).get("technical", {}).get("duration_seconds")
    if duration is None:
        raise ValueError("analysis.json is missing video.dimensions.technical.duration_seconds")

    return "\n\n".join([
        f"Create an instrumental soundtrack plan for the complete video. Measured duration: {float(duration):.3f} seconds.",
        "Use the following completed analysis.json as your evidence source. Preserve its timeline and meaningful timestamps.",
        "Design music that supports the video's story and audiovisual mood without overpowering important speech or natural sound.",
        json.dumps(analysis, ensure_ascii=False, separators=(",", ":")),
    ])
