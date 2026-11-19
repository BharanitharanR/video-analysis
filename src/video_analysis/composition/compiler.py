"""Deterministic compiler from CompositionPlan to SongSpec 1.0."""
from __future__ import annotations

from typing import Any

from .models import CompositionPlan


def _seconds_to_beats(seconds: float, bpm: float) -> int:
    return max(0, round(seconds * bpm / 60.0))


def compile_song_spec(plan: CompositionPlan | dict[str, Any]) -> dict[str, Any]:
    if not isinstance(plan, CompositionPlan):
        plan = CompositionPlan.model_validate(plan)

    sections = []
    for section in plan.sections:
        # energy, mood, texture and transition are not named in the SongSpec
        # 1.0 schema, but arrangement sections allow extra fields, and the
        # text-to-compose ACE-Step adapter turns them into per-section tags.
        # Without them an instrumental score reaches the provider as one flat cue.
        sections.append({
            "name": section.name or section.purpose,
            "start_beat": _seconds_to_beats(section.start_seconds, plan.tempo_bpm),
            "end_beat": _seconds_to_beats(section.end_seconds, plan.tempo_bpm),
            "purpose": section.purpose,
            "instruments": section.instruments or plan.instruments,
            "energy": section.energy,
            "mood": section.mood,
            "texture": section.texture,
            "transition": section.transition,
        })

    return {
        "song_spec_version": "1.0",
        "metadata": {
            "title": plan.title,
            "language": plan.language,
            "duration_seconds": plan.duration_seconds,
            "tags": ["instrumental", "video-score"],
        },
        "creative": {
            "world": plan.world,
            "style": plan.style,
            "tempo": round(plan.tempo_bpm),
            "time_signature": plan.time_signature,
            "key_or_raga": plan.key_or_raga,
            "instruments": plan.instruments,
            "vocals": {
                "required": False,
                "singer": "",
                "narrator": "",
                "language": "Instrumental",
                "range": "",
                "technique": "",
            },
            "ending": plan.ending,
        },
        "lyrics": {
            "approved": True,
            "sections": [],
        },
        "arrangement": {
            "motif": plan.motif,
            "sections": sections,
        },
        "providers": {
            "preferred": "ace-step",
            "requested_capabilities": ["instrumental", "full_duration", "original_music"],
        },
    }
