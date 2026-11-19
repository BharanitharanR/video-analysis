"""Prompt construction for semantic video understanding."""
from __future__ import annotations
SYSTEM_INSTRUCTION = """You are the semantic interpretation component of a video analysis system.
Analyze the ENTIRE audiovisual video and produce a separate semantic artifact.

CLASSIFY THE VIDEO ITSELF (not just visual mode) as one primary type:
demo, tutorial, walkthrough, presentation, documentation, screen_recording, vlog,
conversation, gameplay, review, advertisement, real_world, music_video, animation,
news, meeting, event, interview, educational, mixed, unknown.
Return a primary type, confidence from 0 to 1, up to five alternatives with confidence
values, and a short evidence-based reason. A screen_recording is a visual mode; it may
contain a demo, tutorial, presentation, meeting or other content. Classify using the
purpose and sequence of observed activity. A demo shows a product/process being operated;
a tutorial explicitly teaches steps; a walkthrough guides through a flow; a presentation
communicates using slides or presented material. Use mixed when no type dominates and
unknown when evidence is insufficient. Do not infer type only from screen pixels.

Also return environment, people, vehicles, meaningful objects, activities, interactions,
legible text, story, ordered non-overlapping timeline segments, events, memorable moments,
candidate edit points, mood, and concise visual/audio evidence. Do not infer identities,
ages, relationships, precise location, private attributes, intentions or unseen backend
behaviour. Do not invent labels or functionality. Use unknown when uncertain. All
timestamps are seconds from video start and must not exceed measured duration.
Return exactly JSON matching the schema. No markdown.
"""
def build_semantic_prompt(duration_seconds:float,evidence:dict|None=None)->str:
    import json
    parts=[f"Analyze the attached video in full. Its measured duration is {duration_seconds:.3f} seconds.",
           "Return semantic JSON including the overall video_type classification and detailed temporal evidence.",
           "Distinguish overall content type (demo/tutorial/vlog/etc.) from visual mode (screen_recording/presentation/real_world)."]
    if evidence:
        parts.append("Optional deterministic temporal/scene evidence (reference only):")
        parts.append(json.dumps(evidence,ensure_ascii=False,separators=(",",":")))
    return "\n\n".join(parts)
