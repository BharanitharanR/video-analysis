"""Voice timeline (voiceover-timeline.json) handling for narration-bed composition."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

# Section energy caps applied in code after Gemini returns, so the bed stays
# under speech even if the model ignores the prompt.
NARRATION_ENERGY_CAP = 0.45
GAP_ENERGY_CAP = 0.60


def load_voice_timeline(source: dict | str | Path) -> dict[str, Any]:
    if isinstance(source, (str, Path)):
        source = json.loads(Path(source).read_text(encoding="utf-8"))
    if not isinstance(source, dict) or not isinstance(source.get("lines"), list):
        raise ValueError("voice timeline must be an object with a 'lines' list")
    return source


def narration_windows(timeline: dict[str, Any], merge_gap: float = 0.4) -> list[dict[str, Any]]:
    """Return sorted speech windows; lines closer than merge_gap are merged."""
    lines = sorted(
        (l for l in timeline["lines"] if l.get("end_seconds", 0) > l.get("start_seconds", 0)),
        key=lambda l: l["start_seconds"],
    )
    windows: list[dict[str, Any]] = []
    for line in lines:
        start, end = float(line["start_seconds"]), float(line["end_seconds"])
        speaker = str(line.get("speaker", ""))
        if windows and start - windows[-1]["end"] <= merge_gap:
            windows[-1]["end"] = max(windows[-1]["end"], end)
            if speaker and speaker not in windows[-1]["speakers"]:
                windows[-1]["speakers"].append(speaker)
        else:
            windows.append({"start": start, "end": end, "speakers": [speaker] if speaker else []})
    return windows


def speaking_share(windows: list[dict[str, Any]], duration: float) -> float:
    if duration <= 0:
        return 0.0
    return min(1.0, sum(w["end"] - w["start"] for w in windows) / duration)


def summarize_voice_timeline(timeline: dict[str, Any], duration: float) -> str:
    windows = narration_windows(timeline)
    share = speaking_share(windows, duration)
    rows = [
        f"- {w['start']:.1f}s to {w['end']:.1f}s ({', '.join(w['speakers']) or 'speaker'})"
        for w in windows
    ]
    return "\n".join([
        f"Spoken narration windows ({len(windows)} windows, speech covers {share * 100:.0f}% of the video):",
        *rows,
    ])


def apply_narration_bed(plan, timeline: dict[str, Any]):
    """Flag the plan as a narration bed and cap section energy in code."""
    windows = narration_windows(timeline)
    plan.narration_bed = True
    for section in plan.sections:
        under_voice = any(
            w["start"] < section.end_seconds and w["end"] > section.start_seconds for w in windows
        )
        cap = NARRATION_ENERGY_CAP if under_voice else GAP_ENERGY_CAP
        section.energy = min(section.energy, cap)
    return plan
