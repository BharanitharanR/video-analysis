"""Deterministic audio-stream metadata."""
from __future__ import annotations
from pathlib import Path
from ..ffprobe import summarize

def analyze(path: str | Path) -> dict:
    audio = summarize(path).get("audio", {"present":False,"streams":[]})
    audio["speech"]={"status":"not_analyzed"}
    audio["music"]={"status":"not_analyzed"}
    return audio
