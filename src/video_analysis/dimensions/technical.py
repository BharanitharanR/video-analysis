from __future__ import annotations
from pathlib import Path
from ..ffprobe import summarize

def analyze(path: str | Path) -> dict:
    return summarize(path)
