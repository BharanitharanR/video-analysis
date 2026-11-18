"""Temporal evidence derived from deterministic quality samples."""
from __future__ import annotations
from .quality import analyze as quality_analyze

def analyze(path, interval_seconds: float=1.0) -> dict:
    q=quality_analyze(path, interval_seconds)
    evidence=[{
        "time_seconds":x["time_seconds"],
        "sharpness":x["sharpness"],
        "brightness":x["brightness"],
        "motion":x["motion"],
    } for x in q.get("samples",[])]
    return {"status":"ok","interval_seconds":interval_seconds,"evidence":evidence,
            "note":"Measured temporal evidence; semantic interpretation is separate."}
