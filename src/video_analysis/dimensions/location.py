"""Extract location metadata without reverse geocoding."""
from __future__ import annotations
import json, shutil, subprocess
from pathlib import Path

def analyze(path: str | Path) -> dict:
    if shutil.which("ffprobe") is None:
        return {"status":"unavailable","source":"metadata","confidence":0.0}
    cmd=["ffprobe","-v","quiet","-show_entries","format_tags","-of","json",str(path)]
    p=subprocess.run(cmd,capture_output=True,text=True)
    if p.returncode: return {"status":"not_found","source":"metadata","confidence":0.0}
    tags=json.loads(p.stdout).get("format",{}).get("tags",{}) or {}
    relevant={k:v for k,v in tags.items() if "location" in k.lower() or "gps" in k.lower()}
    iso=tags.get("com.apple.quicktime.location.ISO6709") or tags.get("location")
    if not iso:
        return {"status":"not_found","source":"metadata","metadata_keys":list(relevant),"confidence":0.0}
    return {"status":"found","source":"metadata","raw_location":iso,"metadata_keys":list(relevant),"confidence":1.0}
