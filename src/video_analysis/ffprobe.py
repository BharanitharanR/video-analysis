"""FFprobe helpers for deterministic media metadata."""
from __future__ import annotations
import json, shutil, subprocess
from pathlib import Path
from typing import Any

class FFProbeError(RuntimeError): pass

def probe(path: str | Path) -> dict[str, Any]:
    if shutil.which("ffprobe") is None:
        raise FFProbeError("ffprobe is required. Install FFmpeg and ensure ffprobe is on PATH.")
    cmd = ["ffprobe","-v","error","-show_format","-show_streams","-of","json",str(path)]
    p = subprocess.run(cmd, capture_output=True, text=True)
    if p.returncode:
        raise FFProbeError(p.stderr.strip() or "ffprobe failed")
    return json.loads(p.stdout)

def _num(value):
    try: return float(value)
    except (TypeError, ValueError): return None

def summarize(path: str | Path) -> dict[str, Any]:
    raw = probe(path)
    fmt = raw.get("format", {})
    streams = raw.get("streams", [])
    video = next((s for s in streams if s.get("codec_type") == "video"), None)
    audios = [s for s in streams if s.get("codec_type") == "audio"]
    out: dict[str, Any] = {
        "duration_seconds": _num(fmt.get("duration")),
        "container": fmt.get("format_name"),
        "size_bytes": int(fmt["size"]) if fmt.get("size") else None,
    }
    if video:
        w, h = video.get("width"), video.get("height")
        fps = None
        if video.get("avg_frame_rate") and video["avg_frame_rate"] != "0/0":
            a,b = video["avg_frame_rate"].split("/")
            fps = float(a)/float(b)
        out["video"] = {
            "width": w, "height": h,
            "resolution": f"{w}x{h}" if w and h else None,
            "aspect_ratio": (w/h) if w and h else None,
            "orientation": "landscape" if w and h and w > h else "portrait" if w and h and h > w else "square" if w and h else "unknown",
            "fps": fps,
            "codec": video.get("codec_name"),
            "pixel_format": video.get("pix_fmt"),
            "bitrate": int(video["bit_rate"]) if video.get("bit_rate") else None,
        }
    out["audio"] = {
        "present": bool(audios),
        "streams": [{
            "index": s.get("index"),
            "codec": s.get("codec_name"),
            "sample_rate": s.get("sample_rate"),
            "channels": s.get("channels"),
            "channel_layout": s.get("channel_layout"),
            "duration_seconds": _num(s.get("duration")),
        } for s in audios],
    }
    return out
