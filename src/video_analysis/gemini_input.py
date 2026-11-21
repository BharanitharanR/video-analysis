"""Deterministic, smaller copy of a video for Gemini.

Gemini only needs to understand the video, not keep its full quality, and a
large upload makes it slower and more likely to time out. So before the
semantic call we make a 720p copy with fixed FFmpeg settings and send that.
The original file is never modified: the deterministic analyzer still measures
the original, and the mixer still uses the original picture.

The copy is deterministic: the same input and the same FFmpeg build always
give byte-identical output (fixed encoder settings, x264's default
deterministic mode, and all container metadata and timestamps stripped via
the bitexact flags). The FFmpeg version, the settings and both SHA-256 hashes
are recorded so any run can be checked.

VIDEO_ANALYSIS_GEMINI_COPY=0 turns the copy off and sends the original.
"""
from __future__ import annotations

import hashlib
import os
import shutil
import subprocess
from pathlib import Path
from typing import Any

COPY_ENV = "VIDEO_ANALYSIS_GEMINI_COPY"
COPY_NAME_SUFFIX = ".gemini-input.mp4"

MAX_SHORT_SIDE = 720          # 1080p and above come down to 720p; smaller is never upscaled
VIDEO_CRF = 28
VIDEO_PRESET = "medium"
AUDIO_BITRATE = "96k"
AUDIO_CHANNELS = 1
AUDIO_SAMPLE_RATE = 48000
MAX_DURATION_DRIFT_SECONDS = 0.25  # Gemini timestamps must line up with the original

SETTINGS: dict[str, Any] = {
    "max_short_side": MAX_SHORT_SIDE,
    "video_codec": "libx264",
    "crf": VIDEO_CRF,
    "preset": VIDEO_PRESET,
    "pix_fmt": "yuv420p",
    "audio_codec": "aac",
    "audio_bitrate": AUDIO_BITRATE,
    "audio_channels": AUDIO_CHANNELS,
    "audio_sample_rate": AUDIO_SAMPLE_RATE,
    "frame_rate": "unchanged",
}


def copy_enabled() -> bool:
    return os.environ.get(COPY_ENV, "1").strip() != "0"


def _tool(name: str) -> str:
    path = shutil.which(name)
    if not path:
        raise RuntimeError(f"{name} executable not found on PATH")
    return path


def sha256_of(path: str | Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def ffmpeg_version() -> str:
    completed = subprocess.run([_tool("ffmpeg"), "-version"], capture_output=True, text=True)
    first = completed.stdout.splitlines()[0] if completed.stdout else ""
    return first.strip()


def probe_duration(path: str | Path) -> float:
    completed = subprocess.run(
        [_tool("ffprobe"), "-v", "error", "-show_entries", "format=duration",
         "-of", "default=nw=1:nk=1", str(path)],
        capture_output=True, text=True,
    )
    if completed.returncode != 0 or not completed.stdout.strip():
        raise RuntimeError(f"ffprobe could not read the duration of {path}: {completed.stderr.strip()}")
    return float(completed.stdout.strip())


def build_copy_command(source: str | Path, destination: str | Path) -> list[str]:
    """The exact FFmpeg command. Kept in one place so the settings, and
    therefore the output, only change when this function changes."""
    # Shorter side to at most MAX_SHORT_SIDE, aspect kept, even dimensions,
    # portrait and landscape alike, never upscaled.
    scale = (
        f"scale=w='if(gt(iw,ih),-2,min({MAX_SHORT_SIDE},iw))'"
        f":h='if(gt(iw,ih),min({MAX_SHORT_SIDE},ih),-2)'"
    )
    return [
        _tool("ffmpeg"), "-nostdin", "-hide_banner", "-loglevel", "error", "-y",
        "-i", str(source),
        "-map", "0:v:0", "-map", "0:a:0?",
        "-vf", scale,
        "-c:v", "libx264", "-preset", VIDEO_PRESET, "-crf", str(VIDEO_CRF), "-pix_fmt", "yuv420p",
        "-c:a", "aac", "-b:a", AUDIO_BITRATE, "-ac", str(AUDIO_CHANNELS), "-ar", str(AUDIO_SAMPLE_RATE),
        "-map_metadata", "-1", "-map_chapters", "-1",
        "-fflags", "+bitexact", "-flags:v", "+bitexact", "-flags:a", "+bitexact",
        "-movflags", "+faststart",
        str(destination),
    ]


def copy_path_for(source: str | Path) -> Path:
    source = Path(source)
    return source.with_name(source.stem + COPY_NAME_SUFFIX)


def make_gemini_copy(source: str | Path, destination: str | Path | None = None) -> dict[str, Any]:
    """Write the deterministic Gemini copy and return its record.

    Raises if FFmpeg fails or the copy's duration drifts from the original's,
    since Gemini's timestamps are applied to the original timeline.
    """
    source = Path(source)
    destination = Path(destination) if destination else copy_path_for(source)
    if destination.resolve() == source.resolve():
        raise ValueError("The Gemini copy must not overwrite the original video")

    source_hash_before = sha256_of(source)
    completed = subprocess.run(build_copy_command(source, destination), capture_output=True, text=True)
    if completed.returncode != 0 or not destination.is_file() or destination.stat().st_size == 0:
        raise RuntimeError("FFmpeg could not make the Gemini copy:\n" + completed.stderr.strip())
    if sha256_of(source) != source_hash_before:
        raise RuntimeError("The original video changed while making the Gemini copy")

    original_duration = probe_duration(source)
    copy_duration = probe_duration(destination)
    if abs(original_duration - copy_duration) > MAX_DURATION_DRIFT_SECONDS:
        raise RuntimeError(
            f"Gemini copy duration {copy_duration:.3f}s differs from the original "
            f"{original_duration:.3f}s by more than {MAX_DURATION_DRIFT_SECONDS}s"
        )

    return {
        "path": destination.name,
        "bytes": destination.stat().st_size,
        "sha256": sha256_of(destination),
        "duration_seconds": round(copy_duration, 3),
        "source": source.name,
        "source_bytes": source.stat().st_size,
        "source_sha256": source_hash_before,
        "ffmpeg": ffmpeg_version(),
        "settings": dict(SETTINGS),
    }
