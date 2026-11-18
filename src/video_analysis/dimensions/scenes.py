"""Optional PySceneDetect scene detection."""
from __future__ import annotations
from pathlib import Path

def analyze(path: str | Path) -> dict:
    try:
        from scenedetect import open_video, SceneManager
        from scenedetect.detectors import AdaptiveDetector
    except ImportError:
        return {"status":"not_installed","detector":"AdaptiveDetector","scenes":[]}
    try:
        video=open_video(str(path))
        manager=SceneManager()
        manager.add_detector(AdaptiveDetector())
        manager.detect_scenes(video=video)
        scenes=manager.get_scene_list()
        return {"status":"ok","detector":"AdaptiveDetector","scenes":[
            {"start_seconds":s[0].get_seconds(),"end_seconds":s[1].get_seconds()}
            for s in scenes
        ]}
    except Exception as exc:
        return {"status":"error","detector":"AdaptiveDetector","scenes":[],"error":str(exc)}
