"""Simple deterministic frame-quality sampling."""
from __future__ import annotations
import cv2, numpy as np
from pathlib import Path

def _sample(cap, t):
    cap.set(cv2.CAP_PROP_POS_MSEC, t*1000)
    ok, frame = cap.read()
    if not ok: return None
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    return {
        "time_seconds": round(t, 3),
        "sharpness": float(cv2.Laplacian(gray, cv2.CV_64F).var()),
        "brightness": float(gray.mean()),
    }, gray

def analyze(path: str | Path, interval_seconds: float = 1.0) -> dict:
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened(): return {"status":"error","error":"Unable to open video"}
    fps = cap.get(cv2.CAP_PROP_FPS) or 0
    n = cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0
    duration = n/fps if fps else 0
    times = np.arange(0, duration, interval_seconds).tolist()
    if duration and (not times or times[-1] < duration-0.5): times.append(max(0,duration-0.01))
    samples=[]; prev=None
    for t in times:
        item = _sample(cap,float(t))
        if not item: continue
        data, gray = item
        if prev is None: data["motion"]=0.0
        else:
            data["motion"]=float(cv2.absdiff(prev,gray).mean())
        samples.append(data); prev=gray
    cap.release()
    def stats(key):
        vals=[x[key] for x in samples]
        return {"min":min(vals),"median":float(np.median(vals)),"max":max(vals),"mean":float(np.mean(vals))} if vals else {}
    return {
        "sampling":{"interval_seconds":interval_seconds,"sample_count":len(samples)},
        "sharpness":stats("sharpness"), "brightness":stats("brightness"), "motion":stats("motion"),
        "samples":samples,
    }
