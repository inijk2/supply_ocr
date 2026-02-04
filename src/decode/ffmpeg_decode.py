from __future__ import annotations

from dataclasses import dataclass
from typing import Iterator, List, Tuple

import cv2


@dataclass
class DecodeConfig:
    fps: float = 2.0
    start_sec: float = 0.0
    end_sec: float = 420.0


def iter_frames(video_path: str, cfg: DecodeConfig) -> Iterator[Tuple[float, any]]:
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Failed to open video: {video_path}")
    t = cfg.start_sec
    step = 1.0 / cfg.fps
    while t <= cfg.end_sec:
        cap.set(cv2.CAP_PROP_POS_MSEC, t * 1000.0)
        ok, frame = cap.read()
        if not ok:
            break
        yield t, frame
        t += step
    cap.release()


def sample_frames(
    video_path: str,
    center_t: float,
    window_sec: float,
    count: int,
) -> List[Tuple[float, any]]:
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Failed to open video: {video_path}")
    half = window_sec / 2.0
    if count <= 1:
        times = [center_t]
    else:
        step = window_sec / (count - 1)
        times = [center_t - half + i * step for i in range(count)]
    frames = []
    for t in times:
        cap.set(cv2.CAP_PROP_POS_MSEC, max(0.0, t) * 1000.0)
        ok, frame = cap.read()
        if ok:
            frames.append((t, frame))
    cap.release()
    return frames
