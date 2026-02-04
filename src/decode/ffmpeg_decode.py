from __future__ import annotations

from dataclasses import dataclass
from typing import Iterator, List, Tuple, Optional

import cv2


@dataclass
class DecodeConfig:
    fps: float = 2.0
    start_sec: float = 0.0
    end_sec: float = 420.0


def open_capture(video_path: str) -> cv2.VideoCapture:
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Failed to open video: {video_path}")
    return cap


def get_frame_at(cap: cv2.VideoCapture, t: float) -> Optional[any]:
    cap.set(cv2.CAP_PROP_POS_MSEC, max(0.0, t) * 1000.0)
    ok, frame = cap.read()
    if not ok:
        return None
    return frame


def iter_frames(video_path: str, cfg: DecodeConfig) -> Iterator[Tuple[float, any]]:
    cap = open_capture(video_path)
    try:
        t = cfg.start_sec
        step = 1.0 / cfg.fps
        while t <= cfg.end_sec:
            frame = get_frame_at(cap, t)
            if frame is None:
                break
            yield t, frame
            t += step
    finally:
        cap.release()


def sample_frames(
    video_path: str,
    center_t: float,
    window_sec: float,
    count: int,
) -> List[Tuple[float, any]]:
    cap = open_capture(video_path)
    try:
        return sample_frames_with_capture(cap, center_t, window_sec, count)
    finally:
        cap.release()


def sample_frames_with_capture(
    cap: cv2.VideoCapture,
    center_t: float,
    window_sec: float,
    count: int,
) -> List[Tuple[float, any]]:
    half = window_sec / 2.0
    if count <= 1:
        times = [center_t]
    else:
        step = window_sec / (count - 1)
        times = [center_t - half + i * step for i in range(count)]
    frames = []
    for t in times:
        frame = get_frame_at(cap, t)
        if frame is not None:
            frames.append((t, frame))
    return frames
