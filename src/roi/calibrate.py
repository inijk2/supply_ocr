from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Tuple

import cv2


def _match_template(img, template) -> Tuple[int, int]:
    res = cv2.matchTemplate(img, template, cv2.TM_CCOEFF_NORMED)
    _, _, _, max_loc = cv2.minMaxLoc(res)
    return max_loc


def _read_frame(video_path: str, t_sec: float):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Failed to open video: {video_path}")
    cap.set(cv2.CAP_PROP_POS_MSEC, t_sec * 1000.0)
    ok, frame = cap.read()
    cap.release()
    if not ok:
        raise RuntimeError("Failed to read frame at time")
    return frame


def _select_roi(frame, title: str):
    roi = cv2.selectROI(title, frame, fromCenter=False, showCrosshair=True)
    cv2.destroyWindow(title)
    x, y, w, h = [int(v) for v in roi]
    return x, y, w, h


def main() -> None:
    parser = argparse.ArgumentParser(description="Calibrate ROI profile for 480p.")
    parser.add_argument("video", help="Input video path")
    parser.add_argument("--time", type=float, default=10.0, help="Time (sec) to sample frame")
    parser.add_argument("--out", default=str(Path(__file__).resolve().parent / "profile_480p.json"))
    parser.add_argument("--supply-template", default=str(Path(__file__).resolve().parents[2] / "a" / "supply_frame.png"))
    args = parser.parse_args()

    frame = _read_frame(args.video, args.time)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    tpl = cv2.imread(args.supply_template, cv2.IMREAD_GRAYSCALE)
    if tpl is None:
        raise RuntimeError("Supply template not found")

    sx, sy = _match_template(gray, tpl)
    sh, sw = tpl.shape[:2]

    cv2.namedWindow("selection_panel", cv2.WINDOW_NORMAL)
    cv2.namedWindow("production_queue", cv2.WINDOW_NORMAL)
    sel = _select_roi(frame, "selection_panel")
    queue = _select_roi(frame, "production_queue")
    cv2.destroyAllWindows()

    profile = {
        "resolution": [int(frame.shape[1]), int(frame.shape[0])],
        "rois": {
            "supply": {
                "mode": "static",
                "x": int(sx),
                "y": int(sy),
                "w": int(sw),
                "h": int(sh),
                "enabled": True,
            },
            "selection_panel": {
                "mode": "static",
                "x": int(sel[0]),
                "y": int(sel[1]),
                "w": int(sel[2]),
                "h": int(sel[3]),
                "enabled": True,
            },
            "production_queue": {
                "mode": "static",
                "x": int(queue[0]),
                "y": int(queue[1]),
                "w": int(queue[2]),
                "h": int(queue[3]),
                "enabled": True,
            },
        },
    }

    Path(args.out).write_text(json.dumps(profile, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
