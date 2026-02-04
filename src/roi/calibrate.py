from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Tuple, Optional

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


def _parse_rect(s: Optional[str]) -> Optional[Tuple[int, int, int, int]]:
    if not s:
        return None
    parts = [p.strip() for p in s.split(",")]
    if len(parts) != 4:
        raise ValueError("ROI must be x,y,w,h")
    return tuple(int(p) for p in parts)  # type: ignore


def main() -> None:
    parser = argparse.ArgumentParser(description="Calibrate ROI profile for 480p.")
    parser.add_argument("video", help="Input video path")
    parser.add_argument("--time", type=float, default=10.0, help="Time (sec) to sample frame")
    parser.add_argument("--out", default=str(Path(__file__).resolve().parent / "profile_480p.json"))
    parser.add_argument("--supply-template", default=str(Path(__file__).resolve().parents[2] / "a" / "supply_frame.png"))
    parser.add_argument("--sel", help="Selection ROI as x,y,w,h (skip GUI)")
    parser.add_argument("--queue", help="Queue ROI as x,y,w,h (skip GUI)")
    parser.add_argument("--dump-frame", help="Save the sampled frame to this path")
    args = parser.parse_args()

    frame = _read_frame(args.video, args.time)
    if args.dump_frame:
        Path(args.dump_frame).parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(args.dump_frame, frame)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    tpl = cv2.imread(args.supply_template, cv2.IMREAD_GRAYSCALE)
    if tpl is None:
        raise RuntimeError("Supply template not found")

    sx, sy = _match_template(gray, tpl)
    sh, sw = tpl.shape[:2]

    sel = _parse_rect(args.sel)
    queue = _parse_rect(args.queue)

    if sel is None or queue is None:
        cv2.namedWindow("selection_panel", cv2.WINDOW_NORMAL)
        cv2.namedWindow("production_queue", cv2.WINDOW_NORMAL)
        if sel is None:
            sel = _select_roi(frame, "selection_panel")
        if queue is None:
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
                "template_min_conf": 0.8,
                "enabled": True,
            },
            "selection_panel": {
                "mode": "static",
                "x": int(sel[0]),  # type: ignore[index]
                "y": int(sel[1]),  # type: ignore[index]
                "w": int(sel[2]),  # type: ignore[index]
                "h": int(sel[3]),  # type: ignore[index]
                "enabled": True,
            },
            "production_queue": {
                "mode": "static",
                "x": int(queue[0]),  # type: ignore[index]
                "y": int(queue[1]),  # type: ignore[index]
                "w": int(queue[2]),  # type: ignore[index]
                "h": int(queue[3]),  # type: ignore[index]
                "enabled": True,
            },
        },
    }

    Path(args.out).write_text(json.dumps(profile, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
