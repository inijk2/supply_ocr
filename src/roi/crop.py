from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Tuple

import cv2
import numpy as np


@dataclass
class ROIDefinition:
    mode: str
    x: int = 0
    y: int = 0
    w: int = 0
    h: int = 0
    template: str | None = None
    template_min_conf: float = 0.8
    padding: Tuple[int, int, int, int] = (0, 0, 0, 0)
    enabled: bool = True


def _load_profile(path: Path) -> Dict[str, ROIDefinition]:
    data = json.loads(path.read_text(encoding="utf-8"))
    rois = {}
    for key, val in data.get("rois", {}).items():
        rois[key] = ROIDefinition(
            mode=val.get("mode", "static"),
            x=val.get("x", 0),
            y=val.get("y", 0),
            w=val.get("w", 0),
            h=val.get("h", 0),
            template=val.get("template"),
            template_min_conf=float(val.get("template_min_conf", 0.8)),
            padding=tuple(val.get("padding", [0, 0, 0, 0])),
            enabled=val.get("enabled", True),
        )
    return rois


def _match_template(img: np.ndarray, template: np.ndarray) -> Tuple[Tuple[int, int], float]:
    res = cv2.matchTemplate(img, template, cv2.TM_CCOEFF_NORMED)
    _, max_val, _, max_loc = cv2.minMaxLoc(res)
    return max_loc, float(max_val)


def crop_roi(frame: np.ndarray, profile_path: Path, name: str) -> np.ndarray | None:
    rois = _load_profile(profile_path)
    if name not in rois:
        return None
    roi = rois[name]
    if not roi.enabled:
        return None
    if roi.mode == "static":
        return frame[roi.y : roi.y + roi.h, roi.x : roi.x + roi.w]
    if roi.mode == "template" and roi.template:
        tpl_path = (profile_path.parent / roi.template).resolve()
        if not tpl_path.exists():
            return None
        tpl = cv2.imread(str(tpl_path), cv2.IMREAD_GRAYSCALE)
        gray = frame if len(frame.shape) == 2 else cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        (x, y), conf = _match_template(gray, tpl)
        if conf < roi.template_min_conf:
            return None
        h, w = tpl.shape[:2]
        pad_l, pad_t, pad_r, pad_b = roi.padding
        x0 = max(0, x - pad_l)
        y0 = max(0, y - pad_t)
        x1 = min(frame.shape[1], x + w + pad_r)
        y1 = min(frame.shape[0], y + h + pad_b)
        return frame[y0:y1, x0:x1]
    return None
