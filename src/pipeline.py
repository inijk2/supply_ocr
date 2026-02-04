from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

import cv2

from decode.ffmpeg_decode import DecodeConfig, iter_frames, sample_frames
from detect.diff_trigger import DiffConfig, changed
from ocr.engine import OCREngine
from ocr.preprocess import sharpness_score
from ocr.read_queue import read_queue
from ocr.read_selection import read_selection
from ocr.read_supply import read_supply
from roi.crop import crop_roi


@dataclass
class PipelineConfig:
    video_path: str
    profile_path: str
    output_path: str
    start_sec: float = 0.0
    end_sec: float = 420.0
    supply_fps: float = 2.0
    supply_window_sec: float = 0.25
    supply_samples: int = 7
    roi_window_sec: float = 0.5
    roi_samples: int = 10
    diff_threshold: float = 0.03
    ocr_engine: Optional[str] = None


def _save_evidence(img, path: Path) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(path), img)
    return str(path.as_posix())


def run_pipeline(cfg: PipelineConfig) -> Dict:
    ocr = OCREngine(cfg.ocr_engine)
    profile_path = Path(cfg.profile_path)
    templates_dir = _repo_root() / "a"

    evidence_dir = Path(cfg.output_path).resolve().parent / "evidence"
    evidence_dir.mkdir(parents=True, exist_ok=True)

    supply_series = []
    selection_changes = []
    queue_events = []
    events = []
    first_supply_time = None
    ocr_stats = {
        "supply_parsed": 0,
        "supply_total": 0,
        "selection_nonempty": 0,
        "selection_total": 0,
        "queue_nonempty": 0,
        "queue_total": 0,
    }

    last_supply = None
    supply_idx = 0

    decode_cfg = DecodeConfig(fps=cfg.supply_fps, start_sec=cfg.start_sec, end_sec=cfg.end_sec)
    for t, _ in iter_frames(cfg.video_path, decode_cfg):
        candidates = sample_frames(cfg.video_path, t, cfg.supply_window_sec, cfg.supply_samples)
        best = None
        for ct, frame in candidates:
            roi = crop_roi(frame, profile_path, "supply")
            if roi is None:
                continue
            result = read_supply(roi, templates_dir, ocr)
            ocr_stats["supply_total"] += 1
            if result.used is None or result.total is None:
                continue
            ocr_stats["supply_parsed"] += 1
            if best is None or result.conf > best["conf"]:
                best = {"t": ct, "frame": frame, "roi": roi, "res": result, "conf": result.conf}
        if best is None:
            continue
        current = (best["res"].used, best["res"].total)
        if first_supply_time is None:
            first_supply_time = float(best["t"])
        if last_supply is None or current != last_supply:
            supply_idx += 1
            ev_path = evidence_dir / f"supply_{supply_idx:06d}.jpg"
            frame_path = _save_evidence(best["roi"], ev_path)
            supply_series.append(
                {
                    "t": round(float(best["t"]), 3),
                    "used": best["res"].used,
                    "total": best["res"].total,
                    "raw_text": best["res"].raw_text,
                    "conf": round(float(best["res"].conf), 3),
                    "frame": frame_path,
                }
            )
            last_supply = current

    if first_supply_time is None:
        first_supply_time = cfg.start_sec

    diff_cfg = DiffConfig(cfg.diff_threshold)
    last_sel = None
    last_queue = None
    roi_idx = 0

    decode_cfg_roi = DecodeConfig(fps=cfg.supply_fps, start_sec=first_supply_time, end_sec=cfg.end_sec)
    for t, frame in iter_frames(cfg.video_path, decode_cfg_roi):
        sel_roi = crop_roi(frame, profile_path, "selection_panel")
        if sel_roi is not None:
            if last_sel is None or changed(last_sel, sel_roi, diff_cfg):
                roi_idx += 1
                frames = sample_frames(cfg.video_path, t, cfg.roi_window_sec, cfg.roi_samples)
                best = None
                for ct, f in frames:
                    roi = crop_roi(f, profile_path, "selection_panel")
                    if roi is None:
                        continue
                    sharp = sharpness_score(roi)
                    if best is None or sharp > best["sharp"]:
                        best = {"t": ct, "roi": roi, "sharp": sharp}
                if best:
                    res = read_selection(best["roi"], ocr)
                    ev_path = evidence_dir / f"sel_{roi_idx:06d}.jpg"
                    frame_path = _save_evidence(best["roi"], ev_path)
                    ocr_stats["selection_total"] += 1
                    if res.selected_name.text or res.hp_text.text:
                        ocr_stats["selection_nonempty"] += 1
                    selection_changes.append(
                        {
                            "t": round(float(best["t"]), 3),
                            "frame": frame_path,
                            "ocr": {
                                "selected_name": {"text": res.selected_name.text, "conf": round(float(res.selected_name.conf), 3)},
                                "hp_text": {"text": res.hp_text.text, "conf": round(float(res.hp_text.conf), 3)},
                            },
                        }
                    )
                last_sel = sel_roi

        queue_roi = crop_roi(frame, profile_path, "production_queue")
        if queue_roi is not None:
            if last_queue is None or changed(last_queue, queue_roi, diff_cfg):
                roi_idx += 1
                frames = sample_frames(cfg.video_path, t, cfg.roi_window_sec, cfg.roi_samples)
                best = None
                for ct, f in frames:
                    roi = crop_roi(f, profile_path, "production_queue")
                    if roi is None:
                        continue
                    sharp = sharpness_score(roi)
                    if best is None or sharp > best["sharp"]:
                        best = {"t": ct, "roi": roi, "sharp": sharp}
                if best:
                    res = read_queue(best["roi"], ocr)
                    ev_path = evidence_dir / f"q_{roi_idx:06d}.jpg"
                    frame_path = _save_evidence(best["roi"], ev_path)
                    ocr_stats["queue_total"] += 1
                    if res.queue_text.text:
                        ocr_stats["queue_nonempty"] += 1
                    queue_events.append(
                        {
                            "t": round(float(best["t"]), 3),
                            "frame": frame_path,
                            "ocr": {"queue_text": {"text": res.queue_text.text, "conf": round(float(res.queue_text.conf), 3)}},
                        }
                    )
                    if res.queue_text.text:
                        events.append(
                            {
                                "t": round(float(best["t"]), 3),
                                "id": f"{res.queue_text.text.strip().lower()}_started",
                                "count": 1,
                                "conf": round(float(res.queue_text.conf), 3),
                                "evidence": [frame_path],
                                "source": "queue_ocr",
                            }
                        )
                last_queue = queue_roi

    output = {
        "version": 1,
        "segment": {"start_sec": cfg.start_sec, "end_sec": cfg.end_sec},
        "roi_profile": Path(cfg.profile_path).name.replace(".json", ""),
        "signals": {
            "supply_series": supply_series,
            "selection_changes": selection_changes,
            "queue_events": queue_events,
        },
        "events": events,
        "diagnostics": {
            "warnings": [],
            "ocr_engine": ocr.name,
            "preprocess": "upscale3x+adaptive_threshold",
            "ocr_stats": ocr_stats,
        },
    }

    Path(cfg.output_path).write_text(json.dumps(output, indent=2), encoding="utf-8")
    return output


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]
