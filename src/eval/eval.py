from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple


@dataclass
class Metrics:
    precision: float
    recall: float
    f1: float
    mean_dt: float


def _load(path: str) -> Dict:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _match_events(pred: List[Dict], gt: List[Dict], max_dt: float) -> Tuple[List[Tuple[Dict, Dict]], List[Dict], List[Dict]]:
    matched = []
    used_pred = set()
    used_gt = set()
    for gi, g in enumerate(gt):
        best = None
        best_i = None
        for pi, p in enumerate(pred):
            if pi in used_pred:
                continue
            if p.get("id") != g.get("id"):
                continue
            dt = abs(float(p.get("t", 0.0)) - float(g.get("t", 0.0)))
            if dt <= max_dt and (best is None or dt < best):
                best = dt
                best_i = pi
        if best_i is not None:
            used_pred.add(best_i)
            used_gt.add(gi)
            matched.append((pred[best_i], g))
    unmatched_pred = [p for i, p in enumerate(pred) if i not in used_pred]
    unmatched_gt = [g for i, g in enumerate(gt) if i not in used_gt]
    return matched, unmatched_pred, unmatched_gt


def evaluate(pred_path: str, gt_path: str, max_dt: float = 3.0) -> Dict:
    pred = _load(pred_path)
    gt = _load(gt_path)
    pred_events = pred.get("events", [])
    gt_events = gt.get("events", [])

    matched, un_pred, un_gt = _match_events(pred_events, gt_events, max_dt)
    tp = len(matched)
    fp = len(un_pred)
    fn = len(un_gt)
    precision = tp / (tp + fp) if tp + fp > 0 else 0.0
    recall = tp / (tp + fn) if tp + fn > 0 else 0.0
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0

    if matched:
        mean_dt = sum(abs(float(p["t"]) - float(g["t"])) for p, g in matched) / len(matched)
    else:
        mean_dt = 0.0

    return {
        "metrics": Metrics(precision, recall, f1, mean_dt).__dict__,
        "matched": [{"pred": p, "gt": g} for p, g in matched],
        "unmatched_pred": un_pred,
        "unmatched_gt": un_gt,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate predicted events vs GT.")
    parser.add_argument("pred", help="Predicted output JSON")
    parser.add_argument("gt", help="Ground truth JSON")
    parser.add_argument("--max-dt", type=float, default=3.0, help="Max time delta for match")
    parser.add_argument("-o", "--out", default="eval.json", help="Output evaluation JSON")
    args = parser.parse_args()

    result = evaluate(args.pred, args.gt, args.max_dt)
    Path(args.out).write_text(json.dumps(result, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
