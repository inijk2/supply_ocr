from pathlib import Path
import json

from eval.eval import evaluate


def test_evaluate_simple_match(tmp_path: Path):
    pred = {"events": [{"t": 10.0, "id": "marine_started"}]}
    gt = {"events": [{"t": 11.0, "id": "marine_started"}]}
    pred_path = tmp_path / "pred.json"
    gt_path = tmp_path / "gt.json"
    pred_path.write_text(json.dumps(pred), encoding="utf-8")
    gt_path.write_text(json.dumps(gt), encoding="utf-8")

    result = evaluate(str(pred_path), str(gt_path), max_dt=3.0)
    m = result["metrics"]
    assert m["precision"] == 1.0
    assert m["recall"] == 1.0
