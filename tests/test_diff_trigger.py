import numpy as np

from detect.diff_trigger import diff_score


def test_diff_score_detects_change():
    a = np.zeros((20, 20), dtype=np.uint8)
    b = a.copy()
    b[5:10, 5:10] = 255
    score = diff_score(a, b)
    assert score > 0.0
