import numpy as np
from app.ml.metrics import choose_threshold, evaluate

def test_threshold_is_between_zero_and_one():
    y = np.array([0, 0, 0, 1, 1, 1])
    p = np.array([0.05, 0.1, 0.3, 0.6, 0.8, 0.9])
    threshold = choose_threshold(y, p)
    assert 0 <= threshold <= 1

def test_evaluate_returns_expected_metrics():
    y = np.array([0, 0, 1, 1])
    p = np.array([0.1, 0.2, 0.8, 0.9])
    result = evaluate(y, p, 0.5)
    assert result["average_precision"] == 1.0
    assert result["precision"] == 1.0
    assert result["recall"] == 1.0
    assert result["tp"] == 2
    assert result["tn"] == 2
