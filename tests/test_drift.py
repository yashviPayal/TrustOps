import json
import numpy as np
import pytest
from pathlib import Path
from app.api.monitoring import InputMonitor

@pytest.fixture
def dummy_reference(tmp_path):
    ref_path = tmp_path / "reference.json"
    rng = np.random.default_rng(42)
    ref_data = {}
    for feat in ["V1", "V2", "Amount"]:
        vals = rng.normal(0.0, 1.0, 500)
        edges = np.quantile(vals, np.linspace(0, 1, 11)).tolist()
        counts, _ = np.histogram(vals, bins=edges)
        proportions = (counts / max(counts.sum(), 1)).tolist()
        ref_data[feat] = {
            "edges": edges,
            "proportions": proportions,
            "mean": float(np.mean(vals)),
            "std": float(np.std(vals)),
            "sample": vals.tolist(),
        }
    ref_path.write_text(json.dumps(ref_data), encoding="utf-8")
    return ref_path

def test_drift_collecting_state_before_min_samples(dummy_reference):
    monitor = InputMonitor(dummy_reference, window_size=100, min_samples=30)
    for _ in range(10):
        monitor.add({"V1": 0.1, "V2": -0.2, "Amount": 50.0})
    report = monitor.report()
    assert report["status"] == "collecting"
    assert report["samples"] == 10
    assert report["required"] == 30
    assert report["features"] == {}

def test_baseline_samples_do_not_trigger_severe_drift(dummy_reference):
    monitor = InputMonitor(dummy_reference, window_size=200, min_samples=40, ks_alpha=0.01)
    rng = np.random.default_rng(123)
    # Feed 60 samples from the baseline normal distribution
    for _ in range(60):
        monitor.add({
            "V1": float(rng.normal(0.0, 1.0)),
            "V2": float(rng.normal(0.0, 1.0)),
            "Amount": float(rng.normal(0.0, 1.0)),
        })
    report = monitor.report()
    assert report["status"] == "healthy"
    assert report["drifted_feature_count"] == 0
    assert len(report["features"]) == 3

def test_deliberately_shifted_samples_trigger_expected_drift(dummy_reference):
    monitor = InputMonitor(
        dummy_reference,
        window_size=200,
        min_samples=40,
        effect_threshold=0.15,
        psi_threshold=0.20,
    )
    # Feed heavily shifted samples (mean shifted by +3.5 standard deviations)
    for _ in range(60):
        monitor.add({
            "V1": 4.5,
            "V2": -4.0,
            "Amount": 10.0,
        })
    report = monitor.report()
    assert report["status"] == "drift_detected"
    assert report["drifted_feature_count"] > 0
    assert len(report["alerts"]) > 0

def test_drift_reset_clears_window(dummy_reference):
    monitor = InputMonitor(dummy_reference, window_size=100, min_samples=20)
    for _ in range(25):
        monitor.add({"V1": 3.0, "V2": 3.0, "Amount": 100.0})
    assert monitor.report()["samples"] == 25
    monitor.reset()
    assert monitor.report()["samples"] == 0
    assert monitor.report()["status"] == "collecting"
