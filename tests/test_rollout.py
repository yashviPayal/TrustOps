from app.api.rollout import RolloutManager

def test_champion_mode_never_selects_challenger():
    manager = RolloutManager(mode="champion", canary_percent=100)
    assert all(manager.choose_model() == "champion" for _ in range(20))

def test_canary_rolls_back_after_repeated_slow_candidate_predictions():
    manager = RolloutManager(
        mode="canary", canary_percent=100, max_latency_ms=10, latency_failures=3
    )
    manager.record_candidate_latency(20)
    manager.record_candidate_latency(20)
    assert not manager.rolled_back
    manager.record_candidate_latency(20)
    assert manager.rolled_back
    assert manager.choose_model() == "champion"

def test_canary_rolls_back_after_repeated_errors():
    manager = RolloutManager(
        mode="canary", canary_percent=100, error_failures=3
    )
    manager.record_prediction_error("challenger")
    manager.record_prediction_error("challenger")
    assert not manager.rolled_back
    manager.record_prediction_error("challenger")
    assert manager.rolled_back
    assert "consecutive errors" in manager.status()["rollback_reason"]
    assert manager.choose_model() == "champion"

def test_shadow_mode_is_valid():
    manager = RolloutManager(mode="shadow")
    assert manager.status()["mode"] == "shadow"
    assert manager.choose_model() == "champion"

def test_release_gates_pass_for_valid_candidate():
    manager = RolloutManager(
        gate_min_pr_auc=0.70,
        gate_min_precision=0.55,
        gate_min_recall=0.70,
        gate_max_latency_ms=50.0,
    )
    candidate_metrics = {
        "average_precision": 0.78,
        "precision": 0.65,
        "recall": 0.81,
    }
    features = ["V1", "V2", "Amount"]
    result = manager.evaluate_candidate_gates(
        candidate_metrics=candidate_metrics,
        expected_features=features,
        actual_features=features,
        candidate_avg_latency_ms=3.5,
    )
    assert result["passed"] is True
    assert result["gates"]["min_pr_auc"]["passed"] is True
    assert result["gates"]["schema_compatibility"]["passed"] is True

def test_release_gates_block_candidate_when_metric_fails():
    manager = RolloutManager(
        gate_min_pr_auc=0.75,
        gate_min_precision=0.60,
    )
    failing_metrics = {
        "average_precision": 0.50,  # Below 0.75
        "precision": 0.40,          # Below 0.60
        "recall": 0.80,
    }
    features = ["V1", "V2", "Amount"]
    result = manager.evaluate_candidate_gates(
        candidate_metrics=failing_metrics,
        expected_features=features,
        actual_features=features,
        candidate_avg_latency_ms=4.0,
    )
    assert result["passed"] is False
    assert result["gates"]["min_pr_auc"]["passed"] is False
    assert result["gates"]["min_precision"]["passed"] is False

def test_release_gates_block_candidate_when_schema_mismatch():
    manager = RolloutManager()
    candidate_metrics = {"average_precision": 0.85, "precision": 0.80, "recall": 0.80}
    result = manager.evaluate_candidate_gates(
        candidate_metrics=candidate_metrics,
        expected_features=["V1", "V2", "Amount"],
        actual_features=["V1", "V2"],  # Missing Amount!
    )
    assert result["passed"] is False
    assert result["gates"]["schema_compatibility"]["passed"] is False

def test_manual_rollback_and_reset():
    manager = RolloutManager(mode="canary", canary_percent=50)
    manager.manual_rollback("Emergency test rollback")
    assert manager.rolled_back is True
    assert manager.choose_model() == "champion"
    manager.reset_rollback()
    assert manager.rolled_back is False
