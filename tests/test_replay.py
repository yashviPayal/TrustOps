import pytest
from pathlib import Path
from app.api.replay import TransactionStore, TransactionReplayEngine

def test_transaction_store_crud():
    store = TransactionStore(max_size=10)
    tx = {
        "transaction_id": "tx-001",
        "timestamp": "2026-10-25T12:00:00Z",
        "features": {"Amount": 100.0, "V1": 0.5},
        "prediction": {"fraud_probability": 0.85, "risk": "high", "decision": "review", "model_version": "champion"},
        "latency_ms": 2.5,
        "ground_truth": None,
        "label_status": "pending",
    }
    store.add(tx)
    recent = store.get_recent(limit=5)
    assert len(recent) == 1
    assert recent[0]["transaction_id"] == "tx-001"

    # Resolve label
    store.resolve_label("tx-001", 1)
    retrieved = store.get_by_id("tx-001")
    assert retrieved["ground_truth"] == 1
    assert retrieved["label_status"] == "resolved"

    # Check summary
    summary = store.summary()
    assert summary["total_recorded"] == 1
    assert summary["risk_counts"]["high"] == 1
    assert summary["decision_counts"]["review"] == 1
    assert summary["confirmed_fraud_count"] == 1

def test_replay_engine_scenarios(tmp_path):
    store = TransactionStore()
    def mock_scorer(features):
        return {
            "fraud_probability": 0.1,
            "risk": "low",
            "decision": "allow",
            "threshold": 0.5,
            "model_version": "champion",
        }

    # Pass nonexistent path so it uses synthetic fallback
    engine = TransactionReplayEngine(
        held_out_path=tmp_path / "nonexistent.json",
        store=store,
        score_func=mock_scorer,
    )
    assert len(engine._records) > 0

    # Test baseline sampling
    engine.scenario = "baseline"
    feats_base, label_base = engine._sample_next_record()
    assert "Amount" in feats_base
    assert "V1" in feats_base

    # Test feature drift sampling (Amount and V1 shifted)
    engine.scenario = "feature_drift"
    feats_drift, _ = engine._sample_next_record()
    assert feats_drift["V1"] != 0.0

    # Test single step
    result = engine._step_once()
    assert result["transaction_id"].startswith("tx-")
    assert engine.total_replayed == 1
