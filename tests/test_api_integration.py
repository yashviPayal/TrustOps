import pytest
from fastapi.testclient import TestClient
from app.api.main import app

@pytest.fixture(scope="module")
def client():
    with TestClient(app) as c:
        yield c

def test_api_health(client):
    res = client.get("/health")
    assert res.status_code == 200
    data = res.json()
    assert data["status"] == "ok"
    assert "champion" in data["models_loaded"]

def test_model_info(client):
    res = client.get("/model-info")
    assert res.status_code == 200
    data = res.json()
    assert data["project"] == "TrustOps"
    assert "feature_names" in data
    assert len(data["feature_names"]) == 29
    assert "release_gate_status" in data

def test_predict_endpoint(client):
    tx_features = {f"V{i}": 0.0 for i in range(1, 29)}
    tx_features["Amount"] = 125.50
    res = client.post("/predict", json={"transaction": tx_features})
    assert res.status_code == 200
    data = res.json()
    assert "fraud_probability" in data
    assert "risk" in data
    assert "decision" in data
    assert data["model_version"] in {"champion", "challenger"}

def test_batch_predict_endpoint(client):
    batch = []
    for _ in range(5):
        tx = {f"V{i}": 0.1 for i in range(1, 29)}
        tx["Amount"] = 45.0
        batch.append(tx)
    res = client.post("/predict/batch", json={"transactions": batch})
    assert res.status_code == 200
    data = res.json()
    assert data["count"] == 5
    assert len(data["predictions"]) == 5

def test_drift_endpoints(client):
    res = client.get("/monitoring/drift")
    assert res.status_code == 200
    assert "status" in res.json()

    res_reset = client.post("/monitoring/drift/reset")
    assert res_reset.status_code == 200
    assert res_reset.json()["status"] == "reset_successful"

def test_replay_flow(client):
    res = client.post("/replay/start", json={"scenario": "baseline", "speed_tps": 20, "label_delay_steps": 5})
    assert res.status_code == 200
    assert res.json()["is_running"] is True

    res_status = client.get("/replay/status")
    assert res_status.status_code == 200
    assert res_status.json()["scenario"] == "baseline"

    client.post("/replay/pause")
    client.post("/replay/resume")

    res_stop = client.post("/replay/stop")
    assert res_stop.status_code == 200
    assert res_stop.json()["is_running"] is False

def test_rollout_controls(client):
    res = client.post("/models/rollout-mode", json={"mode": "shadow"})
    assert res.status_code == 200
    assert res.json()["mode"] == "shadow"

    res_canary = client.post("/models/rollout-mode", json={"mode": "canary", "canary_percent": 25})
    assert res_canary.status_code == 200
    assert res_canary.json()["mode"] == "canary"
    assert res_canary.json()["canary_percent"] == 25

    client.post("/models/rollout-mode", json={"mode": "champion"})

def test_scam_analysis_endpoint(client):
    res = client.post(
        "/scam/analyze-text",
        json={"text": "URGENT: CEO wire transfer payment of $50,000 required immediately to updated bank account.", "source_type": "email"}
    )
    assert res.status_code == 200
    data = res.json()
    assert data["risk_score"] >= 0.60
    assert len(data["flagged_categories"]) >= 2

def test_metrics_prometheus_endpoint(client):
    res = client.get("/metrics")
    assert res.status_code == 200
    content = res.text
    assert "trustops_predictions_total" in content
    assert "trustops_input_drift_psi" in content
