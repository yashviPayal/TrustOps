from fastapi.testclient import TestClient
import app.api.main as api

def test_health_without_lifespan_models():
    # Route-level contract test; no trained dataset/artifacts required.
    assert api.root()["service"] == "TrustOps"
