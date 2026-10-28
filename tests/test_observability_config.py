import json
from pathlib import Path

from prometheus_client import generate_latest

ROOT = Path(__file__).resolve().parents[1]


def test_metrics_are_exported_before_first_prediction():
    # Importing the module registers and initializes the dashboard's core series.
    from app.api import monitoring  # noqa: F401

    output = generate_latest().decode("utf-8")
    assert "trustops_predictions_total" in output
    assert 'model="champion"' in output
    assert "trustops_prediction_errors_total" in output
    assert "trustops_input_drift_psi" in output
    assert "trustops_delayed_label_f1" in output


def test_grafana_uses_the_provisioned_prometheus_uid():
    datasource = (ROOT / "monitoring/grafana/provisioning/datasources/datasource.yml").read_text()
    dashboard = json.loads((ROOT / "monitoring/grafana/dashboards/trustops.json").read_text())

    assert "uid: prometheus" in datasource
    assert dashboard["panels"]
    assert all(panel.get("datasource", {}).get("uid") == "prometheus" for panel in dashboard["panels"])
    assert any('up{job="trustops-api"}' in target["expr"] for panel in dashboard["panels"] for target in panel.get("targets", []))


def test_mlflow_is_built_instead_of_installing_at_container_startup():
    compose = (ROOT / "docker-compose.yml").read_text()
    mlflow_dockerfile = (ROOT / "Dockerfile.mlflow").read_text()

    assert "dockerfile: Dockerfile.mlflow" in compose
    assert "pip install --no-cache-dir mlflow==2.22.0" in mlflow_dockerfile
    assert "--serve-artifacts" in mlflow_dockerfile
    assert "pip install" not in compose
