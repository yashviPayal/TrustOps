import json
import os
from datetime import datetime, timezone
from pathlib import Path

import joblib
import mlflow
import mlflow.sklearn
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from app.core.config import DATA_PATH, MODEL_DIR
from app.ml.data import load_dataset, feature_columns, split_data
from app.ml.metrics import choose_threshold, evaluate

EXPERIMENT_NAME = "TrustOps-Credit-Card-Fraud"

def make_model(C: float) -> Pipeline:
    # class_weight='balanced' is a baseline response to severe imbalance.
    # The threshold is separately selected on validation data.
    return Pipeline([
        ("scaler", StandardScaler()),
        ("classifier", LogisticRegression(
            C=C, class_weight="balanced", max_iter=1200, solver="lbfgs",
            random_state=42
        )),
    ])

def make_reference(X, features):
    reference = {}
    for col in features:
        values = X[col].astype(float).to_numpy()
        quantiles = np.quantile(values, np.linspace(0, 1, 11)).tolist()
        # Ensure strictly increasing bin edges for PSI calculations.
        edges = []
        for value in quantiles:
            if not edges or value > edges[-1]:
                edges.append(float(value))
        if len(edges) < 2:
            edges = [float(values.min()) - 1e-9, float(values.max()) + 1e-9]
        counts, bins = np.histogram(values, bins=np.asarray(edges))
        proportions = (counts / max(counts.sum(), 1)).astype(float).tolist()
        reference[col] = {"edges": edges, "proportions": proportions}
    return reference

def main():
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    df = load_dataset(DATA_PATH)
    features = feature_columns(df)
    X_train, X_val, X_test, y_train, y_val, y_test = split_data(df, features)

    tracking_uri = os.getenv("MLFLOW_TRACKING_URI", "file:./mlruns")
    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment(EXPERIMENT_NAME)

    candidates = [
        ("champion", 1.0),
        ("challenger", 0.1),
    ]
    fitted = {}
    val_results = {}

    for name, c_value in candidates:
        model = make_model(c_value)
        with mlflow.start_run(run_name=f"{name}-logistic-regression") as run:
            model.fit(X_train, y_train)
            val_prob = model.predict_proba(X_val)[:, 1]
            threshold = choose_threshold(y_val, val_prob)
            metrics = evaluate(y_val, val_prob, threshold)
            mlflow.log_params({
                "model": "LogisticRegression",
                "C": c_value,
                "class_weight": "balanced",
                "features": ",".join(features),
                "split": "stratified_70_15_15",
                "threshold_selection": "validation_max_f1",
            })
            mlflow.log_metrics({f"val_{k}": float(v) for k, v in metrics.items()})
            mlflow.set_tags({"role": name, "dataset": "ULB-credit-card-fraud"})
            mlflow.sklearn.log_model(model, artifact_path="model")
            fitted[name] = model
            val_results[name] = {"threshold": threshold, "metrics": metrics, "run_id": run.info.run_id}
            print(f"{name} validation: {json.dumps(metrics, indent=2)}")

    # Select a champion by validation average precision; the test set is not
    # used to select the model or threshold.
    winner = max(val_results, key=lambda n: val_results[n]["metrics"]["average_precision"])
    ordered = [winner, "challenger" if winner == "champion" else "champion"]
    final_metadata = {
        "project": "TrustOps",
        "dataset": "ULB Credit Card Fraud Detection",
        "trained_at_utc": datetime.now(timezone.utc).isoformat(),
        "feature_names": features,
        "champion_source_candidate": winner,
        "models": {},
    }

    for output_name, candidate_name in zip(["champion", "challenger"], ordered):
        model = fitted[candidate_name]
        threshold = val_results[candidate_name]["threshold"]
        test_prob = model.predict_proba(X_test)[:, 1]
        test_metrics = evaluate(y_test, test_prob, threshold)
        artifact_path = MODEL_DIR / f"{output_name}.joblib"
        joblib.dump(model, artifact_path)
        final_metadata["models"][output_name] = {
            "file": artifact_path.name,
            "source_candidate": candidate_name,
            "threshold": threshold,
            "validation_metrics": val_results[candidate_name]["metrics"],
            "test_metrics": test_metrics,
            "mlflow_run_id": val_results[candidate_name]["run_id"],
        }
        print(f"{output_name} test metrics: {json.dumps(test_metrics, indent=2)}")

    # Reference distributions are derived only from training inputs.
    reference = make_reference(X_train, features)
    (MODEL_DIR / "reference.json").write_text(json.dumps(reference, indent=2), encoding="utf-8")
    (MODEL_DIR / "metadata.json").write_text(json.dumps(final_metadata, indent=2), encoding="utf-8")
    print(f"\nSaved model artifacts to: {MODEL_DIR.resolve()}")
    print("Training complete. Test metrics are reports, not a guarantee of live performance.")

if __name__ == "__main__":
    main()
