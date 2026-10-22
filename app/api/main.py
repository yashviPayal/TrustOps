import json
import time
from pathlib import Path
from contextlib import asynccontextmanager

import joblib
import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.responses import Response
from prometheus_client import CONTENT_TYPE_LATEST, generate_latest

from app.core.config import (
    MODEL_DIR, ROLLOUT_MODE, CANARY_PERCENT, CANARY_MAX_LATENCY_MS,
    CANARY_LATENCY_FAILURES, DRIFT_WINDOW_SIZE
)
from app.api.schemas import PredictRequest, BatchPredictRequest
from app.api.rollout import RolloutManager
from app.api.monitoring import (
    PREDICTIONS, PREDICTION_ERRORS, PREDICTION_LATENCY, CANARY_ROLLBACKS, InputMonitor
)

models = {}
metadata = {}
features = []
thresholds = {}
rollout = RolloutManager(
    mode=ROLLOUT_MODE, canary_percent=CANARY_PERCENT,
    max_latency_ms=CANARY_MAX_LATENCY_MS, latency_failures=CANARY_LATENCY_FAILURES
)
monitor = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    global models, metadata, features, thresholds, monitor
    champion_path = MODEL_DIR / "champion.joblib"
    challenger_path = MODEL_DIR / "challenger.joblib"
    metadata_path = MODEL_DIR / "metadata.json"
    if not champion_path.exists() or not metadata_path.exists():
        raise RuntimeError(
            f"Model artifacts not found in {MODEL_DIR}. Run `python -m app.ml.train` first."
        )
    models["champion"] = joblib.load(champion_path)
    if challenger_path.exists():
        models["challenger"] = joblib.load(challenger_path)
    else:
        models["challenger"] = models["champion"]
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    features = metadata.get("feature_names", [])
    thresholds = {
        name: info.get("threshold", 0.5)
        for name, info in metadata.get("models", {}).items()
    }
    monitor = InputMonitor(MODEL_DIR / "reference.json", DRIFT_WINDOW_SIZE)
    yield

app = FastAPI(
    title="TrustOps Fraud Detection API",
    description="MLOps demo: credit-card fraud risk scoring, rollout controls, drift and Prometheus metrics.",
    version="1.0.0",
    lifespan=lifespan,
)

def build_frame(transaction_dict):
    row = {}
    for feature in features:
        value = transaction_dict.get(feature)
        if value is None:
            value = 0.0
        row[feature] = float(value)
    return pd.DataFrame([row], columns=features)

def risk_label(probability, threshold):
    if probability >= max(threshold, 0.75):
        return "high"
    if probability >= threshold:
        return "elevated"
    return "low"

def score_one(transaction_dict):
    if not models or not features:
        raise HTTPException(status_code=503, detail="Models are not loaded.")
    start = time.perf_counter()
    try:
        frame = build_frame(transaction_dict)
        chosen = rollout.choose_model()
        chosen_start = time.perf_counter()
        chosen_probability = float(models[chosen].predict_proba(frame)[0, 1])
        chosen_latency = (time.perf_counter() - chosen_start) * 1000

        shadow_probability = None
        # Shadow always serves champion, while evaluating challenger as a comparison.
        if rollout.mode == "shadow":
            shadow_probability = float(models["challenger"].predict_proba(frame)[0, 1])
            chosen = "champion"
            chosen_probability = float(models["champion"].predict_proba(frame)[0, 1])
        if chosen == "challenger":
            rollout.record_candidate_latency(chosen_latency)
            if rollout.rolled_back:
                CANARY_ROLLBACKS.inc()
                chosen = "champion"
                chosen_probability = float(models["champion"].predict_proba(frame)[0, 1])

        threshold = float(thresholds.get(chosen, 0.5))
        decision = "review" if chosen_probability >= threshold else "allow"
        PREDICTIONS.labels(model=chosen, decision=decision).inc()
        monitor.add(transaction_dict)
        return {
            "fraud_probability": round(chosen_probability, 8),
            "risk": risk_label(chosen_probability, threshold),
            "threshold": threshold,
            "model_version": chosen,
            "rollout_mode": rollout.mode,
            "decision": decision,
            "shadow_challenger_probability": (
                round(shadow_probability, 8) if shadow_probability is not None else None
            ),
            "canary_rolled_back": rollout.rolled_back,
        }
    except HTTPException:
        raise
    except Exception as exc:
        PREDICTION_ERRORS.inc()
        raise HTTPException(status_code=500, detail=f"Prediction failed: {type(exc).__name__}") from exc
    finally:
        PREDICTION_LATENCY.observe(time.perf_counter() - start)

@app.get("/")
def root():
    return {"service": "TrustOps", "docs": "/docs", "health": "/health"}

@app.get("/health")
def health():
    return {
        "status": "ok" if models else "not_ready",
        "models_loaded": sorted(models.keys()),
        "feature_count": len(features),
    }

@app.get("/model-info")
def model_info():
    return {
        "project": metadata.get("project", "TrustOps"),
        "dataset": metadata.get("dataset"),
        "trained_at_utc": metadata.get("trained_at_utc"),
        "feature_names": features,
        "models": metadata.get("models", {}),
        "rollout": rollout.status(),
    }

@app.post("/predict")
def predict(request: PredictRequest):
    return score_one(request.transaction.to_feature_dict())

@app.post("/predict/batch")
def predict_batch(request: BatchPredictRequest):
    results = []
    for transaction in request.transactions:
        results.append(score_one(transaction.to_feature_dict()))
    return {"count": len(results), "predictions": results}

@app.get("/monitoring/drift")
def drift_report():
    return monitor.report() if monitor else {"status": "not_ready"}

@app.get("/metrics")
def metrics():
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)
