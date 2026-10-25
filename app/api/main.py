import json
import os
import time
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException, Query, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import Response
from fastapi.staticfiles import StaticFiles
from prometheus_client import CONTENT_TYPE_LATEST, generate_latest

from app.core.config import (
    MODEL_DIR, ROLLOUT_MODE, CANARY_PERCENT, CANARY_MAX_LATENCY_MS,
    CANARY_LATENCY_FAILURES, CANARY_ERROR_FAILURES, DRIFT_WINDOW_SIZE,
    DRIFT_MIN_SAMPLES, DRIFT_KS_ALPHA, DRIFT_PSI_THRESHOLD,
    DRIFT_EFFECT_SIZE_THRESHOLD, PERFORMANCE_WINDOW_SIZE,
    PERFORMANCE_F1_ALERT_THRESHOLD, HELD_OUT_STREAM_PATH, ROOT_DIR
)
from app.api.schemas import (
    PredictRequest, BatchPredictRequest, Prediction,
    ReplayStartRequest, RolloutModeRequest, PromoteRequest,
    RollbackRequest, ScamTextRequest
)
from app.api.rollout import RolloutManager
from app.api.monitoring import (
    PREDICTIONS, PREDICTION_ERRORS, PREDICTION_LATENCY, CANARY_ROLLBACKS,
    PROMOTIONS, InputMonitor, DelayedPerformanceMonitor
)
from app.api.replay import TransactionStore, TransactionReplayEngine
from app.scam.detector import ScamAnalyzer, SAMPLE_SCENARIOS

# Global services & state
models: dict[str, Any] = {}
metadata: dict[str, Any] = {}
features: list[str] = []
thresholds: dict[str, float] = {}

rollout = RolloutManager(
    mode=ROLLOUT_MODE,
    canary_percent=CANARY_PERCENT,
    max_latency_ms=CANARY_MAX_LATENCY_MS,
    latency_failures=CANARY_LATENCY_FAILURES,
    error_failures=CANARY_ERROR_FAILURES,
)
monitor: InputMonitor | None = None
delayed_monitor: DelayedPerformanceMonitor | None = None
transaction_store = TransactionStore(max_size=350)
replay_engine: TransactionReplayEngine | None = None
scam_analyzer = ScamAnalyzer()
server_start_time = time.time()


def build_frame(transaction_dict: dict[str, Any]) -> pd.DataFrame:
    row = {}
    for feature in features:
        value = transaction_dict.get(feature)
        if value is None:
            value = 0.0
        row[feature] = float(value)
    return pd.DataFrame([row], columns=features)


def risk_label(probability: float, threshold: float) -> str:
    if probability >= max(threshold, 0.75):
        return "high"
    if probability >= threshold:
        return "elevated"
    return "low"


def score_one(transaction_dict: dict[str, Any]) -> dict[str, Any]:
    if not models or not features:
        raise HTTPException(status_code=503, detail="Models are not loaded.")
    start = time.perf_counter()
    chosen = "champion"
    try:
        frame = build_frame(transaction_dict)
        chosen = rollout.choose_model()
        chosen_start = time.perf_counter()
        chosen_probability = float(models[chosen].predict_proba(frame)[0, 1])
        chosen_latency = (time.perf_counter() - chosen_start) * 1000

        shadow_probability = None
        # Shadow mode scores challenger concurrently, but serves champion decision
        if rollout.mode == "shadow":
            shadow_probability = float(models["challenger"].predict_proba(frame)[0, 1])
            chosen = "champion"
            chosen_probability = float(models["champion"].predict_proba(frame)[0, 1])

        if chosen == "challenger":
            rollout.record_prediction_success("challenger", chosen_latency)
            if rollout.rolled_back:
                CANARY_ROLLBACKS.labels(reason="canary_latency_or_error").inc()
                chosen = "champion"
                chosen_probability = float(models["champion"].predict_proba(frame)[0, 1])
        else:
            rollout.record_prediction_success("champion", chosen_latency)

        threshold = float(thresholds.get(chosen, 0.5))
        decision = "review" if chosen_probability >= threshold else "allow"
        risk = risk_label(chosen_probability, threshold)

        PREDICTIONS.labels(model=chosen, decision=decision, risk=risk).inc()
        if monitor:
            monitor.add(transaction_dict)

        return {
            "fraud_probability": round(chosen_probability, 8),
            "risk": risk,
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
        if chosen == "challenger":
            rollout.record_prediction_error("challenger")
        raise HTTPException(status_code=500, detail=f"Prediction failed: {type(exc).__name__}") from exc
    finally:
        PREDICTION_LATENCY.observe(time.perf_counter() - start)


@asynccontextmanager
async def lifespan(app: FastAPI):
    global models, metadata, features, thresholds, monitor, delayed_monitor, replay_engine
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

    # Initialize data drift monitor with full statistical configuration
    monitor = InputMonitor(
        reference_path=MODEL_DIR / "reference.json",
        window_size=DRIFT_WINDOW_SIZE,
        min_samples=DRIFT_MIN_SAMPLES,
        ks_alpha=DRIFT_KS_ALPHA,
        psi_threshold=DRIFT_PSI_THRESHOLD,
        effect_threshold=DRIFT_EFFECT_SIZE_THRESHOLD,
    )

    # Initialize concept drift & delayed label monitor
    delayed_monitor = DelayedPerformanceMonitor(
        window_size=PERFORMANCE_WINDOW_SIZE,
        f1_alert_threshold=PERFORMANCE_F1_ALERT_THRESHOLD,
    )

    # Initialize transaction replay engine
    replay_engine = TransactionReplayEngine(
        held_out_path=HELD_OUT_STREAM_PATH,
        store=transaction_store,
        score_func=score_one,
        on_prediction_hook=delayed_monitor.record_prediction,
        on_label_hook=delayed_monitor.record_label,
    )

    yield

    if replay_engine and replay_engine.is_running:
        replay_engine.stop()


app = FastAPI(
    title="TrustOps Fraud Detection & MLOps Platform",
    description="End-to-End MLOps Platform: Inference, Drift Monitoring, Canary/Shadow Rollouts, Automated Rollback, Delayed Labels & Scam Analysis.",
    version="2.0.0",
    lifespan=lifespan,
)

# Enable CORS for local development and dashboard UI
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ---------------------------------------------------------------------------
# Core Prediction Endpoints
# ---------------------------------------------------------------------------
@app.get("/")
def root():
    return {
        "service": "TrustOps",
        "platform": "TrustOps MLOps Fraud Detection Platform",
        "docs": "/docs",
        "health": "/health",
        "metrics": "/metrics",
        "monitoring": "/monitoring/drift",
        "replay": "/replay/status",
    }


@app.get("/health")
def health():
    uptime_sec = round(time.time() - server_start_time, 1)
    return {
        "status": "ok" if models else "not_ready",
        "models_loaded": sorted(models.keys()),
        "feature_count": len(features),
        "uptime_seconds": uptime_sec,
        "rollout_mode": rollout.mode,
        "canary_rolled_back": rollout.rolled_back,
    }


@app.get("/model-info")
def model_info():
    cand_metrics = metadata.get("models", {}).get("challenger", {}).get("test_metrics", {})
    gate_eval = rollout.evaluate_candidate_gates(
        candidate_metrics=cand_metrics,
        expected_features=features,
        actual_features=features,
        candidate_avg_latency_ms=rollout.status()["avg_candidate_latency_ms"] or 2.5,
    )
    return {
        "project": metadata.get("project", "TrustOps"),
        "dataset": metadata.get("dataset"),
        "trained_at_utc": metadata.get("trained_at_utc"),
        "feature_names": features,
        "models": metadata.get("models", {}),
        "rollout": rollout.status(),
        "release_gate_status": gate_eval,
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


# ---------------------------------------------------------------------------
# Data Drift Monitoring Endpoints
# ---------------------------------------------------------------------------
@app.get("/monitoring/drift")
def drift_report():
    return monitor.report() if monitor else {"status": "not_ready"}


@app.post("/monitoring/drift/reset")
def reset_drift():
    if monitor:
        monitor.reset()
    return {"status": "reset_successful", "message": "Input monitoring window cleared."}


@app.get("/monitoring/drift/history")
def drift_history():
    return {
        "count": len(monitor.get_history()) if monitor else 0,
        "history": monitor.get_history() if monitor else [],
    }


# ---------------------------------------------------------------------------
# Concept Drift & Delayed Labels Endpoints
# ---------------------------------------------------------------------------
@app.get("/monitoring/performance")
def performance_report():
    return delayed_monitor.report() if delayed_monitor else {"status": "not_ready"}


@app.get("/monitoring/performance/history")
def performance_history():
    return {
        "count": len(delayed_monitor.get_history()) if delayed_monitor else 0,
        "history": delayed_monitor.get_history() if delayed_monitor else [],
    }


@app.post("/monitoring/performance/reset")
def reset_performance():
    if delayed_monitor:
        delayed_monitor.reset()
    return {"status": "reset_successful", "message": "Performance monitor reset."}


# ---------------------------------------------------------------------------
# Transaction Replay & History Endpoints
# ---------------------------------------------------------------------------
@app.post("/replay/start")
def start_replay(req: ReplayStartRequest):
    if not replay_engine:
        raise HTTPException(status_code=503, detail="Replay engine not initialized.")
    return replay_engine.start(
        scenario=req.scenario,
        speed_tps=req.speed_tps,
        label_delay_steps=req.label_delay_steps,
    )


@app.post("/replay/pause")
def pause_replay():
    if not replay_engine:
        raise HTTPException(status_code=503, detail="Replay engine not initialized.")
    return replay_engine.pause()


@app.post("/replay/resume")
def resume_replay():
    if not replay_engine:
        raise HTTPException(status_code=503, detail="Replay engine not initialized.")
    return replay_engine.resume()


@app.post("/replay/stop")
def stop_replay():
    if not replay_engine:
        raise HTTPException(status_code=503, detail="Replay engine not initialized.")
    return replay_engine.stop()


@app.get("/replay/status")
def replay_status():
    if not replay_engine:
        return {"status": "not_ready"}
    return replay_engine.status()


@app.get("/transactions/recent")
def recent_transactions(
    limit: int = Query(50, ge=1, le=200),
    risk: str | None = None,
    decision: str | None = None,
    model: str | None = None,
):
    return {
        "count": len(transaction_store.get_recent(limit, risk, decision, model)),
        "transactions": transaction_store.get_recent(limit, risk, decision, model),
    }


@app.get("/transactions/{tx_id}")
def get_transaction(tx_id: str):
    tx = transaction_store.get_by_id(tx_id)
    if not tx:
        raise HTTPException(status_code=404, detail=f"Transaction {tx_id} not found.")
    return tx


@app.get("/transactions/stats/summary")
def transaction_stats():
    return transaction_store.summary()


# ---------------------------------------------------------------------------
# Model Lifecycle, Rollout & Release Gates Endpoints
# ---------------------------------------------------------------------------
@app.post("/models/rollout-mode")
def set_rollout_mode(req: RolloutModeRequest):
    try:
        return rollout.set_mode(req.mode, req.canary_percent)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.post("/models/evaluate-candidate")
def evaluate_candidate():
    cand_metrics = metadata.get("models", {}).get("challenger", {}).get("test_metrics", {})
    return rollout.evaluate_candidate_gates(
        candidate_metrics=cand_metrics,
        expected_features=features,
        actual_features=features,
        candidate_avg_latency_ms=rollout.status()["avg_candidate_latency_ms"] or 2.5,
    )


@app.post("/models/promote")
def promote_candidate(req: PromoteRequest):
    cand_metrics = metadata.get("models", {}).get("challenger", {}).get("test_metrics", {})
    gate_eval = rollout.evaluate_candidate_gates(
        candidate_metrics=cand_metrics,
        expected_features=features,
        actual_features=features,
        candidate_avg_latency_ms=rollout.status()["avg_candidate_latency_ms"] or 2.5,
    )

    if not gate_eval["passed"] and not req.force:
        raise HTTPException(
            status_code=412,
            detail={
                "message": "Promotion blocked: Candidate model failed one or more release gates.",
                "gate_evaluation": gate_eval,
                "hint": "Set force=true to override with administrative bypass.",
            }
        )

    # Swap champion and challenger safely
    old_champion = models["champion"]
    models["champion"] = models["challenger"]
    models["challenger"] = old_champion

    old_champ_thresh = thresholds.get("champion", 0.5)
    thresholds["champion"] = thresholds.get("challenger", 0.5)
    thresholds["challenger"] = old_champ_thresh

    rollout.record_promotion(req.candidate_name, req.reason)
    PROMOTIONS.inc()

    return {
        "status": "promoted",
        "message": f"Candidate '{req.candidate_name}' promoted to active champion successfully.",
        "active_champion_threshold": thresholds["champion"],
        "rollout": rollout.status(),
    }


@app.post("/models/rollback")
def rollback_model(req: RollbackRequest):
    return rollout.manual_rollback(reason=req.reason)


@app.post("/models/rollback/reset")
def reset_rollback():
    return rollout.reset_rollback()


# ---------------------------------------------------------------------------
# Voice & Email Scam Analysis Endpoints
# ---------------------------------------------------------------------------
@app.post("/scam/analyze-text")
def analyze_scam_text(req: ScamTextRequest):
    return scam_analyzer.analyze(req.text, source_type=req.source_type)


@app.get("/scam/samples")
def get_scam_samples():
    return SAMPLE_SCENARIOS


# ---------------------------------------------------------------------------
# Prometheus Metrics Endpoint
# ---------------------------------------------------------------------------
@app.get("/metrics")
def metrics():
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)


# ---------------------------------------------------------------------------
# Mount Static Frontend (if built)
# ---------------------------------------------------------------------------
frontend_dist = ROOT_DIR / "frontend" / "dist"
if frontend_dist.exists():
    app.mount("/dashboard", StaticFiles(directory=str(frontend_dist), html=True), name="frontend")
