import json
import math
from collections import deque
from datetime import datetime, timezone
from pathlib import Path
from threading import RLock
from typing import Any

import numpy as np
from prometheus_client import Counter, Histogram, Gauge
from scipy.stats import ks_2samp
from sklearn.metrics import (
    average_precision_score, precision_score, recall_score, f1_score, confusion_matrix
)

# ---------------------------------------------------------------------------
# Prometheus Metrics
# ---------------------------------------------------------------------------
PREDICTIONS = Counter(
    "trustops_predictions_total",
    "Predictions served",
    ["model", "decision", "risk"]
)
PREDICTION_ERRORS = Counter(
    "trustops_prediction_errors_total",
    "Prediction errors"
)
PREDICTION_LATENCY = Histogram(
    "trustops_prediction_latency_seconds",
    "Prediction latency seconds",
    buckets=(0.002, 0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.0, 5.0)
)
INPUT_DRIFT_PSI = Gauge(
    "trustops_input_drift_psi",
    "Maximum PSI among monitored features"
)
INPUT_DRIFT_KS_STAT = Gauge(
    "trustops_input_drift_ks_stat",
    "Maximum KS statistic among monitored features"
)
DRIFT_EVALUATIONS = Counter(
    "trustops_drift_evaluations_total",
    "Drift evaluation counts",
    ["status"]
)
CANARY_ROLLBACKS = Counter(
    "trustops_canary_rollbacks_total",
    "Automatic canary rollbacks",
    ["reason"]
)
PROMOTIONS = Counter(
    "trustops_promotions_total",
    "Model promotions recorded"
)
DELAYED_LABEL_F1 = Gauge(
    "trustops_delayed_label_f1",
    "F1 score on delayed ground truth labels"
)
DELAYED_LABEL_RECALL = Gauge(
    "trustops_delayed_label_recall",
    "Recall score on delayed ground truth labels"
)
DELAYED_LABEL_PRECISION = Gauge(
    "trustops_delayed_label_precision",
    "Precision score on delayed ground truth labels"
)


# ---------------------------------------------------------------------------
# Data Drift Monitor (KS-Test + PSI + Bonferroni Correction)
# ---------------------------------------------------------------------------
class InputMonitor:
    def __init__(
        self,
        reference_path: Path,
        window_size: int = 500,
        min_samples: int = 50,
        ks_alpha: float = 0.05,
        psi_threshold: float = 0.20,
        effect_threshold: float = 0.15,
    ):
        self.reference_path = reference_path
        self.window_size = window_size
        self.min_samples = min_samples
        self.ks_alpha = ks_alpha
        self.psi_threshold = psi_threshold
        self.effect_threshold = effect_threshold

        self._rows = deque(maxlen=window_size)
        self._history = deque(maxlen=60)
        self._lock = RLock()
        self.reference: dict[str, Any] = {}
        self.load_reference()

    def load_reference(self) -> None:
        if self.reference_path.exists():
            try:
                self.reference = json.loads(self.reference_path.read_text(encoding="utf-8"))
            except Exception:
                self.reference = {}

    def add(self, row: dict[str, Any]) -> None:
        with self._lock:
            cleaned = {}
            for k, v in row.items():
                if v is not None:
                    try:
                        cleaned[k] = float(v)
                    except (ValueError, TypeError):
                        continue
            if cleaned:
                self._rows.append(cleaned)

    def reset(self) -> None:
        with self._lock:
            self._rows.clear()
            self._history.clear()
            INPUT_DRIFT_PSI.set(0.0)
            INPUT_DRIFT_KS_STAT.set(0.0)

    @staticmethod
    def _psi(expected: list[float], actual: list[float]) -> float | None:
        eps = 1e-6
        total_e, total_a = sum(expected), sum(actual)
        if total_e <= 0 or total_a <= 0:
            return None
        value = 0.0
        for e, a in zip(expected, actual):
            e_norm = max(e / total_e, eps)
            a_norm = max(a / total_a, eps)
            value += (a_norm - e_norm) * math.log(a_norm / e_norm)
        return float(value)

    def report(self) -> dict[str, Any]:
        with self._lock:
            rows = list(self._rows)

        if not self.reference:
            return {
                "status": "reference_missing",
                "samples": len(rows),
                "required": self.min_samples,
                "window_size": self.window_size,
                "features": {},
                "alerts": ["Reference training distribution missing or corrupted."],
            }

        if len(rows) < self.min_samples:
            return {
                "status": "collecting",
                "samples": len(rows),
                "required": self.min_samples,
                "window_size": self.window_size,
                "features": {},
                "alerts": [],
            }

        feature_names = [f for f in self.reference.keys() if f in rows[0]]
        num_features = max(1, len(feature_names))
        # Bonferroni multiple comparisons adjustment
        bonferroni_alpha = self.ks_alpha / num_features

        feature_results: dict[str, Any] = {}
        drifted_features: list[str] = []
        alerts: list[str] = []
        max_psi = 0.0
        max_ks = 0.0

        for feature in feature_names:
            ref = self.reference[feature]
            values = [r[feature] for r in rows if feature in r and not math.isnan(r[feature])]
            if len(values) < 15:
                continue

            window_arr = np.array(values, dtype=float)
            ref_sample = np.array(ref.get("sample", []), dtype=float)

            # 1. Kolmogorov-Smirnov 2-sample test
            ks_stat = 0.0
            p_value = 1.0
            if len(ref_sample) >= 10:
                try:
                    ks_res = ks_2samp(ref_sample, window_arr)
                    ks_stat = float(ks_res.statistic)
                    p_value = float(ks_res.pvalue)
                except Exception:
                    ks_stat, p_value = 0.0, 1.0

            # 2. Population Stability Index (PSI)
            edges = ref.get("edges", [])
            expected = ref.get("proportions", [])
            psi_val = 0.0
            if len(edges) >= 2 and expected:
                bin_indices = np.digitize(window_arr, edges[1:-1])
                counts = np.bincount(bin_indices, minlength=len(expected))[:len(expected)].tolist()
                psi_calc = self._psi(expected, counts)
                if psi_calc is not None:
                    psi_val = psi_calc

            # 3. Practical effect size (Normalized mean shift)
            ref_mean = float(ref.get("mean", 0.0))
            ref_std = float(ref.get("std", 1.0))
            if ref_std < 1e-6:
                ref_std = 1.0
            mean_shift = abs(float(np.mean(window_arr)) - ref_mean) / ref_std

            # Determine whether feature is meaningfully drifted:
            # Significant KS test with practical effect size OR high PSI with statistical significance
            is_drifted = (
                (p_value < bonferroni_alpha and ks_stat >= self.effect_threshold)
                or (psi_val >= self.psi_threshold and p_value < self.ks_alpha)
            )

            if is_drifted:
                drifted_features.append(feature)

            max_psi = max(max_psi, psi_val)
            max_ks = max(max_ks, ks_stat)

            feature_results[feature] = {
                "ks_statistic": round(ks_stat, 4),
                "p_value": round(p_value, 6),
                "psi": round(psi_val, 4),
                "mean_shift": round(mean_shift, 4),
                "window_mean": round(float(np.mean(window_arr)), 4),
                "reference_mean": round(ref_mean, 4),
                "drift_detected": is_drifted,
            }

        overall_status = "drift_detected" if drifted_features else "healthy"
        if drifted_features:
            alerts.append(
                f"Feature distribution drift detected in {len(drifted_features)} features: "
                f"{', '.join(drifted_features[:5])}{'...' if len(drifted_features) > 5 else ''}"
            )

        INPUT_DRIFT_PSI.set(max_psi)
        INPUT_DRIFT_KS_STAT.set(max_ks)
        DRIFT_EVALUATIONS.labels(status=overall_status).inc()

        eval_time = datetime.now(timezone.utc).isoformat()
        report_data = {
            "status": overall_status,
            "timestamp": eval_time,
            "samples": len(rows),
            "window_size": self.window_size,
            "required": self.min_samples,
            "reference_size": len(self.reference.get("V1", {}).get("sample", [])),
            "alpha_nominal": self.ks_alpha,
            "alpha_bonferroni": round(bonferroni_alpha, 6),
            "psi_threshold": self.psi_threshold,
            "effect_threshold": self.effect_threshold,
            "max_psi": round(max_psi, 4),
            "max_ks_statistic": round(max_ks, 4),
            "drifted_feature_count": len(drifted_features),
            "drifted_features": drifted_features,
            "alerts": alerts,
            "features": feature_results,
        }

        with self._lock:
            self._history.append({
                "timestamp": eval_time,
                "status": overall_status,
                "samples": len(rows),
                "max_psi": round(max_psi, 4),
                "max_ks": round(max_ks, 4),
                "drifted_count": len(drifted_features),
            })

        return report_data

    def get_history(self) -> list[dict[str, Any]]:
        with self._lock:
            return list(self._history)


# ---------------------------------------------------------------------------
# Concept Drift & Delayed Labels Monitor
# ---------------------------------------------------------------------------
class DelayedPerformanceMonitor:
    def __init__(
        self,
        window_size: int = 100,
        f1_alert_threshold: float = 0.60,
    ):
        self.window_size = window_size
        self.f1_alert_threshold = f1_alert_threshold
        # Stores predictions waiting for delayed labels: tx_id -> {prob, decision, threshold, model, time}
        self._pending_predictions: dict[str, dict[str, Any]] = {}
        # Stores resolved labeled predictions in a sliding window
        self._resolved_window: deque[dict[str, Any]] = deque(maxlen=window_size)
        self._history = deque(maxlen=60)
        self._lock = RLock()

    def record_prediction(
        self,
        transaction_id: str,
        fraud_probability: float,
        decision: str,
        threshold: float,
        model_version: str,
    ) -> None:
        with self._lock:
            self._pending_predictions[transaction_id] = {
                "transaction_id": transaction_id,
                "fraud_probability": float(fraud_probability),
                "predicted_class": 1 if decision == "review" or fraud_probability >= threshold else 0,
                "decision": decision,
                "threshold": float(threshold),
                "model_version": model_version,
                "timestamp": datetime.now(timezone.utc).isoformat(),
            }
            # Prevent unbounded memory growth if some labels never arrive
            if len(self._pending_predictions) > 2000:
                # Remove oldest keys
                oldest_keys = list(self._pending_predictions.keys())[:500]
                for k in oldest_keys:
                    self._pending_predictions.pop(k, None)

    def record_label(self, transaction_id: str, true_label: int) -> bool:
        with self._lock:
            pred = self._pending_predictions.pop(transaction_id, None)
            if pred is None:
                return False
            pred["true_label"] = int(true_label)
            pred["resolved_at"] = datetime.now(timezone.utc).isoformat()
            self._resolved_window.append(pred)
            return True

    def reset(self) -> None:
        with self._lock:
            self._pending_predictions.clear()
            self._resolved_window.clear()
            self._history.clear()
            DELAYED_LABEL_F1.set(0.0)
            DELAYED_LABEL_RECALL.set(0.0)
            DELAYED_LABEL_PRECISION.set(0.0)

    def report(self) -> dict[str, Any]:
        with self._lock:
            resolved = list(self._resolved_window)
            pending_count = len(self._pending_predictions)

        if len(resolved) < 15:
            return {
                "status": "collecting_labels",
                "resolved_samples": len(resolved),
                "pending_labels": pending_count,
                "required_samples": 15,
                "metrics": {},
                "alerts": [],
            }

        y_true = np.array([r["true_label"] for r in resolved], dtype=int)
        y_pred = np.array([r["predicted_class"] for r in resolved], dtype=int)
        probabilities = np.array([r["fraud_probability"] for r in resolved], dtype=float)

        # Check if both classes are present
        has_positive = int(np.sum(y_true)) > 0
        prec = float(precision_score(y_true, y_pred, zero_division=0))
        rec = float(recall_score(y_true, y_pred, zero_division=0))
        f1 = float(f1_score(y_true, y_pred, zero_division=0))

        pr_auc = None
        if has_positive and len(np.unique(y_true)) > 1:
            try:
                pr_auc = float(average_precision_score(y_true, probabilities))
            except Exception:
                pr_auc = None

        cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
        tn, fp, fn, tp = int(cm[0, 0]), int(cm[0, 1]), int(cm[1, 0]), int(cm[1, 1])

        DELAYED_LABEL_F1.set(f1)
        DELAYED_LABEL_RECALL.set(rec)
        DELAYED_LABEL_PRECISION.set(prec)

        alerts: list[str] = []
        is_deteriorated = False
        if has_positive and len(resolved) >= 20:
            if f1 < self.f1_alert_threshold:
                is_deteriorated = True
                alerts.append(
                    f"Concept drift / performance deterioration: Delayed F1 score ({f1:.3f}) "
                    f"is below threshold ({self.f1_alert_threshold:.2f})."
                )
            if rec < 0.60:
                alerts.append(f"High false negative rate: Recall has dropped to {rec:.3f}.")

        status = "performance_deteriorated" if is_deteriorated else "healthy"
        now_iso = datetime.now(timezone.utc).isoformat()

        metrics_data = {
            "precision": round(prec, 4),
            "recall": round(rec, 4),
            "f1": round(f1, 4),
            "pr_auc": round(pr_auc, 4) if pr_auc is not None else None,
            "tp": tp,
            "fp": fp,
            "tn": tn,
            "fn": fn,
            "true_fraud_count": int(np.sum(y_true)),
            "true_legit_count": int(len(y_true) - np.sum(y_true)),
        }

        with self._lock:
            self._history.append({
                "timestamp": now_iso,
                "status": status,
                "samples": len(resolved),
                "f1": round(f1, 4),
                "recall": round(rec, 4),
                "precision": round(prec, 4),
            })

        return {
            "status": status,
            "timestamp": now_iso,
            "resolved_samples": len(resolved),
            "window_size": self.window_size,
            "pending_labels": pending_count,
            "f1_alert_threshold": self.f1_alert_threshold,
            "metrics": metrics_data,
            "alerts": alerts,
        }

    def get_history(self) -> list[dict[str, Any]]:
        with self._lock:
            return list(self._history)
