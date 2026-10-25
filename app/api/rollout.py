import random
import time
from collections import deque
from datetime import datetime, timezone
from threading import RLock
from typing import Any

class RolloutManager:
    def __init__(
        self,
        mode: str = "champion",
        canary_percent: int = 10,
        max_latency_ms: float = 500.0,
        latency_failures: int = 5,
        error_failures: int = 3,
        gate_min_pr_auc: float = 0.65,
        gate_min_precision: float = 0.50,
        gate_min_recall: float = 0.70,
        gate_max_latency_ms: float = 50.0,
    ):
        if mode not in {"champion", "shadow", "canary"}:
            mode = "champion"
        self.mode = mode
        self.canary_percent = max(0, min(100, int(canary_percent)))
        self.max_latency_ms = float(max_latency_ms)
        self.latency_failures = max(1, int(latency_failures))
        self.error_failures = max(1, int(error_failures))

        # Default release gate criteria
        self.gate_rules = {
            "min_pr_auc": gate_min_pr_auc,
            "min_precision": gate_min_precision,
            "min_recall": gate_min_recall,
            "max_latency_ms": gate_max_latency_ms,
        }

        self._slow_canary_calls = 0
        self._consecutive_errors = 0
        self._rolled_back = False
        self._rollback_reason = None
        self._lock = RLock()

        self._candidate_latencies = deque(maxlen=100)
        self._champion_latencies = deque(maxlen=100)
        self._audit_log = deque(maxlen=50)

        self._record_audit_event(
            event="initialized",
            details=f"Rollout manager initialized in {self.mode} mode (canary: {self.canary_percent}%)"
        )

    def _record_audit_event(self, event: str, details: str, meta: dict[str, Any] | None = None) -> None:
        self._audit_log.append({
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "event": event,
            "details": details,
            "mode": self.mode,
            "canary_percent": self.canary_percent,
            "meta": meta or {},
        })

    def set_mode(self, mode: str, canary_percent: int | None = None) -> dict[str, Any]:
        with self._lock:
            if mode not in {"champion", "shadow", "canary"}:
                raise ValueError(f"Invalid rollout mode: {mode}. Must be champion, shadow, or canary.")
            old_mode = self.mode
            self.mode = mode
            if canary_percent is not None:
                self.canary_percent = max(0, min(100, int(canary_percent)))
            # When manually configuring mode, reset rollback state
            self._rolled_back = False
            self._rollback_reason = None
            self._slow_canary_calls = 0
            self._consecutive_errors = 0

            self._record_audit_event(
                event="mode_changed",
                details=f"Mode changed from {old_mode} to {self.mode} ({self.canary_percent}% canary)"
            )
            return self.status()

    def choose_model(self) -> str:
        with self._lock:
            if self.mode == "canary" and not self._rolled_back:
                if random.random() * 100 < self.canary_percent:
                    return "challenger"
            return "champion"

    def record_candidate_latency(self, latency_ms: float) -> None:
        """Backward-compatible alias for recording challenger candidate latency."""
        self.record_prediction_success("challenger", latency_ms)

    def record_prediction_success(self, model: str, latency_ms: float) -> None:
        with self._lock:
            if model == "challenger":
                self._candidate_latencies.append(float(latency_ms))
                if latency_ms > self.max_latency_ms:
                    self._slow_canary_calls += 1
                else:
                    self._slow_canary_calls = 0

                if self._slow_canary_calls >= self.latency_failures and not self._rolled_back:
                    self._trigger_rollback_locked(
                        reason=f"Candidate latency exceeded {self.max_latency_ms}ms for {self._slow_canary_calls} consecutive requests"
                    )
            else:
                self._champion_latencies.append(float(latency_ms))
            self._consecutive_errors = 0

    def record_prediction_error(self, model: str) -> None:
        with self._lock:
            if model == "challenger":
                self._consecutive_errors += 1
                if self._consecutive_errors >= self.error_failures and not self._rolled_back:
                    self._trigger_rollback_locked(
                        reason=f"Candidate encountered {self._consecutive_errors} consecutive errors"
                    )

    def trigger_performance_rollback(self, current_metric: float, threshold: float, metric_name: str = "F1") -> None:
        with self._lock:
            if not self._rolled_back:
                self._trigger_rollback_locked(
                    reason=f"Candidate {metric_name} ({current_metric:.3f}) dropped below safeguard threshold ({threshold:.2f})"
                )

    def _trigger_rollback_locked(self, reason: str) -> None:
        self._rolled_back = True
        self._rollback_reason = reason
        self._record_audit_event(
            event="automatic_rollback",
            details=f"Canary rolled back to champion. Reason: {reason}",
            meta={"trigger_reason": reason}
        )

    def manual_rollback(self, reason: str = "Manual operator rollback") -> dict[str, Any]:
        with self._lock:
            self._trigger_rollback_locked(reason=reason)
            return self.status()

    def reset_rollback(self) -> dict[str, Any]:
        with self._lock:
            self._rolled_back = False
            self._rollback_reason = None
            self._slow_canary_calls = 0
            self._consecutive_errors = 0
            self._record_audit_event(
                event="rollback_reset",
                details="Rollback flag cleared by operator"
            )
            return self.status()

    def evaluate_candidate_gates(
        self,
        candidate_metrics: dict[str, Any],
        expected_features: list[str],
        actual_features: list[str],
        candidate_avg_latency_ms: float | None = None,
    ) -> dict[str, Any]:
        """
        Evaluate candidate model against strict release gates before promotion.
        """
        gates = {}
        all_passed = True

        # Gate 1: Schema compatibility
        schema_passed = (set(expected_features) == set(actual_features)) and (len(expected_features) == len(actual_features))
        gates["schema_compatibility"] = {
            "required_feature_count": len(expected_features),
            "actual_feature_count": len(actual_features),
            "schema_match": schema_passed,
            "passed": schema_passed,
        }
        if not schema_passed:
            all_passed = False

        # Gate 2: PR-AUC (Average Precision)
        pr_auc = float(candidate_metrics.get("average_precision", 0.0))
        target_pr_auc = self.gate_rules["min_pr_auc"]
        pr_passed = pr_auc >= target_pr_auc
        gates["min_pr_auc"] = {
            "target": target_pr_auc,
            "actual": round(pr_auc, 4),
            "passed": pr_passed,
        }
        if not pr_passed:
            all_passed = False

        # Gate 3: Precision
        prec = float(candidate_metrics.get("precision", 0.0))
        target_prec = self.gate_rules["min_precision"]
        prec_passed = prec >= target_prec
        gates["min_precision"] = {
            "target": target_prec,
            "actual": round(prec, 4),
            "passed": prec_passed,
        }
        if not prec_passed:
            all_passed = False

        # Gate 4: Recall
        rec = float(candidate_metrics.get("recall", 0.0))
        target_rec = self.gate_rules["min_recall"]
        rec_passed = rec >= target_rec
        gates["min_recall"] = {
            "target": target_rec,
            "actual": round(rec, 4),
            "passed": rec_passed,
        }
        if not rec_passed:
            all_passed = False

        # Gate 5: Latency budget
        actual_latency = candidate_avg_latency_ms if candidate_avg_latency_ms is not None else 5.0
        target_lat = self.gate_rules["max_latency_ms"]
        lat_passed = actual_latency <= target_lat
        gates["latency_budget_ms"] = {
            "budget_ms": target_lat,
            "actual_ms": round(actual_latency, 2),
            "passed": lat_passed,
        }
        if not lat_passed:
            all_passed = False

        reason = "Candidate passed all release gates." if all_passed else "Candidate failed one or more release gates."
        return {
            "passed": all_passed,
            "reason": reason,
            "evaluated_at": datetime.now(timezone.utc).isoformat(),
            "gates": gates,
        }

    def record_promotion(self, candidate_name: str, reason: str = "Passed release gates") -> None:
        with self._lock:
            self._record_audit_event(
                event="model_promoted",
                details=f"Candidate '{candidate_name}' promoted to active champion. Reason: {reason}",
                meta={"promoted_candidate": candidate_name, "reason": reason}
            )

    @property
    def rolled_back(self) -> bool:
        return self._rolled_back

    def status(self) -> dict[str, Any]:
        with self._lock:
            cand_lats = list(self._candidate_latencies)
            champ_lats = list(self._champion_latencies)
            avg_cand_lat = round(sum(cand_lats) / len(cand_lats), 2) if cand_lats else 0.0
            avg_champ_lat = round(sum(champ_lats) / len(champ_lats), 2) if champ_lats else 0.0

            return {
                "mode": self.mode,
                "canary_percent": self.canary_percent,
                "canary_rolled_back": self._rolled_back,
                "rollback_reason": self._rollback_reason,
                "candidate_latency_samples": len(cand_lats),
                "avg_candidate_latency_ms": avg_cand_lat,
                "avg_champion_latency_ms": avg_champ_lat,
                "max_candidate_latency_ms": self.max_latency_ms,
                "consecutive_slow_calls": self._slow_canary_calls,
                "consecutive_errors": self._consecutive_errors,
                "gate_rules": self.gate_rules,
                "recent_audit_events": list(self._audit_log)[-10:],
            }
