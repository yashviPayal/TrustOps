import json
import math
from collections import deque
from pathlib import Path
from threading import Lock

from prometheus_client import Counter, Histogram, Gauge

PREDICTIONS = Counter("trustops_predictions_total", "Predictions served", ["model", "decision"])
PREDICTION_ERRORS = Counter("trustops_prediction_errors_total", "Prediction errors")
PREDICTION_LATENCY = Histogram(
    "trustops_prediction_latency_seconds", "Prediction latency seconds",
    buckets=(0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1, 2, 5)
)
INPUT_DRIFT_PSI = Gauge("trustops_input_drift_psi", "Maximum PSI among monitored features")
CANARY_ROLLBACKS = Counter("trustops_canary_rollbacks_total", "Automatic canary rollbacks")

class InputMonitor:
    def __init__(self, reference_path: Path, window_size: int = 500):
        self.reference_path = reference_path
        self.window_size = window_size
        self._rows = deque(maxlen=window_size)
        self._lock = Lock()
        self.reference = {}
        if reference_path.exists():
            try:
                self.reference = json.loads(reference_path.read_text(encoding="utf-8"))
            except Exception:
                self.reference = {}

    def add(self, row: dict):
        with self._lock:
            self._rows.append({k: float(v) for k, v in row.items() if v is not None})

    @staticmethod
    def _psi(expected, actual):
        eps = 1e-6
        total_e, total_a = sum(expected), sum(actual)
        if total_e <= 0 or total_a <= 0:
            return None
        value = 0.0
        for e, a in zip(expected, actual):
            e = max(e / total_e, eps)
            a = max(a / total_a, eps)
            value += (a - e) * math.log(a / e)
        return float(value)

    def report(self):
        with self._lock:
            rows = list(self._rows)
        if not self.reference:
            return {"status": "reference_missing", "samples": len(rows), "features": {}}
        if len(rows) < min(50, self.window_size):
            return {"status": "collecting", "samples": len(rows), "required": min(50, self.window_size), "features": {}}
        results = {}
        for feature, ref in self.reference.items():
            values = [r[feature] for r in rows if feature in r]
            if len(values) < 10:
                continue
            edges = ref.get("edges", [])
            expected = ref.get("proportions", [])
            if len(edges) < 2 or not expected:
                continue
            # The reference bins may be collapsed when the training feature is
            # constant or heavily tied. Keep counts aligned to saved proportions.
            counts = [0] * len(expected)
            for value in values:
                idx = 0
                while idx < len(edges) - 2 and value > edges[idx + 1]:
                    idx += 1
                counts[min(idx, len(counts) - 1)] += 1
            psi = self._psi(expected, counts)
            if psi is not None:
                results[feature] = round(psi, 6)
        maximum = max(results.values(), default=0.0)
        INPUT_DRIFT_PSI.set(maximum)
        return {
            "status": "ok",
            "samples": len(rows),
            "max_psi": maximum,
            "interpretation": "PSI > 0.2 is a heuristic drift warning; investigate before acting.",
            "features": results,
        }
