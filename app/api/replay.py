import json
import random
import threading
import time
from collections import deque
from datetime import datetime, timezone
from pathlib import Path
from threading import RLock
from typing import Any, Callable

class TransactionStore:
    def __init__(self, max_size: int = 300):
        self._transactions: deque[dict[str, Any]] = deque(maxlen=max_size)
        self._index: dict[str, dict[str, Any]] = {}
        self._lock = RLock()

    def add(self, tx: dict[str, Any]) -> None:
        with self._lock:
            self._transactions.append(tx)
            self._index[tx["transaction_id"]] = tx
            # Keep index bounded
            if len(self._index) > 600:
                oldest_keys = list(self._index.keys())[:150]
                for k in oldest_keys:
                    self._index.pop(k, None)

    def resolve_label(self, tx_id: str, true_label: int) -> None:
        with self._lock:
            if tx_id in self._index:
                self._index[tx_id]["ground_truth"] = true_label
                self._index[tx_id]["label_status"] = "resolved"

    def get_recent(
        self,
        limit: int = 50,
        risk: str | None = None,
        decision: str | None = None,
        model: str | None = None,
    ) -> list[dict[str, Any]]:
        with self._lock:
            items = list(self._transactions)
        items.reverse()  # Newest first

        filtered = []
        for tx in items:
            pred = tx.get("prediction", {})
            if risk and pred.get("risk") != risk:
                continue
            if decision and pred.get("decision") != decision:
                continue
            if model and pred.get("model_version") != model:
                continue
            filtered.append(tx)
            if len(filtered) >= limit:
                break
        return filtered

    def get_by_id(self, tx_id: str) -> dict[str, Any] | None:
        with self._lock:
            return self._index.get(tx_id)

    def summary(self) -> dict[str, Any]:
        with self._lock:
            items = list(self._transactions)

        total = len(items)
        risk_counts = {"low": 0, "elevated": 0, "high": 0}
        decision_counts = {"allow": 0, "review": 0}
        model_counts: dict[str, int] = {}
        resolved_labels = 0
        fraud_labels = 0

        for tx in items:
            pred = tx.get("prediction", {})
            r = pred.get("risk", "low")
            d = pred.get("decision", "allow")
            m = pred.get("model_version", "unknown")
            risk_counts[r] = risk_counts.get(r, 0) + 1
            decision_counts[d] = decision_counts.get(d, 0) + 1
            model_counts[m] = model_counts.get(m, 0) + 1
            if tx.get("label_status") == "resolved":
                resolved_labels += 1
                if tx.get("ground_truth") == 1:
                    fraud_labels += 1

        return {
            "total_recorded": total,
            "risk_counts": risk_counts,
            "decision_counts": decision_counts,
            "model_counts": model_counts,
            "resolved_labels_count": resolved_labels,
            "confirmed_fraud_count": fraud_labels,
        }

    def clear(self) -> None:
        with self._lock:
            self._transactions.clear()
            self._index.clear()


class TransactionReplayEngine:
    def __init__(
        self,
        held_out_path: Path,
        store: TransactionStore,
        score_func: Callable[[dict[str, Any]], dict[str, Any]],
        on_prediction_hook: Callable[[str, float, str, float, str], None] | None = None,
        on_label_hook: Callable[[str, int], None] | None = None,
    ):
        self.held_out_path = held_out_path
        self.store = store
        self.score_func = score_func
        self.on_prediction_hook = on_prediction_hook
        self.on_label_hook = on_label_hook

        self.is_running = False
        self.is_paused = False
        self.scenario = "baseline"
        self.speed_tps = 5.0
        self.label_delay_steps = 10  # number of transactions before delayed label is revealed

        self.total_replayed = 0
        self.fraud_count = 0
        self.alert_count = 0
        self.start_time: float | None = None

        self._stop_event = threading.Event()
        self._pause_event = threading.Event()
        self._thread: threading.Thread | None = None

        self._records: list[dict[str, Any]] = []
        self._fraud_pool: list[dict[str, Any]] = []
        self._legit_pool: list[dict[str, Any]] = []
        self._pending_labels: deque[tuple[int, str, int]] = deque()
        self._lock = RLock()
        self._seq = 0

        self.load_dataset()

    def load_dataset(self) -> None:
        if self.held_out_path.exists():
            try:
                data = json.loads(self.held_out_path.read_text(encoding="utf-8"))
                self._records = data
                self._fraud_pool = [r for r in data if r.get("Class") == 1]
                self._legit_pool = [r for r in data if r.get("Class") == 0]
            except Exception:
                self._generate_fallback_pool()
        else:
            self._generate_fallback_pool()

    def _generate_fallback_pool(self) -> None:
        rng = random.Random(42)
        self._records = []
        self._legit_pool = []
        self._fraud_pool = []
        for i in range(500):
            is_fraud = (i % 25 == 0)
            row = {f"V{j}": rng.gauss(0.0, 1.0) for j in range(1, 29)}
            if is_fraud:
                row["V14"] = rng.gauss(-5.0, 1.5)
                row["V10"] = rng.gauss(-4.0, 1.5)
                row["Amount"] = round(rng.uniform(50.0, 500.0), 2)
                row["Class"] = 1
                self._fraud_pool.append(row)
            else:
                row["V14"] = rng.gauss(0.2, 0.8)
                row["V10"] = rng.gauss(0.1, 0.8)
                row["Amount"] = round(rng.expovariate(1 / 45.0) + 1.0, 2)
                row["Class"] = 0
                self._legit_pool.append(row)
            self._records.append(row)

    def _sample_next_record(self) -> tuple[dict[str, float], int]:
        if self.scenario == "elevated_fraud":
            if random.random() < 0.20 and self._fraud_pool:
                rec = dict(random.choice(self._fraud_pool))
            else:
                rec = dict(random.choice(self._legit_pool))
        else:
            rec = dict(random.choice(self._records))

        true_label = int(rec.pop("Class", 0))
        features = {k: float(v) for k, v in rec.items() if k != "Class"}

        if self.scenario == "feature_drift":
            features["V1"] = features.get("V1", 0.0) + 2.5
            features["V2"] = features.get("V2", 0.0) - 2.0
            features["V3"] = features.get("V3", 0.0) + 1.8
            features["V4"] = features.get("V4", 0.0) - 2.2
            features["Amount"] = round(features.get("Amount", 50.0) * 3.5, 2)

        elif self.scenario == "concept_drift":
            if true_label == 1:
                features["V14"] = 1.2
                features["V10"] = 0.8
                features["V12"] = 0.5
                features["Amount"] = 19.99

        return features, true_label

    def _run_loop(self) -> None:
        while not self._stop_event.is_set():
            if self._pause_event.is_set():
                if self._stop_event.wait(0.05):
                    break
                continue

            interval = 1.0 / max(0.5, self.speed_tps)
            t_start = time.perf_counter()

            try:
                self._step_once()
            except Exception:
                pass

            elapsed = time.perf_counter() - t_start
            sleep_time = max(0.005, interval - elapsed)
            if self._stop_event.wait(sleep_time):
                break

    def _step_once(self) -> dict[str, Any]:
        with self._lock:
            self._seq += 1
            seq = self._seq

        tx_id = f"tx-{datetime.now(timezone.utc).strftime('%Y%m%d%H%M%S')}-{seq:05d}"
        features, true_label = self._sample_next_record()
        now_iso = datetime.now(timezone.utc).isoformat()

        pred_start = time.perf_counter()
        prediction = self.score_func(features)
        pred_latency_ms = (time.perf_counter() - pred_start) * 1000

        risk = prediction.get("risk", "low")
        if risk in {"elevated", "high"}:
            self.alert_count += 1
        if true_label == 1:
            self.fraud_count += 1
        self.total_replayed += 1

        tx_record = {
            "transaction_id": tx_id,
            "timestamp": now_iso,
            "data_type": "replayed_held_out" if self.scenario == "baseline" else f"simulated_{self.scenario}",
            "scenario": self.scenario,
            "features": features,
            "prediction": prediction,
            "latency_ms": round(pred_latency_ms, 2),
            "ground_truth": None,
            "label_status": "pending",
        }
        self.store.add(tx_record)

        if self.on_prediction_hook:
            self.on_prediction_hook(
                tx_id,
                prediction.get("fraud_probability", 0.0),
                prediction.get("decision", "allow"),
                prediction.get("threshold", 0.5),
                prediction.get("model_version", "champion"),
            )

        resolve_at = seq + self.label_delay_steps
        self._pending_labels.append((resolve_at, tx_id, true_label))

        while self._pending_labels and self._pending_labels[0][0] <= seq:
            _, due_id, due_label = self._pending_labels.popleft()
            self.store.resolve_label(due_id, due_label)
            if self.on_label_hook:
                self.on_label_hook(due_id, due_label)

        return tx_record

    def start(
        self,
        scenario: str = "baseline",
        speed_tps: float = 5.0,
        label_delay_steps: int = 10,
    ) -> dict[str, Any]:
        with self._lock:
            if scenario not in {"baseline", "elevated_fraud", "feature_drift", "concept_drift"}:
                scenario = "baseline"
            self.scenario = scenario
            self.speed_tps = max(0.5, min(100.0, float(speed_tps)))
            self.label_delay_steps = max(1, int(label_delay_steps))
            self.is_running = True
            self.is_paused = False
            self.start_time = time.time()
            self._stop_event.clear()
            self._pause_event.clear()

            if self._thread is None or not self._thread.is_alive():
                self._thread = threading.Thread(target=self._run_loop, daemon=True)
                self._thread.start()

        return self.status()

    def pause(self) -> dict[str, Any]:
        self.is_paused = True
        self._pause_event.set()
        return self.status()

    def resume(self) -> dict[str, Any]:
        self.is_paused = False
        self._pause_event.clear()
        return self.status()

    def stop(self) -> dict[str, Any]:
        self.is_running = False
        self.is_paused = False
        self._stop_event.set()
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=1.0)
            self._thread = None
        return self.status()

    def status(self) -> dict[str, Any]:
        uptime_sec = round(time.time() - self.start_time, 1) if (self.is_running and self.start_time) else 0.0
        return {
            "is_running": self.is_running,
            "is_paused": self.is_paused,
            "scenario": self.scenario,
            "speed_tps": self.speed_tps,
            "label_delay_steps": self.label_delay_steps,
            "total_replayed": self.total_replayed,
            "fraud_count": self.fraud_count,
            "alert_count": self.alert_count,
            "uptime_seconds": uptime_sec,
            "dataset_available": len(self._records),
        }
