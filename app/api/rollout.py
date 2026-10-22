import random
import time
from collections import deque
from threading import Lock

class RolloutManager:
    def __init__(self, mode="champion", canary_percent=10, max_latency_ms=500, latency_failures=5):
        if mode not in {"champion", "shadow", "canary"}:
            mode = "champion"
        self.mode = mode
        self.canary_percent = max(0, min(100, int(canary_percent)))
        self.max_latency_ms = float(max_latency_ms)
        self.latency_failures = max(1, int(latency_failures))
        self._slow_canary_calls = 0
        self._rolled_back = False
        self._lock = Lock()
        self._candidate_latencies = deque(maxlen=50)

    def choose_model(self):
        with self._lock:
            if self.mode == "canary" and not self._rolled_back:
                if random.random() * 100 < self.canary_percent:
                    return "challenger"
            return "champion"

    def record_candidate_latency(self, latency_ms):
        with self._lock:
            self._candidate_latencies.append(float(latency_ms))
            if latency_ms > self.max_latency_ms:
                self._slow_canary_calls += 1
            else:
                self._slow_canary_calls = 0
            if self._slow_canary_calls >= self.latency_failures:
                self._rolled_back = True

    @property
    def rolled_back(self):
        return self._rolled_back

    def status(self):
        return {
            "mode": self.mode,
            "canary_percent": self.canary_percent,
            "canary_rolled_back": self.rolled_back,
            "candidate_latency_samples": len(self._candidate_latencies),
            "max_candidate_latency_ms": self.max_latency_ms,
        }
