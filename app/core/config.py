import os
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[2]
MODEL_DIR = Path(os.getenv("MODEL_DIR", ROOT_DIR / "models"))
DATA_PATH = Path(os.getenv("DATA_PATH", ROOT_DIR / "data" / "raw" / "creditcard.csv"))
ROLLOUT_MODE = os.getenv("ROLLOUT_MODE", "champion").lower()
CANARY_PERCENT = max(0, min(100, int(os.getenv("CANARY_PERCENT", "10"))))
CANARY_MAX_LATENCY_MS = float(os.getenv("CANARY_MAX_LATENCY_MS", "500"))
CANARY_LATENCY_FAILURES = max(1, int(os.getenv("CANARY_LATENCY_FAILURES", "5")))
DRIFT_WINDOW_SIZE = max(50, int(os.getenv("DRIFT_WINDOW_SIZE", "500")))
