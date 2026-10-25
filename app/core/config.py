import os
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[2]
MODEL_DIR = Path(os.getenv("MODEL_DIR", ROOT_DIR / "models"))
DATA_PATH = Path(os.getenv("DATA_PATH", ROOT_DIR / "data" / "raw" / "creditcard.csv"))
HELD_OUT_STREAM_PATH = Path(os.getenv("HELD_OUT_STREAM_PATH", ROOT_DIR / "data" / "processed" / "held_out_stream.json"))

ROLLOUT_MODE = os.getenv("ROLLOUT_MODE", "champion").lower()
CANARY_PERCENT = max(0, min(100, int(os.getenv("CANARY_PERCENT", "10"))))
CANARY_MAX_LATENCY_MS = float(os.getenv("CANARY_MAX_LATENCY_MS", "500"))
CANARY_LATENCY_FAILURES = max(1, int(os.getenv("CANARY_LATENCY_FAILURES", "5")))
CANARY_ERROR_FAILURES = max(1, int(os.getenv("CANARY_ERROR_FAILURES", "3")))

DRIFT_WINDOW_SIZE = max(50, int(os.getenv("DRIFT_WINDOW_SIZE", "500")))
DRIFT_MIN_SAMPLES = max(10, int(os.getenv("DRIFT_MIN_SAMPLES", "50")))
DRIFT_KS_ALPHA = float(os.getenv("DRIFT_KS_ALPHA", "0.05"))
DRIFT_PSI_THRESHOLD = float(os.getenv("DRIFT_PSI_THRESHOLD", "0.20"))
DRIFT_EFFECT_SIZE_THRESHOLD = float(os.getenv("DRIFT_EFFECT_SIZE_THRESHOLD", "0.15"))

PERFORMANCE_WINDOW_SIZE = max(20, int(os.getenv("PERFORMANCE_WINDOW_SIZE", "100")))
PERFORMANCE_F1_ALERT_THRESHOLD = float(os.getenv("PERFORMANCE_F1_ALERT_THRESHOLD", "0.60"))
