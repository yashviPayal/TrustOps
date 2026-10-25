#!/usr/bin/env python3
"""
TrustOps Live Transaction Replay CLI Script.

Streams transactions into the TrustOps prediction API with configurable scenarios,
speeds, and batch sizes. Demonstrates realistic traffic replay, fraud attack scenarios,
feature drift, and concept drift.
"""

import argparse
import json
import random
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import requests

ROOT_DIR = Path(__file__).resolve().parents[1]
HELD_OUT_PATH = ROOT_DIR / "data" / "processed" / "held_out_stream.json"


def load_dataset(path: Path) -> list[dict]:
    if path.exists():
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except Exception as e:
            print(f"[Warning] Failed to read {path}: {e}. Falling back to synthetic generator.")
    return generate_synthetic_pool()


def generate_synthetic_pool(count: int = 500) -> list[dict]:
    rng = random.Random(42)
    pool = []
    for i in range(count):
        is_fraud = (i % 20 == 0)
        row = {f"V{j}": rng.gauss(0.0, 1.0) for j in range(1, 29)}
        if is_fraud:
            row["V14"] = rng.gauss(-5.0, 1.5)
            row["V10"] = rng.gauss(-4.0, 1.5)
            row["Amount"] = round(rng.uniform(50.0, 600.0), 2)
            row["Class"] = 1
        else:
            row["V14"] = rng.gauss(0.2, 0.8)
            row["V10"] = rng.gauss(0.1, 0.8)
            row["Amount"] = round(rng.expovariate(1 / 45.0) + 1.0, 2)
            row["Class"] = 0
        pool.append(row)
    return pool


def prepare_transaction(record: dict, scenario: str) -> tuple[dict, int]:
    rec = dict(record)
    true_label = int(rec.pop("Class", 0))
    features = {k: float(v) for k, v in rec.items() if k != "Class"}

    if scenario == "feature_drift":
        # Shift numeric distributions to trigger data drift detection
        features["V1"] = features.get("V1", 0.0) + 2.5
        features["V2"] = features.get("V2", 0.0) - 2.0
        features["V3"] = features.get("V3", 0.0) + 1.8
        features["V4"] = features.get("V4", 0.0) - 2.2
        features["Amount"] = round(features.get("Amount", 50.0) * 3.5, 2)

    elif scenario == "concept_drift":
        # Invert fraud features: fraud masquerades as benign traffic
        if true_label == 1:
            features["V14"] = 1.2
            features["V10"] = 0.8
            features["V12"] = 0.5
            features["Amount"] = 19.99

    return features, true_label


def main():
    parser = argparse.ArgumentParser(description="TrustOps Transaction Streaming / Replay CLI")
    parser.add_argument("--url", default="http://127.0.0.1:8000", help="FastAPI base URL")
    parser.add_argument(
        "--scenario",
        default="baseline",
        choices=["baseline", "elevated_fraud", "feature_drift", "concept_drift"],
        help="Replay scenario (baseline | elevated_fraud | feature_drift | concept_drift)"
    )
    parser.add_argument("--speed", type=float, default=5.0, help="Transactions per second")
    parser.add_argument("--count", type=int, default=50, help="Total transactions to stream (0 for continuous)")
    parser.add_argument("--batch-size", type=int, default=1, help="Transactions per request (1-100)")
    args = parser.parse_args()

    api_url = args.url.rstrip("/")
    predict_endpoint = f"{api_url}/predict" if args.batch_size == 1 else f"{api_url}/predict/batch"

    print("=" * 78)
    print(" TrustOps Live Transaction Streamer")
    print(f" Target API: {api_url}")
    print(f" Scenario:   {args.scenario.upper()} (Data: Held-out ULB test dataset / synthetic)")
    print(f" Speed:      {args.speed} tx/sec | Batch: {args.batch_size} | Count: {args.count or 'Unlimited'}")
    print("=" * 78)

    # Check API health
    try:
        resp = requests.get(f"{api_url}/health", timeout=3.0)
        resp.raise_for_status()
        health_data = resp.json()
        print(f"[API Status] {health_data.get('status')} | Rollout: {health_data.get('rollout_mode')}")
    except Exception as e:
        print(f"[Error] Failed to connect to API at {api_url}: {e}")
        print("Ensure the API is running (e.g. `uvicorn app.api.main:app --port 8000`) before streaming.")
        sys.exit(1)

    records = load_dataset(HELD_OUT_PATH)
    fraud_pool = [r for r in records if r.get("Class") == 1]
    legit_pool = [r for r in records if r.get("Class") == 0]
    print(f"[Dataset] {len(records)} transactions loaded ({len(fraud_pool)} frauds in pool).\n")

    sent_count = 0
    alert_count = 0
    total_latencies = []
    start_time = time.time()

    interval = 1.0 / max(0.5, args.speed)

    try:
        while True:
            if args.count > 0 and sent_count >= args.count:
                break

            loop_start = time.perf_counter()

            if args.batch_size == 1:
                # Single prediction
                if args.scenario == "elevated_fraud" and random.random() < 0.20 and fraud_pool:
                    rec = random.choice(fraud_pool)
                else:
                    rec = random.choice(records)

                features, true_label = prepare_transaction(rec, args.scenario)
                payload = {"transaction": features}

                t0 = time.perf_counter()
                res = requests.post(predict_endpoint, json=payload, timeout=5.0)
                lat_ms = (time.perf_counter() - t0) * 1000
                total_latencies.append(lat_ms)

                if res.status_code == 200:
                    data = res.json()
                    prob = data.get("fraud_probability", 0.0)
                    decision = data.get("decision", "allow")
                    risk = data.get("risk", "low")
                    model_ver = data.get("model_version", "unknown")
                    sent_count += 1

                    if risk in {"elevated", "high"}:
                        alert_count += 1
                        risk_display = f"\033[91m[{risk.upper()}]\033[0m"
                    else:
                        risk_display = f"\033[92m[{risk.upper()}]\033[0m"

                    label_tag = f"[True: {'FRAUD' if true_label==1 else 'NORMAL'}]"
                    print(
                        f"Tx #{sent_count:04d} | Model: {model_ver:10s} | Prob: {prob:0.6f} | "
                        f"Decision: {decision:6s} | Risk: {risk_display:14s} | Latency: {lat_ms:5.1f}ms | {label_tag}"
                    )
                else:
                    print(f"[Error] Request failed: {res.status_code} - {res.text}")

            else:
                # Batch prediction
                batch = []
                for _ in range(args.batch_size):
                    rec = random.choice(records)
                    features, _ = prepare_transaction(rec, args.scenario)
                    batch.append(features)

                t0 = time.perf_counter()
                res = requests.post(predict_endpoint, json={"transactions": batch}, timeout=10.0)
                lat_ms = (time.perf_counter() - t0) * 1000
                total_latencies.append(lat_ms)

                if res.status_code == 200:
                    data = res.json()
                    preds = data.get("predictions", [])
                    sent_count += len(preds)
                    high_risks = sum(1 for p in preds if p.get("risk") in {"elevated", "high"})
                    alert_count += high_risks
                    print(f"Batch #{sent_count // args.batch_size} ({len(preds)} tx) | High Risk: {high_risks} | Latency: {lat_ms:5.1f}ms")
                else:
                    print(f"[Error] Batch failed: {res.status_code} - {res.text}")

            elapsed = time.perf_counter() - loop_start
            sleep_time = max(0.005, interval - elapsed)
            time.sleep(sleep_time)

    except KeyboardInterrupt:
        print("\n[Stopped by user]")

    total_time = round(time.time() - start_time, 2)
    avg_lat = round(sum(total_latencies) / len(total_latencies), 2) if total_latencies else 0.0

    print("\n" + "=" * 78)
    print(" Stream Summary")
    print(f" Total Transactions Sent: {sent_count}")
    print(f" Risk Alerts Raised:      {alert_count}")
    print(f" Total Duration:          {total_time}s")
    print(f" Average Latency:         {avg_lat}ms")
    print("=" * 78)


if __name__ == "__main__":
    main()
