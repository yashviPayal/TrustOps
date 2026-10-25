# TrustOps: Automated MLOps Framework for Cybersecurity & Fraud Detection

TrustOps is a complete, production-grade MLOps demonstration of an end-to-end fraud detection and security lifecycle:
- **Benchmark ML Dataset**: Standard European ULB credit-card fraud dataset (imbalanced classification: 0.172% fraud)
- **Model Governance**: Imbalance-aware pipelines, validation-tuned threshold selection, and MLflow experiment tracking
- **Champion / Challenger**: Controlled model evaluation, automated release gates, and safe promotion
- **Inference & Rollout Service**: FastAPI supporting Champion-Only, Shadow Mode (dual scoring), and Canary Rollouts
- **Automated Rollback**: Safeguard triggers on latency spikes, error bursts, and delayed-label performance drops
- **Two-Factor Drift Monitoring**: Two-Sample Kolmogorov–Smirnov test (`ks_2samp`) + Population Stability Index (PSI) with Bonferroni multiple comparison correction
- **Concept Drift & Delayed Labels**: Delayed ground-truth label ingestion, online confusion matrix, and performance deterioration alerts
- **Live Transaction Replay**: Multi-scenario background replay engine (Baseline, Elevated Fraud Attack, Feature Drift, Concept Drift)
- **Cybersecurity Scam Extension**: NLP heuristic & threat pattern detector for voice transcripts and phishing emails
- **Operations Dashboard**: Modern React + TypeScript + Vite operations UI
- **Prometheus & Grafana**: Production observability metrics and pre-provisioned Grafana dashboards

> [!NOTE]
> **Operational Caveat:** This platform operates on historical anonymized benchmark data (PCA features `V1`–`V28` + `Amount`). Model predictions represent calibrated statistical risk estimates, not definitive legal proof of fraud.

---

## 1. System Architecture

```mermaid
flowchart TD
    subgraph Data Layer
        ULB["ULB Credit Card Dataset\n(creditcard.csv)"]
        HOLDOUT["Held-Out Test Stream\n(held_out_stream.json)"]
        SYNTH["Deterministic Synthetic Generator\n(Schema Compatible)"]
    end

    subgraph Replay & Simulation Engine
        REP["Transaction Replay Controller\n- Baseline (Normal)\n- Elevated Fraud (Surge)\n- Feature Drift (P(X) Shift)\n- Concept Drift (P(Y|X) Inversion)"]
        HOLDOUT --> REP
        SYNTH --> REP
    end

    subgraph FastAPI Inference & Governance Service
        REP -->|"HTTP POST /predict"| API["FastAPI Router\n(Port 8000)"]
        CLIENT["External REST Client / CLI"] -->|"HTTP POST /predict"| API
        
        API --> SCH["Schema & Input Validation\n(29 Features: V1..V28, Amount)"]
        
        subgraph Rollout Controller
            SCH --> ROLL["Rollout Manager\n(Champion / Shadow / Canary)"]
            ROLL -->|"Serve & Compare"| CHAMP["Champion Model\n(Pipeline: StandardScaler + LogReg C=1.0)"]
            ROLL -->|"Evaluate / Shadow"| CAND["Challenger Model\n(Pipeline: StandardScaler + LogReg C=0.1)"]
            
            GATES["Automated Release Gates\n- Min PR-AUC >= 0.65\n- Min Precision >= 0.50\n- Min Recall >= 0.70\n- Latency Budget <= 50ms\n- Schema Match"]
            ROLL --> GATES
            
            ROLLBACK["Auto-Rollback Safeguards\n- Canary Latency > 500ms\n- Error Burst >= 3\n- Performance F1 < 0.60"]
            ROLL --> ROLLBACK
        end

        subgraph Continuous Monitoring
            SCH --> DRIFT["Input Data Drift Monitor\n- Two-Sample KS Test (scipy)\n- Population Stability Index (PSI)\n- Bonferroni Multiple-Comparison Correction\n- Status: Healthy / Drift Detected"]
            REP -->|"Delayed Labels"| PERF["Concept Drift & Performance Monitor\n- Delayed Label Resolution\n- Confusion Matrix (TP, FP, TN, FN)\n- Online F1 / Recall / Precision Tracking"]
        end
        
        subgraph Security Extension
            SCAM["Voice & Phishing Scam Analyzer\n- Suspicious Payment Routing (Crypto, Wire)\n- Executive & Authority Impersonation\n- Urgency & Pressure Tactics\n- Credential & OTP Harvesting"]
        end
    end

    subgraph Observability
        API --> PROM["Prometheus Scraper\n(/metrics on Port 9090)"]
        PROM --> GRAF["Grafana Dashboards\n(Port 3000)"]
        CHAMP --> MLF["MLflow Tracking\n(Port 5000)"]
        CAND --> MLF
    end

    subgraph Frontend Operations Dashboard
        UI["React + TypeScript + Vite Dashboard\n(Port 8000/dashboard or Port 5173)\n- Live Transaction Table & Feature Drawer\n- Interactive Replay & Scenario Controls\n- Side-by-Side Model Rollout & Promotion\n- Feature-Level Drift Analytics Table\n- Concept Drift Alert Banner & Confusion Matrix\n- Scam Intelligence Playground"]
    end

    API --> UI
    DRIFT --> UI
    PERF --> UI
    ROLL --> UI
    SCAM --> UI
```

---

## 2. Repository Layout

```text
TrustOps/
├── app/
│   ├── api/
│   │   ├── main.py              # FastAPI application, routes, lifespan & static dashboard mount
│   │   ├── monitoring.py        # KS-test & PSI drift monitor + delayed performance tracker
│   │   ├── replay.py            # Background replay engine & bounded transaction store
│   │   ├── rollout.py           # Rollout manager, release gates & auto-rollback engine
│   │   └── schemas.py           # Pydantic request/response schemas & model contracts
│   ├── core/
│   │   └── config.py            # Centralized application & threshold configuration
│   ├── ml/
│   │   ├── data.py              # Dataset loading, column verification & stratified split
│   │   ├── metrics.py           # F1 threshold tuning & evaluation metrics (PR-AUC, ROC-AUC)
│   │   └── train.py             # Model training, MLflow logging, reference sample generation
│   └── scam/
│       ├── __init__.py
│       └── detector.py          # Voice transcript & phishing email threat analyzer
├── data/
│   ├── raw/
│   │   └── creditcard.csv       # Canonical ULB dataset
│   └── processed/
│       └── held_out_stream.json # Extracted held-out test split for realistic replay (2,574 records)
├── frontend/
│   ├── src/
│   │   ├── App.tsx              # Operations dashboard UI with full tabbed controls
│   │   ├── api.ts               # Typed client calling FastAPI endpoints
│   │   ├── types.ts             # TypeScript definitions matching backend schemas
│   │   ├── index.css            # Dark cybersecurity theme styling
│   │   └── main.tsx             # Vite entrypoint
│   ├── index.html
│   ├── package.json
│   ├── tsconfig.json
│   └── vite.config.ts           # Proxy to FastAPI port 8000
├── models/
│   ├── champion.joblib          # Active champion pipeline artifact
│   ├── challenger.joblib        # Challenger candidate pipeline artifact
│   ├── metadata.json            # Model lineage, validation metrics & feature schema
│   └── reference.json           # Fixed training baseline distributions (quantiles + samples)
├── monitoring/
│   ├── prometheus.yml           # Prometheus scrape configuration
│   └── grafana/
│       ├── dashboards/
│       │   └── trustops.json    # Pre-provisioned Grafana dashboard (10 metric panels)
│       └── provisioning/
│           ├── dashboards/dashboard.yml
│           └── datasources/datasource.yml
├── scripts/
│   ├── download_data.py         # OpenML dataset mirror downloader
│   └── stream_transactions.py   # CLI transaction streaming & scenario injection tool
├── tests/
│   ├── test_api.py              # Root & health route contract tests
│   ├── test_api_integration.py  # Full FastAPI integration tests across all endpoints
│   ├── test_delayed_labels.py   # Delayed label matching, confusion matrix & concept drift
│   ├── test_drift.py            # Two-sample KS test, PSI, Bonferroni & window reset tests
│   ├── test_metrics.py          # Threshold optimization & classification metrics
│   ├── test_replay.py           # Replay engine scenarios & transaction store CRUD
│   ├── test_rollout.py          # Shadow/canary, latency rollback, release gates & promotion
│   ├── test_scam.py             # Phishing & voice social engineering tests
│   └── test_training_smoke.py   # Deterministic CI training smoke test
├── .env.example                 # Environment variable templates
├── docker-compose.yml           # Complete containerized stack configuration
├── Dockerfile                   # Multi-stage Python + pre-built frontend container
├── pytest.ini                   # Pytest configuration (filters .venv and optimizes runs)
├── requirements.txt             # Pinned Python dependencies
└── README.md                    # Complete technical documentation
```

---

## 3. Local Setup (Windows 11 PowerShell)

### Prerequisites
- **Python 3.11 or 3.12** (`py --version`)
- **Node.js 18+ & npm** (`node --version`)
- **Docker Desktop** (optional for local container orchestration)

### Step 1: Virtual Environment & Python Dependencies

```powershell
# Open Windows PowerShell and enter the project directory
cd D:\projects\TrustOps\TrustOps

# Create a virtual environment
py -3.11 -m venv .venv

# Activate the virtual environment
.\.venv\Scripts\Activate.ps1
# If PowerShell script execution is restricted, run:
# Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass

# Upgrade pip and install pinned requirements
python -m pip install --upgrade pip
pip install -r requirements.txt
```

### Step 2: Build the Frontend Dashboard

```powershell
cd frontend
npm.cmd install
npm.cmd run build
cd ..
```
The built static dashboard files are placed in `frontend/dist/` and will be automatically served by FastAPI at `http://localhost:8000/dashboard/`.

### Step 3: Run the Complete Automated Test Suite

```powershell
python -m pytest
```
Runs 33 automated tests covering preprocessing, threshold tuning, drift detection, shadow/canary rollouts, auto-rollback, delayed labels, scam analysis, and API routes in ~5 seconds.

### Step 4: Start the API Service

```powershell
python -m uvicorn app.api.main:app --host 127.0.0.1 --port 8000 --reload
```

---

## 4. Local URLs & Service Endpoints

| Service | Local URL | Description |
|---|---|---|
| **Frontend Operations Dashboard** | [http://localhost:8000/dashboard/](http://localhost:8000/dashboard/) | Complete UI: Live table, streaming controls, drift, rollout & scam analyzer |
| **FastAPI Swagger Docs** | [http://localhost:8000/docs](http://localhost:8000/docs) | Interactive OpenAPI documentation for all endpoints |
| **Health Check** | [http://localhost:8000/health](http://localhost:8000/health) | API status, models loaded, uptime, rollout mode |
| **Prometheus Metrics** | [http://localhost:8000/metrics](http://localhost:8000/metrics) | Real Prometheus metrics exporter |
| **Grafana Dashboard** *(Docker)* | [http://localhost:3000](http://localhost:3000) | Observability dashboards (`admin` / `admin`) |
| **MLflow Tracking UI** *(Docker)* | [http://localhost:5000](http://localhost:5000) | Model runs, parameters, metrics, and logged artifacts |
| **Prometheus Server** *(Docker)* | [http://localhost:9090](http://localhost:9090) | Time-series metrics engine |
| **Frontend Dev Server** *(Optional)* | [http://localhost:5173](http://localhost:5173) | Vite hot-reloading dev server (`cd frontend; npm run dev`) |

---

## 5. Live Demonstration & Verification Walkthrough

You can execute all scenarios directly through the web UI at `http://localhost:8000/dashboard/` or via CLI / PowerShell:

### Scenario A: Baseline Transaction Streaming
1. Open `http://localhost:8000/dashboard/` and navigate to the **Replay Controls** tab.
2. Select **1. Baseline Traffic**, adjust speed to **5 tx/sec**, and click **Start Replay Stream**.
3. Switch to the **Live Transactions** tab to view transactions streaming in real time with calculated fraud probabilities, decision badges (`allow` or `review`), model versions, and latency.
4. Click **View** on any transaction to inspect all 29 input features and raw output schemas in the detail modal.
5. Alternatively, run via PowerShell:
   ```powershell
   python scripts/stream_transactions.py --scenario baseline --speed 5 --count 50
   ```

### Scenario B: Inject Feature Data Drift
1. In the **Replay Controls** tab, select **3. Feature Data Drift** and click **Start Replay Stream** (or use the CLI):
   ```powershell
   python scripts/stream_transactions.py --scenario feature_drift --speed 10 --count 60
   ```
2. Navigate to the **Data Drift Monitor** tab.
3. Observe the drift status switch to **DRIFT DETECTED** (red alert badge).
4. Review the feature-level statistical table: features such as `V1`, `V2`, `V3`, `V4`, and `Amount` will flag as `DRIFTED` with statistically significant p-values ($< 0.0017$ Bonferroni threshold), elevated KS statistics, and high PSI values.
5. Click **Reset Window** to clear the monitoring window back to baseline.

### Scenario C: Demonstrate Concept Drift & Delayed Labels
1. In the **Replay Controls** tab, select **4. Concept Drift** and click **Start Replay Stream** (or run via CLI):
   ```powershell
   python scripts/stream_transactions.py --scenario concept_drift --speed 10 --count 40
   ```
2. In this scenario, fraud events are disguised as low-amount normal transactions with inverted flags, causing the model to allow them.
3. Switch to the **Concept Drift & Delayed Labels** tab.
4. As delayed ground-truth labels arrive (simulating confirmed fraud reports or chargebacks), watch the confusion matrix update with a surge in **False Negatives (FN)**.
5. When the online F1 score drops below 60% or recall falls, an alert triggers:
   `"Concept drift / performance deterioration: Delayed F1 score is below threshold."`

### Scenario D: Test Shadow & Canary Rollouts
1. Navigate to the **Rollout & Release Gates** tab.
2. **Shadow Mode**: Click **2. Shadow Mode**. Every prediction scores both the Champion and Challenger models. The Champion decision is served while Challenger probabilities are logged.
3. **Canary Mode**: Click **3. Canary Rollout** and set the traffic allocation slider to **25%**. 25% of incoming traffic is routed to the Challenger model.
4. To configure via PowerShell:
   ```powershell
   Invoke-RestMethod -Method Post -Uri http://127.0.0.1:8000/models/rollout-mode -ContentType "application/json" -Body '{"mode": "canary", "canary_percent": 30}'
   ```

### Scenario E: Trigger Automated Canary Rollback
1. TrustOps includes automated safeguards: if the candidate experiences slow predictions ($> 500\text{ ms}$) or repeated errors, traffic is instantly rolled back to the Champion.
2. Trigger an immediate emergency rollback from the UI by clicking **Emergency Rollback**, or via PowerShell:
   ```powershell
   Invoke-RestMethod -Method Post -Uri http://127.0.0.1:8000/models/rollback -ContentType "application/json" -Body '{"reason": "Latency spike observed in upstream microservice"}'
   ```
3. Observe the `ROLLED BACK` alert banner and Prometheus metric `trustops_canary_rollbacks_total` incrementing.

### Scenario F: Release Gates & Model Promotion
1. In the **Rollout & Release Gates** tab, review the automated release gates checklist:
   - **Schema Compatibility**: Verifies exact match of all 29 features.
   - **Min PR-AUC**: Target $\ge 0.65$.
   - **Min Precision**: Target $\ge 0.50$.
   - **Min Recall**: Target $\ge 0.70$.
   - **Latency Budget**: Target $\le 50\text{ ms}$.
2. Click **Promote Challenger to Champion**. If all gates pass, the models and thresholds are swapped safely, recorded in the audit log, and logged to Prometheus.

### Scenario G: Phishing & Voice Scam Analysis
1. Navigate to the **Scam Analyzer** tab.
2. Click **CEO Wire Fraud** or **Fake IT Support (Audio Call)** to load sample attack transcripts.
3. Click **Analyze Communication for Threats**.
4. Review the extracted signals:
   - Threat level (`CRITICAL SCAM` or `SUSPICIOUS`)
   - Risk score ($0 - 100$)
   - Extracted evidence categories: Suspicious payment instructions, authority impersonation, urgency/pressure tactics, and credential requests.
5. Test the **Legitimate Invoice** sample to verify low risk ($< 35/100$) and clean classification.

---

## 6. Docker Compose Deployment

To run the complete containerized stack:

```powershell
# 1. Build the containers
docker compose build

# 2. Start all services in background
docker compose up -d

# 3. Check service health
docker compose ps
```

Services in Compose:
- **API & UI**: `http://localhost:8000/dashboard/`
- **MLflow Tracking**: `http://localhost:5000`
- **Prometheus**: `http://localhost:9090`
- **Grafana**: `http://localhost:3000` (`admin` / `admin`)

To stop:
```powershell
docker compose down
```

---

## 7. Troubleshooting

| Symptom | Cause | Solution |
|---|---|---|
| `Model artifacts not found in models` | Models have not been trained or extracted | Run `python -m app.ml.train` to generate `champion.joblib`, `challenger.joblib`, `reference.json`, and `metadata.json`. |
| Drift endpoint shows `status: "collecting"` | Monitoring window has not reached minimum sample count | The monitor requires at least 50 samples before running statistical tests. Start the replay stream or run `python scripts/stream_transactions.py --count 55`. |
| Port 8000 or 3000 already in use | Another service is using the port | Check running processes with `netstat -ano \| findstr :8000` and stop conflicting services, or configure another port in `.env`. |
| PowerShell script activation disabled | Windows execution policy restriction | Run `Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass`, then activate `.\.venv\Scripts\Activate.ps1`. |
| npm execution policy blocked in PowerShell | Running `.ps1` wrapper blocked | Use `npm.cmd` instead of `npm`, e.g., `npm.cmd run build`. |
