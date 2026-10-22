# TrustOps: End-to-End MLOps for Credit Card Fraud Detection

TrustOps is a modular demonstration of an end-to-end ML lifecycle:
- Standard public ULB credit-card fraud dataset
- Dataset validation and preprocessing
- Imbalance-aware model training and threshold selection
- MLflow experiment tracking
- Champion/challenger model artifacts
- FastAPI inference service
- Shadow and canary rollout modes with latency-triggered canary rollback
- Prometheus metrics and Grafana dashboard
- Input drift monitoring
- Automated tests and GitHub Actions CI
- Docker Compose local deployment

> **Caveat:** This is not a bank-ready fraud prevention system. The ULB dataset is anonymized, historical, and extremely imbalanced. Model outputs are risk signals, not proof of fraud. Do not use this demo to automatically decline real transactions.

## Dataset

Use the ULB Machine Learning Group's **Credit Card Fraud Detection** dataset:
https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud

It contains 284,807 European card transactions from September 2013, with 492 frauds (about 0.172%). Features include `V1`–`V28` (anonymized PCA components), `Amount`, optionally `Time`, and target `Class`. Because the target is rare, PR-AUC / average precision, precision, recall and F1 are more informative than accuracy alone.

### Download dataset (recommended: OpenML mirror)
The downloader attempts to fetch OpenML dataset ID 1597. If it fails, download `creditcard.csv` from Kaggle and put it at `data/raw/creditcard.csv`.

## Windows quick start (PowerShell)

Requirements: Python 3.11 recommended, Git, Docker Desktop.

```powershell
# 1. Extract the ZIP, then enter the project folder
cd .\TrustOps

# 2. Create a virtual environment (no conda)
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1

# If PowerShell blocks activation, use:
# Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
# .\.venv\Scripts\Activate.ps1

# 3. Install dependencies
python -m pip install --upgrade pip
pip install -r requirements.txt

# 4. Download dataset
python scripts/download_data.py

# 5. Train models and log run to local MLflow
python -m app.ml.train

# 6. Run API locally in another terminal (activate venv first)
uvicorn app.api.main:app --reload --host 127.0.0.1 --port 8000

# 7. Open docs
# http://127.0.0.1:8000/docs
# http://127.0.0.1:8000/health
# http://127.0.0.1:8000/metrics
```

If the OpenML download does not work, manually download the Kaggle CSV to `data/raw/creditcard.csv`, then run step 5.

### Example prediction request
Use PowerShell:
```powershell
$body = @{
  transaction = @{
    V1 = -1.359807
    V2 = -0.072781
    V3 = 2.536347
    V4 = 1.378155
    V5 = -0.338321
    V6 = 0.462388
    V7 = 0.239599
    V8 = 0.098698
    V9 = 0.363787
    V10 = 0.090794
    V11 = -0.551600
    V12 = -0.617801
    V13 = -0.991390
    V14 = -0.311169
    V15 = 1.468177
    V16 = -0.470401
    V17 = 0.207971
    V18 = 0.025791
    V19 = 0.403993
    V20 = 0.251412
    V21 = -0.018307
    V22 = 0.277838
    V23 = -0.110474
    V24 = 0.066928
    V25 = 0.128539
    V26 = -0.189115
    V27 = 0.133558
    V28 = -0.021053
    Amount = 149.62
  }
} | ConvertTo-Json -Depth 5
Invoke-RestMethod -Method Post -Uri http://127.0.0.1:8000/predict -ContentType "application/json" -Body $body
```
`Time` is optional. If the trained model used `Time`, the API fills missing `Time` with 0; pass it when available.

## Run the complete stack with Docker Compose

The Docker path is the most repeatable setup. First place `creditcard.csv` in `data/raw/` (or download it locally). Then:

```powershell
docker compose build
docker compose --profile train run --rm trainer
docker compose up -d
docker compose ps
```

Services:
- API: http://localhost:8000/docs
- MLflow UI: http://localhost:5000
- Prometheus: http://localhost:9090
- Grafana: http://localhost:3000 (default local credentials: admin / admin; change before sharing)

To stop:
```powershell
docker compose down
```

To view logs:
```powershell
docker compose logs -f api
docker compose logs -f mlflow
```

## Rollout modes

Set `ROLLOUT_MODE` in `.env` or `docker-compose.yml`:
- `champion`: serve champion only (default)
- `shadow`: return champion prediction while evaluating challenger in the background/request path for comparison
- `canary`: send `CANARY_PERCENT` of requests to challenger; if candidate latency crosses `CANARY_MAX_LATENCY_MS` repeatedly, automatically route traffic back to champion.

For a quick local mode change, edit `.env` then restart the API:
```powershell
docker compose up -d --force-recreate api
```
The canary rollback is intentionally a small demonstration based on latency, not a full production deployment controller. It does not replace traffic management, alerting, or human review.

## Monitoring endpoints
- `GET /health` - liveness and model availability
- `GET /model-info` - current rollout status and model metadata
- `GET /monitoring/drift` - simple population stability index (PSI) by feature using prediction traffic
- `GET /metrics` - Prometheus metrics
- `POST /predict` - one transaction
- `POST /predict/batch` - up to 500 transactions

Drift needs traffic. It compares recent prediction inputs against the training reference distribution. PSI is a heuristic alert, not proof of model degradation.

## Pipeline layout

```text
scripts/download_data.py
          |
          v
data/raw/creditcard.csv
          |
          v
app/ml/data.py -> validate / split
          |
          v
app/ml/train.py -> fit / evaluate / threshold / MLflow
          |
          +--> models/champion.joblib
          +--> models/challenger.joblib
          +--> models/metadata.json
          +--> models/reference.json
                         |
                         v
                 app/api/main.py
                   |      |
             FastAPI   Prometheus
                   |
              Grafana dashboard
```

## Quality gates and tests

Training logs test-set average precision, ROC-AUC, precision, recall, F1, threshold, and confusion matrix. The model is saved only after training completes and metrics are recorded. GitHub Actions installs dependencies, runs unit/API tests, and runs a lightweight compile check. The CI job does not download the full 150 MB dataset or retrain the production-size model.

## Deploying to AWS

This repo is container-ready, but AWS deployment is intentionally not automated because it requires your AWS account, region, registry, IAM permissions, and service choice. A straightforward next step is to push the API image to Amazon ECR and run it on App Runner or ECS; put MLflow and persistent artifacts in managed/persistent storage before production use. Never commit AWS credentials or `.env`.

## Security and limitations
- The model uses anonymized historical transactions, not real customer identities.
- Keep the API private or protected before exposing it publicly.
- Do not expose MLflow or Grafana publicly without authentication/network controls.
- Add authentication, request rate limiting, TLS, secrets management, durable artifact storage, alerting, and retraining governance before production.
- Avoid accuracy-only claims on the highly imbalanced dataset.
