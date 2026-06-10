# Insurance Risk Scorer

Binary risk-scoring model that predicts automobile insurance claim probability, deployed as a live REST API on AWS.

**Dataset:** [Porto Seguro Safe Driver Prediction](https://www.kaggle.com/c/porto-seguro-safe-driver-prediction) (Kaggle)  
**Model:** XGBoost — 5-fold CV, Normalized Gini metric  
**API:** API Gateway → Lambda → SageMaker real-time endpoint

---

## Architecture

```
                   POST /predict
                        │
              ┌─────────▼──────────┐
              │    API Gateway     │
              │  (REST, Regional)  │
              └─────────┬──────────┘
                        │ Lambda proxy
              ┌─────────▼──────────┐
              │  Lambda Function   │
              │ (Python 3.10,128MB)│
              └─────────┬──────────┘
                        │ boto3 invoke_endpoint
              ┌─────────▼──────────┐
              │  SageMaker         │
              │  Real-Time Endpoint│
              │  (ml.m5.large)     │
              └─────────┬──────────┘
                        │
              ┌─────────▼──────────┐
              │  XGBoost Model     │
              │  + Preprocessor    │
              │  (S3 → model.tar)  │
              └────────────────────┘
```

---

## Prerequisites

| Tool | Version | Install |
|---|---|---|
| conda / miniconda | any | [docs.conda.io](https://docs.conda.io) |
| AWS CLI | v2 | `pip install awscli` |
| Kaggle CLI | any | included in conda env |
| Python | 3.10 | via conda env |

You need an AWS account. See **[infra/iam_setup.md](infra/iam_setup.md)** for the full IAM setup (takes ~15 min).

---

## Quick Start

### 1. Clone and create conda environment

```bash
git clone https://github.com/YOUR_USERNAME/insurance-risk-scorer.git
cd insurance-risk-scorer

conda env create -f environment.yml
conda activate insurance-risk
```

### 2. AWS and Kaggle setup

Follow **[infra/iam_setup.md](infra/iam_setup.md)** to create IAM roles and configure the CLI.

Set environment variables (add to your shell profile):

```bash
export SAGEMAKER_ROLE_ARN="arn:aws:iam::ACCOUNT_ID:role/InsuranceRiskScorerSageMakerRole"
export LAMBDA_ROLE_ARN="arn:aws:iam::ACCOUNT_ID:role/InsuranceRiskScorerLambdaRole"
export S3_BUCKET="insurance-risk-scorer-ACCOUNT_ID"
export AWS_REGION="us-east-1"
export ENDPOINT_NAME="insurance-risk-scorer"
```

### 3. Download data

```bash
# Place kaggle.json at ~/.kaggle/kaggle.json first
kaggle competitions download -c porto-seguro-safe-driver-prediction -p data/raw/
cd data/raw && unzip porto-seguro-safe-driver-prediction.zip && cd ../..

# Upload training data to S3
aws s3 cp data/raw/train.csv s3://$S3_BUCKET/porto-seguro/data/train.csv
```

---

## EDA and Feature Engineering

```bash
jupyter lab
# Open and run in order:
#   notebooks/01_eda.ipynb
#   notebooks/02_features.ipynb
#   notebooks/03_local_training.ipynb
```

**Key dataset facts:**
- 595,212 rows × 57 features + target
- 3.64% positive class (severe imbalance → `scale_pos_weight ≈ 26`)
- Missing values encoded as -1
- `_calc_` features dropped (no predictive value, confirmed by community)

---

## Local Training

Run notebook `03_local_training.ipynb` or script-equivalent:

```python
# 5-fold stratified CV
# Expected CV Normalized Gini: ~0.275 – 0.285
```

Artifacts written to `models/`:
- `xgb_model.json` — XGBoost native format
- `preprocessor.pkl` — fitted ColumnTransformer
- `cv_results.json` — per-fold Gini + OOF Gini

---

## SageMaker Training

```bash
# Launch training job (~25 min, ~$0.10)
python sagemaker/deploy.py --train-only

# Check status in AWS Console → SageMaker → Training Jobs
```

---

## Deploy to SageMaker Endpoint

```bash
# Deploy endpoint from completed training job
python sagemaker/deploy.py --deploy --job-name insurance-risk-scorer-YYYYMMDD-HHMMSS

# Or train + deploy in one command:
python sagemaker/deploy.py --train-and-deploy

# Smoke test the endpoint directly
python sagemaker/invoke_test.py
```

**Instance type:** ml.m5.large (~$0.115/hr)  
**WARNING:** The endpoint bills continuously. Run cleanup when done.

---

## Deploy REST API

```bash
python api/deploy_api.py
# Prints: https://{api-id}.execute-api.{region}.amazonaws.com/prod/predict
```

---

## Test the API

```bash
curl -X POST https://{api-id}.execute-api.{region}.amazonaws.com/prod/predict \
  -H "Content-Type: application/json" \
  -d '{
    "ps_ind_01": 2,
    "ps_ind_02_cat": 2,
    "ps_ind_06_bin": 0,
    "ps_reg_01": 0.7,
    "ps_reg_02": 0.2,
    "ps_reg_03": 0.718,
    "ps_car_13": 0.664,
    "missing_count": 0
  }'
```

**Response:**
```json
{
  "claim_probability": 0.043217,
  "risk_label": "LOW"
}
```

| Field | Type | Description |
|---|---|---|
| `claim_probability` | float [0,1] | Model-estimated probability of filing a claim |
| `risk_label` | `"HIGH"` / `"LOW"` | HIGH if probability ≥ 0.10 (~3× base rate) |

---

## Monitoring

```bash
# Creates CloudWatch alarms for errors + latency
python monitoring/stub.py

# Optional: set email for alerts
ALARM_EMAIL=you@example.com python monitoring/stub.py
```

SageMaker Model Monitor baselining is included as a commented stub in `monitoring/stub.py`. Uncomment and run after the endpoint has served real traffic (~1000+ requests).

---

## Cleanup

```bash
# IMPORTANT: run this when done — the endpoint bills ~$0.115/hr
python infra/cleanup.py

# Also delete S3 data (optional):
python infra/cleanup.py --delete-s3

# Preview what would be deleted:
python infra/cleanup.py --dry-run
```

Resources deleted: API Gateway, Lambda, SageMaker endpoint, endpoint config, model, CloudWatch alarms.

---

## Run Tests

```bash
pytest tests/ -v
pytest tests/ -v --cov=src --cov-report=term-missing
```

---

## Results

| Metric | Value |
|---|---|
| CV Normalized Gini | ~0.275 – 0.285 |
| OOF Normalized Gini | ~0.275 |
| Endpoint latency p50 | ~50 ms |
| Endpoint latency p99 | ~200 ms |

---

## Project Structure

```
insurance-risk-scorer/
├── data/                   # gitignored — download locally
├── notebooks/
│   ├── 01_eda.ipynb        # EDA
│   ├── 02_features.ipynb   # Feature pipeline walkthrough
│   └── 03_local_training.ipynb
├── src/
│   ├── preprocessing.py    # load_and_clean, split_feature_columns
│   ├── features.py         # build_preprocessor (sklearn Pipeline)
│   └── metrics.py          # normalized_gini, gini_xgb_eval
├── sagemaker/
│   ├── train.py            # SM training entry point
│   ├── inference.py        # model_fn / input_fn / predict_fn / output_fn
│   ├── deploy.py           # launch training job + deploy endpoint
│   └── invoke_test.py      # endpoint smoke test
├── api/
│   ├── lambda_handler.py   # Lambda: API GW → SageMaker
│   └── deploy_api.py       # boto3: create Lambda + API GW
├── monitoring/
│   └── stub.py             # CloudWatch alarms + Model Monitor skeleton
├── infra/
│   ├── iam_setup.md        # Step-by-step IAM setup
│   └── cleanup.py          # Tear down all AWS resources
├── tests/
│   ├── test_preprocessing.py
│   └── test_metrics.py
├── environment.yml
└── requirements.txt
```
