"""
SageMaker inference handler for the Insurance Risk Scorer.

SageMaker calls these four hooks in order:
  model_fn     → load model artifacts from model_dir
  input_fn     → deserialise the HTTP request body
  predict_fn   → run inference
  output_fn    → serialise the response

Accepts: POST application/json
  Body: {"ps_ind_01": 2, "ps_ind_02_cat": 2, "ps_reg_01": 0.7, ...}

Returns: application/json
  {"claim_probability": 0.043, "risk_label": "LOW"}
"""

import json
import os

import joblib
import numpy as np
import pandas as pd
import xgboost as xgb

# Threshold: roughly 3× the base positive rate (~3.64%)
CLAIM_THRESHOLD = 0.10


def model_fn(model_dir: str) -> dict:
    """Load XGBoost model and fitted preprocessor from model_dir."""
    model = xgb.Booster()
    model.load_model(os.path.join(model_dir, "xgb-model"))

    preprocessor = joblib.load(os.path.join(model_dir, "preprocessor.pkl"))

    return {"model": model, "preprocessor": preprocessor}


def input_fn(request_body: str, content_type: str = "application/json") -> pd.DataFrame:
    """Parse the request body into a one-row DataFrame."""
    if content_type != "application/json":
        raise ValueError(f"Unsupported content type: {content_type}. Expected application/json.")

    data = json.loads(request_body)

    # Accept both a single dict and a list of dicts (batch)
    if isinstance(data, dict):
        data = [data]

    return pd.DataFrame(data)


def predict_fn(df: pd.DataFrame, artifacts: dict) -> np.ndarray:
    """Preprocess features and run XGBoost inference."""
    preprocessor = artifacts["preprocessor"]
    model = artifacts["model"]

    # Drop id/target if accidentally included in the payload
    drop_cols = [c for c in ["id", "target"] if c in df.columns]
    df = df.drop(columns=drop_cols)

    X = preprocessor.transform(df)
    dmat = xgb.DMatrix(X)
    return model.predict(dmat)


def output_fn(probabilities: np.ndarray, accept: str = "application/json") -> str:
    """Serialise predictions as JSON."""
    results = []
    for prob in probabilities:
        results.append(
            {
                "claim_probability": round(float(prob), 6),
                "risk_label": "HIGH" if prob >= CLAIM_THRESHOLD else "LOW",
            }
        )

    # Return a single object for single-row requests, list for batch
    payload = results[0] if len(results) == 1 else results
    return json.dumps(payload)
