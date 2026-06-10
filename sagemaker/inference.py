"""
SageMaker inference handler for the Insurance Risk Scorer.

Accepts: POST application/json
  Body: {"ps_ind_01": 2, "ps_ind_02_cat": 2, "ps_reg_01": 0.7, ...}

Returns: application/json
  {"claim_probability": 0.043, "risk_label": "LOW"}
"""

import json
import os

import numpy as np
import xgboost as xgb

CLAIM_THRESHOLD = 0.10


def model_fn(model_dir: str) -> dict:
    model = xgb.Booster()
    model.load_model(os.path.join(model_dir, "xgb-model"))

    with open(os.path.join(model_dir, "preprocessor_params.json")) as f:
        params = json.load(f)

    return {"model": model, "params": params}


def input_fn(request_body: str, content_type: str = "application/json"):
    if content_type != "application/json":
        raise ValueError(f"Unsupported content type: {content_type}")
    data = json.loads(request_body)
    if isinstance(data, dict):
        data = [data]
    return data  # list of dicts


def predict_fn(rows: list, artifacts: dict) -> np.ndarray:
    params = artifacts["params"]
    model = artifacts["model"]

    cat_cols   = params["cat_cols"]
    cont_cols  = params["cont_cols"]
    bin_cols   = params["bin_cols"]
    cat_med    = np.array(params["cat_medians"])
    cont_med   = np.array(params["cont_medians"])
    cont_mean  = np.array(params["cont_mean"])
    cont_scale = np.array(params["cont_scale"])
    bin_modes  = np.array(params["bin_modes"])

    n = len(rows)
    cat_arr  = np.full((n, len(cat_cols)),  np.nan)
    cont_arr = np.full((n, len(cont_cols)), np.nan)
    bin_arr  = np.full((n, len(bin_cols)),  np.nan)

    for i, row in enumerate(rows):
        for j, col in enumerate(cat_cols):
            v = row.get(col)
            cat_arr[i, j]  = v if v is not None else np.nan
        for j, col in enumerate(cont_cols):
            v = row.get(col)
            cont_arr[i, j] = v if v is not None else np.nan
        for j, col in enumerate(bin_cols):
            v = row.get(col)
            bin_arr[i, j]  = v if v is not None else np.nan

    # Impute
    for j in range(len(cat_cols)):
        mask = np.isnan(cat_arr[:, j])
        cat_arr[mask, j] = cat_med[j]

    for j in range(len(cont_cols)):
        mask = np.isnan(cont_arr[:, j])
        cont_arr[mask, j] = cont_med[j]

    for j in range(len(bin_cols)):
        mask = np.isnan(bin_arr[:, j])
        bin_arr[mask, j] = bin_modes[j]

    # Scale continuous
    cont_arr = (cont_arr - cont_mean) / cont_scale

    X = np.hstack([cat_arr, cont_arr, bin_arr])
    return model.predict(xgb.DMatrix(X))


def output_fn(probabilities: np.ndarray, accept: str = "application/json") -> str:
    results = [
        {"claim_probability": round(float(p), 6),
         "risk_label": "HIGH" if p >= CLAIM_THRESHOLD else "LOW"}
        for p in probabilities
    ]
    payload = results[0] if len(results) == 1 else results
    return json.dumps(payload)
