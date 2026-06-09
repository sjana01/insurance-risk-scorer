"""
SageMaker training entry point for the Insurance Risk Scorer.

Invoked by SageMaker when the training job starts. Reads hyperparameters as
CLI args, loads preprocessed data from SM_CHANNEL_TRAIN, trains XGBoost,
and writes model artifacts to SM_MODEL_DIR so SageMaker can package them.

Usage (local smoke-test):
    python sagemaker/train.py \
        --train data/raw \
        --model-dir /tmp/model \
        --max-depth 6 --eta 0.05
"""

import argparse
import json
import os
import sys

import joblib
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.model_selection import StratifiedKFold

# Allow importing from repo root when running locally
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from src.preprocessing import load_and_clean, get_feature_columns
from src.features import build_preprocessor
from src.metrics import normalized_gini


def parse_args():
    parser = argparse.ArgumentParser()

    # SageMaker injects these automatically
    parser.add_argument("--model-dir", type=str, default=os.environ.get("SM_MODEL_DIR", "/opt/ml/model"))
    parser.add_argument("--train", type=str, default=os.environ.get("SM_CHANNEL_TRAIN", "/opt/ml/input/data/train"))

    # Hyperparameters (passed via SageMaker estimator.hyperparameters or CLI)
    parser.add_argument("--max-depth", type=int, default=6)
    parser.add_argument("--eta", type=float, default=0.05)
    parser.add_argument("--subsample", type=float, default=0.8)
    parser.add_argument("--colsample-bytree", type=float, default=0.8)
    parser.add_argument("--min-child-weight", type=int, default=100)
    parser.add_argument("--num-boost-round", type=int, default=500)
    parser.add_argument("--early-stopping-rounds", type=int, default=50)
    parser.add_argument("--n-folds", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)

    return parser.parse_args()


def load_data(channel_dir: str) -> pd.DataFrame:
    """Load the first CSV found in the channel directory."""
    for fname in os.listdir(channel_dir):
        if fname.endswith(".csv"):
            path = os.path.join(channel_dir, fname)
            print(f"Loading: {path}")
            return load_and_clean(path)
    raise FileNotFoundError(f"No CSV found in {channel_dir}")


def train(args):
    df = load_data(args.train)
    feature_cols = get_feature_columns(df)
    X = df[feature_cols]
    y = df["target"]

    pos_rate = y.mean()
    scale_pos_weight = (1 - pos_rate) / pos_rate
    print(f"Rows: {len(df):,}  positive rate: {pos_rate:.4f}  scale_pos_weight: {scale_pos_weight:.1f}")

    params = {
        "max_depth": args.max_depth,
        "eta": args.eta,
        "subsample": args.subsample,
        "colsample_bytree": args.colsample_bytree,
        "min_child_weight": args.min_child_weight,
        "scale_pos_weight": scale_pos_weight,
        "objective": "binary:logistic",
        "tree_method": "hist",
        "eval_metric": "auc",
        "seed": args.seed,
        "nthread": -1,
    }

    skf = StratifiedKFold(n_splits=args.n_folds, shuffle=True, random_state=args.seed)
    fold_scores = []
    best_model = None
    best_prep = None
    best_gini = -1.0

    for fold, (train_idx, val_idx) in enumerate(skf.split(X, y), 1):
        print(f"\n--- Fold {fold}/{args.n_folds} ---")
        X_tr, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_tr, y_val = y.iloc[train_idx], y.iloc[val_idx]

        prep = build_preprocessor(df)
        X_tr_enc = prep.fit_transform(X_tr, y_tr)
        X_val_enc = prep.transform(X_val)

        dtrain = xgb.DMatrix(X_tr_enc, label=y_tr)
        dval = xgb.DMatrix(X_val_enc, label=y_val)

        model = xgb.train(
            params,
            dtrain,
            num_boost_round=args.num_boost_round,
            evals=[(dtrain, "train"), (dval, "val")],
            early_stopping_rounds=args.early_stopping_rounds,
            verbose_eval=100,
        )

        gini = normalized_gini(y_val.values, model.predict(dval))
        fold_scores.append(gini)
        print(f"Fold {fold} Gini: {gini:.4f}")

        if gini > best_gini:
            best_gini = gini
            best_model = model
            best_prep = prep

    mean_gini = float(np.mean(fold_scores))
    print(f"\nCV Gini: {mean_gini:.4f} ± {np.std(fold_scores):.4f}")

    # --- Save artifacts ---
    model_path = os.path.join(args.model_dir, "xgb-model")
    prep_path = os.path.join(args.model_dir, "preprocessor.pkl")
    meta_path = os.path.join(args.model_dir, "model_meta.json")

    os.makedirs(args.model_dir, exist_ok=True)
    best_model.save_model(model_path)
    joblib.dump(best_prep, prep_path)

    meta = {
        "fold_gini": fold_scores,
        "mean_gini": mean_gini,
        "best_iteration": best_model.best_iteration,
        "params": params,
    }
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)

    print(f"Saved model:       {model_path}")
    print(f"Saved preprocessor:{prep_path}")
    print(f"Saved meta:        {meta_path}")


if __name__ == "__main__":
    args = parse_args()
    train(args)
