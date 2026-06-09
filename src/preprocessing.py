"""
Data loading and cleaning for the Porto Seguro Auto Insurance dataset.

Conventions:
  - Missing values are encoded as -1 in the raw CSV; we convert them to NaN.
  - Features with '_calc_' in their name carry no predictive signal and are dropped.
  - A 'missing_count' feature captures row-wise missingness.
"""

import numpy as np
import pandas as pd


CALC_INFIX = "_calc_"


def load_and_clean(path: str, is_train: bool = True) -> pd.DataFrame:
    """Load raw CSV, replace -1 → NaN, drop _calc_ columns, add missing_count."""
    df = pd.read_csv(path)

    # -1 encodes missing in this dataset
    df.replace(-1, np.nan, inplace=True)

    # _calc_ features are derived from other features; dropping them consistently
    # improves generalisation (confirmed by Kaggle community analysis)
    calc_cols = [c for c in df.columns if CALC_INFIX in c]
    df.drop(columns=calc_cols, inplace=True)

    # Row-wise missingness is a useful proxy for data quality
    meta_cols = {"id", "target"}
    feature_cols = [c for c in df.columns if c not in meta_cols]
    df["missing_count"] = df[feature_cols].isnull().sum(axis=1)

    return df


def split_feature_columns(df: pd.DataFrame) -> dict:
    """Return dict of column lists by suffix type."""
    exclude = {"id", "target", "missing_count"}
    cols = [c for c in df.columns if c not in exclude]
    return {
        "cat": [c for c in cols if c.endswith("_cat")],
        "bin": [c for c in cols if c.endswith("_bin")],
        "cont": [c for c in cols if not c.endswith(("_cat", "_bin"))],
    }


def get_feature_columns(df: pd.DataFrame) -> list:
    """All feature columns (excludes id, target)."""
    return [c for c in df.columns if c not in {"id", "target"}]
