"""
Feature engineering pipeline for Porto Seguro.

Builds a sklearn ColumnTransformer that:
  - Target-encodes _cat columns (with smoothing to reduce leakage risk)
  - Median-imputes + standard-scales continuous columns
  - Mode-imputes binary columns (no scaling needed for XGBoost)
"""

import joblib
import numpy as np
import pandas as pd
from category_encoders import TargetEncoder
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from src.preprocessing import split_feature_columns


def build_preprocessor(df: pd.DataFrame) -> ColumnTransformer:
    """Construct a ColumnTransformer fitted to column structure of df (unfitted)."""
    col_groups = split_feature_columns(df)

    cat_pipeline = Pipeline(
        steps=[
            ("te", TargetEncoder(smoothing=10, min_samples_leaf=5)),
            ("imp", SimpleImputer(strategy="mean")),
        ]
    )

    cont_pipeline = Pipeline(
        steps=[
            ("imp", SimpleImputer(strategy="median")),
            ("scl", StandardScaler()),
        ]
    )

    bin_pipeline = Pipeline(
        steps=[
            ("imp", SimpleImputer(strategy="most_frequent")),
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("cat", cat_pipeline, col_groups["cat"]),
            ("cont", cont_pipeline, col_groups["cont"]),
            ("bin", bin_pipeline, col_groups["bin"]),
        ],
        remainder="drop",
    )

    return preprocessor


def fit_preprocessor(preprocessor: ColumnTransformer, X: pd.DataFrame, y: pd.Series) -> ColumnTransformer:
    """Fit a preprocessor and return it."""
    return preprocessor.fit(X, y)


def get_feature_names(preprocessor: ColumnTransformer, df: pd.DataFrame) -> list:
    """Recover output column names after ColumnTransformer."""
    col_groups = split_feature_columns(df)
    return col_groups["cat"] + col_groups["cont"] + col_groups["bin"]


def save_preprocessor(preprocessor: ColumnTransformer, path: str) -> None:
    joblib.dump(preprocessor, path)


def load_preprocessor(path: str) -> ColumnTransformer:
    return joblib.load(path)
