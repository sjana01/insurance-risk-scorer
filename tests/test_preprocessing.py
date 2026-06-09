"""Tests for src/preprocessing.py"""

import io
import numpy as np
import pandas as pd
import pytest

from src.preprocessing import load_and_clean, split_feature_columns, get_feature_columns


def _make_csv(data: dict) -> str:
    return pd.DataFrame(data).to_csv(index=False)


def _write_tmp_csv(tmp_path, data: dict) -> str:
    path = tmp_path / "train.csv"
    path.write_text(_make_csv(data))
    return str(path)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def sample_raw_data():
    return {
        "id": [1, 2, 3],
        "target": [0, 1, 0],
        "ps_ind_01": [2, -1, 3],         # continuous — has a missing (-1)
        "ps_ind_02_cat": [2, 3, -1],      # categorical
        "ps_ind_06_bin": [0, 1, 0],       # binary
        "ps_calc_01": [0.6, 0.3, 0.1],   # _calc_ — should be dropped
        "ps_calc_02": [1.1, 2.2, 3.3],   # _calc_ — should be dropped
    }


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

def test_replace_missing_with_nan(tmp_path, sample_raw_data):
    path = _write_tmp_csv(tmp_path, sample_raw_data)
    df = load_and_clean(path)
    assert df["ps_ind_01"].isna().sum() == 1
    assert df["ps_ind_02_cat"].isna().sum() == 1
    assert not (df == -1).any().any(), "No -1 values should remain after cleaning"


def test_drop_calc_columns(tmp_path, sample_raw_data):
    path = _write_tmp_csv(tmp_path, sample_raw_data)
    df = load_and_clean(path)
    calc_cols = [c for c in df.columns if "_calc_" in c]
    assert calc_cols == [], f"_calc_ columns were not dropped: {calc_cols}"


def test_missing_count_feature(tmp_path, sample_raw_data):
    path = _write_tmp_csv(tmp_path, sample_raw_data)
    df = load_and_clean(path)
    assert "missing_count" in df.columns
    # Row 0: no missing; Row 1: ps_ind_01 missing; Row 2: ps_ind_02_cat missing
    assert df["missing_count"].iloc[0] == 0
    assert df["missing_count"].iloc[1] == 1
    assert df["missing_count"].iloc[2] == 1


def test_id_and_target_preserved(tmp_path, sample_raw_data):
    path = _write_tmp_csv(tmp_path, sample_raw_data)
    df = load_and_clean(path)
    assert "id" in df.columns
    assert "target" in df.columns


def test_split_feature_columns_cat(tmp_path, sample_raw_data):
    path = _write_tmp_csv(tmp_path, sample_raw_data)
    df = load_and_clean(path)
    groups = split_feature_columns(df)
    assert "ps_ind_02_cat" in groups["cat"]


def test_split_feature_columns_bin(tmp_path, sample_raw_data):
    path = _write_tmp_csv(tmp_path, sample_raw_data)
    df = load_and_clean(path)
    groups = split_feature_columns(df)
    assert "ps_ind_06_bin" in groups["bin"]


def test_split_feature_columns_cont(tmp_path, sample_raw_data):
    path = _write_tmp_csv(tmp_path, sample_raw_data)
    df = load_and_clean(path)
    groups = split_feature_columns(df)
    assert "ps_ind_01" in groups["cont"]


def test_split_feature_columns_excludes_meta(tmp_path, sample_raw_data):
    path = _write_tmp_csv(tmp_path, sample_raw_data)
    df = load_and_clean(path)
    groups = split_feature_columns(df)
    all_features = groups["cat"] + groups["bin"] + groups["cont"]
    assert "id" not in all_features
    assert "target" not in all_features
    assert "missing_count" not in all_features


def test_get_feature_columns_excludes_id_target(tmp_path, sample_raw_data):
    path = _write_tmp_csv(tmp_path, sample_raw_data)
    df = load_and_clean(path)
    feature_cols = get_feature_columns(df)
    assert "id" not in feature_cols
    assert "target" not in feature_cols
    assert "missing_count" in feature_cols
