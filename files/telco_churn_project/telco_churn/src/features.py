"""
features.py — Feature engineering and sklearn Pipeline builder.

Exports:
  build_preprocessor()  — ColumnTransformer for numeric + categorical features
  load_Xy()             — load clean CSV and return X, y split
  get_train_test()      — stratified train/test split

Usage:
    from src.features import get_train_test
    X_train, X_test, y_train, y_test = get_train_test()
"""

import pandas as pd
import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.model_selection import train_test_split
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent))
from config import (
    CLEAN_CSV, TARGET,
    NUMERIC_COLS, CATEGORICAL_COLS,
    RANDOM_STATE, TEST_SIZE,
)


def build_preprocessor() -> ColumnTransformer:
    """
    Return a ColumnTransformer that:
      - StandardScaler on numeric columns
      - OneHotEncoder (drop='if_binary') on categorical columns

    The transformer is unfitted; call .fit_transform(X_train) downstream.
    """
    numeric_pipeline = StandardScaler()

    categorical_pipeline = OneHotEncoder(
        drop="if_binary",       # avoids redundant column for binary features
        handle_unknown="ignore",
        sparse_output=False,
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_pipeline,         NUMERIC_COLS),
            ("cat", categorical_pipeline,     CATEGORICAL_COLS),
        ],
        remainder="drop",
        verbose_feature_names_out=False,
    )
    return preprocessor


def load_Xy(csv_path: Path = CLEAN_CSV):
    """Load clean CSV; return feature matrix X and label vector y."""
    df = pd.read_csv(csv_path)
    X  = df.drop(columns=[TARGET])
    y  = df[TARGET]
    print(f"Loaded: X={X.shape}, y={y.shape}  |  churn rate={y.mean():.2%}")
    return X, y


def get_train_test(csv_path: Path = CLEAN_CSV):
    """
    Return stratified train/test split.
    Returns: X_train, X_test, y_train, y_test
    """
    X, y = load_Xy(csv_path)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=y,
    )
    print(f"Train: {X_train.shape}  |  Test: {X_test.shape}")
    print(f"Train churn rate: {y_train.mean():.2%}  |  Test churn rate: {y_test.mean():.2%}")
    return X_train, X_test, y_train, y_test


def get_feature_names(preprocessor: ColumnTransformer):
    """Extract human-readable feature names after fitting the preprocessor."""
    return preprocessor.get_feature_names_out().tolist()
