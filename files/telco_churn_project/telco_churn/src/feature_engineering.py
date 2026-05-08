"""
feature_engineering.py — Create derived features before modelling.

New features added:
  - charges_per_month     : Total Charges / (Tenure Months + 1)
  - is_new_customer       : 1 if Tenure Months < 6
  - is_long_term          : 1 if Tenure Months >= 48
  - num_services          : count of active add-on services per customer
  - has_no_support        : 1 if no Online Security AND no Tech Support
  - high_spend_new        : 1 if new customer AND Monthly Charges > median

Usage:
    from src.feature_engineering import add_features
    df = add_features(df)
"""

import pandas as pd
import numpy as np
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent))
from config import CLEAN_CSV, TARGET

SERVICE_COLS = [
    "Online Security", "Online Backup", "Device Protection",
    "Tech Support", "Streaming TV", "Streaming Movies",
]


def add_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add all engineered features. Returns a new DataFrame."""
    df = df.copy()

    # Charges per month of tenure (avoids div/0 for tenure=0)
    df["charges_per_month"] = (
        df["Total Charges"] / (df["Tenure Months"] + 1)
    ).round(2)

    # Lifecycle flags
    df["is_new_customer"]  = (df["Tenure Months"] < 6).astype(int)
    df["is_long_term"]     = (df["Tenure Months"] >= 48).astype(int)

    # Count of Yes add-on services
    df["num_services"] = df[SERVICE_COLS].apply(
        lambda row: (row == "Yes").sum(), axis=1
    )

    # No security AND no support — double vulnerability
    df["has_no_support"] = (
        (df["Online Security"] == "No") & (df["Tech Support"] == "No")
    ).astype(int)

    # High spender who is still new — highest risk combo
    median_charge = df["Monthly Charges"].median()
    df["high_spend_new"] = (
        (df["is_new_customer"] == 1) & (df["Monthly Charges"] > median_charge)
    ).astype(int)

    new_cols = [
        "charges_per_month", "is_new_customer", "is_long_term",
        "num_services", "has_no_support", "high_spend_new",
    ]
    print(f"[feature_engineering] Added {len(new_cols)} features: {new_cols}")
    print(f"  New customer rate    : {df['is_new_customer'].mean():.1%}")
    print(f"  Long-term rate       : {df['is_long_term'].mean():.1%}")
    print(f"  Avg services/customer: {df['num_services'].mean():.2f}")
    print(f"  No-support rate      : {df['has_no_support'].mean():.1%}")
    return df


def load_engineered(csv_path: Path = CLEAN_CSV) -> pd.DataFrame:
    """Load clean CSV and apply feature engineering."""
    df = pd.read_csv(csv_path)
    return add_features(df)


if __name__ == "__main__":
    df = load_engineered()
    print(f"\nFinal shape: {df.shape}")
    print(df[["charges_per_month", "is_new_customer", "num_services",
              "has_no_support", "high_spend_new", TARGET]].describe())
