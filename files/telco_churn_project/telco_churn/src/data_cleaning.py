import pandas as pd
import numpy as np
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent))
from config import RAW_CSV, CLEAN_CSV, TARGET, DROP_COLS


def load_raw(path):
    print(f"[1/6] Loading raw data from: {path}")
    df = pd.read_csv(path)
    print(f"      Shape: {df.shape}  |  Columns: {len(df.columns)}")
    return df


def drop_leakage_cols(df):
    existing = [c for c in DROP_COLS if c in df.columns]
    print(f"[2/6] Dropping {len(existing)} columns: {existing}")
    return df.drop(columns=existing, errors="ignore")


def fix_total_charges(df):
    col = "Total Charges"
    if col not in df.columns:
        return df
    print(f"[3/6] Fixing '{col}' dtype (whitespace -> NaN -> median impute)")
    df = df.copy()
    df[col] = pd.to_numeric(df[col].astype(str).str.strip(), errors="coerce")
    n_null = df[col].isna().sum()
    if n_null:
        median_val = df[col].median()
        df[col] = df[col].fillna(median_val)
        print(f"      Imputed {n_null} NaN(s) with median={median_val:.2f}")
    return df


def verify_target(df):
    print(f"[4/6] Verifying target '{TARGET}'")
    df = df.copy()
    df[TARGET] = df[TARGET].astype(int)
    print(f"      Churn rate: {df[TARGET].mean():.1%}  ({df[TARGET].sum()} churned / {len(df)} total)")
    return df


def strip_strings(df):
    print("[5/6] Stripping whitespace from string columns")
    df = df.copy()
    str_cols = [c for c in df.columns if df[c].dtype == object]
    for col in str_cols:
        df[col] = df[col].str.strip()
    return df


def validate(df):
    print("[6/6] Running validation checks")
    assert df.isnull().sum().sum() == 0, "Unexpected NaNs remain!"
    assert df[TARGET].isin([0, 1]).all(), "Target has values outside {0,1}!"
    n_dupes = df.duplicated().sum()
    if n_dupes:
        print(f"      WARNING: {n_dupes} duplicate rows detected")
    print(f"      All checks passed.  Final shape: {df.shape}")


def clean(raw_path=RAW_CSV, out_path=CLEAN_CSV):
    df = load_raw(raw_path)
    df = drop_leakage_cols(df)
    df = fix_total_charges(df)
    df = verify_target(df)
    df = strip_strings(df)
    validate(df)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_path, index=False)
    print(f"\n Clean data saved -> {out_path}  ({df.shape[0]} rows, {df.shape[1]} cols)\n")
    return df


if __name__ == "__main__":
    clean()
