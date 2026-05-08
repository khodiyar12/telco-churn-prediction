"""
run_all.py — Execute the full Telco Churn pipeline end-to-end.

Steps:
  1. Data cleaning
  2. EDA (all plots)
  3. Model training (Logistic Regression + LightGBM/Optuna)
  4. SHAP explainability
  5. Demo prediction

Usage:
    python run_all.py
    python run_all.py --no-optuna      # skip Optuna (faster run ~2 min)
    python run_all.py --skip-eda       # skip EDA plots
"""

import argparse
import time
from pathlib import Path
import sys

sys.path.insert(0, "src")


def header(msg: str) -> None:
    print(f"\n{'='*60}")
    print(f"  {msg}")
    print(f"{'='*60}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--no-optuna",  action="store_true")
    parser.add_argument("--skip-eda",   action="store_true")
    args = parser.parse_args()

    t0 = time.time()

    # ── 1. Cleaning ────────────────────────────────────────────────────────────
    header("STEP 1 / 5 — Data Cleaning")
    from data_cleaning import clean
    clean()

    # ── 2. EDA ────────────────────────────────────────────────────────────────
    if not args.skip_eda:
        header("STEP 2 / 5 — Exploratory Data Analysis")
        from eda import run_eda
        run_eda()
    else:
        print("\n[Skipped] EDA (--skip-eda flag set)")

    # ── 3. Training ───────────────────────────────────────────────────────────
    header("STEP 3 / 5 — Model Training")
    from train import main as train_main
    train_main(use_optuna=not args.no_optuna)

    # ── 4. SHAP ───────────────────────────────────────────────────────────────
    header("STEP 4 / 5 — SHAP Explainability")
    from explain import run_shap
    run_shap()

    # ── 5. Demo prediction ────────────────────────────────────────────────────
    header("STEP 5 / 5 — Demo Prediction")
    from predict import run_demo
    run_demo()

    elapsed = time.time() - t0
    print(f"\n{'='*60}")
    print(f"  Pipeline complete in {elapsed/60:.1f} min")
    print(f"  Outputs -> outputs/eda/  |  outputs/shap/  |  models/")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()
