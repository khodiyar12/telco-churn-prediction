"""
predict.py — Run churn predictions on new customer data.

Usage:
    python src/predict.py --input data/new_customers.csv --output outputs/predictions.csv
    python src/predict.py --demo
"""

import argparse
from pathlib import Path
import sys

import joblib
import pandas as pd
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from config import MODEL_DIR, OUTPUT_DIR, TARGET

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

DEMO_CUSTOMER = {
    "Gender":              "Male",
    "Senior Citizen":      "Yes",
    "Partner":             "No",
    "Dependents":          "No",
    "Tenure Months":       2,
    "Phone Service":       "Yes",
    "Multiple Lines":      "No",
    "Internet Service":    "Fiber optic",
    "Online Security":     "No",
    "Online Backup":       "No",
    "Device Protection":   "No",
    "Tech Support":        "No",
    "Streaming TV":        "Yes",
    "Streaming Movies":    "Yes",
    "Contract":            "Month-to-month",
    "Paperless Billing":   "Yes",
    "Payment Method":      "Electronic check",
    "Monthly Charges":     89.90,
    "Total Charges":       179.80,
}


def load_model(model_path=MODEL_DIR / "lgbm_best.joblib"):
    if not model_path.exists():
        raise FileNotFoundError(f"Model not found at {model_path}. Run train.py first.")
    return joblib.load(model_path)


def predict(pipe, df):
    df = df.copy()
    probs            = pipe.predict_proba(df)[:, 1]
    preds            = pipe.predict(df)
    df["churn_prob"] = np.round(probs, 4)
    df["churn_pred"] = preds
    df["risk_tier"]  = pd.cut(
        probs, bins=[0, 0.30, 0.60, 1.0], labels=["Low", "Medium", "High"]
    )
    return df


def run_demo():
    print("\n-- Demo: single high-risk customer --")
    pipe = load_model()
    df   = pd.DataFrame([DEMO_CUSTOMER])
    out  = predict(pipe, df)
    prob = out["churn_prob"].iloc[0]
    pred = out["churn_pred"].iloc[0]
    tier = out["risk_tier"].iloc[0]

    print("\n  Customer profile:")
    for k, v in DEMO_CUSTOMER.items():
        print(f"    {k:<22} {v}")
    print(f"\n  Churn probability : {prob:.2%}")
    print(f"  Prediction        : {'CHURN' if pred == 1 else 'RETAIN'}")
    print(f"  Risk tier         : {tier}\n")


def run_batch(input_path, output_path):
    print(f"\nLoading customers from: {input_path}")
    df   = pd.read_csv(input_path)
    pipe = load_model()
    out  = predict(pipe, df)
    out.to_csv(output_path, index=False)
    print(f"Predictions saved -> {output_path}")
    print("\n-- Risk distribution --")
    print(out["risk_tier"].value_counts().to_string())
    print(f"\nMean churn probability: {out['churn_prob'].mean():.2%}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--demo",   action="store_true")
    parser.add_argument("--input",  type=Path)
    parser.add_argument("--output", type=Path, default=OUTPUT_DIR / "predictions.csv")
    args = parser.parse_args()

    if args.demo:
        run_demo()
    elif args.input:
        run_batch(args.input, args.output)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
