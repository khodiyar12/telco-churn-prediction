"""
segment_report.py — Customer segment risk report.

Groups customers into Low / Medium / High churn risk tiers
and computes business metrics per segment.

Outputs:
  outputs/segment_report.csv     — full segment stats table
  outputs/segment_summary.png    — visual summary chart

Usage:
    python src/segment_report.py
"""

import warnings
from pathlib import Path
import sys

import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick

warnings.filterwarnings("ignore")

sys.path.insert(0, str(Path(__file__).resolve().parent))
from config import MODEL_DIR, OUTPUT_DIR, TARGET, CLEAN_CSV
from feature_engineering import add_features

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

BG     = "#fafaf8"
COLORS = {"Low": "#0f6e56", "Medium": "#ba7517", "High": "#d85a30"}

plt.rcParams.update({
    "figure.facecolor": BG, "axes.facecolor": BG,
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.spines.left": False, "grid.color": "#e8e6e0",
    "grid.linewidth": 0.6, "axes.grid": True,
    "xtick.color": "#9c9990", "ytick.color": "#9c9990",
})


def load_data_with_predictions():
    """Load clean data, apply features, predict churn probabilities."""
    model_path = MODEL_DIR / "best_model.joblib"
    if not model_path.exists():
        model_path = MODEL_DIR / "lgbm_best.joblib"
    if not model_path.exists():
        raise FileNotFoundError("No trained model found. Run train.py first.")

    pipe = joblib.load(model_path)
    df   = pd.read_csv(CLEAN_CSV)
    df   = add_features(df)

    # Build feature matrix same way as training
    from config import NUMERIC_COLS, CATEGORICAL_COLS
    feature_cols = [c for c in NUMERIC_COLS + CATEGORICAL_COLS if c in df.columns]
    X = df[feature_cols]

    df["churn_prob"] = pipe.predict_proba(X)[:, 1].round(4)
    df["risk_tier"]  = pd.cut(
        df["churn_prob"],
        bins=[0, 0.30, 0.60, 1.0],
        labels=["Low", "Medium", "High"],
    )
    return df


def build_segment_table(df: pd.DataFrame) -> pd.DataFrame:
    """Compute business metrics per risk tier."""
    agg = df.groupby("risk_tier", observed=True).agg(
        num_customers      = (TARGET, "count"),
        actual_churn_rate  = (TARGET, "mean"),
        avg_churn_prob     = ("churn_prob", "mean"),
        avg_monthly_charges= ("Monthly Charges", "mean"),
        avg_tenure_months  = ("Tenure Months", "mean"),
        avg_total_charges  = ("Total Charges", "mean"),
        total_monthly_rev  = ("Monthly Charges", "sum"),
    ).round(2)

    agg["pct_of_customers"] = (agg["num_customers"] / agg["num_customers"].sum() * 100).round(1)
    agg["revenue_at_risk"]  = (agg["total_monthly_rev"] * agg["avg_churn_prob"]).round(0)

    # Most common contract per tier
    contract_mode = (
        df.groupby("risk_tier", observed=True)["Contract"]
          .agg(lambda x: x.value_counts().index[0])
    )
    agg["dominant_contract"] = contract_mode

    return agg[["num_customers", "pct_of_customers", "actual_churn_rate",
                "avg_churn_prob", "avg_monthly_charges", "avg_tenure_months",
                "total_monthly_rev", "revenue_at_risk", "dominant_contract"]]


def plot_segment_summary(df: pd.DataFrame, seg: pd.DataFrame) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(14, 5))
    fig.patch.set_facecolor(BG)
    tiers  = ["Low", "Medium", "High"]
    colors = [COLORS[t] for t in tiers]

    # Customer count
    ax = axes[0]
    counts = [seg.loc[t, "num_customers"] for t in tiers]
    bars = ax.bar(tiers, counts, color=colors, width=0.45, zorder=3)
    for bar, v in zip(bars, counts):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 20,
                f"{v:,}", ha="center", fontsize=11, color=bar.get_facecolor())
    ax.set_title("Customers per risk tier", fontsize=12, fontweight="bold")
    ax.set_ylabel("Customers")
    ax.yaxis.set_major_formatter(mtick.FuncFormatter(lambda x, _: f"{int(x):,}"))

    # Avg monthly charges
    ax = axes[1]
    charges = [seg.loc[t, "avg_monthly_charges"] for t in tiers]
    bars = ax.bar(tiers, charges, color=colors, width=0.45, zorder=3)
    for bar, v in zip(bars, charges):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                f"${v:.0f}", ha="center", fontsize=11, color=bar.get_facecolor())
    ax.set_title("Avg monthly charges by tier", fontsize=12, fontweight="bold")
    ax.set_ylabel("Monthly Charges (USD)")

    # Revenue at risk
    ax = axes[2]
    rev = [seg.loc[t, "revenue_at_risk"] for t in tiers]
    bars = ax.bar(tiers, rev, color=colors, width=0.45, zorder=3)
    for bar, v in zip(bars, rev):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 200,
                f"${v:,.0f}", ha="center", fontsize=10, color=bar.get_facecolor())
    ax.set_title("Monthly revenue at risk by tier", fontsize=12, fontweight="bold")
    ax.set_ylabel("Revenue at risk (USD)")
    ax.yaxis.set_major_formatter(mtick.FuncFormatter(lambda x, _: f"${int(x):,}"))

    for ax in axes:
        ax.set_axisbelow(True)
        ax.spines[["top","right","left"]].set_visible(False)

    fig.suptitle("Customer Churn Risk Segment Report", fontsize=14,
                 fontweight="bold", y=1.02)
    plt.tight_layout()
    out = OUTPUT_DIR / "segment_summary.png"
    plt.savefig(out, dpi=150, bbox_inches="tight", facecolor=BG)
    plt.close()
    print(f"   Saved -> {out}")


def print_recommendations(seg: pd.DataFrame) -> None:
    print("\n── Retention recommendations ─────────────────────────")
    high = seg.loc["High"]
    med  = seg.loc["Medium"]
    print(f"  HIGH risk  ({high['num_customers']:,} customers)")
    print(f"    Avg churn prob : {high['avg_churn_prob']:.0%}")
    print(f"    Revenue at risk: ${high['revenue_at_risk']:,.0f}/month")
    print(f"    Action         : Immediate outreach — offer contract upgrade + discount")
    print(f"\n  MEDIUM risk ({med['num_customers']:,} customers)")
    print(f"    Avg churn prob : {med['avg_churn_prob']:.0%}")
    print(f"    Revenue at risk: ${med['revenue_at_risk']:,.0f}/month")
    print(f"    Action         : Loyalty programme, auto-pay incentive")
    print(f"\n  LOW risk    ({seg.loc['Low','num_customers']:,} customers)")
    print(f"    Action         : Upsell additional services")


def run_segment_report():
    print("=" * 55)
    print("  CUSTOMER SEGMENT RISK REPORT")
    print("=" * 55)

    df  = load_data_with_predictions()
    seg = build_segment_table(df)

    print("\n── Segment table ─────────────────────────────────────")
    print(seg.to_string())

    out_csv = OUTPUT_DIR / "segment_report.csv"
    seg.to_csv(out_csv)
    print(f"\n   Saved -> {out_csv}")

    plot_segment_summary(df, seg)
    print_recommendations(seg)
    print(f"\n✓ Segment report complete\n")


if __name__ == "__main__":
    run_segment_report()
