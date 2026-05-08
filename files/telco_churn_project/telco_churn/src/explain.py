"""
explain.py — SHAP explainability analysis for the trained LightGBM model.

Generates and saves to outputs/shap/:
  shap_summary_beeswarm.png    — beeswarm plot (global feature importance)
  shap_summary_bar.png         — mean |SHAP| bar chart
  shap_waterfall_sample.png    — waterfall plot for one high-risk customer
  shap_dependence_tenure.png   — SHAP dependence plot for tenure

Usage:
    python src/explain.py
"""

import warnings
from pathlib import Path
import sys

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap

warnings.filterwarnings("ignore")

sys.path.insert(0, str(Path(__file__).resolve().parent))
from config import MODEL_DIR, OUTPUT_DIR, RANDOM_STATE
from features import build_preprocessor, get_feature_names, get_train_test

OUT = OUTPUT_DIR / "shap"
OUT.mkdir(parents=True, exist_ok=True)

BG = "#fafaf8"
plt.rcParams.update({"figure.facecolor": BG, "axes.facecolor": BG})


def _save(name: str) -> None:
    path = OUT / name
    plt.savefig(path, dpi=150, bbox_inches="tight", facecolor=BG)
    plt.close()
    print(f"   Saved → {path}")


def load_model_and_data():
    model_path = MODEL_DIR / "lgbm_best.joblib"
    if not model_path.exists():
        raise FileNotFoundError(
            f"Model not found at {model_path}.\n"
            "Run `python src/train.py` first."
        )
    print(f"Loading model: {model_path}")
    pipe = joblib.load(model_path)

    _, X_test, _, y_test = get_train_test()
    return pipe, X_test, y_test


def get_preprocessed_test(pipe, X_test) -> tuple[np.ndarray, list[str]]:
    """Apply preprocessing and return array + feature names."""
    prep         = pipe.named_steps["prep"]
    X_test_prep  = prep.transform(X_test)
    feature_names = get_feature_names(prep)
    return X_test_prep, feature_names


def compute_shap_values(pipe, X_test_prep: np.ndarray):
    """Compute SHAP values using TreeExplainer."""
    clf      = pipe.named_steps["clf"]
    explainer = shap.TreeExplainer(clf)
    shap_values = explainer.shap_values(X_test_prep)

    # LightGBM binary classification: shap_values is a list [neg_class, pos_class]
    if isinstance(shap_values, list):
        shap_vals = shap_values[1] if isinstance(shap_values, list) else shap_values   # SHAP for Churn=1
    else:
        shap_vals = shap_values
    return explainer, shap_vals


def plot_beeswarm(shap_vals, X_test_prep, feature_names):
    print("  Plotting beeswarm …")
    shap.summary_plot(
        shap_vals, X_test_prep,
        feature_names=feature_names,
        max_display=15,
        show=False,
        plot_size=(10, 7),
    )
    plt.title("SHAP feature importance — beeswarm (LightGBM)",
              fontsize=13, fontweight="bold", pad=12)
    _save("shap_summary_beeswarm.png")


def plot_bar(shap_vals, feature_names):
    print("  Plotting mean |SHAP| bar chart …")
    mean_abs = np.abs(shap_vals).mean(axis=0)
    top_idx  = np.argsort(mean_abs)[::-1][:15]

    fig, ax = plt.subplots(figsize=(8, 6))
    colors  = ["#d85a30" if v > mean_abs.mean() else "#9c9990"
               for v in mean_abs[top_idx]]
    ax.barh(
        [feature_names[i] for i in top_idx[::-1]],
        mean_abs[top_idx[::-1]],
        color=colors[::-1], height=0.6, zorder=3,
    )
    ax.set_xlabel("Mean |SHAP value|")
    ax.set_title("Top 15 features — mean absolute SHAP",
                 fontsize=13, fontweight="bold", pad=12)
    ax.grid(axis="x", color="#e8e6e0", linewidth=0.6)
    ax.spines[["top", "right", "left"]].set_visible(False)
    fig.tight_layout()
    _save("shap_summary_bar.png")


def plot_waterfall(explainer, shap_vals, X_test_prep, feature_names, y_test):
    """Waterfall for the highest-risk predicted churn customer."""
    print("  Plotting waterfall for highest-risk customer …")
    # Pick the customer with the highest SHAP sum (highest churn signal)
    risk_scores = shap_vals.sum(axis=1)
    sample_idx  = int(np.argmax(risk_scores))

    explanation = shap.Explanation(
        values          = shap_vals[sample_idx],
        base_values     = explainer.expected_value if not isinstance(
                             explainer.expected_value, list)
                          else explainer.expected_value[1],
        data            = X_test_prep[sample_idx],
        feature_names   = feature_names,
    )
    shap.waterfall_plot(explanation, max_display=12, show=False)
    plt.title(f"SHAP waterfall — highest-risk customer (idx {sample_idx})",
              fontsize=12, fontweight="bold", pad=10)
    _save("shap_waterfall_sample.png")


def plot_dependence(shap_vals, X_test_prep, feature_names):
    """SHAP dependence plot for tenure vs MonthlyCharges."""
    print("  Plotting SHAP dependence — tenure …")
    try:
        tenure_idx = feature_names.index("Tenure Months")
        charges_idx = feature_names.index("Monthly Charges")
    except ValueError:
        print("  Skipping dependence plot — feature not found after encoding.")
        return

    fig, ax = plt.subplots(figsize=(7, 5))
    sc = ax.scatter(
        X_test_prep[:, tenure_idx],
        shap_vals[:, tenure_idx],
        c=X_test_prep[:, charges_idx],
        cmap="RdYlGn_r", alpha=0.6, s=12, edgecolors="none",
    )
    cbar = plt.colorbar(sc, ax=ax)
    cbar.set_label("MonthlyCharges (scaled)", fontsize=10)
    ax.axhline(0, color="#9c9990", linewidth=0.8, linestyle="--")
    ax.set_xlabel("Tenure (scaled)")
    ax.set_ylabel("SHAP value for tenure")
    ax.set_title("SHAP dependence plot — tenure × MonthlyCharges",
                 fontsize=13, fontweight="bold", pad=12)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    _save("shap_dependence_tenure.png")


def print_insights(shap_vals, feature_names) -> None:
    """Print top positive and negative SHAP drivers."""
    mean_shap = shap_vals.mean(axis=0)
    df = (pd.Series(mean_shap, index=feature_names)
            .sort_values(ascending=False))

    print("\n── Top churn-increasing features ────────────────────")
    print(df.head(8).to_string())
    print("\n── Top churn-decreasing features ────────────────────")
    print(df.tail(8).sort_values().to_string())


def run_shap() -> None:
    print("=" * 55)
    print("  TELCO CHURN — SHAP EXPLAINABILITY")
    print("=" * 55)

    pipe, X_test, y_test = load_model_and_data()
    X_test_prep, feature_names = get_preprocessed_test(pipe, X_test)

    print("\nComputing SHAP values (TreeExplainer) …")
    explainer, shap_vals = compute_shap_values(pipe, X_test_prep)
    print(f"  SHAP matrix shape: {shap_vals.shape}")

    print("\nGenerating plots …")
    plot_beeswarm(shap_vals, X_test_prep, feature_names)
    plot_bar(shap_vals, feature_names)
    plot_waterfall(explainer, shap_vals, X_test_prep, feature_names, y_test)
    plot_dependence(shap_vals, X_test_prep, feature_names)

    print_insights(shap_vals, feature_names)
    print(f"\n✓ All SHAP plots saved to {OUT}\n")


if __name__ == "__main__":
    run_shap()
