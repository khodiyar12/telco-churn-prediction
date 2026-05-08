"""
eda.py — Exploratory Data Analysis for Telco Churn.

Generates and saves the following plots to outputs/eda/:
  01_churn_distribution.png      — target class balance
  02_churn_by_contract.png       — churn rate by contract type
  03_churn_by_tenure.png         — tenure distribution by churn
  04_churn_by_monthly_charges.png— KDE of MonthlyCharges by churn
  05_churn_by_payment.png        — churn rate by payment method
  06_churn_by_internet.png       — churn rate by internet service
  07_numeric_correlation.png     — correlation heatmap (numeric features)
  08_churn_by_senior.png         — churn rate for senior vs non-senior

Usage:
    python src/eda.py
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import seaborn as sns
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent))
from config import CLEAN_CSV, OUTPUT_DIR, TARGET, NUMERIC_COLS

# ── Style ──────────────────────────────────────────────────────────────────────
PALETTE   = {"Churned": "#d85a30", "Retained": "#0f6e56"}
ACCENT    = "#d85a30"
NEUTRAL   = "#0f6e56"
WARN      = "#ba7517"
BG        = "#fafaf8"
GREY      = "#9c9990"
FONT      = "DejaVu Sans"

plt.rcParams.update({
    "figure.facecolor":  BG,
    "axes.facecolor":    BG,
    "axes.spines.top":   False,
    "axes.spines.right": False,
    "axes.spines.left":  False,
    "axes.edgecolor":    GREY,
    "axes.labelcolor":   "#5a5854",
    "xtick.color":       GREY,
    "ytick.color":       GREY,
    "text.color":        "#0f0e0d",
    "font.family":       FONT,
    "axes.grid":         True,
    "grid.color":        "#e8e6e0",
    "grid.linewidth":    0.6,
})

OUT = OUTPUT_DIR / "eda"
OUT.mkdir(parents=True, exist_ok=True)


def _save(name: str) -> None:
    path = OUT / name
    plt.savefig(path, dpi=150, bbox_inches="tight", facecolor=BG)
    plt.close()
    print(f"   Saved → {path.relative_to(Path.cwd()) if Path.cwd() in path.parents else path}")


# ── Plot functions ─────────────────────────────────────────────────────────────

def plot_churn_distribution(df: pd.DataFrame) -> None:
    """Class balance bar chart."""
    counts  = df[TARGET].value_counts().sort_index()
    labels  = ["Retained (0)", "Churned (1)"]
    colors  = [NEUTRAL, ACCENT]
    total   = len(df)

    fig, ax = plt.subplots(figsize=(6, 4))
    bars = ax.bar(labels, counts.values, color=colors, width=0.45, zorder=3)
    for bar, cnt in zip(bars, counts.values):
        ax.text(bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 50,
                f"{cnt:,}\n({cnt/total:.1%})",
                ha="center", va="bottom", fontsize=11, color=bar.get_facecolor())
    ax.set_title("Churn class distribution", fontsize=13, fontweight="bold", pad=12)
    ax.set_ylabel("Customers")
    ax.yaxis.set_major_formatter(mtick.FuncFormatter(lambda x, _: f"{int(x):,}"))
    ax.set_axisbelow(True)
    _save("01_churn_distribution.png")


def plot_churn_by_contract(df: pd.DataFrame) -> None:
    """Churn rate % grouped bar by contract type."""
    rates = (df.groupby("Contract")[TARGET]
               .mean()
               .sort_values(ascending=False)
               .reset_index())
    rates.columns = ["Contract", "ChurnRate"]

    fig, ax = plt.subplots(figsize=(7, 4))
    colors = [ACCENT, WARN, NEUTRAL]
    bars = ax.barh(rates["Contract"], rates["ChurnRate"] * 100,
                   color=colors[:len(rates)], height=0.5, zorder=3)
    for bar, val in zip(bars, rates["ChurnRate"]):
        ax.text(val * 100 + 0.5, bar.get_y() + bar.get_height() / 2,
                f"{val:.1%}", va="center", fontsize=11, color=bar.get_facecolor())
    ax.set_xlabel("Churn rate (%)")
    ax.set_title("Churn rate by contract type", fontsize=13, fontweight="bold", pad=12)
    ax.xaxis.set_major_formatter(mtick.PercentFormatter())
    ax.set_axisbelow(True)
    _save("02_churn_by_contract.png")


def plot_churn_by_tenure(df: pd.DataFrame) -> None:
    """KDE of tenure split by churn status."""
    fig, ax = plt.subplots(figsize=(8, 4))
    for churn, label, color in [(0, "Retained", NEUTRAL), (1, "Churned", ACCENT)]:
        subset = df.loc[df[TARGET] == churn, "Tenure Months"]
        subset.plot.kde(ax=ax, label=label, color=color, linewidth=2)
        ax.fill_between(
            ax.lines[-1].get_xdata(),
            ax.lines[-1].get_ydata(),
            alpha=0.12, color=color
        )
    ax.set_xlabel("Tenure (months)")
    ax.set_ylabel("Density")
    ax.set_title("Tenure distribution by churn status", fontsize=13, fontweight="bold", pad=12)
    ax.legend(frameon=False)
    ax.set_xlim(0, None)
    _save("03_churn_by_tenure.png")


def plot_churn_by_monthly_charges(df: pd.DataFrame) -> None:
    """KDE of MonthlyCharges by churn status."""
    fig, ax = plt.subplots(figsize=(8, 4))
    for churn, label, color in [(0, "Retained", NEUTRAL), (1, "Churned", ACCENT)]:
        subset = df.loc[df[TARGET] == churn, "Monthly Charges"]
        subset.plot.kde(ax=ax, label=label, color=color, linewidth=2)
        ax.fill_between(
            ax.lines[-1].get_xdata(),
            ax.lines[-1].get_ydata(),
            alpha=0.12, color=color
        )
    ax.set_xlabel("Monthly charges (USD)")
    ax.set_ylabel("Density")
    ax.set_title("Monthly charges distribution by churn status", fontsize=13, fontweight="bold", pad=12)
    ax.legend(frameon=False)
    _save("04_churn_by_monthly_charges.png")


def plot_churn_by_payment(df: pd.DataFrame) -> None:
    """Churn rate by payment method."""
    rates = (df.groupby("Payment Method")[TARGET]
               .mean()
               .sort_values(ascending=False)
               .reset_index())

    fig, ax = plt.subplots(figsize=(8, 4))
    bar_colors = [ACCENT if r > 0.30 else (WARN if r > 0.15 else NEUTRAL)
                  for r in rates[TARGET]]
    bars = ax.barh(rates["Payment Method"], rates[TARGET] * 100,
                   color=bar_colors, height=0.45, zorder=3)
    for bar, val in zip(bars, rates[TARGET]):
        ax.text(val * 100 + 0.3, bar.get_y() + bar.get_height() / 2,
                f"{val:.1%}", va="center", fontsize=10, color=bar.get_facecolor())
    ax.set_xlabel("Churn rate (%)")
    ax.set_title("Churn rate by payment method", fontsize=13, fontweight="bold", pad=12)
    ax.xaxis.set_major_formatter(mtick.PercentFormatter())
    ax.set_axisbelow(True)
    _save("05_churn_by_payment.png")


def plot_churn_by_internet(df: pd.DataFrame) -> None:
    """Churn rate by internet service type."""
    rates = (df.groupby("Internet Service")[TARGET]
               .mean()
               .sort_values(ascending=False)
               .reset_index())

    fig, ax = plt.subplots(figsize=(6, 4))
    colors = [ACCENT, WARN, NEUTRAL]
    bars = ax.bar(rates["Internet Service"], rates[TARGET] * 100,
                  color=colors[:len(rates)], width=0.4, zorder=3)
    for bar, val in zip(bars, rates[TARGET]):
        ax.text(bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.5,
                f"{val:.1%}", ha="center", va="bottom",
                fontsize=11, color=bar.get_facecolor())
    ax.set_ylabel("Churn rate (%)")
    ax.set_title("Churn rate by internet service type", fontsize=13, fontweight="bold", pad=12)
    ax.yaxis.set_major_formatter(mtick.PercentFormatter())
    ax.set_axisbelow(True)
    _save("06_churn_by_internet.png")


def plot_numeric_correlation(df: pd.DataFrame) -> None:
    """Correlation heatmap of numeric features including target."""
    num_df = df[NUMERIC_COLS + [TARGET]]
    corr   = num_df.corr()

    fig, ax = plt.subplots(figsize=(6, 5))
    mask = np.triu(np.ones_like(corr, dtype=bool), k=1)
    sns.heatmap(corr, annot=True, fmt=".2f", cmap="RdYlGn",
                center=0, vmin=-1, vmax=1,
                linewidths=0.5, linecolor=BG,
                ax=ax, mask=~mask ^ np.eye(len(corr), dtype=bool),
                annot_kws={"size": 10})
    ax.set_title("Numeric feature correlations", fontsize=13, fontweight="bold", pad=12)
    fig.tight_layout()
    _save("07_numeric_correlation.png")


def plot_churn_by_senior(df: pd.DataFrame) -> None:
    """Churn rate: senior citizen vs non-senior."""
    rates = (df.groupby("Senior Citizen")[TARGET]
               .mean()
               .reset_index())
    rates["label"] = rates["Senior Citizen"].map({0: "Non-senior", 1: "Senior citizen"})

    fig, ax = plt.subplots(figsize=(5, 4))
    bars = ax.bar(rates["label"], rates[TARGET] * 100,
                  color=[NEUTRAL, ACCENT], width=0.4, zorder=3)
    for bar, val in zip(bars, rates[TARGET]):
        ax.text(bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.4,
                f"{val:.1%}", ha="center", va="bottom",
                fontsize=12, color=bar.get_facecolor())
    ax.set_ylabel("Churn rate (%)")
    ax.set_title("Churn rate: senior vs non-senior", fontsize=13, fontweight="bold", pad=12)
    ax.yaxis.set_major_formatter(mtick.PercentFormatter())
    ax.set_axisbelow(True)
    _save("08_churn_by_senior.png")


# ── Summary stats ──────────────────────────────────────────────────────────────

def print_summary(df: pd.DataFrame) -> None:
    print("\n── Dataset overview ──────────────────────────────────")
    print(df.describe(include="all").T.to_string())
    print("\n── Churn breakdown ───────────────────────────────────")
    print(df[TARGET].value_counts(normalize=True).rename({1: "Churned", 0: "Retained"}).to_string())
    print("\n── Missing values ────────────────────────────────────")
    missing = df.isnull().sum()
    print(missing[missing > 0].to_string() if missing.any() else "  None detected.")


# ── Entry point ────────────────────────────────────────────────────────────────

def run_eda(csv_path: Path = CLEAN_CSV) -> None:
    print(f"Loading clean data from: {csv_path}")
    df = pd.read_csv(csv_path)
    print_summary(df)

    print("\nGenerating EDA plots...")
    plot_churn_distribution(df)
    plot_churn_by_contract(df)
    plot_churn_by_tenure(df)
    plot_churn_by_monthly_charges(df)
    plot_churn_by_payment(df)
    plot_churn_by_internet(df)
    plot_numeric_correlation(df)
    plot_churn_by_senior(df)
    print(f"\n✓ All EDA plots saved to {OUT}\n")


if __name__ == "__main__":
    run_eda()
