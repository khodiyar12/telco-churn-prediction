"""
train.py - Model training: Logistic Regression baseline + LightGBM with Optuna.

Workflow:
  1. Load data -> stratified train/test split
  2. Train Logistic Regression baseline (sklearn Pipeline)
  3. Optuna search for LightGBM hyperparameters
  4. Evaluate both models: Accuracy, F1, ROC-AUC, full classification report
  5. Save best model to models/lgbm_best.joblib

Usage:
    python src/train.py
    python src/train.py --no-optuna        # skip tuning, use default LightGBM
"""

import argparse
import json
import warnings
from pathlib import Path
import sys

import joblib
import numpy as np
import optuna
import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, classification_report,
    f1_score, roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.pipeline import Pipeline

warnings.filterwarnings("ignore", category=UserWarning)
optuna.logging.set_verbosity(optuna.logging.WARNING)

sys.path.insert(0, str(Path(__file__).resolve().parent))
from config import (
    CV_FOLDS, LGBM_PARAM_SPACE, MODEL_DIR,
    OPTUNA_TRIALS, RANDOM_STATE, SCORING,
)
from features import build_preprocessor, get_train_test

MODEL_DIR.mkdir(parents=True, exist_ok=True)

# -- Evaluation helper --

def evaluate(name: str, pipeline: Pipeline, X_test, y_test) -> dict:
    y_pred = pipeline.predict(X_test)
    y_prob = pipeline.predict_proba(X_test)[:, 1]

    acc     = accuracy_score(y_test, y_pred)
    f1      = f1_score(y_test, y_pred)
    roc_auc = roc_auc_score(y_test, y_prob)

    print(f"\n{'-'*50}")
    print(f"  {name}")
    print(f"{'-'*50}")
    print(f"  Accuracy : {acc:.4f}")
    print(f"  F1-Score : {f1:.4f}")
    print(f"  ROC-AUC  : {roc_auc:.4f}")
    print(f"\n{classification_report(y_test, y_pred, target_names=['Retained','Churned'])}")

    return {"model": name, "accuracy": acc, "f1": f1, "roc_auc": roc_auc}


# -- Baseline: Logistic Regression --

def train_baseline(X_train, y_train) -> Pipeline:
    print("\n[1/3] Training Logistic Regression baseline ...")
    prep  = build_preprocessor()
    model = LogisticRegression(
        max_iter=1000, class_weight="balanced",
        random_state=RANDOM_STATE, solver="lbfgs",
    )
    pipe = Pipeline([("prep", prep), ("clf", model)])

    cv = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_STATE)
    scores = cross_val_score(pipe, X_train, y_train, cv=cv, scoring=SCORING, n_jobs=-1)
    print(f"  CV ROC-AUC: {scores.mean():.4f} ± {scores.std():.4f}")

    pipe.fit(X_train, y_train)
    return pipe


# -- LightGBM with Optuna --

def _make_objective(X_train, y_train):
    cv = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_STATE)

    def objective(trial: optuna.Trial) -> float:
        space = LGBM_PARAM_SPACE
        params = {
            "n_estimators":      trial.suggest_int("n_estimators",   *space["n_estimators"]),
            "learning_rate":     trial.suggest_float("learning_rate", *space["learning_rate"], log=True),
            "num_leaves":        trial.suggest_int("num_leaves",      *space["num_leaves"]),
            "max_depth":         trial.suggest_int("max_depth",       *space["max_depth"]),
            "min_child_samples": trial.suggest_int("min_child_samples", *space["min_child_samples"]),
            "subsample":         trial.suggest_float("subsample",     *space["subsample"]),
            "colsample_bytree":  trial.suggest_float("colsample_bytree", *space["colsample_bytree"]),
            "reg_alpha":         trial.suggest_float("reg_alpha",     *space["reg_alpha"], log=True),
            "reg_lambda":        trial.suggest_float("reg_lambda",    *space["reg_lambda"], log=True),
            "class_weight":      "balanced",
            "random_state":      RANDOM_STATE,
            "n_jobs":            -1,
            "verbose":           -1,
        }
        prep = build_preprocessor()
        pipe = Pipeline([("prep", prep), ("clf", LGBMClassifier(**params))])
        scores = cross_val_score(pipe, X_train, y_train, cv=cv,
                                 scoring=SCORING, n_jobs=1)
        return scores.mean()

    return objective


def train_lgbm(X_train, y_train, n_trials: int = OPTUNA_TRIALS) -> Pipeline:
    print(f"\n[2/3] Optuna hyperparameter search - {n_trials} trials ...")

    study = optuna.create_study(
        direction="maximize",
        sampler=optuna.samplers.TPESampler(seed=RANDOM_STATE),
        pruner=optuna.pruners.MedianPruner(n_startup_trials=10),
    )
    study.optimize(_make_objective(X_train, y_train),
                   n_trials=n_trials, show_progress_bar=True)

    best = study.best_params
    print(f"\n  Best ROC-AUC: {study.best_value:.4f}")
    print(f"  Best params: {json.dumps(best, indent=4)}")

    # Save best params for reproducibility
    params_path = MODEL_DIR / "lgbm_best_params.json"
    with open(params_path, "w") as f:
        json.dump(best, f, indent=4)
    print(f"  Params saved -> {params_path}")

    prep = build_preprocessor()
    clf  = LGBMClassifier(
        **best,
        class_weight="balanced",
        random_state=RANDOM_STATE,
        n_jobs=-1,
        verbose=-1,
    )
    pipe = Pipeline([("prep", prep), ("clf", clf)])
    pipe.fit(X_train, y_train)
    return pipe


def train_lgbm_default(X_train, y_train) -> Pipeline:
    """Fast path: LightGBM with sensible defaults, no tuning."""
    print("\n[2/3] Training LightGBM with default params (no Optuna) ...")
    prep = build_preprocessor()
    clf  = LGBMClassifier(
        n_estimators=500, learning_rate=0.05, num_leaves=63,
        class_weight="balanced", random_state=RANDOM_STATE,
        n_jobs=-1, verbose=-1,
    )
    pipe = Pipeline([("prep", prep), ("clf", clf)])

    cv = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_STATE)
    scores = cross_val_score(pipe, X_train, y_train, cv=cv, scoring=SCORING, n_jobs=-1)
    print(f"  CV ROC-AUC: {scores.mean():.4f} ± {scores.std():.4f}")

    pipe.fit(X_train, y_train)
    return pipe


# -- Main --

def main(use_optuna: bool = True) -> None:
    print("=" * 55)
    print("  TELCO CHURN - MODEL TRAINING")
    print("=" * 55)

    X_train, X_test, y_train, y_test = get_train_test()

    results = []

    # Baseline
    lr_pipe = train_baseline(X_train, y_train)
    results.append(evaluate("Logistic Regression (baseline)", lr_pipe, X_test, y_test))
    joblib.dump(lr_pipe, MODEL_DIR / "logreg_baseline.joblib")

    # LightGBM
    print(f"\n[2/3] LightGBM {'+ Optuna tuning' if use_optuna else '(defaults)'} ...")
    lgbm_pipe = (train_lgbm(X_train, y_train) if use_optuna
                 else train_lgbm_default(X_train, y_train))
    results.append(evaluate("LightGBM", lgbm_pipe, X_test, y_test))

    # Save best model
    model_path = MODEL_DIR / "lgbm_best.joblib"
    joblib.dump(lgbm_pipe, model_path)
    print(f"\n[3/3] Best model saved -> {model_path}")

    # Results table
    print("\n" + "=" * 55)
    print("  RESULTS SUMMARY")
    print("=" * 55)
    results_df = pd.DataFrame(results).set_index("model")
    print(results_df.to_string(float_format="{:.4f}".format))
    results_df.to_csv(MODEL_DIR / "results.csv")
    print(f"\n  Results table saved -> {MODEL_DIR / 'results.csv'}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--no-optuna", action="store_true",
                        help="Skip Optuna tuning; use default LightGBM params")
    args = parser.parse_args()
    main(use_optuna=not args.no_optuna)
