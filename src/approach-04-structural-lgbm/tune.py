"""
Hyperparameter tuning for Approach-04 XGBoost classifier.

Uses the pre-built parquets (no dataset rebuild needed).
Evaluates each trial on the fixed val split using early stopping —
no cross-validation folds, so each trial completes in ~2-5 min on H100.

Requires Optuna (smart Bayesian search):
    pip install optuna

Falls back to a manual grid if Optuna is not installed.

Usage
-----
    # 30 Optuna trials (default)
    python tune.py

    # More trials
    python tune.py --trials 60

    # Manual grid instead of Optuna
    python tune.py --backend grid

    # After tuning, retrain final model with best params and save
    python tune.py --retrain

Artifacts saved
---------------
    models/approach04/tune_best_params.json   — best XGB params found
    models/approach04/tune_history.csv        — all trial results sorted by val AUC
    models/approach04/model.joblib            — (only if --retrain) final model
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from config import PATHS, RANDOM_SEED, XGB_CONFIG
from structural_features import FEATURE_NAMES

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# Data loading
# ─────────────────────────────────────────────────────────────────────────────

def load_data() -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    logger.info("Loading feature parquets …")
    df_train = pd.read_parquet(PATHS["train_feats"])
    df_val   = pd.read_parquet(PATHS["val_feats"])
    logger.info("  train: %d rows   val: %d rows   features: %d",
                len(df_train), len(df_val), len(FEATURE_NAMES))

    X_train = df_train[FEATURE_NAMES].to_numpy(dtype=np.float32)
    y_train = df_train["label"].to_numpy(dtype=np.int8)
    X_val   = df_val[FEATURE_NAMES].to_numpy(dtype=np.float32)
    y_val   = df_val["label"].to_numpy(dtype=np.int8)
    return X_train, y_train, X_val, y_val


# ─────────────────────────────────────────────────────────────────────────────
# Single trial evaluation
# ─────────────────────────────────────────────────────────────────────────────

def _eval_params(
    params: dict[str, Any],
    X_train: np.ndarray, y_train: np.ndarray,
    X_val: np.ndarray, y_val: np.ndarray,
    n_estimators_max: int = 2000,
    early_stopping_rounds: int = 50,
) -> tuple[float, int]:
    """
    Train XGB with given params using early stopping on val AUC.
    Returns (val_auc, best_n_estimators).
    """
    from xgboost import XGBClassifier

    clf = XGBClassifier(
        **params,
        n_estimators=n_estimators_max,
        objective="binary:logistic",
        eval_metric="auc",
        use_label_encoder=False,
        tree_method="hist",
        device="cuda",
        random_state=RANDOM_SEED,
        early_stopping_rounds=early_stopping_rounds,
        verbosity=0,
    )
    clf.fit(
        X_train, y_train,
        eval_set=[(X_val, y_val)],
        verbose=False,
    )
    best_n = int(clf.best_iteration + 1)
    val_proba = clf.predict_proba(X_val)[:, 1]
    auc = float(roc_auc_score(y_val, val_proba))
    return auc, best_n


# ─────────────────────────────────────────────────────────────────────────────
# Optuna search
# ─────────────────────────────────────────────────────────────────────────────

def run_optuna(
    X_train, y_train, X_val, y_val,
    n_trials: int = 30,
) -> list[dict]:
    import optuna
    optuna.logging.set_verbosity(optuna.logging.WARNING)

    results: list[dict] = []

    def objective(trial: optuna.Trial) -> float:
        params = {
            "max_depth":        trial.suggest_int("max_depth", 3, 5),  # depth=4 dominated on 1M; narrow range
            "min_child_weight": trial.suggest_int("min_child_weight", 1, 20),
            "subsample":        trial.suggest_float("subsample", 0.6, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
            "learning_rate":    trial.suggest_float("learning_rate", 0.01, 0.15, log=True),
            "reg_alpha":        trial.suggest_float("reg_alpha", 1e-4, 2.0, log=True),
            "reg_lambda":       trial.suggest_float("reg_lambda", 0.1, 5.0, log=True),
            "gamma":            trial.suggest_float("gamma", 0.0, 1.0),
        }
        t0 = time.time()
        auc, n_est = _eval_params(params, X_train, y_train, X_val, y_val)
        elapsed = time.time() - t0
        logger.info(
            "Trial %3d | AUC=%.4f | n_est=%4d | depth=%d | lr=%.4f | "
            "subsample=%.2f | colsample=%.2f | min_cw=%2d | elapsed=%.0fs",
            trial.number, auc, n_est,
            params["max_depth"], params["learning_rate"],
            params["subsample"], params["colsample_bytree"],
            params["min_child_weight"], elapsed,
        )
        results.append({**params, "n_estimators": n_est, "val_auc": auc})
        return auc

    study = optuna.create_study(
        direction="maximize",
        sampler=optuna.samplers.TPESampler(seed=RANDOM_SEED),
    )
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)
    return results


# ─────────────────────────────────────────────────────────────────────────────
# Manual grid search (fallback)
# ─────────────────────────────────────────────────────────────────────────────

MANUAL_GRID = [
    # (max_depth, min_child_weight, subsample, colsample_bytree, lr, reg_alpha, reg_lambda, gamma)
    # Baseline (current config)
    {"max_depth": 6, "min_child_weight": 5,  "subsample": 0.8, "colsample_bytree": 0.8, "learning_rate": 0.05, "reg_alpha": 0.1,  "reg_lambda": 1.0, "gamma": 0.0},
    # Shallower trees (less overfit)
    {"max_depth": 4, "min_child_weight": 5,  "subsample": 0.8, "colsample_bytree": 0.8, "learning_rate": 0.05, "reg_alpha": 0.1,  "reg_lambda": 1.0, "gamma": 0.0},
    {"max_depth": 4, "min_child_weight": 10, "subsample": 0.8, "colsample_bytree": 0.8, "learning_rate": 0.05, "reg_alpha": 0.1,  "reg_lambda": 1.0, "gamma": 0.0},
    # Deeper + more reg
    {"max_depth": 8, "min_child_weight": 10, "subsample": 0.7, "colsample_bytree": 0.7, "learning_rate": 0.05, "reg_alpha": 0.5,  "reg_lambda": 2.0, "gamma": 0.1},
    # Lower lr + more estimators (early stopping finds right n)
    {"max_depth": 6, "min_child_weight": 5,  "subsample": 0.8, "colsample_bytree": 0.8, "learning_rate": 0.01, "reg_alpha": 0.1,  "reg_lambda": 1.0, "gamma": 0.0},
    {"max_depth": 4, "min_child_weight": 5,  "subsample": 0.9, "colsample_bytree": 0.9, "learning_rate": 0.01, "reg_alpha": 0.05, "reg_lambda": 0.5, "gamma": 0.0},
    # Higher lr (faster, may underfit)
    {"max_depth": 6, "min_child_weight": 5,  "subsample": 0.8, "colsample_bytree": 0.8, "learning_rate": 0.10, "reg_alpha": 0.1,  "reg_lambda": 1.0, "gamma": 0.0},
    # High reg_lambda (L2 penalty)
    {"max_depth": 6, "min_child_weight": 5,  "subsample": 0.8, "colsample_bytree": 0.8, "learning_rate": 0.05, "reg_alpha": 0.0,  "reg_lambda": 5.0, "gamma": 0.0},
    # High gamma (min split loss)
    {"max_depth": 6, "min_child_weight": 5,  "subsample": 0.8, "colsample_bytree": 0.8, "learning_rate": 0.05, "reg_alpha": 0.1,  "reg_lambda": 1.0, "gamma": 0.5},
    # Wider colsample
    {"max_depth": 5, "min_child_weight": 5,  "subsample": 0.8, "colsample_bytree": 1.0, "learning_rate": 0.05, "reg_alpha": 0.1,  "reg_lambda": 1.0, "gamma": 0.0},
]


def run_grid(X_train, y_train, X_val, y_val) -> list[dict]:
    results = []
    for i, params in enumerate(MANUAL_GRID):
        logger.info("Grid point %2d/%d — %s", i + 1, len(MANUAL_GRID), params)
        t0 = time.time()
        auc, n_est = _eval_params(params, X_train, y_train, X_val, y_val)
        elapsed = time.time() - t0
        logger.info("  → AUC=%.4f | n_est=%d | %.0fs", auc, n_est, elapsed)
        results.append({**params, "n_estimators": n_est, "val_auc": auc})
    return results


# ─────────────────────────────────────────────────────────────────────────────
# Save results
# ─────────────────────────────────────────────────────────────────────────────

def _save_results(results: list[dict], best_params: dict) -> None:
    out_dir = os.path.dirname(PATHS["model"])
    os.makedirs(out_dir, exist_ok=True)

    # Full history as CSV, sorted by val_auc descending
    df = pd.DataFrame(results).sort_values("val_auc", ascending=False)
    history_path = os.path.join(out_dir, "tune_history.csv")
    df.to_csv(history_path, index=False)
    logger.info("Trial history saved → %s", history_path)

    # Best params as JSON
    best_path = os.path.join(out_dir, "tune_best_params.json")
    with open(best_path, "w") as f:
        json.dump(best_params, f, indent=2)
    logger.info("Best params saved → %s", best_path)

    # Print top-10
    logger.info("\n═══ Top-10 trials by val AUC ═══")
    for _, row in df.head(10).iterrows():
        logger.info(
            "  AUC=%.4f | depth=%d | lr=%.4f | subsample=%.2f | "
            "colsample=%.2f | min_cw=%2d | n_est=%d",
            row["val_auc"], int(row["max_depth"]), row["learning_rate"],
            row["subsample"], row["colsample_bytree"],
            int(row["min_child_weight"]), int(row["n_estimators"]),
        )


# ─────────────────────────────────────────────────────────────────────────────
# Retrain final model with best params
# ─────────────────────────────────────────────────────────────────────────────

def retrain_best(best_params: dict, X_train, y_train, X_val, y_val) -> None:
    from xgboost import XGBClassifier
    logger.info("\nRetraining final model with best params …")

    # Drop bookkeeping keys — not XGB params
    best_params.pop("n_estimators", None)   # ignore trial n_est; early stopping finds it
    val_auc_ref = best_params.pop("val_auc", None)

    logger.info("  params: %s", best_params)

    # Use a high cap + early stopping so the model trains until it genuinely
    # converges, rather than stopping at the (often too-small) trial n_est.
    clf = XGBClassifier(
        **best_params,
        n_estimators=3000,
        objective="binary:logistic",
        eval_metric="auc",
        use_label_encoder=False,
        tree_method="hist",
        device="cuda",
        early_stopping_rounds=50,
        random_state=RANDOM_SEED,
        verbosity=0,
    )
    clf.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=25)

    val_proba = clf.predict_proba(X_val)[:, 1]
    auc = roc_auc_score(y_val, val_proba)
    logger.info("  Final model val AUC=%.4f  (search best was %.4f)",
                auc, val_auc_ref or 0)

    joblib.dump(clf, PATHS["model"])
    logger.info("Model saved → %s", PATHS["model"])

    # Restore val_auc for completeness
    best_params["val_auc"] = val_auc_ref
    best_params["n_estimators"] = int(clf.best_iteration + 1)


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Approach-04 XGB hyperparameter tuner")
    p.add_argument(
        "--backend", choices=["optuna", "grid"], default="optuna",
        help="Search backend (default: optuna). Use grid if Optuna not installed.",
    )
    p.add_argument(
        "--trials", type=int, default=30,
        help="Number of Optuna trials (ignored for grid search, default: 30)",
    )
    p.add_argument(
        "--retrain", action="store_true",
        help="After tuning, retrain final model with best params and save model.joblib",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    t_wall = time.time()

    logger.info("=" * 65)
    logger.info("  Approach-04 XGB Hyperparameter Tuning")
    logger.info("  Backend: %-10s  Trials: %d", args.backend, args.trials)
    logger.info("=" * 65)

    X_train, y_train, X_val, y_val = load_data()

    if args.backend == "optuna":
        try:
            import optuna  # noqa: F401
        except ImportError:
            logger.warning("Optuna not installed. Falling back to manual grid.")
            logger.warning("  To install: pip install optuna")
            args.backend = "grid"

    if args.backend == "optuna":
        results = run_optuna(X_train, y_train, X_val, y_val, n_trials=args.trials)
    else:
        results = run_grid(X_train, y_train, X_val, y_val)

    # Best trial
    best = max(results, key=lambda r: r["val_auc"])
    logger.info("\n═══ Best params (val AUC=%.4f) ═══", best["val_auc"])
    for k, v in best.items():
        logger.info("  %-20s %s", k, v)

    _save_results(results, best.copy())

    if args.retrain:
        retrain_best(best.copy(), X_train, y_train, X_val, y_val)

    logger.info("\nTotal tuning time: %.1f min", (time.time() - t_wall) / 60)
    logger.info("Run predict.py next to generate submission with best model.")


if __name__ == "__main__":
    main()
