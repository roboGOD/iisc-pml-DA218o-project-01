"""
Train a tree-based binary classifier on pre-built structural feature tables.

Usage
-----
  # Build feature tables first (if not cached)
  python dataset_builder.py

  # Train with default config (LightGBM)
  python train.py

  # Train with specific classifier
  python train.py --classifier hgb
  python train.py --classifier lgbm
  python train.py --classifier xgb

Artifacts saved
---------------
  models/approach04/model.joblib
  models/approach04/feature_names.json
  models/approach04/metrics.json
  models/approach04/threshold.json
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
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.metrics import (
    average_precision_score,
    f1_score,
    roc_auc_score,
)

from config import (
    CLASSIFIER,
    HGB_CONFIG,
    LGBM_CONFIG,
    PATHS,
    RANDOM_SEED,
    RF_CONFIG,
    XGB_CONFIG,
)
from structural_features import FEATURE_NAMES

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# Classifier factory
# ─────────────────────────────────────────────────────────────────────────────

def _build_classifier(name: str) -> Any:
    name = name.lower()
    if name == "lgbm":
        try:
            from lightgbm import LGBMClassifier
        except ImportError:
            logger.warning("LightGBM not available, falling back to HGB")
            return HistGradientBoostingClassifier(**HGB_CONFIG)
        cfg = LGBM_CONFIG.copy()
        n_est = cfg.pop("n_estimators")
        return LGBMClassifier(n_estimators=n_est, **cfg)

    if name == "xgb":
        try:
            from xgboost import XGBClassifier
        except ImportError:
            logger.warning("XGBoost not available, falling back to HGB")
            return HistGradientBoostingClassifier(**HGB_CONFIG)
        cfg = XGB_CONFIG.copy()
        return XGBClassifier(**cfg)

    if name == "hgb":
        return HistGradientBoostingClassifier(**HGB_CONFIG)

    if name == "rf":
        return RandomForestClassifier(**RF_CONFIG)

    raise ValueError(f"Unknown classifier: {name!r}. Choose lgbm/xgb/hgb/rf")


# ─────────────────────────────────────────────────────────────────────────────
# Threshold search
# ─────────────────────────────────────────────────────────────────────────────

def find_best_threshold(
    proba: np.ndarray, labels: np.ndarray, step: float = 0.01
) -> tuple[float, float]:
    """
    Search threshold in [0.05, 0.95] that maximises F1 on validation set.

    Returns (best_threshold, best_f1).
    """
    best_thr, best_f1 = 0.5, 0.0
    for thr in np.arange(0.05, 0.96, step):
        preds = (proba >= thr).astype(int)
        f1 = f1_score(labels, preds, zero_division=0)
        if f1 > best_f1:
            best_f1 = f1
            best_thr = float(thr)
    return best_thr, best_f1


# ─────────────────────────────────────────────────────────────────────────────
# Main training logic
# ─────────────────────────────────────────────────────────────────────────────

def train(
    train_feats_path: str = PATHS["train_feats"],
    val_feats_path: str = PATHS["val_feats"],
    model_path: str = PATHS["model"],
    feature_names_path: str = PATHS["feature_names"],
    metrics_path: str = PATHS["metrics"],
    threshold_path: str = PATHS["threshold"],
    classifier_name: str = CLASSIFIER,
    rebuild_if_missing: bool = True,
) -> None:
    t_total = time.time()

    # ── 0. Auto-build datasets if missing ────────────────────────────
    if rebuild_if_missing and not os.path.exists(train_feats_path):
        logger.info("Feature tables not found — running dataset_builder first …")
        from dataset_builder import build_datasets
        build_datasets()

    # ── 1. Load feature tables ────────────────────────────────────────
    logger.info("Loading feature tables …")
    df_train = pd.read_parquet(train_feats_path)
    df_val   = pd.read_parquet(val_feats_path)

    logger.info(
        "  train: %d rows   val: %d rows   features: %d",
        len(df_train), len(df_val), len(FEATURE_NAMES),
    )

    X_train = df_train[FEATURE_NAMES].to_numpy(dtype=np.float32)
    y_train = df_train["label"].to_numpy(dtype=np.int8)
    X_val   = df_val[FEATURE_NAMES].to_numpy(dtype=np.float32)
    y_val   = df_val["label"].to_numpy(dtype=np.int8)

    logger.info(
        "  train pos=%.1f%%  val pos=%.1f%%",
        100 * y_train.mean(), 100 * y_val.mean(),
    )

    # ── 2. Baseline: Logistic Regression ─────────────────────────────
    logger.info("Running Logistic Regression baseline …")
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler
    from sklearn.pipeline import Pipeline

    lr = Pipeline([
        ("scaler", StandardScaler()),
        ("lr", LogisticRegression(max_iter=500, C=1.0, random_state=RANDOM_SEED)),
    ])
    lr.fit(X_train, y_train)
    lr_proba = lr.predict_proba(X_val)[:, 1]
    lr_auc = roc_auc_score(y_val, lr_proba)
    lr_ap  = average_precision_score(y_val, lr_proba)
    logger.info("  [LR baseline]  AUC=%.4f  AP=%.4f", lr_auc, lr_ap)

    # ── 3. Main classifier ────────────────────────────────────────────
    logger.info("Training %s classifier …", classifier_name.upper())
    clf = _build_classifier(classifier_name)

    t0 = time.time()

    # LightGBM and XGBoost accept early stopping via eval_set
    if classifier_name.lower() in ("lgbm",):
        try:
            from lightgbm import LGBMClassifier, early_stopping, log_evaluation
            clf.fit(
                X_train, y_train,
                eval_set=[(X_val, y_val)],
                callbacks=[
                    early_stopping(50, verbose=False),
                    log_evaluation(100),
                ],
            )
        except Exception:
            clf.fit(X_train, y_train)
    elif classifier_name.lower() == "xgb":
        try:
            clf.fit(
                X_train, y_train,
                eval_set=[(X_val, y_val)],
                verbose=100,
            )
        except Exception:
            clf.fit(X_train, y_train)
    else:
        clf.fit(X_train, y_train)

    train_time = time.time() - t0
    logger.info("  Training complete in %.1f s", train_time)

    # ── 4. Validation metrics ─────────────────────────────────────────
    logger.info("Evaluating on validation set …")
    val_proba = clf.predict_proba(X_val)[:, 1]
    auc = roc_auc_score(y_val, val_proba)
    ap  = average_precision_score(y_val, val_proba)
    best_thr, best_f1 = find_best_threshold(val_proba, y_val)

    logger.info("  [%s]  AUC=%.4f  AP=%.4f  F1=%.4f  thr=%.2f",
                classifier_name.upper(), auc, ap, best_f1, best_thr)
    logger.info("  [LR baseline]  AUC=%.4f (for comparison)", lr_auc)

    metrics = {
        "classifier": classifier_name,
        "val_auc":   round(auc,  6),
        "val_ap":    round(ap,   6),
        "val_f1":    round(best_f1, 6),
        "threshold": round(best_thr, 4),
        "lr_baseline_auc": round(lr_auc, 6),
        "train_rows": int(len(df_train)),
        "val_rows":   int(len(df_val)),
        "features":   FEATURE_NAMES,
        "train_time_s": round(train_time, 2),
    }

    # ── 5. Feature importances (if available) ────────────────────────
    if hasattr(clf, "feature_importances_"):
        imp = dict(zip(FEATURE_NAMES, clf.feature_importances_.tolist()))
        imp_sorted = sorted(imp.items(), key=lambda x: -x[1])
        logger.info("Top-10 feature importances:")
        for feat, score in imp_sorted[:10]:
            logger.info("  %-30s %.6f", feat, score)
        metrics["feature_importances"] = imp

    # ── 6. Save artifacts ─────────────────────────────────────────────
    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    joblib.dump(clf, model_path)
    logger.info("Model saved → %s", model_path)

    with open(feature_names_path, "w") as f:
        json.dump(FEATURE_NAMES, f, indent=2)

    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)
    logger.info("Metrics saved → %s", metrics_path)

    with open(threshold_path, "w") as f:
        json.dump({"threshold": best_thr}, f, indent=2)
    logger.info("Threshold saved → %s  (%.4f)", threshold_path, best_thr)

    total_elapsed = time.time() - t_total
    logger.info("Training pipeline complete in %.1f s", total_elapsed)
    logger.info("=" * 60)
    logger.info("  Final:  AUC=%.4f   AP=%.4f   F1=%.4f   threshold=%.2f",
                auc, ap, best_f1, best_thr)
    logger.info("=" * 60)


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train Approach-04 structural GBDT")
    p.add_argument(
        "--classifier", default=CLASSIFIER,
        choices=["lgbm", "xgb", "hgb", "rf"],
        help="Classifier backend (default: %(default)s)",
    )
    p.add_argument("--seed", type=int, default=RANDOM_SEED)
    p.add_argument(
        "--no-rebuild", action="store_true",
        help="Fail if feature tables are missing instead of rebuilding",
    )
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    train(
        classifier_name=args.classifier,
        rebuild_if_missing=not args.no_rebuild,
    )
