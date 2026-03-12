"""
Kaggle inference script for Approach-05 (Hybrid: structural + Node2Vec).

Usage
-----
  python predict.py

  # Override learned threshold
  python predict.py --threshold 0.45

Output
------
  data/processed/approach05/predictions.csv  — Id, Predictions
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import sys

import joblib
import numpy as np
import pandas as pd

# ── path setup ────────────────────────────────────────────────────────────────
_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)

from config import PATHS, RANDOM_SEED

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _load_feature_names() -> list[str]:
    fn_path = PATHS["feature_names"]
    if not os.path.exists(fn_path):
        raise FileNotFoundError(
            f"Feature names file not found: {fn_path}\n"
            "Run dataset_builder.py then train.py first."
        )
    with open(fn_path) as f:
        return json.load(f)


def _load_threshold(threshold_path: str, override: float | None) -> float:
    if override is not None:
        logger.info("Using provided threshold: %.4f", override)
        return override
    if os.path.exists(threshold_path):
        with open(threshold_path) as f:
            thr = json.load(f)["threshold"]
        logger.info("Loaded learned threshold: %.4f", thr)
        return thr
    logger.warning("Threshold file not found; using 0.5")
    return 0.5


# ─────────────────────────────────────────────────────────────────────────────
# Main predict
# ─────────────────────────────────────────────────────────────────────────────

def predict(
    model_path: str = PATHS["model"],
    test_feats_path: str = PATHS["test_feats"],
    threshold_path: str = PATHS["threshold"],
    out_path: str = PATHS["predictions"],
    threshold_override: float | None = None,
) -> None:

    # ── 1. Load model ─────────────────────────────────────────────────
    if not os.path.exists(model_path):
        raise FileNotFoundError(
            f"Model not found at {model_path}. Run train.py first."
        )
    logger.info("Loading model: %s", model_path)
    clf = joblib.load(model_path)

    FEATURE_NAMES = _load_feature_names()
    threshold = _load_threshold(threshold_path, threshold_override)

    # ── 2. Load test features (built by dataset_builder) ─────────────
    if not os.path.exists(test_feats_path):
        raise FileNotFoundError(
            f"Test features not found: {test_feats_path}\n"
            "Run dataset_builder.py first."
        )
    logger.info("Loading test features: %s", test_feats_path)
    df_test = pd.read_parquet(test_feats_path)
    logger.info("  Loaded %d test pairs × %d cols", len(df_test), len(df_test.columns))

    # Guard against missing feature columns
    missing = [c for c in FEATURE_NAMES if c not in df_test.columns]
    if missing:
        raise ValueError(
            f"Test parquet missing {len(missing)} feature columns: {missing[:5]} …\n"
            "Regenerate with dataset_builder.py."
        )

    # ── 3. Predict probabilities ──────────────────────────────────────
    X_test = df_test[FEATURE_NAMES].to_numpy(dtype=np.float32)
    logger.info("Scoring %d test pairs with %d features …", len(X_test), len(FEATURE_NAMES))
    proba = clf.predict_proba(X_test)[:, 1]

    # ── 4. Apply threshold ────────────────────────────────────────────
    predictions = (proba >= threshold).astype(int)

    pos_count = predictions.sum()
    logger.info(
        "Predictions: total=%d  positive=%d (%.1f%%)  threshold=%.4f",
        len(predictions), pos_count, 100 * pos_count / len(predictions), threshold,
    )

    # ── 5. Write submission CSV ───────────────────────────────────────
    ids = (
        df_test["Id"].to_numpy(dtype=np.int64)
        if "Id" in df_test.columns
        else np.arange(1, len(predictions) + 1)
    )

    out_df = pd.DataFrame({"Id": ids, "Predictions": predictions})
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    out_df.to_csv(out_path, index=False)
    logger.info("Submission written → %s", out_path)

    # ── 6. Save with probabilities for downstream stacking ───────────
    proba_path = out_path.replace(".csv", "_with_proba.csv")
    pd.DataFrame({
        "Id": ids,
        "Predictions": predictions,
        "probability": proba.astype(np.float32),
    }).to_csv(proba_path, index=False)
    logger.info("Probabilities saved → %s", proba_path)


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Approach-05 Kaggle inference")
    p.add_argument(
        "--threshold", type=float, default=None,
        help="Override learned threshold (default: use models/approach05/threshold.json)",
    )
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    predict(threshold_override=args.threshold)
