"""
Kaggle inference script for Approach-04.

Usage
-----
  python predict.py

  # Use custom threshold instead of the learned one
  python predict.py --threshold 0.45

  # Use pre-built test feature table (skip feature extraction)
  python predict.py --use-cached-feats

Output
------
  data/processed/approach04/predictions.csv  — Id, Predictions
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

from config import PATHS, RANDOM_SEED
from structural_features import FEATURE_NAMES

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
    test_edges_path: str = PATHS["test_edges"],
    test_feats_path: str = PATHS["test_feats"],
    threshold_path: str = PATHS["threshold"],
    out_path: str = PATHS["predictions"],
    threshold_override: float | None = None,
    use_cached_feats: bool = True,
) -> None:

    # ── 1. Load model ─────────────────────────────────────────────────
    if not os.path.exists(model_path):
        raise FileNotFoundError(
            f"Model not found at {model_path}. Run train.py first."
        )
    logger.info("Loading model: %s", model_path)
    clf = joblib.load(model_path)

    threshold = _load_threshold(threshold_path, threshold_override)

    # ── 2. Get test features ──────────────────────────────────────────
    if use_cached_feats and os.path.exists(test_feats_path):
        logger.info("Loading cached test features: %s", test_feats_path)
        df_test = pd.read_parquet(test_feats_path)
    else:
        logger.info("Building test features from scratch …")
        from graph_store import GraphStore
        from structural_features import build_dataframe
        from dataset_builder import read_test_edges

        gs_full = GraphStore.from_adjacency_csv(PATHS["train_graph"])
        test_ids, test_src, test_dst = read_test_edges(test_edges_path)
        test_pairs = np.stack([test_src, test_dst], axis=1).astype(np.int32)
        df_test = build_dataframe(
            gs_full, test_pairs,
            labels=np.zeros(len(test_pairs), dtype=np.int8),
        )
        df_test["Id"] = test_ids
        # Cache for next time
        os.makedirs(os.path.dirname(test_feats_path), exist_ok=True)
        df_test.to_parquet(test_feats_path, index=False)

    # ── 3. Predict probabilities ──────────────────────────────────────
    X_test = df_test[FEATURE_NAMES].to_numpy(dtype=np.float32)
    logger.info("Scoring %d test pairs …", len(X_test))
    proba = clf.predict_proba(X_test)[:, 1]

    # ── 4. Apply threshold ────────────────────────────────────────────
    predictions = (proba >= threshold).astype(int)

    pos_count = predictions.sum()
    logger.info(
        "Predictions: total=%d  positive=%d (%.1f%%)  threshold=%.4f",
        len(predictions), pos_count, 100 * pos_count / len(predictions), threshold,
    )

    # ── 5. Write submission CSV ───────────────────────────────────────
    ids = df_test["Id"].to_numpy(dtype=np.int64) if "Id" in df_test.columns else np.arange(1, len(predictions) + 1)

    out_df = pd.DataFrame({
        "Id": ids,
        "Predictions": predictions,
    })

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    out_df.to_csv(out_path, index=False)
    logger.info("Submission written → %s", out_path)

    # ── 6. Also save with probabilities for hybrid stacking ──────────
    proba_path = out_path.replace(".csv", "_with_proba.csv")
    proba_df = pd.DataFrame({
        "Id":          ids,
        "Predictions": predictions,
        "probability": proba.astype(np.float32),
    })
    proba_df.to_csv(proba_path, index=False)
    logger.info("Probabilities saved → %s (for hybrid stacking)", proba_path)


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Approach-04 Kaggle inference")
    p.add_argument(
        "--threshold", type=float, default=None,
        help="Override learned threshold (default: use models/approach04/threshold.json)",
    )
    p.add_argument(
        "--use-cached-feats", action="store_true", default=True,
        help="Use cached test feature parquet if it exists (default: True)",
    )
    p.add_argument(
        "--recompute-feats", dest="use_cached_feats", action="store_false",
        help="Force recompute test features even if cache exists",
    )
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    predict(
        threshold_override=args.threshold,
        use_cached_feats=args.use_cached_feats,
    )
