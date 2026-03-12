"""
Configuration for Approach-04: Directed Structural Feature Engineering + GBDT.

Node IDs in this dataset are dense integers in [0, 4_867_135], so no ID
mapping dictionary is needed — the array index IS the node ID.
"""
import os as _os

# Project root = two levels above this file (src/approach-04-structural-lgbm/ → project root)
_ROOT = _os.path.abspath(_os.path.join(_os.path.dirname(__file__), "..", ".."))

def _p(*parts: str) -> str:
    """Return an absolute path anchored at the project root."""
    return _os.path.join(_ROOT, *parts)

# ── Reproducibility ────────────────────────────────────────────────────────────
RANDOM_SEED = 42

# ── File paths ─────────────────────────────────────────────────────────────────
PATHS = {
    "train_graph":   _p("data", "raw", "train.csv"),
    "test_edges":    _p("data", "raw", "test.csv"),
    # Cached feature tables (parquet); created by dataset_builder.py
    "train_feats":   _p("data", "processed", "approach04", "train_features.parquet"),
    "val_feats":     _p("data", "processed", "approach04", "val_features.parquet"),
    "test_feats":    _p("data", "processed", "approach04", "test_features.parquet"),
    # Model artifacts
    "model":         _p("models", "approach04", "model.joblib"),
    "feature_names": _p("models", "approach04", "feature_names.json"),
    "metrics":       _p("models", "approach04", "metrics.json"),
    "threshold":     _p("models", "approach04", "threshold.json"),
    # Predictions
    "predictions":   _p("data", "processed", "approach04", "predictions.csv"),
}

# ── Graph ─────────────────────────────────────────────────────────────────────
GRAPH_CONFIG = {
    "directed": True,
    # Dense ID range — max node ID in dataset
    "num_nodes": 4_867_136,   # max_id + 1; verified from data
}

# ── Train / Val split ─────────────────────────────────────────────────────────
SPLIT_CONFIG = {
    "val_ratio":    0.10,   # fraction of positive edges held out for validation
    "test_ratio":   0.00,   # internal test; 0 because Kaggle test.csv is external
}

# ── Negative sampling ─────────────────────────────────────────────────────────
NEG_SAMPLING = {
    # Ratio of negatives to positives (1.0 = balanced)
    "neg_ratio": 1.0,
    # Kaggle test negatives are random (confirmed by experiments: hard_frac>0
    # consistently drops Kaggle AUC from 0.77 to 0.67-0.70). Keep both at 0.0
    # so the model learns a boundary against random negatives — matching test.
    "train_hard_frac": 0.0,
    "val_hard_frac":   0.0,
    "max_train_pos": 1_000_000,
    "max_val_pos":   200_000,
    # Degree cap for hub nodes in hard negative sampling (kept for reference)
    "hub_degree_cap": 2_000,
}

# ── Structural features ────────────────────────────────────────────────────────
FEATURE_CONFIG = {
    # Maximum number of intermediaries used for Adamic-Adar / RA
    # Speeds up extraction for high-degree node pairs
    "max_intermediaries": 500,
    # Degree cap for log-normalisation display (does not affect extraction)
    "log_eps": 1e-6,
}

# ── Classifier ────────────────────────────────────────────────────────────────
# "lgbm"  → LightGBM (CPU-only build; no GPU support in standard pip package)
# "xgb"   → XGBoost  (CUDA GPU, recommended on H100)
# "xgb"   → XGBoost
# "hgb"   → sklearn HistGradientBoostingClassifier (no extra deps)
# "rf"    → sklearn RandomForestClassifier (benchmark)
CLASSIFIER = "xgb"   # XGBoost CUDA (GPU); use --classifier lgbm for CPU LightGBM

LGBM_CONFIG = {
    "n_estimators":    1000,
    "learning_rate":   0.05,
    "num_leaves":      127,
    "max_depth":       -1,
    "min_child_samples": 20,
    "feature_fraction": 0.8,
    "bagging_fraction": 0.8,
    "bagging_freq":    5,
    "reg_alpha":       0.1,
    "reg_lambda":      0.1,
    "objective":       "binary",
    "metric":          "auc",
    "device":          "cpu",       # standard pip lightgbm has no GPU; use xgb for GPU
    "n_jobs":          -1,
    "verbose":         -1,
    "random_state":    RANDOM_SEED,
}

XGB_CONFIG = {
    "n_estimators":         2000,   # high cap; early stopping will find true optimum
    "learning_rate":        0.05,
    "max_depth":            6,
    "min_child_weight":     5,
    "subsample":            0.8,
    "colsample_bytree":     0.8,
    "reg_alpha":            0.1,
    "reg_lambda":           1.0,
    "objective":            "binary:logistic",
    "eval_metric":          "auc",
    "use_label_encoder":    False,
    "tree_method":          "hist",
    "device":               "cuda",      # GPU-accelerated tree building (XGBoost CUDA)
    "early_stopping_rounds": 50,         # stop if val AUC doesn't improve for 50 rounds
    "n_jobs":               -1,
    "random_state":         RANDOM_SEED,
}

HGB_CONFIG = {
    "max_iter":          500,
    "learning_rate":     0.05,
    "max_leaf_nodes":    127,
    "max_depth":         None,
    "min_samples_leaf":  20,
    "l2_regularization": 0.1,
    "random_state":      RANDOM_SEED,
    "early_stopping":    True,
    "n_iter_no_change":  30,
    "validation_fraction": 0.1,
}

RF_CONFIG = {
    "n_estimators":    300,
    "max_depth":       None,
    "min_samples_leaf": 5,
    "n_jobs":          -1,
    "random_state":    RANDOM_SEED,
}
