"""
Configuration for Approach-04b:
Structural Features Phase A (18) + Phase B (8) = 26 features total.

Phase B additions over approach-04:
  - fm_proxy                : friends-measure proxy (avg transitive via sampled N_out(u))
  - same_community          : 1 if u and v share a Leiden community
  - log1p_comm_size_u/v     : log-scaled community sizes
  - avg_trans_nbr_out       : avg transitive(x,v) for x sampled from N_out(u)
  - avg_jac_trans_nbr_out   : avg jaccard_trans(x→v) for x sampled from N_out(u)
  - avg_jac_trans_nbr_in    : avg jaccard_trans(u→y) for y sampled from N_in(v)
  - fm_truncated            : 1 when nbr_list_cap was hit (hub-pair indicator)
"""
import os as _os

# Project root = two levels above this file
_ROOT = _os.path.abspath(_os.path.join(_os.path.dirname(__file__), "..", ".."))

def _p(*parts: str) -> str:
    return _os.path.join(_ROOT, *parts)

# ── Reproducibility ────────────────────────────────────────────────────────────
RANDOM_SEED = 42

# ── File paths ─────────────────────────────────────────────────────────────────
PATHS = {
    "train_graph":   _p("data", "raw", "train.csv"),
    "test_edges":    _p("data", "raw", "test.csv"),
    # Leiden community cache
    "communities":   _p("data", "processed", "approach04b", "communities.npy"),
    # Feature tables
    "train_feats":   _p("data", "processed", "approach04b", "train_features.parquet"),
    "val_feats":     _p("data", "processed", "approach04b", "val_features.parquet"),
    "test_feats":    _p("data", "processed", "approach04b", "test_features.parquet"),
    # Model artifacts
    "model":         _p("models", "approach04b", "model.joblib"),
    "feature_names": _p("models", "approach04b", "feature_names.json"),
    "metrics":       _p("models", "approach04b", "metrics.json"),
    "threshold":     _p("models", "approach04b", "threshold.json"),
    # Predictions
    "predictions":   _p("data", "processed", "approach04b", "predictions.csv"),
}

# ── Graph ─────────────────────────────────────────────────────────────────────
GRAPH_CONFIG = {
    "directed": True,
    "num_nodes": 4_867_136,
}

# ── Train / Val split ─────────────────────────────────────────────────────────
SPLIT_CONFIG = {
    "val_ratio":  0.10,
    "test_ratio": 0.00,
}

# ── Negative sampling ─────────────────────────────────────────────────────────
NEG_SAMPLING = {
    "neg_ratio":       1.0,
    "train_hard_frac": 0.0,
    "val_hard_frac":   0.0,
    "max_train_pos":   1_000_000,
    "max_val_pos":     200_000,
    "hub_degree_cap":  2_000,
}

# ── Structural features (Phase A) ─────────────────────────────────────────────
FEATURE_CONFIG = {
    "max_intermediaries": 500,
}

# ── Phase B specific ────────────────────────────────────────────────────────────
PHASE_B_CONFIG = {
    # Number of nodes to sample from N_out(u) / N_in(v) for neighbor meta-features
    "nbr_sample_k":     10,
    # Cap neighbour list length before sampling to avoid hub blowup
    "nbr_list_cap":     200,
    # Leiden resolution parameter (higher = smaller communities)
    "leiden_resolution": 1.0,
    # Whether to include community features (requires leidenalg + igraph)
    "use_community":    True,
}

# ── Classifier ────────────────────────────────────────────────────────────────
CLASSIFIER = "xgb"   # XGBoost CUDA (GPU); use --classifier lgbm for CPU LightGBM

LGBM_CONFIG = {
    "n_estimators":      1000,
    "learning_rate":     0.05,
    "num_leaves":        127,
    "max_depth":         -1,
    "min_child_samples": 20,
    "feature_fraction":  0.8,
    "bagging_fraction":  0.8,
    "bagging_freq":      5,
    "reg_alpha":         0.1,
    "reg_lambda":        0.1,
    "objective":         "binary",
    "metric":            "auc",
    "device":            "cpu",       # standard pip lightgbm has no GPU; use xgb for GPU
    "n_jobs":            -1,
    "verbose":           -1,
    "random_state":      RANDOM_SEED,
}

XGB_CONFIG = {
    "n_estimators":      1000,
    "learning_rate":     0.05,
    "max_depth":         6,
    "min_child_weight":  5,
    "subsample":         0.8,
    "colsample_bytree":  0.8,
    "reg_alpha":         0.1,
    "reg_lambda":        1.0,
    "objective":         "binary:logistic",
    "eval_metric":       "auc",
    "use_label_encoder": False,
    "tree_method":       "hist",
    "device":            "cuda",
    "n_jobs":            -1,
    "random_state":      RANDOM_SEED,
}

HGB_CONFIG = {
    "max_iter":            500,
    "learning_rate":       0.05,
    "max_leaf_nodes":      127,
    "max_depth":           None,
    "min_samples_leaf":    20,
    "l2_regularization":   0.1,
    "random_state":        RANDOM_SEED,
    "early_stopping":      True,
    "n_iter_no_change":    30,
    "validation_fraction": 0.1,
}

RF_CONFIG = {
    "n_estimators":  300,
    "max_depth":     None,
    "min_samples_leaf": 5,
    "n_jobs":        -1,
    "random_state":  RANDOM_SEED,
}
