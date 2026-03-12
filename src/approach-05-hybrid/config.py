"""
Configuration for Approach-05: Hybrid = Structural (26) + Node2Vec Hadamard (128) = 154 features.

Depends on:
  - approach-04b parquets (structural features, already built)
  - approach-03 Node2Vec model (Gensim Word2Vec, trained on graph walks)
"""
import os as _os

_ROOT = _os.path.abspath(_os.path.join(_os.path.dirname(__file__), "..", ".."))

def _p(*parts: str) -> str:
    return _os.path.join(_ROOT, *parts)

# ── Reproducibility ────────────────────────────────────────────────────────────
RANDOM_SEED = 42

# ── Source feature tables (from approach-04b) ──────────────────────────────────
SOURCE_PATHS = {
    "train_feats": _p("data", "processed", "approach04b", "train_features.parquet"),
    "val_feats":   _p("data", "processed", "approach04b", "val_features.parquet"),
    "test_feats":  _p("data", "processed", "approach04b", "test_features.parquet"),
}

# ── approach-03 Node2Vec embedding model ───────────────────────────────────────
# Gensim KeyedVectors saved via wv.save(); approach-03 discards the full Word2Vec
# object and keeps only the .wv (KeyedVectors). Two files on disk:
#   node_embeddings.model          (small pickle)
#   node_embeddings.model.vectors.npy  (~2.5 GB float32)
NODE2VEC_MODEL_PATH = _p("model", "approach-03", "node_embeddings.model")

# approach-03 source scripts (used to auto-train if model not found)
APPROACH03_TRAIN_SCRIPT = _p("src", "approach-03-node2vec-lr", "train.py")

# ── Output paths for approach-05 combined parquets ───────────────────────────
PATHS = {
    "train_feats":   _p("data", "processed", "approach05", "train_features.parquet"),
    "val_feats":     _p("data", "processed", "approach05", "val_features.parquet"),
    "test_feats":    _p("data", "processed", "approach05", "test_features.parquet"),
    "train_graph":   _p("data", "raw", "train.csv"),
    "test_edges":    _p("data", "raw", "test.csv"),
    "model":         _p("models", "approach05", "model.joblib"),
    "feature_names": _p("models", "approach05", "feature_names.json"),
    "metrics":       _p("models", "approach05", "metrics.json"),
    "threshold":     _p("models", "approach05", "threshold.json"),
    "predictions":   _p("data", "processed", "approach05", "predictions.csv"),
}

# ── Node2Vec edge feature operator ────────────────────────────────────────────
# "hadamard" | "average" | "l1" | "l2"
EMBEDDING_OPERATOR = "hadamard"
EMBEDDING_DIM = 128     # must match NODE2VEC_CONFIG["dimensions"] in approach-03

# ── Structural feature names (from approach-04b) ─────────────────────────────
# Imported at runtime from structural_features.py to stay in sync
# NUM_STRUCTURAL = 26, NUM_EMBEDDING = 128, TOTAL = 154

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
    "n_estimators":     300,
    "max_depth":        None,
    "min_samples_leaf": 5,
    "n_jobs":           -1,
    "random_state":     RANDOM_SEED,
}
