"""
Configuration and hyperparameters for the
Node2Vec + Logistic Regression link prediction pipeline.
"""

# ── Reproducibility ────────────────────────────────────────────────────────────
RANDOM_SEED = 42

# ── Graph ─────────────────────────────────────────────────────────────────────
GRAPH_CONFIG = {
    "directed": True,
}

# ── Node2Vec (lightweight config) ─────────────────────────────────────────────
#   Tuned for large graphs (5M nodes / 24M edges) on limited hardware.
#   Increase num_walks / walk_length / epochs for better accuracy if compute allows.
NODE2VEC_CONFIG = {
    "dimensions":  128,   # embedding size — 64 is the sweet spot for LR
    "walk_length": 20,    # steps per random walk
    "num_walks":   10,    # walks per node — keep low for speed
    "p":           1.0,   # return parameter  (1 = neutral BFS/DFS)
    "q":           1.0,   # in-out parameter  (1 = neutral)
    "workers":     4,     # parallel workers for walk generation
}

# ── Word2Vec (gensim) ─────────────────────────────────────────────────────────
W2V_CONFIG = {
    "window":      5,
    "min_count":   1,
    "sg":          1,     # skip-gram (better for rare nodes)
    "epochs":      3,     # lightweight: keep low; raise to 5–10 for accuracy
    "batch_words": 4,
}

# ── Train / Val / Test split ──────────────────────────────────────────────────
SPLIT_CONFIG = {
    "val_ratio":  0.10,   # fraction of positive edges held out for validation
    "test_ratio": 0.10,   # fraction held out for final test
    "neg_ratio":  1.0,    # negatives per positive (1.0 = balanced classes)
}

# ── Edge feature operator ─────────────────────────────────────────────────────
# Options: "hadamard" | "average" | "l1" | "l2"
# Hadamard (element-wise product) gives best empirical performance with LR.
FEATURE_OPERATOR = "hadamard"

# ── Logistic Regression ───────────────────────────────────────────────────────
LR_CONFIG = {
    "C":            1.0,
    "max_iter":     1000,
    "solver":       "lbfgs",
    "class_weight": "balanced",   # handles any residual class imbalance
    "random_state": RANDOM_SEED,
}

# ── File paths ─────────────────────────────────────────────────────────────────
PATHS = {
    "train_graph": "data/raw/train.csv",
    "test_edges":  "data/raw/test.csv",
    "embeddings":  "model/approach-03/node_embeddings.model",
    "classifier":  "model/approach-03/lr_classifier.joblib",
    "predictions": "data/processed/node2vec_lr_predictions_v1.csv",
    "metrics":     "artifacts/metrics.json",
}
