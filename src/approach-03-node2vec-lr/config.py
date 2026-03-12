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

# ── Node2Vec via PecanPy ───────────────────────────────────────────────────────
#
#  mode — controls the memory / speed trade-off:
#
#    "SparseOTF"  transition probs on-the-fly from sparse matrix
#                 → lowest RAM (~2–4 GB for 24M edges)   ← default
#    "DenseOTF"   transition probs on-the-fly from dense matrix
#                 → faster walks, needs 32 GB+ RAM
#    "PreComp"    probs precomputed & cached (like the old node2vec library)
#                 → fastest walks, highest RAM — avoid on large graphs
#
NODE2VEC_CONFIG = {
    "mode":        "SparseOTF",  # change to DenseOTF if you have 32 GB+ RAM
    "dimensions":  128,   # 128 is the minimum for 5M-node Twitter graph
    "walk_length": 20,    # steps per random walk
    "num_walks":   10,    # walks per node
    "p":           2.0,   # discourage backtracking — follow chains are directional
    "q":           0.75,  # slight BFS bias — interest clusters dominate
    "workers":     4,     # parallel workers (walk gen + Word2Vec)
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
# MAX_LR_PAIRS: cap training pairs fed to LR (LR is convex — 500K rows gives
# the same accuracy as 19M rows at 1/38th the memory). The 19M-pair OOM killed
# the previous run; this prevents it.
MAX_LR_PAIRS = 500_000

LR_CONFIG = {
    "C":            0.9,
    "max_iter":     300,
    "solver":       "saga",        # mini-batch SGD; memory-efficient for large N
    "class_weight": "balanced",
    "random_state": RANDOM_SEED,
    "n_jobs":       -1,
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
