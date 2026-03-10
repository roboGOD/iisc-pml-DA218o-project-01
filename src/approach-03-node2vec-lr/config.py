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
    "dimensions":  128,   # doubled from 128 → richer representations for 5M nodes
    "walk_length": 40,    # doubled from 20 → deeper structural capture
    "num_walks":   20,    # doubled from 10 → better coverage of sparse regions
    "p":           1.0,   # unbiased return (was 2.0 — too aggressive backtrack penalty)
    "q":           0.5,   # stronger BFS bias → capture local community structure
    "workers":     4,     # parallel workers (walk gen + Word2Vec)
}

# ── Word2Vec (gensim) ─────────────────────────────────────────────────────────
W2V_CONFIG = {
    "window":      10,    # larger window = more context per walk step
    "min_count":   1,
    "sg":          1,     # skip-gram (better for rare nodes)
    "epochs":      10,    # doubled from 5 → better embedding convergence
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

# When FEATURE_OPERATORS is a list with >1 entry, all operators are concatenated.
# For a single operator, leave this as None and FEATURE_OPERATOR is used instead.
# Combining operators captures complementary signal: Hadamard for multiplicative
# interaction, L1 for asymmetry, L2 for discrepancy magnitude.
FEATURE_OPERATORS = ["hadamard", "l1", "l2", "average"]

# Whether to compute and append graph-structural features (common neighbors,
# Jaccard, Adamic-Adar, preferential attachment, degree features).
# Adds 11 features.  Requires passing G_train into fit_classifier.
USE_GRAPH_FEATURES = True

# ── Classifier selection ───────────────────────────────────────────────────────
# Options: "sgd" | "xgboost"
#   "sgd"     — SGDClassifier(loss="log_loss"), trained via mini-batch partial_fit.
#               Memory-safe on arbitrarily large datasets.
#   "xgboost" — XGBClassifier with histogram tree method.  Better accuracy but
#               needs more RAM than SGD (~2× the feature matrix).
CLASSIFIER_TYPE = "xgboost"

# ── SGD Logistic Regression (memory-efficient) ────────────────────────────────
SGD_CONFIG = {
    "alpha":        1e-4,           # L2 regularisation strength
    "class_weight": "balanced",     # handles any residual class imbalance
    "random_state": RANDOM_SEED,
    "n_epochs":     10,             # passes over the training data
    "chunk_size":   500_000,        # rows per mini-batch
}

# ── XGBoost ───────────────────────────────────────────────────────────────────
XGB_CONFIG = {
    "n_estimators":      2000,       # more trees with lower LR (early stopping guards)
    "max_depth":         6,          # reduced from 8 → less overfitting
    "learning_rate":     0.03,       # reduced from 0.05 → gentler boosting
    "subsample":         0.7,        # reduced from 0.8 → more regularization
    "colsample_bytree":  0.6,        # reduced from 0.8 → decorrelate trees
    "min_child_weight":  10,         # increased from 5 → fewer spurious splits
    "gamma":             0.3,        # increased from 0.1 → prune more aggressively
    "reg_alpha":         0.3,        # increased L1 → sparser trees
    "reg_lambda":        3.0,        # increased L2 → smoother predictions
    "tree_method":       "hist",     # histogram-based — fast & memory-efficient
    "scale_pos_weight":  1.0,        # adjust if classes are imbalanced
    "random_state":      RANDOM_SEED,
    "n_jobs":            4,
    "verbosity":         1,
    "early_stopping_rounds": 50,     # more patience with lower LR
}

# Backward compat: LR_CONFIG points to whichever classifier is active
LR_CONFIG = SGD_CONFIG if CLASSIFIER_TYPE == "sgd" else XGB_CONFIG

# ── File paths ─────────────────────────────────────────────────────────────────
PATHS = {
    "train_graph": "data/raw/train.csv",
    "test_edges":  "data/raw/test.csv",
    "embeddings":  "model/approach-03-node2vec/node_embeddings.model",
    "classifier":  "model/approach-03-node2vec/"+CLASSIFIER_TYPE+"_classifier.joblib",
    "predictions": "data/processed/node2vec_"+CLASSIFIER_TYPE+"_predictions_v1.csv",
    "metrics":     "artifacts/node2vec/metrics.json",
}
