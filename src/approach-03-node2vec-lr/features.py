"""
Edge feature engineering: convert a pair of node embeddings into a single
feature vector suitable for a binary classifier.
"""
from typing import List, Tuple

import numpy as np
from gensim.models import Word2Vec


# ── Embedding lookup ───────────────────────────────────────────────────────────

def get_embedding(wv, node, dim: int) -> np.ndarray:
    """
    Return the embedding vector for *node*.
    Falls back to a zero vector for nodes not seen during training
    (e.g. isolated nodes that were never part of a walk).
    """
    key = str(node)
    if key in wv:
        return wv[key].astype(np.float32)
    return np.zeros(dim, dtype=np.float32)


# ── Feature operators ──────────────────────────────────────────────────────────

_OPERATORS = {
    # Hadamard: captures multiplicative interaction; best for LR on most benchmarks
    "hadamard": lambda u, v: u * v,
    # Average: simple centroid of the two embeddings
    "average":  lambda u, v: (u + v) * 0.5,
    # L1: captures absolute difference; sensitive to asymmetric relationships
    "l1":       lambda u, v: np.abs(u - v),
    # L2: squared difference; penalises large discrepancies more
    "l2":       lambda u, v: (u - v) ** 2,
}


def edge_features(
    wv,
    edges:    List[Tuple],
    operator: str = "hadamard",
    dim:      int = 64,
) -> np.ndarray:
    """
    Build a feature matrix for a list of (src, dst) node pairs.

    Parameters
    ----------
    wv       : gensim KeyedVectors (model.wv)
    edges    : list of (src_node, dst_node) tuples
    operator : one of "hadamard" | "average" | "l1" | "l2"
    dim      : embedding dimension (used for zero-vector fallback)

    Returns
    -------
    X : np.ndarray of shape (len(edges), dim)
    """
    if operator not in _OPERATORS:
        raise ValueError(
            f"Unknown operator '{operator}'. "
            f"Valid options: {list(_OPERATORS.keys())}"
        )
    fn = _OPERATORS[operator]
    return np.array(
        [fn(get_embedding(wv, u, dim), get_embedding(wv, v, dim)) for u, v in edges],
        dtype=np.float32,
    )


def build_dataset(
    wv,
    pos_edges: List[Tuple],
    neg_edges: List[Tuple],
    operator:  str = "hadamard",
    dim:       int = 64,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Combine positive and negative edges into a labelled (X, y) dataset.

    Returns
    -------
    X : np.ndarray, shape (n_pos + n_neg, dim)
    y : np.ndarray, shape (n_pos + n_neg,)  — 1 = edge, 0 = no edge
    """
    X_pos = edge_features(wv, pos_edges, operator, dim)
    X_neg = edge_features(wv, neg_edges, operator, dim)
    X = np.vstack([X_pos, X_neg])
    y = np.array([1] * len(pos_edges) + [0] * len(neg_edges), dtype=np.int32)
    return X, y
