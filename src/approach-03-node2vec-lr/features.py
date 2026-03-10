"""
Edge feature engineering: convert a pair of node embeddings into a single
feature vector suitable for a binary classifier.

Feature groups
--------------
1. **Embedding operators** — Hadamard, L1, L2, average of node embeddings.
   Multiple operators can be concatenated for richer representation.

2. **Graph-structural features** — classical link-prediction heuristics
   computed from the training graph (common neighbors, Jaccard, Adamic-Adar,
   preferential attachment, degree features).  These capture local topology
   that random-walk embeddings may miss, especially on very large graphs
   where walk coverage is sparse.
"""
import logging
import os
import tempfile
from typing import List, Optional, Tuple

import networkx as nx
import numpy as np
from gensim.models import Word2Vec
from tqdm import tqdm

logger = logging.getLogger(__name__)


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


# ── Scalar embedding similarities (appended as extra columns) ─────────────────

def _embedding_similarities(u_emb: np.ndarray, v_emb: np.ndarray) -> np.ndarray:
    """
    Return a small array of scalar similarity measures between two embeddings.

    Features:
        0  cosine_similarity   dot(u,v) / (||u|| * ||v||)
        1  dot_product         dot(u,v)
        2  l2_distance         ||u - v||
    """
    dot   = np.dot(u_emb, v_emb)
    norm_u = np.linalg.norm(u_emb)
    norm_v = np.linalg.norm(v_emb)
    cos   = dot / (norm_u * norm_v) if (norm_u > 0 and norm_v > 0) else 0.0
    l2    = np.linalg.norm(u_emb - v_emb)
    return np.array([cos, dot, l2], dtype=np.float32)


EMBEDDING_SIM_DIM = 3  # number of scalar similarity features


def edge_features(
    wv,
    edges:    List[Tuple],
    operator: str = "hadamard",
    dim:      int = 64,
    out:      np.ndarray = None,
) -> np.ndarray:
    """
    Build a feature matrix for a list of (src, dst) node pairs.

    Parameters
    ----------
    wv       : gensim KeyedVectors (model.wv)
    edges    : list of (src_node, dst_node) tuples
    operator : one of "hadamard" | "average" | "l1" | "l2"
    dim      : embedding dimension (used for zero-vector fallback)
    out      : optional pre-allocated array of shape (len(edges), dim)

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
    n = len(edges)
    if out is None:
        out = np.empty((n, dim), dtype=np.float32)
    for i, (u, v) in enumerate(edges):
        out[i] = fn(get_embedding(wv, u, dim), get_embedding(wv, v, dim))
    return out


# ── Multi-operator embedding features ─────────────────────────────────────────

def multi_operator_features(
    wv,
    edges:     List[Tuple],
    operators: List[str],
    dim:       int = 64,
    out:       np.ndarray = None,
) -> np.ndarray:
    """
    Concatenate multiple embedding operators into a single feature vector.

    Parameters
    ----------
    operators : e.g. ["hadamard", "l1", "l2"] → output dim = 3 * dim

    Returns
    -------
    X : np.ndarray, shape (len(edges), len(operators) * dim)
    """
    n = len(edges)
    total_dim = len(operators) * dim
    if out is None:
        out = np.empty((n, total_dim), dtype=np.float32)
    for idx, op in enumerate(operators):
        col_start = idx * dim
        col_end   = col_start + dim
        edge_features(wv, edges, op, dim, out=out[:, col_start:col_end])
    return out


# ── Graph-structural features ─────────────────────────────────────────────────

_LOG2 = np.log(2.0)


def graph_structural_features(
    G:     nx.DiGraph,
    edges: List[Tuple],
    out:   np.ndarray = None,
    pagerank: dict = None,
) -> np.ndarray:
    """
    Compute classical link-prediction heuristics for each (u, v) pair.

    Features (20 total):
        0  common_neighbors_out     |N_out(u) ∩ N_out(v)|
        1  common_neighbors_in      |N_in(u)  ∩ N_in(v)|
        2  jaccard_out              common_out / |N_out(u) ∪ N_out(v)|
        3  jaccard_in               common_in  / |N_in(u)  ∪ N_in(v)|
        4  adamic_adar_out          Σ 1/log(|N_out(w)|) for w in common_out
        5  adamic_adar_in           Σ 1/log(|N_in(w)|)  for w in common_in
        6  preferential_attachment  out_deg(u) × in_deg(v)
        7  u_out_degree             out-degree of source
        8  v_in_degree              in-degree of target
        9  u_in_degree              in-degree of source
       10  v_out_degree             out-degree of target
       11  reciprocal               1 if edge v→u exists, else 0
       12  resource_alloc_out       Σ 1/|N_out(w)| for w in common_out
       13  resource_alloc_in        Σ 1/|N_in(w)|  for w in common_in
       14  total_neighbors_out      |N_out(u) ∪ N_out(v)|
       15  total_neighbors_in       |N_in(u)  ∪ N_in(v)|
       16  u_follow_ratio           out_deg(u) / max(in_deg(u), 1)
       17  v_follow_ratio           out_deg(v) / max(in_deg(v), 1)
       18  u_pagerank               PageRank of source node
       19  v_pagerank               PageRank of target node

    All features are computed on the directed training graph.
    """
    n = len(edges)
    n_feats = 20
    if out is None:
        out = np.empty((n, n_feats), dtype=np.float32)

    # Compute PageRank once if not provided
    if pagerank is None:
        logger.info("Computing PageRank (this may take a few minutes) ...")
        pagerank = nx.pagerank(G, alpha=0.85, max_iter=50, tol=1e-4)

    # ── Collect only the nodes that appear in the edge list ────────────────────
    # For a 50K test set this builds sets for ~100K nodes, not all 5M.
    edge_nodes = set()
    for u, v in edges:
        edge_nodes.add(u)
        edge_nodes.add(v)

    logger.info(
        f"Pre-building adjacency sets for {len(edge_nodes):,} edge-list nodes ..."
    )

    _empty = frozenset()
    has_node = G.has_node

    # Eagerly build adjacency sets only for nodes in the edge list
    succ_sets = {}
    pred_sets = {}
    for node in edge_nodes:
        if has_node(node):
            succ_sets[node] = set(G.successors(node))
            pred_sets[node] = set(G.predecessors(node))
        else:
            succ_sets[node] = _empty
            pred_sets[node] = _empty

    # Pre-compute degrees from the sets (avoids repeated len() calls)
    out_deg = {node: len(s) for node, s in succ_sets.items()}
    in_deg  = {node: len(s) for node, s in pred_sets.items()}

    # Pre-compute 1/log(out_deg) and 1/out_deg for Adamic-Adar and Resource
    # Allocation inner loops.  These are keyed on *common-neighbor* nodes
    # which may not be in edge_nodes, so we build lazily into a cache.
    _inv_log_out = {}
    _inv_out     = {}
    _inv_log_in  = {}
    _inv_in      = {}

    def _get_succ(node):
        """Get successor set, building on demand for non-edge-list nodes."""
        if node not in succ_sets:
            if has_node(node):
                succ_sets[node] = set(G.successors(node))
                out_deg[node] = len(succ_sets[node])
            else:
                succ_sets[node] = _empty
                out_deg[node] = 0
        return succ_sets[node]

    def _get_pred(node):
        """Get predecessor set, building on demand for non-edge-list nodes."""
        if node not in pred_sets:
            if has_node(node):
                pred_sets[node] = set(G.predecessors(node))
                in_deg[node] = len(pred_sets[node])
            else:
                pred_sets[node] = _empty
                in_deg[node] = 0
        return pred_sets[node]

    def _aa_out_weight(w):
        if w not in _inv_log_out:
            d = len(_get_succ(w))
            _inv_log_out[w] = (1.0 / np.log(d)) if d > 1 else 0.0
        return _inv_log_out[w]

    def _aa_in_weight(w):
        if w not in _inv_log_in:
            d = len(_get_pred(w))
            _inv_log_in[w] = (1.0 / np.log(d)) if d > 1 else 0.0
        return _inv_log_in[w]

    def _ra_out_weight(w):
        if w not in _inv_out:
            d = len(_get_succ(w))
            _inv_out[w] = (1.0 / d) if d > 0 else 0.0
        return _inv_out[w]

    def _ra_in_weight(w):
        if w not in _inv_in:
            d = len(_get_pred(w))
            _inv_in[w] = (1.0 / d) if d > 0 else 0.0
        return _inv_in[w]

    # ── Vectorise simple features (no set ops needed) ─────────────────────────
    logger.info("Computing vectorisable features (degrees, PageRank, etc.) ...")
    edges_arr_u = np.array([e[0] for e in edges])
    edges_arr_v = np.array([e[1] for e in edges])

    u_out_arr = np.array([out_deg.get(u, 0) for u in edges_arr_u], dtype=np.float32)
    v_in_arr  = np.array([in_deg.get(v, 0)  for v in edges_arr_v], dtype=np.float32)
    u_in_arr  = np.array([in_deg.get(u, 0)  for u in edges_arr_u], dtype=np.float32)
    v_out_arr = np.array([out_deg.get(v, 0) for v in edges_arr_v], dtype=np.float32)

    out[:, 6]  = u_out_arr * v_in_arr                           # pref. attachment
    out[:, 7]  = u_out_arr
    out[:, 8]  = v_in_arr
    out[:, 9]  = u_in_arr
    out[:, 10] = v_out_arr
    out[:, 16] = u_out_arr / np.maximum(u_in_arr, 1.0)          # u follow ratio
    out[:, 17] = v_out_arr / np.maximum(v_in_arr, 1.0)          # v follow ratio
    out[:, 18] = np.array([pagerank.get(u, 0.0) for u in edges_arr_u], dtype=np.float32)
    out[:, 19] = np.array([pagerank.get(v, 0.0) for v in edges_arr_v], dtype=np.float32)

    # Reciprocity — batch check via edge set
    edge_set = set(G.edges())
    out[:, 11] = np.array(
        [1.0 if (v, u) in edge_set else 0.0 for u, v in edges],
        dtype=np.float32,
    )

    # ── Topology features (require set intersections) ─────────────────────────
    logger.info("Computing topology features (common neighbors, AA, RA) ...")
    for i, (u, v) in enumerate(tqdm(edges, desc="Graph features", mininterval=5.0)):
        s_u = succ_sets[u]
        s_v = succ_sets[v]
        p_u = pred_sets[u]
        p_v = pred_sets[v]

        # Common neighbors (out / in)
        cn_out = s_u & s_v
        cn_in  = p_u & p_v
        n_cn_out = len(cn_out)
        n_cn_in  = len(cn_in)

        # Jaccard (union size computed from inclusion-exclusion to avoid building union set)
        len_union_out = len(s_u) + len(s_v) - n_cn_out
        len_union_in  = len(p_u) + len(p_v) - n_cn_in

        # Adamic-Adar & Resource Allocation (share the common-neighbor loop)
        aa_out = 0.0
        ra_out = 0.0
        for w in cn_out:
            aa_out += _aa_out_weight(w)
            ra_out += _ra_out_weight(w)

        aa_in = 0.0
        ra_in = 0.0
        for w in cn_in:
            aa_in += _aa_in_weight(w)
            ra_in += _ra_in_weight(w)

        out[i, 0]  = n_cn_out
        out[i, 1]  = n_cn_in
        out[i, 2]  = n_cn_out / len_union_out if len_union_out else 0.0
        out[i, 3]  = n_cn_in  / len_union_in  if len_union_in  else 0.0
        out[i, 4]  = aa_out
        out[i, 5]  = aa_in
        out[i, 12] = ra_out
        out[i, 13] = ra_in
        out[i, 14] = len_union_out
        out[i, 15] = len_union_in

    return out


GRAPH_FEATURE_DIM = 20   # number of structural features (was 11, now 20)


# ── Memory-safe allocation ─────────────────────────────────────────────────────

_MEMMAP_THRESHOLD_BYTES = 4 * 1024**3  # 4 GB — safe headroom on a 48 GB machine


def _allocate_array(shape, dtype=np.float32):
    """Allocate a numpy array; use a temp-file memmap if size exceeds threshold."""
    nbytes = int(np.prod(shape)) * np.dtype(dtype).itemsize
    if nbytes > _MEMMAP_THRESHOLD_BYTES:
        fd, path = tempfile.mkstemp(suffix=".mmap")
        os.close(fd)
        logger.info(
            f"Allocating {nbytes / 1e9:.1f} GB feature matrix via memmap → '{path}'"
        )
        return np.memmap(path, dtype=dtype, mode="w+", shape=shape)
    return np.empty(shape, dtype=dtype)


def cleanup_memmap(*arrays):
    """Remove backing files of any memmap arrays. No-op for regular arrays."""
    for arr in arrays:
        if isinstance(arr, np.memmap) and getattr(arr, "filename", None):
            try:
                os.unlink(arr.filename)
            except OSError:
                pass


# ── Dataset builders ───────────────────────────────────────────────────────────

def build_dataset(
    wv,
    pos_edges:  List[Tuple],
    neg_edges:  List[Tuple],
    operator:   str  = "hadamard",
    dim:        int  = 64,
    G:          Optional[nx.DiGraph] = None,
    operators:  Optional[List[str]]  = None,
    pagerank:   Optional[dict] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Combine positive and negative edges into a labelled (X, y) dataset.

    If ``operators`` is given (e.g. ["hadamard", "l1", "l2"]), multiple
    embedding operators are concatenated.  Otherwise only ``operator`` is used.

    If ``G`` is provided, graph-structural features are appended.
    Embedding similarity features (cosine, dot, L2 distance) are always appended.

    Returns
    -------
    X : np.ndarray, shape (n_total, feature_dim)
    y : np.ndarray, shape (n_total,)
    """
    n_pos = len(pos_edges)
    n_neg = len(neg_edges)
    all_edges = pos_edges + neg_edges
    n_total = n_pos + n_neg

    # ── Determine total feature dimension ──────────────────────────────────────
    if operators and len(operators) > 1:
        emb_dim = len(operators) * dim
    else:
        emb_dim = dim
    graph_dim = GRAPH_FEATURE_DIM if G is not None else 0
    sim_dim   = EMBEDDING_SIM_DIM
    total_dim = emb_dim + sim_dim + graph_dim

    # ── Allocate unified feature matrix (memmap-backed if large) ───────────────
    X = _allocate_array((n_total, total_dim))

    # ── Embedding features ─────────────────────────────────────────────────────
    if operators and len(operators) > 1:
        multi_operator_features(wv, all_edges, operators, dim, out=X[:, :emb_dim])
    else:
        edge_features(wv, all_edges[:n_pos], operator, dim, out=X[:n_pos, :emb_dim])
        edge_features(wv, all_edges[n_pos:], operator, dim, out=X[n_pos:, :emb_dim])

    # ── Embedding similarity features (cosine, dot, L2 dist) ──────────────────
    logger.info("Computing embedding similarity features ...")
    sim_start = emb_dim
    sim_end   = emb_dim + sim_dim
    for i, (u, v) in enumerate(all_edges):
        u_emb = get_embedding(wv, u, dim)
        v_emb = get_embedding(wv, v, dim)
        X[i, sim_start:sim_end] = _embedding_similarities(u_emb, v_emb)

    # ── Graph-structural features ──────────────────────────────────────────────
    if G is not None:
        logger.info("Computing graph-structural features ...")
        graph_structural_features(G, all_edges, out=X[:, sim_end:], pagerank=pagerank)

    # ── Labels ─────────────────────────────────────────────────────────────────
    y = np.empty(n_total, dtype=np.int32)
    y[:n_pos] = 1
    y[n_pos:] = 0
    return X, y
