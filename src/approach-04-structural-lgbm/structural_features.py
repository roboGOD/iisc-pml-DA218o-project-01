"""
Directed structural feature extractor.

Phase A features (18 total)
---------------------------
Node-level:
  out_deg_u, in_deg_u, out_deg_v, in_deg_v
  log1p_out_deg_u, log1p_in_deg_u, log1p_out_deg_v, log1p_in_deg_v

Edge-level:
  reciprocal        — 1 if v→u exists in the training graph

Neighbourhood overlap (all counts use SORTED np.intersect1d):
  common_out        — |N_out(u) ∩ N_out(v)|   shared followees
  common_in         — |N_in(u)  ∩ N_in(v)|    shared followers
  transitive        — |N_out(u) ∩ N_in(v)|    "friends-of-friends" in directed sense
                      = nodes that u follows AND that follow v back

Normalised overlaps:
  jaccard_out       — common_out / |N_out(u) ∪ N_out(v)|
  jaccard_in        — common_in  / |N_in(u)  ∪ N_in(v)|
  jaccard_trans     — transitive / (out_deg_u + in_deg_v - transitive)

Global attachment:
  pref_attach       — out_deg_u * in_deg_v   (unnormalised PA for directed graphs)

Weighted overlap over transitive intermediaries w ∈ N_out(u) ∩ N_in(v):
  adamic_adar_trans — Σ_w  1 / log(out_deg_w + in_deg_w + 2)
  resource_alloc    — Σ_w  1 / (out_deg_w + in_deg_w + 1)

Why these features work
-----------------------
* out_deg/in_deg capture the power-law popularity bias.
* transitive is the single strongest local signal for DIRECTED graphs:
  if u follows w AND w is followed by v, then u→v is a natural recommendation.
* Adamic-Adar down-weights hub intermediaries (they are less informative).
* Resource Allocation is even more aggressive down-weighting of hubs.
* reciprocal exploits the Twitter convention: mutual-follow pairs are very
  likely to be real connections.
* log1p transforms tame the long-tailed distribution for tree learners.
"""
from __future__ import annotations

import logging
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from graph_store import GraphStore

logger = logging.getLogger(__name__)

# ── Feature names (canonical order) ──────────────────────────────────────────
FEATURE_NAMES: List[str] = [
    # degrees
    "out_deg_u",
    "in_deg_u",
    "out_deg_v",
    "in_deg_v",
    "log1p_out_deg_u",
    "log1p_in_deg_u",
    "log1p_out_deg_v",
    "log1p_in_deg_v",
    # edge flag
    "reciprocal",
    # overlap counts
    "common_out",
    "common_in",
    "transitive",
    # Jaccard variants
    "jaccard_out",
    "jaccard_in",
    "jaccard_trans",
    # attachment
    "pref_attach",
    # weighted overlap
    "adamic_adar_trans",
    "resource_alloc_trans",
]

NUM_FEATURES = len(FEATURE_NAMES)


# ─────────────────────────────────────────────────────────────────────────────
# Internal low-level helpers
# ─────────────────────────────────────────────────────────────────────────────

def _intersect_sorted(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """
    Return the sorted intersection of two sorted int arrays.
    O((|a| + |b|) log(|a| + |b|)) via numpy merge path.
    Assumes both arrays are already sorted (GraphStore guarantees this).
    """
    if len(a) == 0 or len(b) == 0:
        return _EMPTY
    return np.intersect1d(a, b, assume_unique=True)


_EMPTY = np.array([], dtype=np.int32)


def _union_size(a_len: int, b_len: int, inter_len: int) -> int:
    return a_len + b_len - inter_len


def _jaccard(inter_len: int, a_len: int, b_len: int) -> float:
    union = _union_size(a_len, b_len, inter_len)
    return inter_len / union if union > 0 else 0.0


def _aa_ra(
    inter: np.ndarray,
    out_deg: np.ndarray,
    in_deg: np.ndarray,
    max_inter: int,
) -> tuple[float, float]:
    """
    Compute Adamic-Adar and Resource Allocation over a set of intermediaries.
    Both sums are over nodes w in `inter`.

    Returns (adamic_adar, resource_allocation).
    """
    if len(inter) == 0:
        return 0.0, 0.0

    # Cap to avoid O(degree²) blow-up on hub nodes
    if len(inter) > max_inter:
        inter = inter[:max_inter]

    total_deg = out_deg[inter].astype(np.float64) + in_deg[inter].astype(np.float64) + 2.0
    log_deg = np.log(total_deg)

    aa = float(np.sum(1.0 / log_deg))
    ra = float(np.sum(1.0 / (total_deg - 1.0)))   # out+in+1 in denominator
    return aa, ra


# ─────────────────────────────────────────────────────────────────────────────
# Per-pair feature extraction
# ─────────────────────────────────────────────────────────────────────────────

def extract_features_pair(
    gs: GraphStore,
    u: int,
    v: int,
    max_intermediaries: int = 500,
) -> np.ndarray:
    """
    Extract all Phase A structural features for a single edge (u→v).

    Unseen nodes (id >= gs.num_nodes) are treated as isolated nodes
    (all neighbourhood features = 0).

    Returns
    -------
    np.ndarray shape (NUM_FEATURES,) float32
    """
    n = gs.num_nodes

    # Safety guard for unseen nodes
    if u < 0 or u >= n or v < 0 or v >= n:
        return np.zeros(NUM_FEATURES, dtype=np.float32)

    # ── Degrees ───────────────────────────────────────────────────────
    od_u = int(gs.out_deg[u])
    id_u = int(gs.in_deg[u])
    od_v = int(gs.out_deg[v])
    id_v = int(gs.in_deg[v])

    # ── Reciprocal ────────────────────────────────────────────────────
    recip = float(gs.has_edge(v, u))

    # ── Neighbourhood sets ────────────────────────────────────────────
    n_out_u = gs.out_neighbors(u)   # sorted int32
    n_out_v = gs.out_neighbors(v)
    n_in_u  = gs.in_neighbors(u)
    n_in_v  = gs.in_neighbors(v)

    # ── Overlap counts ────────────────────────────────────────────────
    common_out_arr = _intersect_sorted(n_out_u, n_out_v)
    common_in_arr  = _intersect_sorted(n_in_u,  n_in_v)
    trans_arr      = _intersect_sorted(n_out_u, n_in_v)   # key directed feature

    c_out  = len(common_out_arr)
    c_in   = len(common_in_arr)
    c_tran = len(trans_arr)

    # ── Jaccard ───────────────────────────────────────────────────────
    jac_out   = _jaccard(c_out,  od_u, od_v)
    jac_in    = _jaccard(c_in,   id_u, id_v)
    # Jaccard-transitive: inter = transitive, sets are N_out(u) and N_in(v)
    jac_trans = _jaccard(c_tran, od_u, id_v)

    # ── Preferential Attachment ───────────────────────────────────────
    pa = float(od_u) * float(id_v)

    # ── Adamic-Adar & Resource Allocation (transitive intermediaries) ─
    aa_trans, ra_trans = _aa_ra(
        trans_arr, gs.out_deg, gs.in_deg, max_intermediaries
    )

    return np.array([
        od_u, id_u, od_v, id_v,
        np.log1p(od_u), np.log1p(id_u), np.log1p(od_v), np.log1p(id_v),
        recip,
        c_out, c_in, c_tran,
        jac_out, jac_in, jac_trans,
        pa,
        aa_trans, ra_trans,
    ], dtype=np.float32)


# ─────────────────────────────────────────────────────────────────────────────
# Batch extraction
# ─────────────────────────────────────────────────────────────────────────────

def extract_features_batch(
    gs: GraphStore,
    pairs: np.ndarray,
    max_intermediaries: int = 500,
    log_every: int = 50_000,
) -> np.ndarray:
    """
    Extract features for a batch of (u, v) pairs.

    Parameters
    ----------
    pairs : np.ndarray shape (N, 2) int32/int64
    max_intermediaries : cap to avoid hub blow-up
    log_every : log progress every this many pairs

    Returns
    -------
    np.ndarray shape (N, NUM_FEATURES) float32
    """
    N = len(pairs)
    out = np.zeros((N, NUM_FEATURES), dtype=np.float32)

    for i in range(N):
        u, v = int(pairs[i, 0]), int(pairs[i, 1])
        out[i] = extract_features_pair(gs, u, v, max_intermediaries)
        if log_every > 0 and (i + 1) % log_every == 0:
            logger.info("  features: %d / %d pairs done …", i + 1, N)

    return out


def build_dataframe(
    gs: GraphStore,
    pairs: np.ndarray,
    labels: np.ndarray,
    max_intermediaries: int = 500,
    batch_size: int = 100_000,
    log_every: int = 50_000,
) -> pd.DataFrame:
    """
    Build a feature DataFrame for supervised training / evaluation.

    Parameters
    ----------
    gs : GraphStore (training graph for feature extraction)
    pairs : (N, 2) int32 — source and target nodes
    labels : (N,) int — 1 for positive edges, 0 for negatives
    batch_size : process this many pairs at once to bound peak memory
    log_every : passed through to extract_features_batch

    Returns
    -------
    pd.DataFrame with columns FEATURE_NAMES + ['label', 'u', 'v']
    """
    N = len(pairs)
    all_feats: list[np.ndarray] = []

    for start in range(0, N, batch_size):
        end = min(start + batch_size, N)
        logger.info(
            "Feature extraction: pairs %d–%d / %d …", start + 1, end, N
        )
        chunk = extract_features_batch(
            gs, pairs[start:end], max_intermediaries, log_every
        )
        all_feats.append(chunk)

    X = np.concatenate(all_feats, axis=0)
    df = pd.DataFrame(X, columns=FEATURE_NAMES)
    df["label"] = labels.astype(np.int8)
    df["u"] = pairs[:, 0]
    df["v"] = pairs[:, 1]
    return df
