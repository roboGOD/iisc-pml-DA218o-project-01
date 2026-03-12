"""
Directed structural feature extractor — Phase A (18) + Phase B (8) = 26 features.

Phase A (same as approach-04)
------------------------------
  out_deg_u, in_deg_u, out_deg_v, in_deg_v
  log1p_* versions
  reciprocal
  common_out, common_in, transitive
  jaccard_out, jaccard_in, jaccard_trans
  pref_attach
  adamic_adar_trans, resource_alloc_trans

Phase B additions
-----------------
  fm_proxy            — avg |N_out(x) ∩ N_in(v)| for x sampled from N_out(u)
                        (friends-measure proxy: expected transitive via u's followees)
  avg_trans_nbr_in    — avg |N_out(u) ∩ N_in(y)| for y sampled from N_in(v)
                        (dual direction: expected transitive via v's followers)
  avg_jac_trans_nbr_out — avg jaccard_trans(x→v) for x sampled from N_out(u)
  avg_jac_trans_nbr_in  — avg jaccard_trans(u→y) for y sampled from N_in(v)
  same_community      — 1 if u and v share a Leiden community, else 0
  log1p_comm_size_u   — log1p(size of u's community)
  log1p_comm_size_v   — log1p(size of v's community)
  fm_truncated        — 1 when nbr_list_cap was hit for N_out(u) or N_in(v);
                        allows the model to learn a separate rule for hub pairs
                        whose neighbourhood aggregates are estimates rather than exact.

Community features gracefully fall back to 0 when CommunityStore is None.
"""
from __future__ import annotations

import logging
from typing import List, Optional
import time

import numpy as np
import pandas as pd

from graph_store import GraphStore
from community_store import CommunityStore

logger = logging.getLogger(__name__)

# ── Phase A feature names ─────────────────────────────────────────────────────
FEATURE_NAMES_A: List[str] = [
    "out_deg_u",
    "in_deg_u",
    "out_deg_v",
    "in_deg_v",
    "log1p_out_deg_u",
    "log1p_in_deg_u",
    "log1p_out_deg_v",
    "log1p_in_deg_v",
    "reciprocal",
    "common_out",
    "common_in",
    "transitive",
    "jaccard_out",
    "jaccard_in",
    "jaccard_trans",
    "pref_attach",
    "adamic_adar_trans",
    "resource_alloc_trans",
]

# ── Phase B feature names ─────────────────────────────────────────────────────
FEATURE_NAMES_B: List[str] = [
    "fm_proxy",
    "avg_trans_nbr_in",
    "avg_jac_trans_nbr_out",
    "avg_jac_trans_nbr_in",
    "same_community",
    "log1p_comm_size_u",
    "log1p_comm_size_v",
    "fm_truncated",   # 1 if nbr_list_cap was hit for N_out(u) or N_in(v)
]

FEATURE_NAMES: List[str] = FEATURE_NAMES_A + FEATURE_NAMES_B
NUM_FEATURES = len(FEATURE_NAMES)      # 26
NUM_FEATURES_A = len(FEATURE_NAMES_A)  # 18
NUM_FEATURES_B = len(FEATURE_NAMES_B)  # 8


# ─────────────────────────────────────────────────────────────────────────────
# Low-level helpers (Phase A)
# ─────────────────────────────────────────────────────────────────────────────

_EMPTY = np.array([], dtype=np.int32)


def _intersect_sorted(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    if len(a) == 0 or len(b) == 0:
        return _EMPTY
    return np.intersect1d(a, b, assume_unique=True)


def _jaccard(inter_len: int, a_len: int, b_len: int) -> float:
    union = a_len + b_len - inter_len
    return inter_len / union if union > 0 else 0.0


def _aa_ra(
    inter: np.ndarray,
    out_deg: np.ndarray,
    in_deg: np.ndarray,
    max_inter: int,
) -> tuple[float, float]:
    if len(inter) == 0:
        return 0.0, 0.0
    if len(inter) > max_inter:
        inter = inter[:max_inter]
    total_deg = out_deg[inter].astype(np.float64) + in_deg[inter].astype(np.float64) + 2.0
    aa = float(np.sum(1.0 / np.log(total_deg)))
    ra = float(np.sum(1.0 / (total_deg - 1.0)))
    return aa, ra


# ─────────────────────────────────────────────────────────────────────────────
# Phase B helpers
# ─────────────────────────────────────────────────────────────────────────────

def _sample_neighbors(
    neighbors: np.ndarray,
    k: int,
    cap: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """
    Sample up to k node IDs from *neighbors*, pre-capped at *cap*.
    Returns an empty array if neighbors is empty.
    """
    if len(neighbors) == 0:
        return _EMPTY
    nbrs = neighbors if len(neighbors) <= cap else neighbors[:cap]
    if len(nbrs) <= k:
        return nbrs
    idx = rng.choice(len(nbrs), k, replace=False)
    return nbrs[idx]


def _transitive_count(
    n_out_x: np.ndarray, n_in_v: np.ndarray
) -> int:
    """Count |N_out(x) ∩ N_in(v)|."""
    if len(n_out_x) == 0 or len(n_in_v) == 0:
        return 0
    return len(np.intersect1d(n_out_x, n_in_v, assume_unique=True))


def _jac_trans(
    n_out_x: np.ndarray, n_in_y: np.ndarray, od_x: int, id_y: int
) -> float:
    """Jaccard_trans(x→y) = |N_out(x) ∩ N_in(y)| / (od_x + id_y - |inter|)."""
    c = _transitive_count(n_out_x, n_in_y)
    return _jaccard(c, od_x, id_y)


def _compute_phase_b(
    gs: GraphStore,
    u: int,
    v: int,
    n_out_u: np.ndarray,
    n_in_v: np.ndarray,
    od_u: int,
    id_v: int,
    cs: Optional[CommunityStore],
    nbr_sample_k: int,
    nbr_list_cap: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """
    Compute the 7 Phase B features for edge (u→v).

    Parameters
    ----------
    n_out_u, n_in_v : already-fetched sorted neighbour arrays
    od_u, id_v      : degree scalars (already computed in Phase A)
    cs              : CommunityStore (or None to zero-fill community features)
    rng             : seeded Generator for neighbour sampling
    """
    # ── fm_proxy & avg_jac_trans_nbr_out ────────────────────────────
    # Sampled nodes x from N_out(u); compute transitive(x→v) for each
    sampled_out = _sample_neighbors(n_out_u, nbr_sample_k, nbr_list_cap, rng)

    fm_sum = 0.0
    jac_sum_out = 0.0
    for x in sampled_out:
        xi = int(x)
        n_out_x = gs.out_neighbors(xi)
        t_xv = _transitive_count(n_out_x, n_in_v)
        fm_sum += t_xv
        jac_sum_out += _jaccard(t_xv, int(gs.out_deg[xi]), id_v)

    k_out = max(len(sampled_out), 1)
    fm_proxy = fm_sum / k_out
    avg_jac_trans_nbr_out = jac_sum_out / k_out

    # ── avg_trans_nbr_in & avg_jac_trans_nbr_in ─────────────────────
    # Sampled nodes y from N_in(v); compute transitive(u→y) for each
    n_in_u = gs.in_neighbors(u)   # needed for transitive(u→y) = |N_out(u) ∩ N_in(y)|
    # Note: N_out(u) is n_out_u; N_in(y) fetched per y
    sampled_in = _sample_neighbors(n_in_v, nbr_sample_k, nbr_list_cap, rng)

    trans_sum_in = 0.0
    jac_sum_in = 0.0
    for y in sampled_in:
        yi = int(y)
        n_in_y = gs.in_neighbors(yi)
        t_uy = _transitive_count(n_out_u, n_in_y)
        trans_sum_in += t_uy
        jac_sum_in += _jaccard(t_uy, od_u, int(gs.in_deg[yi]))

    k_in = max(len(sampled_in), 1)
    avg_trans_nbr_in = trans_sum_in / k_in
    avg_jac_trans_nbr_in = jac_sum_in / k_in

    # ── Community features ───────────────────────────────────────────
    if cs is not None:
        same_comm = float(cs.same_community(u, v))
        log_sz_u  = float(np.log1p(cs.get_size(u)))
        log_sz_v  = float(np.log1p(cs.get_size(v)))
    else:
        same_comm = 0.0
        log_sz_u  = 0.0
        log_sz_v  = 0.0

    return np.array([
        fm_proxy,
        avg_trans_nbr_in,
        avg_jac_trans_nbr_out,
        avg_jac_trans_nbr_in,
        same_comm,
        log_sz_u,
        log_sz_v,
    ], dtype=np.float32)


# ─────────────────────────────────────────────────────────────────────────────
# Per-pair feature extraction (Phase A + B)
# ─────────────────────────────────────────────────────────────────────────────

def extract_features_pair(
    gs: GraphStore,
    u: int,
    v: int,
    max_intermediaries: int = 500,
    cs: Optional[CommunityStore] = None,
    nbr_sample_k: int = 10,
    nbr_list_cap: int = 200,
    rng: Optional[np.random.Generator] = None,
) -> np.ndarray:
    """
    Extract all 25 Phase A + B structural features for edge (u→v).

    Unseen nodes (id >= gs.num_nodes) return a zero vector.
    Community features are zeroed when cs=None.
    """
    if rng is None:
        rng = np.random.default_rng(hash((u, v)) & 0xFFFFFFFF)

    n = gs.num_nodes
    if u < 0 or u >= n or v < 0 or v >= n:
        return np.zeros(NUM_FEATURES, dtype=np.float32)

    # ── Phase A ───────────────────────────────────────────────────────
    od_u = int(gs.out_deg[u])
    id_u = int(gs.in_deg[u])
    od_v = int(gs.out_deg[v])
    id_v = int(gs.in_deg[v])

    recip = float(gs.has_edge(v, u))

    n_out_u = gs.out_neighbors(u)
    n_out_v = gs.out_neighbors(v)
    n_in_u  = gs.in_neighbors(u)
    n_in_v  = gs.in_neighbors(v)

    common_out_arr = _intersect_sorted(n_out_u, n_out_v)
    common_in_arr  = _intersect_sorted(n_in_u,  n_in_v)
    trans_arr      = _intersect_sorted(n_out_u, n_in_v)

    c_out  = len(common_out_arr)
    c_in   = len(common_in_arr)
    c_tran = len(trans_arr)

    jac_out   = _jaccard(c_out,  od_u, od_v)
    jac_in    = _jaccard(c_in,   id_u, id_v)
    jac_trans = _jaccard(c_tran, od_u, id_v)
    pa = float(od_u) * float(id_v)

    aa_trans, ra_trans = _aa_ra(trans_arr, gs.out_deg, gs.in_deg, max_intermediaries)

    feats_a = np.array([
        od_u, id_u, od_v, id_v,
        np.log1p(od_u), np.log1p(id_u), np.log1p(od_v), np.log1p(id_v),
        recip,
        c_out, c_in, c_tran,
        jac_out, jac_in, jac_trans,
        pa,
        aa_trans, ra_trans,
    ], dtype=np.float32)

    # ── Phase B ───────────────────────────────────────────────────────
    # fm_truncated: 1 when the cap fires, so the model can learn a separate
    # rule for hub pairs whose neighbourhood aggregates are estimates only.
    fm_truncated = np.array([
        1.0 if (len(n_out_u) > nbr_list_cap or len(n_in_v) > nbr_list_cap) else 0.0
    ], dtype=np.float32)

    feats_b = _compute_phase_b(
        gs, u, v,
        n_out_u=n_out_u, n_in_v=n_in_v,
        od_u=od_u, id_v=id_v,
        cs=cs,
        nbr_sample_k=nbr_sample_k,
        nbr_list_cap=nbr_list_cap,
        rng=rng,
    )

    return np.concatenate([feats_a, feats_b, fm_truncated])


# ─────────────────────────────────────────────────────────────────────────────
# Batch extraction
# ─────────────────────────────────────────────────────────────────────────────

def extract_features_batch(
    gs: GraphStore,
    pairs: np.ndarray,
    max_intermediaries: int = 500,
    cs: Optional[CommunityStore] = None,
    nbr_sample_k: int = 10,
    nbr_list_cap: int = 200,
    seed: int = 42,
) -> np.ndarray:
    """
    Extract features for every (u, v) pair in *pairs* (shape N×2).

    Returns np.ndarray of shape (N, NUM_FEATURES) float32.
    """
    rng = np.random.default_rng(seed)
    N = len(pairs)
    out = np.empty((N, NUM_FEATURES), dtype=np.float32)
    for i in range(N):
        out[i] = extract_features_pair(
            gs,
            int(pairs[i, 0]), int(pairs[i, 1]),
            max_intermediaries=max_intermediaries,
            cs=cs,
            nbr_sample_k=nbr_sample_k,
            nbr_list_cap=nbr_list_cap,
            rng=rng,
        )
    return out


def build_dataframe(
    gs: GraphStore,
    pairs: np.ndarray,
    labels: np.ndarray,
    max_intermediaries: int = 500,
    cs: Optional[CommunityStore] = None,
    nbr_sample_k: int = 10,
    nbr_list_cap: int = 200,
    batch_size: int = 50_000,
    seed: int = 42,
) -> pd.DataFrame:
    """
    Build a feature DataFrame for supervised training / evaluation.

    Parameters
    ----------
    pairs  : (N, 2) int32 — (u, v) pairs
    labels : (N,) int8 — 1 for positive, 0 for negative
    cs     : CommunityStore (None → community features zeroed out)
    """
    N = len(pairs)
    n_batches = (N + batch_size - 1) // batch_size
    phase = "A+B" if cs is not None else "A-only (no communities yet)"
    logger.info(
        "Extracting %s features for %d pairs (%d batches) …",
        phase, N, n_batches,
    )

    all_feats: list[np.ndarray] = []
    t_start = time.time()

    for bi, start in enumerate(range(0, N, batch_size), 1):
        end = min(start + batch_size, N)
        t_batch = time.time()
        chunk = extract_features_batch(
            gs,
            pairs[start:end],
            max_intermediaries=max_intermediaries,
            cs=cs,
            nbr_sample_k=nbr_sample_k,
            nbr_list_cap=nbr_list_cap,
            seed=seed + bi,
        )
        all_feats.append(chunk)
        elapsed = time.time() - t_start
        batch_t = time.time() - t_batch
        pct = 100.0 * end / N
        eta = (elapsed / bi) * (n_batches - bi) if bi < n_batches else 0.0
        logger.info(
            "  [%d/%d] pairs %d–%d (%.1f%%) | batch=%.1fs | elapsed=%.1fs | ETA=%.1fs",
            bi, n_batches, start + 1, end, pct, batch_t, elapsed, eta,
        )

    logger.info("Feature extraction complete in %.1fs.", time.time() - t_start)

    X = np.concatenate(all_feats, axis=0)
    df = pd.DataFrame(X, columns=FEATURE_NAMES)
    df["label"] = labels.astype(np.int8)
    df["u"] = pairs[:, 0]
    df["v"] = pairs[:, 1]
    return df
