"""
Negative edge sampler for directed link-prediction.

Strategy
--------
Easy  — random (u, v) pairs drawn uniformly.  Since the graph is sparse
        (24M / 4.87M² ≈ 0.001 %), almost no random pair is a real edge, so
        we skip the edge-existence check for easy negatives, saving time.

Hard  — 2-hop negatives u → w → v where u → v does NOT exist.
        These are realistic negatives that force the model to learn
        second-order structure rather than just memorising degree effects.

Mixed — blends easy and hard according to configurable fractions.

Guarantees
----------
* No self-loops
* No true positive edges (positives excluded via has_edge_batch)
* No duplicates within a returned batch
* Reproducible via numpy Generator seeds
"""
from __future__ import annotations

import logging
from typing import Tuple

import numpy as np

from graph_store import GraphStore

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# Internal helpers
# ─────────────────────────────────────────────────────────────────────────────

def _deduplicate(
    pairs: np.ndarray,
) -> np.ndarray:
    """Remove duplicate (u, v) pairs from an (N, 2) int array."""
    codes = pairs[:, 0].astype(np.int64) * 10_000_000 + pairs[:, 1].astype(np.int64)
    _, unique_idx = np.unique(codes, return_index=True)
    return pairs[unique_idx]


def _remove_self_loops(pairs: np.ndarray) -> np.ndarray:
    mask = pairs[:, 0] != pairs[:, 1]
    return pairs[mask]


def _exclude_true_edges(
    pairs: np.ndarray, gs: GraphStore
) -> np.ndarray:
    """Remove any pair that is an actual edge in gs."""
    is_edge = gs.has_edge_batch(pairs[:, 0], pairs[:, 1])
    return pairs[~is_edge]


# ─────────────────────────────────────────────────────────────────────────────
# Easy negatives
# ─────────────────────────────────────────────────────────────────────────────

def sample_easy_negatives(
    gs: GraphStore,
    n: int,
    seed: int = 0,
    exclude_pairs: np.ndarray | None = None,
) -> np.ndarray:
    """
    Sample n easy (random) negative edges.

    Parameters
    ----------
    gs : GraphStore
    n  : number of negatives to return
    seed : RNG seed for reproducibility
    exclude_pairs : optional (M, 2) array of pairs to exclude
                   (e.g. the positive training pairs)

    Returns
    -------
    np.ndarray shape (n, 2) int32  — unique negative pairs
    """
    rng = np.random.default_rng(seed)
    num_nodes = gs.num_nodes

    collected: list[np.ndarray] = []
    collected_n = 0
    overshoot = 1.05   # slight oversample then trim

    max_attempts = 20
    for attempt in range(max_attempts):
        need = int((n - collected_n) * overshoot) + 128
        u = rng.integers(0, num_nodes, size=need, dtype=np.int32)
        v = rng.integers(0, num_nodes, size=need, dtype=np.int32)
        pairs = np.stack([u, v], axis=1)
        pairs = _remove_self_loops(pairs)

        if exclude_pairs is not None and len(exclude_pairs) > 0:
            # Encode both sets and subtract
            codes_new = (
                pairs[:, 0].astype(np.int64) * num_nodes
                + pairs[:, 1].astype(np.int64)
            )
            codes_excl = (
                exclude_pairs[:, 0].astype(np.int64) * num_nodes
                + exclude_pairs[:, 1].astype(np.int64)
            )
            excl_set = set(codes_excl.tolist())
            mask = np.array([c not in excl_set for c in codes_new.tolist()], dtype=bool)
            pairs = pairs[mask]

        collected.append(pairs)
        collected_n += len(pairs)
        if collected_n >= n:
            break

    if collected_n == 0:
        raise RuntimeError("Could not sample any easy negatives")

    result = np.concatenate(collected, axis=0)[:n]
    logger.debug("Easy negatives sampled: %d", len(result))
    return result


# ─────────────────────────────────────────────────────────────────────────────
# Hard negatives (2-hop)
# ─────────────────────────────────────────────────────────────────────────────

def sample_hard_negatives(
    gs: GraphStore,
    n: int,
    seed: int = 1,
    hub_degree_cap: int = 2_000,
    exclude_pairs: np.ndarray | None = None,
) -> np.ndarray:
    """
    Sample n hard (2-hop) negative edges u→w→v where u→v does not exist.

    Strategy: iterate over a random subset of edges u→w; for each, pick a
    random out-neighbour v of w.  Discard if u→v exists.

    Parameters
    ----------
    hub_degree_cap : int
        For nodes w whose out-degree exceeds this, we sub-sample their
        out-neighbours to avoid O(degree) cost and hub bias.

    Returns
    -------
    np.ndarray shape (≤n, 2) int32
    """
    rng = np.random.default_rng(seed)
    src, dst = gs.edge_list()

    # Shuffle edges for random traversal
    perm = rng.permutation(len(src))
    src = src[perm]
    dst = dst[perm]

    collected: list[np.ndarray] = []
    collected_n = 0
    target = int(n * 1.3)   # oversample to account for filtration loss

    for i in range(len(src)):
        if collected_n >= target:
            break
        u = int(src[i])
        w = int(dst[i])

        w_out = gs.out_neighbors(w)
        if len(w_out) == 0:
            continue

        # Sub-sample hub nodes
        if len(w_out) > hub_degree_cap:
            w_out = rng.choice(w_out, hub_degree_cap, replace=False)

        # Candidate 2-hop targets
        candidates = w_out[w_out != u]   # no self-loop
        if len(candidates) == 0:
            continue

        # Check which do not have a direct edge from u
        u_arr = np.full(len(candidates), u, dtype=np.int32)
        is_edge = gs.has_edge_batch(u_arr, candidates.astype(np.int32))
        candidates = candidates[~is_edge]

        if len(candidates) == 0:
            continue

        # Pick one at random
        v = int(rng.choice(candidates))
        collected.append(np.array([[u, v]], dtype=np.int32))
        collected_n += 1

    if collected_n == 0:
        logger.warning("No hard negatives found; falling back to easy negatives")
        return sample_easy_negatives(gs, n, seed=seed + 99, exclude_pairs=exclude_pairs)

    result = np.concatenate(collected, axis=0)
    result = _deduplicate(result)

    if exclude_pairs is not None and len(exclude_pairs) > 0:
        codes_r = result[:, 0].astype(np.int64) * gs.num_nodes + result[:, 1].astype(np.int64)
        codes_e = exclude_pairs[:, 0].astype(np.int64) * gs.num_nodes + exclude_pairs[:, 1].astype(np.int64)
        excl_set = set(codes_e.tolist())
        mask = np.array([c not in excl_set for c in codes_r.tolist()], dtype=bool)
        result = result[mask]

    result = result[:n]
    logger.debug("Hard negatives sampled: %d (target %d)", len(result), n)
    return result


# ─────────────────────────────────────────────────────────────────────────────
# Mixed sampler (public API)
# ─────────────────────────────────────────────────────────────────────────────

def sample_mixed_negatives(
    gs: GraphStore,
    n: int,
    hard_frac: float = 0.5,
    seed: int = 42,
    hub_degree_cap: int = 2_000,
    exclude_pairs: np.ndarray | None = None,
) -> np.ndarray:
    """
    Sample n negatives as a mix of hard (2-hop) and easy (random) negatives.

    Parameters
    ----------
    n : total number of negatives
    hard_frac : fraction that should be hard negatives (0.0 = all easy)
    seed : base seed; hard and easy receive seed and seed+1
    hub_degree_cap : passed through to sample_hard_negatives
    exclude_pairs : pairs (typically positive edges) to exclude

    Returns
    -------
    np.ndarray shape (n, 2) int32 — deduplicated and sorted
    """
    n_hard = int(n * hard_frac)
    n_easy = n - n_hard

    parts: list[np.ndarray] = []

    if n_hard > 0:
        logger.info("Sampling %d hard negatives …", n_hard)
        hard = sample_hard_negatives(
            gs, n_hard,
            seed=seed,
            hub_degree_cap=hub_degree_cap,
            exclude_pairs=exclude_pairs,
        )
        parts.append(hard)

    if n_easy > 0:
        logger.info("Sampling %d easy negatives …", n_easy)
        easy = sample_easy_negatives(
            gs, n_easy,
            seed=seed + 1,
            exclude_pairs=exclude_pairs,
        )
        parts.append(easy)

    combined = np.concatenate(parts, axis=0)
    combined = _deduplicate(combined)
    combined = _remove_self_loops(combined)

    if len(combined) > n:
        combined = combined[:n]
    elif len(combined) < n:
        logger.warning(
            "Could only sample %d / %d negatives; padding with easy negatives",
            len(combined), n,
        )
        extra = sample_easy_negatives(
            gs, n - len(combined), seed=seed + 100,
            exclude_pairs=exclude_pairs,
        )
        combined = np.concatenate([combined, extra], axis=0)[:n]

    logger.info("Mixed negatives ready: %d total", len(combined))
    return combined.astype(np.int32)
