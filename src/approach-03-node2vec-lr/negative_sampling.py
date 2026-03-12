"""
Negative edge sampler — approach-03 (Node2Vec + Logistic Regression).

Formalises the three-tier hard-negative strategy that was previously embedded
as a closure inside ``graph_utils.train_val_test_split``.  The logic is now
exposed as three independent, seed-reproducible public functions.

Strategy
--------
Easy  — uniform random (u, v) pairs from the training-graph node set.
        The graph is so sparse (24M / 4.87M² ≈ 0.001 %) that random pairs
        are almost never real edges; no vectorised existence check needed.

Hard  — Two complementary notions of "structural plausibility":
        Tier 1 — 2-hop paths  (u → w → v,  u → v absent)
          "friend of a friend" — the strongest structural negative.
          Directly mirrors Twitter's "who to follow" recommendation surface.
        Tier 2 — shared-follower  (w → u and w → v,  u → v absent)
          u and v are both followed by the same account w, placing them in
          the same interest cluster.  Tests whether audience overlap alone
          is enough to predict a follow.
        Tier 3 — random fallback
          Fills any shortfall left by Tiers 1 / 2 (sparse nodes, very dense
          subgraphs).  The budget is controlled by ``hard_frac``; only the
          unfilled fraction of that budget falls back to Tier 3.

Mixed — combines hard and easy in caller-specified proportions.
        Recommended defaults:
          train:               50 % hard, 50 % easy
          val / test-offline:  70 % hard, 30 % easy

Cross-split isolation
---------------------
Pass a single mutable ``seen_codes: set[int]`` across all split calls.
Each accepted pair is encoded as  u * n_nodes + v  and added to it.
Subsequent calls treat already-seen codes as if they were real edges — so
no negative can appear in two splits simultaneously.  Pass ``None`` (default)
to get per-call deduplication only.

Guarantees
----------
* No self-loops
* No true positive edges (checked against the edge set of the graph passed in)
* No duplicates within a returned batch
* No duplicates across splits when a shared ``seen_codes`` set is provided
* Reproducible from ``seed``
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Optional

import networkx as nx
import numpy as np

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# GraphAdapter — O(1) adjacency sampling wrapper around nx.DiGraph
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class GraphAdapter:
    """
    Thin, pre-indexed wrapper around a ``nx.DiGraph`` for fast sampling.

    Build once per graph with ``GraphAdapter.from_digraph(G)``, then pass to
    the sampling functions.  Pre-building the adjacency lists and node array
    pays off when the sampling functions are called many times.

    Attributes
    ----------
    successors   : node_id → list of out-neighbour ids
    predecessors : node_id → list of in-neighbour ids
    nodes        : array of all node ids (for uniform random node selection)
    n_nodes      : number of nodes (alias for len(nodes))
    edge_set     : set of (u, v) tuples — O(1) true-edge look-up
    """
    successors:   dict[int, list[int]]
    predecessors: dict[int, list[int]]
    nodes:        np.ndarray          # int64, shape (n_nodes,)
    n_nodes:      int
    edge_set:     set[tuple[int, int]]

    @classmethod
    def from_digraph(cls, G: nx.DiGraph) -> "GraphAdapter":
        """
        Build a GraphAdapter from a NetworkX DiGraph.

        This is O(V + E) and allocates memory proportional to the graph size.
        On the 4.87 M-node Twitter graph it takes ~30 s and ~4 GB; call it
        once and reuse across all split operations.
        """
        logger.info(
            "GraphAdapter: building adjacency index  "
            "(%d nodes, %d edges) ...",
            G.number_of_nodes(), G.number_of_edges(),
        )
        succ   = {u: list(G.successors(u))   for u in G.nodes()}
        pred   = {v: list(G.predecessors(v)) for v in G.nodes()}
        nodes  = np.array(list(G.nodes()), dtype=np.int64)
        logger.info("GraphAdapter: adjacency index ready.")
        return cls(
            successors=succ,
            predecessors=pred,
            nodes=nodes,
            n_nodes=len(nodes),
            edge_set=set(G.edges()),
        )


# ─────────────────────────────────────────────────────────────────────────────
# Internal helpers
# ─────────────────────────────────────────────────────────────────────────────

def _encode(u: int, v: int, n_nodes: int) -> int:
    return u * n_nodes + v


def _is_valid(
    u: int,
    v: int,
    adapter: GraphAdapter,
    seen_codes: set[int],
) -> bool:
    """Return True iff (u, v) is a genuine, previously-unseen non-edge."""
    if u == v:
        return False
    if (u, v) in adapter.edge_set:
        return False
    code = _encode(u, v, adapter.n_nodes)
    if code in seen_codes:
        return False
    return True


# ─────────────────────────────────────────────────────────────────────────────
# Public API
# ─────────────────────────────────────────────────────────────────────────────

def sample_easy_negatives(
    adapter: GraphAdapter,
    n: int,
    seed: int = 0,
    seen_codes: Optional[set[int]] = None,
) -> list[tuple[int, int]]:
    """
    Sample ``n`` easy (uniformly random) negative edges.

    Parameters
    ----------
    adapter    : pre-built GraphAdapter for the training graph
    n          : number of negatives to return
    seed       : numpy Generator seed for reproducibility
    seen_codes : shared mutable set of encoded pairs (u * n_nodes + v) already
                 used in previous split calls.  Updated in-place.
                 Pass ``None`` to skip cross-split isolation.

    Returns
    -------
    list of (u, v) integer tuples, len == n (or fewer if the quota cannot be
    met within the attempt budget — extremely unlikely on a sparse graph).
    """
    if seen_codes is None:
        seen_codes = set()

    rng      = np.random.default_rng(seed)
    nodes    = adapter.nodes
    n_nodes  = adapter.n_nodes
    negatives: list[tuple[int, int]] = []

    max_attempts = n * 20
    attempts = 0
    while len(negatives) < n and attempts < max_attempts:
        attempts += 1
        u = int(nodes[rng.integers(n_nodes)])
        v = int(nodes[rng.integers(n_nodes)])
        if _is_valid(u, v, adapter, seen_codes):
            negatives.append((u, v))
            seen_codes.add(_encode(u, v, n_nodes))

    if len(negatives) < n:
        logger.warning(
            "sample_easy_negatives: only collected %d / %d pairs "
            "(attempt budget exhausted).",
            len(negatives), n,
        )
    logger.debug("Easy negatives sampled: %d", len(negatives))
    return negatives


def sample_hard_negatives(
    adapter: GraphAdapter,
    n: int,
    seed: int = 1,
    hard_frac: float = 0.8,
    seen_codes: Optional[set[int]] = None,
) -> list[tuple[int, int]]:
    """
    Sample ``n`` hard negative edges using two structural strategies.

    The function mirrors the original three-tier logic in
    ``graph_utils._sample_hard_negatives`` but is now a standalone, testable
    function with an explicit interface.

    Tier 1 — 2-hop paths  (u → w → v,  u → v absent)
        Covers ``hard_frac * n / 2`` of the budget.
    Tier 2 — shared-follower  (w → u, w → v,  u → v absent)
        Covers the remaining ``hard_frac * n / 2`` of the budget.
    Tier 3 — random fallback
        Fills whatever Tiers 1 & 2 could not fill within their attempt budgets.

    Parameters
    ----------
    adapter    : pre-built GraphAdapter for the training graph
    n          : number of negatives to return
    seed       : numpy Generator seed for reproducibility
    hard_frac  : fraction of ``n`` to attempt via Tier-1/2 before Tier-3
                 random fallback kicks in.  Default 0.8 = 80 % hard.
    seen_codes : shared mutable set — see ``sample_easy_negatives``.

    Returns
    -------
    list of (u, v) integer tuples.
    """
    if seen_codes is None:
        seen_codes = set()

    rng     = np.random.default_rng(seed)
    nodes   = adapter.nodes
    n_nodes = adapter.n_nodes
    succ    = adapter.successors
    negatives: list[tuple[int, int]] = []

    n_hard            = int(n * hard_frac)
    max_per_tier      = n_hard * 8   # generous attempt budget per tier

    def _accept(u: int, v: int) -> bool:
        if not _is_valid(u, v, adapter, seen_codes):
            return False
        negatives.append((u, v))
        seen_codes.add(_encode(u, v, n_nodes))
        return True

    # ── Tier 1: 2-hop paths  (u → w → v,  no u → v) ─────────────────────────
    tier1_target  = n_hard // 2
    attempts      = 0
    while len(negatives) < tier1_target and attempts < max_per_tier:
        attempts += 1
        u     = int(nodes[rng.integers(n_nodes)])
        u_out = succ.get(u)
        if not u_out:
            continue
        w     = u_out[int(rng.integers(len(u_out)))]
        w_out = succ.get(w)
        if not w_out:
            continue
        v = w_out[int(rng.integers(len(w_out)))]
        _accept(u, v)

    tier1_count = len(negatives)
    logger.debug("Hard negatives — Tier 1 (2-hop): %d", tier1_count)

    # ── Tier 2: shared-follower  (w → u, w → v,  no u → v) ──────────────────
    attempts = 0
    while len(negatives) < n_hard and attempts < max_per_tier:
        attempts += 1
        w     = int(nodes[rng.integers(n_nodes)])
        w_out = succ.get(w)
        if not w_out or len(w_out) < 2:
            continue
        idx_u, idx_v = rng.choice(len(w_out), size=2, replace=False)
        u = w_out[int(idx_u)]
        v = w_out[int(idx_v)]
        _accept(u, v)

    tier2_count = len(negatives) - tier1_count
    logger.debug("Hard negatives — Tier 2 (shared-follower): %d", tier2_count)

    # ── Tier 3: random fallback ───────────────────────────────────────────────
    fallback_budget = (n - len(negatives)) * 30
    fallback_attempts = 0
    while len(negatives) < n and fallback_attempts < fallback_budget:
        fallback_attempts += 1
        u = int(nodes[rng.integers(n_nodes)])
        v = int(nodes[rng.integers(n_nodes)])
        _accept(u, v)

    fallback_count = len(negatives) - tier1_count - tier2_count
    logger.debug("Hard negatives — Tier 3 (random fallback): %d", fallback_count)

    if len(negatives) < n:
        logger.warning(
            "sample_hard_negatives: only collected %d / %d pairs  "
            "(Tier1=%d, Tier2=%d, Fallback=%d).  "
            "Graph may be very dense — consider reducing neg_ratio.",
            len(negatives), n, tier1_count, tier2_count, fallback_count,
        )
    logger.debug("Hard negatives sampled: %d total", len(negatives))
    return negatives[:n]


def sample_mixed_negatives(
    adapter: GraphAdapter,
    n: int,
    hard_frac: float = 0.5,
    easy_frac: float = 0.5,
    seed: int = 42,
    seen_codes: Optional[set[int]] = None,
) -> list[tuple[int, int]]:
    """
    Sample ``n`` negatives as a blend of hard and easy.

    Recommended split-specific defaults
    ------------------------------------
    train:              hard_frac=0.5, easy_frac=0.5
    val / test-offline: hard_frac=0.7, easy_frac=0.3

    Parameters
    ----------
    adapter    : pre-built GraphAdapter for the training graph
    n          : total negatives to return
    hard_frac  : fraction to draw from ``sample_hard_negatives``
    easy_frac  : fraction to draw from ``sample_easy_negatives``
                 (hard_frac + easy_frac need not equal 1.0 exactly; both are
                  rescaled so that hard + easy == n in total)
    seed       : base RNG seed; hard uses ``seed``, easy uses ``seed + 1``
    seen_codes : shared mutable set for cross-split isolation

    Returns
    -------
    list of (u, v) integer tuples, len == n
    """
    if seen_codes is None:
        seen_codes = set()

    total_frac = hard_frac + easy_frac
    if total_frac <= 0:
        raise ValueError("hard_frac + easy_frac must be > 0")

    n_hard = round(n * hard_frac / total_frac)
    n_easy = n - n_hard

    negatives: list[tuple[int, int]] = []

    if n_hard > 0:
        logger.info("Sampling %d hard negatives (seed=%d) ...", n_hard, seed)
        hard = sample_hard_negatives(
            adapter, n_hard,
            seed=seed,
            seen_codes=seen_codes,
        )
        negatives.extend(hard)

    if n_easy > 0:
        logger.info("Sampling %d easy negatives (seed=%d) ...", n_easy, seed + 1)
        easy = sample_easy_negatives(
            adapter, n_easy,
            seed=seed + 1,
            seen_codes=seen_codes,
        )
        negatives.extend(easy)

    # Final dedup pass (should be a no-op given seen_codes, but be safe)
    seen_final: set[tuple[int, int]] = set()
    unique: list[tuple[int, int]] = []
    for pair in negatives:
        if pair not in seen_final:
            seen_final.add(pair)
            unique.append(pair)

    if len(unique) < n:
        logger.warning(
            "sample_mixed_negatives: only %d / %d unique pairs collected.",
            len(unique), n,
        )

    logger.info(
        "Mixed negatives ready: %d total  (hard=%d, easy=%d)",
        len(unique), n_hard, n_easy,
    )
    return unique[:n]
