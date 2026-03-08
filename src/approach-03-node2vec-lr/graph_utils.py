"""
Graph utilities: loading, edge splitting, and negative sampling.
"""
import csv
import logging
import random
from typing import List, Set, Tuple

import networkx as nx
import numpy as np
from tqdm import tqdm

logger = logging.getLogger(__name__)


# ── Graph Loading ──────────────────────────────────────────────────────────────

def _is_header_cell(value: str) -> bool:
    """
    Return True if *value* looks like a column label rather than a node ID.
    A cell is treated as a header if it is non-empty and cannot be parsed
    as an integer or float.
    """
    stripped = value.strip()
    if not stripped:
        return False
    try:
        float(stripped)
        return False
    except ValueError:
        return True


def load_graph_from_adjacency_csv(filepath: str, directed: bool = True) -> nx.DiGraph:
    """
    Load a directed (or undirected) graph from a *ragged* adjacency-list CSV.

    Each row has a variable number of columns:
        source_node, neighbor_1, neighbor_2, ..., neighbor_N

    Rows are read one at a time with the built-in ``csv`` module so the file
    is never fully loaded into memory — no NaN-padded DataFrame, no up-front
    column-count scan.  This is safe and efficient for graphs with tens of
    millions of edges and highly variable degree distributions.

    Format rules
    ------------
    - The first column of every data row is the source node ID (integer).
    - All subsequent non-empty fields on that row are out-neighbour IDs.
    - A header row is auto-detected: if the first cell of the first row
      cannot be parsed as a number it is skipped.
    - Empty fields within a row (e.g. trailing commas) are silently ignored.
    - Rows whose source cell is missing or non-numeric are skipped and
      counted so you can spot malformed input early.

    Parameters
    ----------
    filepath : str
        Path to the adjacency-list CSV file.
    directed : bool
        If True (default) builds a ``DiGraph``; otherwise an undirected ``Graph``.

    Returns
    -------
    nx.DiGraph or nx.Graph
    """
    logger.info(f"Loading graph from '{filepath}' ...")

    G: nx.Graph = nx.DiGraph() if directed else nx.Graph()

    skipped_rows     = 0   # rows whose source cell is absent or non-numeric
    skipped_neighbor = 0   # individual neighbor cells that could not be parsed
    total_rows       = 0

    with open(filepath, newline="", encoding="utf-8") as fh:
        reader = csv.reader(fh)

        # ── Header detection ──────────────────────────────────────────────────
        # Peek at the first row; put it back if it contains data.
        try:
            first_row = next(reader)
        except StopIteration:
            logger.warning("CSV file is empty — returning empty graph.")
            return G

        if first_row and _is_header_cell(first_row[0]):
            logger.info(f"Header row detected and skipped: {first_row}")
        else:
            # Not a header — process it as a data row
            reader = _prepend_row(first_row, reader)  # type: ignore[assignment]

        # ── Row-by-row streaming parse ────────────────────────────────────────
        for row in tqdm(reader, desc="Building graph", unit="rows", mininterval=2.0):
            total_rows += 1

            if not row:          # completely blank line
                skipped_rows += 1
                continue

            src_cell = row[0].strip()
            if not src_cell:     # source field is empty
                skipped_rows += 1
                continue

            try:
                src = int(src_cell)
            except ValueError:
                logger.debug(f"Row {total_rows}: cannot parse source '{src_cell}' — skipped.")
                skipped_rows += 1
                continue

            G.add_node(src)

            # Every field after the first is a potential neighbour
            for cell in row[1:]:
                cell = cell.strip()
                if not cell:     # empty field (e.g. trailing comma)
                    continue
                try:
                    G.add_edge(src, int(cell))
                except ValueError:
                    logger.debug(
                        f"Row {total_rows}: cannot parse neighbour '{cell}' "
                        f"for source {src} — skipped."
                    )
                    skipped_neighbor += 1

    logger.info(
        f"Graph loaded  →  {G.number_of_nodes():,} nodes, "
        f"{G.number_of_edges():,} edges  |  "
        f"rows processed: {total_rows:,}  |  "
        f"skipped rows: {skipped_rows:,}  |  "
        f"skipped neighbour cells: {skipped_neighbor:,}"
    )
    return G


# ── Internal helper ────────────────────────────────────────────────────────────

def _prepend_row(row, iterator):
    """Yield *row* first, then yield everything from *iterator*."""
    yield row
    yield from iterator


# ── Edge Splitting ─────────────────────────────────────────────────────────────

def train_val_test_split(
    G: nx.DiGraph,
    val_ratio:  float = 0.10,
    test_ratio: float = 0.10,
    neg_ratio:  float = 1.0,
    seed:       int   = 42,
) -> Tuple:
    """
    Split the graph edges into train / validation / test sets.

    Strategy
    --------
    1. Randomly shuffle all positive edges.
    2. Reserve the first `test_ratio` fraction for test, the next
       `val_ratio` fraction for validation, the rest for training.
    3. Build G_train by removing val + test edges.
    4. Sample balanced negative edges (node pairs with no edge).

    Returns
    -------
    G_train                       : training graph (val/test edges removed)
    train_pos, val_pos, test_pos  : positive edge lists
    train_neg, val_neg, test_neg  : negative edge lists
    """
    rng    = random.Random(seed)
    np_rng = np.random.default_rng(seed)

    all_edges = list(G.edges())
    rng.shuffle(all_edges)
    n = len(all_edges)

    n_test = int(n * test_ratio)
    n_val  = int(n * val_ratio)

    test_pos  = all_edges[:n_test]
    val_pos   = all_edges[n_test: n_test + n_val]
    train_pos = all_edges[n_test + n_val:]

    # Training graph: remove edges that appear in val / test sets
    G_train = G.copy()
    G_train.remove_edges_from(val_pos + test_pos)

    logger.info(
        f"Edge split  →  "
        f"train: {len(train_pos):,}  |  "
        f"val: {len(val_pos):,}  |  "
        f"test: {len(test_pos):,}"
    )
    logger.info(
        f"Training graph  →  {G_train.number_of_nodes():,} nodes, "
        f"{G_train.number_of_edges():,} edges"
    )

    # ── Hard Negative sampling ────────────────────────────────────────────────
    #
    # Why hard negatives matter for Twitter
    # ─────────────────────────────────────
    # Random node pairs are "easy" negatives — two arbitrary users almost
    # certainly share no common follows and look nothing alike structurally.
    # A classifier trained on easy negatives learns a trivial rule and
    # achieves inflated val/test scores that don't reflect real performance.
    #
    # Hard negatives are node pairs that LOOK like they should have an edge
    # but don't.  On a follow graph the most natural hard negative is:
    #
    #   u follows w,  w follows v,  but u does NOT follow v
    #
    # i.e. "friend-of-a-friend" pairs.  These share structural context
    # (they appear in the same walk windows) yet have no direct edge, so
    # the classifier must learn finer-grained structural distinctions.
    #
    # Strategy — three tiers, sampled in priority order:
    #
    #   Tier 1 — 2-hop paths  (u → w → v, no u → v)
    #     Strongest signal: v is reachable from u through a mutual follow.
    #     Directly mimics how Twitter's "who to follow" surface works.
    #
    #   Tier 2 — shared follower pairs  (w → u, w → v, no u → v)
    #     u and v are both followed by the same account w — they occupy
    #     similar audience positions (e.g. two sports journalists) but
    #     u doesn't follow v.  Tests whether interest-cluster similarity
    #     alone is enough to predict a follow.
    #
    #   Tier 3 — random fallback
    #     For any samples not filled by Tiers 1/2 (sparse nodes, dense
    #     graphs) we fall back to random sampling so n_samples is always met.
    #
    # hard_ratio controls the tier-1/2 budget:
    #   hard_ratio=0.8 → 80 % hard, 20 % random fallback

    existing_edges: Set[Tuple] = set(G.edges())
    nodes   = list(G.nodes())
    n_nodes = len(nodes)

    # Pre-build successor / predecessor index as plain lists for O(1) sampling.
    # NetworkX's adjacency views are fine for lookup but slow to sample from
    # repeatedly at scale; converting to lists pays off for millions of samples.
    logger.info("Pre-building adjacency index for hard negative sampling ...")
    successors   = {u: list(G.successors(u))   for u in G.nodes()}
    predecessors = {v: list(G.predecessors(v)) for v in G.nodes()}

    def _sample_hard_negatives(
        n_samples:  int,
        hard_ratio: float = 0.8,
    ) -> List[Tuple]:
        """
        Sample hard negatives for a Twitter-style directed follow graph.

        Parameters
        ----------
        n_samples  : total number of negative pairs to return
        hard_ratio : fraction of n_samples to attempt via Tier-1/2 strategies
                     before falling back to random sampling

        Returns
        -------
        List of (u, v) node pairs guaranteed to have no edge in G and not
        already present in existing_edges.
        """
        negatives:  List[Tuple] = []
        seen_candidates: Set[Tuple] = set()   # local dedup within this call

        def _is_valid(u, v) -> bool:
            """True iff (u,v) is a genuine unseen non-edge."""
            if u == v:
                return False
            candidate = (u, v)
            if candidate in existing_edges:
                return False
            if candidate in seen_candidates:
                return False
            return True

        def _accept(u, v) -> bool:
            """Validate, register in both dedup sets, append."""
            if not _is_valid(u, v):
                return False
            candidate = (u, v)
            negatives.append(candidate)
            seen_candidates.add(candidate)
            existing_edges.add(candidate)   # prevent re-use across splits
            return True

        n_hard    = int(n_samples * hard_ratio)
        max_attempts_per_tier = n_hard * 8   # generous budget before giving up

        # ── Tier 1: 2-hop paths  (u → w → v,  no u → v) ─────────────────────
        # Intuition: v is a "friend of a friend" — the most natural candidate
        # for a new follow that the model must learn to reject when absent.
        attempts = 0
        while len(negatives) < n_hard // 2 and attempts < max_attempts_per_tier:
            attempts += 1
            u = nodes[int(np_rng.integers(n_nodes))]
            u_out = successors.get(u)
            if not u_out:
                continue
            # Pick a random account u follows
            w = u_out[int(np_rng.integers(len(u_out)))]
            w_out = successors.get(w)
            if not w_out:
                continue
            # Pick someone w follows — candidate for u to follow
            v = w_out[int(np_rng.integers(len(w_out)))]
            _accept(u, v)

        tier1_count = len(negatives)
        logger.debug(f"Hard negatives — Tier 1 (2-hop): {tier1_count:,}")

        # ── Tier 2: shared-follower pairs  (w → u, w → v,  no u → v) ────────
        # Intuition: u and v share an audience (both followed by w) so they
        # likely occupy the same interest niche, making (u,v) a plausible but
        # absent edge.
        attempts = 0
        while len(negatives) < n_hard and attempts < max_attempts_per_tier:
            attempts += 1
            # Pick a random "pivot" follower w
            w = nodes[int(np_rng.integers(n_nodes))]
            w_out = successors.get(w)
            if not w_out or len(w_out) < 2:
                continue
            # Pick two distinct accounts that w follows
            idx_u, idx_v = np_rng.choice(len(w_out), size=2, replace=False)
            u = w_out[int(idx_u)]
            v = w_out[int(idx_v)]
            _accept(u, v)

        tier2_count = len(negatives) - tier1_count
        logger.debug(f"Hard negatives — Tier 2 (shared follower): {tier2_count:,}")

        # ── Tier 3: random fallback ───────────────────────────────────────────
        # Fills any shortfall from Tiers 1/2 (isolated nodes, very dense graph).
        fallback_attempts = 0
        max_fallback      = (n_samples - len(negatives)) * 30
        while len(negatives) < n_samples and fallback_attempts < max_fallback:
            fallback_attempts += 1
            u = nodes[int(np_rng.integers(n_nodes))]
            v = nodes[int(np_rng.integers(n_nodes))]
            _accept(u, v)

        fallback_count = len(negatives) - tier1_count - tier2_count
        logger.debug(f"Hard negatives — Tier 3 (random fallback): {fallback_count:,}")

        if len(negatives) < n_samples:
            logger.warning(
                f"Hard negative sampling: only collected {len(negatives):,}/"
                f"{n_samples:,} pairs  "
                f"(Tier1={tier1_count:,}, Tier2={tier2_count:,}, "
                f"Fallback={fallback_count:,}).  "
                "Graph may be very dense — consider reducing neg_ratio."
            )

        return negatives[:n_samples]

    logger.info("Sampling hard negative edges ...")
    n_train_neg = int(len(train_pos) * neg_ratio)
    n_val_neg   = int(len(val_pos)   * neg_ratio)
    n_test_neg  = int(len(test_pos)  * neg_ratio)

    train_neg = _sample_hard_negatives(n_train_neg)
    val_neg   = _sample_hard_negatives(n_val_neg)
    test_neg  = _sample_hard_negatives(n_test_neg)

    logger.info(
        f"Negatives sampled  →  "
        f"train: {len(train_neg):,}  |  "
        f"val: {len(val_neg):,}  |  "
        f"test: {len(test_neg):,}"
    )

    return (
        G_train,
        train_pos, val_pos,  test_pos,
        train_neg, val_neg,  test_neg,
    )
