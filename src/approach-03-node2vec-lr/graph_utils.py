"""
Graph utilities: loading, edge splitting, and negative sampling.
"""
import logging
import random
from typing import List, Set, Tuple

import networkx as nx
import numpy as np
import pandas as pd
from tqdm import tqdm

logger = logging.getLogger(__name__)


# ── Graph Loading ──────────────────────────────────────────────────────────────

def load_graph_from_adjacency_csv(filepath: str, directed: bool = True) -> nx.DiGraph:
    """
    Load a directed (or undirected) graph from an adjacency-list CSV.

    Expected CSV format — header is optional and auto-detected:
        node_id, neighbor_1, neighbor_2, neighbor_3, ...

    The first column is the source node; every subsequent non-NaN value is
    an out-neighbour of that source.

    Parameters
    ----------
    filepath : path to the adjacency-list CSV file
    directed : if True, builds a DiGraph; otherwise an undirected Graph

    Returns
    -------
    NetworkX DiGraph (or Graph)
    """
    logger.info(f"Loading graph from '{filepath}' ...")

    # Auto-detect whether the first row is a header
    peek = pd.read_csv(filepath, header=None, nrows=1)
    first_cell = str(peek.iloc[0, 0]).strip().lower()
    has_header = not first_cell.lstrip("-").replace(".", "", 1).isdigit()

    df = pd.read_csv(filepath, header=0 if has_header else None, low_memory=False)

    G = nx.DiGraph() if directed else nx.Graph()
    skipped_rows = 0

    for _, row in tqdm(df.iterrows(), total=len(df), desc="Building graph", unit="rows"):
        src_raw = row.iloc[0]
        if pd.isna(src_raw):
            skipped_rows += 1
            continue
        src = int(src_raw)
        G.add_node(src)
        for val in row.iloc[1:]:
            if pd.notna(val):
                try:
                    G.add_edge(src, int(val))
                except (ValueError, TypeError):
                    pass  # ignore non-numeric neighbour cells

    logger.info(
        f"Graph loaded  →  {G.number_of_nodes():,} nodes, "
        f"{G.number_of_edges():,} edges  "
        f"(skipped {skipped_rows} malformed rows)"
    )
    return G


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

    # ── Negative sampling ────────────────────────────────────────────────────
    existing_edges: Set[Tuple] = set(G.edges())
    nodes = list(G.nodes())
    n_nodes = len(nodes)

    def _sample_negatives(n_samples: int) -> List[Tuple]:
        """Sample node pairs that are guaranteed not to be edges in G."""
        negatives: List[Tuple] = []
        max_attempts = n_samples * 30
        attempts = 0
        while len(negatives) < n_samples and attempts < max_attempts:
            idx_u = int(np_rng.integers(n_nodes))
            idx_v = int(np_rng.integers(n_nodes))
            u, v = nodes[idx_u], nodes[idx_v]
            candidate = (u, v)
            if u != v and candidate not in existing_edges:
                negatives.append(candidate)
                existing_edges.add(candidate)   # prevent re-use across splits
            attempts += 1

        if len(negatives) < n_samples:
            logger.warning(
                f"Negative sampling: only found {len(negatives)}/{n_samples} "
                "pairs (graph may be very dense)."
            )
        return negatives

    logger.info("Sampling negative edges ...")
    n_train_neg = int(len(train_pos) * neg_ratio)
    n_val_neg   = int(len(val_pos)   * neg_ratio)
    n_test_neg  = int(len(test_pos)  * neg_ratio)

    train_neg = _sample_negatives(n_train_neg)
    val_neg   = _sample_negatives(n_val_neg)
    test_neg  = _sample_negatives(n_test_neg)

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
