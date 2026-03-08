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
