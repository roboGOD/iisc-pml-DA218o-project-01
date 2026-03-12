"""
Community detection using the Leiden algorithm (leidenalg + igraph).

Runs once offline on the undirected projection of the training graph,
then caches community IDs to a .npy file.  Subsequent loads are instant.

Install dependencies:
    pip install leidenalg igraph

Memory note (4.87M nodes, 24M directed edges → ~48M undirected edges):
  igraph edge list    ~1.5 GB RAM
  Leiden partition   ~100 MB RAM
  Total              ~2 GB additional peak (safe on 96 GB machine)

Usage
-----
    from graph_store import GraphStore
    from community_store import CommunityStore, build_communities

    gs = GraphStore.from_adjacency_csv("data/raw/train.csv")
    community_ids = build_communities(gs, cache_path="data/processed/approach04b/communities.npy")
    cs = CommunityStore(community_ids)

    cs.same_community(u, v)          # bool
    cs.get_size(u)                    # int — size of u's community
"""
from __future__ import annotations

import logging
import os
import time

import numpy as np

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# Leiden builder (runs once, cached)
# ─────────────────────────────────────────────────────────────────────────────

def build_communities(
    gs,
    cache_path: str,
    resolution: float = 1.0,
    seed: int = 42,
) -> np.ndarray:
    """
    Run Leiden community detection on the undirected projection of *gs*.

    Parameters
    ----------
    gs         : GraphStore instance
    cache_path : path to save / load the int32 community-ID array
    resolution : Leiden resolution parameter (higher → smaller communities)
    seed       : random seed for reproducibility

    Returns
    -------
    np.ndarray shape (num_nodes,) int32
        community_ids[u] is the community index of node u.
    """
    if os.path.exists(cache_path):
        logger.info("Loading cached communities from %s", cache_path)
        ids = np.load(cache_path)
        n_comms = len(np.unique(ids))
        logger.info("  %d communities for %d nodes", n_comms, len(ids))
        return ids

    try:
        import igraph as ig
        import leidenalg
    except ImportError as e:
        raise ImportError(
            "leidenalg and igraph are required for community features.\n"
            "Install with:  pip install leidenalg igraph\n"
            f"Original error: {e}"
        ) from e

    # ── Build undirected edge list ─────────────────────────────────────
    logger.info("Building undirected edge list for Leiden (%d directed edges)…",
                gs.num_edges)
    t0 = time.time()

    src, dst = gs.edge_list()

    # Combine both directions, deduplicate, drop self-loops
    all_u = np.concatenate([src, dst])
    all_v = np.concatenate([dst, src])

    mask_nsl = all_u != all_v
    all_u = all_u[mask_nsl]
    all_v = all_v[mask_nsl]

    # Canonicalize u ≤ v to halve the edge count
    swap = all_u > all_v
    all_u[swap], all_v[swap] = all_v[swap], all_u[swap]

    codes = all_u.astype(np.int64) * gs.num_nodes + all_v.astype(np.int64)
    _, unique_idx = np.unique(codes, return_index=True)
    all_u = all_u[unique_idx]
    all_v = all_v[unique_idx]

    logger.info("  Undirected edges: %d (in %.1fs). Building igraph…",
                len(all_u), time.time() - t0)

    t1 = time.time()
    g = ig.Graph(
        n=gs.num_nodes,
        edges=list(zip(all_u.tolist(), all_v.tolist())),
        directed=False,
    )
    del all_u, all_v
    logger.info("  igraph built in %.1fs", time.time() - t1)

    # ── Run Leiden ─────────────────────────────────────────────────────
    logger.info("Running Leiden (resolution=%.2f, seed=%d)…", resolution, seed)
    t2 = time.time()
    partition = leidenalg.find_partition(
        g,
        leidenalg.RBConfigurationVertexPartition,
        resolution_parameter=resolution,
        seed=seed,
    )
    n_comms = len(partition)
    logger.info("  Leiden done in %.1fs — %d communities (avg size %.1f)",
                time.time() - t2, n_comms, gs.num_nodes / n_comms)

    community_ids = np.array(partition.membership, dtype=np.int32)

    os.makedirs(os.path.dirname(cache_path), exist_ok=True)
    np.save(cache_path, community_ids)
    logger.info("Communities saved → %s", cache_path)

    return community_ids


# ─────────────────────────────────────────────────────────────────────────────
# Fast accessor
# ─────────────────────────────────────────────────────────────────────────────

class CommunityStore:
    """
    Wraps a community-ID array for fast per-node and per-pair lookups.

    Attributes
    ----------
    community_ids : np.ndarray shape (num_nodes,) int32
    num_communities : int
    """

    def __init__(self, community_ids: np.ndarray) -> None:
        self.community_ids = community_ids.astype(np.int32)
        unique, counts = np.unique(self.community_ids, return_counts=True)
        self.num_communities = int(unique.max()) + 1
        # Size lookup: _sizes[community_id] = size of that community
        self._sizes = np.zeros(self.num_communities, dtype=np.int32)
        self._sizes[unique] = counts
        logger.info(
            "CommunityStore ready: %d nodes, %d communities",
            len(community_ids), len(unique),
        )

    # ── Scalar accessors ──────────────────────────────────────────────

    def get_community(self, u: int) -> int:
        return int(self.community_ids[u])

    def get_size(self, u: int) -> int:
        return int(self._sizes[self.community_ids[u]])

    def same_community(self, u: int, v: int) -> bool:
        return bool(self.community_ids[u] == self.community_ids[v])

    # ── Batch accessors (numpy arrays) ────────────────────────────────

    def get_community_batch(self, u_arr: np.ndarray) -> np.ndarray:
        return self.community_ids[u_arr]

    def get_size_batch(self, u_arr: np.ndarray) -> np.ndarray:
        return self._sizes[self.community_ids[u_arr]]

    def same_community_batch(
        self, u_arr: np.ndarray, v_arr: np.ndarray
    ) -> np.ndarray:
        return self.community_ids[u_arr] == self.community_ids[v_arr]

    # ── Factory ───────────────────────────────────────────────────────

    @classmethod
    def from_cache(cls, cache_path: str) -> "CommunityStore":
        if not os.path.exists(cache_path):
            raise FileNotFoundError(
                f"Community cache not found: {cache_path}\n"
                "Run build_communities() first via dataset_builder.py"
            )
        return cls(np.load(cache_path))

    def __repr__(self) -> str:
        return (
            f"CommunityStore(nodes={len(self.community_ids):,}, "
            f"communities={self.num_communities:,})"
        )
