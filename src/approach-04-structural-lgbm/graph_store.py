"""
GraphStore: memory-efficient directed graph for repeated structural feature extraction.

Design
------
* No NetworkX — raw numpy / scipy operations only.
* Node IDs in this dataset are DENSE integers in [0, num_nodes-1], so the
  node index IS the node ID.  No id→idx dictionary is needed.
* Adjacency is stored in two scipy CSR matrices (out and in) whose .indices
  arrays are sorted, enabling fast np.intersect1d intersection.
* has_edge uses a sorted int64 edge_codes array + binary search, which is
  O(log E) and fully vectorised over batches.

Memory footprint (24M edges, 4.87M nodes)
------------------------------------------
  out_csr.indices   96 MB  (int32)
  in_csr.indices    96 MB  (int32)
  indptr arrays      2 × 19 MB (int32)
  edge_codes        192 MB (int64, sorted)
  deg arrays          2 × 19 MB (int32)
  Total            ~441 MB            (fits comfortably in 96 GB RAM)
"""
from __future__ import annotations

import csv
import logging
import os
import time
from typing import Tuple

import numpy as np
from scipy.sparse import csr_matrix

logger = logging.getLogger(__name__)


class GraphStore:
    """
    Immutable directed graph optimised for batch structural feature extraction.

    Attributes
    ----------
    num_nodes : int
    num_edges : int
    out_indptr  : np.ndarray shape (num_nodes+1,) int32
    out_indices : np.ndarray shape (num_edges,)   int32  — sorted within each row
    in_indptr   : np.ndarray shape (num_nodes+1,) int32
    in_indices  : np.ndarray shape (num_edges,)   int32  — sorted within each row
    out_deg     : np.ndarray shape (num_nodes,)   int32
    in_deg      : np.ndarray shape (num_nodes,)   int32
    edge_codes  : np.ndarray shape (num_edges,)   int64  — sorted; u*num_nodes+v
    """

    # ------------------------------------------------------------------
    # Construction
    # ------------------------------------------------------------------

    def __init__(
        self,
        num_nodes: int,
        src: np.ndarray,
        dst: np.ndarray,
    ) -> None:
        """
        Build from flat edge arrays.

        Parameters
        ----------
        num_nodes : int
            Total number of nodes (max_id + 1 for dense IDs).
        src, dst : int32/int64 arrays of equal length
            Source and destination node IDs for every directed edge.
        """
        t0 = time.time()
        self.num_nodes = num_nodes
        self.num_edges = len(src)

        logger.info(
            "Building GraphStore: %d nodes, %d edges …", num_nodes, self.num_edges
        )

        src32 = src.astype(np.int32)
        dst32 = dst.astype(np.int32)

        # ── Out-adjacency CSR ─────────────────────────────────────────
        ones = np.ones(self.num_edges, dtype=np.int8)
        out_csr = csr_matrix(
            (ones, (src32, dst32)), shape=(num_nodes, num_nodes), dtype=np.int8
        )
        out_csr.sort_indices()  # guarantee sorted columns within each row

        self.out_indptr  = out_csr.indptr.astype(np.int32)
        self.out_indices = out_csr.indices.astype(np.int32)
        del out_csr  # free the full sparse object; keep only the arrays

        # ── In-adjacency CSR (transpose) ──────────────────────────────
        in_csr = csr_matrix(
            (ones, (dst32, src32)), shape=(num_nodes, num_nodes), dtype=np.int8
        )
        in_csr.sort_indices()

        self.in_indptr  = in_csr.indptr.astype(np.int32)
        self.in_indices = in_csr.indices.astype(np.int32)
        del in_csr
        del ones

        # ── Degree vectors ─────────────────────────────────────────────
        self.out_deg = (
            self.out_indptr[1:] - self.out_indptr[:-1]
        ).astype(np.int32)
        self.in_deg = (
            self.in_indptr[1:] - self.in_indptr[:-1]
        ).astype(np.int32)

        # ── Sorted edge codes for O(log E) has_edge ───────────────────
        codes = src.astype(np.int64) * num_nodes + dst.astype(np.int64)
        codes.sort()
        self.edge_codes = codes

        elapsed = time.time() - t0
        logger.info("GraphStore built in %.1f s", elapsed)

    # ------------------------------------------------------------------
    # Neighbourhood accessors
    # ------------------------------------------------------------------

    def out_neighbors(self, u: int) -> np.ndarray:
        """Sorted int32 array of out-neighbours of node u."""
        lo, hi = int(self.out_indptr[u]), int(self.out_indptr[u + 1])
        return self.out_indices[lo:hi]

    def in_neighbors(self, v: int) -> np.ndarray:
        """Sorted int32 array of in-neighbours of node v."""
        lo, hi = int(self.in_indptr[v]), int(self.in_indptr[v + 1])
        return self.in_indices[lo:hi]

    # ------------------------------------------------------------------
    # Edge existence
    # ------------------------------------------------------------------

    def has_edge(self, u: int, v: int) -> bool:
        """O(log E) scalar edge check."""
        code = int(u) * self.num_nodes + int(v)
        idx = np.searchsorted(self.edge_codes, code)
        return bool(idx < self.num_edges and self.edge_codes[idx] == code)

    def has_edge_batch(
        self, u_arr: np.ndarray, v_arr: np.ndarray
    ) -> np.ndarray:
        """
        Vectorised O(n log E) edge-existence check for arrays of pairs.

        Returns
        -------
        np.ndarray of bool, same length as u_arr.
        """
        codes = u_arr.astype(np.int64) * self.num_nodes + v_arr.astype(np.int64)
        indices = np.searchsorted(self.edge_codes, codes)
        valid = indices < self.num_edges
        result = valid.copy()
        result[valid] = self.edge_codes[indices[valid]] == codes[valid]
        return result

    # ------------------------------------------------------------------
    # Classic factory
    # ------------------------------------------------------------------

    @classmethod
    def from_adjacency_csv(
        cls,
        path: str,
        num_nodes: int | None = None,
    ) -> "GraphStore":
        """
        Parse a ragged adjacency-list CSV (no header).

        Format:  node_id, neighbor1, neighbor2, ...
        Node IDs must be non-negative integers.
        If num_nodes is None it is inferred from max_id + 1.

        Returns
        -------
        GraphStore instance.
        """
        if not os.path.exists(path):
            raise FileNotFoundError(f"Train graph not found: {path}")

        logger.info("Parsing adjacency CSV: %s …", path)
        t0 = time.time()

        src_list: list[int] = []
        dst_list: list[int] = []
        max_id = 0

        with open(path, newline="") as fh:
            reader = csv.reader(fh)
            for row in reader:
                if not row or not row[0].strip():
                    continue
                u = int(row[0].strip())
                if u > max_id:
                    max_id = u
                for tok in row[1:]:
                    tok = tok.strip()
                    if tok:
                        v = int(tok)
                        if v > max_id:
                            max_id = v
                        src_list.append(u)
                        dst_list.append(v)

        parse_time = time.time() - t0
        logger.info(
            "Parsed %d edges in %.1f s; max_id=%d", len(src_list), parse_time, max_id
        )

        if num_nodes is None:
            num_nodes = max_id + 1

        src = np.array(src_list, dtype=np.int32)
        dst = np.array(dst_list, dtype=np.int32)
        del src_list, dst_list

        return cls(num_nodes, src, dst)

    # ------------------------------------------------------------------
    # Utility
    # ------------------------------------------------------------------

    def edge_list(self) -> Tuple[np.ndarray, np.ndarray]:
        """Return (src, dst) int32 arrays of all edges."""
        # Reconstruct from CSR
        src = np.repeat(
            np.arange(self.num_nodes, dtype=np.int32),
            self.out_indptr[1:] - self.out_indptr[:-1],
        )
        return src, self.out_indices.copy()

    def subgraph_without_edges(
        self, remove_src: np.ndarray, remove_dst: np.ndarray
    ) -> "GraphStore":
        """
        Return a new GraphStore with the specified edges removed.

        Used to build a training graph that excludes validation/test edges,
        preventing feature leakage during dataset construction.
        """
        full_src, full_dst = self.edge_list()
        remove_codes = (
            remove_src.astype(np.int64) * self.num_nodes
            + remove_dst.astype(np.int64)
        )
        remove_set = set(remove_codes.tolist())

        existing_codes = (
            full_src.astype(np.int64) * self.num_nodes
            + full_dst.astype(np.int64)
        )
        mask = np.array(
            [c not in remove_set for c in existing_codes.tolist()], dtype=bool
        )
        logger.info(
            "subgraph_without_edges: removing %d edges, keeping %d",
            mask.size - mask.sum(),
            mask.sum(),
        )
        return GraphStore(self.num_nodes, full_src[mask], full_dst[mask])

    def __repr__(self) -> str:
        return (
            f"GraphStore(num_nodes={self.num_nodes:,}, "
            f"num_edges={self.num_edges:,})"
        )
