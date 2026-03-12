"""
EmbeddingStore: fast numpy lookup layer over a Gensim Word2Vec model.

Uses a pre-built dense matrix so that edge features (hadamard, L1, L2, etc.)
are computed with numpy vectorisation instead of per-node Python lookups.

Memory (128-dim, 4.87M nodes):
  matrix float32   2.5 GB — fits in 96 GB RAM

Usage
-----
    es = EmbeddingStore.from_gensim_model("model/approach-03/node_embeddings.model")
    feats = es.edge_features_batch(u_arr, v_arr, operator="hadamard")
    # → np.ndarray shape (N, 128) float32
"""
from __future__ import annotations

import logging
import os
import time
from typing import Callable

import numpy as np

logger = logging.getLogger(__name__)

_OPERATORS: dict[str, Callable[[np.ndarray, np.ndarray], np.ndarray]] = {
    "hadamard": lambda u, v: u * v,
    "average":  lambda u, v: (u + v) * 0.5,
    "l1":       lambda u, v: np.abs(u - v),
    "l2":       lambda u, v: (u - v) ** 2,
}


class EmbeddingStore:
    """
    Dense embedding matrix for all graph nodes.

    Nodes not seen during Node2Vec training (isolated nodes that were never
    part of a walk) receive a zero vector.

    Attributes
    ----------
    matrix : np.ndarray shape (num_nodes, dim) float32
    dim    : int
    """

    def __init__(self, matrix: np.ndarray) -> None:
        self.matrix = matrix.astype(np.float32)
        self.num_nodes, self.dim = matrix.shape

    # ── Lookup / edge features ────────────────────────────────────────

    def get_vector(self, node: int) -> np.ndarray:
        """Return the float32 embedding of *node* (zero if out-of-range)."""
        if 0 <= node < self.num_nodes:
            return self.matrix[node]
        return np.zeros(self.dim, dtype=np.float32)

    def get_vectors_batch(self, nodes: np.ndarray) -> np.ndarray:
        """
        Return shape (N, dim) embedding matrix for *nodes*.
        Out-of-range IDs map to zero rows.
        """
        valid = (nodes >= 0) & (nodes < self.num_nodes)
        out = np.zeros((len(nodes), self.dim), dtype=np.float32)
        out[valid] = self.matrix[nodes[valid]]
        return out

    def edge_features_batch(
        self,
        u_arr: np.ndarray,
        v_arr: np.ndarray,
        operator: str = "hadamard",
    ) -> np.ndarray:
        """
        Compute edge feature matrix for (u, v) pairs.

        Parameters
        ----------
        u_arr, v_arr : int arrays of equal length N
        operator     : "hadamard" | "average" | "l1" | "l2"

        Returns
        -------
        np.ndarray shape (N, dim) float32
        """
        if operator not in _OPERATORS:
            raise ValueError(
                f"Unknown operator {operator!r}. "
                f"Choose from: {list(_OPERATORS)}"
            )
        fn = _OPERATORS[operator]
        emb_u = self.get_vectors_batch(u_arr.astype(np.int64))
        emb_v = self.get_vectors_batch(v_arr.astype(np.int64))
        return fn(emb_u, emb_v)

    # ── Factory ───────────────────────────────────────────────────────

    @classmethod
    def from_gensim_model(
        cls,
        model_path: str,
        num_nodes: int | None = None,
    ) -> "EmbeddingStore":
        """
        Build a dense matrix from a Gensim KeyedVectors file.

        approach-03 saves ``self.wv`` (KeyedVectors) via ``wv.save(path)``,
        which writes two files:
          - ``path``               — small metadata pickle
          - ``path.vectors.npy``   — ~2.5 GB float32 matrix

        Node IDs are stored as string keys (str(node_id)) in the vocab.
        If *num_nodes* is None it is inferred from ``max(int(key)) + 1``.

        Parameters
        ----------
        model_path : path matching ``PATHS["embeddings"]`` in approach-03 config
        num_nodes  : expected number of nodes (optional)
        """
        if not os.path.exists(model_path):
            raise FileNotFoundError(
                f"Node2Vec model not found: {model_path}\n"
                "Run approach-03's train.py first to generate embeddings."
            )

        # approach-03 saves self.wv (KeyedVectors) directly via wv.save(),
        # NOT the full Word2Vec model — so we must load with KeyedVectors.load().
        logger.info("Loading Gensim KeyedVectors: %s …", model_path)
        t0 = time.time()

        try:
            from gensim.models import KeyedVectors
        except ImportError:
            raise ImportError(
                "gensim is required. Install with:  pip install gensim"
            )

        wv = KeyedVectors.load(model_path)
        dim = wv.vector_size

        # Infer num_nodes from vocab if not provided
        int_keys = [int(k) for k in wv.key_to_index.keys()]
        inferred_n = max(int_keys) + 1
        actual_n = num_nodes if num_nodes is not None else inferred_n
        logger.info(
            "  vocab size=%d, dim=%d, num_nodes=%d (in %.1fs)",
            len(int_keys), dim, actual_n, time.time() - t0,
        )

        # Build dense matrix (zero-filled; only known nodes filled in)
        t1 = time.time()
        matrix = np.zeros((actual_n, dim), dtype=np.float32)
        for key in wv.key_to_index:
            node_id = int(key)
            if 0 <= node_id < actual_n:
                matrix[node_id] = wv[key]

        coverage = len(int_keys) / actual_n * 100
        logger.info(
            "  Dense matrix built in %.1fs  (coverage %.1f%% nodes have embeddings)",
            time.time() - t1, coverage,
        )
        del wv

        return cls(matrix)

    def __repr__(self) -> str:
        return (
            f"EmbeddingStore(num_nodes={self.num_nodes:,}, dim={self.dim}, "
            f"memory={self.matrix.nbytes / 1e9:.2f} GB)"
        )
