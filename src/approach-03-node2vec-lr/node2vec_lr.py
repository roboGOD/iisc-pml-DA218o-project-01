"""
Core model: Node2Vec embeddings (via PecanPy) → Logistic Regression link predictor.

Why PecanPy over the node2vec library
──────────────────────────────────────
The `node2vec` library precomputes and stores transition probabilities for every
edge before walks begin — O(edges × avg_degree) memory.  On a 24M-edge Twitter
graph this requires 30–50 GB RAM and hours of preprocessing.

PecanPy fixes both problems:
  • Transition probs computed on-the-fly during walks → O(edges) memory
  • Walk generation parallelised with numba JIT → 10–100× faster in practice
  • Three modes for different memory/speed trade-offs (see fit_embeddings)
  • Reads a CSR-format graph directly from an edge-list file, bypassing NetworkX

Training workflow
-----------------
1. predictor.fit_embeddings(edge_list_path)  — PecanPy Node2Vec walks
2. predictor.fit_classifier(pos, neg)        — LR on Hadamard edge features
3. predictor.evaluate(pos, neg, "val")       — print + return metrics
4. predictor.predict(edges)                  — binary 0 / 1 labels
5. predictor.predict_proba(edges)            — probability scores
6. predictor.save(...)  /  .load(...)        — model persistence
"""
import logging
import os
import tempfile
from typing import Dict, List, Optional, Tuple

import joblib
import networkx as nx
import numpy as np
from pecanpy import pecanpy as ppy
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    classification_report,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from features import build_dataset, edge_features

logger = logging.getLogger(__name__)

# PecanPy modes — choose based on available RAM:
#
#   SparseOTF  — transition probs computed on-the-fly from a sparse adjacency
#                matrix.  Lowest memory (a few GB for 24M edges).
#                Best default for large graphs on limited hardware.
#
#   DenseOTF   — probs computed on-the-fly from a dense matrix.  Faster walks
#                than SparseOTF but uses more RAM.  Good if you have 32 GB+.
#
#   PreComp    — probs precomputed and cached (closest to the node2vec library).
#                Fastest walk generation but highest memory — only viable for
#                small-to-medium graphs.
_PECANPY_MODES = {
    "SparseOTF": ppy.SparseOTF,
    "DenseOTF":  ppy.DenseOTF,
    "PreComp":   ppy.PreComp,
}


class Node2VecLinkPredictor:
    """
    Directed link-prediction via PecanPy Node2Vec embeddings + Logistic Regression.

    All heavy lifting is split into explicit steps so each stage can be
    inspected, swapped, or rerun independently.
    """

    def __init__(
        self,
        n2v_config:       Dict,
        w2v_config:       Dict,
        lr_config:        Dict,
        feature_operator: str = "hadamard",
        seed:             int = 42,
    ):
        self.n2v_config       = n2v_config
        self.w2v_config       = w2v_config
        self.lr_config        = lr_config
        self.feature_operator = feature_operator
        self.seed             = seed

        self.wv         = None   # gensim KeyedVectors (after fit_embeddings)
        self.classifier = None   # sklearn Pipeline (after fit_classifier)
        self._dim       = n2v_config.get("dimensions", 128)

    # ── Step 1: Embeddings ─────────────────────────────────────────────────────

    def fit_embeddings(
        self,
        G:              nx.DiGraph,
        edge_list_path: Optional[str] = None,
    ) -> None:
        """
        Generate Node2Vec random-walk embeddings using PecanPy.

        PecanPy reads an edge-list file rather than a NetworkX object.
        If ``edge_list_path`` points to an already-written edge list that
        matches G, it is used directly (saves I/O).  Otherwise the edges of G
        are written to a temporary file automatically.

        Edge-list format expected by PecanPy (tab-separated, no header):
            src_node  dst_node  [weight]
        Weights are optional; unweighted edges are written without a weight
        column and PecanPy treats them as weight 1.0.

        Parameters
        ----------
        G              : training graph (val/test edges already removed)
        edge_list_path : optional path to a pre-written edge list for G;
                         when None a temp file is created and deleted after use
        """
        mode_name = self.n2v_config.get("mode", "SparseOTF")
        if mode_name not in _PECANPY_MODES:
            raise ValueError(
                f"Unknown PecanPy mode '{mode_name}'. "
                f"Valid options: {list(_PECANPY_MODES.keys())}"
            )

        logger.info(
            f"Running PecanPy Node2Vec  "
            f"(mode={mode_name}, "
            f"dim={self.n2v_config['dimensions']}, "
            f"walks={self.n2v_config['num_walks']} × {self.n2v_config['walk_length']} steps, "
            f"p={self.n2v_config['p']}, q={self.n2v_config['q']}, "
            f"workers={self.n2v_config['workers']}) ..."
        )

        # ── Write edge list if not provided ────────────────────────────────────
        _tmp_file  = None
        _owns_file = edge_list_path is None

        if _owns_file:
            # Write to a named temp file; PecanPy needs a real path on disk
            _tmp_file = tempfile.NamedTemporaryFile(
                mode="w", suffix=".edgelist", delete=False
            )
            logger.info(f"Writing edge list to temp file '{_tmp_file.name}' ...")
            for u, v in G.edges():
                _tmp_file.write(f"{u}\t{v}\n")
            _tmp_file.flush()
            _tmp_file.close()
            edge_list_path = _tmp_file.name

        try:
            # ── Initialise PecanPy graph ───────────────────────────────────────
            g = _PECANPY_MODES[mode_name](
                p=self.n2v_config["p"],
                q=self.n2v_config["q"],
                workers=self.n2v_config["workers"],
                verbose=True,
                extend=False,          # standard Node2Vec (not Node2Vec+)
            )

            # directed=True preserves edge direction in walks
            # weighted=False since we have no edge weights
            g.read_edg(
                edge_list_path,
                weighted=False,
                directed=True,
            )

            # ── Generate walks ─────────────────────────────────────────────────
            logger.info("Generating random walks ...")
            walks = g.simulate_walks(
                num_walks=self.n2v_config["num_walks"],
                walk_length=self.n2v_config["walk_length"],
            )

            # ── Train Word2Vec (Skip-Gram) on the walks ────────────────────────
            # PecanPy returns walks as lists of strings (node IDs as str).
            # We pass them directly to gensim Word2Vec.
            logger.info("Training Word2Vec on walks ...")
            from gensim.models import Word2Vec

            w2v = Word2Vec(
                sentences=walks,
                vector_size=self.n2v_config["dimensions"],
                window=self.w2v_config["window"],
                min_count=self.w2v_config["min_count"],
                sg=self.w2v_config["sg"],
                epochs=self.w2v_config["epochs"],
                workers=self.n2v_config["workers"],
                seed=self.seed,
            )

        finally:
            # Always clean up the temp file even if training raises
            if _owns_file and _tmp_file is not None:
                try:
                    os.unlink(_tmp_file.name)
                except OSError:
                    pass

        # Keep only the lightweight KeyedVectors; discard the full Word2Vec model
        self.wv   = w2v.wv
        self._dim = self.n2v_config["dimensions"]
        logger.info(f"Embeddings ready  →  {len(self.wv):,} nodes covered.")

    # ── Step 2: Classifier ─────────────────────────────────────────────────────

    def fit_classifier(
        self,
        train_pos: List[Tuple],
        train_neg: List[Tuple],
    ) -> None:
        """
        Build Hadamard (or chosen operator) edge features and train the
        Logistic Regression pipeline (StandardScaler + LR).
        """
        logger.info("Building training feature matrix ...")
        X_train, y_train = build_dataset(
            self.wv, train_pos, train_neg,
            operator=self.feature_operator, dim=self._dim,
        )
        logger.info(
            f"Training set  →  "
            f"{y_train.sum():,} positives  |  "
            f"{(y_train == 0).sum():,} negatives  |  "
            f"feature dim: {X_train.shape[1]}"
        )

        self.classifier = Pipeline([
            ("scaler", StandardScaler()),
            ("lr",     LogisticRegression(**self.lr_config)),
        ])
        self.classifier.fit(X_train, y_train)
        logger.info("Logistic Regression fitted successfully.")

    # ── Step 3: Evaluation ─────────────────────────────────────────────────────

    def evaluate(
        self,
        pos_edges:  List[Tuple],
        neg_edges:  List[Tuple],
        split_name: str = "val",
    ) -> Dict:
        """
        Evaluate the fitted model on a labelled edge set.

        Returns a metrics dictionary (also logged at INFO level).
        """
        X, y = build_dataset(
            self.wv, pos_edges, neg_edges,
            operator=self.feature_operator, dim=self._dim,
        )
        y_pred  = self.classifier.predict(X)
        y_proba = self.classifier.predict_proba(X)[:, 1]

        metrics = {
            "split":         split_name,
            "roc_auc":       round(float(roc_auc_score(y, y_proba)),         4),
            "avg_precision": round(float(average_precision_score(y, y_proba)), 4),
            "f1":            round(float(f1_score(y, y_pred)),               4),
            "precision":     round(float(precision_score(y, y_pred)),        4),
            "recall":        round(float(recall_score(y, y_pred)),           4),
            "n_pos":         int(y.sum()),
            "n_neg":         int((y == 0).sum()),
        }

        sep = "─" * 52
        logger.info(
            f"\n{sep}\n"
            f"  [{split_name.upper()}] Evaluation\n"
            f"{sep}\n"
            f"  ROC-AUC        : {metrics['roc_auc']}\n"
            f"  Avg Precision  : {metrics['avg_precision']}\n"
            f"  F1 Score       : {metrics['f1']}\n"
            f"  Precision      : {metrics['precision']}\n"
            f"  Recall         : {metrics['recall']}\n"
            f"{sep}\n"
            + classification_report(y, y_pred, target_names=["No Edge (0)", "Edge (1)"])
        )
        return metrics

    # ── Step 4 & 5: Inference ──────────────────────────────────────────────────

    def predict(self, edges: List[Tuple]) -> np.ndarray:
        """Return binary predictions (0 / 1) for a list of node pairs."""
        X = edge_features(self.wv, edges, self.feature_operator, self._dim)
        return self.classifier.predict(X)

    def predict_proba(self, edges: List[Tuple]) -> np.ndarray:
        """Return probability scores (P(edge=1)) for a list of node pairs."""
        X = edge_features(self.wv, edges, self.feature_operator, self._dim)
        return self.classifier.predict_proba(X)[:, 1]

    # ── Persistence ────────────────────────────────────────────────────────────

    def save(self, embedding_path: str, classifier_path: str) -> None:
        """Persist the KeyedVectors and sklearn pipeline to disk."""
        os.makedirs(os.path.dirname(embedding_path), exist_ok=True)
        self.wv.save(embedding_path)
        joblib.dump(
            {"classifier": self.classifier, "dim": self._dim},
            classifier_path,
        )
        logger.info(
            f"Model saved  →  embeddings: '{embedding_path}'  |  "
            f"classifier: '{classifier_path}'"
        )

    @classmethod
    def load(
        cls,
        embedding_path:   str,
        classifier_path:  str,
        n2v_config:       Dict,
        w2v_config:       Dict,
        lr_config:        Dict,
        feature_operator: str = "hadamard",
    ) -> "Node2VecLinkPredictor":
        """Restore a previously saved predictor from disk."""
        from gensim.models import KeyedVectors

        instance = cls(n2v_config, w2v_config, lr_config, feature_operator)
        instance.wv = KeyedVectors.load(embedding_path)

        payload            = joblib.load(classifier_path)
        instance.classifier = payload["classifier"]
        instance._dim       = payload["dim"]

        logger.info("Model loaded successfully.")
        return instance
    