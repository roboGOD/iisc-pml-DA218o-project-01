"""
Core model: Node2Vec embeddings (via PecanPy) → binary link predictor.

Supported classifiers (controlled by ``classifier_type``):
  • "sgd"     — SGDClassifier(loss="log_loss") trained incrementally via
                partial_fit.  Lowest memory; suitable for 10M+ edge datasets.
  • "xgboost" — XGBClassifier with histogram tree method.  Better accuracy;
                needs ~2× the feature matrix in RAM.

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
2. predictor.fit_classifier(pos, neg)        — classifier on Hadamard edge features
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
from sklearn.linear_model import SGDClassifier
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

from features import build_dataset, cleanup_memmap, edge_features, multi_operator_features, graph_structural_features, GRAPH_FEATURE_DIM, get_embedding, _embedding_similarities, EMBEDDING_SIM_DIM

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


_VALID_CLASSIFIERS = ("sgd", "xgboost")


class Node2VecLinkPredictor:
    """
    Directed link-prediction via PecanPy Node2Vec embeddings + a configurable
    binary classifier (SGDClassifier or XGBClassifier).

    All heavy lifting is split into explicit steps so each stage can be
    inspected, swapped, or rerun independently.
    """

    def __init__(
        self,
        n2v_config:        Dict,
        w2v_config:        Dict,
        lr_config:         Dict,
        feature_operator:  str = "hadamard",
        classifier_type:   str = "sgd",
        seed:              int = 42,
        feature_operators: Optional[List[str]] = None,
        use_graph_features: bool = False,
    ):
        if classifier_type not in _VALID_CLASSIFIERS:
            raise ValueError(
                f"Unknown classifier_type '{classifier_type}'. "
                f"Valid options: {_VALID_CLASSIFIERS}"
            )
        self.n2v_config         = n2v_config
        self.w2v_config         = w2v_config
        self.lr_config          = lr_config
        self.feature_operator   = feature_operator
        self.feature_operators  = feature_operators
        self.use_graph_features = use_graph_features
        self.classifier_type    = classifier_type
        self.seed               = seed

        self.wv         = None   # gensim KeyedVectors (after fit_embeddings)
        self.classifier = None   # sklearn Pipeline or XGBClassifier
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
        _tmp_file   = None
        _walk_path  = None
        _owns_file  = edge_list_path is None

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

            # ── Write walks to disk to free memory before Word2Vec training ────
            _walk_fd, _walk_path = tempfile.mkstemp(suffix=".walks")
            os.close(_walk_fd)
            logger.info(
                f"Writing {len(walks):,} walks to '{_walk_path}' "
                f"to free memory before Word2Vec training ..."
            )
            with open(_walk_path, "w") as wf:
                for walk in walks:
                    wf.write(" ".join(str(w) for w in walk) + "\n")
            del walks

            # ── Train Word2Vec (Skip-Gram) on the walks ────────────────────────
            logger.info("Training Word2Vec on walks (corpus_file mode) ...")
            from gensim.models import Word2Vec

            w2v = Word2Vec(
                corpus_file=_walk_path,
                vector_size=self.n2v_config["dimensions"],
                window=self.w2v_config["window"],
                min_count=self.w2v_config["min_count"],
                sg=self.w2v_config["sg"],
                epochs=self.w2v_config["epochs"],
                workers=self.n2v_config["workers"],
                seed=self.seed,
            )

        finally:
            # Always clean up temp files even if training raises
            if _owns_file and _tmp_file is not None:
                try:
                    os.unlink(_tmp_file.name)
                except OSError:
                    pass
            if _walk_path is not None:
                try:
                    os.unlink(_walk_path)
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
        G_train:   Optional[nx.DiGraph] = None,
        pagerank:  Optional[dict] = None,
        val_pos:   Optional[List[Tuple]] = None,
        val_neg:   Optional[List[Tuple]] = None,
    ) -> None:
        """
        Build edge features and train the selected classifier.

        Parameters
        ----------
        G_train  : if provided and ``self.use_graph_features`` is True,
                   graph-structural features are appended to embedding features.
        pagerank : precomputed PageRank dict; computed on-the-fly if None.
        val_pos  : validation positive edges (for proper early stopping).
        val_neg  : validation negative edges (for proper early stopping).
        """
        logger.info("Building training feature matrix ...")
        X_train, y_train = build_dataset(
            self.wv, train_pos, train_neg,
            operator=self.feature_operator, dim=self._dim,
            G=G_train if self.use_graph_features else None,
            operators=self.feature_operators,
            pagerank=pagerank,
        )
        logger.info(
            f"Training set  →  "
            f"{y_train.sum():,} positives  |  "
            f"{(y_train == 0).sum():,} negatives  |  "
            f"feature dim: {X_train.shape[1]}"
        )

        if self.classifier_type == "sgd":
            self._fit_sgd(X_train, y_train)
        else:
            # Build proper val set for early stopping if provided
            X_val, y_val = None, None
            if val_pos is not None and val_neg is not None:
                logger.info("Building validation feature matrix for early stopping ...")
                X_val, y_val = build_dataset(
                    self.wv, val_pos, val_neg,
                    operator=self.feature_operator, dim=self._dim,
                    G=G_train if self.use_graph_features else None,
                    operators=self.feature_operators,
                    pagerank=pagerank,
                )
            self._fit_xgboost(X_train, y_train, X_val=X_val, y_val=y_val)
            if X_val is not None:
                cleanup_memmap(X_val)
        cleanup_memmap(X_train)

    # ── SGD path ───────────────────────────────────────────────────────────────

    def _fit_sgd(self, X_train: np.ndarray, y_train: np.ndarray) -> None:
        """Incremental StandardScaler + SGDClassifier(log_loss) via partial_fit."""
        n = X_train.shape[0]

        chunk_size = self.lr_config.get("chunk_size", 500_000)
        n_epochs   = self.lr_config.get("n_epochs", 10)
        sgd_params = {
            k: v for k, v in self.lr_config.items()
            if k not in ("chunk_size", "n_epochs")
        }

        # 1. Incremental StandardScaler (avoids float64 full copy)
        logger.info("Fitting StandardScaler (incremental) ...")
        scaler = StandardScaler()
        for start in range(0, n, chunk_size):
            scaler.partial_fit(X_train[start : start + chunk_size])

        # Transform in-place chunk-by-chunk, keeping float32
        for start in range(0, n, chunk_size):
            end = min(start + chunk_size, n)
            X_train[start:end] = scaler.transform(
                X_train[start:end]
            ).astype(np.float32)

        # 2. SGDClassifier with partial_fit (mini-batch LR)
        sgd = SGDClassifier(loss="log_loss", **sgd_params)
        classes = np.array([0, 1])
        rng = np.random.RandomState(self.seed)

        for epoch in range(1, n_epochs + 1):
            perm = rng.permutation(n)
            for start in range(0, n, chunk_size):
                idx = perm[start : start + chunk_size]
                sgd.partial_fit(X_train[idx], y_train[idx], classes=classes)
            logger.info(f"  SGD epoch {epoch}/{n_epochs} done.")

        self.classifier = Pipeline([("scaler", scaler), ("lr", sgd)])
        logger.info("SGDClassifier (logistic) fitted successfully.")

    # ── XGBoost path ───────────────────────────────────────────────────────────

    def _fit_xgboost(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val:   Optional[np.ndarray] = None,
        y_val:   Optional[np.ndarray] = None,
    ) -> None:
        """XGBClassifier with histogram tree method and early stopping."""
        from xgboost import XGBClassifier

        xgb_params = {k: v for k, v in self.lr_config.items()}
        early_stopping = xgb_params.pop("early_stopping_rounds", None)

        xgb = XGBClassifier(
            objective="binary:logistic",
            eval_metric="logloss",
            use_label_encoder=False,
            early_stopping_rounds=early_stopping,
            **xgb_params,
        )
        logger.info("Training XGBClassifier ...")

        if early_stopping:
            if X_val is not None and y_val is not None:
                # Use proper validation set for early stopping
                logger.info(
                    f"Using proper validation set for early stopping "
                    f"({X_val.shape[0]:,} samples) ..."
                )
                xgb.fit(
                    X_train, y_train,
                    eval_set=[(X_val, y_val)],
                    verbose=50,
                )
            else:
                # Fall back to random sample from training data
                n = X_train.shape[0]
                rng = np.random.RandomState(self.seed)
                n_eval = min(500_000, max(1, int(n * 0.05)))
                eval_idx = np.sort(rng.choice(n, size=n_eval, replace=False))
                X_eval = np.array(X_train[eval_idx], dtype=np.float32)
                y_eval = y_train[eval_idx].copy()
                xgb.fit(
                    X_train, y_train,
                    eval_set=[(X_eval, y_eval)],
                    verbose=50,
                )
                del X_eval, y_eval
            logger.info(f"Best iteration: {xgb.best_iteration}")
        else:
            xgb.fit(X_train, y_train)

        self.classifier = xgb
        logger.info("XGBClassifier fitted successfully.")

    # ── Step 3: Evaluation ─────────────────────────────────────────────────────

    def evaluate(
        self,
        pos_edges:  List[Tuple],
        neg_edges:  List[Tuple],
        split_name: str = "val",
        G_train:    Optional[nx.DiGraph] = None,
        pagerank:   Optional[dict] = None,
    ) -> Dict:
        """
        Evaluate the fitted model on a labelled edge set.

        Returns a metrics dictionary (also logged at INFO level).
        """
        X, y = build_dataset(
            self.wv, pos_edges, neg_edges,
            operator=self.feature_operator, dim=self._dim,
            G=G_train if self.use_graph_features else None,
            operators=self.feature_operators,
            pagerank=pagerank,
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
        cleanup_memmap(X)
        return metrics

    # ── Step 4 & 5: Inference ──────────────────────────────────────────────────

    def _build_inference_features(
        self,
        edges:   List[Tuple],
        G_train: Optional[nx.DiGraph] = None,
        pagerank: Optional[dict] = None,
    ) -> np.ndarray:
        """Build the full feature matrix for inference edges."""
        if self.feature_operators and len(self.feature_operators) > 1:
            X_emb = multi_operator_features(
                self.wv, edges, self.feature_operators, self._dim,
            )
        else:
            X_emb = edge_features(
                self.wv, edges, self.feature_operator, self._dim,
            )

        # Embedding similarity features (cosine, dot, L2 distance)
        n = len(edges)
        X_sim = np.empty((n, EMBEDDING_SIM_DIM), dtype=np.float32)
        for i, (u, v) in enumerate(edges):
            u_emb = get_embedding(self.wv, u, self._dim)
            v_emb = get_embedding(self.wv, v, self._dim)
            X_sim[i] = _embedding_similarities(u_emb, v_emb)

        parts = [X_emb, X_sim]

        if self.use_graph_features and G_train is not None:
            X_graph = graph_structural_features(G_train, edges, pagerank=pagerank)
            parts.append(X_graph)

        return np.hstack(parts)

    def predict(
        self,
        edges:   List[Tuple],
        G_train: Optional[nx.DiGraph] = None,
        pagerank: Optional[dict] = None,
    ) -> np.ndarray:
        """Return binary predictions (0 / 1) for a list of node pairs."""
        X = self._build_inference_features(edges, G_train, pagerank=pagerank)
        preds = self.classifier.predict(X)
        return preds.astype(int)

    def predict_proba(
        self,
        edges:   List[Tuple],
        G_train: Optional[nx.DiGraph] = None,
        pagerank: Optional[dict] = None,
    ) -> np.ndarray:
        """Return probability scores (P(edge=1)) for a list of node pairs."""
        X = self._build_inference_features(edges, G_train, pagerank=pagerank)
        return self.classifier.predict_proba(X)[:, 1]

    # ── Persistence ────────────────────────────────────────────────────────────

    def save(self, embedding_path: str, classifier_path: str) -> None:
        """Persist the KeyedVectors and classifier to disk."""
        os.makedirs(os.path.dirname(embedding_path), exist_ok=True)
        self.wv.save(embedding_path)
        joblib.dump(
            {
                "classifier": self.classifier,
                "classifier_type": self.classifier_type,
                "dim": self._dim,
            },
            classifier_path,
        )
        logger.info(
            f"Model saved  →  embeddings: '{embedding_path}'  |  "
            f"classifier ({self.classifier_type}): '{classifier_path}'"
        )

    @classmethod
    def load(
        cls,
        embedding_path:     str,
        classifier_path:    str,
        n2v_config:         Dict,
        w2v_config:         Dict,
        lr_config:          Dict,
        feature_operator:   str = "hadamard",
        classifier_type:    str = "sgd",
        feature_operators:  Optional[List[str]] = None,
        use_graph_features: bool = False,
    ) -> "Node2VecLinkPredictor":
        """Restore a previously saved predictor from disk."""
        from gensim.models import KeyedVectors

        payload = joblib.load(classifier_path)
        # Prefer the type stored at save-time; fall back to caller's arg
        saved_type = payload.get("classifier_type", classifier_type)

        instance = cls(
            n2v_config, w2v_config, lr_config,
            feature_operator, classifier_type=saved_type,
            feature_operators=feature_operators,
            use_graph_features=use_graph_features,
        )
        instance.wv         = KeyedVectors.load(embedding_path)
        instance.classifier = payload["classifier"]
        instance._dim       = payload["dim"]

        logger.info(f"Model loaded successfully (classifier={saved_type}).")
        return instance
    