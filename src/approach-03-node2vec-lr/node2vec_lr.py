"""
Core model: Node2Vec embeddings → Logistic Regression link predictor.

Training workflow
-----------------
1. predictor.fit_embeddings(G_train)     — random-walk Node2Vec
2. predictor.fit_classifier(pos, neg)    — LR on Hadamard edge features
3. predictor.evaluate(pos, neg, "val")   — print + return metrics
4. predictor.predict(edges)              — binary 0 / 1 labels
5. predictor.predict_proba(edges)        — probability scores
6. predictor.save(...)  /  .load(...)    — model persistence
"""
import logging
import os
from typing import Dict, List, Tuple

import joblib
import networkx as nx
import numpy as np
from gensim.models import Word2Vec
from node2vec import Node2Vec
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


class Node2VecLinkPredictor:
    """
    Directed link-prediction via Node2Vec embeddings + Logistic Regression.

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
        self._dim       = n2v_config.get("dimensions", 64)

    # ── Step 1: Embeddings ─────────────────────────────────────────────────────

    def fit_embeddings(self, G: nx.DiGraph) -> None:
        """
        Generate Node2Vec random-walk embeddings on the training graph.

        Node IDs are converted to strings internally by the node2vec library;
        we store only the KeyedVectors for a smaller memory footprint.
        """
        logger.info(
            "Running Node2Vec  "
            f"(dim={self.n2v_config['dimensions']}, "
            f"walks={self.n2v_config['num_walks']} × {self.n2v_config['walk_length']} steps, "
            f"p={self.n2v_config['p']}, q={self.n2v_config['q']}) ..."
        )

        n2v = Node2Vec(
            G,
            dimensions=self.n2v_config["dimensions"],
            walk_length=self.n2v_config["walk_length"],
            num_walks=self.n2v_config["num_walks"],
            p=self.n2v_config["p"],
            q=self.n2v_config["q"],
            workers=self.n2v_config["workers"],
            seed=self.seed,
            quiet=False,
        )

        w2v_model = n2v.fit(
            window=self.w2v_config["window"],
            min_count=self.w2v_config["min_count"],
            sg=self.w2v_config["sg"],
            epochs=self.w2v_config["epochs"],
            batch_words=self.w2v_config["batch_words"],
        )

        # Keep only the lightweight KeyedVectors; discard the full Word2Vec model
        self.wv   = w2v_model.wv
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
