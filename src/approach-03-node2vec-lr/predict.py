"""
Inference entry-point — generates predictions for a test.csv file.

Expected test CSV format:
    Id,From,To
    1,3360982,4457271

Output CSV format:
    Id,Predictions
    1,0

Usage
-----
    python predict.py
    python predict.py --test test.csv --output predictions.csv
    python predict.py --proba          # also write edge probabilities
"""
import argparse
import logging
import sys

import pandas as pd
import networkx as nx

from config import (
    CLASSIFIER_TYPE,
    FEATURE_OPERATOR,
    FEATURE_OPERATORS,
    GRAPH_CONFIG,
    LR_CONFIG,
    NODE2VEC_CONFIG,
    PATHS,
    USE_GRAPH_FEATURES,
    W2V_CONFIG,
)
from graph_utils import load_graph_from_adjacency_csv
from node2vec_lr import Node2VecLinkPredictor

# ── Logging setup ──────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Run link-prediction inference on a test CSV"
    )
    p.add_argument(
        "--test",
        default=PATHS["test_edges"],
        help="Path to test CSV with columns [Id, From, To] (default: %(default)s)",
    )
    p.add_argument(
        "--output",
        default=PATHS["predictions"],
        help="Path for the output predictions CSV (default: %(default)s)",
    )
    p.add_argument(
        "--proba",
        action="store_true",
        help="If set, also include an edge-probability column in the output",
    )
    p.add_argument(
        "--threshold",
        type=float,
        default=0.9,
        help="Probability threshold for classifying an edge as 1 (default: 0.9)",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()

    # ── 1. Load saved model ────────────────────────────────────────────────────
    logger.info("Loading saved model artefacts ...")
    predictor = Node2VecLinkPredictor.load(
        embedding_path=PATHS["embeddings"],
        classifier_path=PATHS["classifier"],
        n2v_config=NODE2VEC_CONFIG,
        w2v_config=W2V_CONFIG,
        lr_config=LR_CONFIG,
        feature_operator=FEATURE_OPERATOR,
        classifier_type=CLASSIFIER_TYPE,
        feature_operators=FEATURE_OPERATORS,
        use_graph_features=USE_GRAPH_FEATURES,
    )

    # Load training graph if graph-structural features are enabled
    # IMPORTANT: Use the same graph that was used during training (G_train)
    # to avoid distribution mismatch between train-time and inference-time features.
    G_train = None
    pagerank = None
    if USE_GRAPH_FEATURES:
        logger.info("Loading training graph for structural features ...")
        G_train = load_graph_from_adjacency_csv(
            PATHS["train_graph"], directed=GRAPH_CONFIG["directed"]
        )
        logger.info("Computing PageRank (cached for all predictions) ...")
        pagerank = nx.pagerank(G_train, alpha=0.85, max_iter=50, tol=1e-4)

    # ── 2. Load test edges ─────────────────────────────────────────────────────
    logger.info(f"Loading test edges from '{args.test}' ...")
    df_test = pd.read_csv(args.test)

    required_cols = {"Id", "From", "To"}
    missing = required_cols - set(df_test.columns)
    if missing:
        raise ValueError(
            f"Test CSV is missing required columns: {missing}. "
            f"Found: {list(df_test.columns)}"
        )

    edges = list(zip(df_test["From"].tolist(), df_test["To"].tolist()))
    logger.info(f"Predicting {len(edges):,} candidate edges ...")

    # ── 3. Run inference ───────────────────────────────────────────────────────
    thrs = [0.9, 0.95, 0.97, 0.98, 0.99, 0.995, 0.999]
    probas = predictor.predict_proba(edges, G_train=G_train, pagerank=pagerank)
    for t in thrs:
        logger.info(f"Applying threshold {t:.3f} to edge probabilities ...")
        preds = (probas >= t).astype(int)
        n_edge    = int(preds.sum())
        n_no_edge = int((preds == 0).sum())
        logger.info(
            f"Threshold {t:.3f} → edge=1: {n_edge:,}  |  no-edge=0: {n_no_edge:,}"
        )
        out = pd.DataFrame({"Id": df_test["Id"], "Predictions": preds.astype(int)})
        out.to_csv(f"data/processed/node2vec_{CLASSIFIER_TYPE}_predictions_{t:.3f}.csv", index=False)

    # ── 4. Build and save output ───────────────────────────────────────────────
    out = pd.DataFrame({"Id": df_test["Id"], "Predictions": preds.astype(int)})

    if args.proba:
        out["Probability"] = probas.round(4)
        logger.info("Edge probabilities included in output (--proba flag set).")

    out.to_csv(args.output, index=False)

    n_edge    = int(preds.sum())
    n_no_edge = int((preds == 0).sum())
    logger.info(
        f"Predictions saved → '{args.output}'  "
        f"( edge=1: {n_edge:,}  |  no-edge=0: {n_no_edge:,} )"
    )
    logger.info("Inference complete.")


if __name__ == "__main__":
    main()
