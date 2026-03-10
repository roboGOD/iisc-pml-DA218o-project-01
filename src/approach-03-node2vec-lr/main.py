"""
End-to-end entry-point: train the Node2Vec + Logistic Regression model,
then immediately run inference on the test set.

Usage
-----
    python main.py
    python main.py --graph data/raw/train.csv --seed 123
    python main.py --resume
    python main.py --output data/processed/my_predictions.csv --proba
"""
import argparse
import json
import logging
import os
import sys
import time

import networkx as nx
import pandas as pd

from checkpoint import CheckpointManager
from config import (
    CLASSIFIER_TYPE,
    FEATURE_OPERATOR,
    FEATURE_OPERATORS,
    GRAPH_CONFIG,
    LR_CONFIG,
    NODE2VEC_CONFIG,
    PATHS,
    RANDOM_SEED,
    SPLIT_CONFIG,
    USE_GRAPH_FEATURES,
    W2V_CONFIG,
)
from graph_utils import load_graph_from_adjacency_csv, train_val_test_split
from node2vec_lr import Node2VecLinkPredictor

# ── Logging setup ──────────────────────────────────────────────────────────────
os.makedirs("artifacts/node2vec", exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler("artifacts/node2vec/training.log", mode="w"),
    ],
)
logger = logging.getLogger(__name__)

# ── Checkpoint setup ───────────────────────────────────────────────────────────
_STAGES = ["graph_loaded", "split_done", "embeddings_done",
           "classifier_done", "evaluation_done", "complete"]

_REGISTRY = {
    "G":            "graph",
    "G_train":      "graph",
    "train_pos":    "edges",
    "val_pos":      "edges",
    "test_pos":     "edges",
    "train_neg":    "edges",
    "val_neg":      "edges",
    "test_neg":     "edges",
    "wv":           "keyed_vectors",
    "classifier":   "sklearn",
    "val_metrics":  "json",
    "test_metrics": "json",
}

ckpt = CheckpointManager(
    stages=_STAGES,
    checkpoint_dir=os.path.join("artifacts", "node2vec", "checkpoints"),
    registry=_REGISTRY,
)


# ── CLI ────────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Train Node2Vec + LR, then predict on the test set"
    )
    p.add_argument(
        "--graph",
        default=PATHS["train_graph"],
        help="Path to the adjacency-list CSV (default: %(default)s)",
    )
    p.add_argument(
        "--seed",
        type=int,
        default=RANDOM_SEED,
        help="Global random seed (default: %(default)s)",
    )
    p.add_argument(
        "--resume",
        action="store_true",
        help="Resume training from the last checkpoint (if one exists)",
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


# ── Training phase ─────────────────────────────────────────────────────────────

def run_training(args) -> Node2VecLinkPredictor:
    logger.info("=" * 60)
    logger.info("  PHASE 1 — Training")
    logger.info("=" * 60)
    logger.info(f"Graph file : {args.graph}")
    logger.info(f"Random seed: {args.seed}")

    ckpt_stage, ckpt_data = (None, {})
    if args.resume:
        ckpt_stage, ckpt_data = ckpt.load()
        if ckpt_stage is None:
            logger.info("No checkpoint found — starting from scratch.")

    # ── 1. Load graph ──────────────────────────────────────────────────────────
    if ckpt.past_stage(ckpt_stage, "graph_loaded"):
        logger.info("[checkpoint] Skipping graph loading (already done).")
        G = ckpt_data["G"]
    else:
        G = load_graph_from_adjacency_csv(
            args.graph, directed=GRAPH_CONFIG["directed"]
        )
        ckpt.save("graph_loaded", G=G)

    # ── 2. Train / val / test split ────────────────────────────────────────────
    if ckpt.past_stage(ckpt_stage, "split_done"):
        logger.info("[checkpoint] Skipping split (already done).")
        G_train   = ckpt_data["G_train"]
        train_pos = ckpt_data["train_pos"]
        val_pos   = ckpt_data["val_pos"]
        test_pos  = ckpt_data["test_pos"]
        train_neg = ckpt_data["train_neg"]
        val_neg   = ckpt_data["val_neg"]
        test_neg  = ckpt_data["test_neg"]
    else:
        (
            G_train,
            train_pos, val_pos,  test_pos,
            train_neg, val_neg,  test_neg,
        ) = train_val_test_split(
            G,
            val_ratio=SPLIT_CONFIG["val_ratio"],
            test_ratio=SPLIT_CONFIG["test_ratio"],
            neg_ratio=SPLIT_CONFIG["neg_ratio"],
            seed=args.seed,
        )
        ckpt.save(
            "split_done",
            G=G, G_train=G_train,
            train_pos=train_pos, val_pos=val_pos, test_pos=test_pos,
            train_neg=train_neg, val_neg=val_neg, test_neg=test_neg,
        )

    del G
    ckpt_data.pop("G", None)

    # ── 3. Build and train the model ───────────────────────────────────────────
    predictor = Node2VecLinkPredictor(
        n2v_config=NODE2VEC_CONFIG,
        w2v_config=W2V_CONFIG,
        lr_config=LR_CONFIG,
        feature_operator=FEATURE_OPERATOR,
        classifier_type=CLASSIFIER_TYPE,
        seed=args.seed,
        feature_operators=FEATURE_OPERATORS,
        use_graph_features=USE_GRAPH_FEATURES,
    )

    if ckpt.past_stage(ckpt_stage, "embeddings_done"):
        logger.info("[checkpoint] Skipping embedding training (already done).")
        predictor.wv   = ckpt_data["wv"]
        predictor._dim = NODE2VEC_CONFIG["dimensions"]
    else:
        predictor.fit_embeddings(G_train)
        ckpt.save("embeddings_done", wv=predictor.wv)

    # ── Compute PageRank once (used by graph features) ──────────────────────
    pagerank = None
    if USE_GRAPH_FEATURES:
        logger.info("Computing PageRank on G_train (one-time) ...")
        pagerank = nx.pagerank(G_train, alpha=0.85, max_iter=50, tol=1e-4)
        logger.info(f"PageRank computed for {len(pagerank):,} nodes.")

    if ckpt.past_stage(ckpt_stage, "classifier_done"):
        logger.info("[checkpoint] Skipping classifier training (already done).")
        predictor.classifier = ckpt_data["classifier"]
    else:
        predictor.fit_classifier(
            train_pos, train_neg,
            G_train=G_train,
            pagerank=pagerank,
            val_pos=val_pos,
            val_neg=val_neg,
        )
        ckpt.save("classifier_done", classifier=predictor.classifier)

    # ── 4. Evaluate ────────────────────────────────────────────────────────────
    if ckpt.past_stage(ckpt_stage, "evaluation_done"):
        logger.info("[checkpoint] Skipping evaluation (already done).")
        val_metrics  = ckpt_data["val_metrics"]
        test_metrics = ckpt_data["test_metrics"]
    else:
        val_metrics  = predictor.evaluate(val_pos,  val_neg,  split_name="val",  G_train=G_train, pagerank=pagerank)
        test_metrics = predictor.evaluate(test_pos, test_neg, split_name="test", G_train=G_train, pagerank=pagerank)
        ckpt.save("evaluation_done", val_metrics=val_metrics, test_metrics=test_metrics)

    # ── 5. Persist model and metrics ───────────────────────────────────────────
    if not ckpt.past_stage(ckpt_stage, "complete"):
        predictor.save(PATHS["embeddings"], PATHS["classifier"])

        all_metrics = {"val": val_metrics, "test": test_metrics}
        with open(PATHS["metrics"], "w") as fh:
            json.dump(all_metrics, fh, indent=2)
        logger.info(f"Metrics saved → '{PATHS['metrics']}'")

        ckpt.save("complete")

    return predictor


# ── Prediction phase ───────────────────────────────────────────────────────────

def run_prediction(predictor: Node2VecLinkPredictor, args) -> None:
    logger.info("")
    logger.info("=" * 60)
    logger.info("  PHASE 2 — Inference")
    logger.info("=" * 60)

    # Load G_train for structural features if needed
    G_train = None
    pagerank = None
    if USE_GRAPH_FEATURES:
        logger.info("Loading training graph for structural features ...")
        G_train = load_graph_from_adjacency_csv(
            args.graph, directed=GRAPH_CONFIG["directed"]
        )
        logger.info("Computing PageRank for inference ...")
        pagerank = nx.pagerank(G_train, alpha=0.85, max_iter=50, tol=1e-4)

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

    logger.info(f"Using probability threshold: {args.threshold}")
    probas = predictor.predict_proba(edges, G_train=G_train, pagerank=pagerank)
    preds = (probas >= args.threshold).astype(int)

    out = pd.DataFrame({"Id": df_test["Id"], "Predictions": preds.astype(int)})
    if args.proba:
        out["Probability"] = probas.round(4)
        logger.info("Edge probabilities included in output (--proba flag set).")

    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    out.to_csv(args.output, index=False)

    n_edge    = int(preds.sum())
    n_no_edge = int((preds == 0).sum())
    logger.info(
        f"Predictions saved → '{args.output}'  "
        f"( edge=1: {n_edge:,}  |  no-edge=0: {n_no_edge:,} )"
    )
    logger.info("Inference complete.")


# ── Entry point ────────────────────────────────────────────────────────────────

def main() -> None:
    args = parse_args()
    t_start = time.time()

    predictor = run_training(args)
    run_prediction(predictor, args)

    elapsed = time.time() - t_start
    logger.info(f"\nTotal elapsed time: {elapsed / 60:.1f} min  ({elapsed:.0f}s)")
    logger.info("Done.")


if __name__ == "__main__":
    main()
