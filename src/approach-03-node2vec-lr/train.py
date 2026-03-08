"""
Training entry-point for the Node2Vec + Logistic Regression link predictor.

Usage
-----
    python train.py --graph train_graph.csv
    python train.py --graph train_graph.csv --seed 123
"""
import argparse
import json
import logging
import os
import sys
import time

from config import (
    FEATURE_OPERATOR,
    GRAPH_CONFIG,
    LR_CONFIG,
    NODE2VEC_CONFIG,
    PATHS,
    RANDOM_SEED,
    SPLIT_CONFIG,
    W2V_CONFIG,
)
from graph_utils import load_graph_from_adjacency_csv, train_val_test_split
from node2vec_lr import Node2VecLinkPredictor

# ── Logging setup ──────────────────────────────────────────────────────────────
os.makedirs("artifacts", exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler("artifacts/training.log", mode="w"),
    ],
)
logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Train Node2Vec + Logistic Regression for link prediction"
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
    return p.parse_args()


def main() -> None:
    args = parse_args()
    t_start = time.time()

    logger.info("=" * 60)
    logger.info("  Node2Vec + Logistic Regression — Link Prediction")
    logger.info("=" * 60)
    logger.info(f"Graph file : {args.graph}")
    logger.info(f"Random seed: {args.seed}")

    # ── 1. Load graph ──────────────────────────────────────────────────────────
    G = load_graph_from_adjacency_csv(
        args.graph, directed=GRAPH_CONFIG["directed"]
    )

    # ── 2. Train / val / test split ────────────────────────────────────────────
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

    # ── 3. Build and train the model ───────────────────────────────────────────
    predictor = Node2VecLinkPredictor(
        n2v_config=NODE2VEC_CONFIG,
        w2v_config=W2V_CONFIG,
        lr_config=LR_CONFIG,
        feature_operator=FEATURE_OPERATOR,
        seed=args.seed,
    )

    predictor.fit_embeddings(G_train)
    predictor.fit_classifier(train_pos, train_neg)

    # ── 4. Evaluate on validation and test sets ────────────────────────────────
    val_metrics  = predictor.evaluate(val_pos,  val_neg,  split_name="val")
    test_metrics = predictor.evaluate(test_pos, test_neg, split_name="test")

    # ── 5. Persist model and metrics ───────────────────────────────────────────
    predictor.save(PATHS["embeddings"], PATHS["classifier"])

    all_metrics = {"val": val_metrics, "test": test_metrics}
    with open(PATHS["metrics"], "w") as fh:
        json.dump(all_metrics, fh, indent=2)
    logger.info(f"Metrics saved → '{PATHS['metrics']}'")

    elapsed = time.time() - t_start
    logger.info(f"\nTotal training time: {elapsed / 60:.1f} min  ({elapsed:.0f}s)")
    logger.info("Training complete.")


if __name__ == "__main__":
    main()
