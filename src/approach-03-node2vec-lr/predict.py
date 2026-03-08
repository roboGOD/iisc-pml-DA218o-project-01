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

from config import (
    FEATURE_OPERATOR,
    LR_CONFIG,
    NODE2VEC_CONFIG,
    PATHS,
    W2V_CONFIG,
)
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
    )

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
    preds = predictor.predict(edges)

    # ── 4. Build and save output ───────────────────────────────────────────────
    out = pd.DataFrame({"Id": df_test["Id"], "Predictions": preds.astype(int)})

    if args.proba:
        probas = predictor.predict_proba(edges)
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
