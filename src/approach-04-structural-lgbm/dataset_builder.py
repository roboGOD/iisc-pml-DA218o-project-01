"""
Dataset builder: parses the raw graph, splits edges, samples negatives,
extracts structural features, and saves train/val/test tables as parquet.

Run standalone:
    python dataset_builder.py

Artifacts created
-----------------
  data/processed/approach04/train_features.parquet
  data/processed/approach04/val_features.parquet
  data/processed/approach04/test_features.parquet   (Kaggle test set)
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time

import numpy as np
import pandas as pd

from config import (
    FEATURE_CONFIG,
    NEG_SAMPLING,
    PATHS,
    RANDOM_SEED,
    SPLIT_CONFIG,
)
from graph_store import GraphStore
from negative_sampling import sample_mixed_negatives
from structural_features import FEATURE_NAMES, build_dataframe

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# Edge split
# ─────────────────────────────────────────────────────────────────────────────

def train_val_split(
    src: np.ndarray,
    dst: np.ndarray,
    val_ratio: float,
    seed: int = RANDOM_SEED,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Randomly split all directed edges into train and validation sets.

    Returns (train_src, train_dst, val_src, val_dst).
    """
    rng = np.random.default_rng(seed)
    N = len(src)
    perm = rng.permutation(N)

    n_val = int(N * val_ratio)
    val_idx   = perm[:n_val]
    train_idx = perm[n_val:]

    return (
        src[train_idx], dst[train_idx],
        src[val_idx],   dst[val_idx],
    )


# ─────────────────────────────────────────────────────────────────────────────
# Kaggle test reader
# ─────────────────────────────────────────────────────────────────────────────

def read_test_edges(path: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Read test.csv (Id, From, To) — handles BOM.

    Returns (ids, src_arr, dst_arr), all as int32/int64.
    """
    df = pd.read_csv(path, encoding="utf-8-sig")  # utf-8-sig strips BOM
    df.columns = [c.strip() for c in df.columns]
    ids = df["Id"].to_numpy(dtype=np.int64)
    src = df["From"].to_numpy(dtype=np.int32)
    dst = df["To"].to_numpy(dtype=np.int32)
    return ids, src, dst


# ─────────────────────────────────────────────────────────────────────────────
# Main build
# ─────────────────────────────────────────────────────────────────────────────

def build_datasets(
    train_graph_path: str = PATHS["train_graph"],
    test_edges_path: str = PATHS["test_edges"],
    out_dir_train: str = PATHS["train_feats"],
    out_dir_val: str = PATHS["val_feats"],
    out_dir_test: str = PATHS["test_feats"],
    val_ratio: float = SPLIT_CONFIG["val_ratio"],
    max_train_pos: int | None = NEG_SAMPLING["max_train_pos"],
    max_val_pos: int | None = NEG_SAMPLING["max_val_pos"],
    neg_ratio: float = NEG_SAMPLING["neg_ratio"],
    train_hard_frac: float = NEG_SAMPLING["train_hard_frac"],
    val_hard_frac: float = NEG_SAMPLING["val_hard_frac"],
    hub_degree_cap: int = NEG_SAMPLING["hub_degree_cap"],
    max_intermediaries: int = FEATURE_CONFIG["max_intermediaries"],
    seed: int = RANDOM_SEED,
) -> None:
    t_total = time.time()

    # ── 1. Load full graph ────────────────────────────────────────────
    logger.info("=" * 60)
    logger.info("Step 1/6 — Loading training graph")
    gs_full = GraphStore.from_adjacency_csv(train_graph_path)
    full_src, full_dst = gs_full.edge_list()
    logger.info("Full graph: %s", gs_full)

    # ── 2. Train / val split ──────────────────────────────────────────
    logger.info("Step 2/6 — Splitting edges (val_ratio=%.2f)", val_ratio)
    tr_src, tr_dst, val_src, val_dst = train_val_split(
        full_src, full_dst, val_ratio, seed
    )
    logger.info(
        "  train positives: %d   val positives: %d",
        len(tr_src), len(val_src),
    )

    # Build training graph with val edges removed (avoid feature leakage)
    logger.info("Step 3/6 — Building training sub-graph (val edges removed)")
    gs_train = gs_full.subgraph_without_edges(val_src, val_dst)

    # ── 3. Subsample positives ────────────────────────────────────────
    rng = np.random.default_rng(seed)

    def _subsample(
        s: np.ndarray, d: np.ndarray, max_n: int | None
    ) -> tuple[np.ndarray, np.ndarray]:
        if max_n is not None and len(s) > max_n:
            idx = rng.choice(len(s), max_n, replace=False)
            return s[idx], d[idx]
        return s, d

    tr_src_s, tr_dst_s = _subsample(tr_src, tr_dst, max_train_pos)
    val_src_s, val_dst_s = _subsample(val_src, val_dst, max_val_pos)
    logger.info(
        "  subsampled train=%d, val=%d",
        len(tr_src_s), len(val_src_s),
    )

    # ── 4. Sample negatives ───────────────────────────────────────────
    logger.info("Step 4/6 — Sampling negatives")
    train_pos_pairs = np.stack([tr_src_s, tr_dst_s], axis=1).astype(np.int32)
    val_pos_pairs   = np.stack([val_src_s, val_dst_s], axis=1).astype(np.int32)

    n_train_neg = int(len(tr_src_s) * neg_ratio)
    n_val_neg   = int(len(val_src_s) * neg_ratio)

    logger.info("  sampling %d train negatives (hard_frac=%.2f)…", n_train_neg, train_hard_frac)
    train_neg_pairs = sample_mixed_negatives(
        gs_train, n_train_neg,
        hard_frac=train_hard_frac,
        seed=seed,
        hub_degree_cap=hub_degree_cap,
        exclude_pairs=train_pos_pairs,
    )

    logger.info("  sampling %d val negatives (hard_frac=%.2f)…", n_val_neg, val_hard_frac)
    val_neg_pairs = sample_mixed_negatives(
        gs_train, n_val_neg,
        hard_frac=val_hard_frac,
        seed=seed + 10,
        hub_degree_cap=hub_degree_cap,
        exclude_pairs=val_pos_pairs,
    )

    # ── 5. Combine and featurise ──────────────────────────────────────
    logger.info("Step 5/6 — Extracting structural features")

    def _make_table(pos: np.ndarray, neg: np.ndarray) -> pd.DataFrame:
        pairs  = np.concatenate([pos, neg], axis=0)
        labels = np.concatenate([
            np.ones(len(pos), dtype=np.int8),
            np.zeros(len(neg), dtype=np.int8),
        ])
        # Shuffle
        perm = rng.permutation(len(pairs))
        return build_dataframe(
            gs_train, pairs[perm], labels[perm],
            max_intermediaries=max_intermediaries,
        )

    logger.info("  Building TRAIN table (%d pairs)…", len(train_pos_pairs) + len(train_neg_pairs))
    df_train = _make_table(train_pos_pairs, train_neg_pairs)

    logger.info("  Building VAL table (%d pairs)…", len(val_pos_pairs) + len(val_neg_pairs))
    df_val = _make_table(val_pos_pairs, val_neg_pairs)

    # ── 6. Kaggle test features ───────────────────────────────────────
    logger.info("  Building TEST table (Kaggle, %s)…", test_edges_path)
    test_ids, test_src, test_dst = read_test_edges(test_edges_path)
    test_pairs = np.stack([test_src, test_dst], axis=1).astype(np.int32)
    test_X = build_dataframe(
        gs_full, test_pairs,
        labels=np.zeros(len(test_pairs), dtype=np.int8),
        max_intermediaries=max_intermediaries,
    )
    test_X["Id"] = test_ids

    # ── 7. Save ───────────────────────────────────────────────────────
    logger.info("Step 6/6 — Saving feature tables")
    for path, df, name in [
        (out_dir_train, df_train, "train"),
        (out_dir_val,   df_val,   "val"),
        (out_dir_test,  test_X,   "test"),
    ]:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        df.to_parquet(path, index=False)
        logger.info("  saved %s → %s (%d rows)", name, path, len(df))

    elapsed = time.time() - t_total
    logger.info("Dataset build complete in %.1f s", elapsed)

    # ── Summary stats ─────────────────────────────────────────────────
    for name, df in [("train", df_train), ("val", df_val)]:
        pos = df["label"].sum()
        logger.info(
            "  [%s] rows=%d, pos=%d (%.1f%%), features=%d",
            name, len(df), pos, 100 * pos / len(df), len(FEATURE_NAMES),
        )


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Build structural feature tables for Approach 04"
    )
    p.add_argument("--graph",    default=PATHS["train_graph"], help="Adjacency CSV path")
    p.add_argument("--test",     default=PATHS["test_edges"],  help="Kaggle test CSV path")
    p.add_argument("--max-train-pos", type=int, default=NEG_SAMPLING["max_train_pos"],
                   help="Max positive pairs for training set (None = all)")
    p.add_argument("--max-val-pos",   type=int, default=NEG_SAMPLING["max_val_pos"],
                   help="Max positive pairs for validation set")
    p.add_argument("--seed",     type=int, default=RANDOM_SEED)
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    build_datasets(
        train_graph_path=args.graph,
        test_edges_path=args.test,
        max_train_pos=args.max_train_pos,
        max_val_pos=args.max_val_pos,
        seed=args.seed,
    )
