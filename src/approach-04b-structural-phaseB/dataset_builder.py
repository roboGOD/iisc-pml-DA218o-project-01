"""
Dataset builder for Approach-04b (Phase A + Phase B features).

Additions over approach-04:
  - Builds Leiden community partition (once, cached to .npy)
  - Passes CommunityStore to build_dataframe for 7 extra Phase B features

Run:
    python dataset_builder.py
"""
from __future__ import annotations

import argparse
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
    PHASE_B_CONFIG,
    RANDOM_SEED,
    SPLIT_CONFIG,
)
from community_store import CommunityStore, build_communities
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
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def train_val_split(
    src: np.ndarray,
    dst: np.ndarray,
    val_ratio: float,
    seed: int = RANDOM_SEED,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    perm = rng.permutation(len(src))
    n_val = int(len(src) * val_ratio)
    val_idx   = perm[:n_val]
    train_idx = perm[n_val:]
    return src[train_idx], dst[train_idx], src[val_idx], dst[val_idx]


def read_test_edges(path: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    df = pd.read_csv(path, encoding="utf-8-sig")
    df.columns = [c.strip() for c in df.columns]
    return (
        df["Id"].to_numpy(dtype=np.int64),
        df["From"].to_numpy(dtype=np.int32),
        df["To"].to_numpy(dtype=np.int32),
    )


# ─────────────────────────────────────────────────────────────────────────────
# Main build
# ─────────────────────────────────────────────────────────────────────────────

def build_datasets(
    train_graph_path: str = PATHS["train_graph"],
    test_edges_path: str = PATHS["test_edges"],
    communities_path: str = PATHS["communities"],
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
    use_community: bool = PHASE_B_CONFIG["use_community"],
    leiden_resolution: float = PHASE_B_CONFIG["leiden_resolution"],
    nbr_sample_k: int = PHASE_B_CONFIG["nbr_sample_k"],
    nbr_list_cap: int = PHASE_B_CONFIG["nbr_list_cap"],
    seed: int = RANDOM_SEED,
) -> None:
    t_total = time.time()

    # ── 1. Load full graph ────────────────────────────────────────────
    logger.info("=" * 60)
    logger.info("Step 1/7 — Loading training graph")
    gs_full = GraphStore.from_adjacency_csv(train_graph_path)
    full_src, full_dst = gs_full.edge_list()
    logger.info("Full graph: %s", gs_full)

    # ── 2. Build / load communities ───────────────────────────────────
    cs: CommunityStore | None = None
    if use_community:
        logger.info("Step 2/7 — Building Leiden communities")
        try:
            community_ids = build_communities(
                gs_full, communities_path,
                resolution=leiden_resolution, seed=seed,
            )
            cs = CommunityStore(community_ids)
            logger.info("  %s", cs)
        except ImportError as e:
            logger.warning(
                "Community detection unavailable (%s). "
                "Community features will be zeros.",
                e,
            )
    else:
        logger.info("Step 2/7 — Community features disabled (use_community=False)")

    # ── 3. Train / val split ──────────────────────────────────────────
    logger.info("Step 3/7 — Splitting edges (val_ratio=%.2f)", val_ratio)
    tr_src, tr_dst, val_src, val_dst = train_val_split(
        full_src, full_dst, val_ratio, seed
    )
    logger.info("  train positives: %d   val positives: %d", len(tr_src), len(val_src))

    # ── Build two graph stores ────────────────────────────────────────
    # gs_train : train edges ONLY — used for train and val feature extraction.
    #   Val positive edges are NOT in this graph, so val features honestly
    #   reflect the "edge not yet present" state — identical to the test setup.
    #   Without this, val positives (u→v) appear in gs_full, inflating every
    #   structural feature for those pairs and giving a falsely high val AUC
    #   (0.98–0.99) that does not generalise to Kaggle.
    # gs_full  : all train.csv edges — used ONLY for Kaggle test features.
    #   The competition has already removed test edges from train.csv, so
    #   gs_full correctly represents the graph at inference time.
    logger.info("Step 3b — Building train-only graph store (val leakage fix)")
    gs_train = GraphStore(num_nodes=gs_full.num_nodes, src=tr_src, dst=tr_dst)
    logger.info("  gs_train: %d nodes, %d edges", gs_train.num_nodes, gs_train.num_edges)
    logger.info("  gs_full:  %d nodes, %d edges (Kaggle test features only)",
                gs_full.num_nodes, gs_full.num_edges)
    # train + val features both use gs_train; Kaggle test uses gs_full
    gs_feat = gs_train

    # ── 4. Subsample positives ────────────────────────────────────────
    logger.info("Step 4/7 — Subsampling positives")
    rng = np.random.default_rng(seed)

    def _sub(s, d, maxn):
        if maxn and len(s) > maxn:
            idx = rng.choice(len(s), maxn, replace=False)
            return s[idx], d[idx]
        return s, d

    tr_src, tr_dst = _sub(tr_src, tr_dst, max_train_pos)
    val_src, val_dst = _sub(val_src, val_dst, max_val_pos)
    logger.info("  subsampled train=%d, val=%d", len(tr_src), len(val_src))

    # ── 5. Sample negatives ───────────────────────────────────────────
    logger.info("Step 5/7 — Sampling negatives")
    train_pos = np.stack([tr_src, tr_dst], axis=1).astype(np.int32)
    val_pos   = np.stack([val_src, val_dst], axis=1).astype(np.int32)

    n_train_neg = int(len(tr_src) * neg_ratio)
    n_val_neg   = int(len(val_src) * neg_ratio)

    t4 = time.time()
    logger.info("  Sampling %d train negatives (hard_frac=%.2f)…", n_train_neg, train_hard_frac)
    train_neg = sample_mixed_negatives(
        gs_feat, n_train_neg, hard_frac=train_hard_frac,
        seed=seed, hub_degree_cap=hub_degree_cap, exclude_pairs=train_pos,
    )
    logger.info("  Train negatives done in %.1fs", time.time() - t4)

    t4v = time.time()
    logger.info("  Sampling %d val negatives (hard_frac=%.2f)…", n_val_neg, val_hard_frac)
    val_neg = sample_mixed_negatives(
        gs_feat, n_val_neg, hard_frac=val_hard_frac,
        seed=seed + 10, hub_degree_cap=hub_degree_cap, exclude_pairs=val_pos,
    )
    logger.info("  Val negatives done in %.1fs | Step 5 total %.1fs",
                time.time() - t4v, time.time() - t4)

    # ── 6. Featurize ──────────────────────────────────────────────────
    logger.info("Step 6/7 — Extracting structural features (Phase A + B)")

    def _make_table(pos: np.ndarray, neg: np.ndarray, name: str) -> pd.DataFrame:
        pairs  = np.concatenate([pos, neg], axis=0)
        labels = np.concatenate([
            np.ones(len(pos),  dtype=np.int8),
            np.zeros(len(neg), dtype=np.int8),
        ])
        perm = rng.permutation(len(pairs))
        logger.info("  [%s] %d pairs (%d pos + %d neg)", name, len(pairs), len(pos), len(neg))
        t = time.time()
        df = build_dataframe(
            gs_feat, pairs[perm], labels[perm],
            max_intermediaries=max_intermediaries,
            cs=cs,
            nbr_sample_k=nbr_sample_k,
            nbr_list_cap=nbr_list_cap,
            seed=seed,
        )
        logger.info("  [%s] done in %.1fs", name, time.time() - t)
        return df

    df_train = _make_table(train_pos, train_neg, "TRAIN")
    df_val   = _make_table(val_pos,   val_neg,   "VAL")

    # ── Kaggle test ───────────────────────────────────────────────────
    # Use gs_full here: all train.csv edges are available at inference time
    # (Kaggle already removed test edges from train.csv before publishing).
    logger.info("  Building TEST table …")
    t_test = time.time()
    test_ids, test_src, test_dst = read_test_edges(test_edges_path)
    test_pairs = np.stack([test_src, test_dst], axis=1).astype(np.int32)
    test_X = build_dataframe(
        gs_full, test_pairs,
        labels=np.zeros(len(test_pairs), dtype=np.int8),
        max_intermediaries=max_intermediaries,
        cs=cs,
        nbr_sample_k=nbr_sample_k,
        nbr_list_cap=nbr_list_cap,
        seed=seed,
    )
    test_X["Id"] = test_ids
    logger.info("  [TEST] done in %.1fs", time.time() - t_test)

    # ── 7. Save ───────────────────────────────────────────────────────
    logger.info("Step 7/7 — Saving feature tables")
    for path, df, name in [
        (out_dir_train, df_train, "train"),
        (out_dir_val,   df_val,   "val"),
        (out_dir_test,  test_X,   "test"),
    ]:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        df.to_parquet(path, index=False)
        logger.info("  saved %s → %s (%d rows, %d features)",
                    name, path, len(df), len(FEATURE_NAMES))

    logger.info("Dataset build complete in %.1fs", time.time() - t_total)
    for name, df in [("train", df_train), ("val", df_val)]:
        pos = df["label"].sum()
        logger.info("  [%s] rows=%d, pos=%d (%.1f%%)", name, len(df), pos, 100*pos/len(df))


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Build Phase A+B feature tables")
    p.add_argument("--graph",   default=PATHS["train_graph"])
    p.add_argument("--test",    default=PATHS["test_edges"])
    p.add_argument("--no-community", action="store_true",
                   help="Skip Leiden (zero-fill community features)")
    p.add_argument("--seed", type=int, default=RANDOM_SEED)
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    build_datasets(
        train_graph_path=args.graph,
        test_edges_path=args.test,
        use_community=not args.no_community,
        seed=args.seed,
    )
