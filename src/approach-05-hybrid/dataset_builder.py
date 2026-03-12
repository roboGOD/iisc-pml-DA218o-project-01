"""
Approach-05 dataset builder: structural (25) + Node2Vec hadamard (128) = 153 features.

Prerequisites
-------------
1. Run approach-04b dataset_builder.py first → generates approach04b parquets
2. Run approach-03 train.py first → generates Node2Vec Gensim model

Usage
-----
    cd src/approach-05-hybrid
    python dataset_builder.py

The script loads approach-04b parquets, computes Node2Vec hadamard edge
embeddings for every (u, v) pair, horizontally concatenates them, and saves
combined parquets to data/processed/approach05/.
"""
from __future__ import annotations

import json
import logging
import os
import sys
import time

import numpy as np
import pandas as pd

# ── allow running from both src/approach-05-hybrid/ and project root ──────────
_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)

import config
from embedding_store import EmbeddingStore

# ── Structural feature names come from approach-04b ───────────────────────────
_APPROACH04B = os.path.join(_HERE, "..", "approach-04b-structural-phaseB")
sys.path.insert(0, os.path.abspath(_APPROACH04B))
try:
    from structural_features import FEATURE_NAMES as STRUCTURAL_FEATURE_NAMES
except ImportError:
    # Fallback: read from saved feature_names.json in approach-04b models dir
    _fn_path = os.path.join(
        os.path.dirname(__file__), "..", "..",
        "models", "approach04b", "feature_names.json"
    )
    if os.path.exists(_fn_path):
        with open(_fn_path) as _f:
            STRUCTURAL_FEATURE_NAMES = json.load(_f)
    else:
        raise ImportError(
            "Cannot import structural_features from approach-04b. "
            "Ensure approach-04b-structural-phaseB exists in the src folder."
        )

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


# ────────────────────────────────────────────────────────────────────────────
def _check_source_parquets() -> None:
    """Abort early with clear message if approach-04b parquets are missing."""
    missing = [
        k for k, v in config.SOURCE_PATHS.items() if not os.path.exists(v)
    ]
    if missing:
        logger.error(
            "Missing approach-04b parquets: %s\n"
            "Run approach-04b/dataset_builder.py first.",
            missing,
        )
        sys.exit(1)


def _check_node2vec_model() -> None:
    """Abort early with clear instructions if Node2Vec model is missing."""
    if not os.path.exists(config.NODE2VEC_MODEL_PATH):
        logger.error(
            "\nNode2Vec model not found: %s\n"
            "\nGenerate it first:\n"
            "  cd src/approach-03-node2vec-lr\n"
            "  conda run -n node_pred python train.py\n"
            "\nNote: training takes 30-60 min on the full graph (4.87M nodes).",
            config.NODE2VEC_MODEL_PATH,
        )
        sys.exit(1)


def _build_embedding_feature_names() -> list[str]:
    op = config.EMBEDDING_OPERATOR  # e.g. "hadamard"
    dim = config.EMBEDDING_DIM
    return [f"n2v_{op[:3]}_{i}" for i in range(dim)]


def _process_split(
    es: EmbeddingStore,
    split_name: str,
    src_path: str,
    dst_path: str,
    emb_feature_names: list[str],
) -> None:
    """Load one parquet, append embedding features, save combined parquet."""
    logger.info("Processing split: %s → %s", split_name, os.path.basename(dst_path))
    t0 = time.time()

    df = pd.read_parquet(src_path)
    n = len(df)
    logger.info("  Loaded %d rows", n)

    # Verify required structural columns exist
    missing_cols = [c for c in STRUCTURAL_FEATURE_NAMES if c not in df.columns]
    if missing_cols:
        raise ValueError(
            f"Structural features missing from {src_path}: {missing_cols[:5]} …"
        )

    # Extract node pairs
    u_arr = df["u"].values.astype(np.int64)
    v_arr = df["v"].values.astype(np.int64)

    # Compute embedding features
    t1 = time.time()
    emb_feats = es.edge_features_batch(u_arr, v_arr, operator=config.EMBEDDING_OPERATOR)
    logger.info("  Embedding features computed in %.1fs (shape %s)", time.time() - t1, emb_feats.shape)

    # Append embedding columns
    emb_df = pd.DataFrame(emb_feats, columns=emb_feature_names, index=df.index)
    out_df = pd.concat([df, emb_df], axis=1)

    # Save
    os.makedirs(os.path.dirname(dst_path), exist_ok=True)
    out_df.to_parquet(dst_path, index=False)
    logger.info(
        "  Saved %d rows × %d cols in %.1fs → %s",
        len(out_df), len(out_df.columns), time.time() - t0, dst_path,
    )


# ────────────────────────────────────────────────────────────────────────────
def main() -> None:
    logger.info("=" * 65)
    logger.info("Approach-05 dataset builder")
    logger.info("  Structural features : %d (%s…)", len(STRUCTURAL_FEATURE_NAMES), STRUCTURAL_FEATURE_NAMES[:3])
    logger.info("  Embedding operator  : %s (dim=%d)", config.EMBEDDING_OPERATOR, config.EMBEDDING_DIM)
    logger.info("  Total features      : %d", len(STRUCTURAL_FEATURE_NAMES) + config.EMBEDDING_DIM)
    logger.info("=" * 65)

    _check_source_parquets()
    _check_node2vec_model()

    # Infer num_nodes from graph if possible (for EmbeddingStore matrix sizing)
    num_nodes = None
    if os.path.exists(config.PATHS["train_graph"]):
        import csv
        max_id = 0
        with open(config.PATHS["train_graph"]) as f:
            reader = csv.reader(f)
            next(reader, None)  # skip header
            for row in reader:
                if len(row) >= 2:
                    max_id = max(max_id, int(row[0]), int(row[1]))
        num_nodes = max_id + 1
        logger.info("Graph max node id=%d → num_nodes=%d", max_id, num_nodes)

    # Load embedding store (dense matrix, one-time cost)
    es = EmbeddingStore.from_gensim_model(
        config.NODE2VEC_MODEL_PATH, num_nodes=num_nodes
    )
    logger.info("EmbeddingStore: %s", es)

    emb_feature_names = _build_embedding_feature_names()
    logger.info("Embedding feature names: %s … (total %d)", emb_feature_names[:3], len(emb_feature_names))

    # Save combined feature names for train.py
    combined_feature_names = list(STRUCTURAL_FEATURE_NAMES) + emb_feature_names
    fn_path = config.PATHS["feature_names"]
    os.makedirs(os.path.dirname(fn_path), exist_ok=True)
    with open(fn_path, "w") as fp:
        json.dump(combined_feature_names, fp)
    logger.info("Saved combined feature names (%d) → %s", len(combined_feature_names), fn_path)

    # Process each split
    splits = [
        ("train", config.SOURCE_PATHS["train_feats"], config.PATHS["train_feats"]),
        ("val",   config.SOURCE_PATHS["val_feats"],   config.PATHS["val_feats"]),
        ("test",  config.SOURCE_PATHS["test_feats"],  config.PATHS["test_feats"]),
    ]

    t_total = time.time()
    for split_name, src, dst in splits:
        _process_split(es, split_name, src, dst, emb_feature_names)

    logger.info("All splits done in %.1f min", (time.time() - t_total) / 60)
    logger.info("Run train.py next.")


if __name__ == "__main__":
    main()
