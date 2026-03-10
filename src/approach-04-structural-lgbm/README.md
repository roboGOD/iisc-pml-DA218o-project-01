# Approach 04 — Structural Features + LightGBM

Directed link prediction on the Twitter graph using hand-crafted structural
graph features and a gradient-boosted classifier (LightGBM primary).

---

## Problem

Binary prediction: given a directed edge (u → v), does the edge exist in
the underlying social network?

- Train graph: `data/raw/train.csv` — ragged adjacency list, ~24 M edges
- Test pairs : `data/raw/test.csv`  — 2 000 pairs (1 000 real + 1 000 fake)

---

## Why this approach

1. **Dense node IDs** (0 … 4 867 135) allow scipy CSR to replace all Python
   dicts → ~6× less memory, faster neighbour look-ups.
2. **18 directed structural features** capture both symmetric network effects
   (common neighbours, Jaccard) and directed effects (transitive closure,
   reciprocal edge).
3. **Hard negative sampling** (2-hop u → w → v without direct edge) makes the
   classifier learn finer decision boundaries compared to random negatives.
4. **LightGBM** with early stopping prevents overfitting and trains in
   O(minutes) on CPU.

---

## Pipeline

```
train.csv  ──► GraphStore (CSR)
                   │
                   ├─► train / val edge split (90 / 10 %)
                   │
                   ├─► gs_train (val edges removed for leak-free features)
                   │
                   ├─► sample mixed negatives (50 % hard / 70 % hard)
                   │
                   ├─► extract 18 structural features   ──► *.parquet
                   │
                   └─► LightGBM  ──► model.joblib
                                 ──► threshold.json
                                 ──► metrics.json

test.csv   ──► extract features (cached in test_features.parquet)
           ──► predict_proba  ──► threshold  ──► predictions.csv
```

---

## Features (18 total)

| # | Name | Description |
|---|------|-------------|
| 1 | out_deg_u | out-degree of source node |
| 2 | in_deg_u | in-degree of source node |
| 3 | out_deg_v | out-degree of target node |
| 4 | in_deg_v | in-degree of target node |
| 5–8 | log1p_* | log1p of above degrees |
| 9 | reciprocal | 1 if v→u exists, else 0 |
| 10 | common_out | \|N_out(u) ∩ N_out(v)\| |
| 11 | common_in | \|N_in(u) ∩ N_in(v)\| |
| 12 | **transitive** | \|N_out(u) ∩ N_in(v)\| ← most predictive |
| 13 | jaccard_out | transitive Jaccard using out(u) and in(v) |
| 14 | jaccard_in | N_in(u) ∩ N_in(v) Jaccard |
| 15 | jaccard_trans | transitive Jaccard normalised |
| 16 | pref_attach | out_deg_u × in_deg_v |
| 17 | adamic_adar_trans | AA over shared transitive intermediaries |
| 18 | resource_alloc_trans | RA over shared transitive intermediaries |

---

## File structure

```
src/approach-04-structural-lgbm/
    config.py              # all hyper-parameters & paths
    graph_store.py         # memory-efficient directed graph (scipy CSR)
    negative_sampling.py   # easy / hard / mixed negative sampler
    structural_features.py # 18-feature extractor
    dataset_builder.py     # offline pipeline: graph → parquet
    train.py               # classifier training + evaluation
    predict.py             # Kaggle inference → submission CSV

data/processed/approach04/
    train_features.parquet
    val_features.parquet
    test_features.parquet
    predictions.csv
    predictions_with_proba.csv   # for hybrid stacking (Approach 5)

models/approach04/
    model.joblib
    threshold.json
    metrics.json
    feature_names.json
```

---

## Quick start

```bash
conda activate node_pred
cd src/approach-04-structural-lgbm

# 1. Build feature tables (once, takes ~15–30 min on CPU)
python dataset_builder.py

# 2. Train LightGBM
python train.py

# 3. Generate submission
python predict.py
```

---

## Config knobs

Edit `config.py` to change:

| Key | Default | Effect |
|-----|---------|--------|
| `SPLIT_CONFIG["val_ratio"]` | 0.10 | fraction of edges held out for validation |
| `NEG_SAMPLING["max_train_pos"]` | 500 000 | positive examples for training |
| `NEG_SAMPLING["hard_frac_train"]` | 0.50 | fraction of hard negatives in training set |
| `CLASSIFIER` | `"lgbm"` | `"lgbm"` / `"xgb"` / `"hgb"` / `"rf"` |
| `LGBM_CONFIG["n_estimators"]` | 1 000 | max boosting rounds (early stopping active) |
| `FEATURE_CONFIG["max_intermediaries"]` | 500 | cap on AA/RA summation |

---

## Notes

- `test.csv` has a UTF-8 BOM character; handled with `encoding="utf-8-sig"`.
- AA/RA features are capped at `max_intermediaries=500` to avoid hub-node
  computation blow-up (high-degree hubs can have 100 k+ neighbours).
- `predictions_with_proba.csv` is automatically written for future
  Approach-05 meta-learning stacking experiments.
