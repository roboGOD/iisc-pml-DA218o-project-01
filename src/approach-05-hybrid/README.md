# Approach-05: Hybrid (Structural + Node2Vec) → LightGBM

Combines 25 structural/community features from Approach-04b with 128-dim
Node2Vec edge embeddings (hadamard product) → 153 total features trained on
LightGBM / XGBoost / HGB.

---

## Feature breakdown

| Source | Type | Count |
|---|---|---|
| Approach-04b Phase A | Structural (degree, overlap, centrality) | 18 |
| Approach-04b Phase B | Community / transitive neighbourhood | 7 |
| Node2Vec hadamard | Embedding dot-product | 128 |
| **Total** | | **153** |

---

## Run order (critical — two hard prerequisites)

### Step 1 — Generate Node2Vec embeddings (30-60 min, one-time)

```bash
cd src/approach-03-node2vec-lr
conda run -n node_pred python train.py
```

Writes exactly **2 files** (~2.5 GB total, no intermediate checkpoints):
- `model/approach-03/node_embeddings.model` — small metadata pickle (KeyedVectors)
- `model/approach-03/node_embeddings.model.vectors.npy` — float32 matrix (the actual vectors)

### Step 2 — Build approach-04b structural parquets (first time runs Leiden community detection ≈ 10-30 min)

```bash
cd src/approach-04b-structural-phaseB
conda run -n node_pred python dataset_builder.py
```

Generates: `data/processed/approach04b/train_features.parquet`, `val_features.parquet`, `test_features.parquet`

### Step 3 — Build approach-05 combined parquets

```bash
cd src/approach-05-hybrid
conda run -n node_pred python dataset_builder.py
```

- Loads approach-04b parquets (structural features already there)
- Builds dense EmbeddingStore from Gensim model (≈2.5 GB float32 matrix)
- Computes Node2Vec hadamard edge features for every pair
- Saves combined parquets to `data/processed/approach05/`

### Step 4 — Train

```bash
conda run -n node_pred python train.py
# or with a different backend:
conda run -n node_pred python train.py --classifier xgb
```

Saves artifacts to `models/approach05/`.

### Step 5 — Predict

```bash
conda run -n node_pred python predict.py
# or with custom threshold:
conda run -n node_pred python predict.py --threshold 0.45
```

Writes: `data/processed/approach05/predictions.csv`

---

## Embedding operators

Configurable in `config.py` → `EMBEDDING_OPERATOR`:

| Operator | Formula | Notes |
|---|---|---|
| `hadamard` | $e_u \odot e_v$ | Default; preserves sign, good for directed links |
| `average` | $(e_u + e_v)/2$ | Symmetric |
| `l1` | $\|e_u - e_v\|$ | Unsigned distance |
| `l2` | $(e_u - e_v)^2$ | Squared distance |

---

## File structure

```
src/approach-05-hybrid/
├── config.py          # All paths, hyperparams, embedding config
├── embedding_store.py # Dense matrix over Gensim Word2Vec, hadamard etc.
├── dataset_builder.py # Loads 04b parquets + appends Node2Vec features
├── train.py           # LGBM/XGB/HGB/RF training + threshold tuning
├── predict.py         # Kaggle submission inference
└── README.md          # This file
```

---

## Expected performance

| Metric | Estimate |
|---|---|
| Kaggle public AUC | 0.78–0.88 |
| vs Approach-04b | +0.01–0.04 AUC (embeddings add complementary signal) |

> **Note**: Node2Vec learns global proximity (nodes that appear in similar walks
> are close in embedding space). Structural features capture local topology.
> Together they cover both ranges.

---

## Troubleshooting

**`FileNotFoundError: Node2Vec model not found`**
→ Run Step 1 above.

**`Missing approach-04b parquets`**
→ Run Step 2 above.

**Out of memory in `dataset_builder.py`**
→ Dense matrix is ≈2.5 GB (4.87M × 128 × float32). Requires ≥6 GB free RAM.
  If tight, add `del es` after building all splits.

**`ImportError: gensim`**
→ `conda run -n node_pred pip install gensim`
