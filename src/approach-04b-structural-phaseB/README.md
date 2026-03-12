# Approach-04b — Structural Features Phase A + Phase B

Extends approach-04 with 7 additional Phase B features (25 total) and
Leiden community detection.

## Phase A features (18) — same as approach-04
degrees, log-degrees, reciprocal, common-out/in, transitive, Jaccard
variants, preferential attachment, Adamic-Adar, Resource Allocation.

## Phase B features (7 new)
| Feature | Description |
|---|---|
| `fm_proxy` | avg \|N\_out(x) ∩ N\_in(v)\| for x sampled from N\_out(u) |
| `avg_trans_nbr_in` | avg transitive(u→y) for y sampled from N\_in(v) |
| `avg_jac_trans_nbr_out` | avg jaccard\_trans(x→v) for x sampled from N\_out(u) |
| `avg_jac_trans_nbr_in` | avg jaccard\_trans(u→y) for y sampled from N\_in(v) |
| `same_community` | 1 if u and v share a Leiden community |
| `log1p_comm_size_u` | log1p community size of u |
| `log1p_comm_size_v` | log1p community size of v |

## Install dependencies
```bash
pip install leidenalg igraph
```

## Run pipeline
```bash
cd src/approach-04b-structural-phaseB

# Build features (runs Leiden on first call, ~10-20 min for 4.87M nodes)
python dataset_builder.py

# Skip community detection (faster, community features will be zeros):
python dataset_builder.py --no-community

# Train
python train.py

# Predict
python predict.py
```

## Artifacts
| Path | Description |
|---|---|
| `data/processed/approach04b/communities.npy` | Leiden community IDs (cached) |
| `data/processed/approach04b/train_features.parquet` | 25 features + label |
| `models/approach04b/model.joblib` | Trained LGBM |
| `models/approach04b/metrics.json` | AUC, AP, F1, threshold |
| `data/processed/approach04b/predictions.csv` | Kaggle submission |
