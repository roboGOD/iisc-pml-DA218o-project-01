# Node2Vec + Logistic Regression — Directed Link Prediction

Scalable link-prediction pipeline for large directed graphs (tested at
5M nodes / 24M edges).  Structural embeddings from lightweight Node2Vec random
walks are combined with a Logistic Regression classifier trained on Hadamard
edge features.

---

## Project layout

```
node2vec_link_prediction/
├── config.py          ← all hyperparameters & file paths (edit here)
├── graph_utils.py     ← graph loading, edge splitting, negative sampling
├── features.py        ← edge feature construction (Hadamard, L1, L2, avg)
├── node2vec_lr.py     ← core model class
├── train.py           ← training entry-point
├── predict.py         ← inference entry-point
├── requirements.txt
└── artifacts/         ← created automatically during training
    ├── node_embeddings.model
    ├── lr_classifier.joblib
    ├── metrics.json
    └── training.log
```

---

## Quick start

### 1  Install dependencies
```bash
pip install -r requirements.txt
```

### 2  Prepare your data

**Train graph** — adjacency-list CSV (header optional):
```
3360982,4457271,9912345
4457271,3360982
9912345,3360982,4457271,1234567
```
Each row: `source_node, neighbour_1, neighbour_2, ...`

**Test file** — fixed format:
```csv
Id,From,To
1,3360982,4457271
2,9912345,1234567
```

### 3  Train
```bash
python train.py --graph train_graph.csv
```

Optional flags:
| Flag | Default | Description |
|------|---------|-------------|
| `--graph` | `train_graph.csv` | Path to adjacency-list CSV |
| `--seed`  | `42`              | Global random seed |

### 4  Predict
```bash
python predict.py --test test.csv --output predictions.csv
```

Optional flags:
| Flag | Default | Description |
|------|---------|-------------|
| `--test`   | `test.csv`        | Test CSV (Id, From, To) |
| `--output` | `predictions.csv` | Output path |
| `--proba`  | off               | Also write edge probability column |

---

## Output format

```csv
Id,Predictions
1,0
2,1
```
`1` = edge predicted to exist · `0` = no edge

With `--proba`:
```csv
Id,Predictions,Probability
1,0,0.1732
2,1,0.8841
```

---

## Tuning guide

All knobs live in **`config.py`**.

| Parameter | Lightweight | Accurate | Notes |
|-----------|-------------|----------|-------|
| `dimensions` | 64 | 128 | Larger → richer but slower LR |
| `walk_length` | 20 | 80 | More context per walk |
| `num_walks` | 10 | 20 | More coverage per node |
| `epochs` (W2V) | 3 | 10 | More passes over walks |
| `p / q` | 1.0 / 1.0 | tune | p<1 → BFS; q<1 → DFS |
| `FEATURE_OPERATOR` | `hadamard` | try all | Best: hadamard or l1 |
| `C` (LR) | 1.0 | tune 0.1–10 | Regularisation strength |

For **very large graphs** consider replacing `node2vec` with
[`pecanpy`](https://github.com/krishnanlab/PecanPy) which is 10–100× faster
on disk-based sparse graphs.

---

## Metrics reported after training

- **ROC-AUC** — primary ranking metric
- **Average Precision** — area under the Precision-Recall curve
- **F1 / Precision / Recall** — threshold-based metrics at 0.5
- Full classification report printed and saved to `artifacts/training.log`
