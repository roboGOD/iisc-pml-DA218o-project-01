# Link Prediction on a Directed Social Graph
## DA218o — Practical Machine Learning, Project Report

**Team:** [Team Name]  
**Members:** [Name(s) — Roll No(s)]  
**Institution:** Indian Institute of Science, Bangalore  
**Course:** DA218o — Practical Machine Learning  
**Date:** March 2026

---

## Table of Contents

1. [Problem Statement and Notation](#1-problem-statement-and-notation)
2. [Dataset Description](#2-dataset-description)
3. [Approaches Overview](#3-approaches-overview)
4. [Approach 1 — Simple Heuristic Baselines](#4-approach-1--simple-heuristic-baselines)
5. [Approach 2 — DeepWalk Embeddings](#5-approach-2--deepwalk-embeddings)
6. [Approach 3 — Node2Vec + Logistic Regression](#6-approach-3--node2vec--logistic-regression)
7. [Approach 4 — Structural Features + XGBoost (Phase A)](#7-approach-4--structural-features--xgboost-phase-a)
8. [Approach 4b — Phase A + Phase B Structural Features](#8-approach-4b--phase-a--phase-b-structural-features)
9. [Approach 5 — Hybrid: Structural + Node2Vec Embeddings](#9-approach-5--hybrid-structural--node2vec-embeddings)
10. [Preprocessing and Data Pipeline](#10-preprocessing-and-data-pipeline)
11. [Negative Sampling Strategy](#11-negative-sampling-strategy)
12. [Ablation Studies and Key Findings](#12-ablation-studies-and-key-findings)
13. [Results Summary](#13-results-summary)
14. [Problems Encountered and Solutions](#14-problems-encountered-and-solutions)
15. [Future Scope](#15-future-scope)
16. [References](#16-references)

---

## 1. Problem Statement and Notation

**Task:** Given a directed social graph G = (V, E) representing a Twitter-like follow network, predict whether a directed link (u → v) exists for a set of candidate pairs (u, v) not present in the training edge list. This is a binary classification problem evaluated using Area Under the ROC Curve (AUC).

**Formal definition:** Let G_train = (V, E_train) be the observed graph. For each test pair (u, v) with u ≠ v and (u, v) ∉ E_train, predict P((u, v) ∈ E_true).

**Notation used throughout this report:**

| Symbol | Meaning |
|---|---|
| V | Set of all nodes (users) |
| E | Set of directed edges (follow relationships) |
| N_out(u) | Out-neighbourhood of u: {w : (u, w) ∈ E} |
| N_in(v) | In-neighbourhood of v: {w : (w, v) ∈ E} |
| d_out(u) | Out-degree of u: \|N_out(u)\| |
| d_in(v) | In-degree of v: \|N_in(v)\| |
| G_train | Graph built from 90% of training edges |
| G_full | Graph built from all training edges (used for test features only) |
| AUC | Area Under the ROC Curve (Kaggle evaluation metric) |

---

## 2. Dataset Description

The dataset represents a directed Twitter follow graph with the following characteristics:

| Property | Value |
|---|---|
| Total nodes \|V\| | 4,867,136 |
| Total training edges \|E_train\| | 24,003,361 |
| Test pairs (Kaggle) | ~400,000 |
| Graph type | Directed, unweighted |
| Node ID range | [0, 4,867,135] (dense, contiguous) |

**Key structural observation (critical for modelling):**  
Only ~19,468 nodes (~0.4% of V) have d_out(u) > 0 — i.e., the vast majority of nodes have never followed anyone. All test `From` nodes, however, come from this active-source population (median d_out = 341). This distribution mismatch was the root cause of the 0.5 AUC failure described in Section 14.

**Class distribution:** The Kaggle test set is balanced 50/50 positive/negative; the competition withholds which test edges are true positives. Based on empirical evidence across all experiments, Kaggle test negatives appear to be sampled randomly (not as structural near-misses), which significantly shaped our negative sampling strategy.

---

## 3. Approaches Overview

We developed five approaches in sequence, each building on lessons learned from the previous:

| Approach | Core Method | Features | Kaggle AUC |
|---|---|---|---|
| 1 | Simple heuristic baselines | Common neighbours, Jaccard | N/A |
| 2 | DeepWalk embeddings + classifier | 128-dim random walk embeddings | N/A |
| 3 | Node2Vec + Logistic Regression | 128-dim biased random walk embeddings | N/A |
| 4 | Structural features + XGBoost | 18 hand-crafted graph features (Phase A) | 0.780 → 0.770 |
| 4b | Structural + community + neighbourhood meta-features + XGBoost | 26 features (Phase A + Phase B) | 0.700 |
| 5 | Hybrid: Structural + Node2Vec hadamard + XGBoost | 154 features (26 structural + 128 embedding) | 0.708 |

The progression reflects a deliberate design strategy: start with interpretable structural features (Approach 4), then incrementally add richer but harder-to-debug features (Phase B, embeddings), driven by empirical AUC measurements at each step.

---

## 4. Approach 1 — Simple Heuristic Baselines

**Motivation:** Establish lower-bound baselines using classical link prediction heuristics from the network science literature (Liben-Nowell & Kleinberg, 2003). These require no training and serve as sanity checks.

**Features used:**
- Common neighbours: \|N_out(u) ∩ N_in(v)\| (transitive count in directed setting)
- Jaccard coefficient
- Preferential attachment score

**Implementation:** `src/predict.py`, `src/preprocessor.py`

**Result:** Kaggle AUC — N/A (used as baseline reference, not submitted independently)

**Why superseded:** Pure heuristics cannot be combined or weighted optimally without a trained classifier. They also ignore degree distributions, which are highly skewed (power-law) in this graph.

---

## 5. Approach 2 — DeepWalk Embeddings

**Motivation:** DeepWalk (Perozzi et al., 2014) treats random walks on the graph as sentences and applies Word2Vec to learn node representations. The hypothesis is that nodes appearing in similar walk contexts are likely to form edges.

**Implementation:** `src/approach-02-deepwalk/`

**Edge features:** For a candidate pair (u, v), the edge embedding is formed as the elementwise product (Hadamard) of the node embeddings:

```
f(u, v) = z_u ⊙ z_v     where z_u, z_v ∈ R^128
```

**Classifier:** Trained on top of the 128-dim Hadamard features.

**Result:** Kaggle AUC — N/A

**Why superseded:** DeepWalk uses unbiased uniform random walks, treating the directed graph as undirected. This discards the directional structure of follow relationships. Node2Vec's p/q parameters (Approach 3) allow walks to respect directionality more faithfully.

---

## 6. Approach 3 — Node2Vec + Logistic Regression

**Motivation:** Node2Vec (Grover & Leskovec, 2016) generalises DeepWalk by introducing return parameter p and in-out parameter q to interpolate between depth-first (DFS) and breadth-first (BFS) walk strategies. For a directed follow graph, DFS-leaning walks capture community structure; BFS-leaning walks capture local structural roles.

**Implementation:** `src/approach-03-node2vec-lr/`  
**Library:** PecanPy (SparseOTF mode for memory efficiency on 24M edges)

**Walk hyperparameters:**

| Parameter | Value | Rationale |
|---|---|---|
| dimensions | 128 | Minimum for 4.87M-node graph expressiveness |
| walk_length | 20 | Captures 2nd/3rd order neighbourhood |
| num_walks | 10 | 10 walks per node (total corpus: ~50M sentences) |
| p | 2.0 | Discourages backtracking (directional chains) |
| q | 0.75 | Slight BFS bias (interest clusters) |
| Word2Vec window | 5 | Skip-gram, 3 epochs |

**Edge representation:** Hadamard product of node embeddings, same as Approach 2.

**Classifier:** Logistic Regression with SAGA solver (memory-efficient for large N), capped at 500,000 training pairs to avoid OOM on the embedding matrix multiplication.

**Data split:** 80% train / 10% val / 10% test (internal).

**Result:** Kaggle AUC — N/A (embeddings were reused in Approach 5)

**Why partially superseded:** LR is a linear classifier over 128 non-linear embedding dimensions. The structural features in Approach 4 are far more interpretable and directly target the topology signals relevant to link prediction. However, the trained Node2Vec embeddings were preserved and reused as additional features in Approach 5.

---

## 7. Approach 4 — Structural Features + XGBoost (Phase A)

### 7.1 Motivation

Instead of learning representations implicitly via random walks, we hypothesised that a small set of well-designed structural features would give a stronger, more interpretable signal. Structural features have the advantage of being directly computable from the graph topology without a separate embedding training phase.

For directed graphs specifically, the **transitive neighbourhood** |N_out(u) ∩ N_in(v)| — nodes that u follows AND that follow v — is theoretically the strongest local signal: it approximates the "friends-of-friends" pattern prevalent in social networks.

### 7.2 Feature Set — Phase A (18 features)

All features are computed from the graph at prediction time. For train and validation, features are extracted from G_train (90% of edges). For Kaggle test pairs, features are extracted from G_full (all edges).

**Degree features (8):**

```
out_deg_u = d_out(u)
in_deg_u  = d_in(u)
out_deg_v = d_out(v)
in_deg_v  = d_in(v)

log1p_out_deg_u = log(1 + d_out(u))
log1p_in_deg_u  = log(1 + d_in(u))
log1p_out_deg_v = log(1 + d_out(v))
log1p_in_deg_v  = log(1 + d_in(v))
```

**Rationale:** Degree features capture popularity bias. In a follow network, high d_in(v) nodes (celebrities) attract disproportionate new follows. The log1p transform tames the power-law tail for tree-based learners.

**Edge flag (1):**

```
reciprocal = 1  if (v → u) ∈ E_train,  else 0
```

**Rationale:** Twitter mutual-follow pairs are strongly correlated with real connections. If v already follows u, the probability of u → v completing a reciprocal link is high.

**Neighbourhood overlap counts (3):**

```
common_out   = |N_out(u) ∩ N_out(v)|   (shared followees)
common_in    = |N_in(u)  ∩ N_in(v)|    (shared followers)
transitive   = |N_out(u) ∩ N_in(v)|    (u follows w AND w follows v)
```

**Rationale:** `transitive` is the directed analogue of common neighbours. It identifies walk patterns u → w → v that strongly suggest a latent u → v edge. Computed efficiently using sorted NumPy intersection on pre-sorted adjacency arrays.

**Jaccard normalised overlaps (3):**

```
jaccard_out   = common_out   / (d_out(u) + d_out(v) - common_out)
jaccard_in    = common_in    / (d_in(u)  + d_in(v)  - common_in)
jaccard_trans = transitive   / (d_out(u) + d_in(v)  - transitive)
```

**Rationale:** Raw counts favour high-degree nodes. Jaccard normalisation removes this bias. `jaccard_trans` is the normalised version of the transitive count.

**Preferential attachment (1):**

```
pref_attach = d_out(u) × d_in(v)
```

**Rationale:** From the Barabási-Albert preferential attachment model — high-degree nodes attract new edges proportionally to their degree. This feature was empirically the strongest single predictor (63% XGBoost feature importance) but is also responsible for degree-bias overfitting (see Section 12).

**Weighted transitive overlap (2):**

Let W = N_out(u) ∩ N_in(v) be the set of transitive intermediaries.

```
adamic_adar_trans = Σ_{w ∈ W}  1 / log(d_out(w) + d_in(w) + 2)

resource_alloc_trans = Σ_{w ∈ W}  1 / (d_out(w) + d_in(w) + 1)
```

**Rationale:** Both metrics weight intermediaries inversely by their degree — low-degree intermediaries are more informative signals of a specific connection. Adamic-Adar (Adamic & Adar, 2003) uses a logarithmic penalty; Resource Allocation (Zhou et al., 2009) uses a linear penalty (more aggressive hub down-weighting). Hub intermediaries (w with high degree) contribute little information because they connect nearly everyone.

### 7.3 Graph Store Implementation

Adjacency lists are stored in a custom `GraphStore` class using sorted NumPy int32 arrays (CSR-style). This enables:
- O(n log n) set intersection via `np.intersect1d(assume_unique=True)`
- O(1) degree lookup
- Efficient GPU-to-CPU array operations compatible with XGBoost

### 7.4 Classifier

**XGBoost** with GPU histogram method (`tree_method=hist, device=cuda`). Key hyperparameters (post-tuning with Optuna):

| Parameter | Value |
|---|---|
| max_depth | 4 (shallower generalises better than depth 6-8) |
| n_estimators | up to 3000 with early_stopping_rounds=50 |
| learning_rate | ~0.03 (found by Optuna) |
| subsample | ~0.84 |
| colsample_bytree | ~0.63 |
| reg_alpha | ~0.096 |
| reg_lambda | ~0.22 |
| gamma | ~0.34 |

Early stopping on validation AUC prevents overfitting — without it, training ran all 1000 trees past the AUC peak (peak at iteration 75, final at iteration 999 with AUC 0.05 below peak).

### 7.5 Results

| Configuration | Val AUC | Kaggle AUC |
|---|---|---|
| 1M rows, hard_frac=0.0/0.0 (pre-leakage fix) | ~0.99 (inflated) | 0.780 |
| 1M rows, hard_frac=0.3/0.0 | 0.84 | 0.679 |
| 5M rows, hard_frac=0.0/0.0 (leakage fixed) | 0.85 | 0.770 |
| 5M rows, hard_frac=0.3/0.3 | 0.91 | 0.670 |

---

## 8. Approach 4b — Phase A + Phase B Structural Features

### 8.1 Motivation

After Approach 4 established a baseline, we hypothesised that second-order neighbourhood signals and community membership would add discriminative power beyond local overlap counts.

### 8.2 Feature Set — Phase B (8 additional features, total 26)

Phase B features require sampling from neighbour lists and community detection, making them significantly more expensive to compute (~4× slower per pair than Phase A).

**Neighbourhood meta-features (4):**

For each pair (u, v), sample k=10 nodes from N_out(u) (call them {x_1, ..., x_k}) and k=10 nodes from N_in(v) (call them {y_1, ..., y_k}).

```
fm_proxy = (1/k) × Σ_{x ∈ sample(N_out(u))}  |N_out(x) ∩ N_in(v)|

avg_trans_nbr_in = (1/k) × Σ_{y ∈ sample(N_in(v))}  |N_out(u) ∩ N_in(y)|

avg_jac_trans_nbr_out = (1/k) × Σ_{x ∈ sample(N_out(u))}  jaccard_trans(x, v)

avg_jac_trans_nbr_in  = (1/k) × Σ_{y ∈ sample(N_in(v))}   jaccard_trans(u, y)
```

**Rationale:** `fm_proxy` is a "friends-measure" — it estimates how well u's followee community is already followed-by v. High `fm_proxy` means u and v are deeply embedded in the same interest cluster. These features capture 2nd-order transitivity that the raw `transitive` count misses.

**Leiden community features (3):**

Community detection was run using the Leiden algorithm (Traag et al., 2019) on the full training graph.

```
same_community    = 1 if community(u) == community(v),  else 0
log1p_comm_size_u = log(1 + |community(u)|)
log1p_comm_size_v = log(1 + |community(v)|)
```

**Rationale:** Users in the same community on a follow graph are more likely to follow each other. Community size features allow the model to distinguish node pairs within small tight clusters (high probability) from those in large loose communities (lower probability).

**Hub pair indicator (1):**

```
fm_truncated = 1  if (d_out(u) > nbr_list_cap  OR  d_in(v) > nbr_list_cap)  else 0
              where nbr_list_cap = 200
```

**Rationale:** For hub nodes, the neighbourhood sampling is truncated (we can only afford to sample 10 of potentially 10,000 neighbours). `fm_truncated` signals to the model that the neighbourhood meta-features for this pair are estimates rather than exact values, allowing it to learn a separate rule for hub pairs.

### 8.3 Critical Bug — Community Feature Inconsistency

A significant implementation bug was identified: Leiden communities were built on G_full (all 24M edges) in Step 2 of the dataset builder, **before** the train/val split in Step 3. This meant:

- Phase A features (jaccard, transitive, etc.) were computed on G_train (21.6M edges, 90%)
- Phase B community features were computed on G_full (24M edges, 100%)

The model was trained on a **mixed feature distribution** — Phase A and Phase B features described different graphs. At Kaggle test time, both Phase A and Phase B features came from G_full (consistent), but the learned thresholds from training did not transfer correctly.

**Effect:** Approach 4b scored 0.700 Kaggle AUC vs Approach 4's 0.770 — Phase B features **hurt** by 0.07 AUC due to this inconsistency. The fix (build communities on G_train in Step 3b, after the split) was identified but requires a full dataset rebuild (~3-5 hours) before the improvement can be measured.

### 8.4 Results

| Configuration | Val AUC | Kaggle AUC |
|---|---|---|
| 1M rows, 26 features, leakage fix, hard_frac=0.0/0.0 | 0.84 | 0.700 |

---

## 9. Approach 5 — Hybrid: Structural + Node2Vec Embeddings

### 9.1 Motivation

Structural features and embedding-based features capture complementary information. Structural features are exact and interpretable but local (1-2 hop). Node2Vec embeddings encode multi-hop topological context through random walks but are approximate. Combining them should, in theory, be strictly better than either alone.

### 9.2 Feature Engineering

For each candidate pair (u, v), the 154-dimensional feature vector is:

```
f(u, v) = [ structural(u, v) || hadamard(z_u, z_v) ]
         = [ f_A(u,v), f_B(u,v), z_u ⊙ z_v ]
                18          8          128
```

Where `z_u` and `z_v` are the 128-dimensional Node2Vec embedding vectors from Approach 3, pre-loaded from a saved Gensim KeyedVectors file.

The Hadamard product was selected as the edge operator over alternatives (average, L1, L2) because:
- It is asymmetric — `z_u ⊙ z_v ≠ z_v ⊙ z_u` for directed edges
- Empirically it performs best for link prediction with LR (Hamilton et al., 2017)

Dataset builder for Approach 5 simply loads Approach 4b parquets and appends the embedding columns — no graph re-traversal needed.

### 9.3 Results

| Configuration | Val AUC | Kaggle AUC |
|---|---|---|
| 1M rows, 154 features (inherits 4b parquets) | 0.85 | 0.708 |

The +0.008 Kaggle gain from 128 extra embedding dimensions over 4b alone is nearly noise-level. This suggests the Node2Vec embeddings added minimal signal, likely because:
1. Embeddings were trained with only 3 Word2Vec epochs (undertrained)
2. `p=2.0, q=0.75` walk parameters may not be optimal for link prediction
3. The community feature inconsistency bug in the 4b parquets was inherited by Approach 5

---

## 10. Preprocessing and Data Pipeline

### 10.1 Graph Loading

The raw training graph is stored as an adjacency list CSV (`data/raw/train.csv`) with columns `From, To`. The graph is loaded into a custom `GraphStore` object:

- Nodes are assigned dense integer IDs [0, N-1]
- Out-neighbours and in-neighbours for each node are stored as sorted int32 NumPy arrays
- Out-degree and in-degree arrays are pre-computed for O(1) lookup

### 10.2 Train / Validation Split

All training edges are randomly split:

```
E_train : 90% of edges  →  used for all feature extraction (train + val pairs)
E_val   : 10% of edges  →  used as positive labels only
```

The split is performed by randomly permuting edge indices with a fixed seed (42) for reproducibility.

**Critical design decision — Leakage fix (gs_train vs gs_full):**

In early experiments, features for validation pairs were extracted from G_full (all edges). This caused **target leakage**: the positive val pairs (u → v) were themselves present in G_full, so features like `transitive = |N_out(u) ∩ N_in(v)|` included walks through the very edge being predicted. This inflated val AUC to 0.98-0.99, which did not transfer to Kaggle (where the target edges are withheld).

Fix: Build G_train from E_train only. All train and val features are computed from G_train. Only Kaggle test features use G_full (which does not contain any test edges — the competition has already removed them from the provided `train.csv`).

```
G_train  →  train features  (on E_train positives + sampled negatives)
G_train  →  val features    (on E_val positives + sampled negatives)
G_full   →  Kaggle test features  (all training edges, no test edges)
```

### 10.3 Label Construction

For each split, the dataset consists of:

```
Positive pairs:  (u, v) drawn from E_split,   label = 1
Negative pairs:  (u, v) sampled (no edge),      label = 0
```

Balanced sampling: `neg_ratio = 1.0` (equal positives and negatives).

For 5M training rows: 5M positive pairs + 5M negative pairs = 10M total rows.

### 10.4 Feature Extraction

For each (u, v, label) triple, all 18 (or 26) features are computed by traversing G_train's adjacency arrays. The resulting rows are saved as Apache Parquet files for fast subsequent training runs without graph re-traversal.

```
data/processed/approach04/
    train_features.parquet   (10M rows × 19 cols [18 features + label])
    val_features.parquet     (400K rows × 19 cols)
    test_features.parquet    (~400K rows × 19 cols)
```

---

## 11. Negative Sampling Strategy

### 11.1 Motivation

The choice of negative samples is as important as the features. A model trained on easy negatives (random pairs with low-degree nodes) will learn a trivially different boundary than what Kaggle evaluates.

### 11.2 Easy Negatives (Random)

**Implementation:**
1. Sample source u from **active nodes only** — those with d_out(u) > 0 (~19,468 nodes)
2. Sample target v uniformly from all V
3. Reject if edge (u → v) already exists

**Critical bug (fixed):** In early versions, u was sampled from all 4.87M nodes. Since 99.6% of nodes have d_out = 0, the model learned to predict 0 for any u with no outgoing edges — which never appears in the test set (all test From nodes have high d_out). This caused 0.5 Kaggle AUC (equivalent to random guessing on test).

After fixing u to be sampled only from active nodes, Kaggle AUC jumped from 0.5 to 0.78.

### 11.3 Hard Negatives (2-hop)

We also implemented hard negatives: pairs (u, v) connected by exactly 2 hops (u → w → v), which are structural near-misses that the classifier should not confuse with true edges.

**Tier 1 (2-hop):** Sample w from N_out(u), then sample v from N_out(w). Accept if (u, v) ∉ E.

**Tier 2 (shared-follower):** Sample pivot w, pick two followees v1, v2 of w. Use (v1, v2) as a hard negative.

**Ablation finding:** Hard negatives consistently **hurt** Kaggle AUC regardless of fraction used:

| train_hard_frac | val_hard_frac | Kaggle AUC |
|---|---|---|
| 0.0 | 0.0 | 0.780 (best) |
| 0.3 | 0.0 | 0.679 |
| 0.3 | 0.3 | 0.670 |

**Conclusion:** Kaggle test negatives are sampled randomly (not as 2-hop pairs). Training with hard negatives teaches the model a boundary that misaligns with the test distribution — it learns to score 2-hop structural patterns low, but some of those patterns characterise true test positives.

### 11.4 Cross-Split Isolation

A `seen_codes` set is shared across train/val/test negative sampling to ensure no negative pair appears in more than one split, preventing evaluation leakage through the negative set.

---

## 12. Ablation Studies and Key Findings

### 12.1 Active-Source Fix (0.5 → 0.78 AUC)

| Setting | Kaggle AUC |
|---|---|
| u sampled from all 4.87M nodes | 0.500 |
| u sampled from active nodes only | 0.780 |

The 0.5 AUC was equivalent to random guessing. The root cause was that the training negative distribution was completely different from the test distribution: 99.6% of training negatives had d_out(u) = 0, while 0% of test pairs have d_out(From) = 0.

### 12.2 Leakage Fix (fake 0.99 → honest ~0.85 val AUC)

| Setting | Val AUC | Kaggle AUC |
|---|---|---|
| Val features from G_full (leakage) | 0.98–0.99 | — |
| Val features from G_train (fixed) | ~0.84–0.85 | 0.77–0.78 |

The corrected val AUC is a much more reliable proxy for Kaggle performance.

### 12.3 Data Volume (1M vs 5M training rows)

| Training rows | Kaggle AUC |
|---|---|
| 1M | 0.780 |
| 5M | 0.770 |

Increasing training data by 5× provided no clear improvement. This indicates that the **feature ceiling** for Phase A (18 features) has been reached — more data cannot extract more signal than the feature set encodes.

### 12.4 Hard Negative Fraction

(See Section 11.3 table.) Hard negatives consistently degraded Kaggle performance, confirming Kaggle's test negatives are randomly sampled.

### 12.5 Phase B Features (26 vs 18 features)

| Approach | Features | Kaggle AUC |
|---|---|---|
| 4 (Phase A only) | 18 | 0.770 |
| 4b (Phase A + Phase B) | 26 | 0.700 |

Phase B hurt by 0.07 AUC due to the community inconsistency bug (Section 8.3). The true effect of Phase B features when correctly computed remains to be measured.

### 12.6 Node2Vec Embeddings (154 vs 26 features)

| Approach | Features | Kaggle AUC |
|---|---|---|
| 4b (structural only) | 26 | 0.700 |
| 5 (structural + embeddings) | 154 | 0.708 |

The embedding contribution (+0.008) is marginal. The embeddings were undertrained (3 epochs) with potentially suboptimal walk parameters.

### 12.7 Tree Depth (Optuna Finding)

Optuna consistently found `max_depth = 4` as the best configuration across all top-10 trials on the 1M-row parquets. This suggests the structural features do not require deep interaction trees — the signal is captured within 4 splits of depth.

### 12.8 LR vs XGBoost (degree-bias indicator)

An unexpected finding: Logistic Regression on the 18 structural features achieved **val AUC = 0.977** vs XGBoost's **0.853** for the 0.0/0.0 hard_frac configuration. LR outperforming a boosted ensemble signals that the val set is linearly separable — specifically by `pref_attach = d_out(u) × d_in(v)`.

Val negatives (random u, random v) have typically low d_in(v), while positives always have non-trivial d_in(v). LR exploits this trivial degree boundary effortlessly. XGBoost's lower val AUC actually reflects **better generalisation** — its tree splits are not as dominated by the degree shortcut.

---

## 13. Results Summary

### 13.1 Kaggle Leaderboard Results

| Approach | Features | Rows | hard_frac | Kaggle AUC | Notes |
|---|---|---|---|---|---|
| 4 — Phase A | 18 | 1M | 0.3 / 0.0 | 0.679 | hard_frac mismatch hurt |
| 4b — Phase B | 26 | 1M | 0.0 / 0.0 | 0.700 | community inconsistency bug |
| 5 — Hybrid | 154 | 1M | 0.0 / 0.0 | 0.708 | inherits 4b bug |
| 4 — Phase A | 18 | 5M | 0.3 / 0.3 | 0.670 | hard negatives hurt Kaggle |
| **4 — Phase A** | **18** | **5M** | **0.0 / 0.0** | **0.770** | **best result** |
| 4 — Phase A (pre-leakage fix) | 18 | 1M | — | 0.780 | inflated val, lucky Kaggle |

### 13.2 Feature Importance (Best Model — Approach 4, 5M rows)

| Rank | Feature | XGB Importance |
|---|---|---|
| 1 | pref_attach | 63.3% |
| 2 | resource_alloc_trans | 17.2% |
| 3 | out_deg_u | 4.7% |
| 4 | log1p_out_deg_u | 4.5% |
| 5 | adamic_adar_trans | 2.5% |
| 6 | reciprocal | 1.5% |
| 7 | in_deg_v | 1.3% |
| 8 | transitive | 1.3% |

`pref_attach` dominates at 63%, corroborating the degree-bias finding. The Adamic-Adar and Resource Allocation features carry meaningful signal despite low importance scores.

---

## 14. Problems Encountered and Solutions

### 14.1 Zero AUC (0.5) on First Kaggle Submission

**Problem:** Approach 4's first submission scored 0.5 AUC — equivalent to random guessing — despite 0.9999 validation AUC.

**Root cause:** Negative sampling drew source node u from all 4.87M nodes. 99.6% of those nodes have d_out = 0 (no outgoing edges). The classifier learned to predict 0 for any node with d_out = 0. But all Kaggle test pairs have `From` nodes with d_out ≥ 1 (median = 341) — so the model predicted 0 for every test pair.

**Solution:** Sample u exclusively from the ~19,468 nodes with d_out > 0. Applied consistently across all approaches.

**Impact:** 0.5 → 0.78 Kaggle AUC.

### 14.2 Inflated Validation AUC (Target Leakage)

**Problem:** Validation AUC was 0.98–0.99 but Kaggle AUC was only ~0.78, indicating the validation metric was not a reliable proxy for competition performance.

**Root cause:** Feature extraction used G_full (all edges including validation positive edges). The edge being predicted was present in the feature graph, inflating structural features for positive pairs.

**Solution:** Build G_train from the 90% training split, use it for all train + val feature extraction. Reserve G_full exclusively for Kaggle test feature extraction.

**Impact:** Val AUC corrected to honest ~0.84–0.85 which tracks Kaggle AUC more faithfully.

### 14.3 Hard Negatives Degrading Kaggle AUC

**Problem:** Using 2-hop hard negatives in training (hard_frac = 0.3) caused Kaggle AUC to drop from 0.78 to 0.67.

**Root cause:** Kaggle test negatives appear to be randomly sampled, not structural near-misses. Training with hard negatives teaches the classifier to score 2-hop structural patterns low, but some true Kaggle positives exhibit that same structural fingerprint.

**Solution:** Set train_hard_frac = val_hard_frac = 0.0. Random negatives match the Kaggle test distribution.

### 14.4 Phase B Community Inconsistency

**Problem:** Approach 4b scored 0.700 Kaggle AUC — worse than Approach 4's 0.770 despite having 8 more features.

**Root cause:** Leiden community detection ran on G_full before the train/val split. Phase B community features (same_community, log1p_comm_size) described the 100%-edge graph, while Phase A features described the 90%-edge G_train. The model trained on this mixed incoherent signal.

**Solution (pending rebuild):** Run Leiden on G_train (after the split), use a separate community store built on G_full for Kaggle test features only.

### 14.5 XGBoost Early Stopping Not Applied in train.py

**Problem:** XGBoost was configured with n_estimators=1000 and no early stopping. The model peaked in validation AUC at iteration 75 (0.907) and degraded monotonically to 0.853 by iteration 999. The saved model was significantly overfit.

**Solution:** Added `early_stopping_rounds=50` to XGB_CONFIG and increased `n_estimators=2000` as a high cap, letting the training stop at the true optimum. Also fixed the same issue in `tune.py::retrain_best` — the original code reused the trial's n_estimators (44 trees) for the final model instead of letting early stopping find the true convergence point.

### 14.6 Node2Vec OOM During LR Training

**Problem:** Training Logistic Regression on all 19M+ edge embedding pairs caused out-of-memory errors.

**Solution:** Capped training pairs at MAX_LR_PAIRS = 500,000. LR is a convex model; the optimal hyperplane does not change significantly beyond 500K balanced samples. Switched solver from `lbfgs` (full-batch) to `saga` (mini-batch SGD) for further memory efficiency.

---

## 15. Future Scope

### 15.1 Fix Phase B Community Inconsistency
Rebuild Approach 4b parquets with Leiden run on G_train only. This would give a fair comparison of Phase B features and potentially recover the ~0.07 AUC gap observed between Approach 4 and 4b.

### 15.2 Retrain Node2Vec Properly
Current embeddings were trained with:
- Only 3 Word2Vec epochs (too few for a 4.87M node graph)
- `p=2.0, q=0.75` (not standard for link prediction)

Recommended settings: `p=1.0, q=1.0` (uniform walks), `epochs=5–10`. This would likely improve the embedding quality and the Approach 5 Kaggle AUC.

### 15.3 Graph Neural Networks (GCN / GraphSAGE)
The hard ceiling on structural features (~0.78 AUC) suggests that hand-crafted local features have limited discriminative power. Graph Neural Networks aggregate multi-hop neighbourhood information with learnable weights, potentially breaking through this ceiling. GraphSAGE with mean aggregation over 2-hop neighbourhoods would be a natural next step.

### 15.4 Proper Hyperparameter Tuning on 5M Rows
Optuna tuning was performed on 1M rows where the model converged in only 44 trees. Re-running 40 Optuna trials on the 5M-row parquets with the corrected early stopping would find the true optimal hyperparameters for the larger dataset.

### 15.5 Alternative Edge Operators for Embeddings
Only the Hadamard operator was tested in Approach 5. Other operators (average, L1-distance, L2-distance, concatenation) capture different aspects of node similarity and may perform better for directed graphs.

### 15.6 Temporal Features
If edge timestamps were available, features like "time since u joined" or "relative temporal activity" could provide significant signal. The current formulation ignores all temporal structure.

### 15.7 Degree Bias Mitigation
`pref_attach` dominated feature importances at 63%, indicating the model relies heavily on degree-based signals. Explicit bias mitigation (e.g., normalising features by expected degree under a configuration model null hypothesis) could reduce overfitting to popular nodes and improve performance on non-popular users.

### 15.8 Larger Training Data
The 5M-row experiment showed no clear improvement over 1M rows for Phase A features. However, with Phase B community features fixed and hard_frac=0.0, a full 21.6M row training set might capture long-tail structural patterns not present in the 5M subsample. Memory-efficient XGBoost with external memory would be required.

---

## 16. References

1. Liben-Nowell, D., & Kleinberg, J. (2003). *The link prediction problem for social networks.* CIKM 2003.
2. Perozzi, B., Al-Rfou, R., & Skiena, S. (2014). *DeepWalk: Online learning of social representations.* KDD 2014.
3. Grover, A., & Leskovec, J. (2016). *node2vec: Scalable feature learning for networks.* KDD 2016.
4. Adamic, L. A., & Adar, E. (2003). *Friends and neighbors on the web.* Social Networks, 25(3).
5. Zhou, T., Lü, L., & Zhang, Y. C. (2009). *Predicting missing links via local information.* EPJ B, 71(4).
6. Traag, V. A., Waltman, L., & Van Eck, N. J. (2019). *From Louvain to Leiden: guaranteeing well-connected communities.* Scientific Reports.
7. Hamilton, W. L., Ying, R., & Leskovec, J. (2017). *Inductive representation learning on large graphs.* NeurIPS 2017.
8. Chen, T., & Guestrin, C. (2016). *XGBoost: A scalable tree boosting system.* KDD 2016.
9. Ma, L., et al. (2021). *PecanPy: A fast, efficient and parallelized Python implementation of node2vec.* Bioinformatics.

---

*This report was prepared as part of the DA218o Practical Machine Learning course at IISc, Bangalore.*
