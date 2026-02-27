
from collections import defaultdict
import csv
import os
import numpy as np
import torch
import torch.nn as nn
from torch_geometric.data import Data
from torch_geometric.utils import negative_sampling
from sklearn.metrics import roc_auc_score, average_precision_score

loss_fn = torch.nn.BCEWithLogitsLoss()


class NodeEmbeddingModel(nn.Module):
    def __init__(self, num_nodes, dim):
        super().__init__()
        self.emb = nn.Embedding(num_nodes, dim)
        nn.init.xavier_uniform_(self.emb.weight)

    def forward(self):
        return self.emb.weight


# ---------------------------------------------------------------------------
# Graph I/O
# ---------------------------------------------------------------------------

def read_graph():
    adj = defaultdict(list)
    with open('data/raw/train.csv', 'r') as f:
        reader = csv.reader(f)
        for row in reader:
            node, *neighbors = row
            adj[node].extend(neighbors)
    return adj


def get_edges_list(adj):
    """Build directed edge list and node-id mapping."""
    edges_list = []
    node_id_to_idx = {}
    idx = 0
    for u, vs in adj.items():
        if u not in node_id_to_idx:
            node_id_to_idx[u] = idx
            idx += 1
        for v in vs:
            if v not in node_id_to_idx:
                node_id_to_idx[v] = idx
                idx += 1
            edges_list.append((node_id_to_idx[u], node_id_to_idx[v]))
    return edges_list, node_id_to_idx


# ---------------------------------------------------------------------------
# Validation helpers
# ---------------------------------------------------------------------------

def split_edges(edge_index, num_nodes, val_ratio=0.05, seed=42):
    """Hold out *val_ratio* of edges (+ equal negatives) for validation."""
    num_edges = edge_index.size(1)
    num_val = int(num_edges * val_ratio)

    rng = torch.Generator().manual_seed(seed)
    perm = torch.randperm(num_edges, generator=rng)

    val_idx = perm[:num_val]
    train_idx = perm[num_val:]

    val_pos_edge_index = edge_index[:, val_idx]
    train_edge_index = edge_index[:, train_idx]

    # Negative edges for validation
    val_neg_edge_index = negative_sampling(
        edge_index=edge_index,
        num_nodes=num_nodes,
        num_neg_samples=num_val,
        method="sparse",
    )

    return train_edge_index, val_pos_edge_index, val_neg_edge_index


@torch.no_grad()
def evaluate(model, val_pos_edge_index, val_neg_edge_index):
    """Compute AUC-ROC, Average Precision, and find the optimal threshold."""
    model.eval()
    z = model()

    pos_scores = edge_score(z, val_pos_edge_index)
    neg_scores = edge_score(z, val_neg_edge_index)

    scores = torch.cat([pos_scores, neg_scores]).sigmoid().cpu().numpy()
    labels = np.concatenate([
        np.ones(pos_scores.size(0)),
        np.zeros(neg_scores.size(0)),
    ])

    auc = roc_auc_score(labels, scores)
    ap = average_precision_score(labels, scores)

    # Find optimal threshold (maximise F1 on validation)
    best_f1, best_thr = 0.0, 0.5
    for thr in np.arange(0.05, 0.96, 0.01):
        preds = (scores >= thr).astype(int)
        tp = ((preds == 1) & (labels == 1)).sum()
        fp = ((preds == 1) & (labels == 0)).sum()
        fn = ((preds == 0) & (labels == 1)).sum()
        prec = tp / (tp + fp + 1e-8)
        rec = tp / (tp + fn + 1e-8)
        f1 = 2 * prec * rec / (prec + rec + 1e-8)
        if f1 > best_f1:
            best_f1 = f1
            best_thr = float(thr)

    model.train()
    return auc, ap, best_f1, best_thr


# ---------------------------------------------------------------------------
# Scoring & training
# ---------------------------------------------------------------------------

def edge_score(z, edge_index):
    src, dst = edge_index
    return (z[src] * z[dst]).sum(dim=1)


def train_step(model, optimizer, train_edge_index, num_nodes, batch_size=65536):
    model.train()
    optimizer.zero_grad()

    num_edges = train_edge_index.size(1)
    perm = torch.randint(0, num_edges, (batch_size,), device=train_edge_index.device)
    pos_edge_index = train_edge_index[:, perm]

    z = model()

    pos_logits = edge_score(z, pos_edge_index)
    pos_labels = torch.ones_like(pos_logits)

    neg_edge_index = negative_sampling(
        edge_index=train_edge_index,
        num_nodes=num_nodes,
        num_neg_samples=pos_edge_index.size(1),
        method="sparse",
    )

    neg_logits = edge_score(z, neg_edge_index)
    neg_labels = torch.zeros_like(neg_logits)

    logits = torch.cat([pos_logits, neg_logits], dim=0)
    labels = torch.cat([pos_labels, neg_labels], dim=0)

    loss = loss_fn(logits, labels)
    loss.backward()

    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    optimizer.step()

    return loss.item()


# ---------------------------------------------------------------------------
# Checkpointing
# ---------------------------------------------------------------------------

def save_checkpoint(model, optimizer, scheduler, node_id_to_idx, step,
                    best_threshold, metrics, path):
    """Save full training state so training can be resumed."""
    torch.save({
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict() if scheduler else None,
        "node_id_to_idx": node_id_to_idx,
        "step": step,
        "best_threshold": best_threshold,
        "metrics": metrics,
    }, path)


def save_embeddings(model, node_id_to_idx, best_threshold, metrics,
                    path='model/node_embeddings.pt'):
    """Save final embeddings + learned threshold for the prediction script."""
    model.eval()
    with torch.no_grad():
        node_embeddings = model().cpu()
    torch.save({
        "embeddings": node_embeddings,
        "num_nodes": node_embeddings.size(0),
        "dim": node_embeddings.size(1),
        "node_id_to_idx": node_id_to_idx,
        "best_threshold": best_threshold,
        "metrics": metrics,
    }, path)


# ---------------------------------------------------------------------------
# Main entry-point
# ---------------------------------------------------------------------------

def generate_embeddings(
    max_steps: int = 2000,
    log_every: int = 1,
    val_every: int = 100,
    save_every: int = 100,
    embedding_dim: int = 128,
    lr: float = 0.01,
    batch_size: int = 65536,
    val_ratio: float = 0.05,
    resume_from: str = None,
):
    print("Reading graph...")
    adj = read_graph()
    edges_list, node_id_to_idx = get_edges_list(adj)

    num_nodes = len(node_id_to_idx)
    num_edges = len(edges_list)
    print(f"Total nodes: {num_nodes}, Total edges (directed): {num_edges}")

    edge_index = torch.as_tensor(edges_list, dtype=torch.long).t().contiguous()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # --- Train / val split ---------------------------------------------------
    print(f"Splitting edges (val_ratio={val_ratio})...")
    train_edge_index, val_pos_edge_index, val_neg_edge_index = split_edges(
        edge_index, num_nodes, val_ratio=val_ratio
    )
    train_edge_index = train_edge_index.to(device)
    val_pos_edge_index = val_pos_edge_index.to(device)
    val_neg_edge_index = val_neg_edge_index.to(device)
    print(f"  Train edges: {train_edge_index.size(1)}, "
          f"Val pos: {val_pos_edge_index.size(1)}, "
          f"Val neg: {val_neg_edge_index.size(1)}")

    # --- Model / optimiser / scheduler ----------------------------------------
    model = NodeEmbeddingModel(num_nodes=num_nodes, dim=embedding_dim).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=max_steps, eta_min=1e-5
    )

    start_step = 1
    best_threshold = 0.5
    best_metrics = {}
    
    torch.set_num_threads(os.cpu_count())

    # Resume from checkpoint if requested
    if resume_from and os.path.isfile(resume_from):
        print(f"Resuming from checkpoint: {resume_from}")
        ckpt = torch.load(resume_from, map_location=device)
        model.load_state_dict(ckpt["model_state_dict"])
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        if ckpt.get("scheduler_state_dict"):
            scheduler.load_state_dict(ckpt["scheduler_state_dict"])
        start_step = ckpt["step"] + 1
        best_threshold = ckpt.get("best_threshold", 0.5)
        best_metrics = ckpt.get("metrics", {})
        print(f"  Resuming from step {start_step}")

    # --- Training loop --------------------------------------------------------
    os.makedirs("model", exist_ok=True)
    os.makedirs("logs", exist_ok=True)

    log_path = "logs/training.log"
    print(f"Starting training (step-based, logging to {log_path})...")

    with open(log_path, "a") as log_f:
        log_f.write(f"# Nodes: {num_nodes}, Edges (train): {train_edge_index.size(1)}, "
                    f"dim={embedding_dim}, lr={lr}, batch={batch_size}\n")

        for step in range(start_step, max_steps + 1):
            loss = train_step(model, optimizer, train_edge_index, num_nodes,
                              batch_size=batch_size)
            scheduler.step()

            if torch.isnan(torch.tensor(loss)):
                raise RuntimeError("NaN loss detected — aborting")

            if step % log_every == 0:
                cur_lr = optimizer.param_groups[0]["lr"]
                msg = (f"Step {step}/{max_steps} | Loss: {loss:.4f} | "
                       f"LR: {cur_lr:.6f}")
                print(msg)
                log_f.write(msg + "\n")

            # --- Validation ---------------------------------------------------
            if step % val_every == 0:
                auc, ap, f1, thr = evaluate(
                    model, val_pos_edge_index, val_neg_edge_index
                )
                best_threshold = thr
                best_metrics = {"auc": auc, "ap": ap, "f1": f1,
                                "threshold": thr, "step": step}
                val_msg = (f"  [VAL] Step {step} | AUC: {auc:.4f} | "
                           f"AP: {ap:.4f} | F1: {f1:.4f} | "
                           f"Threshold: {thr:.2f}")
                print(val_msg)
                log_f.write(val_msg + "\n")
                log_f.flush()

            # --- Checkpoint every `save_every` steps --------------------------
            if step % save_every == 0:
                ckpt_path = f"model/checkpoint_step_{step}.pt"
                save_checkpoint(model, optimizer, scheduler, node_id_to_idx,
                                step, best_threshold, best_metrics, ckpt_path)
                print(f"  [CKPT] Saved checkpoint → {ckpt_path}")
                log_f.write(f"  [CKPT] Saved → {ckpt_path}\n")
                log_f.flush()

    # --- Final save -----------------------------------------------------------
    print("Training completed. Saving final embeddings...")
    save_embeddings(model, node_id_to_idx, best_threshold, best_metrics)
    print(f"Embeddings saved. Best threshold: {best_threshold:.4f}")
    print(f"Final metrics: {best_metrics}")


if __name__ == '__main__':
    generate_embeddings()