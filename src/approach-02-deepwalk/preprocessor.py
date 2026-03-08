import csv
import os
from collections import defaultdict
from dataclasses import dataclass

import gc
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import average_precision_score, roc_auc_score

# GPU: Don't limit threads — let CUDA handle parallelism.
# torch.set_num_threads() caps CPU threads for CPU ops (DataLoader, etc.)
# but on RunPod we want full CPU cores for data prep while GPU trains.
torch.set_num_threads(os.cpu_count())


@dataclass
class DeepWalkConfig:
    # --- Embedding ---
    # VRAM budget (4.87M nodes, fp32):
    #   weights:       2 × 4.87M × 128 × 4B =  5.0 GB
    #   SparseAdam m1: 2 × 4.87M × 128 × 4B =  5.0 GB
    #   SparseAdam m2: 2 × 4.87M × 128 × 4B =  5.0 GB
    #   peak batch:    524288 × 10 × 128 × 4B = 2.7 GB
    #   total:                                ~17.7 GB ✓
    # dim=256 tripled static footprint to ~30 GB via SparseAdam moments → OOM.
    embedding_dim: int = 128

    # --- Walk hyperparameters ---
    # walk_length=40: longer context captures higher-order neighborhoods.
    # Doubled from 20 — CPU walk gen is the bottleneck, not GPU, so the
    # extra pairs are "free" from a GPU utilization standpoint.
    walk_length: int = 40
    # window_size=10: wider context window pairs with the longer walk.
    # Pairs scale as O(walk_length × window_size), so GPU batch sizes
    # below absorb the ~4x increase comfortably.
    window_size: int = 10
    # num_walks_per_node=10: 2x more walks → denser training signal.
    # Each node batch of 16 384 launches 163 840 walkers — still fast
    # with vectorized numpy walks on a 48-core RunPod CPU.
    num_walks_per_node: int = 10
    # num_negative_samples=10: peak neg tensor = 524288 × 10 × 128 × 4B = 2.7 GB
    # Was 15 with dim=256 → 524288 × 15 × 256 × 4B = 32 GB → OOM.
    num_negative_samples: int = 10

    # --- Batch sizes ---
    # batch_nodes=16384: 2x larger node batch per walk round.
    # Produces more (center, context) pairs per GPU step, improving
    # GPU occupancy and amortizing CPU→GPU transfer overhead.
    batch_nodes: int = 16384
    # skipgram_batch_size=524288 (512K): 4x larger than before.
    # A100 SXM has 80 GB VRAM and 2 TB/s bandwidth — the bottleneck
    # is arithmetic throughput, not memory. Larger sub-batches reduce
    # kernel-launch overhead and improve SM utilization.
    skipgram_batch_size: int = 524288

    # --- Optimizer ---
    # lr=0.01: scale lr with effective batch size (linear scaling rule).
    # batch_nodes×num_walks×pairs/walk is ~4x larger → lr ×2 is conservative.
    lr: float = 0.01
    num_epochs: int = 10
    val_ratio: float = 0.05
    seed: int = 42

    # --- Logging / checkpointing ---
    log_every_steps: int = 10
    val_every_steps: int = 100
    save_every_steps: int = 500
    checkpoint_dir: str = "model/deepwalk"
    checkpoint_name: str = "checkpoint_latest.pt"
    final_embeddings_path: str = "model/deepwalk_node_embeddings.pt"


class DirectedDeepWalkModel(nn.Module):
    def __init__(self, num_nodes: int, dim: int):
        super().__init__()
        # sparse=True enables SparseAdam and avoids dense gradient allocation.
        # NOTE: sparse embeddings on CUDA require SparseAdam (not Adam).
        # On GPU, sparse embedding lookups are handled natively by CUDA.
        self.in_emb = nn.Embedding(num_nodes, dim, sparse=True)
        self.out_emb = nn.Embedding(num_nodes, dim, sparse=True)
        init_range = 0.5
        nn.init.uniform_(self.in_emb.weight, -init_range, init_range)
        nn.init.uniform_(self.out_emb.weight, -init_range, init_range)

    def score(self, src_idx: torch.Tensor, dst_idx: torch.Tensor) -> torch.Tensor:
        src = self.in_emb(src_idx)
        dst = self.out_emb(dst_idx)
        return (src * dst).sum(dim=-1)


# ---------------------------------------------------------------------------
# Graph I/O
# ---------------------------------------------------------------------------

def read_graph(path: str = "data/raw/train.csv"):
    adj = defaultdict(list)
    with open(path, "r") as f:
        reader = csv.reader(f)
        for row in reader:
            if not row:
                continue
            src, *neighbors = row
            adj[src].extend(neighbors)
    return adj


def build_edge_list_and_mapping(adj):
    edges = []
    node_id_to_idx = {}

    def ensure_idx(node_id):
        if node_id not in node_id_to_idx:
            node_id_to_idx[node_id] = len(node_id_to_idx)
        return node_id_to_idx[node_id]

    for src, dsts in adj.items():
        src_idx = ensure_idx(src)
        for dst in dsts:
            dst_idx = ensure_idx(dst)
            edges.append((src_idx, dst_idx))

    return np.asarray(edges, dtype=np.int64), node_id_to_idx


def split_train_val_edges(edges, val_ratio: float, seed: int):
    rng = np.random.default_rng(seed)
    edges_arr = np.asarray(edges, dtype=np.int64)
    perm = rng.permutation(len(edges_arr))
    num_val = max(1, int(len(edges_arr) * val_ratio))
    val_idx = perm[:num_val]
    train_idx = perm[num_val:]
    train_edges = edges_arr[train_idx]
    val_pos_edges = edges_arr[val_idx]
    return train_edges, val_pos_edges


def build_csr_bidirectional(num_nodes: int, edges: np.ndarray):
    """Build CSR-format bidirectional adjacency for vectorized walks."""
    src = np.concatenate([edges[:, 0], edges[:, 1]])
    dst = np.concatenate([edges[:, 1], edges[:, 0]])

    order = np.argsort(src, kind="mergesort")
    src_sorted = src[order]
    dst_sorted = dst[order]

    degree = np.bincount(src_sorted, minlength=num_nodes).astype(np.int64)
    indptr = np.zeros(num_nodes + 1, dtype=np.int64)
    np.cumsum(degree, out=indptr[1:])

    reachable = int((degree > 0).sum())
    avg_deg = float(dst_sorted.shape[0]) / max(reachable, 1)
    print(f"  Walk adjacency (CSR): {reachable}/{num_nodes} nodes reachable "
          f"({100.0 * reachable / num_nodes:.1f}%), avg degree: {avg_deg:.1f}")

    return indptr, dst_sorted, degree


def sample_negative_edges(num_samples: int, num_nodes: int, edge_set_encoded: set, seed: int):
    """Sample negative edges."""
    rng = np.random.default_rng(seed)
    negatives = np.empty((num_samples, 2), dtype=np.int64)
    filled = 0

    while filled < num_samples:
        remaining = num_samples - filled
        chunk = int(remaining * 1.1) + 1024
        src = rng.integers(0, num_nodes, size=chunk, dtype=np.int64)
        dst = rng.integers(0, num_nodes, size=chunk, dtype=np.int64)

        mask = src != dst
        src, dst = src[mask], dst[mask]
        encoded = src * num_nodes + dst

        for j in range(len(encoded)):
            if filled >= num_samples:
                break
            if int(encoded[j]) not in edge_set_encoded:
                negatives[filled, 0] = src[j]
                negatives[filled, 1] = dst[j]
                filled += 1

    return negatives


# ---------------------------------------------------------------------------
# Walk + skip-gram batching
# ---------------------------------------------------------------------------

def generate_walks_vectorized(
    source_nodes: np.ndarray, indptr: np.ndarray, indices: np.ndarray,
    degree: np.ndarray, walk_length: int, num_walks: int, rng,
) -> np.ndarray:
    """Generate random walks for all source nodes in parallel using numpy.
    
    Walk generation runs on CPU (numpy) — the graph structure lives in
    CPU RAM and would require costly GPU transfers per step. The resulting
    (center, context) pairs are then transferred to GPU for training.
    """
    walkers = np.repeat(source_nodes, num_walks)
    num_walkers = len(walkers)

    walks = np.full((num_walkers, walk_length), -1, dtype=np.int64)
    walks[:, 0] = walkers
    alive = np.ones(num_walkers, dtype=bool)

    for step in range(1, walk_length):
        cur = walks[alive, step - 1]
        cur_deg = degree[cur]

        has_nbrs = cur_deg > 0
        alive_idx = np.where(alive)[0]

        alive[alive_idx[~has_nbrs]] = False

        active_idx = alive_idx[has_nbrs]
        if len(active_idx) == 0:
            break

        active_nodes = cur[has_nbrs]
        active_deg = cur_deg[has_nbrs]

        rand_offset = (rng.random(len(active_idx)) * active_deg).astype(np.int64)
        next_nodes = indices[indptr[active_nodes] + rand_offset]
        walks[active_idx, step] = next_nodes

    return walks


def build_context_pairs_vectorized(walks: np.ndarray, window_size: int, device):
    """Build (center, context) pairs from a 2D walk array.
    
    Pairs are assembled on CPU (numpy) then transferred to GPU in one
    contiguous block — much faster than incremental GPU tensor ops.
    """
    walk_length = walks.shape[1]
    center_parts = []
    context_parts = []

    for w in range(1, min(window_size + 1, walk_length)):
        c = walks[:, :walk_length - w]
        x = walks[:, w:walk_length]
        valid = (c >= 0) & (x >= 0)
        center_parts.append(c[valid])
        context_parts.append(x[valid])

    if not center_parts:
        return None, None

    centers_np = np.concatenate(center_parts)
    contexts_np = np.concatenate(context_parts)

    if len(centers_np) == 0:
        return None, None

    # pin_memory=True + non_blocking=True: overlap CPU→GPU transfer with
    # the next batch's walk generation for higher GPU utilization.
    centers = torch.from_numpy(centers_np).pin_memory().to(device, non_blocking=True)
    contexts = torch.from_numpy(contexts_np).pin_memory().to(device, non_blocking=True)
    return centers, contexts


def train_skipgram_batch(
    model, optimizer, centers, contexts, num_nodes: int,
    num_negative_samples: int, skipgram_batch_size: int = 65536,
):
    """Train one skip-gram step with gradient-accumulating sub-batches.
    
    Sub-batching keeps peak GPU memory proportional to skipgram_batch_size.
    On a 24GB GPU (A100/A10), skipgram_batch_size=131072 is safe with
    dim=128 and num_negative_samples=10.
    """
    optimizer.zero_grad()
    total_loss = 0.0
    num_pairs = centers.size(0)

    for i in range(0, num_pairs, skipgram_batch_size):
        c = centers[i:i + skipgram_batch_size]
        ctx = contexts[i:i + skipgram_batch_size]

        pos_logits = model.score(c, ctx)
        pos_loss = F.logsigmoid(pos_logits)

        neg_nodes = torch.randint(
            low=0, high=num_nodes,
            size=(c.size(0), num_negative_samples),
            device=c.device,
        )
        center_vec = model.in_emb(c).unsqueeze(1)
        neg_vec = model.out_emb(neg_nodes)
        neg_logits = (center_vec * neg_vec).sum(dim=-1)
        neg_loss = F.logsigmoid(-neg_logits).sum(dim=1)

        sub_loss = -(pos_loss + neg_loss).sum() / num_pairs
        sub_loss.backward()
        total_loss += float(sub_loss.item())

    optimizer.step()
    return total_loss


# ---------------------------------------------------------------------------
# Validation + checkpointing
# ---------------------------------------------------------------------------

@torch.no_grad()
def evaluate_link_prediction(model, val_pos_edges, val_neg_edges, device, eval_batch_size: int = 262144):
    """Evaluate link prediction with batched scoring to avoid GPU OOM."""
    model.eval()

    def _batched_scores(edges_np):
        all_scores = []
        for i in range(0, len(edges_np), eval_batch_size):
            batch = torch.as_tensor(
                edges_np[i:i + eval_batch_size], dtype=torch.long, device=device
            )
            scores = torch.sigmoid(model.score(batch[:, 0], batch[:, 1])).cpu().numpy()
            all_scores.append(scores)
        return np.concatenate(all_scores)

    pos_scores = _batched_scores(val_pos_edges)
    neg_scores = _batched_scores(val_neg_edges)

    pos_mean = float(pos_scores.mean())
    neg_mean = float(neg_scores.mean())

    labels = np.concatenate(
        [np.ones_like(pos_scores, dtype=np.int64), np.zeros_like(neg_scores, dtype=np.int64)]
    )
    scores = np.concatenate([pos_scores, neg_scores])

    auc = roc_auc_score(labels, scores)
    ap = average_precision_score(labels, scores)

    best_f1, best_thr = 0.0, 0.5
    for thr in np.arange(0.05, 0.96, 0.01):
        preds = (scores >= thr).astype(np.int64)
        tp = ((preds == 1) & (labels == 1)).sum()
        fp = ((preds == 1) & (labels == 0)).sum()
        fn = ((preds == 0) & (labels == 1)).sum()
        precision = tp / (tp + fp + 1e-8)
        recall = tp / (tp + fn + 1e-8)
        f1 = 2 * precision * recall / (precision + recall + 1e-8)
        if f1 > best_f1:
            best_f1, best_thr = float(f1), float(thr)

    model.train()
    return {
        "auc": float(auc), "ap": float(ap), "f1": best_f1,
        "threshold": best_thr, "pos_mean": pos_mean, "neg_mean": neg_mean,
    }


def save_checkpoint(path, model, optimizer, epoch, step, best_metrics, node_id_to_idx, config):
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "epoch": epoch,
            "step": step,
            "best_metrics": best_metrics,
            "node_id_to_idx": node_id_to_idx,
            "config": config.__dict__,
        },
        path,
    )


def save_final_embeddings(path, model, node_id_to_idx, best_metrics):
    model.eval()
    with torch.no_grad():
        src_embeddings = model.in_emb.weight.detach().cpu()
        dst_embeddings = model.out_emb.weight.detach().cpu()
        merged_embeddings = ((src_embeddings + dst_embeddings) / 2.0).contiguous()

    torch.save(
        {
            "src_embeddings": src_embeddings,
            "dst_embeddings": dst_embeddings,
            "embeddings": merged_embeddings,
            "node_id_to_idx": node_id_to_idx,
            "num_nodes": int(merged_embeddings.size(0)),
            "dim": int(merged_embeddings.size(1)),
            "best_threshold": best_metrics.get("threshold", 0.5),
            "metrics": best_metrics,
        },
        path,
    )


# ---------------------------------------------------------------------------
# Main entry-point
# ---------------------------------------------------------------------------

def generate_embeddings(
    embedding_dim: int = 128,
    walk_length: int = 40,
    window_size: int = 10,
    num_walks_per_node: int = 10,
    num_negative_samples: int = 10,
    batch_nodes: int = 16384,
    skipgram_batch_size: int = 524288,
    lr: float = 0.01,
    num_epochs: int = 10,
    val_ratio: float = 0.05,
    seed: int = 42,
    log_every_steps: int = 10,
    val_every_steps: int = 50,
    save_every_steps: int = 100,
    checkpoint_dir: str = "model/deepwalk",
    checkpoint_name: str = "checkpoint_latest.pt",
    final_embeddings_path: str = "model/deepwalk_node_embeddings.pt",
    resume_from: str = None,
):
    config = DeepWalkConfig(
        embedding_dim=embedding_dim,
        walk_length=walk_length,
        window_size=window_size,
        num_walks_per_node=num_walks_per_node,
        num_negative_samples=num_negative_samples,
        batch_nodes=batch_nodes,
        skipgram_batch_size=skipgram_batch_size,
        lr=lr,
        num_epochs=num_epochs,
        val_ratio=val_ratio,
        seed=seed,
        log_every_steps=log_every_steps,
        val_every_steps=val_every_steps,
        save_every_steps=save_every_steps,
        checkpoint_dir=checkpoint_dir,
        checkpoint_name=checkpoint_name,
        final_embeddings_path=final_embeddings_path,
    )

    # -----------------------------------------------------------------------
    # GPU setup
    # -----------------------------------------------------------------------
    if not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA not available. On RunPod, ensure you selected a GPU instance "
            "and that the correct PyTorch (CUDA) image is used.\n"
            "Check with: python -c \"import torch; print(torch.cuda.is_available())\""
        )

    device = torch.device("cuda")
    gpu_name = torch.cuda.get_device_name(0)
    gpu_mem_gb = torch.cuda.get_device_properties(0).total_memory / 1024 ** 3
    print(f"GPU: {gpu_name} ({gpu_mem_gb:.1f} GB VRAM)")

    # TF32 is enabled by default on Ampere (A100, A10, RTX 3090+).
    # Explicitly confirm it's on for matmuls — gives ~2x throughput vs FP32
    # on supported GPUs with negligible accuracy loss for embeddings.
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    # -----------------------------------------------------------------------
    # Data loading
    # -----------------------------------------------------------------------
    print("Reading graph...")
    adj = read_graph()
    edges, node_id_to_idx = build_edge_list_and_mapping(adj)
    del adj
    gc.collect()

    num_nodes = len(node_id_to_idx)
    num_edges = len(edges)
    print(f"Total nodes: {num_nodes}, Total edges (directed): {num_edges}")

    train_edges, val_pos_edges = split_train_val_edges(edges, val_ratio=config.val_ratio, seed=config.seed)

    print("Building edge set for negative sampling...")
    edge_set_encoded = set((edges[:, 0] * num_nodes + edges[:, 1]).tolist())
    val_neg_edges = sample_negative_edges(
        num_samples=len(val_pos_edges),
        num_nodes=num_nodes,
        edge_set_encoded=edge_set_encoded,
        seed=config.seed + 1,
    )
    del edge_set_encoded

    print("Building CSR adjacency for vectorized walks...")
    walk_indptr, walk_indices, walk_degree = build_csr_bidirectional(num_nodes, train_edges)

    print(f"Using device: {device}")
    print(f"Train edges: {len(train_edges)}, Val pos: {len(val_pos_edges)}, Val neg: {len(val_neg_edges)}")

    # -----------------------------------------------------------------------
    # Model + optimizer
    # -----------------------------------------------------------------------
    model = DirectedDeepWalkModel(num_nodes=num_nodes, dim=config.embedding_dim).to(device)

    # SparseAdam: required for sparse=True embeddings on both CPU and GPU.
    # On CUDA, sparse embedding gradients are coalesced before the Adam update,
    # so only rows touched in the forward pass are updated — memory-efficient
    # for 4M+ node tables.
    optimizer = torch.optim.SparseAdam(model.parameters(), lr=config.lr)

    # Print GPU memory after model allocation
    allocated = torch.cuda.memory_allocated(0) / 1024 ** 3
    print(f"GPU memory after model init: {allocated:.2f} GB / {gpu_mem_gb:.1f} GB")

    os.makedirs(config.checkpoint_dir, exist_ok=True)
    os.makedirs("logs", exist_ok=True)
    ckpt_path = os.path.join(config.checkpoint_dir, config.checkpoint_name)
    log_path = "logs/deepwalk_training.log"

    start_epoch = 1
    global_step = 0
    best_metrics = {"auc": 0.0, "ap": 0.0, "f1": 0.0, "threshold": 0.5, "step": 0, "epoch": 0}

    if resume_from and os.path.isfile(resume_from):
        print(f"Resuming from checkpoint: {resume_from}")
        # map_location=device ensures tensors load directly onto GPU
        # without a CPU→GPU copy step.
        ckpt = torch.load(resume_from, map_location=device, weights_only=False)
        model.load_state_dict(ckpt["model_state_dict"])
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        start_epoch = int(ckpt.get("epoch", 0)) + 1
        global_step = int(ckpt.get("step", 0))
        best_metrics = ckpt.get("best_metrics", best_metrics)
        print(f"Resumed at epoch={start_epoch}, step={global_step}")

    rng = np.random.default_rng(config.seed)

    # -----------------------------------------------------------------------
    # Training loop
    # -----------------------------------------------------------------------
    with open(log_path, "a") as log_f:
        log_f.write(
            f"# DeepWalk directed | gpu={gpu_name} | nodes={num_nodes} edges(train)={len(train_edges)} "
            f"dim={config.embedding_dim} walk={config.walk_length} window={config.window_size} "
            f"walks/node={config.num_walks_per_node}\n"
        )

        all_train_nodes = np.arange(num_nodes, dtype=np.int64)

        for epoch in range(start_epoch, config.num_epochs + 1):
            rng.shuffle(all_train_nodes)
            epoch_losses = []

            for start in range(0, num_nodes, config.batch_nodes):
                batch_source_nodes = all_train_nodes[start:start + config.batch_nodes]

                # Walk generation runs on CPU — graph CSR lives in CPU RAM.
                walks = generate_walks_vectorized(
                    source_nodes=batch_source_nodes,
                    indptr=walk_indptr,
                    indices=walk_indices,
                    degree=walk_degree,
                    walk_length=config.walk_length,
                    num_walks=config.num_walks_per_node,
                    rng=rng,
                )

                # Pairs assembled on CPU, then pinned+transferred to GPU.
                centers, contexts = build_context_pairs_vectorized(
                    walks=walks,
                    window_size=config.window_size,
                    device=device,
                )

                if centers is None or contexts is None or centers.numel() == 0:
                    continue

                loss = train_skipgram_batch(
                    model=model,
                    optimizer=optimizer,
                    centers=centers,
                    contexts=contexts,
                    num_nodes=num_nodes,
                    num_negative_samples=config.num_negative_samples,
                    skipgram_batch_size=config.skipgram_batch_size,
                )

                global_step += 1
                epoch_losses.append(loss)

                if global_step % config.log_every_steps == 0:
                    # Report GPU memory utilization alongside loss.
                    gpu_used = torch.cuda.memory_allocated(0) / 1024 ** 3
                    gpu_reserved = torch.cuda.memory_reserved(0) / 1024 ** 3
                    msg = (
                        f"Epoch {epoch}/{config.num_epochs} | Step {global_step} | "
                        f"Loss: {loss:.4f} | Pairs: {centers.size(0)} | "
                        f"GPU mem: {gpu_used:.2f}/{gpu_reserved:.2f} GB"
                    )
                    print(msg)
                    log_f.write(msg + "\n")

                if global_step % config.val_every_steps == 0:
                    metrics = evaluate_link_prediction(
                        model=model,
                        val_pos_edges=val_pos_edges,
                        val_neg_edges=val_neg_edges,
                        device=device,
                    )
                    metrics["step"] = global_step
                    metrics["epoch"] = epoch

                    if metrics["f1"] >= best_metrics["f1"]:
                        best_metrics = metrics

                    val_msg = (
                        f"  [VAL] Step {global_step} | AUC: {metrics['auc']:.4f} | "
                        f"AP: {metrics['ap']:.4f} | F1: {metrics['f1']:.4f} | "
                        f"Threshold: {metrics['threshold']:.2f} | "
                        f"Pos/Neg mean: {metrics['pos_mean']:.4f}/{metrics['neg_mean']:.4f}"
                    )
                    print(val_msg)
                    log_f.write(val_msg + "\n")
                    log_f.flush()

                if global_step % config.save_every_steps == 0:
                    save_checkpoint(
                        path=ckpt_path,
                        model=model,
                        optimizer=optimizer,
                        epoch=epoch,
                        step=global_step,
                        best_metrics=best_metrics,
                        node_id_to_idx=node_id_to_idx,
                        config=config,
                    )
                    print(f"  [CKPT] Saved checkpoint: {ckpt_path}")

            if epoch_losses:
                mean_loss = float(np.mean(epoch_losses))
                gpu_used = torch.cuda.memory_allocated(0) / 1024 ** 3
                msg = f"Epoch {epoch} completed | Mean loss: {mean_loss:.4f} | GPU mem: {gpu_used:.2f} GB"
                print(msg)
                log_f.write(msg + "\n")
                log_f.flush()

            new_lr = config.lr * (1 - (epoch - 1) / config.num_epochs)
            for g in optimizer.param_groups:
                g['lr'] = max(new_lr, 3e-4)

    # -----------------------------------------------------------------------
    # Final evaluation + save
    # -----------------------------------------------------------------------
    print("Training completed. Running final validation...")
    final_metrics = evaluate_link_prediction(model, val_pos_edges, val_neg_edges, device)
    final_metrics["step"] = global_step
    final_metrics["epoch"] = config.num_epochs

    if final_metrics["f1"] >= best_metrics["f1"]:
        best_metrics = final_metrics

    print(
        f"Final metrics | AUC: {final_metrics['auc']:.4f} | AP: {final_metrics['ap']:.4f} | "
        f"F1: {final_metrics['f1']:.4f} | Threshold: {final_metrics['threshold']:.2f}"
    )

    save_checkpoint(
        path=ckpt_path,
        model=model,
        optimizer=optimizer,
        epoch=config.num_epochs,
        step=global_step,
        best_metrics=best_metrics,
        node_id_to_idx=node_id_to_idx,
        config=config,
    )
    save_final_embeddings(
        path=config.final_embeddings_path,
        model=model,
        node_id_to_idx=node_id_to_idx,
        best_metrics=best_metrics,
    )

    print(f"Saved final checkpoint: {ckpt_path}")
    print(f"Saved final embeddings: {config.final_embeddings_path}")
    print(f"Best validation metrics: {best_metrics}")


if __name__ == "__main__":
    generate_embeddings()