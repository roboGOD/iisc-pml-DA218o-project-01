import argparse
import os

import torch
import torch.nn as nn


# ---------------------------------------------------------------------------
# Model definition (must match preprocessor.py exactly)
# ---------------------------------------------------------------------------

class DirectedDeepWalkModel(nn.Module):
    """Directed DeepWalk model with separate in/out embedding tables."""

    def __init__(self, num_nodes: int, dim: int):
        super().__init__()
        self.in_emb  = nn.Embedding(num_nodes, dim, sparse=True)
        self.out_emb = nn.Embedding(num_nodes, dim, sparse=True)

    def score(self, src_idx: torch.Tensor, dst_idx: torch.Tensor) -> torch.Tensor:
        return (self.in_emb(src_idx) * self.out_emb(dst_idx)).sum(dim=-1)


# ---------------------------------------------------------------------------
# Core export logic
# ---------------------------------------------------------------------------

def export_embeddings(checkpoint_path: str, output_path: str) -> None:
    if not os.path.isfile(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    print(f"Loading checkpoint: {checkpoint_path}")
    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)

    # ---- Validate required keys ----------------------------------------
    for key in ("model_state_dict", "node_id_to_idx"):
        if key not in ckpt:
            raise KeyError(f"Checkpoint is missing required key: '{key}'")

    node_id_to_idx: dict = ckpt["node_id_to_idx"]
    num_nodes = len(node_id_to_idx)

    # ---- Infer embedding dim from the saved weights ---------------------
    state_dict = ckpt["model_state_dict"]
    in_weight = state_dict.get("in_emb.weight")
    if in_weight is None:
        raise KeyError("model_state_dict does not contain 'in_emb.weight'.")
    embedding_dim = in_weight.shape[1]

    # Also accept dim from config if present (sanity check)
    cfg = ckpt.get("config", {})
    cfg_dim = cfg.get("embedding_dim")
    if cfg_dim is not None and cfg_dim != embedding_dim:
        print(
            f"  Warning: config embedding_dim={cfg_dim} differs from "
            f"weight shape dim={embedding_dim}. Using weight shape."
        )

    # ---- Reconstruct and load model ------------------------------------
    print(f"  Nodes: {num_nodes:,}  |  Embedding dim: {embedding_dim}")
    model = DirectedDeepWalkModel(num_nodes=num_nodes, dim=embedding_dim)
    model.load_state_dict(state_dict)
    model.eval()

    # ---- Extract embeddings (CPU, no grad) -----------------------------
    with torch.no_grad():
        src_embeddings    = model.in_emb.weight.detach().cpu()
        dst_embeddings    = model.out_emb.weight.detach().cpu()
        merged_embeddings = ((src_embeddings + dst_embeddings) / 2.0).contiguous()

    # ---- Retrieve best metrics / threshold -----------------------------
    best_metrics: dict = ckpt.get("best_metrics", {})
    best_threshold = float(best_metrics.get("threshold", 0.5))

    epoch = ckpt.get("epoch", "?")
    step  = ckpt.get("step",  "?")
    print(f"  Checkpoint epoch: {epoch}  |  step: {step}")
    print(f"  Best threshold:   {best_threshold:.4f}")

    auc = best_metrics.get("auc")
    ap  = best_metrics.get("ap")
    f1  = best_metrics.get("f1")
    fmt = lambda v: f"{v:.4f}" if isinstance(v, (float, int)) else "N/A"
    print(f"  Best val metrics — AUC: {fmt(auc)}, AP: {fmt(ap)}, F1: {fmt(f1)}")

    # ---- Write final-embeddings file -----------------------------------
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    payload = {
        "src_embeddings": src_embeddings,
        "dst_embeddings": dst_embeddings,
        "embeddings":     merged_embeddings,
        "node_id_to_idx": node_id_to_idx,
        "num_nodes":      int(merged_embeddings.size(0)),
        "dim":            int(merged_embeddings.size(1)),
        "best_threshold": best_threshold,
        "metrics":        best_metrics,
    }
    torch.save(payload, output_path)
    print(f"\nEmbeddings saved to: {output_path}")
    print(
        f"  src_embeddings : {tuple(src_embeddings.shape)}\n"
        f"  dst_embeddings : {tuple(dst_embeddings.shape)}\n"
        f"  merged         : {tuple(merged_embeddings.shape)}"
    )


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Export DeepWalk node embeddings from a training checkpoint."
    )
    parser.add_argument(
        "--checkpoint",
        default="model/deepwalk/checkpoint_latest.pt",
        help="Path to the training checkpoint (.pt) saved by preprocessor.py "
             "(default: model/deepwalk/checkpoint_latest.pt).",
    )
    parser.add_argument(
        "--output",
        default="model/deepwalk_node_embeddings.pt",
        help="Destination path for the final-embeddings file consumed by predict.py "
             "(default: model/deepwalk_node_embeddings.pt).",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    export_embeddings(checkpoint_path=args.checkpoint, output_path=args.output)
