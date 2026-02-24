
from collections import defaultdict
import csv
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.nn import SAGEConv
from torch_geometric.utils import negative_sampling

loss_fn = torch.nn.BCEWithLogitsLoss()

class GNN(nn.Module):
    def __init__(self, num_nodes, hidden_dim):
        super().__init__()
        self.node_emb = nn.Embedding(num_nodes, hidden_dim)
        self.conv1 = SAGEConv(hidden_dim, hidden_dim)
        self.conv2 = SAGEConv(hidden_dim, hidden_dim)

    def forward(self, edge_index):
        x = self.node_emb.weight
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = self.conv2(x, edge_index)
        return x 

def read_graph():
    adj = defaultdict(list)

    with open('data/raw/train.csv', 'r') as f:
        reader = csv.reader(f)
        for row in reader:
            node, *neighbors = row
            adj[node].extend(neighbors)
    return adj

def get_edges_list(adj):
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

def edge_score(z, edge_index):
    src, dst = edge_index
    return (z[src] * z[dst]).sum(dim=1)


def train_step(
    model,
    optimizer,
    data,
    batch_size: int = 65536,
):
    model.train()
    optimizer.zero_grad()

    # ---- 1. Sample positive edges ----
    num_edges = data.edge_index.size(1)
    perm = torch.randint(0, num_edges, (batch_size,), device=data.edge_index.device)
    pos_edge_index = data.edge_index[:, perm]

    # ---- 2. Forward pass ----
    z = model(data.edge_index)

    # Positive logits
    pos_logits = edge_score(z, pos_edge_index)
    pos_labels = torch.ones(pos_logits.size(0), device=pos_logits.device)

    # ---- 3. Negative sampling ----
    neg_edge_index = negative_sampling(
        edge_index=data.edge_index,
        num_nodes=data.num_nodes,
        num_neg_samples=pos_edge_index.size(1),
        method="sparse",
    )

    neg_logits = edge_score(z, neg_edge_index)
    neg_labels = torch.zeros(neg_logits.size(0), device=neg_logits.device)

    # ---- 4. Loss ----
    logits = torch.cat([pos_logits, neg_logits], dim=0)
    labels = torch.cat([pos_labels, neg_labels], dim=0)

    loss = loss_fn(logits, labels)
    loss.backward()

    # ---- 5. Stability ----
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    optimizer.step()

    return loss.item()

def predict_edge(u, v, node_embeddings):
    score = (node_embeddings[u] * node_embeddings[v]).sum()
    prob = torch.sigmoid(score).item()
    return prob

def save_embeddings(node_embeddings, node_id_to_idx, path='model/node_embeddings.pt'):
    torch.save(
        {
        "embeddings": node_embeddings.cpu(),
        "num_nodes": node_embeddings.size(0),
        "dim": node_embeddings.size(1),
        "node_id_to_idx": node_id_to_idx
        }, 
        path
    )

def load_embeddings(path='model/node_embeddings.pt'):
    ckpt = torch.load(path, map_location="cpu")
    node_embeddings = ckpt["embeddings"]
    return node_embeddings

def generate_embeddings(
    max_steps: int = 80_000,
    log_every: int = 1_000,
):
    print("Reading graph...")
    adj = read_graph()
    edges_list, node_id_to_idx = get_edges_list(adj)

    num_nodes = len(node_id_to_idx)
    num_edges = len(edges_list)
    print(f"Total nodes: {num_nodes}, Total edges: {num_edges}")

    # Convert to tensor safely
    edge_index = torch.as_tensor(edges_list, dtype=torch.long).t().contiguous()

    data = Data(edge_index=edge_index, num_nodes=num_nodes)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    data = data.to(device)

    model = GNN(num_nodes=num_nodes, hidden_dim=64).to(device)

    # Lower LR for stability on large graphs
    optimizer = torch.optim.Adam(model.parameters(), lr=0.005)

    print("Starting training (step-based)...")
    model.train()

    for step in range(1, max_steps + 1):
        loss = train_step(model, optimizer, data)

        if torch.isnan(torch.tensor(loss)):
            raise RuntimeError("NaN loss detected — aborting")

        if step % log_every == 0:
            print(f"Step {step}/{max_steps} | Loss: {loss:.4f}")

    print("Training completed. Generating embeddings...")

    model.eval()
    with torch.no_grad():
        node_embeddings = model(data.edge_index).cpu()

    save_embeddings(node_embeddings, node_id_to_idx)
    print("Embeddings generated and saved.")

if __name__ == '__main__':
    generate_embeddings()