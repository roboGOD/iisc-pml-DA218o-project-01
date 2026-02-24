
from collections import defaultdict
import csv
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.nn import SAGEConv
from torch_geometric.utils import negative_sampling

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

def train(model, optimizer, data):
    model.train()
    optimizer.zero_grad()

    z = model(data.edge_index)

    # Positive edges
    pos_edge_index = data.edge_index
    pos_score = edge_score(z, pos_edge_index)
    pos_loss = -torch.log(torch.sigmoid(pos_score)).mean()

    # Negative edges
    neg_edge_index = negative_sampling(
        edge_index=data.edge_index,
        num_nodes=data.num_nodes,
        num_neg_samples=pos_edge_index.size(1)
    )
    neg_score = edge_score(z, neg_edge_index)
    neg_loss = -torch.log(1 - torch.sigmoid(neg_score)).mean()

    loss = pos_loss + neg_loss
    loss.backward()
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

def generate_embeddings():
    print("Reading graph...")
    adj = read_graph()
    edges_list, node_id_to_idx = get_edges_list(adj)

    num_nodes = len(node_id_to_idx)
    print(f"Total nodes: {len(node_id_to_idx)}, Total edges: {len(edges_list)}")

    edge_index = torch.tensor(edges_list, dtype=torch.long).t().contiguous()
    
    data = Data(edge_index=edge_index, num_nodes=num_nodes)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = GNN(num_nodes=num_nodes, hidden_dim=64).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    data = data.to(device)

    print("Starting training...")
    for epoch in range(5):
        loss = train(model, optimizer, data)
        print(f'Epoch {epoch}, Loss: {loss:.4f}')
    print("Training completed. Generating embeddings...")

    model.eval()
    with torch.no_grad():
        node_embeddings = model(data.edge_index)
    save_embeddings(node_embeddings, node_id_to_idx)
    print("Embeddings generated and saved.")

if __name__ == '__main__':
    generate_embeddings()