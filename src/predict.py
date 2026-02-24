
import csv
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.nn import SAGEConv
from torch_geometric.utils import negative_sampling


def load_embeddings(path='model/node_embeddings.pt'):
    ckpt = torch.load(path, map_location="cpu")
    node_embeddings = ckpt["embeddings"]
    node_id_to_idx = ckpt["node_id_to_idx"]
    return node_embeddings, node_id_to_idx

def predict_edge(u, v, node_embeddings):
    with torch.no_grad():
        score = (node_embeddings[u] * node_embeddings[v]).sum()
        prob = torch.sigmoid(score).item()
    return prob

def read_test_edges(path='data/raw/test.csv'):
    edge_list = []
    with open(path, 'r') as f:
        reader = csv.reader(f)
        next(reader)
        for row in reader:
            id, u, v = row
            edge_list.append((id, u, v))
    return edge_list

def predict_edges(edge_list, node_embeddings, node_id_to_idx):
    predictions = []
    for id, u, v in edge_list:
        u_idx = node_id_to_idx.get(u)
        v_idx = node_id_to_idx.get(v)
        if u_idx is not None and v_idx is not None:
            prob = predict_edge(u_idx, v_idx, node_embeddings)
        else:
            prob = 0.0
        predictions.append((id, prob))
    return predictions

def write_predictions(predictions, path='processed/predictions.csv'):
    with open(path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['Id', 'Predictions'])
        for id, prob in predictions:
            writer.writerow([id, 1 if prob >= 0.5 else 0])

if __name__ == "__main__":
    edges_list = read_test_edges()
    node_embeddings, node_id_to_idx = load_embeddings()
    predictions = predict_edges(edges_list, node_embeddings, node_id_to_idx)
    write_predictions(predictions)