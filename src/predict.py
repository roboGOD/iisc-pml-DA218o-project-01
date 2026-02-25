
import csv
import torch
import torch.nn.functional as F


def load_embeddings(path='model/node_embeddings.pt'):
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    node_embeddings = ckpt["embeddings"]
    node_id_to_idx = ckpt["node_id_to_idx"]
    best_threshold = ckpt.get("best_threshold", 0.5)
    metrics = ckpt.get("metrics", {})
    print(f"Loaded embeddings: {node_embeddings.shape}")
    print(f"Learned threshold: {best_threshold:.4f}")
    if metrics:
        print(f"Validation metrics — AUC: {metrics.get('auc', 'N/A'):.4f}, "
              f"AP: {metrics.get('ap', 'N/A'):.4f}, "
              f"F1: {metrics.get('f1', 'N/A'):.4f}")
    return node_embeddings, node_id_to_idx, best_threshold


def predict_edge(u, v, node_embeddings):
    with torch.no_grad():
        score = (node_embeddings[u] * node_embeddings[v]).sum()
        prob = torch.sigmoid(score).item()
    return prob


def read_test_edges(path='data/raw/test.csv'):
    edge_list = []
    with open(path, 'r') as f:
        reader = csv.reader(f)
        next(reader)  # skip header
        for row in reader:
            id_, u, v = row
            edge_list.append((id_, u, v))
    return edge_list


def predict_edges(edge_list, node_embeddings, node_id_to_idx):
    predictions = []
    missing_count = 0
    for id_, u, v in edge_list:
        u_idx = node_id_to_idx.get(u)
        v_idx = node_id_to_idx.get(v)
        if u_idx is not None and v_idx is not None:
            prob = predict_edge(u_idx, v_idx, node_embeddings)
        else:
            prob = 0.0
            missing_count += 1
        predictions.append((id_, prob))
    if missing_count > 0:
        print(f"Warning: {missing_count}/{len(edge_list)} edges had unknown nodes "
              "(predicted as 0)")
    return predictions


def write_predictions(predictions, threshold, path='data/processed/predictions.csv'):
    with open(path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['Id', 'Predictions'])
        pos_count = 0
        for id_, prob in predictions:
            pred = 1 if prob >= threshold else 0
            pos_count += pred
            writer.writerow([id_, pred])
    print(f"Predictions written to {path}")
    print(f"  Total: {len(predictions)}, Positive: {pos_count}, "
          f"Negative: {len(predictions) - pos_count}")


if __name__ == "__main__":
    edges_list = read_test_edges()
    node_embeddings, node_id_to_idx, threshold = load_embeddings()
    print(f"Using learned threshold: {threshold:.4f}")
    predictions = predict_edges(edges_list, node_embeddings, node_id_to_idx)
    write_predictions(predictions, threshold)