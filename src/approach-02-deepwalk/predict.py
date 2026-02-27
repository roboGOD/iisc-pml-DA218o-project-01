import csv
import torch


def load_embeddings(path="model/deepwalk_node_embeddings.pt"):
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    src_embeddings = ckpt.get("src_embeddings")
    dst_embeddings = ckpt.get("dst_embeddings")
    merged_embeddings = ckpt.get("embeddings")

    if src_embeddings is None or dst_embeddings is None:
        if merged_embeddings is None:
            raise ValueError("No embeddings found in checkpoint")
        src_embeddings = merged_embeddings
        dst_embeddings = merged_embeddings

    node_id_to_idx = ckpt["node_id_to_idx"]
    best_threshold = float(ckpt.get("best_threshold", 0.5))
    metrics = ckpt.get("metrics", {})

    print(f"Loaded source embeddings: {tuple(src_embeddings.shape)}")
    print(f"Loaded target embeddings: {tuple(dst_embeddings.shape)}")
    print(f"Learned threshold: {best_threshold:.4f}")

    if metrics:
        auc = metrics.get("auc")
        ap = metrics.get("ap")
        f1 = metrics.get("f1")
        auc_s = f"{auc:.4f}" if isinstance(auc, (float, int)) else "N/A"
        ap_s = f"{ap:.4f}" if isinstance(ap, (float, int)) else "N/A"
        f1_s = f"{f1:.4f}" if isinstance(f1, (float, int)) else "N/A"
        print(f"Validation metrics — AUC: {auc_s}, AP: {ap_s}, F1: {f1_s}")

    return src_embeddings, dst_embeddings, node_id_to_idx, best_threshold


def predict_edge(u_idx, v_idx, src_embeddings, dst_embeddings):
    with torch.no_grad():
        score = (src_embeddings[u_idx] * dst_embeddings[v_idx]).sum()
        prob = torch.sigmoid(score).item()
    return prob


def read_test_edges(path="data/raw/test.csv"):
    edge_list = []
    with open(path, "r") as f:
        reader = csv.reader(f)
        next(reader)
        for row in reader:
            if not row:
                continue
            id_, u, v = row
            edge_list.append((id_, u, v))
    return edge_list


def predict_edges(edge_list, src_embeddings, dst_embeddings, node_id_to_idx):
    predictions = []
    missing_count = 0

    for id_, u, v in edge_list:
        u_idx = node_id_to_idx.get(u)
        v_idx = node_id_to_idx.get(v)

        if u_idx is None or v_idx is None:
            prob = 0.0
            missing_count += 1
        else:
            prob = predict_edge(u_idx, v_idx, src_embeddings, dst_embeddings)

        predictions.append((id_, prob))

    if missing_count > 0:
        print(
            f"Warning: {missing_count}/{len(edge_list)} edges include unknown nodes (predicted as 0)."
        )

    return predictions


def write_predictions(predictions, threshold, path="data/processed/deepwalk_predictions.csv"):
    positive_count = 0

    with open(path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["Id", "Predictions"])
        for id_, prob in predictions:
            pred = 1 if prob >= threshold else 0
            positive_count += pred
            writer.writerow([id_, pred])

    print(f"Predictions written to {path}")
    print(
        f"Total: {len(predictions)}, Positive: {positive_count}, "
        f"Negative: {len(predictions) - positive_count}"
    )


if __name__ == "__main__":
    test_edges = read_test_edges()
    src_emb, dst_emb, node_id_to_idx, threshold = load_embeddings()
    print(f"Using learned threshold: {threshold:.4f}")
    preds = predict_edges(test_edges, src_emb, dst_emb, node_id_to_idx)
    write_predictions(preds, threshold)
