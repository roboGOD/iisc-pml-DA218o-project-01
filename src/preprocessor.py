
from collections import defaultdict
import csv
import math


def read_graph():
    adj = defaultdict(list)

    with open('data/raw/train.csv', 'r') as f:
        reader = csv.reader(f)
        for row in reader:
            node, *neighbors = row
            adj[node].extend(neighbors)
    return adj


def build_reverse_adj(adj):
    rev = defaultdict(list)
    for u, vs in adj.items():
        for v in vs:
            rev[v].append(u)
    return rev


def similarity(u, v, adj):
    """
    Weighted similarity between nodes u and v
    based on shared outgoing neighbors.
    """
    Nu = set(adj[u])
    Nv = set(adj[v])
    common = Nu & Nv

    score = 0.0
    for w in common:
        deg = len(adj[w])
        if deg > 1:
            score += 1.0 / math.log(deg)
    return score

def compute_global_top_k(adj, K=10):
    nodes = list(adj.keys())
    topk = defaultdict(list)

    print('Computing global top-K similarities...')
    for u in nodes:
        scores = []
        total = len(nodes) ** 2
        for v in nodes:
            count_completed = (nodes.index(u) * len(nodes)) + nodes.index(v) + 1
            print(f'Percentage: {100 * count_completed / total:.4f}%', end='\r')
            if u == v:
                continue
            sim = similarity(u, v, adj)
            if sim > 0:
                scores.append((sim, v))

        scores.sort(reverse=True)
        topk[u] = scores[:K]

    return topk

def preprocess():
    adj = read_graph()
    topk = compute_global_top_k(adj, K=10)
    with open('data/processed/topk.csv', 'w', newline='') as f:
        writer = csv.writer(f)
        for u, neighbors in topk.items():
            row = [u] + [v for _, v in neighbors]
            writer.writerow(row)    

if __name__ == '__main__':
    preprocess()