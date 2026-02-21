
from collections import defaultdict
import csv


def read_graph():
    adj = defaultdict(list)

    with open('data/raw/train.csv', 'r') as f:
        reader = csv.reader(f)
        for row in reader:
            node, *neighbors = row
            adj[node].extend(neighbors)
    return adj

def process():
    adj = read_graph()
    for node, neighbors in adj.items():
        print(f'Node: {node}\n Neighbors: {neighbors}')
        break

if __name__ == '__main__':
    process()