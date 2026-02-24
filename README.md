# iisc-pml-DA218o-project-01

## Problem
We are given an adjacency list for a graph and we want to train a model which will take two nodes as input and predict if there's an edge or not. 
The graph is directed (if A follows B does not mean that B follows A).

# Preprocessing Approaches

## Approach 1
- We will first flatten the adjacency list into three column structure, where a each row indicates that there's an edge from column 1 to column 2.
- The third column will be the label column with value as 1 for all the rows generated from adjacency list. 
- We will then generate some sample data for false (negative) case with label as 0, where edge is not present between column 1 and column 2. 
- This structure should help with the model training process.

### Problems
- Raw node IDs would not work in col 1 and col 2. 
  - Since node IDs are just identifiers we don't want model to learn some inherent meaning from numeric values.
  - We might need to learn embeddings for each node (use a graph neural network maybe?)
- For negative case sampling, using random nodes might disturb the underlying graph distribution.
  - We need to come up with some algorithm to carefully generate negative samples.

## Approach 2
- First, for each node, find the top K nodes which share almost the same neighbors as current node from the given graph. 
- Then, whenever we need to check if an edge exists from node A to node B, first we check whether B exists as a neighbor in top K similar nodes. 
- If it does in most of the nodes, we predict that it could be a neighbor to A as well
- We'll use some simple similarity score formula based on common neighbors and corresponding degrees.


### Kaggle Link
https://www.kaggle.com/t/766a2f80039a48608afcdaa760a4bf73
