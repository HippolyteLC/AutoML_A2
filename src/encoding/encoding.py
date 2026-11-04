import re

# from grakel import Graph

import numpy as np

from src.utils import parse_arch

from collections import Counter

OPS = ["none", "skip_connect", "nor_conv_1x1", "nor_conv_3x3", "avg_pool_3x3"]
OPS_DICT = {op: i for i, op in enumerate(OPS)}
EDGES = [
        (0, 1),
        (0, 2),
        (1, 2),
        (0, 3),
        (1, 3),
        (2, 3),
    ]

def construct_arch(arch_vec):
    vec = [OPS[i] for i in arch_vec]
    return (
        f"|{vec[0]}~0|"
        f"|{vec[1]}~0|{vec[2]}~1|"
        f"|{vec[3]}~0|{vec[4]}~1|{vec[5]}~2|"
    )  

def construct_graph(arch_vec):
    raise NotImplementedError("Grakel is not suited to the problem. ")
    adj_matrix = np.zeros((4, 4))
    for src, dst in EDGES:
        adj_matrix[src, dst] = 1
    edge_labels = {
        edge: arch_vec[idx] for
        idx, edge in enumerate(EDGES)
    }
    return Graph(adj_matrix, edge_labels=edge_labels, graph_format="adjacency")

def to_one_hot(arch_vec):
    encoding = np.zeros(6 * len(OPS))
    for position, op_id in enumerate(arch_vec):
        encoding[position * len(OPS) + op_id] = 1
    return encoding

def edge_to_node_labeled_dag(arch_vec) -> tuple[dict[int, list[int]]]:
    """
    Edge to node labeling!!!

    Converting the arch vector to tuple of labels (in and out labels added, total 8 nodes),
    parent nodes: using nested for loop with idx shift to account for new parent ("in"),
    children: finds the children of parent nodes. The two latter are dicts of [int, lists]. 
    
    """
    labels = ["in"] + [OPS[op] for op in arch_vec] + ["out"]
    n_labs = len(labels)
    parent_nodes = {node: [] for node in range(n_labs)}
    for i, (src_i, dst_i) in enumerate(EDGES, start=1):
        if src_i == 0:
            parent_nodes[i].append(0)
        if dst_i == 3:
            parent_nodes[n_labs-1].append(3)
        for j, (src_j, _) in enumerate(EDGES, start=1):
            if src_j == dst_i:
                parent_nodes[j].append(i)

    children = {node: [] for node in range(n_labs)}
    for node, node_parents in parent_nodes.items():
        for parent in node_parents:
            children[parent].append(node)
    return labels, parent_nodes, children


def wl_counts(arch_vec: list[int], vocab: dict, n_iter: int):
    """
    arch_vec: arch vector as ints
    vocab: input as empty dict, use setdefault to update dict as we go
    
    function counts the order and occurence of the label vocab. 
    assigns child and parent nodes to each node inside the arch. Ensures order matters
    so that different label orderings aren't duplicates (with sorted()).
    
    For the num of iterations it adds distinct signatures to the vocab, updates the 
    labels and repeats. Final counts of signatures are returned 
    """
    labels, parents, children = edge_to_node_labeled_dag(arch_vec)
    labels = [vocab.setdefault(label, len(vocab)) for label in labels]
    counts = Counter(labels)
    for _ in range(n_iter):
        new_labels = []
        for node in range(len(labels)):
            signature = (
                labels[node],
                tuple(sorted(labels[p] for p in parents[node])),
                tuple(sorted(labels[c] for c in children[node])),
            )
            new_labels.append(vocab.setdefault(signature, len(vocab)))
        labels = new_labels
        counts.update(labels)
    return counts


def wl_feature_matrix(arch_vecs, n_iter=2):
    """
    outputs normalized 
    """
    vocab = {}

    all_counts = [wl_counts(vec, vocab, n_iter) for vec in arch_vecs]
    features = np.zeros((len(arch_vecs), len(vocab)), dtype=np.float32)
    for row, counts in enumerate(all_counts):
        for label, count in counts.items():
            features[row, label] = count
    norms = np.linalg.norm(features, axis=1, keepdims=True)
    return features / norms


def arch_to_vector(arch_str, verbose=False):
    if verbose:
        print(parse_arch(arch_str))
    return [OPS_DICT[op] for _,_, op in parse_arch(arch_str) if len(op) > 0]


def idx_to_graph(idx):
    return construct_graph(arch_to_vector(idx))

def test_arch_round_trip():
    arch_vec = [1, 2, 3, 4, 0, 1]

    arch_str = construct_arch(arch_vec)
    recovered_vec = arch_to_vector(arch_str, verbose=True)

    assert recovered_vec == arch_vec, (
        f"Round trip failed!\n"
        f"Original:  {arch_vec}\n"
        f"Recovered: {recovered_vec}\n"
        f"Architecture: {arch_str}"
    )

    print("Round trip successful!")
    print("Architecture:", arch_str)
    print("Vector:", recovered_vec)


if __name__ == "__main__":
    test_arch_round_trip()