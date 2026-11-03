import re

from grakel import Graph

import numpy as np

OPS = ["none", "skip_connect", "nor_conv_1x1", "nor_conv_3x3", "avg_pool_3x3"]
OPS_DICT = {op: i for i, op in enumerate(OPS)}

def construct_arch(arch_vec):
    return (
        f"|{arch_vec[0]}~0|"
        f"|{arch_vec[1]}~0|{arch_vec[2]}~1|"
        f"|{arch_vec[3]}~0|{arch_vec[4]}~1|{arch_vec[5]}~2|"
    )  

def construct_graph(arch_vec):
    edges = [
        (0, 1),
        (0, 2),
        (1, 2),
        (0, 3),
        (1, 3),
        (2, 3),
    ]
    adj_matrix = np.zeros((4, 4))
    for src, dst in edges:
        adj_matrix[src, dst] = 1
    edge_labels = {
        edge: arch_vec[idx] for
        idx, edge in enumerate(edges)
    }
    return Graph(adj_matrix, edge_labels=edge_labels, graph_format="adjacency")

def idx_to_vector(arch_str):
    groups = arch_str.strip("|").split("+")
    def clean_str(str):
        return re.sub(r'[~ℤℕℤ]', '', str)
    incoming_edges = [clean_str(i) for i in groups]
    return [OPS_DICT[edge] for edge in incoming_edges]

def idx_to_graph(idx):
    return construct_graph(idx_to_vector(idx))