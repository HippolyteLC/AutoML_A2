import re

from grakel import Graph

import numpy as np

from src.utils import parse_arch

OPS = ["none", "skip_connect", "nor_conv_1x1", "nor_conv_3x3", "avg_pool_3x3"]
OPS_DICT = {op: i for i, op in enumerate(OPS)}

def construct_arch(arch_vec):
    vec = [OPS[i] for i in arch_vec]
    return (
        f"|{vec[0]}~0|"
        f"|{vec[1]}~0|{vec[2]}~1|"
        f"|{vec[3]}~0|{vec[4]}~1|{vec[5]}~2|"
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

def arch_to_vector(arch_str):
    return [OPS_DICT[op] for _,_, op in parse_arch(arch_str)]

def idx_to_graph(idx):
    return construct_graph(arch_to_vector(idx))