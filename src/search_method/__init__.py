OPS = ["none", "skip_connect", "nor_conv_1x1", "nor_conv_3x3", "avg_pool_3x3"]

def construct_arch(arch_vec):
    arch_str = (
        f"|{arch_vec[0]}~0|"
        f"|{arch_vec[1]}~0|{arch_vec[2]}~1|"
        f"|{arch_vec[3]}~0|{arch_vec[4]}~1|{arch_vec[5]}~2|"
    )  

def construct_graph(arch_vec):
    adj_matrix = [
        (0, 1),
        (0, 2),
        (1, 2),
        (0, 3),
        (1, 3),
        (2, 3),
    ]
    edge_labels = {
        edge: arch_vec[idx] for
        idx, edge in enumerate(adj_matrix)
    }
    return adj_matrix, edge_labels
