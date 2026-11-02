from grakel import GraphletSampling

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
    return adj_matrix, None, edge_labels


graphlet_kernel = GraphletSampling(k=4, normalize=False)

test_archictectures = [
    [1,1,1,2,2,2],
    [1,2,3,1,2,3],
    [0,4,4,2,3,1],
]

graphs_input = [construct_graph(a) for a in test_archictectures]

kernel = graphlet_kernel.fit_transform(graphs_input)
print(kernel)

print("Architecture 1 vs 2:", kernel[0, 1])
print("Architecture 1 vs 3:", kernel[0, 2])
print("Architecture 2 vs 3:", kernel[1, 2])