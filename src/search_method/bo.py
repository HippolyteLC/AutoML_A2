from base import BaseSearch
import networkx as nx

import re

class NASBowl(BaseSearch):
    """
    Use a specific WL kernel to 
    
    """
    def __init__(self, seed, nats_api, budget):
        super().__init__(seed, nats_api, budget)

    def run(self):
        raise NotImplementedError("Should use NASBowl method to search for optimal topology.")

        G = nx.DiGraph()

    @staticmethod
    def parse_arch_str(arch_str):
        groups = arch_str.strip("|").split("+")
        result = []
        for group in groups:
            tokens = re.findall(r'([^|]+)~(\d+)', group)
            result.append([(int(idx), op) for op, idx in tokens])

        return result

    def to_graph(self, idx):
        arch_str = self.nats_api.get_arch_str(idx)

        G, nodes, idx = nx.DiGraph(), [0], 1

        for part in arch_str.strip("|").split("|+"):
            if not part: continue
            layer_nodes = []
            for edge in part.strip("|").split("|"):
                op, src = edge.split("~")
                src = int(src)
                dst = idx
                G.add_edge(nodes[src], dst, operation=op)
                layer_nodes.append(dst)
            nodes.append(idx)
            idx += 1

    def to_idx(graph):
        pass
s = "|nor_conv_1x1~0|+|none~0|nor_conv_1x1~1|+|none~0|nor_conv_1x1~1|nor_conv_1x1~2|"

NASBowl.parse_arch_str(s)