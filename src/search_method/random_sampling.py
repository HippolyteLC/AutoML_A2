import numpy as np
from base import BaseSearch
from search_method import OPS, construct_arch


class RandomSampling(BaseSearch):
    def __init__(self, seed, api, budget):
        super().__init__(seed, api, budget)

    def run(self):
        rng = np.random.default_rng(self.seed)
        for i in range(self.budget):
            arch_vec = [rng.choice(5) for _ in range(6)]
            arch_str = construct_arch(arch_vec)
            idx = self.api.get_idx(arch_str)
            self.evaluator.evaluate(idx, 200)

            
            
            
