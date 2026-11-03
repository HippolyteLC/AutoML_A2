from base import BaseSearch

import numpy as np

from encoding.encoding import construct_arch


class RandomSampling(BaseSearch):
    def __init__(self, seed, api, budget, evaluator, hp="200", max_epochs=200):
        super().__init__(seed, api, budget, evaluator, hp, max_epochs)

    def run(self):
        for i in range(self.budget):
            arch_vec = [self.rng_idx()[0] for _ in range(6)]
            arch_str = construct_arch(arch_vec)
            idx = self.api.get_idx(arch_str)
            self.evaluator.evaluate(idx, self.max_epochs)

            
            
            
