import numpy as np
from base import BaseSearch

class RandomSearch(BaseSearch):
    def __init__(self, seed, api, budget):
        super().__init__(seed, api, budget)

    def search(self):
        for i in range(self.budget):
            idx = self.rng_idx(self.seed, self.api, 200, self.hp)
            self.evaluator.evaluate(idx, 200)
        
