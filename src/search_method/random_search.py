import numpy as np
from src.search_method.base import BaseSearch

class RandomSearch(BaseSearch):
    def __init__(self, seed, api, budget, evaluator, hp, max_epochs):
        super().__init__(seed, api, budget, evaluator, hp, max_epochs)
    def __str__(self):
        return f"RandomSearch"
    def search(self):
        for _ in range(self.budget):
            idx = self.rng_idx(1)[0]
            self.evaluator.evaluate(idx, self.max_epochs)

        
