import numpy as np
from src.search_method.base import BaseSearch

class RandomSearch(BaseSearch):
    def __init__(self, seed, api, budget, evaluator):
        super().__init__(seed, api, budget, evaluator)

    def propose(self):
        return self.sample_archs(1, replace=True)[0]
