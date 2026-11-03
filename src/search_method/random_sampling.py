from src.search_method.base import BaseSearch

import numpy as np

from src.encoding.encoding import construct_arch


class RandomSampling(BaseSearch):
    def __init__(self, seed, api, budget, evaluator):
        super().__init__(seed, api, budget, evaluator)

    def propose(self):
        operations = self.rng.integers(0, 5, size=6).tolist()
        return self.api.get_idx(construct_arch(operations))
    

