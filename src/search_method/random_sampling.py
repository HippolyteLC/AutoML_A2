from src.search_method.base import BaseSearch

import numpy as np

from src.encoding.encoding import construct_arch


class RandomSampling(BaseSearch):
    def __init__(self, seed, evaluator):
        super().__init__(seed, evaluator)

    def propose(self):
        operations = self.rng.integers(0, 5, size=6).tolist()
        assert len([op for op in operations if op < 0]) == 0, "adding invalid operation"
        return self.evaluator.api.get_idx(construct_arch(operations))
    

if __name__=="__main__":
    rng = np.random.default_rng(11)
    
    s = rng.integers(0, 5, size=6).tolist()
    print(s)