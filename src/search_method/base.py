import numpy as np

class BaseSearch:
    def __init__(self, seed, api, budget, evaluator, hp="200"):
        self.seed = seed
        self.api = api
        self.budget = budget
        self.evaluator = evaluator
        self.hp = hp
        self.max_archs = api.get_size()

    @staticmethod
    def rng_idx(seed, max_archs):
        # TODO: instead, provide an rng per method so it does not need to generate a new rng every time.
        rng = np.random.default_rng(seed=seed)
        idx = rng.choice(max_archs)
        return idx
    
    def run(self):
        raise NotImplementedError("Must return history of evals, archs, and performance metrics.")

    
    