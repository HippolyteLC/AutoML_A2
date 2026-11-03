import numpy as np

class BaseSearch:
    def __init__(self, seed, api, budget, evaluator, hp="200", max_epochs=200):
        self.seed = seed
        self.api = api
        self.budget = budget #set to 50k maybe? 
        self.evaluator = evaluator
        self.hp = hp
        self.max_archs = api.get_size()
        self.rng = np.random.default_rng(seed)
        self.max_epochs = max_epochs

    def rng_idx(self, pop_size=1, high=None):
        if high is None:
            high = self.max_archs
        # TODO: instead, provide an rng per method so it does not need to generate a new rng every time.
        """ 
        Returns pop size shaped array of rand architectures
        """
        return self.rng.integers(low=0, high=high, size=pop_size)
    
    def run(self):
        raise NotImplementedError("Must return history of evals, archs, and performance metrics.")

    
    