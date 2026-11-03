import numpy as np

class BaseSearch:
    def __init__(self, seed, api, evaluator):
        self.api = api
        self.evaluator = evaluator
        self.max_archs = api.get_size()
        self.max_epochs = self.evaluator.max_epochs
        self.rng = np.random.default_rng(seed)
        
    def __str__(self):
        return type(self).__name__

    def rng_idx(self, pop_size=1, high=None):
        if high is None:
            high = self.max_archs
        # TODO: instead, provide an rng per method so it does not need to generate a new rng every time.
        """ 
        Returns pop size shaped array of rand architectures
        """
        return self.rng.integers(low=0, high=high, size=pop_size)

    def sample_archs(self, pop_size=1, replace=False):
        """ 
        Returns pop size shaped array of rand architectures
        """        
        return self.rng.choice(self.max_archs, size=pop_size, replace=replace)

    def score(self, idx, epochs):
        """ useful for partial epoch evals """
        return self.evaluator.evaluate(idx, epochs)

    def propose(self):
        raise NotImplementedError

    def observe(self, idx, score):
        pass

    def run(self):
        while True:
            idx = int(self.propose())
            if not self.evaluator.can_afford(idx, self.max_epochs):
                break
            score = self.evaluator.evaluate(idx, self.max_epochs)
            self.observe(idx, score)
    
    