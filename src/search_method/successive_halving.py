from base import BaseSearch

import numpy as np

import math

class SuccessiveHalving(BaseSearch):
    def __init__(self, seed, api, budget, evaluator, hp="200", n_arms=1000):
        super().__init__(seed, api, budget, evaluator, hp)
        self.n_arms = n_arms

    def compute_resource(self, current_pop_size):
        return np.floor(self.budget/(current_pop_size * np.ceil(np.log2(self.n_arms))))
    
    def run(self):
        n_iters = math.ceil(np.log2(self.n_arms)) 

        # initialize_candidates for some r amount of epochs
        population = self.rng_idx(pop_size=self.n_arms)

        for k in range(n_iters):
            resource_k = self.compute_resource(len(population))

            cut_off_point = int(np.floor(len(population)/2))
            _ = [self.evaluator.evaluate(idx) for idx in population]
            population = sorted(
                population, 
                key=lambda idx: self.api.accum_val_loss(idx, self.hp, resource_k), 
                reverse=True
            )[:cut_off_point]

        
        


            
