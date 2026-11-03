from base import BaseSearch

import numpy as np

from scipy.optimize import curve_fit

import math

class SuccessiveHalving(BaseSearch):
    def __init__(self, seed, api, budget, evaluator, hp="200", n_arms=1000, mode="sh"):
        super().__init__(seed, api, budget, evaluator, hp)
        self.n_arms = n_arms
        self.mode = mode

    def compute_resource(self, current_pop_size):
        return np.floor(self.budget/(current_pop_size * np.ceil(np.log2(self.n_arms))))
    
    def pow3(self, x, c, a, alpha):
        return c - a * (x ** (-alpha))

    def predict_lc(self, idx, n_epochs):
        val_accuracies, _ = self.api.learning_curve(idx, self.hp, n_epochs)
        ymax, ymin = np.max(val_accuracies), np.min(val_accuracies)
        xmax, xmin = np.max(n_epochs), np.min(n_epochs)

        c, a, alpha = ymax, ymax-ymin, (xmax-xmin)/xmax
        p0 = [c, a, alpha]

        low_bounds, up_bounds = [ymax, 0.0, 0.0], [1, np.inf, np.inf]
        parameters_optimized, _ = curve_fit(
                self.pow3, 
                n_epochs, 
                val_accuracies, 
                p0=p0, 
                bounds=(low_bounds, up_bounds)
        )   
        c_opt, a_opt, alpha_opt = parameters_optimized
        return self.pow3(self.max_epochs, c_opt, a_opt, alpha_opt)

    def compute_accum_val_loss(self, idx, resource_k):
        return self.api.accum_val_loss(idx, self.hp, resource_k)

    def run(self):
        n_iters = math.ceil(np.log2(self.n_arms)) 

        # initialize_candidates for some r amount of epochs
        population = self.rng_idx(pop_size=self.n_arms)

        for k in range(n_iters):
            resource_k = self.compute_resource(len(population))

            cut_off_point = int(np.floor(len(population)/2))
            _ = [self.evaluator.evaluate(idx, resource_k) for idx in population]
            if self.mode == "sh":
                objective_func = self.compute_accum_val_loss
            elif self.mode == "lc":
                objective_func = self.predict_lc
            
            population = sorted(
                population, 
                key=lambda idx: objective_func(idx, resource_k), 
                reverse=True
            )[:cut_off_point]



        
        


            
