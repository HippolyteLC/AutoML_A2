from src.search_method.base import BaseSearch

import numpy as np

from scipy.optimize import curve_fit

import math


def pow3(x, c, a, alpha):
    return c - a * (x ** (-alpha))


class SuccessiveHalving(BaseSearch):
    def __init__(self, seed, api, evaluator, n_arms=1000):
        super().__init__(seed, api, evaluator)
        self.n_arms = n_arms
        self.n_iters = math.ceil(np.log2(self.n_arms)) 

    def compute_resource(self, current_pop_size):
        r_k = np.floor(self.budget/(current_pop_size * np.ceil(np.log2(self.n_arms))))
        return int(min(self.max_epochs, max(1, r_k)))

    def compute_accum_val_loss(self, idx, resource_k):
        return self.api.accum_val_loss(idx, self.evaluator.hp, resource_k)

    def run(self):
        population = self.sample_archs(pop_size=self.n_arms)

        for k in range(self.n_iters):
            resource_k = self.compute_resource(len(population))
            scores = {self.score(idx, resource_k) for idx in population}
            population = sorted(population, key=scores.get, reverse=True)

            cut_off_point = max(1, int(len(population)//2))
            population = population[:cut_off_point]


class LearningCurvesSH(SuccessiveHalving):
    def predict_final_accuracy(self, val_curve):
        x = np.arange(1, len(val_curve)+1)
        ymax, ymin = val_curve.max(), val_curve.min()

        p0 = [ymax, max(ymax-ymin, 1e-2), 0.5] # use the 1e-2 fallback in case the val curve is stagnant
        bounds = [ymax, 0.0, 0.0], [100, np.inf, np.inf]
        
        try:
            parameters_optimized, _ = curve_fit(pow3, x, val_curve, p0=p0, bounds=bounds)   
        except Exception:
            return val_curve[-1] 
        c_opt, a_opt, alpha_opt = parameters_optimized
        return pow3(x, c_opt, a_opt, alpha_opt)

    def score(self, idx, epochs):
        self.evaluator.evaluate(idx, epochs)
        seed = self.evaluator.arch_seeds[idx]
        curve = self.api.learning_curve(idx, epochs, hp_schedule=self.evaluator.hp, is_random=seed)
        return self.predict_final_accuracy(curve)


        
        


            
