from src.search_method.base import BaseSearch

import numpy as np

from scipy.optimize import curve_fit, OptimizeWarning

import math

import warnings


warnings.filterwarnings("ignore", category=OptimizeWarning)




def pow3(x, c, a, alpha):
    return c - a * (x ** (-alpha))



class SuccessiveHalving(BaseSearch):
    def __init__(self, seed, evaluator, n_arms=1000, budget_iter=20):
        super().__init__(seed, evaluator)
        self.n_arms = n_arms
        self.n_iters = math.ceil(np.log2(self.n_arms)) 
        self.budget_iter = budget_iter

    def _compute_resource(self, current_pop_size):
        # TODO: cross-validate this implementation with the paper (jamieson et al.)
        r_k = np.floor(self.budget/(current_pop_size * np.ceil(np.log2(self.n_arms))))
        return int(min(self.max_epochs, max(1, r_k)))

    def compute_resource(self, current_pop_size, budget):
        return min(
            self.max_epochs,
            max(1, int(budget/(current_pop_size*self.n_iters)))
        )

    def compute_accum_val_loss(self, idx, resource_k):
        return self.evaluator.api.accum_val_loss(idx, self.evaluator.hp, resource_k)

    def simulate_cost(self, own_budget):
        trained = 0
        used_resources = 0
        current_pop_size = self.n_arms
        for _ in range(self.n_iters):
            resource = self.compute_resource(current_pop_size, own_budget)
            used_resources += current_pop_size * max(0, resource-trained)
            trained = max(trained, resource)
            current_pop_size = max(1, current_pop_size // 2)
        return used_resources 

    def get_own_budget(self):
        """
        method to compute own budget, so that it uses same amount of evaluations as other 
        methods (rsearch, rsampl, bo)
        binary search algorithm (log(n)))
        """
        l, h = self.evaluator.budget, self.evaluator.budget*5
        for _ in range(self.budget_iter):
            m = (l+h)/2
            if self.simulate_cost(m) <= self.evaluator.budget:
                l = m
            else:
                h = m
        return l 
    
    def run(self):
        population = self.sample_archs(pop_size=self.n_arms).tolist()
        own_budget = self.get_own_budget()
        

        for _ in range(self.n_iters):
            resource_k = self.compute_resource(len(population), own_budget)
            scores = {}
            for idx in population:
                if not self.evaluator.can_afford(idx, resource_k):
                    return 
                scores[idx] = self.score(idx, resource_k)
                
            population = sorted(population, key=lambda x: scores[x], reverse=True)
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
        return pow3(self.max_epochs, c_opt, a_opt, alpha_opt)


    def score(self, idx, epochs):
        self.evaluator.evaluate(idx, epochs)
        curve = self.evaluator.get_lc(idx, epochs)
        return self.predict_final_accuracy(curve)



def main(budget=100000, max_epochs=200, n_iters=10, n_arms=1000, budget_iter=20):
    def compute_resource(current_pop_size, budget):
        return min(
            max_epochs,
            max(1, int(budget/(current_pop_size*n_iters)))
        )   

    def simulate_cost(own_budget):
        trained = 0
        used_resources = 0
        current_pop_size = n_arms
        for _ in range(n_iters):
            resource = compute_resource(current_pop_size, own_budget)
            used_resources += current_pop_size * max(0, resource-trained)
            trained = max(trained, resource)
            current_pop_size = max(1, current_pop_size // 2)
        return used_resources 

    def get_own_budget():
        """
        method to compute own budget, so that it uses same amount of evaluations as other 
        methods (rsearch, rsampl, bo)
        binary search algorithm (log(n)))
        """
        l, h = budget, budget*5
        for _ in range(budget_iter):
            m = (l+h)/2
            if simulate_cost(m) <= budget:
                l = m
            else:
                h = m
        return l 
    return get_own_budget()

if __name__=="__main__":
    print(main())


            
