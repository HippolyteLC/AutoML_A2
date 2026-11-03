from base import BaseSearch

import numpy as np

from sklearn.gaussian_process import GaussianProcessRegressor

from grakel import GraphletSampling

from encoding.encoding import construct_graph


class BONAS(BaseSearch):
    """
    Use a specific WL kernel to 
    
    """
    def __init__(self, seed, api, budget, evaluator, hp="200", max_epochs=200, doe=10, exploration_factor=2, alpha=1e-2, k=4):
        super().__init__(seed, api, budget, evaluator, hp, max_epochs)
        self.doe = doe
        self.exploration_factor = exploration_factor
        self.alpha = alpha
        self.k = k # graphlet size (i.e. sub graph size)

    def __str__(self):
        return f"BONAS"

    def run(self):
        # TODO: clean the code up so that less lines are used
        # raise NotImplementedError("Should use NASBowl method to search for optimal topology.")     

        observed_archs = self.rng_idx(pop_size=self.doe)
        observed_performances = [self.evaluator.evaluate(idx, self.max_epochs) for idx in observed_archs]
        
        observed_graphs = [construct_graph(arch) for arch in observed_archs]
        graphlet_kernel = GraphletSampling(k=4, normalize=False)
        k_train = graphlet_kernel.fit_transform(observed_archs)
        gp = GaussianProcessRegressor(kernel="precomputed", alpha=self.alpha)
        gp.fit(k_train, observed_performances)

        while True:
            candidates_idxs = [
                self.rng_idx(self.seed, self.max_archs)
                for _ in range(self.max_archs)
            ]
            candidates = [
                construct_graph(idx)
                for idx in candidates_idxs
                ]
            k_candidates = graphlet_kernel.transform(candidates)
            mu, sigma = gp.predict(k_candidates, return_std=True)

            ucb_scores = mu + self.exploration_factor * sigma
            
            next_cand = np.argmax(ucb_scores)
            arch_idx = candidates_idxs[next_cand]
            if not self.evaluator.can_afford(arch_idx, self.max_epochs):
                break
            next_cand_score = self.evaluator.evaluate(arch_idx, self.max_epochs)

            observed_archs.append(arch_idx)
            observed_performances.append(next_cand_score)
            observed_graphs.append(construct_graph(arch_idx))
            k_train = graphlet_kernel.fit_transform(observed_archs)
            gp.fit(k_train, observed_performances)


            

    