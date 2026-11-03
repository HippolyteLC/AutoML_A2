from base import BaseSearch

import numpy as np

from sklearn.gaussian_process import GaussianProcessRegressor

from grakel import GraphletSampling

from encoding.encoding import construct_graph


class BONAS(BaseSearch):
    """
    Use a specific WL kernel to 
    
    """

    def __init__(self, seed, api, budget, evaluator, doe=10, exploration_factor=2):
        super().__init__(seed, api, budget, evaluator)
        self.doe = doe
        self.n_candidates
        self.exploration_factor = exploration_factor

    def run(self):
        # TODO: fix the conversion from idx to graph
        # TODO: clean the code up so that less lines are used
        # TODO: move helper functions to other files
        # TODO: set a max epoch amount, or check default iepoc value in evaluator 
        # raise NotImplementedError("Should use NASBowl method to search for optimal topology.")

        observed_archs = []
        observed_performances = []
        for _ in range(self.doe):
            idx = self.rng_idx(self.seed, self.max_archs)
            observed_archs.append(idx)
            observed_performances.append(self.evaluator.evaluate(idx, 200))        

        observed_graphs = [construct_graph(arch) for arch in observed_archs]
        graphlet_kernel = GraphletSampling(k=4, normalize=False)
        k_train = graphlet_kernel.fit_transform(observed_archs)
        gp = GaussianProcessRegressor(kernel="precomputed", alpha=1e-2)
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


            

    