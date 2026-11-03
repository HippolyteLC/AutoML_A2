from src.search_method.base import BaseSearch

import numpy as np

from sklearn.gaussian_process import GaussianProcessRegressor

from grakel import WeisfeilerLehman, EdgeHistogram

from src.encoding.encoding import idx_to_graph


class BONAS(BaseSearch):
    """
    Use a specific WL kernel to 
    
    """
    def __init__(self, seed, api, budget, evaluator, 
                 doe=10, exploration_factor=2, alpha=1e-2, n_iter=3, pool_size=500):
        super().__init__(seed, api, budget, evaluator)
        self.doe = doe
        self.exploration_factor = exploration_factor
        self.alpha = alpha
        self.pool_size = pool_size
        self.n_iter = n_iter # graphlet size (i.e. sub graph size)

        self.sampled_x = []
        self.obj_func_evals = []
        self.gp = GaussianProcessRegressor(
            kernel=self.WLKernel,
            alpha=self.alpha,
        )

    @staticmethod
    def WLKernel(X, Y=None):
        wl_kernel = WeisfeilerLehman(n_iter=3, base_graph_kernel=EdgeHistogram)
        graphs = [g[0] for g in X]
        if Y is None:
            return wl_kernel.fit_transform(graphs)
        else:
            graphs_y = [g[0] for g in Y]
            wl_kernel.fit(graphs)
            return wl_kernel.transform(graphs_y)

    def propose(self):
        if len(self.obj_func_evals) < self.doe:
            return self.sample_archs(1)
        pool = self.sample_archs(self.pool_size)
        X_train = np.array([[idx_to_graph(vec)] for vec in pool], dtype=object)
        mu, sigma = self.gp.predict(X_train, return_std=True)
        ucb_scores = mu + self.exploration_factor * sigma
        return X_train[np.argmax(ucb_scores)]

    def observe(self, idx, score):
        self.sampled_x.append(self.)

    def run(self):
        # TODO: clean the code up so that less lines are used
        # raise NotImplementedError("Should use NASBowl method to search for optimal topology.")     

        observed_archs = self.rng_idx(pop_size=self.doe)
        observed_performances = [self.evaluator.evaluate(idx, self.max_epochs) for idx in observed_archs]
        
        observed_graphs = [idx_to_graph(idx) for idx in observed_archs]
        wl_kernel = WeisfeilerLehman(n_iter=3, base_graph_kernel=EdgeHistogram)
        k_train = wl_kernel.fit_transform(observed_graphs)
        gp = GaussianProcessRegressor(kernel="precomputed", alpha=self.alpha)
        gp.fit(k_train, observed_performances)

        while True:
            candidates_idxs = self.rng.choice(
                [idx for idx in range(self.max_archs) if idx not in set(observed_archs)],
                size=self.pool_size,
                replace=False
            )
            
            [
                self.rng_idx(self.seed)
                for _ in range(self.max_archs)
            ]
            candidates = [
                idx_to_graph(idx)
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
            observed_graphs.append(idx_to_graph(arch_idx))
            k_train = graphlet_kernel.fit_transform(observed_archs)
            gp.fit(k_train, observed_performances)


            

    