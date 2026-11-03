from src.search_method.base import BaseSearch

import numpy as np

from sklearn.gaussian_process import GaussianProcessRegressor

from grakel import WeisfeilerLehman, EdgeHistogram

from src.encoding.encoding import idx_to_graph


class NASBOWL(BaseSearch):
    """
    Use a specific WL kernel to 
    
    """
    def __init__(self, seed, evaluator, 
                 doe=10, exploration_factor=2, alpha=1e-2, n_iter=3, pool_size=500):
        super().__init__(seed, evaluator)
        self.doe = doe
        self.exploration_factor = exploration_factor
        self.alpha = alpha
        self.pool_size = pool_size
        self.n_iter = n_iter # WL parameter

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
        X_train = np.array([[idx_to_graph(idx)] for idx in pool], dtype=object)
        mu, sigma = self.gp.predict(X_train, return_std=True)
        ucb_scores = mu + self.exploration_factor * sigma
        return X_train[np.argmax(ucb_scores)]

    def observe(self, idx, score):
        self.sampled_x.append(idx)
        self.obj_func_evals.append(score)
        if len(self.obj_func_evals) >= self.doe:
            X_train = np.array([[idx_to_graph(idx)]], dtype=object)
            self.gp.fit([idx_to_graph(X_train)], self.obj_func_evals)

        