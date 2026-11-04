from src.search_method.base import BaseSearch

import numpy as np

from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern, ConstantKernel, DotProduct

from grakel import WeisfeilerLehman, EdgeHistogram

from src.encoding.encoding import arch_to_vector, wl_feature_matrix, to_one_hot



class BO(BaseSearch):
    """
    Use a specific WL kernel to 
    
    """
    def __init__(self, seed, evaluator, 
                 doe=10, exploration_factor=2, alpha=1e-2, pool_size=500):
        super().__init__(seed, evaluator)
        self.doe = doe
        self.exploration_factor = exploration_factor
        self.alpha = alpha
        self.pool_size = pool_size

        self.arch_vechs = self.get_all_archs()
        self.feats = self.build_feats()
        self.sampled_x = []
        self.obj_func_evals = []
        self.gp = GaussianProcessRegressor(
            kernel=self.build_kernel(),
            alpha=self.alpha,
            normalize_y=True,
            random_state=seed,
        )
        self.mask = np.ones(self.max_archs, dtype=bool)

    def get_all_archs(self):
        return [arch_to_vector(self.evaluator.api.get_arch_str(idx)) for idx in range(self.max_archs)]
        
    def build_feats(self):
        return np.array([to_one_hot(arch_vec) for arch_vec in self.all])

    def build_kernel(self):
        return Matern(nu=1.5) # default val from slk;learn

    def unobserved_pool(self, size):
        self.mask[self.sampled_x] = False
        candidates = np.flatnonzero(self.mask)
        return self.sample_archs(candidates=candidates, size=size, replace=False)

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
            return self.unobserved_pool(1)[0]
        pool = self.unobserved_pool(self.pool_size)
        mu, sigma = self.gp.predict(self.feats[pool], return_std=True)
        ucb_scores = mu + self.exploration_factor * sigma
        return pool[np.argmax(ucb_scores)]

    def observe(self, idx, score):
        self.sampled_x.append(idx)
        self.obj_func_evals.append(score)
        if len(self.obj_func_evals) >= self.doe:
            self.gp.fit(self.feats[self.sampled_x], self.obj_func_evals)

class NASBOWL(BO):
    def __init__(self, seed, evaluator, 
                 doe=10, exploration_factor=2, alpha=0.01, pool_size=500,
                 n_iter=2):
        super().__init__(seed, evaluator, doe, exploration_factor, alpha, pool_size)
        self.n_iter=2

    def build_feats(self):
        return wl_feature_matrix(self.arch_vechs, n_iter=self.n_iter)
    
    def build_kernel(self):
        return ConstantKernel(constant_value=1.0) * DotProduct(sigma_0_bounds="fixed")# again sklearn default, except for fixed