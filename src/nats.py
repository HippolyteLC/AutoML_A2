import os

from src.utils import draw_cell

import numpy as np
import matplotlib.pyplot as plt

from nats_bench import create

PATH_SIMPLE = "NATS-tss-v1_0-3ffb9-simple"
PATH_LOCAL = "/local/.cache/torch/NATS-tss-v1_0-3ffb9.pickle.pbz2"

SEARCH_SPACE = 'tss'
FAST_MODE = False
BENCH_PATH = os.environ.get('NATS_BENCH_PATH', PATH_LOCAL)

class NATSWrapper:
    """
    A wrapper class for NATS-Bench to simplify querying performance, 
    learning curves, and architecture information for NAS methods.
    """
    def __init__(self, dataset_name='cifar10', mode='topology', fast_mode=FAST_MODE, path=BENCH_PATH):
        """
        Initializes the NATS-Bench API.
        
        Args:
            dataset_name (str): Dataset to use ('cifar10', 'cifar100', 'ImageNet16-120').
            mode (str): 'topology' (tss) or 'size' (sss).
            fast_mode (bool): Fast mode loads faster and uses less memory (ideal for laptops).
            path (str): Optional path to the NATS-Bench .pth file if not in the default directory.
        """
        self.dataset_name = dataset_name
        self.mode = mode
        
        print(f"Loading NATS-Bench ({mode}) for dataset '{dataset_name}' (fast_mode={fast_mode})...")
        
        self.api = create(path, mode, fast_mode=fast_mode, verbose=False)
        self.max_archs = len(self.api)

    def get_size(self):
        """Returns the total number of architectures in the search space."""
        return self.max_archs

    def get_idx(self, arch_str):
        return self.api.query_index_by_arch(arch_str)
    
    def get_arch_str(self, idx, verbose=False):
        arch_str = self.api.arch(idx)
        if verbose:
            print('Architecture 0:', arch_str)
            print('Round-trip index:', self.api.query_index_by_arch(arch_str))
        return arch_str
    
    def evaluate_arch(self, arch_index, iepoch=None, hp=12, is_random=False, use_eval=True):
        """
        Queries the performance of a specific architecture.
        
        Args:
            arch_index (int): Index of the architecture.
            iepoch (int, optional): Epoch up to which to evaluate (useful for Successive Halving / Learning Curves).
            use_eval (bool): Whether to use test/validation performance.
            
        Returns:
            dict: Performance metrics (accuracy, loss, latency, etc.)
        """
        if self.dataset_name == "cifar10":
            name = self.dataset_name if not use_eval else f"{self.dataset_name}-valid"
        else:
            name = self.dataset_name
        info = self.api.get_more_info(
            index=arch_index, 
            dataset=name, 
            iepoch=iepoch,
            hp=hp,
            is_random=is_random,
            )
        cost_info = self.get_cost(arch_index=arch_index, hp=hp)
        info["latency"] = cost_info[""]
        return info | cost_info

    def learning_curve(self, idx, hp_schedule, n_epochs):
        vals, tests = [], []
        for e in range(200):
            name = 'cifar10-valid' if self.dataset_name == 'cifar10' else self.dataset_name
            vals.append(self.api.get_more_info(idx, name, hp=hp_schedule, iepoch=e, is_random=False)['valid-accuracy'])
            tests.append(self.api.get_more_info(idx, self.dataset_name, hp=hp_schedule, iepoch=e, is_random=False)['test-accuracy'])
            
        return np.array(vals)[:n_epochs], np.array(tests)[:n_epochs]

    def accum_val_loss(self, idx, hp_schedule, n_epochs):
        return np.cumsum(self.learning_curve(idx, hp_schedule, n_epochs)[0])

    def get_cost(self, arch_index, hp):
        """Returns training time or FLOPs info for budget tracking."""
        cost_info = self.api.get_cost_info(arch_index, self.dataset_name, hp=hp)
        return cost_info

    def draw_arch(self, idx, save=False):
        fig, ax = plt.subplots(figsize=(6, 3))
        arch_str = self.get_arch_str(idx)
        draw_cell(arch_str, ax=ax, title='Architecture %d: %s' % (idx, arch_str))
        if save:
            plt.savefig('Architecture %d' % (idx))
        plt.show()
        # self.api.show(idx)

if __name__=="__main__":
    pass
    # DATASETS = ['cifar10', 'cifar100', 'ImageNet16-120']
# DATASET = 'cifar10'

# if not os.path.exists(BENCH_PATH):
#     raise FileNotFoundError('Benchmark not found at {}. See section 1.'.format(BENCH_PATH))

# api = create(BENCH_PATH, SEARCH_SPACE, fast_mode=FAST_MODE, verbose=False)

# print('Search space: %s' % SEARCH_SPACES[SEARCH_SPACE])
# print('Number of architectures:', len(api))