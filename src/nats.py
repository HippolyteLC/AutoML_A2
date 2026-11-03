import os

from src.utils import draw_cell

import numpy as np
import matplotlib.pyplot as plt

from nats_bench import create

PATH_SIMPLE = "NATS-tss-v1_0-3ffb9-simple"
PATH_LOCAL = "/local/.cache/torch/NATS-tss-v1_0-3ffb9.pickle.pbz2"

SEARCH_SPACE = 'tss'
FAST_MODE = False
if FAST_MODE:
    PATH_LOCAL = PATH_SIMPLE
BENCH_PATH = os.environ.get('NATS_BENCH_PATH', PATH_LOCAL)

class NATSWrapper:
    """
    A wrapper class for NATS-Bench to simplify querying performance, 
    learning curves, and architecture information for NAS methods.
    """
    def __init__(self, dataset_name='cifar10', mode='topology', fast_mode=FAST_MODE, path=BENCH_PATH):
        self.dataset_name = dataset_name
        self.max_archs = len(self.api)

        print(f"Loading NATS-Bench ({mode}) for dataset '{dataset_name}' (fast_mode={fast_mode})...")
        self.api = create(path, mode, fast_mode=fast_mode, verbose=False)

    def update_dataset(self, ds_name):
        self.dataset_name = ds_name

    def get_size(self):
        """Returns the total number of architectures in the search space."""
        return self.max_archs

    def valid_split_name(self):
        if self.dataset_name == "cifar10":
            return f"{self.dataset_name}-valid"
        return self.dataset_name

    def get_validation_accuracy(self, idx, n_epochs, hp, is_random):
        return self.api.get_more_info(
            idx, self.valid_split_name(), iepoch=n_epochs - 1, hp=hp, is_random=is_random,
        )["valid-accuracy"]
    
    def get_test_accuracy(self, idx, max_epochs, hp, is_random):
        return self.api.get_more_info(
            idx, self.dataset_name, iepoch=max_epochs-1, hp=hp, is_random=is_random,
        )["test-accuracy"]

    def get_idx(self, arch_str):
        return self.api.query_index_by_arch(arch_str)
    
    def get_arch_str(self, idx, verbose=False):
        arch_str = self.api.arch(idx)
        if verbose:
            print('Architecture 0:', arch_str)
            print('Round-trip index:', self.api.query_index_by_arch(arch_str))
        return arch_str

    def learning_curve(self, idx, n_epochs, hp_schedule, is_random):
        vals = []
        for e in range(n_epochs):
            vals.append(self.get_validation_accuracy(
                idx, e+1, hp_schedule, is_random
            ))
        return np.array(vals)

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