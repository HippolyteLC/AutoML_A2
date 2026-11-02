import numpy as np
from base import BaseSearch

class RandomSearch(BaseSearch):
    def __init__(self, seed, budget):
        super().__init__(seed, budget)

    def search(seed, hp_schedule, n_evals, dataset):
        records = []
        rng = np.random.default_rng(seed)
        current_best_val = float('-inf')
        current_best_arch = None
        for _ in range(n_evals):
            idx = rng.choice(15625)
            name = 'cifar10-valid' if dataset == 'cifar10' else dataset
            info = self.api.get_more_info(idx, name, hp=hp_schedule, is_random=True)
            # note that CIFAR10 has some weird key issues
            # use a debugger to get insights in the proper keys
            val_acc = info['valid-accuracy']
            if val_acc > current_best_val:
                current_best_arch = idx
            records.append((current_best_arch, val_acc))

            # info = api.get_more_info(idx, name, hp=hp_schedule, is_random=False)

            # train_cost = api.get_cost_info(arch_idx, DATASET, hp=200)
            
        return records