import csv

import numpy as np

import os

API_SEEDS = [14098, 3421934, 340181]

class Evaluator:
    def __init__(
        self,
        api,
        dataset,
        budget_epochs,
        max_epochs=200,
        hp="200",
        seed=2026,
    ):
        self.api = api
        self.dataset = dataset
        self.budget = budget_epochs
        self.max_epochs = max_epochs
        self.hp = hp
        self.rng = np.random.default_rng(seed)
        self.reset()

    def reset(self):
        self.used_epochs = 0
        self.training_progress = {}
        self.arch_seeds = {}
        self.best_architecture = None
        self.best_validation_accuracy = -1.0
        self.best_test = None
        self.trace = []

    def api_seeds(self, idx):
        if idx not in self.arch_seeds:
            self.arch_seeds[idx] = int(self.rng.choice(API_SEEDS))
        return self.arch_seeds[idx]

    def cost(self, arch_idx, target_epochs):
        trained_epochs = self.training_progress.get(arch_idx, 0)
        return max(0, target_epochs - trained_epochs)

    def remaining_budget(self):
        return self.budget - self.used_epochs

    def can_afford(self, arch_idx, target_epochs):
        return self.cost(arch_idx, target_epochs) <= self.remaining_budget()

    def evaluate(self, arch_idx, n_epochs):
        if not 1 <= n_epochs <= self.max_epochs:
            raise ValueError(f"epochs must be in [1, {self.max_epochs}], got {n_epochs}")
        additional_epochs = self.cost(arch_idx, n_epochs)

        if additional_epochs > self.remaining_budget():
            raise RuntimeError("Budget exceeded: check can_afford() before evaluate()")

        self.used_epochs += additional_epochs

        alrdy_trained_epochs = self.training_progress.get(arch_idx, 0)
        self.training_progress[arch_idx] = max(alrdy_trained_epochs, n_epochs)

        seed = self.api_seeds(arch_idx)
        validation_accuracy = self.api.get_validation_accuracy(arch_idx, n_epochs, self.hp, seed)

        if validation_accuracy > self.best_validation_accuracy:
            self.best_validation_accuracy = validation_accuracy
            self.best_architecture = arch_idx
            self.best_test = self.api.get_test_accuracy(self, arch_idx, n_epochs, self.hp, seed)

        self.trace.append((self.used_epochs, self.best_architecture, self.best_validation_accuracy, self.best_test))

        return validation_accuracy

    def save_trace(self, filepath):
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        with open(filepath, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["epochs_used", "best_architecture",
                             "best_validation_accuracy", "best_test_accuracy"])
            writer.writerows(self.trace)