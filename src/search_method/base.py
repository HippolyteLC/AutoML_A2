import numpy as np

class BaseSearch:
    def __init__(self, seed, nats_api, budget):
        self.seed = seed
        self.nats_api = nats_api
        self.budget = budget

    def run(self):
        raise NotImplementedError("Must return history of evals, archs, and performance metrics.")
    