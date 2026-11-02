import numpy as np

from search_method import api 

class BaseSearch:
    def __init__(self, seed, api, budget):
        self.seed = seed
        self.api = api
        self.budget = budget

    def run(self):
        raise NotImplementedError("Must return history of evals, archs, and performance metrics.")
    