from search_method.bo import BONAS
from search_method.random_sampling import RandomSampling
from search_method.random_search import RandomSearch
from search_method.successive_halving import SuccessiveHalving

from eval import Evaluator

from nats import NATSWrapper

budget = 100000 #epoch evals

hp = "200"

seeds = [1, 2, 3, 11, 22, 50, 30, 20, 22, 34192]

max_epochs = 200

api = NATSWrapper()

evaluator = Evaluator()

search_methods = [
    BONAS, 
    RandomSearch,
    RandomSampling,
    SuccessiveHalving,
]

for method in search_methods:
    method.run()

    # log evaluator results as json


