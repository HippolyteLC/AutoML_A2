from search_method.bo import BONAS
from search_method.random_sampling import RandomSampling
from search_method.random_search import RandomSearch
from search_method.successive_halving import SuccessiveHalving

from eval import Evaluator

from nats import NATSWrapper

from datetime import datetime

timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')

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


    fp = f"{timestamp}_{str(method)}.csv"

    evaluator.save_trace(fp)

    # log evaluator results as json


