from src.search_method.bo import NASBOWL
from src.search_method.random_sampling import RandomSampling
from src.search_method.random_search import RandomSearch
from src.search_method.successive_halving import SuccessiveHalving, LearningCurvesSH

from src.eval import Evaluator

from src.nats import NATSWrapper

from datetime import datetime

timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')

BUDGET = 100000 #epoch evals
HP = "200"
MAX_EPOCHS = 200

SEEDS = [1, 2, 3, 11, 24, 50, 30, 20, 22, 34192]
DATASETS = ['cifar10', 'cifar100', 'ImageNet16-120']
METHODS = [NASBOWL, RandomSampling, RandomSearch, SuccessiveHalving, LearningCurvesSH]

api = NATSWrapper()
for dataset in DATASETS:
    api.update_dataset(dataset)
    for seed in SEEDS:
        evaluator = Evaluator(api, dataset, BUDGET, MAX_EPOCHS, HP, seed)
        for method in METHODS:
            evaluator.reset()
            method.run()

            fp = f"{timestamp}_{dataset}_{str(method)}_seed{seed}.csv"

            evaluator.save_trace(fp)





