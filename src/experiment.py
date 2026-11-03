from src.search_method.bo import BONAS
from src.search_method.random_sampling import RandomSampling
from src.search_method.random_search import RandomSearch
from src.search_method.successive_halving import SuccessiveHalving

from src.eval import Evaluator

from src.nats import NATSWrapper

from datetime import datetime

timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')

budget = 100000 #epoch evals

hp = "200"

seeds = [1, 2, 3, 11, 22, 50, 30, 20, 22, 34192]

DATASETS = ['cifar10', 'cifar100', 'ImageNet16-120']

max_epochs = 200


for dataset in DATASETS:
    api = NATSWrapper(dataset_name=dataset)

    for seed in seeds:

        evaluator = Evaluator(
            api,
            dataset,
            budget_epochs=budget,
            total_epochs=max_epochs,
            training_schedule=hp,
            is_random=True, # to sim real noisy training
        )

        search_methods = [
            BONAS(
                seed=seed, 
                api=api,
                budget=budget,
                evaluator=evaluator,
                hp=hp,
                max_epochs=max_epochs,
            ), 
            RandomSearch(
                seed=seed, 
                api=api,
                budget=budget,
                evaluator=evaluator,
                hp=hp,
                max_epochs=max_epochs,
            ),
            RandomSampling(
                seed=seed, 
                api=api,
                budget=budget,
                evaluator=evaluator,
                hp=hp,
                max_epochs=max_epochs,
            ),
            SuccessiveHalving(
                seed=seed, 
                api=api,
                budget=budget,
                evaluator=evaluator,
                hp=hp,
                max_epochs=max_epochs,
            ),
            SuccessiveHalving(
                seed=seed, 
                api=api,
                budget=budget,
                evaluator=evaluator,
                hp=hp,
                max_epochs=max_epochs,
                mode="lc" # sh with learning curve power law mode
            ),
        ]

        for method in search_methods:

            method.run()

            fp = f"{timestamp}_{str(method)}.csv"

            evaluator.save_trace(fp)





