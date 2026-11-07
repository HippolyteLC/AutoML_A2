from src.search_method.bo import BO, NASBOWL
from src.search_method.random_sampling import RandomSampling
from src.search_method.random_search import RandomSearch
from src.search_method.successive_halving import SuccessiveHalving, LearningCurvesSH

from src.eval import Evaluator

from src.nats import NATSWrapper

from datetime import datetime

timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')

BUDGET = 3000 #epoch evals
HP = "200"
MAX_EPOCHS = 200

SEEDS = [1, 2, 3, 11, 24, 50, 30, 20, 22, 34192]
DATASETS = ['cifar10', 'cifar100', 'ImageNet16-120']
METHODS = [BO, NASBOWL, RandomSampling, RandomSearch, SuccessiveHalving, LearningCurvesSH]

def main():
    api = NATSWrapper()
    for dataset in DATASETS[1:]:
        api.update_dataset(dataset)
        for seed in SEEDS[:2]:
            evaluator = Evaluator(api, dataset, BUDGET, MAX_EPOCHS, HP, seed)
            for method_class in METHODS[4:5]:
                evaluator.reset()
                search_method = method_class(seed, evaluator)
                search_method.run()
                fp = f"results/{dataset}/{search_method}_seed{seed}.csv"
                # evaluator.save_trace(fp)
                print(dataset, seed, search_method, evaluator.best_test)

if __name__=="__main__":
    main()


