import os
import csv
import random

import numpy as np
import matplotlib.pyplot as plt

from nats_bench import create


PATH_SIMPLE = "NATS-tss-v1_0-3ffb9-simple"
PATH_NOT_SIMPLE = "/local/.cache/torch/NATS-tss-v1_0-3ffb9.pickle.pbz2"

SEARCH_SPACES = {'tss': 'NATS-Bench topology (TSS)'} # TODO: expand, there's also 
SEARCH_SPACE = 'tss'
FAST_MODE = False
BENCH_PATH = os.environ.get('NATS_BENCH_PATH', PATH_NOT_SIMPLE)

DATASETS = ['cifar10', 'cifar100', 'ImageNet16-120']
DATASET = 'cifar10'

if not os.path.exists(BENCH_PATH):
    raise FileNotFoundError('Benchmark not found at {}. See section 1.'.format(BENCH_PATH))

api = create(BENCH_PATH, SEARCH_SPACE, fast_mode=FAST_MODE, verbose=False)

print('Search space: %s' % SEARCH_SPACES[SEARCH_SPACE])
print('Number of architectures:', len(api))