import matplotlib.pyplot as plt

from typing import List, Tuple, Dict

import numpy as np

# --------------------------------------------------------------------------- #
# Global configuration
# --------------------------------------------------------------------------- #

plt.style.use(
    "seaborn-v0_8-whitegrid"
    if "seaborn-v0_8-whitegrid" in plt.style.available
    else "default"
)
plt.rcParams.update({
    "font.size": 11,
    "axes.labelsize": 12,
    "axes.titlesize": 13,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "legend.fontsize": 10,
    "pdf.fonttype": 42,   # embed TrueType fonts (required by many venues)
    "ps.fonttype": 42,
})

# Dataset display order / labels (keys are lower-cased, "_" -> "-").
DATASET_ORDER = ["cifar10", "cifar100", "imagenet16-120"]
DATASET_LABELS = {
    "cifar10": "CIFAR-10",
    "cifar100": "CIFAR-100",
    "imagenet16-120": "ImageNet16-120",
}

METRIC_COLUMNS = {
    "validation": "best_validation_accuracy",
    "test": "best_test_accuracy",
}

# Okabe-Ito-based colour-blind-safe palette. Yellow is dropped (poor contrast on
# white) and black is replaced by purple so hatching stays visible on bars.
COLORS = ["#0072B2", "#D55E00", "#009E73", "#CC79A7",
          "#E69F00", "#56B4E9", "#6A3D9A", "#999999"]
LINESTYLES = ["-", "--", "-.", ":"]
LINEWIDTHS = [2.4, 2.0, 2.2, 2.0, 2.4, 2.0, 2.2, 2.0]

# Sample std (ddof=1) is the usual choice with few seeds. Set to 0 to match
# the population std (np.std default) used in the earlier version of the script.
STD_DDOF = 1

# "RandomSample" baseline = the *first* evaluation of BASELINE_SOURCE_METHOD, aggregated
# over runs (mean and std). Anytime plots draw its mean as a horizontal line; bar charts
# draw it as a regular bar with a std whisker. Methods in EXCLUDED_METHODS are not plotted.
# (Names are matched case-insensitively against the method folder names.)
BASELINE_SOURCE_METHOD = "RandomSearch"
BASELINE_LABEL = "RandomSample"
EXCLUDED_METHODS = ("RandomSampling",)
BASELINE_STYLE = {"color": "#808080", "linestyle": (0, (6, 3)), "linewidth": 1.6}  # anytime line
BASELINE_BAR_COLOR = "#EDEDED"                                                      # bar chart

Trace = Tuple[np.ndarray, np.ndarray]                   # (epochs, incumbent)
TraceDict = Dict[str, Dict[str, List[Trace]]]           # dataset -> method -> traces

