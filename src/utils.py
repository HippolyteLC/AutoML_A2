from matplotlib.patches import Circle, FancyArrowPatch

import matplotlib.pyplot as plt

import pandas as pd

import numpy as np

import glob


OP_COLORS = {
    'nor_conv_3x3': '#1f77b4',
    'nor_conv_1x1': '#2ca02c',
    'avg_pool_3x3': '#ff7f0e',
    'skip_connect': '#7f7f7f',
    'none': '#bbbbbb',
}


def anytime_perf_plot(all_runs):
    """
    all_runs: a dict of seed: incumbent arch list
    """
    for seed, records in all_runs.items():
        accuracies = [
            acc for idx, acc in records
        ]
        plt.plot(np.maximum.accumulate(accuracies), alpha=0.5)

    plt.xlabel('Number of evaluations')
    plt.ylabel('Best validation accuracy')
    plt.title('Anytime Performance of Random Search')
    plt.grid(alpha=0.3)
    plt.show()


def get_incumbent_on_grid(trace, grid):
    positions = np.searchsorted(trace["epochs_used"].values, grid, side="right") - 1
    values = np.full(len(grid), np.nan)
    has_data = positions >= 0
    values[has_data] = trace["best_test_accuracy"].values[positions[has_data]]
    return values


def plot_anytime(dataset, methods, budget=100000):
    grid = np.linspace(0, budget, 500)
    for method in methods:
        curves = []
        for path in glob.glob(f"results/{dataset}/{method}_seed*.csv"):
            curves.append(get_incumbent_on_grid(pd.read_csv(path), grid))
        curves = np.array(curves)
        mean = np.nanmean(curves, axis=0)
        std = np.nanstd(curves, axis=0)
        plt.plot(grid, mean, label=method)
        plt.fill_between(grid, mean - std, mean + std, alpha=0.2)

    plt.xlabel("Epochs used")
    plt.ylabel("Test accuracy of incumbent (%)")
    plt.title(dataset)
    plt.legend()
    plt.grid(alpha=0.3)
    plt.show()


def parse_arch(arch_str):
    """Turn '|op~0|+|op~0|op~1|+|op~0|op~1|op~2|' into a list of (src, dst, op)."""
    edges = []
    for dst, group in enumerate(arch_str.split('+'), start=1):
        items = group.strip().strip('|').split('|')
        for src, item in enumerate(items):
            edges.append((src, dst, item.split('~')[0]))
    return edges


def draw_cell(arch_str, ax=None, title=None):
    if ax is None:
        _, ax = plt.subplots(figsize=(6, 3))
    for src, dst, op in parse_arch(arch_str):
        color = OP_COLORS.get(op, 'black')
        arrow = FancyArrowPatch((src, 0), (dst, 0),
                                connectionstyle='arc3,rad=-0.6',  # arcs above the baseline
                                arrowstyle='-|>', mutation_scale=14,
                                color=color,
                                linewidth=1 if op == 'none' else 2,
                                linestyle=':' if op == 'none' else '-',
                                shrinkA=10, shrinkB=10)
        ax.add_patch(arrow)
        # With rad=-0.6 the arc peaks at 0.3 * span above the baseline.
        ax.text((src + dst) / 2, 0.3 * (dst - src) + 0.12, op,
                ha='center', va='bottom', fontsize=8, color=color)
    for node in range(4):
        ax.add_patch(Circle((node, 0), 0.12, fc='white', ec='black', zorder=3))
        name = 'input' if node == 0 else ('output' if node == 3 else 'node {}'.format(node))
        ax.text(node, -0.25, name, ha='center', va='top', fontsize=9)
    ax.set_xlim(-0.5, 3.5)
    ax.set_ylim(-0.6, 1.3)
    ax.axis('off')
    if title:
        ax.set_title(title, fontsize=10)
    return ax