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


# +++++++++++++++
# This is the results plotting code
# ++++++++++++++++++?

import os
import re
import glob
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d


plt.style.use(
    'seaborn-v0_8-whitegrid'
    if 'seaborn-v0_8-whitegrid' in plt.style.available
    else 'default'
)

plt.rcParams.update({
    'font.size': 11,
    'axes.labelsize': 12,
    'axes.titlesize': 13,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'legend.fontsize': 10,
    'figure.titlesize': 14
})


def parse_logs(results_dir: str) -> pd.DataFrame:
    results_path = Path(results_dir)

    if not results_path.is_dir():
        raise FileNotFoundError(
            f"Results directory does not exist: {results_dir}"
        )

    files = glob.glob(
        os.path.join(str(results_path), "**", "*.csv"),
        recursive=True
    )

    if not files:
        raise FileNotFoundError(
            f"No CSV files found in: {results_dir}"
        )

    all_records = []

    required_cols = {
        'epochs_used',
        'best_architecture',
        'best_validation_accuracy',
        'best_test_accuracy'
    }

    for file_path in files:
        path_obj = Path(file_path)

        try:
            rel_path = path_obj.relative_to(results_path)
            parts = rel_path.parts

            if len(parts) < 3:
                continue

            dataset = parts[0]
            search_method = parts[1]
            filename = path_obj.stem

            seed_match = re.search(
                r'seed[_-]?(\d+)',
                filename,
                flags=re.IGNORECASE
            )

            if seed_match:
                seed = int(seed_match.group(1))
            else:
                seed = 0

            df_run = pd.read_csv(file_path)

            if not required_cols.issubset(df_run.columns):
                print(f"Warning: Missing required columns in {file_path}")
                continue

            df_run = df_run.copy()

            df_run['epochs_used'] = pd.to_numeric(
                df_run['epochs_used'],
                errors='coerce'
            )

            df_run['best_validation_accuracy'] = pd.to_numeric(
                df_run['best_validation_accuracy'],
                errors='coerce'
            )

            df_run['best_test_accuracy'] = pd.to_numeric(
                df_run['best_test_accuracy'],
                errors='coerce'
            )

            df_run = df_run.dropna(
                subset=[
                    'epochs_used',
                    'best_validation_accuracy',
                    'best_test_accuracy'
                ]
            )

            if df_run.empty:
                print(f"Warning: No valid records in {file_path}")
                continue

            df_run['dataset'] = dataset
            df_run['search_method'] = search_method
            df_run['seed'] = seed

            all_records.append(df_run)

        except Exception as e:
            print(f"Error parsing {file_path}: {e}")

    if not all_records:
        raise ValueError(
            "No valid experiment log records were loaded."
        )

    return pd.concat(all_records, ignore_index=True)


def process_and_aggregate_data(
    df: pd.DataFrame,
    target_accuracy: float = 1.0,
    num_grid_points: int = 100
):
    required_cols = {
        'dataset',
        'search_method',
        'seed',
        'epochs_used',
        'best_validation_accuracy',
        'best_test_accuracy'
    }

    if not required_cols.issubset(df.columns):
        raise ValueError("Input DataFrame is missing required columns.")

    if num_grid_points < 2:
        raise ValueError("num_grid_points must be at least 2.")

    if not np.isfinite(target_accuracy):
        raise ValueError("target_accuracy must be finite.")

    datasets = df['dataset'].dropna().unique()
    aggregated_curves = {}
    summary_metrics = []

    for dataset in datasets:

        ds_subset = df[df['dataset'] == dataset].copy()

        max_epoch = ds_subset['epochs_used'].max()

        if not np.isfinite(max_epoch) or max_epoch <= 0:
            continue

        common_epochs = np.linspace(
            0,
            max_epoch,
            num_grid_points
        )

        aggregated_curves[dataset] = {}

        for method in ds_subset['search_method'].dropna().unique():

            method_subset = ds_subset[
                ds_subset['search_method'] == method
            ]

            interpolated_val_curves = []
            final_val_accs = []
            final_test_accs = []
            aocc_list = []

            for seed in method_subset['seed'].dropna().unique():

                run_df = method_subset[
                    method_subset['seed'] == seed
                ].copy()

                run_df = run_df.sort_values('epochs_used')

                run_df = run_df.groupby(
                    'epochs_used',
                    as_index=False
                ).last()

                epochs = run_df['epochs_used'].to_numpy(
                    dtype=float
                )

                val_acc = run_df[
                    'best_validation_accuracy'
                ].to_numpy(dtype=float)

                test_acc = run_df[
                    'best_test_accuracy'
                ].to_numpy(dtype=float)

                valid = (
                    np.isfinite(epochs)
                    & np.isfinite(val_acc)
                    & np.isfinite(test_acc)
                )

                epochs = epochs[valid]
                val_acc = val_acc[valid]
                test_acc = test_acc[valid]

                if len(epochs) == 0:
                    continue

                val_incumbent = np.maximum.accumulate(val_acc)
                test_incumbent = np.maximum.accumulate(test_acc)

                final_val_accs.append(val_incumbent[-1])
                final_test_accs.append(test_incumbent[-1])

                t_acc = max(
                    target_accuracy,
                    np.max(val_incumbent)
                )

                gap = t_acc - val_incumbent

                if len(epochs) > 1:
                    epoch_span = epochs[-1] - epochs[0]

                    aocc = np.trapezoid(
                        gap,
                        x=epochs
                    ) / epoch_span if epoch_span > 0 else 0.0
                else:
                    aocc = 0.0

                aocc_list.append(aocc)

                if len(epochs) == 1:
                    interp_vals = np.full(
                        num_grid_points,
                        val_incumbent[0]
                    )
                else:
                    f_interp = interp1d(
                        epochs,
                        val_incumbent,
                        kind='linear',
                        bounds_error=False,
                        fill_value=(
                            val_incumbent[0],
                            val_incumbent[-1]
                        ),
                        assume_sorted=True
                    )

                    interp_vals = f_interp(common_epochs)

                interpolated_val_curves.append(interp_vals)

            if not interpolated_val_curves:
                continue

            interpolated_val_curves = np.asarray(
                interpolated_val_curves
            )

            mean_curve = np.mean(
                interpolated_val_curves,
                axis=0
            )

            std_curve = np.std(
                interpolated_val_curves,
                axis=0
            )

            aggregated_curves[dataset][method] = {
                'epochs': common_epochs,
                'mean': mean_curve,
                'std': std_curve
            }

            summary_metrics.append({
                'dataset': dataset,
                'search_method': method,
                'num_seeds': len(final_val_accs),
                'final_val_acc_mean': np.mean(final_val_accs),
                'final_val_acc_std': np.std(final_val_accs),
                'final_test_acc_mean': np.mean(final_test_accs),
                'final_test_acc_std': np.std(final_test_accs),
                'aocc_mean': np.mean(aocc_list),
                'aocc_std': np.std(aocc_list)
            })

    summary_df = pd.DataFrame(summary_metrics)

    return aggregated_curves, summary_df


def plot_anytime_performance(
    aggregated_curves: dict,
    output_path: str = "anytime_performance_plots.pdf"
):
    datasets = list(aggregated_curves.keys())

    if not datasets:
        print("No datasets available for plotting.")
        return

    preferred = ['cifar10', 'cifar100']
    datasets = (
        [ds for name in preferred for ds in datasets if str(ds).lower() == name]
        + [ds for ds in datasets if str(ds).lower() not in preferred]
    )[:3]

    fig, axes = plt.subplots(
        1, 3, figsize=(18, 5), squeeze=False
    )
    axes = axes[0]

    methods = sorted({
        method for ds in datasets
        for method in aggregated_curves[ds]
    })

    colors = plt.cm.tab10(np.linspace(0, 1, max(len(methods), 1)))
    color_map = dict(zip(methods, colors))
    legend_handles = {}

    for i, dataset in enumerate(datasets):
        ax = axes[i]

        for method, data in aggregated_curves[dataset].items():
            epochs = np.asarray(data['epochs'], dtype=float)
            mean = np.asarray(data['mean'], dtype=float)
            std = np.asarray(data['std'], dtype=float)

            valid = np.isfinite(epochs) & np.isfinite(mean) & np.isfinite(std)
            epochs, mean, std = epochs[valid], mean[valid], std[valid]

            if not len(epochs):
                continue

            if np.max(mean) > 1:
                mean, std = mean / 100, std / 100

            color = color_map[method]

            line, = ax.plot(
                epochs, mean,
                color=color, linewidth=1.2, label=method
            )

            ax.fill_between(
                epochs, mean - std, mean + std,
                color=color, alpha=0.15
            )

            legend_handles.setdefault(method, line)

        ax.set_title(dataset.upper(), fontweight='normal')
        ax.set_xlabel("Epochs")
        ax.set_ylabel("Validation Accuracy")
        ax.grid(True, linestyle='--', alpha=0.6)

    for ax in axes[len(datasets):]:
        ax.set_visible(False)

    if legend_handles:
        fig.legend(
            legend_handles.values(),
            legend_handles.keys(),
            loc='center',
            bbox_to_anchor=(0.5, 0.96),
            ncol=len(legend_handles),
            mode=None,
            frameon=True,
            fontsize=10,
            handlelength=2,
            columnspacing=1.5
        )

    fig.tight_layout(rect=[0, 0, 1, 0.88])
    fig.savefig(output_path, dpi=300, bbox_inches='tight', format="pdf")
    print(f"Visualization saved to '{output_path}'.")
    plt.show()
    plt.close(fig)




def main():

    RESULTS_DIR = "results"
    SUMMARY_CSV_OUTPUT = "experiment_summary_metrics.csv"
    PLOT_OUTPUT_PATH = "anytime_performance_plots.pdf"

    print(
        f"1. Parsing experiment logs from '{RESULTS_DIR}'..."
    )

    try:
        df_raw = parse_logs(RESULTS_DIR)

    except Exception as e:
        print(f"Error: {e}")
        return

    print(
        "2. Processing data and aggregating across random seeds..."
    )

    try:
        aggregated_curves, summary_df = process_and_aggregate_data(
            df_raw
        )

    except Exception as e:
        print(f"Error processing experiment data: {e}")
        return

    if summary_df.empty:
        print("No valid experiment results available.")
        return

    print(
        f"3. Exporting summary metrics to '{SUMMARY_CSV_OUTPUT}'..."
    )

    summary_df.to_csv(
        SUMMARY_CSV_OUTPUT,
        index=False
    )

    print(summary_df.to_string(index=False))

    print(
        "4. Generating anytime performance visualization..."
    )

    plot_anytime_performance(
        aggregated_curves,
        output_path=PLOT_OUTPUT_PATH
    )

    print("Pipeline completed successfully!")


if __name__ == "__main__":
    main()
