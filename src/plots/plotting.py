"""
Plotting and analysis utilities for NAS / HPO experiments.

Pipeline
--------
    parse_logs()                    -> long-format DataFrame (one row per logged step)
    extract_incumbent_traces()      -> per-run best-so-far step functions
    plot_anytime_curves()           -> Output 1: log-x anytime curves + RandomSample line (PDF/PNG)
    compute_efficiency_csv()        -> Output 2: final validation accuracy & std summary (CSV)

Expected input DataFrame columns (this is what `parse_logs` produces):
    dataset, search_method, seed, epochs_used,
    best_validation_accuracy, best_test_accuracy

Accuracies may be given in [0, 1] or in [0, 100]; everything is converted to
percent internally.
"""

from __future__ import annotations

import argparse
import datetime as dt
import glob
import os
import re
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D

from src.plots.config import *
from src.plots.utils import _order_datasets, make_run_dir,_ensure_parent
from src.plots.utils import _dataset_label, _std, build_method_styles


# --------------------------------------------------------------------------- #
# Data loading and preparation
# --------------------------------------------------------------------------- #

def parse_logs(results_dir: str) -> pd.DataFrame:
    """
    Load every CSV under  <results_dir>/<dataset>/<search_method>/**/<...seedN...>.csv
    into one long-format DataFrame.
    """
    results_path = Path(results_dir)
    if not results_path.is_dir():
        raise FileNotFoundError(f"Results directory does not exist: {results_dir}")

    files = glob.glob(os.path.join(str(results_path), "**", "*.csv"), recursive=True)
    if not files:
        raise FileNotFoundError(f"No CSV files found in: {results_dir}")

    metric_cols = list(METRIC_COLUMNS.values())
    numeric = ["epochs_used"] + metric_cols
    records = []

    for file_path in files:
        parts = Path(file_path).relative_to(results_path).parts
        if len(parts) < 3:                      # need dataset/method/file.csv
            continue
        dataset, method = parts[0], parts[1]

        seed_match = re.search(r"seed[_-]?(\d+)", Path(file_path).stem, flags=re.IGNORECASE)
        seed = int(seed_match.group(1)) if seed_match else 0

        try:
            run = pd.read_csv(file_path)
        except Exception as exc:                # noqa: BLE001 - report and move on
            print(f"Error parsing {file_path}: {exc}")
            continue

        if "epochs_used" not in run.columns or not set(metric_cols) & set(run.columns):
            print(f"Warning: missing required columns in {file_path}")
            continue

        # A metric column may be absent or partly empty (None/NaN), e.g. the test
        # accuracy of Successive Halving / Learning Curves. Keep those rows: each
        # metric is cleaned separately later, so validation data is never lost
        # just because the test value is missing.
        for col in metric_cols:
            if col not in run.columns:
                run[col] = np.nan
        run[numeric] = run[numeric].apply(pd.to_numeric, errors="coerce")
        run = run.dropna(subset=["epochs_used"])
        run = run[run[metric_cols].notna().any(axis=1)]   # drop rows empty for every metric
        if run.empty:
            print(f"Warning: no valid records in {file_path}")
            continue

        run = run.assign(dataset=dataset, search_method=method, seed=seed)
        records.append(run)

    if not records:
        raise ValueError("No valid experiment log records were loaded.")
    df = pd.concat(records, ignore_index=True)
    _report_missing(df)
    return df


def _report_missing(df: pd.DataFrame) -> None:
    """Print which (dataset, method) pairs have missing metric values."""
    for col in METRIC_COLUMNS.values():
        frac = df[col].isna().groupby([df["dataset"], df["search_method"]]).mean()
        frac = frac[frac > 0]
        for (dataset, method), f in frac.items():
            kind = "entirely missing" if f >= 1 else f"{f:.0%} of rows missing"
            print(f"Note: {col} is {kind} for {dataset}/{method}")


def _to_percent(df: pd.DataFrame) -> pd.DataFrame:
    """Return a copy with accuracy columns on a 0-100 scale (auto-detected)."""
    df = df.copy()
    for col in METRIC_COLUMNS.values():
        if df[col].max() <= 1.0:
            df[col] = df[col] * 100.0
    return df


def extract_incumbent_traces(
    df: pd.DataFrame,
    metric: str = "validation",
    budget: Optional[float] = 100_000,
    exclude_methods: Sequence[str] = (),
) -> TraceDict:
    """
    Turn the raw logs into one best-so-far step function per run.
    Methods listed in `exclude_methods` (case-insensitive) are skipped.

    Returns {dataset: {method: [(epochs, incumbent), ...one per seed...]}}.
    Records beyond `budget` are dropped; accuracies are in percent.
    """
    if metric not in METRIC_COLUMNS:
        raise ValueError(f"metric must be one of {list(METRIC_COLUMNS)}")
    col = METRIC_COLUMNS[metric]

    needed = {"dataset", "search_method", "seed", "epochs_used", col}
    if not needed.issubset(df.columns):
        raise ValueError(f"Input DataFrame is missing columns: {needed - set(df.columns)}")

    df = _to_percent(df)
    excluded = {m.lower() for m in exclude_methods}
    traces: TraceDict = {}

    for (dataset, method, _seed), run in df.groupby(["dataset", "search_method", "seed"]):
        if str(method).lower() in excluded:
            continue
        # Rows where *this* metric is missing (None/NaN) are skipped for this metric
        # only; a method with no values at all simply has no trace for it.
        run = run.dropna(subset=[col]).sort_values("epochs_used")
        run = run.drop_duplicates("epochs_used", keep="last")
        if budget is not None:
            run = run[run["epochs_used"] <= budget]
        if run.empty:
            continue

        epochs = run["epochs_used"].to_numpy(dtype=float)
        incumbent = np.maximum.accumulate(run[col].to_numpy(dtype=float))
        traces.setdefault(dataset, {}).setdefault(method, []).append((epochs, incumbent))

    if not traces:
        raise ValueError("No traces could be extracted (check budget / input data).")
    return traces


def evaluate_on_grid(trace: Trace, grid: np.ndarray) -> np.ndarray:
    """
    Evaluate an incumbent step function on `grid`.

    Points before the run's first record are NaN; points after its last record
    carry the final incumbent forward.
    """
    epochs, values = trace
    pos = np.searchsorted(epochs, grid, side="right") - 1
    out = np.full(grid.shape, np.nan)
    has_data = pos >= 0
    out[has_data] = values[pos[has_data]]
    return out


def _mean_curve(traces: List[Trace], grid: np.ndarray) -> np.ndarray:
    """
    Mean incumbent across runs on `grid`.

    Grid points where any run has no data yet are set to NaN, so the mean is
    never computed over a changing subset of runs (which would bias the curve).
    """
    mean, _ = _mean_and_std_curve(traces, grid)
    return mean


def _mean_and_std_curve(traces: List[Trace], grid: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Mean and standard deviation of the incumbent across runs on `grid`.

    Grid points where any run has no data yet are set to NaN, so statistics are
    never computed over a changing subset of runs (which would bias the curve).
    """
    stack = np.vstack([evaluate_on_grid(t, grid) for t in traces])
    missing = np.isnan(stack).any(axis=0)
    mean = stack.mean(axis=0)
    if stack.shape[0] > STD_DDOF:
        std = np.std(stack, axis=0, ddof=STD_DDOF)
    else:
        std = np.zeros_like(mean)
    mean[missing] = np.nan
    std[missing] = np.nan
    return mean, std


def compute_baselines(
    traces: TraceDict,
    source_method: str = BASELINE_SOURCE_METHOD,
) -> Dict[str, Tuple[float, float]]:
    """
    Per dataset: (mean, std) over runs of the first logged incumbent of `source_method`.
    Datasets where that method has no data are simply absent from the result.
    """
    baselines = {}
    for dataset, by_method in traces.items():
        key = next((m for m in by_method if m.lower() == source_method.lower()), None)
        if key is not None:
            first_values = [values[0] for _, values in by_method[key]]
            baselines[dataset] = (float(np.mean(first_values)), _std(first_values))
    return baselines


def _final_scores(traces: List[Trace]) -> np.ndarray:
    """Final incumbent of every run."""
    return np.array([values[-1] for _, values in traces], dtype=float)



# --------------------------------------------------------------------------- #
# Function 1: log-x anytime curves with linear y-axis
# --------------------------------------------------------------------------- #

def plot_anytime_curves(
    df: pd.DataFrame,
    output_path: str = "anytime_performance.pdf",
    metric: str = "validation",
    budget: float = 100_000,
    num_grid_points: int = 400,
    method_styles: Optional[Dict[str, dict]] = None,
    exclude_methods: Sequence[str] = EXCLUDED_METHODS,
    show: bool = False,
) -> plt.Figure:
    """
    One subplot per dataset showing the mean +/- std incumbent vs. epochs on a log-x
    axis (linear y-axis). The RandomSample baseline (first RandomSearch evaluation) is
    drawn as a horizontal line.
    """
    # Baselines come from the unfiltered traces; plotted methods exclude `exclude_methods`.
    baselines = compute_baselines(extract_incumbent_traces(df, metric=metric, budget=budget))
    traces = extract_incumbent_traces(df, metric=metric, budget=budget,
                                      exclude_methods=exclude_methods)
    datasets = _order_datasets(list(traces))
    methods = sorted({m for ds in datasets for m in traces[ds]})
    styles = build_method_styles(methods, method_styles)

    fig, axes = plt.subplots(1, len(datasets), figsize=(5.2 * len(datasets), 4.2),
                             squeeze=False)

    for ax, dataset in zip(axes[0], datasets):
        all_traces = [t for ts in traces[dataset].values() for t in ts]
        # A log axis cannot show epoch 0, so start at the earliest logged epoch (>= 1).
        lo = max(1.0, min(t[0][0] for t in all_traces))
        hi = max(t[0][-1] for t in all_traces)
        grid = np.logspace(np.log10(lo), np.log10(hi), num_grid_points)

        first_valid_x = hi
        for method in methods:
            if method not in traces[dataset]:
                continue
            curve, std = _mean_and_std_curve(traces[dataset][method], grid)
            color = styles[method]["color"]
            ax.plot(grid, curve, label=method, **styles[method])
            ax.fill_between(grid, curve - std, curve + std, color=color,
                            alpha=0.18, linewidth=0)
            finite = np.flatnonzero(np.isfinite(curve))
            if len(finite):
                first_valid_x = min(first_valid_x, grid[finite[0]])

        ax.set_xscale("log")
        # Start the axis where the earliest mean curve starts (the mean is only
        # defined once every seed has logged a record), not at epoch 1.
        ax.set_xlim(first_valid_x, hi)
        if dataset in baselines:                 # horizontal RandomSample baseline
            ax.axhline(baselines[dataset][0], zorder=1.5, **BASELINE_STYLE)

        ax.set_title(_dataset_label(dataset))
        ax.set_xlabel("Epochs (log scale)")
        ax.set_ylabel(f"{metric.capitalize()} accuracy (%)")
        ax.grid(True, which="major", linestyle="--", alpha=0.6)
        ax.grid(True, which="minor", linestyle=":", alpha=0.25)

    handles = [Line2D([0], [0], label=m, **styles[m]) for m in methods]
    if baselines:
        handles.append(Line2D([0], [0], label=BASELINE_LABEL, **BASELINE_STYLE))
    legend_cols = min(len(handles), 5)
    legend_rows = int(np.ceil(len(handles) / legend_cols))
    fig.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, 1.02),
               ncol=legend_cols, frameon=True, handlelength=3.2,
               columnspacing=1.5)
    fig.tight_layout(rect=[0, 0, 1, 1 - 0.02 - 0.06 * legend_rows])  # room for legend rows

    _ensure_parent(output_path)
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    print(f"Saved anytime plot to '{output_path}'.")
    if show:
        plt.show()
    plt.close(fig)
    return fig


# --------------------------------------------------------------------------- #
# Function 2: final validation accuracy & std summary CSV
# --------------------------------------------------------------------------- #

def compute_efficiency_csv(
    df: pd.DataFrame,
    output_path: str = "efficiency_summary.csv",
    metric: str = "validation",
    budget: float = 100_000,
    exclude_methods: Sequence[str] = EXCLUDED_METHODS,
    baseline_source_method: str = BASELINE_SOURCE_METHOD,
) -> pd.DataFrame:
    """
    For every (dataset, method): output final accuracy mean and std across runs.
    Also includes `RandomSample` (aggregated across seeds from the first logged
    sample of `baseline_source_method`).
    """
    unfiltered_traces = extract_incumbent_traces(df, metric=metric, budget=budget)
    baselines = compute_baselines(unfiltered_traces, source_method=baseline_source_method)
    traces = extract_incumbent_traces(df, metric=metric, budget=budget,
                                      exclude_methods=exclude_methods)

    all_datasets = _order_datasets(list(set(traces) | set(baselines)))
    rows = []

    for dataset in all_datasets:
        if dataset in baselines:
            base_mean, base_std = baselines[dataset]
            base_key = next(
                (m for m in unfiltered_traces.get(dataset, {})
                 if m.lower() == baseline_source_method.lower()),
                None,
            )
            num_base_runs = len(unfiltered_traces[dataset][base_key]) if base_key else 0
            rows.append({
                "dataset": _dataset_label(dataset),
                "method": BASELINE_LABEL,
                "metric": metric,
                "num_runs": num_base_runs,
                "final_mean": round(float(base_mean), 4),
                "final_std": round(float(base_std), 4),
                "final_performance": f"{base_mean:.2f} ± {base_std:.2f}",
            })

        for method in sorted(traces.get(dataset, {})):
            runs = traces[dataset][method]
            finals = _final_scores(runs)
            final_mean, final_std = float(finals.mean()), _std(finals)

            rows.append({
                "dataset": _dataset_label(dataset),
                "method": method,
                "metric": metric,
                "num_runs": len(runs),
                "final_mean": round(final_mean, 4),
                "final_std": round(final_std, 4),
                "final_performance": f"{final_mean:.2f} ± {final_std:.2f}",
            })

    out = pd.DataFrame(rows)
    # utf-8-sig so Excel displays the "±" correctly
    _ensure_parent(output_path)
    out.to_csv(output_path, index=False, encoding="utf-8-sig")
    print(f"Saved efficiency summary to '{output_path}'.")
    return out


# --------------------------------------------------------------------------- #
# Entry point
# --------------------------------------------------------------------------- #

def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--results-dir", default="results")
    parser.add_argument("--out-dir", default="figures")
    parser.add_argument("--budget", type=float, default=100_000)
    args = parser.parse_args()

    # Every run writes all its outputs into its own  <out_dir>/<YYYYmmdd_HHMMSS>/  folder.
    out_dir = make_run_dir(args.out_dir)
    print(f"Writing outputs to '{out_dir}'.")

    df = parse_logs(args.results_dir)

    plot_anytime_curves(df, str(out_dir / "anytime_performance.pdf"), budget=args.budget)
    plot_anytime_curves(df, str(out_dir / "anytime_performance.png"), budget=args.budget)

    summary = compute_efficiency_csv(df, str(out_dir / "efficiency_summary.csv"),
                                     budget=args.budget)
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()