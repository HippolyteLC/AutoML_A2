"""
Plotting and analysis utilities for NAS / HPO experiments.

Pipeline
--------
    parse_logs()                    -> long-format DataFrame (one row per logged step)
    extract_incumbent_traces()      -> per-run best-so-far step functions
    plot_anytime_curves()           -> Output 1: log-x anytime curves (PDF)
    plot_final_performance_bars()   -> Output 2: final-score bars with std whiskers (PDF)
    compute_efficiency_csv()        -> Output 3: epochs-to-95%-of-final table (CSV)

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
from matplotlib.patches import Patch

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

Trace = Tuple[np.ndarray, np.ndarray]                   # (epochs, incumbent)
TraceDict = Dict[str, Dict[str, List[Trace]]]           # dataset -> method -> traces


# --------------------------------------------------------------------------- #
# Small helpers
# --------------------------------------------------------------------------- #

def _norm(name: str) -> str:
    """Canonical dataset key: lower-case, '_' -> '-'."""
    return str(name).lower().replace("_", "-")


def _dataset_label(name: str) -> str:
    return DATASET_LABELS.get(_norm(name), str(name))


def _order_datasets(names: Sequence[str]) -> List[str]:
    """Known datasets first (in DATASET_ORDER), then any others alphabetically."""
    def key(n: str):
        k = _norm(n)
        return (DATASET_ORDER.index(k), "") if k in DATASET_ORDER else (len(DATASET_ORDER), k)
    return sorted(names, key=key)


def _std(x: Sequence[float]) -> float:
    """Std across runs; 0 for a single run instead of NaN."""
    x = np.asarray(x, dtype=float)
    return float(np.std(x, ddof=STD_DDOF)) if len(x) > STD_DDOF else 0.0


def make_run_dir(base_dir: str = "figures", timestamp: Optional[str] = None) -> Path:
    """Create and return <base_dir>/<YYYYmmdd_HHMMSS>/ for one analysis run."""
    timestamp = timestamp or dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = Path(base_dir) / timestamp
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir


def _ensure_parent(path: str) -> None:
    """Make sure the folder of an output file exists."""
    Path(path).parent.mkdir(parents=True, exist_ok=True)


def build_method_styles(
    methods: Sequence[str],
    overrides: Optional[Dict[str, dict]] = None,
) -> Dict[str, dict]:
    """
    Assign every method a unique (colour, linestyle, linewidth) combination.

    The assignment depends only on the sorted method names, so a method looks
    identical in every subplot and every figure. Pass `overrides`
    ({method: {"color":..., "linestyle":..., "linewidth":...}}) to pin specific
    methods (e.g. to highlight your proposed method).
    """
    styles = {}
    for i, method in enumerate(sorted(methods)):
        styles[method] = {
            "color": COLORS[i % len(COLORS)],
            # (i // len(COLORS)) shifts the linestyle once colours start repeating
            "linestyle": LINESTYLES[(i + i // len(COLORS)) % len(LINESTYLES)],
            "linewidth": LINEWIDTHS[i % len(LINEWIDTHS)],
        }
    for method, style in (overrides or {}).items():
        styles.setdefault(method, {}).update(style)
    return styles


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
) -> TraceDict:
    """
    Turn the raw logs into one best-so-far step function per run.

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
    traces: TraceDict = {}

    for (dataset, method, _seed), run in df.groupby(["dataset", "search_method", "seed"]):
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
    stack = np.vstack([evaluate_on_grid(t, grid) for t in traces])
    mean = stack.mean(axis=0)
    mean[np.isnan(stack).any(axis=0)] = np.nan
    return mean


def _final_scores(traces: List[Trace]) -> np.ndarray:
    """Final incumbent of every run."""
    return np.array([values[-1] for _, values in traces], dtype=float)


# --------------------------------------------------------------------------- #
# Function 1: log-scale anytime curves
# --------------------------------------------------------------------------- #

def plot_anytime_curves(
    df: pd.DataFrame,
    output_path: str = "anytime_performance.pdf",
    metric: str = "validation",
    budget: float = 100_000,
    num_grid_points: int = 400,
    method_styles: Optional[Dict[str, dict]] = None,
    show: bool = False,
) -> plt.Figure:
    """
    One subplot per dataset showing the mean incumbent vs. epochs on a log-x axis.

    No std bands are drawn (to avoid overlapping clutter); methods are separated
    by unique colour / linestyle / linewidth combinations.
    """
    traces = extract_incumbent_traces(df, metric=metric, budget=budget)
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
            curve = _mean_curve(traces[dataset][method], grid)
            ax.plot(grid, curve, label=method, **styles[method])
            finite = np.flatnonzero(np.isfinite(curve))
            if len(finite):
                first_valid_x = min(first_valid_x, grid[finite[0]])

        ax.set_xscale("log")                     # the key change: log-scale x
        # Start the axis where the earliest mean curve starts (the mean is only
        # defined once every seed has logged a record), not at epoch 1.
        ax.set_xlim(first_valid_x, hi)
        ax.set_title(_dataset_label(dataset))
        ax.set_xlabel("Epochs (log scale)")
        ax.set_ylabel(f"{metric.capitalize()} accuracy (%)")
        ax.grid(True, which="major", linestyle="--", alpha=0.6)
        ax.grid(True, which="minor", linestyle=":", alpha=0.25)

    handles = [Line2D([0], [0], label=m, **styles[m]) for m in methods]
    fig.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, 1.02),
               ncol=min(len(methods), 5), frameon=True, handlelength=3.2,
               columnspacing=1.5)
    fig.tight_layout(rect=[0, 0, 1, 0.90])

    _ensure_parent(output_path)
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    print(f"Saved anytime plot to '{output_path}'.")
    if show:
        plt.show()
    plt.close(fig)
    return fig


# --------------------------------------------------------------------------- #
# Function 2: final-performance bar chart with std whiskers
# --------------------------------------------------------------------------- #

def _group_stats(source: TraceDict, dataset: str, methods: Sequence[str]):
    """Mean / std of the final incumbent per method; NaN where a method has no data."""
    means = np.full(len(methods), np.nan)
    stds = np.full(len(methods), np.nan)
    for i, method in enumerate(methods):
        runs = source.get(dataset, {}).get(method)
        if runs:
            scores = _final_scores(runs)
            means[i], stds[i] = scores.mean(), _std(scores)
    return means, stds


def plot_final_performance_bars(
    df: pd.DataFrame,
    output_path: str = "final_performance_bars.pdf",
    budget: float = 100_000,
    single_metric: str = "test",
    truncate_y: bool = True,
    method_styles: Optional[Dict[str, dict]] = None,
    show: bool = False,
) -> plt.Figure:
    """
    Final converged performance (mean +/- std over runs), one subplot per dataset.

    * CIFAR-10      : grouped bars per method, Validation (solid) next to Test (hatched).
    * other datasets: one bar per method, showing `single_metric` ("test" or "validation").

    All panels have the same width: in the paired CIFAR-10 panel each bar is half
    as wide as a single bar elsewhere. Methods with no data for a metric (e.g. no
    test accuracy logged) are marked "n/a" instead of drawing a bar.

    `truncate_y=True` zooms the y-axis around the data so small gaps between
    methods are visible. State this in the paper's caption, since a truncated
    bar baseline exaggerates differences; set it to False for a zero baseline.
    """
    val_traces = extract_incumbent_traces(df, "validation", budget)
    test_traces = extract_incumbent_traces(df, "test", budget)
    single_traces = val_traces if single_metric == "validation" else test_traces

    datasets = _order_datasets(list(set(val_traces) | set(test_traces)))
    methods = sorted({m for src in (val_traces, test_traces)
                      for ds in src for m in src[ds]})
    styles = build_method_styles(methods, method_styles)

    fig, axes = plt.subplots(1, len(datasets), figsize=(5.4 * len(datasets), 4.4),
                             squeeze=False)

    single_width = 0.62                       # bar width when there is one bar per method
    for ax, dataset in zip(axes[0], datasets):
        paired = _norm(dataset) == "cifar10"
        present = [m for m in methods
                   if m in val_traces.get(dataset, {}) or m in test_traces.get(dataset, {})]
        x = np.arange(len(present))

        if paired:
            width = single_width / 2          # two bars share the space of one
            groups = [("Validation", val_traces, -width / 2, None),
                      ("Test", test_traces, +width / 2, "//")]
        else:
            width = single_width
            groups = [(single_metric.capitalize(), single_traces, 0.0, None)]

        tops, bottoms, missing_x = [], [], []
        for _label, source, offset, hatch in groups:
            means, stds = _group_stats(source, dataset, present)
            ok = np.isfinite(means)
            ax.bar(x[ok] + offset, means[ok], width,
                   color=[styles[m]["color"] for m, k in zip(present, ok) if k],
                   edgecolor="black", linewidth=0.8, hatch=hatch, zorder=3)
            ax.errorbar(x[ok] + offset, means[ok], yerr=stds[ok], fmt="none",
                        ecolor="black", elinewidth=1.3, capsize=3, capthick=1.3,
                        zorder=4)
            tops.extend(means[ok] + stds[ok])
            bottoms.extend(means[ok] - stds[ok])
            missing_x.extend(x[~ok] + offset)

        title = _dataset_label(dataset)
        if not paired:
            title += f" ({single_metric.capitalize()})"
        ax.set_title(title)
        ax.set_ylabel("Accuracy (%)")
        ax.set_xticks(x)
        ax.set_xticklabels(present, rotation=30, ha="right")
        ax.grid(True, axis="y", linestyle="--", alpha=0.6, zorder=0)
        ax.grid(False, axis="x")
        if not tops:                          # nothing to draw for this dataset
            continue

        # The paired (CIFAR-10) panel gets extra headroom for its legend.
        headroom = 3.0 if paired else 1.0
        if truncate_y:
            lo, hi = min(bottoms), max(tops)
            pad = max(0.1 * (hi - lo), 0.5)
            ax.set_ylim(lo - 2 * pad, hi + headroom * pad)
        else:
            ax.set_ylim(0, max(tops) * (1.0 + 0.05 * headroom))

        # Mark methods without data for this metric.
        y0, y1 = ax.get_ylim()
        for xm in missing_x:
            ax.text(xm, y0 + 0.02 * (y1 - y0), "n/a", rotation=90, ha="center",
                    va="bottom", fontsize=9, color="0.3")

    # Solid vs. hatched legend (colour already identifies the method via x tick labels).
    if any(_norm(ds) == "cifar10" for ds in datasets):
        legend_ax = axes[0][[_norm(ds) == "cifar10" for ds in datasets].index(True)]
        legend_ax.legend(handles=[
            Patch(facecolor="white", edgecolor="black", label="Validation"),
            Patch(facecolor="white", edgecolor="black", hatch="//", label="Test"),
        ], loc="upper right", ncol=2, frameon=True)

    fig.tight_layout()
    _ensure_parent(output_path)
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    print(f"Saved final-performance bar chart to '{output_path}'.")
    if show:
        plt.show()
    plt.close(fig)
    return fig


# --------------------------------------------------------------------------- #
# Function 3: efficiency CSV (epochs to reach 95% of final score)
# --------------------------------------------------------------------------- #

def compute_efficiency_csv(
    df: pd.DataFrame,
    output_path: str = "efficiency_summary.csv",
    metric: str = "validation",
    threshold: float = 0.95,
    budget: float = 100_000,
    relative_to_start: bool = False,
) -> pd.DataFrame:
    """
    For every (dataset, method): how many epochs until the incumbent reaches
    `threshold` (default 95%) of the final converged score?

    Definitions
    -----------
    final score   : mean over runs of each run's final incumbent.
    target        : threshold * final score
                    (or start + threshold * (final - start) if `relative_to_start`;
                     "start" = mean first incumbent. This is stricter, because 95%
                     of a ~94% accuracy is already reached almost immediately.)
    main column   : first epoch at which the *run-averaged* incumbent curve reaches
                    the target (exact, evaluated at every logged epoch).
    per-run cols  : the same crossing computed for each run separately, then
                    averaged (runs that never reach it are excluded and counted).
    """
    if not 0 < threshold <= 1:
        raise ValueError("threshold must be in (0, 1].")

    traces = extract_incumbent_traces(df, metric=metric, budget=budget)
    pct = int(round(threshold * 100))
    rows = []

    for dataset in _order_datasets(list(traces)):
        for method in sorted(traces[dataset]):
            runs = traces[dataset][method]

            finals = _final_scores(runs)
            final_mean, final_std = finals.mean(), _std(finals)

            if relative_to_start:
                start_mean = np.mean([values[0] for _, values in runs])
                target = start_mean + threshold * (final_mean - start_mean)
            else:
                target = threshold * final_mean

            # Seed-averaged curve, evaluated at every epoch logged by any run.
            union = np.unique(np.concatenate([epochs for epochs, _ in runs]))
            mean_curve = _mean_curve(runs, union)
            hit = np.flatnonzero(mean_curve >= target - 1e-12)
            epochs_mean_curve = float(union[hit[0]]) if len(hit) else np.nan

            # Per-run crossing epochs.
            per_run = []
            for epochs, values in runs:
                reached = np.flatnonzero(values >= target - 1e-12)
                if len(reached):
                    per_run.append(float(epochs[reached[0]]))

            rows.append({
                "dataset": _dataset_label(dataset),
                "method": method,
                "metric": metric,
                "num_runs": len(runs),
                f"epochs_to_{pct}pct": epochs_mean_curve,
                f"epochs_to_{pct}pct_per_run_mean": np.mean(per_run) if per_run else np.nan,
                f"epochs_to_{pct}pct_per_run_std": _std(per_run) if per_run else np.nan,
                "runs_reaching_target": len(per_run),
                "target_score": round(float(target), 4),
                "final_mean": round(float(final_mean), 4),
                "final_std": round(float(final_std), 4),
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
    plot_final_performance_bars(df, str(out_dir / "final_performance_bars.pdf"),
                                budget=args.budget)
    summary = compute_efficiency_csv(df, str(out_dir / "efficiency_summary.csv"),
                                     budget=args.budget)
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()