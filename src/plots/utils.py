from src.plots.config import * 

from typing import Sequence, Optional

from pathlib import Path

import numpy as np

import pandas as pd

import datetime as dt

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

