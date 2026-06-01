import numpy as np
import pandas as pd
from scipy.stats import spearmanr


def equal_weight_alpha(factors: dict) -> pd.DataFrame:
    """Simple average of all factor z-scores."""
    all_dates = sorted(set().union(*[f.index for f in factors.values()]))
    result = {}
    for date in all_dates:
        cols = {name: f.loc[date] for name, f in factors.items() if date in f.index}
        if not cols:
            continue
        result[date] = pd.DataFrame(cols).mean(axis=1)
    return pd.DataFrame(result).T


def ic_weighted_alpha(
    factors: dict,
    forward_returns: pd.DataFrame,
    min_stocks: int = 30,
) -> pd.DataFrame:
    """
    Weight each factor by its expanding-window mean IC, estimated on data strictly
    before the current rebalance date (no look-ahead).
    Falls back to equal weighting until each factor has at least one IC observation.
    """
    names = list(factors.keys())
    all_dates = sorted(set().union(*[f.index for f in factors.values()]))
    ic_history = {name: [] for name in names}
    result = {}

    for date in all_dates:
        # Compute weights from history (before this date)
        mean_ics = {name: np.mean(ic_history[name]) if ic_history[name] else 0.0
                    for name in names}
        total = sum(abs(v) for v in mean_ics.values())
        if total < 1e-12:
            weights = {name: 1.0 / len(names) for name in names}
        else:
            weights = {name: mean_ics[name] / total for name in names}

        # Compute composite score
        available = {name: factors[name].loc[date] for name in names if date in factors[name].index}
        if not available:
            continue
        composite = sum(weights[name] * available[name] for name in available)
        result[date] = composite

        # Update IC history using this date's realized IC
        if date in forward_returns.index:
            for name in names:
                if date not in factors[name].index:
                    continue
                f = factors[name].loc[date].dropna()
                r = forward_returns.loc[date].reindex(f.index).dropna()
                common = f.index.intersection(r.index)
                if len(common) < min_stocks:
                    continue
                ic, _ = spearmanr(f[common].values, r[common].values)
                ic_history[name].append(ic)

    return pd.DataFrame(result).T
