import numpy as np
import pandas as pd
from evaluation.ic import compute_ic_series, compute_ic_stats


def compute_factor_decay(
    factor_df: pd.DataFrame,
    monthly_returns: pd.DataFrame,
    horizons: list[int] = None,
) -> pd.DataFrame:
    """
    IC at each forward return horizon (in months).
    Returns DataFrame: index=horizons, columns=[mean_ic, icir].
    """
    if horizons is None:
        horizons = [1, 2, 3, 6, 12]
    rows = []
    for h in horizons:
        fwd = monthly_returns.shift(-h)
        ic = compute_ic_series(factor_df, fwd)
        stats = compute_ic_stats(ic)
        rows.append({"horizon": h, "mean_ic": stats["mean_ic"], "icir": stats["icir"]})
    return pd.DataFrame(rows).set_index("horizon")


def compute_factor_correlation(
    factors: dict,
    min_stocks: int = 30,
) -> pd.DataFrame:
    """
    Average cross-sectional Spearman correlation between factor pairs across all dates.
    Returns symmetric correlation matrix.
    """
    names = list(factors.keys())
    n = len(names)
    mat = np.eye(n)

    for i in range(n):
        for j in range(i + 1, n):
            corr_series = compute_ic_series(factors[names[i]], factors[names[j]], min_stocks=min_stocks)
            avg = corr_series.dropna().mean() if not corr_series.dropna().empty else np.nan
            mat[i, j] = mat[j, i] = avg

    return pd.DataFrame(mat, index=names, columns=names)
