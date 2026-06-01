import numpy as np
import pandas as pd
from scipy.stats import spearmanr
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
            fa, fb = factors[names[i]], factors[names[j]]
            common_dates = fa.index.intersection(fb.index)
            daily_corrs = []
            for date in common_dates:
                a = fa.loc[date].dropna()
                b = fb.loc[date].reindex(a.index).dropna()
                common = a.index.intersection(b.index)
                if len(common) < min_stocks:
                    continue
                c, _ = spearmanr(a[common].values, b[common].values)
                daily_corrs.append(c)
            avg = np.mean(daily_corrs) if daily_corrs else np.nan
            mat[i, j] = mat[j, i] = avg

    return pd.DataFrame(mat, index=names, columns=names)
