import numpy as np
import pandas as pd
from scipy.stats import spearmanr, ttest_1samp


def compute_ic_series(
    factor_df: pd.DataFrame,
    forward_returns: pd.DataFrame,
    min_stocks: int = 30,
) -> pd.Series:
    """
    Monthly Spearman IC between factor scores and next-month returns.
    Returns pd.Series indexed by date, with NaN where data is insufficient.
    """
    common_dates = factor_df.index.intersection(forward_returns.index)
    ics = {}
    for date in common_dates:
        f = factor_df.loc[date].dropna()
        r = forward_returns.loc[date].reindex(f.index).dropna()
        common = f.index.intersection(r.index)
        if len(common) < min_stocks:
            ics[date] = np.nan
            continue
        ic, _ = spearmanr(f[common].values, r[common].values)
        ics[date] = ic
    return pd.Series(ics)


def compute_ic_stats(ic_series: pd.Series) -> dict:
    """Mean IC, std, ICIR, t-stat (vs zero), pct positive, n months."""
    clean = ic_series.dropna()
    if len(clean) < 2:
        return {"mean_ic": np.nan, "std_ic": np.nan, "icir": np.nan,
                "t_stat": np.nan, "pct_positive": np.nan, "n_months": len(clean)}
    mean_ic = clean.mean()
    std_ic = clean.std(ddof=1)
    icir = mean_ic / std_ic if std_ic > 0 else np.nan
    t_stat = ttest_1samp(clean, 0).statistic
    pct_pos = (clean > 0).mean()
    return {
        "mean_ic": mean_ic,
        "std_ic": std_ic,
        "icir": icir,
        "t_stat": t_stat,
        "pct_positive": pct_pos,
        "n_months": len(clean),
    }
