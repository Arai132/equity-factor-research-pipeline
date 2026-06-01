import numpy as np
import pandas as pd


def compute_quintile_returns(
    factor_df: pd.DataFrame,
    forward_returns: pd.DataFrame,
    n_quantiles: int = 5,
    min_stocks: int = 50,
) -> pd.DataFrame:
    """Equal-weighted returns for each quintile bucket, indexed by date."""
    labels = [f"Q{i + 1}" for i in range(n_quantiles)]
    rows = []
    common_dates = factor_df.index.intersection(forward_returns.index)

    for date in common_dates:
        f = factor_df.loc[date].dropna()
        r = forward_returns.loc[date].reindex(f.index).dropna()
        common = f.index.intersection(r.index)
        if len(common) < min_stocks:
            continue
        buckets = pd.qcut(f[common], q=n_quantiles, labels=labels)
        row = {"date": date}
        for q in labels:
            stocks = buckets[buckets == q].index
            row[q] = r[stocks].mean()
        rows.append(row)

    return pd.DataFrame(rows).set_index("date")[labels]


def compute_quintile_stats(quintile_returns: pd.DataFrame) -> pd.DataFrame:
    """Annualized return, vol, Sharpe, and hit rate per quintile."""
    stats = {}
    for q in quintile_returns.columns:
        r = quintile_returns[q].dropna()
        ann_ret = (1 + r.mean()) ** 12 - 1
        ann_vol = r.std(ddof=1) * np.sqrt(12)
        stats[q] = {
            "ann_return": ann_ret,
            "ann_vol": ann_vol,
            "sharpe": ann_ret / ann_vol if ann_vol > 0 else np.nan,
            "hit_rate": (r > 0).mean(),
        }
    return pd.DataFrame(stats).T
