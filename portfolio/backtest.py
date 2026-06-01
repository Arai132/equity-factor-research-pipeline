import numpy as np
import pandas as pd


def run_backtest(
    composite_alpha: pd.DataFrame,
    monthly_returns: pd.DataFrame,
    spy_returns: pd.Series = None,
    transaction_cost_bps: float = 10.0,
    n_quantiles: int = 5,
    min_stocks: int = 50,
) -> dict:
    """
    Long top quintile, short bottom quintile of composite alpha.
    Applies 10bps one-way transaction costs on turnover.
    Returns gross and net-of-cost performance.
    """
    tc = transaction_cost_bps / 10_000
    rows = []
    prev_long: set = set()
    prev_short: set = set()

    dates = composite_alpha.index[:-1]
    for i, date in enumerate(dates):
        alpha = composite_alpha.loc[date].dropna()
        if len(alpha) < min_stocks:
            continue

        n = len(alpha)
        q_size = n // n_quantiles
        sorted_stocks = alpha.sort_values()
        short_stocks = set(sorted_stocks.iloc[:q_size].index)
        long_stocks = set(sorted_stocks.iloc[-q_size:].index)

        fwd_date = composite_alpha.index[i + 1]
        if fwd_date not in monthly_returns.index:
            continue

        fwd = monthly_returns.loc[fwd_date]
        long_ret = fwd.reindex(list(long_stocks)).mean()
        short_ret = fwd.reindex(list(short_stocks)).mean()
        gross = long_ret - short_ret

        long_to = len(long_stocks.symmetric_difference(prev_long)) / max(len(long_stocks), 1)
        short_to = len(short_stocks.symmetric_difference(prev_short)) / max(len(short_stocks), 1)
        turnover = (long_to + short_to) / 2
        net = gross - 2 * turnover * tc

        rows.append({"date": fwd_date, "gross": gross, "net": net, "turnover": turnover})
        prev_long, prev_short = long_stocks, short_stocks

    df = pd.DataFrame(rows).set_index("date")
    spy_aligned = spy_returns.reindex(df.index) if spy_returns is not None else None

    return {
        "returns": df,
        "metrics_gross": compute_performance_metrics(df["gross"], spy_aligned),
        "metrics_net": compute_performance_metrics(df["net"], spy_aligned),
    }


def compute_performance_metrics(
    returns: pd.Series,
    spy_returns: pd.Series = None,
) -> dict:
    """Annualized metrics for a monthly return series."""
    ann_ret = (1 + returns.mean()) ** 12 - 1
    ann_vol = returns.std(ddof=1) * np.sqrt(12)
    sharpe = ann_ret / ann_vol if ann_vol > 0 else np.nan

    cumulative = (1 + returns).cumprod()
    rolling_max = cumulative.cummax()
    drawdown = (cumulative - rolling_max) / rolling_max
    max_dd = drawdown.min()
    calmar = ann_ret / abs(max_dd) if max_dd < 0 else np.nan

    beta = np.nan
    if spy_returns is not None:
        common = returns.index.intersection(spy_returns.index)
        if len(common) > 12:
            beta = np.cov(returns[common], spy_returns[common])[0, 1] / np.var(spy_returns[common])

    return {
        "annualized_return": ann_ret,
        "annualized_vol": ann_vol,
        "sharpe_ratio": sharpe,
        "max_drawdown": max_dd,
        "calmar_ratio": calmar,
        "monthly_win_rate": (returns > 0).mean(),
        "n_months": len(returns),
        "beta_to_spy": beta,
    }
