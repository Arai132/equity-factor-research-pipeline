import numpy as np
import pandas as pd
import pytest
from evaluation.ic import compute_ic_series, compute_ic_stats


def make_factor_and_returns(n_dates=36, n_stocks=100, seed=0):
    np.random.seed(seed)
    dates = pd.date_range("2020-01-31", periods=n_dates, freq="ME")
    tickers = [f"S{i}" for i in range(n_stocks)]
    factor = pd.DataFrame(np.random.randn(n_dates, n_stocks), index=dates, columns=tickers)
    # Returns correlated with factor (IC ~ 0.05) + noise
    returns = 0.05 * factor + pd.DataFrame(
        np.random.randn(n_dates, n_stocks) * 0.1, index=dates, columns=tickers
    )
    return factor, returns


def test_ic_series_length():
    factor, returns = make_factor_and_returns()
    ic = compute_ic_series(factor, returns)
    assert isinstance(ic, pd.Series)
    assert len(ic) == len(factor)


def test_ic_stats_keys():
    factor, returns = make_factor_and_returns()
    ic = compute_ic_series(factor, returns)
    stats = compute_ic_stats(ic)
    for key in ["mean_ic", "std_ic", "icir", "t_stat", "pct_positive", "n_months"]:
        assert key in stats


def test_ic_positive_for_correlated_factor():
    factor, returns = make_factor_and_returns(n_dates=120, n_stocks=200)
    ic = compute_ic_series(factor, returns)
    stats = compute_ic_stats(ic)
    assert stats["mean_ic"] > 0
    assert stats["t_stat"] > 2.0


def test_ic_negative_for_inverted_factor():
    factor, returns = make_factor_and_returns(n_dates=120, n_stocks=200)
    ic_pos = compute_ic_series(factor, returns)
    ic_neg = compute_ic_series(-factor, returns)
    assert compute_ic_stats(ic_neg)["mean_ic"] < 0
