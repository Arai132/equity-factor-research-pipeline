import numpy as np
import pandas as pd
import pytest
from evaluation.ic import compute_ic_series, compute_ic_stats


def make_factor_and_returns(n_dates=36, n_stocks=100, seed=0):
    np.random.seed(seed)
    dates = pd.date_range("2020-01-31", periods=n_dates, freq="ME")
    tickers = [f"S{i}" for i in range(n_stocks)]
    factor = pd.DataFrame(np.random.randn(n_dates, n_stocks), index=dates, columns=tickers)
    # Returns strongly correlated with factor + noise
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


def test_ic_series_nan_when_insufficient_stocks():
    """Dates with fewer than min_stocks valid observations should produce NaN."""
    dates = pd.date_range("2020-01-31", periods=3, freq="ME")
    tickers = [f"S{i}" for i in range(10)]
    factor = pd.DataFrame(np.random.randn(3, 10), index=dates, columns=tickers)
    returns = pd.DataFrame(np.random.randn(3, 10), index=dates, columns=tickers)
    ic = compute_ic_series(factor, returns, min_stocks=20)  # 10 stocks < 20 threshold
    assert ic.isna().all()


def test_ic_stats_degenerate_series():
    """IC series with fewer than 2 observations should return NaN for all stats."""
    empty = pd.Series(dtype=float)
    stats = compute_ic_stats(empty)
    assert np.isnan(stats["mean_ic"])
    assert np.isnan(stats["t_stat"])
    assert stats["n_months"] == 0

    single = pd.Series([0.05])
    stats_single = compute_ic_stats(single)
    assert np.isnan(stats_single["mean_ic"])
    assert stats_single["n_months"] == 1
