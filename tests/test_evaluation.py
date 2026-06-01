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


from evaluation.quintile import compute_quintile_returns, compute_quintile_stats
from evaluation.decay import compute_factor_decay, compute_factor_correlation


def test_quintile_returns_shape():
    factor, returns = make_factor_and_returns(n_dates=60, n_stocks=200)
    q_returns = compute_quintile_returns(factor, returns)
    assert isinstance(q_returns, pd.DataFrame)
    assert list(q_returns.columns) == ["Q1", "Q2", "Q3", "Q4", "Q5"]


def test_quintile_monotonic_for_strong_factor():
    np.random.seed(7)
    n, m = 120, 300
    dates = pd.date_range("2010-01-31", periods=n, freq="ME")
    tickers = [f"S{i}" for i in range(m)]
    factor = pd.DataFrame(np.random.randn(n, m), index=dates, columns=tickers)
    returns = 0.2 * factor + pd.DataFrame(np.random.randn(n, m) * 0.05, index=dates, columns=tickers)
    q_returns = compute_quintile_returns(factor, returns)
    means = q_returns.mean()
    assert means["Q1"] < means["Q2"] < means["Q3"] < means["Q4"] < means["Q5"]


def test_decay_returns_correct_horizons():
    factor, returns = make_factor_and_returns(n_dates=60, n_stocks=100)
    # Pad returns for multi-horizon computation
    extra = pd.DataFrame(0.0, index=pd.date_range(returns.index[-1] + pd.offsets.MonthEnd(1),
                         periods=12, freq="ME"), columns=returns.columns)
    long_returns = pd.concat([returns, extra])
    decay = compute_factor_decay(factor, long_returns, horizons=[1, 3, 6])
    assert list(decay.index) == [1, 3, 6]
    assert "mean_ic" in decay.columns


def test_factor_correlation_diagonal_is_one():
    factor, _ = make_factor_and_returns()
    corr = compute_factor_correlation({"F1": factor, "F2": factor * 2})
    assert abs(corr.loc["F1", "F1"] - 1.0) < 1e-6
    assert abs(corr.loc["F2", "F2"] - 1.0) < 1e-6
