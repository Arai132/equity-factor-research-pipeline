import numpy as np
import pandas as pd
from portfolio.combination import equal_weight_alpha, ic_weighted_alpha


def make_factors(n_dates=36, n_stocks=50, seed=0):
    np.random.seed(seed)
    dates = pd.date_range("2020-01-31", periods=n_dates, freq="ME")
    tickers = [f"S{i}" for i in range(n_stocks)]
    return {
        "F1": pd.DataFrame(np.random.randn(n_dates, n_stocks), index=dates, columns=tickers),
        "F2": pd.DataFrame(np.random.randn(n_dates, n_stocks), index=dates, columns=tickers),
        "F3": pd.DataFrame(np.random.randn(n_dates, n_stocks), index=dates, columns=tickers),
    }


def make_returns(factors):
    dates = list(factors.values())[0].index
    tickers = list(factors.values())[0].columns
    np.random.seed(1)
    return pd.DataFrame(np.random.randn(len(dates), len(tickers)) * 0.05, index=dates, columns=tickers)


def test_equal_weight_shape():
    factors = make_factors()
    alpha = equal_weight_alpha(factors)
    f1 = list(factors.values())[0]
    assert alpha.shape == f1.shape


def test_equal_weight_is_mean_of_factors():
    factors = make_factors(n_dates=5, n_stocks=4)
    alpha = equal_weight_alpha(factors)
    for date in alpha.index:
        expected = pd.concat([f.loc[date] for f in factors.values()], axis=1).mean(axis=1)
        pd.testing.assert_series_equal(alpha.loc[date].dropna(), expected.dropna(), check_names=False)


def test_ic_weighted_shape():
    factors = make_factors()
    returns = make_returns(factors)
    alpha = ic_weighted_alpha(factors, returns)
    assert isinstance(alpha, pd.DataFrame)
    assert len(alpha) <= len(list(factors.values())[0])


def test_ic_weighted_uses_only_past_ic():
    """Weights at date t must be estimated from IC at dates < t only."""
    factors = make_factors(n_dates=24)
    returns = make_returns(factors)
    alpha_full = ic_weighted_alpha(factors, returns)
    # Truncate returns at month 12 — weights for months 1-12 should be identical
    returns_trunc = returns.iloc[:12]
    alpha_trunc = ic_weighted_alpha(factors, returns_trunc)
    # Months 1-12 overlap: same weights since IC history up to each point is identical
    common = alpha_full.index.intersection(alpha_trunc.index)
    pd.testing.assert_frame_equal(
        alpha_full.loc[common[:12]], alpha_trunc.loc[common[:12]], check_exact=False, atol=1e-10
    )


from portfolio.backtest import run_backtest, compute_performance_metrics


def make_alpha_and_returns(n_dates=60, n_stocks=200):
    np.random.seed(42)
    dates = pd.date_range("2010-01-31", periods=n_dates, freq="ME")
    tickers = [f"S{i}" for i in range(n_stocks)]
    alpha = pd.DataFrame(np.random.randn(n_dates, n_stocks), index=dates, columns=tickers)
    returns = pd.DataFrame(np.random.randn(n_dates, n_stocks) * 0.05, index=dates, columns=tickers)
    spy = pd.Series(np.random.randn(n_dates) * 0.04, index=dates, name="SPY")
    return alpha, returns, spy


def test_backtest_returns_expected_keys():
    alpha, returns, spy = make_alpha_and_returns()
    result = run_backtest(alpha, returns, spy_returns=spy)
    assert "returns" in result
    assert "metrics_gross" in result
    assert "metrics_net" in result


def test_backtest_net_less_than_gross():
    alpha, returns, spy = make_alpha_and_returns()
    result = run_backtest(alpha, returns, spy_returns=spy)
    gross = result["metrics_gross"]["annualized_return"]
    net = result["metrics_net"]["annualized_return"]
    assert net <= gross


def test_performance_metrics_keys():
    returns = pd.Series(np.random.randn(60) * 0.05, index=pd.date_range("2010-01-31", periods=60, freq="ME"))
    spy = returns * 0.8 + pd.Series(np.random.randn(60) * 0.01, index=returns.index)
    metrics = compute_performance_metrics(returns, spy_returns=spy)
    for key in ["annualized_return", "annualized_vol", "sharpe_ratio",
                "max_drawdown", "calmar_ratio", "monthly_win_rate", "beta_to_spy"]:
        assert key in metrics


def test_max_drawdown_is_negative():
    returns = pd.Series([-0.05] * 12, index=pd.date_range("2010-01-31", periods=12, freq="ME"))
    metrics = compute_performance_metrics(returns)
    assert metrics["max_drawdown"] < 0
