import numpy as np
import pandas as pd
import pytest
from factors.volatility import IVolFactor


def make_data():
    np.random.seed(0)
    dates = pd.date_range("2020-01-01", periods=120, freq="B")
    spy_ret = np.random.randn(120) * 0.01
    spy_prices = pd.DataFrame(
        {"SPY": 100 * np.cumprod(1 + spy_ret)}, index=dates
    )
    # LOW_VOL closely tracks SPY; HIGH_VOL adds large idiosyncratic noise
    prices = pd.DataFrame({
        "LOW_VOL":  100 * np.cumprod(1 + spy_ret + np.random.randn(120) * 0.001),
        "HIGH_VOL": 100 * np.cumprod(1 + spy_ret + np.random.randn(120) * 0.05),
    }, index=dates)
    return prices, spy_prices


def test_ivol_low_vol_scores_higher():
    prices, spy_prices = make_data()
    factor = IVolFactor(spy_prices=spy_prices)
    result = factor.compute(prices, None, None, prices.index[-1:])
    assert result.iloc[0]["LOW_VOL"] > result.iloc[0]["HIGH_VOL"]


def test_ivol_no_lookahead():
    prices, spy_prices = make_data()
    factor = IVolFactor(spy_prices=spy_prices)
    target = prices.index[80]
    r_full = factor.compute(prices, None, None, pd.DatetimeIndex([target]))
    r_trunc = factor.compute(
        prices.loc[:target], spy_prices.loc[:target], None, pd.DatetimeIndex([target])
    )
    pd.testing.assert_frame_equal(r_full, r_trunc)


def test_ivol_insufficient_window_returns_empty():
    prices, spy_prices = make_data()
    factor = IVolFactor(spy_prices=spy_prices)
    result = factor.compute(prices.iloc[:10], spy_prices.iloc[:10], None, prices.index[9:10])
    assert result.empty or result.isna().all().all()
