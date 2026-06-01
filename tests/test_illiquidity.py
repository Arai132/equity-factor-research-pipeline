import numpy as np
import pandas as pd
from factors.illiquidity import IlliquidityFactor


def make_data():
    np.random.seed(1)
    dates = pd.date_range("2020-01-01", periods=42, freq="B")
    returns = np.random.randn(42, 2) * 0.01
    prices = pd.DataFrame(
        100 * np.cumprod(1 + returns, axis=0),
        index=dates, columns=["LIQUID", "ILLIQUID"]
    )
    volume = pd.DataFrame({
        "LIQUID":   prices["LIQUID"] * 1e8,
        "ILLIQUID": prices["ILLIQUID"] * 1e4,
    }, index=dates)
    return prices, volume


def test_liquid_stock_scores_higher():
    prices, volume = make_data()
    factor = IlliquidityFactor()
    result = factor.compute(prices, volume, None, prices.index[-1:])
    assert result.iloc[0]["LIQUID"] > result.iloc[0]["ILLIQUID"]


def test_illiquidity_no_lookahead():
    prices, volume = make_data()
    factor = IlliquidityFactor()
    target = prices.index[20]
    r_full = factor.compute(prices, volume, None, pd.DatetimeIndex([target]))
    r_trunc = factor.compute(prices.loc[:target], volume.loc[:target], None, pd.DatetimeIndex([target]))
    pd.testing.assert_frame_equal(r_full, r_trunc)


def test_zero_volume_returns_nan():
    dates = pd.date_range("2020-01-01", periods=25, freq="B")
    prices = pd.DataFrame({"A": [100.0] * 25}, index=dates)
    volume = pd.DataFrame({"A": [0.0] * 25}, index=dates)
    factor = IlliquidityFactor()
    result = factor.compute(prices, volume, None, prices.index[-1:])
    assert result.empty or result.isna().all().all()
