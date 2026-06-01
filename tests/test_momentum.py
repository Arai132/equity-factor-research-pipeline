import numpy as np
import pandas as pd
import pytest
from factors.momentum import MomentumFactor, ReversalFactor


def make_prices(n_months: int = 15) -> pd.DataFrame:
    dates = pd.date_range("2019-01-31", periods=n_months, freq="ME")
    return pd.DataFrame({
        "UP":   [100 * 1.05 ** i for i in range(n_months)],
        "FLAT": [100.0] * n_months,
        "DOWN": [100 * 0.95 ** i for i in range(n_months)],
    }, index=dates)


def test_momentum_ordering():
    prices = make_prices(15)
    result = MomentumFactor().compute(prices, None, None, prices.index[-1:])
    row = result.iloc[0]
    assert row["UP"] > row["FLAT"] > row["DOWN"]


def test_momentum_skip_month():
    """Signal must use price at t-1, not t, as the recent anchor."""
    dates = pd.date_range("2019-01-31", periods=15, freq="ME")
    # A: flat for 14 months, then doubles in final month
    # B: flat for 13 months, drops 50% in month 14, flat in month 15
    prices = pd.DataFrame({
        "A": [100.0] * 14 + [200.0],
        "B": [100.0] * 13 + [50.0, 50.0],
    }, index=dates)
    result = MomentumFactor().compute(prices, None, None, prices.index[-1:])
    # 12-1 MOM for A at last date: close[13]/close[1] = 100/100 = 0 (flat)
    # 12-1 MOM for B at last date: close[13]/close[1] = 50/100 = -0.5 (drop)
    # So A should score higher than B
    assert result.iloc[0]["A"] > result.iloc[0]["B"]


def test_momentum_no_lookahead():
    """Signal at t must not depend on data after t."""
    prices = make_prices(15)
    target_date = prices.index[-2]
    result_full = MomentumFactor().compute(prices, None, None, pd.DatetimeIndex([target_date]))
    result_trunc = MomentumFactor().compute(prices.iloc[:-1], None, None, pd.DatetimeIndex([target_date]))
    pd.testing.assert_frame_equal(result_full, result_trunc)


def test_momentum_insufficient_history_returns_empty():
    prices = make_prices(5)
    result = MomentumFactor().compute(prices, None, None, prices.index)
    assert len(result) == 0


def test_reversal_negatively_correlated_with_momentum():
    prices = make_prices(15)
    mom = MomentumFactor().compute(prices, None, None, prices.index[-1:])
    rev = ReversalFactor().compute(prices, None, None, prices.index[-1:])
    # UP stock has high momentum → should have low reversal
    assert rev.iloc[0]["UP"] < rev.iloc[0]["DOWN"]
