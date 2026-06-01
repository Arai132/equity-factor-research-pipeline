import numpy as np
import pandas as pd
from factors.earnings_yield import EarningsYieldFactor


def make_data():
    quarters = pd.to_datetime([
        "2018-12-01", "2019-03-01", "2019-06-01", "2019-09-01",
        "2019-12-01", "2020-03-01",
    ])
    fundamentals = pd.DataFrame({
        "HIGH_EP": [5.0, 5.0, 5.0, 5.0, 5.0, 5.0],
        "LOW_EP":  [0.1, 0.1, 0.1, 0.1, 0.1, 0.1],
    }, index=quarters)
    dates = pd.date_range("2018-01-01", "2020-06-30", freq="B")
    prices = pd.DataFrame({
        "HIGH_EP": [100.0] * len(dates),
        "LOW_EP":  [100.0] * len(dates),
    }, index=dates)
    return prices, fundamentals


def test_ep_ordering():
    prices, fundamentals = make_data()
    # Rebalance 60 days after last report to satisfy 45-day lag
    rebalance = pd.DatetimeIndex(["2020-05-01"])
    result = EarningsYieldFactor().compute(prices, None, fundamentals, rebalance)
    assert result.iloc[0]["HIGH_EP"] > result.iloc[0]["LOW_EP"]


def test_ep_45_day_lag_excludes_recent_quarter():
    """A quarter filed 10 days ago must not appear in TTM EPS."""
    dates = pd.date_range("2019-01-01", "2020-12-31", freq="B")
    prices = pd.DataFrame({"A": [100.0] * len(dates), "B": [100.0] * len(dates)}, index=dates)

    # 4 old quarters (equal EPS for A and B) + 1 recent quarter (A jumps to 100)
    quarters = pd.to_datetime(["2019-09-01", "2019-12-01", "2020-03-01", "2020-06-01", "2020-06-20"])
    fundamentals = pd.DataFrame({
        "A": [1.0, 1.0, 1.0, 1.0, 100.0],
        "B": [1.0, 1.0, 1.0, 1.0, 1.0],
    }, index=quarters)

    # Rebalance 10 days after the big EPS quarter (2020-06-20 + 10 = 2020-06-30)
    # 45-day lag: cutoff = 2020-06-30 - 45 days = 2020-05-16 → excludes the 2020-06-20 quarter
    result = EarningsYieldFactor().compute(
        prices, None, fundamentals, pd.DatetimeIndex(["2020-06-30"])
    )
    if not result.empty:
        a_score = result.iloc[0]["A"]
        b_score = result.iloc[0]["B"]
        # A's TTM uses quarters before cutoff: 2019-12, 2020-03, 2020-06-01 only 3 < 4 required
        # OR 4 quarters if 2020-06-01 is included: 1+1+1+1=4, same as B
        # Either way A should NOT benefit from the 100 EPS quarter
        assert abs(a_score - b_score) < 0.01 or (pd.isna(a_score) and pd.isna(b_score))


def test_ep_empty_fundamentals_returns_empty_dataframe():
    prices, _ = make_data()
    result = EarningsYieldFactor().compute(prices, None, pd.DataFrame(), pd.DatetimeIndex(["2020-05-01"]))
    assert result.empty or result.isna().all().all()
