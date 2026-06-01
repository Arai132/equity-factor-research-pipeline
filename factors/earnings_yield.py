import numpy as np
import pandas as pd
from factors.base import BaseFactor


class EarningsYieldFactor(BaseFactor):
    """
    Trailing twelve-month earnings yield: TTM EPS / price.
    EPS data is lagged 45 days before use to simulate SEC filing delays.
    Requires at least 4 quarterly observations for TTM computation.
    """

    def __init__(self, lag_days: int = 45, min_quarters: int = 4):
        self.lag_days = lag_days
        self.min_quarters = min_quarters

    def compute(self, prices, volume, fundamentals, rebalance_dates):
        if fundamentals is None or fundamentals.empty:
            return pd.DataFrame()

        signals = {}
        for date in rebalance_dates:
            cutoff = date - pd.Timedelta(days=self.lag_days)
            available = fundamentals.loc[fundamentals.index <= cutoff]
            if available.empty:
                continue

            ep = {}
            for ticker in available.columns:
                series = available[ticker].dropna()
                if len(series) < self.min_quarters:
                    continue
                ttm_eps = series.iloc[-self.min_quarters:].sum()
                if ticker not in prices.columns:
                    continue
                price = prices.loc[:date, ticker].dropna()
                if price.empty or price.iloc[-1] <= 0:
                    continue
                ep[ticker] = ttm_eps / price.iloc[-1]

            if ep:
                signals[date] = pd.Series(ep)

        return self._normalize_signals(signals)
