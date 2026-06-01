import pandas as pd
from factors.base import BaseFactor, normalize


class MomentumFactor(BaseFactor):
    """12-1 month momentum: cumulative return from t-13 to t-1 monthly closes."""

    def compute(self, prices, volume, fundamentals, rebalance_dates):
        monthly = prices.resample("ME").last()
        signals = {}
        for date in rebalance_dates:
            try:
                loc = monthly.index.get_loc(date)
            except KeyError:
                continue
            if loc < 13:
                continue
            raw = monthly.iloc[loc - 1] / monthly.iloc[loc - 13] - 1
            signals[date] = raw
        return self._normalize_signals(signals)


class MediumMomentumFactor(BaseFactor):
    """6-1 month momentum: cumulative return from t-7 to t-1 monthly closes."""

    def compute(self, prices, volume, fundamentals, rebalance_dates):
        monthly = prices.resample("ME").last()
        signals = {}
        for date in rebalance_dates:
            try:
                loc = monthly.index.get_loc(date)
            except KeyError:
                continue
            if loc < 7:
                continue
            raw = monthly.iloc[loc - 1] / monthly.iloc[loc - 7] - 1
            signals[date] = raw
        return self._normalize_signals(signals)


class ReversalFactor(BaseFactor):
    """Short-term reversal: most recent 1-month return, negated."""

    def compute(self, prices, volume, fundamentals, rebalance_dates):
        monthly = prices.resample("ME").last()
        signals = {}
        for date in rebalance_dates:
            try:
                loc = monthly.index.get_loc(date)
            except KeyError:
                continue
            if loc < 1:
                continue
            raw = monthly.iloc[loc] / monthly.iloc[loc - 1] - 1
            signals[date] = -raw
        return self._normalize_signals(signals)
