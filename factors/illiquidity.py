import numpy as np
import pandas as pd
from factors.base import BaseFactor


class IlliquidityFactor(BaseFactor):
    """
    Amihud illiquidity: 21-day avg of |daily return| / daily dollar volume.
    Illiquid stocks (high ILLIQ) receive positive scores, capturing the illiquidity premium.
    """

    def __init__(self, window: int = 21):
        self.window = window

    def compute(self, prices, volume, fundamentals, rebalance_dates):
        daily_returns = prices.pct_change()
        signals = {}

        for date in rebalance_dates:
            ret_w = daily_returns.loc[:date].tail(self.window)
            vol_w = volume.loc[ret_w.index]
            safe_vol = vol_w.replace(0, np.nan)
            illiq = (ret_w.abs() / safe_vol).mean(axis=0)

            if illiq.isna().all():
                continue

            signals[date] = illiq

        return self._normalize_signals(signals)
