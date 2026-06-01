import numpy as np
import pandas as pd
from factors.base import BaseFactor


class IVolFactor(BaseFactor):
    """
    Idiosyncratic volatility: residual std from 60-day rolling OLS on SPY returns.
    Negated so that low-volatility stocks receive positive scores.
    """

    def __init__(self, spy_prices: pd.DataFrame = None, window: int = 60):
        self.spy_prices = spy_prices
        self.window = window

    def compute(self, prices, volume, fundamentals, rebalance_dates):
        daily_returns = prices.pct_change().dropna(how="all")
        spy_source = self.spy_prices if self.spy_prices is not None else volume
        spy = spy_source.iloc[:, 0].pct_change().dropna() if spy_source is not None else None
        signals = {}

        for date in rebalance_dates:
            window_returns = daily_returns.loc[:date].tail(self.window)
            if spy is not None:
                spy_window = spy.loc[spy.index <= date].tail(self.window)
                common_idx = window_returns.index.intersection(spy_window.index)
            else:
                common_idx = window_returns.index

            if len(common_idx) < 30:
                continue

            stocks = window_returns.loc[common_idx].values  # (T, N)

            if spy is not None:
                mkt = spy_window.loc[common_idx].values      # (T,)
                mkt_dm = mkt - mkt.mean()
                mkt_var = (mkt_dm ** 2).sum()
                if mkt_var < 1e-12:
                    continue
                betas = (stocks * mkt_dm[:, None]).sum(axis=0) / mkt_var
                alphas = stocks.mean(axis=0) - betas * mkt.mean()
                residuals = stocks - alphas[None, :] - mkt[:, None] * betas[None, :]
            else:
                residuals = stocks - stocks.mean(axis=0)

            ivol = np.std(residuals, axis=0, ddof=1)
            signals[date] = pd.Series(-ivol, index=prices.columns)

        return self._normalize_signals(signals)
