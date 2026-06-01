"""
End-to-end pipeline runner. Downloads data and verifies all components work.
Estimated runtime: ~5 min on first run (prices + volume download). Subsequent runs: ~2 min.
"""
import pandas as pd
from data.loader import download_prices, download_volume
from data.universe import get_sp500_tickers, build_return_matrix
from factors.momentum import MomentumFactor, MediumMomentumFactor, ReversalFactor
from factors.volatility import IVolFactor
from factors.illiquidity import IlliquidityFactor
from evaluation.ic import compute_ic_series, compute_ic_stats
from evaluation.quintile import compute_quintile_returns
from evaluation.decay import compute_factor_decay, compute_factor_correlation
from portfolio.combination import equal_weight_alpha, ic_weighted_alpha
from portfolio.backtest import run_backtest

START_DAILY = "2009-01-01"   # extra year for 12-month momentum lookback
END_DAILY = "2024-12-31"
START_SIGNAL = "2010-01-01"
END_SIGNAL = "2024-12-31"
IS_END = "2019-12-31"        # in-sample cutoff

print("Fetching tickers...")
tickers = get_sp500_tickers()

print("Downloading prices and volume...")
prices = download_prices(tickers, START_DAILY, END_DAILY)
volume = download_volume(tickers, START_DAILY, END_DAILY)

print("Downloading SPY...")
import yfinance as yf
spy_daily = yf.download("SPY", start=START_DAILY, end=END_DAILY, auto_adjust=True, progress=False)[["Close"]]
spy_monthly = spy_daily.resample("ME").last().loc[START_SIGNAL:END_SIGNAL].iloc[:, 0]
spy_monthly_ret = spy_monthly.pct_change()

print("Building return matrix...")
monthly_closes, monthly_returns = build_return_matrix(prices, START_SIGNAL, END_SIGNAL)
rebalance_dates = monthly_closes.index
forward_returns = monthly_returns.shift(-1)

print("Computing factors...")
ivol_factor = IVolFactor(spy_prices=spy_daily)
factors = {
    "MOM":   MomentumFactor().compute(prices, volume, None, rebalance_dates),
    "MOM6":  MediumMomentumFactor().compute(prices, volume, None, rebalance_dates),
    "REV":   ReversalFactor().compute(prices, volume, None, rebalance_dates),
    "IVOL":  ivol_factor.compute(prices, volume, None, rebalance_dates),
    "ILLIQ": IlliquidityFactor().compute(prices, volume, None, rebalance_dates),
}

print("\n=== Factor Coverage (number of stocks with valid signal) ===")
for name, f in factors.items():
    print(f"  {name}: {f.notna().sum(axis=1).mean():.0f} avg stocks/month, {len(f)} months")

print("\n=== In-Sample IC Statistics (2010-2019) ===")
for name, f in factors.items():
    f_is = f.loc[:IS_END]
    fwd_is = forward_returns.loc[:IS_END]
    ic = compute_ic_series(f_is, fwd_is)
    stats = compute_ic_stats(ic)
    print(f"  {name}: IC={stats['mean_ic']:.4f}, ICIR={stats['icir']:.2f}, t={stats['t_stat']:.2f}")

print("\nPipeline completed successfully.")
