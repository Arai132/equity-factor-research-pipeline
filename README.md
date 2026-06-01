# Cross-Sectional Factor Research Pipeline

End-to-end equity factor research on S&P 500 stocks (2010–2024), built as a Walleye Capital quant research internship application project.

## What This Builds

Five systematic equity factors are constructed, evaluated, combined into a composite alpha, and backtested in a long-short portfolio.

| Factor | Description | Data |
|--------|-------------|------|
| MOM | 12-1 month price momentum | Price |
| MOM6 | 6-1 month price momentum (medium-term) | Price |
| REV | 1-month short-term reversal (negated) | Price |
| IVOL | Idiosyncratic volatility from 60-day OLS on SPY (negated) | Price |
| ILLIQ | Amihud illiquidity: avg \|ret\|/dvol over 21 days | Price + Volume |

Each factor is cross-sectionally winsorized (±3σ) and z-scored before use.

## Evaluation Methodology

- **IC/ICIR**: Monthly Spearman rank correlation between factor and next-month returns
- **Quintile analysis**: Equal-weighted returns per quintile, Q5-Q1 spread
- **Factor decay**: IC at 1, 2, 3, 6, 12 month horizons
- **Factor correlation**: Average cross-sectional Spearman correlation matrix
- **In-sample**: 2010–2019 | **Out-of-sample**: 2020–2024

## Portfolio Construction

Long top quintile (~100 stocks), short bottom quintile (~100 stocks), equal-weighted, monthly rebalancing. Two alpha combination methods are compared: equal-weight average and IC-weighted (expanding window, no look-ahead).

Transaction costs: 10bps per-side applied to monthly turnover.

## Known Limitations

1. **Survivorship bias**: Uses current S&P 500 constituents. Companies removed from the index (bankruptcies, delistings) are excluded, which inflates all factor returns. Production research requires a point-in-time constituent universe (e.g., CRSP).
2. **Research backtest only**: No market impact, slippage, borrow costs, or capacity modeling.
3. **No fundamental factor**: yfinance's earnings history is limited to recent quarters only, preventing an EP factor with 2010–2024 coverage. MOM6 (6-1 month momentum) is used as the fifth factor instead.
4. **Equal weighting within legs**: No risk model or optimizer. Beta neutrality is approximate.

## Setup

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python run_pipeline.py   # downloads data (~5 min first run), prints IC stats
jupyter notebook notebooks/research.ipynb
```

## Tests

```bash
pytest tests/ -v
```
