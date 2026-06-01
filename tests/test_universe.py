import pandas as pd
import pytest
from data.universe import get_sp500_tickers, build_return_matrix

def test_get_sp500_tickers_returns_list():
    tickers = get_sp500_tickers()
    assert isinstance(tickers, list)
    assert len(tickers) >= 490
    assert "AAPL" in tickers

def test_get_sp500_tickers_no_dots():
    tickers = get_sp500_tickers()
    assert all("." not in t for t in tickers)

def test_build_return_matrix_shape():
    dates = pd.date_range("2020-01-31", periods=6, freq="ME")
    prices = pd.DataFrame(
        {"A": [100, 110, 105, 120, 115, 130], "B": [50, 55, 52, 58, 56, 62]},
        index=dates,
    )
    monthly_closes, returns = build_return_matrix(prices, start="2020-01-01", end="2020-06-30")
    assert monthly_closes.shape[1] == 2
    assert returns.shape == monthly_closes.shape
    assert returns.iloc[0].isna().all()

def test_build_return_matrix_correct_values():
    dates = pd.date_range("2020-01-31", periods=3, freq="ME")
    prices = pd.DataFrame({"A": [100.0, 110.0, 99.0]}, index=dates)
    _, returns = build_return_matrix(prices, "2020-01-01", "2020-03-31")
    assert abs(returns["A"].iloc[1] - 0.10) < 1e-6
    assert abs(returns["A"].iloc[2] - (-0.10)) < 1e-6

def test_build_return_matrix_drops_sparse_tickers():
    dates = pd.date_range("2020-01-31", periods=10, freq="ME")
    prices = pd.DataFrame(
        {"FULL": [100.0] * 10, "SPARSE": [100.0] + [None] * 9},
        index=dates,
    )
    monthly_closes, _ = build_return_matrix(prices, "2020-01-01", "2020-10-31", min_coverage=0.9)
    assert "FULL" in monthly_closes.columns
    assert "SPARSE" not in monthly_closes.columns
