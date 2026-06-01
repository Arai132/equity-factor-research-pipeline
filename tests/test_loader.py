import shutil
import pandas as pd
import pytest
from pathlib import Path
from data.loader import download_prices, download_volume, CACHE_DIR

@pytest.fixture(autouse=True)
def clean_cache():
    shutil.rmtree(CACHE_DIR, ignore_errors=True)
    yield
    shutil.rmtree(CACHE_DIR, ignore_errors=True)

def test_download_prices_returns_dataframe():
    prices = download_prices(["AAPL", "MSFT"], start="2023-01-01", end="2023-03-31")
    assert isinstance(prices, pd.DataFrame)
    assert {"AAPL", "MSFT"}.issubset(set(prices.columns))
    assert len(prices) > 40

def test_download_prices_creates_cache():
    download_prices(["AAPL", "MSFT"], start="2023-01-01", end="2023-03-31")
    assert (CACHE_DIR / "prices.parquet").exists()

def test_download_prices_reads_from_cache(mocker):
    download_prices(["AAPL", "MSFT"], start="2023-01-01", end="2023-03-31")
    mock_yf = mocker.patch("data.loader.yf.download")
    download_prices(["AAPL", "MSFT"], start="2023-01-01", end="2023-03-31")
    mock_yf.assert_not_called()

def test_download_volume_returns_positive_values():
    vol = download_volume(["AAPL", "MSFT"], start="2023-01-01", end="2023-03-31")
    assert isinstance(vol, pd.DataFrame)
    assert (vol[["AAPL", "MSFT"]].dropna() > 0).all().all()
