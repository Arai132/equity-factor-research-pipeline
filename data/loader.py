from pathlib import Path
import pandas as pd
import yfinance as yf

CACHE_DIR = Path(__file__).resolve().parent / "cache"


def download_prices(tickers: list[str], start: str, end: str) -> pd.DataFrame:
    cache_path = CACHE_DIR / "prices.parquet"
    if cache_path.exists():
        return pd.read_parquet(cache_path)

    raw = yf.download(tickers, start=start, end=end, auto_adjust=True, progress=False)
    prices = raw["Close"] if isinstance(raw.columns, pd.MultiIndex) else raw

    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    prices.to_parquet(cache_path)
    return prices


def download_volume(tickers: list[str], start: str, end: str) -> pd.DataFrame:
    """Returns dollar volume (adjusted close × share volume)."""
    cache_path = CACHE_DIR / "volume.parquet"
    if cache_path.exists():
        return pd.read_parquet(cache_path)

    raw = yf.download(tickers, start=start, end=end, auto_adjust=True, progress=False)
    prices = raw["Close"]
    shares = raw["Volume"]
    dollar_vol = prices * shares

    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    dollar_vol.to_parquet(cache_path)
    return dollar_vol


def download_fundamentals(tickers: list[str]) -> pd.DataFrame:
    """
    Downloads quarterly Basic EPS for each ticker.
    Returns DataFrame: index=report dates, columns=tickers.
    Note: slow on first run (~10 min for 500 tickers).
    """
    cache_path = CACHE_DIR / "fundamentals.parquet"
    if cache_path.exists():
        return pd.read_parquet(cache_path)

    records = {}
    for i, ticker in enumerate(tickers):
        if i % 50 == 0:
            print(f"Fundamentals: {i}/{len(tickers)}", flush=True)
        try:
            stmt = yf.Ticker(ticker).quarterly_income_stmt
            if stmt is not None and not stmt.empty and "Basic EPS" in stmt.index:
                records[ticker] = stmt.loc["Basic EPS"]
        except Exception:
            pass

    if not records:
        return pd.DataFrame()

    df = pd.DataFrame(records)
    df.index = pd.to_datetime(df.index)
    df = df.sort_index()

    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    df.to_parquet(cache_path)
    return df
