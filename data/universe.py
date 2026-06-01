import pandas as pd
import urllib.request


def get_sp500_tickers() -> list[str]:
    """Fetches current S&P 500 constituents from Wikipedia."""
    # Add User-Agent header to avoid 403 errors from Wikipedia
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
    }
    request = urllib.request.Request(
        "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies",
        headers=headers
    )
    with urllib.request.urlopen(request) as response:
        tables = pd.read_html(response)
    tickers = tables[0]["Symbol"].tolist()
    return [t.replace(".", "-") for t in tickers]


def build_return_matrix(
    prices: pd.DataFrame,
    start: str,
    end: str,
    min_coverage: float = 0.9,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Resample daily prices to month-end, filter to [start, end], drop sparse tickers.
    Returns (monthly_closes, monthly_returns).
    """
    monthly_closes = prices.resample("ME").last().loc[start:end]
    coverage = monthly_closes.notna().mean()
    monthly_closes = monthly_closes.loc[:, coverage >= min_coverage]
    monthly_returns = monthly_closes.pct_change()
    return monthly_closes, monthly_returns
