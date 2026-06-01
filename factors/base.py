from abc import ABC, abstractmethod
import numpy as np
import pandas as pd


def winsorize_cross_section(s: pd.Series, n_std: float = 3.0) -> pd.Series:
    """Clip values to [-n_std, n_std]. Assumes input is already z-scored."""
    return s.clip(-n_std, n_std)


def zscore_cross_section(s: pd.Series) -> pd.Series:
    mean, std = s.mean(), s.std(ddof=1)
    if std == 0 or np.isnan(std):
        return pd.Series(np.nan, index=s.index)
    return (s - mean) / std


def normalize(s: pd.Series, n_std: float = 3.0) -> pd.Series:
    """Z-score then winsorize. Standard cross-sectional normalization pipeline."""
    valid = s.dropna()
    if len(valid) < 2:
        return pd.Series(np.nan, index=s.index)
    return winsorize_cross_section(zscore_cross_section(valid), n_std).reindex(s.index)


class BaseFactor(ABC):
    """
    Abstract base for all equity factors.
    compute() returns a DataFrame: index=rebalance_dates, columns=tickers.
    Each row contains cross-sectionally winsorized and z-scored factor scores.
    """

    @abstractmethod
    def compute(
        self,
        prices: pd.DataFrame,
        volume: pd.DataFrame,
        fundamentals: pd.DataFrame,
        rebalance_dates: pd.DatetimeIndex,
    ) -> pd.DataFrame:
        ...

    def _normalize_signals(self, signals: dict) -> pd.DataFrame:
        """Convert {date: raw_series} to a normalized DataFrame."""
        if not signals:
            return pd.DataFrame()
        df = pd.DataFrame(signals).T
        df.index = pd.DatetimeIndex(df.index)
        df.index.name = "date"
        return df.apply(normalize, axis=1)
