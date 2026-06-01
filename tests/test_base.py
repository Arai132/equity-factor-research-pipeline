import numpy as np
import pandas as pd
from factors.base import winsorize_cross_section, zscore_cross_section, normalize

def test_winsorize_clips_at_n_std():
    s = pd.Series([-100.0, -1.0, 0.0, 1.0, 100.0])
    result = winsorize_cross_section(s, n_std=2.0)
    mean, std = s.mean(), s.std()
    assert result.max() <= mean + 2.0 * std + 1e-9
    assert result.min() >= mean - 2.0 * std - 1e-9

def test_zscore_zero_mean_unit_std():
    s = pd.Series([1.0, 2.0, 3.0, 4.0, 5.0])
    result = zscore_cross_section(s)
    assert abs(result.mean()) < 1e-10
    assert abs(result.std(ddof=1) - 1.0) < 1e-10

def test_zscore_preserves_rank_order():
    s = pd.Series([1.0, 3.0, 2.0, 5.0, 4.0])
    result = zscore_cross_section(s)
    assert result.rank().equals(s.rank())

def test_normalize_handles_constant_series():
    s = pd.Series([5.0, 5.0, 5.0])
    result = normalize(s)
    assert result.isna().all() or (result == 0).all()

def test_normalize_removes_outlier_effect():
    np.random.seed(42)
    s = pd.Series(np.random.randn(100))
    s_with_outlier = s.copy()
    s_with_outlier.iloc[0] = 1000.0
    result = normalize(s_with_outlier)
    # Outlier should be clipped to ≤ 3 std, not dominate the distribution
    assert result.max() < 4.0
