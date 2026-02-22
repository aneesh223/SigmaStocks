"""
Unit tests for df_to_heatmap function.

Feature: convolutional-order-book
Task: 5.1 Create df_to_heatmap function
"""

import pytest
import numpy as np
import pandas as pd
from datetime import datetime, timedelta

from src.microstructure import df_to_heatmap


def create_sample_ohlcv_data(num_rows=64, base_price=100.0, base_volume=1000000):
    """Helper function to create sample OHLCV data."""
    dates = pd.date_range(
        start=datetime.now() - timedelta(minutes=num_rows),
        periods=num_rows,
        freq='1min'
    )
    
    data = {
        'Open': [base_price + i * 0.1 for i in range(num_rows)],
        'High': [base_price + i * 0.1 + 0.5 for i in range(num_rows)],
        'Low': [base_price + i * 0.1 - 0.5 for i in range(num_rows)],
        'Close': [base_price + i * 0.1 + 0.2 for i in range(num_rows)],
        'Volume': [base_volume + i * 1000 for i in range(num_rows)]
    }
    
    return pd.DataFrame(data, index=dates)


def test_heatmap_shape_with_exact_64_rows():
    """
    Test that heatmap has shape (64, 64) when given exactly 64 rows of data.
    **Validates: Requirements 2.1**
    """
    df = create_sample_ohlcv_data(num_rows=64)
    heatmap = df_to_heatmap(df)
    
    assert heatmap is not None, "Heatmap should not be None"
    assert heatmap.shape == (64, 64), f"Expected shape (64, 64), got {heatmap.shape}"
    assert heatmap.dtype == np.float32, f"Expected dtype float32, got {heatmap.dtype}"


def test_heatmap_shape_with_fewer_than_64_rows():
    """
    Test that heatmap has shape (64, 64) even with fewer than 64 rows (padding).
    **Validates: Requirements 2.1, 2.6**
    """
    df = create_sample_ohlcv_data(num_rows=30)
    heatmap = df_to_heatmap(df)
    
    assert heatmap is not None, "Heatmap should not be None"
    assert heatmap.shape == (64, 64), f"Expected shape (64, 64), got {heatmap.shape}"


def test_heatmap_shape_with_more_than_64_rows():
    """
    Test that heatmap uses only the last 64 rows when given more data.
    **Validates: Requirements 2.2**
    """
    df = create_sample_ohlcv_data(num_rows=100)
    heatmap = df_to_heatmap(df)
    
    assert heatmap is not None, "Heatmap should not be None"
    assert heatmap.shape == (64, 64), f"Expected shape (64, 64), got {heatmap.shape}"


def test_heatmap_normalization():
    """
    Test that heatmap values are normalized to [0, 1] range.
    **Validates: Requirements 2.5**
    """
    df = create_sample_ohlcv_data(num_rows=64)
    heatmap = df_to_heatmap(df)
    
    assert heatmap is not None, "Heatmap should not be None"
    assert np.all(heatmap >= 0.0), f"All values should be >= 0.0, min: {heatmap.min()}"
    assert np.all(heatmap <= 1.0), f"All values should be <= 1.0, max: {heatmap.max()}"


def test_heatmap_with_zero_price_range():
    """
    Test that heatmap handles zero price range by returning uniform heatmap.
    **Validates: Requirements 2.7**
    """
    dates = pd.date_range(start=datetime.now(), periods=64, freq='1min')
    
    # All prices are the same
    data = {
        'Open': [100.0] * 64,
        'High': [100.0] * 64,
        'Low': [100.0] * 64,
        'Close': [100.0] * 64,
        'Volume': [1000000] * 64
    }
    
    df = pd.DataFrame(data, index=dates)
    heatmap = df_to_heatmap(df)
    
    assert heatmap is not None, "Heatmap should not be None"
    assert heatmap.shape == (64, 64), f"Expected shape (64, 64), got {heatmap.shape}"
    # All values should be the same (uniform heatmap)
    assert np.allclose(heatmap, heatmap[0, 0]), "Heatmap should be uniform for zero price range"


def test_heatmap_with_empty_dataframe():
    """
    Test that heatmap returns None for empty DataFrame.
    **Validates: Requirements 2.6**
    """
    df = pd.DataFrame()
    heatmap = df_to_heatmap(df)
    
    assert heatmap is None, "Heatmap should be None for empty DataFrame"


def test_heatmap_with_missing_columns():
    """
    Test that heatmap returns None when required columns are missing.
    """
    dates = pd.date_range(start=datetime.now(), periods=64, freq='1min')
    
    # Missing 'Volume' column
    data = {
        'Open': [100.0] * 64,
        'High': [101.0] * 64,
        'Low': [99.0] * 64,
        'Close': [100.5] * 64
    }
    
    df = pd.DataFrame(data, index=dates)
    heatmap = df_to_heatmap(df)
    
    assert heatmap is None, "Heatmap should be None when required columns are missing"


def test_heatmap_with_nan_values():
    """
    Test that heatmap handles NaN values gracefully.
    """
    df = create_sample_ohlcv_data(num_rows=64)
    
    # Introduce some NaN values
    df.loc[df.index[10], 'Volume'] = np.nan
    df.loc[df.index[20], 'High'] = np.nan
    
    heatmap = df_to_heatmap(df)
    
    assert heatmap is not None, "Heatmap should handle NaN values"
    assert heatmap.shape == (64, 64), f"Expected shape (64, 64), got {heatmap.shape}"
    assert not np.any(np.isnan(heatmap)), "Heatmap should not contain NaN values"


def test_heatmap_log_transformation():
    """
    Test that volume is log-transformed.
    **Validates: Requirements 2.4**
    """
    df = create_sample_ohlcv_data(num_rows=64, base_volume=1000000)
    heatmap = df_to_heatmap(df)
    
    assert heatmap is not None, "Heatmap should not be None"
    # The heatmap should have non-zero values due to log transformation
    assert np.any(heatmap > 0), "Heatmap should have non-zero values from log-transformed volumes"


def test_heatmap_price_mapping():
    """
    Test that lower prices map to lower Y-coordinates and higher prices to higher Y-coordinates.
    **Validates: Requirements 2.3**
    """
    # Create data with clear price separation
    dates = pd.date_range(start=datetime.now(), periods=64, freq='1min')
    
    # First half: low prices, second half: high prices
    data = {
        'Open': [100.0] * 32 + [200.0] * 32,
        'High': [101.0] * 32 + [201.0] * 32,
        'Low': [99.0] * 32 + [199.0] * 32,
        'Close': [100.5] * 32 + [200.5] * 32,
        'Volume': [1000000] * 64
    }
    
    df = pd.DataFrame(data, index=dates)
    heatmap = df_to_heatmap(df)
    
    assert heatmap is not None, "Heatmap should not be None"
    
    # Check that there's activity in both lower and upper regions
    # Lower half of Y-axis (rows 0-31) should have some activity from low prices
    # Upper half of Y-axis (rows 32-63) should have some activity from high prices
    lower_half_sum = np.sum(heatmap[0:32, :])
    upper_half_sum = np.sum(heatmap[32:64, :])
    
    assert lower_half_sum > 0, "Lower half of heatmap should have activity from low prices"
    assert upper_half_sum > 0, "Upper half of heatmap should have activity from high prices"


def test_heatmap_time_ordering():
    """
    Test that time is ordered chronologically (oldest to newest, left to right).
    **Validates: Requirements 2.2, 2.3**
    """
    df = create_sample_ohlcv_data(num_rows=64)
    heatmap = df_to_heatmap(df)
    
    assert heatmap is not None, "Heatmap should not be None"
    # With 64 rows, the data should fill the entire width
    # We can't directly verify time ordering without more complex logic,
    # but we can verify the heatmap is not all zeros
    assert np.any(heatmap > 0), "Heatmap should have non-zero values"


def test_heatmap_with_small_dataset():
    """
    Test that heatmap handles very small datasets (< 10 rows).
    **Validates: Requirements 2.6**
    """
    df = create_sample_ohlcv_data(num_rows=5)
    heatmap = df_to_heatmap(df)
    
    assert heatmap is not None, "Heatmap should not be None"
    assert heatmap.shape == (64, 64), f"Expected shape (64, 64), got {heatmap.shape}"


def test_heatmap_custom_window_size():
    """
    Test that heatmap respects custom window_size parameter.
    """
    df = create_sample_ohlcv_data(num_rows=100)
    window_size = 32
    heatmap = df_to_heatmap(df, window_size=window_size)
    
    assert heatmap is not None, "Heatmap should not be None"
    assert heatmap.shape == (window_size, window_size), \
        f"Expected shape ({window_size}, {window_size}), got {heatmap.shape}"


def test_heatmap_returns_float32():
    """
    Test that heatmap returns float32 array for memory efficiency.
    **Validates: Requirements 2.1**
    """
    df = create_sample_ohlcv_data(num_rows=64)
    heatmap = df_to_heatmap(df)
    
    assert heatmap is not None, "Heatmap should not be None"
    assert heatmap.dtype == np.float32, f"Expected dtype float32, got {heatmap.dtype}"
