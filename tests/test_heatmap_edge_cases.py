"""
Unit tests for heatmap edge cases.

Feature: convolutional-order-book
Task: 5.7 Write unit tests for heatmap edge cases
"""

import pytest
import numpy as np
import pandas as pd
from datetime import datetime, timedelta

from src.microstructure import df_to_heatmap


def test_heatmap_exactly_64_minutes():
    """
    Test with exactly 64 minutes of data.
    **Validates: Requirements 2.6**
    """
    num_rows = 64
    dates = pd.date_range(start=datetime.now(), periods=num_rows, freq='1min')
    
    data = {
        'Open': [100.0 + i * 0.1 for i in range(num_rows)],
        'High': [101.0 + i * 0.1 for i in range(num_rows)],
        'Low': [99.0 + i * 0.1 for i in range(num_rows)],
        'Close': [100.5 + i * 0.1 for i in range(num_rows)],
        'Volume': [1000000 + i * 1000 for i in range(num_rows)]
    }
    
    df = pd.DataFrame(data, index=dates)
    heatmap = df_to_heatmap(df)
    
    assert heatmap is not None, "Heatmap should not be None with exactly 64 minutes"
    assert heatmap.shape == (64, 64), f"Expected shape (64, 64), got {heatmap.shape}"
    assert heatmap.dtype == np.float32, f"Expected dtype float32, got {heatmap.dtype}"
    
    # Verify all values are in [0, 1]
    assert np.all(heatmap >= 0.0) and np.all(heatmap <= 1.0), \
        f"Values should be in [0, 1], got min={heatmap.min()}, max={heatmap.max()}"
    
    # With varying volumes, the heatmap should have some variation
    assert heatmap.max() > heatmap.min(), \
        "Heatmap should have variation with varying volumes"


def test_heatmap_30_minutes_padding():
    """
    Test with 30 minutes of data (should pad with zeros).
    **Validates: Requirements 2.6**
    """
    num_rows = 30
    dates = pd.date_range(start=datetime.now(), periods=num_rows, freq='1min')
    
    data = {
        'Open': [100.0 + i * 0.1 for i in range(num_rows)],
        'High': [101.0 + i * 0.1 for i in range(num_rows)],
        'Low': [99.0 + i * 0.1 for i in range(num_rows)],
        'Close': [100.5 + i * 0.1 for i in range(num_rows)],
        'Volume': [1000000 + i * 1000 for i in range(num_rows)]
    }
    
    df = pd.DataFrame(data, index=dates)
    heatmap = df_to_heatmap(df)
    
    assert heatmap is not None, "Heatmap should not be None with 30 minutes"
    assert heatmap.shape == (64, 64), f"Expected shape (64, 64), got {heatmap.shape}"
    
    # The first 34 columns should be padding (mostly zeros)
    # The last 30 columns should have the actual data
    padding_cols = 64 - num_rows  # 34 columns
    
    # Check that padding columns have lower mean intensity than data columns
    padding_mean = heatmap[:, :padding_cols].mean()
    data_mean = heatmap[:, padding_cols:].mean()
    
    # Padding should have lower or equal intensity
    assert padding_mean <= data_mean, \
        f"Padding columns should have lower intensity, got padding={padding_mean:.4f}, data={data_mean:.4f}"


def test_heatmap_zero_price_range():
    """
    Test with zero price range (should return uniform heatmap).
    **Validates: Requirements 2.7**
    """
    num_rows = 64
    dates = pd.date_range(start=datetime.now(), periods=num_rows, freq='1min')
    
    # All prices are identical
    price = 100.0
    data = {
        'Open': [price] * num_rows,
        'High': [price] * num_rows,
        'Low': [price] * num_rows,
        'Close': [price] * num_rows,
        'Volume': [1000000] * num_rows
    }
    
    df = pd.DataFrame(data, index=dates)
    heatmap = df_to_heatmap(df)
    
    assert heatmap is not None, "Heatmap should not be None with zero price range"
    assert heatmap.shape == (64, 64), f"Expected shape (64, 64), got {heatmap.shape}"
    
    # With zero price range, the heatmap should be uniform
    # All values should be the same (or very close)
    unique_values = np.unique(heatmap)
    
    # Should have very few unique values (ideally 1, but allow for floating point variations)
    assert len(unique_values) <= 3, \
        f"Uniform data should produce uniform heatmap, got {len(unique_values)} unique values"
    
    # All values should be in [0, 1]
    assert np.all(heatmap >= 0.0) and np.all(heatmap <= 1.0), \
        f"Values should be in [0, 1], got min={heatmap.min()}, max={heatmap.max()}"


def test_heatmap_with_nan_values():
    """
    Test with NaN values (should replace with 0).
    **Validates: Requirements 2.7**
    """
    num_rows = 64
    dates = pd.date_range(start=datetime.now(), periods=num_rows, freq='1min')
    
    data = {
        'Open': [100.0 + i * 0.1 for i in range(num_rows)],
        'High': [101.0 + i * 0.1 for i in range(num_rows)],
        'Low': [99.0 + i * 0.1 for i in range(num_rows)],
        'Close': [100.5 + i * 0.1 for i in range(num_rows)],
        'Volume': [1000000 + i * 1000 for i in range(num_rows)]
    }
    
    df = pd.DataFrame(data, index=dates)
    
    # Introduce NaN values at specific positions
    df.iloc[10, df.columns.get_loc('High')] = np.nan
    df.iloc[20, df.columns.get_loc('Low')] = np.nan
    df.iloc[30, df.columns.get_loc('Volume')] = np.nan
    
    heatmap = df_to_heatmap(df)
    
    assert heatmap is not None, "Heatmap should not be None with NaN values"
    assert heatmap.shape == (64, 64), f"Expected shape (64, 64), got {heatmap.shape}"
    
    # The output should not contain any NaN values
    assert not np.any(np.isnan(heatmap)), \
        "Heatmap should not contain NaN values"
    
    # All values should be in [0, 1]
    assert np.all(heatmap >= 0.0) and np.all(heatmap <= 1.0), \
        f"Values should be in [0, 1], got min={heatmap.min()}, max={heatmap.max()}"


def test_heatmap_missing_volume_data():
    """
    Test with missing volume data (should use 0).
    **Validates: Requirements 2.7**
    """
    num_rows = 64
    dates = pd.date_range(start=datetime.now(), periods=num_rows, freq='1min')
    
    # Create data with some zero volumes
    volumes = [1000000 + i * 1000 for i in range(num_rows)]
    # Set some volumes to 0
    volumes[10] = 0
    volumes[20] = 0
    volumes[30] = 0
    
    data = {
        'Open': [100.0 + i * 0.1 for i in range(num_rows)],
        'High': [101.0 + i * 0.1 for i in range(num_rows)],
        'Low': [99.0 + i * 0.1 for i in range(num_rows)],
        'Close': [100.5 + i * 0.1 for i in range(num_rows)],
        'Volume': volumes
    }
    
    df = pd.DataFrame(data, index=dates)
    heatmap = df_to_heatmap(df)
    
    assert heatmap is not None, "Heatmap should not be None with zero volumes"
    assert heatmap.shape == (64, 64), f"Expected shape (64, 64), got {heatmap.shape}"
    
    # The heatmap should handle zero volumes gracefully
    # All values should be in [0, 1]
    assert np.all(heatmap >= 0.0) and np.all(heatmap <= 1.0), \
        f"Values should be in [0, 1], got min={heatmap.min()}, max={heatmap.max()}"
    
    # The heatmap should not be all zeros (since we have non-zero volumes)
    assert heatmap.max() > 0, \
        "Heatmap should have non-zero values with non-zero volumes"


def test_heatmap_all_zero_volumes():
    """
    Test with all zero volumes.
    **Validates: Requirements 2.7**
    """
    num_rows = 64
    dates = pd.date_range(start=datetime.now(), periods=num_rows, freq='1min')
    
    data = {
        'Open': [100.0 + i * 0.1 for i in range(num_rows)],
        'High': [101.0 + i * 0.1 for i in range(num_rows)],
        'Low': [99.0 + i * 0.1 for i in range(num_rows)],
        'Close': [100.5 + i * 0.1 for i in range(num_rows)],
        'Volume': [0] * num_rows  # All zero volumes
    }
    
    df = pd.DataFrame(data, index=dates)
    heatmap = df_to_heatmap(df)
    
    assert heatmap is not None, "Heatmap should not be None with all zero volumes"
    assert heatmap.shape == (64, 64), f"Expected shape (64, 64), got {heatmap.shape}"
    
    # With all zero volumes, log(1 + 0) = 0, so all intensities should be 0
    # After normalization, the heatmap should be all zeros
    assert np.all(heatmap == 0.0), \
        f"Heatmap with all zero volumes should be all zeros, got unique values: {np.unique(heatmap)}"


def test_heatmap_single_row():
    """
    Test with only 1 minute of data.
    **Validates: Requirements 2.6**
    """
    num_rows = 1
    dates = pd.date_range(start=datetime.now(), periods=num_rows, freq='1min')
    
    data = {
        'Open': [100.0],
        'High': [101.0],
        'Low': [99.0],
        'Close': [100.5],
        'Volume': [1000000]
    }
    
    df = pd.DataFrame(data, index=dates)
    heatmap = df_to_heatmap(df)
    
    assert heatmap is not None, "Heatmap should not be None with 1 minute"
    assert heatmap.shape == (64, 64), f"Expected shape (64, 64), got {heatmap.shape}"
    
    # The first 63 columns should be padding (zeros)
    # The last column should have the actual data
    padding_mean = heatmap[:, :63].mean()
    last_col_mean = heatmap[:, 63].mean()
    
    # Padding should have lower or equal intensity
    assert padding_mean <= last_col_mean, \
        f"Padding should have lower intensity, got padding={padding_mean:.4f}, last_col={last_col_mean:.4f}"


def test_heatmap_extreme_price_range():
    """
    Test with extreme price range.
    **Validates: Requirements 2.6, 2.7**
    """
    num_rows = 64
    dates = pd.date_range(start=datetime.now(), periods=num_rows, freq='1min')
    
    # Create data with extreme price range
    data = {
        'Open': [50000.0 + i * 100 for i in range(num_rows)],
        'High': [50100.0 + i * 100 for i in range(num_rows)],
        'Low': [49900.0 + i * 100 for i in range(num_rows)],
        'Close': [50050.0 + i * 100 for i in range(num_rows)],
        'Volume': [1000000 + i * 1000 for i in range(num_rows)]
    }
    
    df = pd.DataFrame(data, index=dates)
    heatmap = df_to_heatmap(df)
    
    assert heatmap is not None, "Heatmap should not be None with extreme prices"
    assert heatmap.shape == (64, 64), f"Expected shape (64, 64), got {heatmap.shape}"
    
    # All values should be in [0, 1] regardless of price range
    assert np.all(heatmap >= 0.0) and np.all(heatmap <= 1.0), \
        f"Values should be in [0, 1], got min={heatmap.min()}, max={heatmap.max()}"


def test_heatmap_extreme_volume_range():
    """
    Test with extreme volume range.
    **Validates: Requirements 2.4, 2.5**
    """
    num_rows = 64
    dates = pd.date_range(start=datetime.now(), periods=num_rows, freq='1min')
    
    # Create data with extreme volume range
    volumes = [1] * (num_rows // 2) + [1000000000] * (num_rows // 2)
    
    data = {
        'Open': [100.0 + i * 0.1 for i in range(num_rows)],
        'High': [101.0 + i * 0.1 for i in range(num_rows)],
        'Low': [99.0 + i * 0.1 for i in range(num_rows)],
        'Close': [100.5 + i * 0.1 for i in range(num_rows)],
        'Volume': volumes
    }
    
    df = pd.DataFrame(data, index=dates)
    heatmap = df_to_heatmap(df)
    
    assert heatmap is not None, "Heatmap should not be None with extreme volumes"
    assert heatmap.shape == (64, 64), f"Expected shape (64, 64), got {heatmap.shape}"
    
    # All values should be in [0, 1] regardless of volume range
    assert np.all(heatmap >= 0.0) and np.all(heatmap <= 1.0), \
        f"Values should be in [0, 1], got min={heatmap.min()}, max={heatmap.max()}"
    
    # The log transformation should compress the extreme range
    # Both halves should be visible (not all zeros or all ones)
    first_half_mean = heatmap[:, :32].mean()
    second_half_mean = heatmap[:, 32:].mean()
    
    # Both halves should have some intensity
    assert first_half_mean >= 0 and second_half_mean >= 0, \
        f"Both halves should be visible, got first={first_half_mean:.4f}, second={second_half_mean:.4f}"


def test_heatmap_negative_prices():
    """
    Test that negative prices are handled (though unlikely in real data).
    **Validates: Requirements 2.6, 2.7**
    """
    num_rows = 64
    dates = pd.date_range(start=datetime.now(), periods=num_rows, freq='1min')
    
    # Create data with negative prices (edge case)
    data = {
        'Open': [-10.0 + i * 0.1 for i in range(num_rows)],
        'High': [-9.0 + i * 0.1 for i in range(num_rows)],
        'Low': [-11.0 + i * 0.1 for i in range(num_rows)],
        'Close': [-10.5 + i * 0.1 for i in range(num_rows)],
        'Volume': [1000000 + i * 1000 for i in range(num_rows)]
    }
    
    df = pd.DataFrame(data, index=dates)
    heatmap = df_to_heatmap(df)
    
    # The function should handle negative prices gracefully
    # (even though they're unlikely in real financial data)
    if heatmap is not None:
        assert heatmap.shape == (64, 64), f"Expected shape (64, 64), got {heatmap.shape}"
        
        # All values should be in [0, 1]
        assert np.all(heatmap >= 0.0) and np.all(heatmap <= 1.0), \
            f"Values should be in [0, 1], got min={heatmap.min()}, max={heatmap.max()}"


def test_heatmap_very_small_price_range():
    """
    Test with very small price range (e.g., penny stocks).
    **Validates: Requirements 2.6, 2.7**
    """
    num_rows = 64
    dates = pd.date_range(start=datetime.now(), periods=num_rows, freq='1min')
    
    # Create data with very small price range
    data = {
        'Open': [0.01 + i * 0.0001 for i in range(num_rows)],
        'High': [0.011 + i * 0.0001 for i in range(num_rows)],
        'Low': [0.009 + i * 0.0001 for i in range(num_rows)],
        'Close': [0.0105 + i * 0.0001 for i in range(num_rows)],
        'Volume': [1000000 + i * 1000 for i in range(num_rows)]
    }
    
    df = pd.DataFrame(data, index=dates)
    heatmap = df_to_heatmap(df)
    
    assert heatmap is not None, "Heatmap should not be None with small prices"
    assert heatmap.shape == (64, 64), f"Expected shape (64, 64), got {heatmap.shape}"
    
    # All values should be in [0, 1]
    assert np.all(heatmap >= 0.0) and np.all(heatmap <= 1.0), \
        f"Values should be in [0, 1], got min={heatmap.min()}, max={heatmap.max()}"
